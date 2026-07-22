from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class APIClientError(RuntimeError):
    def __init__(self, message: str, *, failure_type: str, retryable: bool):
        super().__init__(message)
        self.failure_type = failure_type
        self.retryable = retryable


class TransientAPIError(APIClientError):
    def __init__(self, message: str, *, failure_type: str = "api_transient"):
        super().__init__(message, failure_type=failure_type, retryable=True)


class NonRetryableAPIError(APIClientError):
    def __init__(self, message: str, *, failure_type: str = "api_nonretryable"):
        super().__init__(message, failure_type=failure_type, retryable=False)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    raw_json: dict[str, Any] | None = None


def normalize_chat_completions_url(api_base_url: str) -> str:
    base = (api_base_url or "").strip()
    if not base:
        raise NonRetryableAPIError("LLM_API_BASE_URL is empty", failure_type="missing_api_base_url")
    if base.rstrip("/").endswith("/chat/completions"):
        return base.rstrip("/")
    return base.rstrip("/") + "/chat/completions"


def _safe_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: ("<redacted>" if k.lower() == "authorization" else v) for k, v in headers.items()}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_error_body(exc: error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_usage(data: dict[str, Any]) -> tuple[int, int, int]:
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = int(details.get("reasoning_tokens") or 0)
    return prompt_tokens, completion_tokens, reasoning_tokens


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def call_chat_completion(
    *,
    prompt_text: str,
    model: str,
    api_base_url: str,
    api_key: str,
    debug_dir: Path,
    max_attempts: int = 5,
    backoff_seconds: float = 2.0,
    timeout_seconds: float = 300.0,
    max_tokens: int = 16384,
    temperature: float = 0.0,
    reasoning_effort: str = "",
) -> LLMResponse:
    """Call an OpenAI-compatible chat completions endpoint with per-step retries."""
    if not model:
        raise NonRetryableAPIError("model is empty", failure_type="missing_model")
    if not api_key:
        raise NonRetryableAPIError("API key is empty", failure_type="missing_api_key")

    url = normalize_chat_completions_url(api_base_url)
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    encoded_body = json.dumps(body).encode("utf-8")

    debug_dir.mkdir(parents=True, exist_ok=True)
    last_error = "unknown API failure"

    for attempt in range(1, max_attempts + 1):
        attempt_dir = debug_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            attempt_dir / "api_request.json",
            {"url": url, "headers": _safe_headers(headers), "body": body},
        )

        req = request.Request(url, data=encoded_body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                status = int(resp.status)
                resp_headers = dict(resp.headers.items())
                resp_text = resp.read().decode("utf-8", errors="replace")

            (attempt_dir / "api_status.txt").write_text(
                f"status_code: {status}\nheaders: {resp_headers}",
                encoding="utf-8",
            )
            (attempt_dir / "api_response.txt").write_text(resp_text, encoding="utf-8")

            try:
                data = json.loads(resp_text)
            except json.JSONDecodeError as exc:
                last_error = f"invalid JSON response: {exc}"
                (attempt_dir / "api_error.txt").write_text(last_error, encoding="utf-8")
                if attempt < max_attempts:
                    time.sleep(backoff_seconds * (2 ** (attempt - 1)))
                    continue
                raise TransientAPIError(last_error, failure_type="api_invalid_json") from exc

            _write_json(attempt_dir / "api_response.json", data)
            content = _extract_content(data).strip()
            if not content:
                last_error = "empty assistant content"
                (attempt_dir / "api_error.txt").write_text(last_error, encoding="utf-8")
                if attempt < max_attempts:
                    time.sleep(backoff_seconds * (2 ** (attempt - 1)))
                    continue
                raise TransientAPIError(last_error, failure_type="api_empty_content")

            prompt_tokens, completion_tokens, reasoning_tokens = _extract_usage(data)
            (debug_dir / "api_status.txt").write_text(
                f"status_code: {status}\nheaders: {resp_headers}",
                encoding="utf-8",
            )
            (debug_dir / "api_response.txt").write_text(resp_text, encoding="utf-8")
            _write_json(debug_dir / "api_response.json", data)
            return LLMResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                raw_json=data,
            )

        except error.HTTPError as exc:
            status = int(exc.code)
            resp_text = _read_error_body(exc)
            (attempt_dir / "api_status.txt").write_text(
                f"status_code: {status}\nheaders: {dict(exc.headers.items()) if exc.headers else {}}",
                encoding="utf-8",
            )
            (attempt_dir / "api_response.txt").write_text(resp_text, encoding="utf-8")
            last_error = f"HTTP {status}: {resp_text[:500]}"
            (attempt_dir / "api_error.txt").write_text(last_error, encoding="utf-8")
            if status in RETRYABLE_HTTP_STATUS and attempt < max_attempts:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
                continue
            if status in RETRYABLE_HTTP_STATUS:
                raise TransientAPIError(last_error, failure_type=f"http_{status}") from exc
            raise NonRetryableAPIError(last_error, failure_type=f"http_{status}") from exc

        except (error.URLError, socket.timeout, TimeoutError) as exc:
            last_error = repr(exc)
            (attempt_dir / "api_error.txt").write_text(last_error, encoding="utf-8")
            if attempt < max_attempts:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
                continue
            raise TransientAPIError(last_error, failure_type="api_connection_error") from exc

    raise TransientAPIError(last_error)
