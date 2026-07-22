from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_api_client import call_chat_completion
from navigation_core import (
    MAX_STEPS_DEFAULT,
    NonModelRunFailure,
    infer_initial_facing,
    infer_prompt_mode,
    infer_prompt_template,
    resolve_landmark_retention,
    run_navigation_episode,
    validate_trial_setup,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def default_memory_type(experiment_type: str) -> str:
    if experiment_type == "goal_trap_memory":
        return "recent"
    return "omniscient"


def default_memory_size(experiment_type: str) -> int:
    if experiment_type == "goal_trap_memory":
        return 1
    return 5


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.result_path:
        return Path(args.result_path).expanduser().resolve().parent
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    raise SystemExit("--result-path 或 --output-dir 必须提供一个")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one LLM navigation trial into a canonical result directory.")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL_NAME", ""), help="API model id, e.g. gpt-5")
    parser.add_argument("--api-base-url", default=os.environ.get("LLM_API_BASE_URL", ""), help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key-env", default="LLM_API_KEY", help="environment variable that stores the API key")
    parser.add_argument("--map", required=True, dest="map_id")
    parser.add_argument("--trial", required=True, type=int)
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument("--experiment-type", default="main_fpv_baseline")
    parser.add_argument("--condition", default="omniscient")
    parser.add_argument("--prompt-mode", choices=["auto", "action_only", "cm_cot"], default="auto")
    parser.add_argument("--prompt-template", default="", help="override prompt template path")
    parser.add_argument("--result-path", default="", help="canonical .../result.json path from manifest")
    parser.add_argument("--output-dir", default="", help="repeat directory; ignored when --result-path is given")
    parser.add_argument("--memory-type", default="", help="default: omniscient, or recent for goal_trap_memory")
    parser.add_argument("--memory-size", type=int, default=-1, help="default: 5, or 1 for goal_trap_memory")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--sleep", type=float, default=float(os.environ.get("LLM_SLEEP_SECONDS", "0")))
    parser.add_argument("--api-max-attempts", type=int, default=int(os.environ.get("LLM_API_MAX_ATTEMPTS", "5")))
    parser.add_argument("--api-timeout", type=float, default=float(os.environ.get("LLM_API_TIMEOUT_SECONDS", "300")))
    parser.add_argument("--api-backoff", type=float, default=float(os.environ.get("LLM_API_BACKOFF_SECONDS", "2")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("LLM_MAX_TOKENS", "16384")))
    parser.add_argument("--reasoning-effort", default=os.environ.get("LLM_REASONING_EFFORT", ""), help="optional reasoning effort passed to the API")
    parser.add_argument("--initial-facing", default="", help="override initial orientation")
    parser.add_argument("--dry-run", action="store_true", help="validate paths and print plan only; no API call and no result.json")
    parser.add_argument("--local-policy", action="store_true", help="run with deterministic local policy for smoke testing")
    parser.add_argument("--visualize", action="store_true", help="also save result.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_output_dir(args)
    prompt_mode = infer_prompt_mode(args.experiment_type, args.condition) if args.prompt_mode == "auto" else args.prompt_mode
    prompt_template = (
        Path(args.prompt_template).expanduser().resolve()
        if args.prompt_template
        else infer_prompt_template(prompt_mode, args.experiment_type, args.condition)
    )
    resolve_landmark_retention(args.experiment_type, args.condition)
    initial_facing = args.initial_facing or infer_initial_facing(args.experiment_type)
    valid_facings = {"UP", "DOWN", "LEFT", "RIGHT", "NORTH", "SOUTH", "WEST", "EAST"}
    if initial_facing not in valid_facings:
        raise SystemExit(f"unsupported initial orientation: {initial_facing}")
    memory_type = args.memory_type or default_memory_type(args.experiment_type)
    memory_size = args.memory_size if args.memory_size >= 0 else default_memory_size(args.experiment_type)

    setup = validate_trial_setup(args.map_id, args.trial, prompt_template)
    plan = {
        "model": args.model or ("local-policy" if args.local_policy else ""),
        "map_id": args.map_id,
        "trial": args.trial,
        "repeat": args.repeat,
        "experiment_type": args.experiment_type,
        "condition": args.condition,
        "prompt_mode": prompt_mode,
        "prompt_template": str(prompt_template),
        "memory_type": memory_type,
        "memory_size": memory_size,
        "initial_facing": initial_facing,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "output_dir": str(output_dir),
        "result_path": str(output_dir / "result.json"),
        "setup": setup,
    }

    if args.dry_run:
        print(json.dumps({"status": "dry_run_ok", "plan": plan}, ensure_ascii=False, indent=2))
        return 0

    if not args.local_policy and not args.model:
        print("缺少 --model 或环境变量 LLM_MODEL_NAME", file=sys.stderr)
        return 4
    if not args.local_policy and not args.api_base_url:
        print("缺少 --api-base-url 或环境变量 LLM_API_BASE_URL", file=sys.stderr)
        return 4

    api_key = os.environ.get(args.api_key_env, "")
    if not args.local_policy and not api_key:
        print(f"缺少 API key 环境变量: {args.api_key_env}", file=sys.stderr)
        return 4

    started_at = utc_now()
    status_path = output_dir / "run_status.json"
    write_status(
        status_path,
        {
            "status": "running",
            "started_at": started_at,
            "plan": plan,
        },
    )

    def call_model(prompt_text: str, step_dir: Path):
        return call_chat_completion(
            prompt_text=prompt_text,
            model=args.model,
            api_base_url=args.api_base_url,
            api_key=api_key,
            debug_dir=step_dir,
            max_attempts=args.api_max_attempts,
            backoff_seconds=args.api_backoff,
            timeout_seconds=args.api_timeout,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )

    try:
        result = run_navigation_episode(
            map_id=args.map_id,
            trial_num=args.trial,
            repeat_idx=args.repeat,
            model=args.model,
            experiment_type=args.experiment_type,
            condition=args.condition,
            output_dir=output_dir,
            call_model=None if args.local_policy else call_model,
            prompt_mode=prompt_mode,
            prompt_template=prompt_template,
            memory_type=memory_type,
            memory_size=memory_size,
            max_steps=args.max_steps,
            sleep_seconds=args.sleep,
            initial_facing=initial_facing,
            local_policy=args.local_policy,
            visualize=args.visualize,
        )
    except NonModelRunFailure as exc:
        write_status(
            status_path,
            {
                "status": "incomplete_non_model_failure",
                "failure_type": exc.failure_type,
                "failure_reason": str(exc),
                "failure_step": exc.step,
                "retryable": exc.retryable,
                "started_at": started_at,
                "ended_at": utc_now(),
                "plan": plan,
            },
        )
        print(f"incomplete_non_model_failure: {exc.failure_type}: {exc}", file=sys.stderr)
        return 2 if exc.retryable else 3
    except Exception as exc:
        write_status(
            status_path,
            {
                "status": "failed_runner_error",
                "failure_type": type(exc).__name__,
                "failure_reason": str(exc),
                "started_at": started_at,
                "ended_at": utc_now(),
                "plan": plan,
            },
        )
        print(f"failed_runner_error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    write_status(
        status_path,
        {
            "status": "completed_valid_repeat",
            "started_at": started_at,
            "ended_at": utc_now(),
            "result_path": str(output_dir / "result.json"),
            "reached_goal": result.get("reached_goal"),
            "num_steps": result.get("num_steps"),
            "accumulated_cost": result.get("accumulated_cost"),
            "plan": plan,
        },
    )
    print(json.dumps({"status": "completed_valid_repeat", "result_path": str(output_dir / "result.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
