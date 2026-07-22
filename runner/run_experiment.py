from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from navigation_core import infer_initial_facing, infer_prompt_mode, infer_prompt_template, resolve_landmark_retention


REPO_ROOT = Path(__file__).resolve().parents[1]
SINGLE_RUNNER = Path(__file__).resolve().parent / "run_single_trial.py"
SELECTED_PATHS_FILE = REPO_ROOT / "LlmFPV-main" / "LLM" / "map_setting" / "selected_paths.json"
EXPERIMENTS_FILE = REPO_ROOT / "configs" / "experiments.json"


def load_experiments() -> dict[str, dict[str, Any]]:
    return json.loads(EXPERIMENTS_FILE.read_text(encoding="utf-8"))


def load_all_tasks() -> list[dict[str, str]]:
    selected = json.loads(SELECTED_PATHS_FILE.read_text(encoding="utf-8"))
    tasks = []
    for map_id, map_data in selected.get("maps", {}).items():
        for trial in map_data.get("trials", []):
            tasks.append({"map_id": str(map_id), "trial": str(trial["trial"])})
    return sorted(tasks, key=lambda row: (int(row["map_id"]), int(row["trial"])))


def load_task_file(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("tasks", []) if isinstance(payload, dict) else payload
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    tasks = []
    for row in rows:
        if not isinstance(row, dict) or "map_id" not in row or "trial" not in row:
            raise ValueError("task file rows must contain map_id and trial")
        tasks.append({"map_id": str(row["map_id"]), "trial": str(row["trial"])})
    return sorted(tasks, key=lambda row: (int(row["map_id"]), int(row["trial"])))


def derive_hard_set_tasks(
    baseline_root: Path,
    *,
    threshold: float,
    minimum_runs: int,
    baseline_model: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, bool]] = {}
    discovered_models: set[str] = set()
    duplicate_results_ignored = 0
    conflicting_repeats = []

    result_paths = []
    for dir_path, dir_names, filenames in os.walk(baseline_root, followlinks=True):
        dir_names[:] = [name for name in dir_names if name != "_archive_duplicate"]
        if "result.json" in filenames:
            result_paths.append(Path(dir_path) / "result.json")
    for result_path in result_paths:
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict) or not {"map_id", "trial", "reached_goal"}.issubset(data):
            continue
        model = str(data.get("model") or "")
        if model:
            discovered_models.add(model)
        if baseline_model and model != baseline_model:
            continue
        key = (str(data["map_id"]), str(data["trial"]))
        repeat_key = str(data.get("repeat_idx", result_path.parent.name))
        outcome = bool(data["reached_goal"])
        existing = groups.setdefault(key, {}).get(repeat_key)
        if existing is None:
            groups[key][repeat_key] = outcome
        elif existing == outcome:
            duplicate_results_ignored += 1
        else:
            conflicting_repeats.append({"map_id": key[0], "trial": key[1], "repeat": repeat_key})

    if not baseline_model and len(discovered_models) > 1:
        models = ", ".join(sorted(discovered_models))
        raise ValueError(f"baseline directory contains multiple models ({models}); provide --baseline-model")
    if not groups:
        raise ValueError("no valid baseline result.json files matched the requested model")
    if conflicting_repeats:
        raise ValueError(f"conflicting duplicate baseline repeats detected: {conflicting_repeats[:10]}")

    eligible = []
    incomplete = []
    for (map_id, trial), repeat_outcomes in sorted(groups.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        outcomes = list(repeat_outcomes.values())
        if len(outcomes) < minimum_runs:
            incomplete.append({"map_id": map_id, "trial": trial, "runs": len(outcomes)})
            continue
        success_rate = sum(outcomes) / len(outcomes)
        if success_rate <= threshold:
            eligible.append({"map_id": map_id, "trial": trial})

    metadata = {
        "source": str(baseline_root),
        "baseline_model": baseline_model or (next(iter(discovered_models)) if discovered_models else ""),
        "valid_results": sum(len(outcomes) for outcomes in groups.values()),
        "duplicate_results_ignored": duplicate_results_ignored,
        "map_trials_with_results": len(groups),
        "minimum_baseline_runs": minimum_runs,
        "hard_threshold": threshold,
        "hard_threshold_operator": "<=",
        "incomplete_map_trials": incomplete,
        "selected_hard_tasks": len(eligible),
    }
    if not eligible:
        raise ValueError("no map-trials satisfied the dynamic hard-set criteria")
    return eligible, metadata


def parse_args() -> argparse.Namespace:
    experiments = load_experiments()
    parser = argparse.ArgumentParser(description="Run a complete FPV navigation experiment condition.")
    parser.add_argument("--experiment-type", required=True, choices=sorted(experiments))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL_NAME", ""))
    parser.add_argument("--api-base-url", default=os.environ.get("LLM_API_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--baseline-results", default="", help="baseline output directory used to derive an intervention hard set")
    parser.add_argument("--baseline-model", default="", help="model id to select when the baseline directory contains multiple models")
    parser.add_argument(
        "--hard-threshold",
        type=float,
        default=0.4,
        help="inclusive maximum baseline success rate for the dynamic hard set (default: 0.4)",
    )
    parser.add_argument("--minimum-baseline-runs", type=int, default=5)
    parser.add_argument("--task-file", default="", help="optional CSV or JSON with map_id and trial columns")
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--map", action="append", dest="map_ids", default=[])
    parser.add_argument("--trial", action="append", dest="trials", type=int, default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-policy", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def select_tasks(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if args.experiment_type == "intervention":
        if args.task_file:
            task_file = Path(args.task_file).expanduser().resolve()
            tasks = load_task_file(task_file)
            selection = {"source": str(task_file), "selection_method": "task_file"}
        elif args.baseline_results:
            tasks, selection = derive_hard_set_tasks(
                Path(args.baseline_results).expanduser().resolve(),
                threshold=args.hard_threshold,
                minimum_runs=args.minimum_baseline_runs,
                baseline_model=args.baseline_model,
            )
            selection["selection_method"] = "baseline_success_rate"
        else:
            raise ValueError("intervention requires --baseline-results or --task-file")
    else:
        tasks = load_all_tasks()
        selection = {"selection_method": "all_180_map_trials"}

    if args.map_ids:
        allowed_maps = set(args.map_ids)
        tasks = [task for task in tasks if task["map_id"] in allowed_maps]
    if args.trials:
        allowed_trials = {str(value) for value in args.trials}
        tasks = [task for task in tasks if task["trial"] in allowed_trials]
    if args.limit > 0:
        tasks = tasks[: args.limit]
    selection["selected_after_filters"] = len(tasks)
    return tasks, selection


def build_command(args: argparse.Namespace, task: dict[str, str], repeat: int, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(SINGLE_RUNNER),
        "--model",
        args.model or ("local-policy" if args.local_policy else ""),
        "--api-base-url",
        args.api_base_url,
        "--api-key-env",
        args.api_key_env,
        "--map",
        task["map_id"],
        "--trial",
        task["trial"],
        "--repeat",
        str(repeat),
        "--experiment-type",
        args.experiment_type,
        "--condition",
        args.condition,
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(args.max_steps),
        "--max-tokens",
        str(args.max_tokens),
    ]
    if args.local_policy:
        command.append("--local-policy")
    if args.visualize:
        command.append("--visualize")
    return command


def main() -> int:
    args = parse_args()
    experiments = load_experiments()
    definition = experiments[args.experiment_type]
    if args.condition not in definition["conditions"]:
        expected = ", ".join(definition["conditions"])
        raise SystemExit(f"unsupported condition for {args.experiment_type}: {args.condition}; expected one of: {expected}")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    prompt_mode = infer_prompt_mode(args.experiment_type, args.condition)
    prompt_template = infer_prompt_template(prompt_mode, args.experiment_type, args.condition)
    retain_goal, retain_traps = resolve_landmark_retention(args.experiment_type, args.condition)
    try:
        tasks, task_selection = select_tasks(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    repeats = list(range(args.repeat_start, args.repeat_start + args.repeats))
    condition_dir = args.condition.replace("/", "__")
    output_root = Path(args.output_root).expanduser().resolve()

    summary = {
        "experiment_type": args.experiment_type,
        "condition": args.condition,
        "model": args.model or ("local-policy" if args.local_policy else ""),
        "task_scope": definition["task_scope"],
        "task_selection": task_selection,
        "tasks": len(tasks),
        "repeats": repeats,
        "planned_runs": len(tasks) * len(repeats),
        "prompt_mode": prompt_mode,
        "prompt_template": str(prompt_template.relative_to(REPO_ROOT)),
        "initial_facing": infer_initial_facing(args.experiment_type),
        "retain_goal": retain_goal,
        "retain_traps": retain_traps,
        "preview": tasks[:10],
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    completed = 0
    failed = 0
    for task in tasks:
        for repeat in repeats:
            output_dir = (
                output_root
                / args.experiment_type
                / condition_dir
                / task["map_id"]
                / f"trial_{task['trial']}"
                / f"repeat_{repeat}"
            )
            command = build_command(args, task, repeat, output_dir)
            process = subprocess.run(command, check=False)
            if process.returncode == 0:
                completed += 1
            else:
                failed += 1

    result = {**summary, "completed": completed, "failed": failed}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
