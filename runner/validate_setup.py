from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LLM_DIR = REPO_ROOT / "LlmFPV-main" / "LLM"
ANNOTATIONS_DIR = REPO_ROOT / "LlmFPV-main" / "organized_data" / "matched" / "annotations"
EXPERIMENTS_FILE = REPO_ROOT / "configs" / "experiments.json"


def main() -> int:
    selected_paths_file = LLM_DIR / "map_setting" / "selected_paths.json"
    trap_lists_file = LLM_DIR / "map_setting" / "trap_lists.json"
    prompt_files = [
        LLM_DIR / "prompts" / "prompt_template_v6.txt",
        LLM_DIR / "prompts" / "cm_cot_template.txt",
        LLM_DIR / "prompts" / "cm_cot_legacy.txt",
        LLM_DIR / "prompts" / "intervention_none.txt",
        LLM_DIR / "prompts" / "intervention_higher_elevation.txt",
        LLM_DIR / "prompts" / "intervention_wider_exploration.txt",
        LLM_DIR / "prompts" / "intervention_combined.txt",
    ]

    selected_paths = json.loads(selected_paths_file.read_text(encoding="utf-8"))
    trap_lists = json.loads(trap_lists_file.read_text(encoding="utf-8"))
    maps = selected_paths.get("maps", {})
    map_ids = sorted(maps, key=int)

    missing_annotations = [map_id for map_id in map_ids if not (ANNOTATIONS_DIR / f"{map_id}.json").exists()]
    missing_trap_entries = [map_id for map_id in map_ids if map_id not in trap_lists]
    missing_prompts = [str(path.relative_to(REPO_ROOT)) for path in prompt_files if not path.exists()]
    trial_count = sum(len(map_data.get("trials", [])) for map_data in maps.values())
    experiments = json.loads(EXPERIMENTS_FILE.read_text(encoding="utf-8"))

    summary = {
        "status": "ok" if not (missing_annotations or missing_trap_entries or missing_prompts) else "invalid",
        "maps": len(map_ids),
        "trials": trial_count,
        "annotations": len(map_ids) - len(missing_annotations),
        "experiment_types": sorted(experiments),
        "missing_annotations": missing_annotations,
        "missing_trap_entries": missing_trap_entries,
        "missing_prompts": missing_prompts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
