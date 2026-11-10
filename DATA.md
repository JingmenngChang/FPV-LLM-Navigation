# Data and provenance

This document describes the public materials for **Humans and language models differ in prospective information sampling during navigation**. The immutable large-file release is `v1.0.0-data`:

<https://github.com/JingmenngChang/FPV-LLM-Navigation/releases/tag/v1.0.0-data>

All Release assets have SHA-256 digests in `SHA256SUMS` and structured provenance in `release-manifest.json`.

## Release assets

| Asset | Contents | Expected records |
|---|---|---:|
| `human-fpv-deidentified.jsonl.gz` | All cleaned human FPV trials, with analysis-inclusion flags | 8,263 (8,141 included) |
| `human-gv-deidentified.jsonl.gz` | Cleaned human global-view trials | 9,581 |
| `llm-primary-fpv.jsonl.gz` | Main FPV model episodes, raw step outputs, prompts, parsed actions, and trajectories | 8,100 |
| `llm-posthoc-fpv.jsonl.gz` | GPT-5.6 Sol and GPT-6 Astra post hoc FPV episodes | 1,800 |
| `llm-trace-and-memory-ablation.jsonl.gz` | Externalized-trace and persistent-landmark-memory episodes | 720 |
| `llm-interventions.jsonl.gz` | Frozen GPT-5 and Gemini intervention episodes | 1,400 |
| `llm-global-view.jsonl.gz` | Six models × 180 global-view configurations | 1,080 |
| `baseline-agent-outputs.tar.gz` | Q-learning and multiseed random-agent outputs and source | — |
| `analysis-and-figure-source.tar.gz` | Frozen metrics, statistical scripts, manifests, and figure builders | — |

`README-RELEASE.md`, `LICENSE-DATA`, `release-manifest.json`, and `SHA256SUMS` accompany the data assets.

## Streaming examples

The JSONL files are compressed and should be streamed instead of loaded wholly into memory.

```python
import gzip
import json

with gzip.open("llm-primary-fpv.jsonl.gz", "rt", encoding="utf-8") as source:
    for line in source:
        episode = json.loads(line)
        print(episode["model"], episode["map_id"], episode["trial"])
```

## Human data

### FPV inclusion

The source archive and public asset contain 8,263 cleaned trial records from 168 participants. Each record has an `analysis_inclusion` object. Filtering `analysis_inclusion.current_start_matched == true` yields exactly 8,141 records, matching the strict current-start analysis set reported in the manuscript. The remaining 122 records are retained for transparency instead of being silently discarded.

### GV inclusion

The public GV asset contains all 9,581 records in the frozen cleaned global-view archive, from 138 participants.

Every human record receives a unique public `record_id`. Repeated participant–map–trial combinations in the cleaned archive are retained as separate attempts rather than silently deduplicated; their aggregate count is reported in `release-manifest.json`.

### De-identification

- Original participant identifiers are replaced, in deterministic source order, with `FPV-P0001…` or `GV-P0001…`.
- No identifier crosswalk is created or retained.
- `_id`, `system_info`, session/device/network fields, and all keys containing `timestamp` are removed recursively.
- Absolute trial-start times are removed.
- Scientific variables and relative timing variables such as `t`, `relative_time`, `completion_time`, and `first_movement_latency` are retained.
- Trajectories, projected grid paths, view directions, key events, start/end positions, outcomes, and task identifiers are retained.
- FPV records include the canonical trial inferred from the actual current start, whether that start matches one of the 180 formal configurations, and whether the reported trial label agrees with that canonical trial.

The two participant namespaces are intentionally separate; the release does not assert that a code in one view condition maps to a code in the other.

## FPV model episode schema

Each line in an FPV model asset is one episode:

```json
{
  "dataset": "llm_fpv",
  "experiment": "main_fpv_baseline",
  "condition": "omniscient",
  "model": "GPT-5",
  "map_id": "851",
  "trial": 2,
  "repeat": 0,
  "result": {},
  "step_io_archive_status": "archived_steps",
  "step_io": [
    {
      "step": 1,
      "prompt": "...",
      "prompt_archive_status": "archived_text",
      "raw_output": "...",
      "raw_output_archive_status": "archived_text"
    }
  ],
  "source_result_sha256": "..."
}
```

`result` is the frozen parsed episode record and includes observations, histories, parsed actions, trajectories, success/failure state, step counts, costs, and available token-use fields. Where the historical archive retained them, `step_io` preserves the exact prompt and raw model text selected by the archived parser. The four comparison-model step archives were restored from their compressed text mirrors, but only episodes represented by the frozen local 900-result set for each model are included. A missing historical prompt or response is never reconstructed: the value is `null`, or the episode has an empty `step_io`, and its archive status is `not_archived`. In particular, some GPT-5 requests and empty-response artifacts were not retained in the source archive. Exact missing-field counts are reported in `release-manifest.json`.

Duplicate API response envelopes and request metadata are excluded because they add credentials/transport risk without adding scientific content.

For each of DeepSeek, GLM, Kimi, and Qwen, step-level prompt/raw mirrors were retained for 720 of the 900 frozen episodes; the remaining 180 episodes have their parsed `result` but no archived step text. All 3,600 frozen result files were matched byte-for-byte by SHA-256 to the server result archive. An additional 84 DeepSeek results present only in the server archive are excluded because they are not members of the frozen 900-result analysis set. The machine-readable audit is [`data/model_archive_audit.json`](data/model_archive_audit.json).

### Main FPV counts

| Model | Episodes |
|---|---:|
| GPT-5 | 3,600 |
| Gemini 2.5 Pro | 900 |
| DeepSeek-V3.1-Thinking | 900 |
| GLM-4.5 | 900 |
| Kimi-K2-Thinking | 900 |
| Qwen3-235B-A22B | 900 |

### Post hoc FPV counts

| Model | Episodes |
|---|---:|
| GPT-5.6 Sol | 900 |
| GPT-6 Astra | 900 |

## Externalized trace and persistent memory

`llm-trace-and-memory-ablation.jsonl.gz` contains:

- 180 formal GPT-5 externalized-trace episodes, selected by the frozen Figure 5 trial-level manifest after excluding eight historical repeat-1 residual files.
- 540 GPT-5 persistent-memory episodes: 180 configurations each for `none`, `goal_only`, and `trap_only`, using `repeat = 0` as reported in the final analysis.

The externalized-trace protocol is separate from the main action-only evaluation. `previous_memory` is an input field. The response contains `action`, `updated_memory`, and reasoning (including `situation_analysis` where present); `previous_memory` is not a third model-generated output.

The archive `analysis-and-figure-source.tar.gz` contains the frozen coordinate–feature extraction tables, normalized-progress curves, configuration-level metrics, and recomputation scripts.

## Interventions

`llm-interventions.jsonl.gz` uses the exact result paths and SHA-256 values in the frozen canonical manifests:

- GPT-5: 14 hard configurations × 10 repeats × 4 conditions = 560 episodes.
- Gemini 2.5 Pro: 21 configurations × 10 repeats × 4 conditions = 840 episodes.

Conditions are `none`, `higher_elevation`, `wider_exploration`, and `higher_elevation_wider_exploration`. The Gemini set combines the 555 parser-fix reruns and 285 eligible legacy episodes selected by the canonical manifest; noncanonical batches are excluded.

## Global-view model data

`llm-global-view.jsonl.gz` contains one response for each of 180 start–goal configurations for GPT-5, Gemini 2.5 Pro, DeepSeek-V3.1-Thinking, GLM-4.5, Kimi-K2-Thinking, and Qwen3-235B-A22B. Each record includes the exact prompt, raw output, parsed/validated path, available reasoning text, token usage, and a source hash.

Kimi-K2-Thinking and Qwen3-235B-A22B use the exact-version 2026-08-25 reruns; they are not cross-variant sensitivity analyses.

## Baseline agents

`baseline-agent-outputs.tar.gz` contains the frozen Q-learning and multiseed random-agent scientific outputs together with their source code. Execution logs are excluded because they duplicate outcomes while carrying absolute run times and local machine paths; those fields are not analysis inputs. Archive member metadata are normalized, and absolute times, emails, credentials, and machine-specific paths are removed from retained text files.

## Maps and task definitions

The repository includes:

- `LlmFPV-main/organized_data/matched/annotations/`: 60 final map annotations.
- `LlmFPV-main/LLM/map_setting/selected_paths.json`: 180 start–goal definitions.
- `LlmFPV-main/LLM/map_setting/trap_lists.json`: trap definitions for all final maps.
- `data/map_generation/Features_final.csv`: frozen original-language source table for 10,000 maps (84 columns including filename and difficulty rate).
- `data/map_generation/Features_final_translated.csv`: analysis-ready English-column table with filename, difficulty rate, and 84 objective complexity features.
- `data/map_generation/difficulty_per_image_final.csv`: frozen difficulty estimates.
- `data/map_generation/final_60_map_ids.csv`: final-map selection list.
- `data/map_generation/selected_maps_1000.json`: archived 1,000-map candidate list.
- `data/map_generation/generation_stats.json`: map-generation summary.
- `map_generation/`: WFC/PCG source code retained from the study.

### Known map-selection archive limitation

The available archive does not contain a single end-to-end historical script that reproduces the original 10,000→1,000→60 feature-selection, Lasso, clustering, and final sampling sequence byte for byte. The public release therefore provides the frozen 84-feature table, difficulty table, archived 1,000-map list, final 60 IDs, final maps, and generation code, plus the contemporaneous audit. It does not fabricate a retrospective reconstruction of the missing orchestration script. In that audit, `${MAP_GENERATION_ARCHIVE_ROOT}` denotes the historical map-generation workspace whose relevant public files are now under `data/map_generation/`, `map_generation/`, and `LlmFPV-main/`.

## Analysis and figure source

`analysis-and-figure-source.tar.gz` contains the frozen scripts and data used for the main and supplementary figures, including the final no-Sol main-figure sources, cross-model ordering, visible-tile analysis, cognitive-map and memory metrics, intervention recomputation, exact-version GV comparisons, and supplementary figure rebuilds. Local or remote machine paths in text files are replaced with `${NAVIGATION_PROJECT_ROOT}`, `${ARCHIVE_ROOT}`, `${HOME}`, `${REMOTE_ARCHIVE_ROOT}`, `${REMOTE_HOST}`, or `${SSH_KEY_FILE}`; configure these placeholders before rerunning a historical script.

## Integrity and privacy checks

The release builder verifies:

- exact expected record counts and per-model/condition counts;
- unique episode keys;
- frozen source hashes where manifests provide them;
- readable gzip streams and valid JSON on every line;
- readable tar archives;
- absence of original participant hashes, timestamp fields, epoch-like absolute times, credentials, and local user paths;
- GitHub's per-asset size limit.

Use `scripts/verify_release.py` and `SHA256SUMS` after downloading the assets.

## Licensing and citation

Data, prompts, figures, and documentation are licensed under CC BY 4.0. Code is licensed under MIT. Cite the paper and the `v1.0.0-data` release when reusing the materials.
