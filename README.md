# FPV-LLM-Navigation

Code, task materials, and public research data for:

> **Humans and language models differ in prospective information sampling during navigation**
>
> Jingmeng Chang, Shan Xu, Zi Yin, Xizhi Xiao, and Jia Liu

The study compares human and large-language-model navigation under text-mediated first-person partial observability (FPV), a global-view control (GV), externalized spatial traces, persistent landmark memory, and strategy-cue interventions.

## Public release

- Repository: <https://github.com/JingmenngChang/FPV-LLM-Navigation>
- Versioned data: <https://github.com/JingmenngChang/FPV-LLM-Navigation/releases/tag/v1.0.0-data>
- Data dictionary, inclusion rules, and provenance: [DATA.md](DATA.md)
- Machine-readable citation: [CITATION.cff](CITATION.cff)

The repository contains the executable task code, all 60 final maps, 180 fixed start–goal configurations, trap definitions, model prompts, map-generation metadata, and release-validation utilities. Large de-identified human data, archived raw model responses where retained, parsed trajectories/actions, ablations, interventions, baseline-agent outputs, and analysis/figure sources are distributed as versioned Release assets.

## Task overview

- Environment: 9 × 9 grid with flat, slope, and corner tiles.
- Observation: current position, orientation, locally visible tiles, tile properties, and directed adjacency.
- Partial observability: the complete map is never provided in the FPV condition.
- Main FPV initial orientation: `DOWN`.
- Maximum episode length: 100 steps by default.
- Actions: `MoveForward`, `MoveBackward`, `MoveLeftward`, `MoveRightward`, `TurnLeft`, and `TurnRight`.
- Movement cost: −1.0 per movement action.
- Turning cost: −0.2 per turning action.

Actions are relative to the agent's current orientation.

## Repository structure

```text
.
├── runner/                         # Reproducible FPV experiment runner
├── configs/experiments.json        # Reported conditions
├── LlmFPV-main/
│   ├── LLM/prompts/                # FPV, trace, intervention, and GV prompts
│   ├── LLM/map_setting/            # Start–goal and trap definitions
│   └── organized_data/...          # Final 60 map annotations
├── data/map_generation/            # Features, difficulty, selection metadata
├── map_generation/                 # WFC/PCG source retained from the study
├── scripts/verify_release.py       # Checks downloaded Release assets
├── release-manifest.json           # Counts, provenance, and privacy audit
├── SHA256SUMS                      # Digests for the versioned Release assets
├── DATA.md
├── LICENSE                         # MIT code license
└── LICENSE-DATA                    # CC BY 4.0 data/documentation license
```

## Installation

Python 3.11 or later is recommended.

```bash
python3 -m venv ".venv"
source ".venv/bin/activate"
python3 -m pip install -r "requirements.txt"
```

## Validate the repository

```bash
python3 "runner/validate_setup.py"
```

Expected summary:

```json
{
  "status": "ok",
  "maps": 60,
  "trials": 180,
  "annotations": 60
}
```

Run a trial without an external API call:

```bash
python3 "runner/run_single_trial.py" \
  --model "dry-run-model" \
  --map 851 \
  --trial 2 \
  --repeat 0 \
  --experiment-type main_fpv_baseline \
  --condition omniscient \
  --output-dir "outputs/dry_run" \
  --dry-run
```

Run a short deterministic environment smoke test:

```bash
python3 "runner/run_single_trial.py" \
  --model "local-policy" \
  --map 851 \
  --trial 2 \
  --repeat 0 \
  --experiment-type main_fpv_baseline \
  --condition omniscient \
  --output-dir "outputs/local_smoke" \
  --local-policy \
  --max-steps 3
```

## Run one LLM trial

Set endpoint credentials locally. Never commit them.

```bash
export LLM_API_BASE_URL="https://your-llm-endpoint.example/v1"
export LLM_API_KEY="set-locally"
export LLM_MODEL_NAME="your-model-id"

python3 "runner/run_single_trial.py" \
  --model "$LLM_MODEL_NAME" \
  --map 851 \
  --trial 2 \
  --repeat 0 \
  --experiment-type main_fpv_baseline \
  --condition omniscient \
  --output-dir "outputs/map_851_trial_2_repeat_0"
```

The endpoint must expose an OpenAI-compatible `/chat/completions` interface.

## Reported conditions

- `main_fpv_baseline`: action-only FPV navigation.
- `cognitive_map_prompt`: separate externalized spatial-knowledge analysis with `previous_memory` as an input and `action`, `updated_memory`, and reasoning as outputs.
- `goal_trap_memory`: one-step ordinary observation memory plus condition-specific persistent goal/trap landmarks.
- `intervention`: baseline task with a strategy cue for higher elevation, wider exploration, both, or neither.
- `global_view`: full directed graph and trap list, with a five-line path output format.

The exact prompts are experimental materials and should not be edited when reproducing the reported results.

## Download and verify the data

Download every asset from the `v1.0.0-data` Release into one directory, then run:

```bash
python3 "scripts/verify_release.py" \
  --asset-dir "/path/to/downloaded/assets" \
  --checksums "/path/to/downloaded/assets/SHA256SUMS"
```

See [DATA.md](DATA.md) for expected record counts and examples for streaming `.jsonl.gz` files.

## Reproducibility note

The final maps, their 84-feature table, difficulty estimates, 1,000-map candidate list, and generation code are public. The historical archive does not contain a single end-to-end script that reproduces the original 10,000→1,000→60 selection byte for byte. This limitation is documented rather than reconstructed retrospectively; see [DATA.md](DATA.md#known-map-selection-archive-limitation).

## Licenses

- Source code: [MIT](LICENSE)
- Data, prompts, figures, and documentation: [Creative Commons Attribution 4.0 International](LICENSE-DATA)

When using the data, cite the paper and this versioned release, and preserve the attribution and provenance notices.
