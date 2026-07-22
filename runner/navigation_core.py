from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llm_api_client import LLMResponse


LLM_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_LLM_DIR = LLM_ROOT / "LlmFPV-main" / "LLM"
ORIGINAL_PROJECT_DIR = LLM_ROOT / "LlmFPV-main"

if str(ORIGINAL_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_LLM_DIR))

from utils.get_observation import accumulate_visibility, compute_visible_now  # noqa: E402
from utils.utils import build_graph, get_agent_height, map_pos_to_property, visualize_path  # noqa: E402


PROMPT_TEMPLATE_FILE = ORIGINAL_LLM_DIR / "prompts" / "prompt_template_v6.txt"
CM_COT_TEMPLATE_FILE = ORIGINAL_LLM_DIR / "prompts" / "cm_cot_legacy.txt"
INTERVENTION_PROMPT_FILES = {
    "none": ORIGINAL_LLM_DIR / "prompts" / "intervention_none.txt",
    "higher_elevation": ORIGINAL_LLM_DIR / "prompts" / "intervention_higher_elevation.txt",
    "wider_exploration": ORIGINAL_LLM_DIR / "prompts" / "intervention_wider_exploration.txt",
    "higher_elevation_wider_exploration": ORIGINAL_LLM_DIR / "prompts" / "intervention_combined.txt",
}
SELECTED_PATHS_FILE = ORIGINAL_LLM_DIR / "map_setting" / "selected_paths.json"
TRAP_LISTS_FILE = ORIGINAL_LLM_DIR / "map_setting" / "trap_lists.json"
XML_FILE = ORIGINAL_LLM_DIR / "map_setting" / "data.xml"
ANNOTATIONS_DIR = ORIGINAL_PROJECT_DIR / "organized_data" / "matched" / "annotations"
ANNOTATION_DIR_CANDIDATES = [
    ANNOTATIONS_DIR,
    ORIGINAL_PROJECT_DIR / "organized_data" / "matched",
]

ACTION_SPACE = ["MoveForward", "MoveBackward", "MoveLeftward", "MoveRightward", "TurnLeft", "TurnRight"]
MAP_SIZE = 9
MAX_STEPS_DEFAULT = 100
MAX_STAY = 5
MAX_INVALID = 5
MAX_VISIT = 10

MOVE_OFFSETS = {
    "MoveForward": {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1), "NORTH": (-1, 0), "SOUTH": (1, 0), "WEST": (0, -1), "EAST": (0, 1)},
    "MoveBackward": {"UP": (1, 0), "DOWN": (-1, 0), "LEFT": (0, 1), "RIGHT": (0, -1), "NORTH": (1, 0), "SOUTH": (-1, 0), "WEST": (0, 1), "EAST": (0, -1)},
    "MoveLeftward": {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (1, 0), "RIGHT": (-1, 0), "NORTH": (0, -1), "SOUTH": (0, 1), "WEST": (1, 0), "EAST": (-1, 0)},
    "MoveRightward": {"UP": (0, 1), "DOWN": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0), "NORTH": (0, 1), "SOUTH": (0, -1), "WEST": (-1, 0), "EAST": (1, 0)},
}
TURN_LEFT_MAP = {"UP": "LEFT", "DOWN": "RIGHT", "LEFT": "DOWN", "RIGHT": "UP", "NORTH": "WEST", "SOUTH": "EAST", "WEST": "SOUTH", "EAST": "NORTH"}
TURN_RIGHT_MAP = {"UP": "RIGHT", "DOWN": "LEFT", "LEFT": "UP", "RIGHT": "DOWN", "NORTH": "EAST", "SOUTH": "WEST", "WEST": "NORTH", "EAST": "SOUTH"}


class NonModelRunFailure(RuntimeError):
    def __init__(self, message: str, *, failure_type: str, step: int | None = None, retryable: bool = True):
        super().__init__(message)
        self.failure_type = failure_type
        self.step = step
        self.retryable = retryable


@dataclass
class GameState:
    position: list[int]
    facing: str
    agent_height: float
    stay_counter: int = 0
    invalid_counter: int = 0
    visit_counter: dict[int, int] | None = None
    last_position: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.visit_counter is None:
            self.visit_counter = {}

    def get_tile_id(self) -> int:
        return self.position[0] * MAP_SIZE + self.position[1]

    def update_position(self, new_pos: list[int], new_facing: str, new_height: float) -> None:
        new_pos_tuple = (new_pos[0], new_pos[1])
        if self.last_position is not None and new_pos_tuple == self.last_position:
            self.stay_counter += 1
        else:
            self.stay_counter = 0
        self.last_position = new_pos_tuple

        if new_pos != self.position:
            tile_id = new_pos[0] * MAP_SIZE + new_pos[1]
            self.visit_counter[tile_id] = self.visit_counter.get(tile_id, 0) + 1

        self.position = new_pos
        self.facing = new_facing
        self.agent_height = new_height
        self.invalid_counter = 0

    def increment_invalid(self) -> None:
        self.invalid_counter += 1


def predict_next_state(row: int, col: int, facing: str, action: str) -> tuple[int, int, str]:
    if action in MOVE_OFFSETS:
        delta_row, delta_col = MOVE_OFFSETS[action].get(facing, (0, 0))
        return row + delta_row, col + delta_col, facing
    if action == "TurnLeft":
        return row, col, TURN_LEFT_MAP.get(facing, facing)
    if action == "TurnRight":
        return row, col, TURN_RIGHT_MAP.get(facing, facing)
    return row, col, facing


def calculate_action_cost(action: str) -> float:
    return -1.0 if action.startswith("Move") else -0.2


def is_position_in_bounds(row: int, col: int) -> bool:
    return 0 <= row < MAP_SIZE and 0 <= col < MAP_SIZE


def is_move_reachable(current_tile_id: int, next_tile_id: int, neighbors_current: dict, neighbors_accumulated: dict) -> bool:
    current_neighbors = neighbors_current.get(str(current_tile_id), [])
    accumulated_neighbors = neighbors_accumulated.get(str(current_tile_id), [])
    return str(next_tile_id) in (set(current_neighbors) | set(accumulated_neighbors))


def load_trial_info(map_id: str, trial_num: int) -> dict[str, Any]:
    with SELECTED_PATHS_FILE.open("r", encoding="utf-8") as f:
        selected = json.load(f)
    trials = selected.get("maps", {}).get(str(map_id), {}).get("trials", [])
    for trial in trials:
        if int(trial.get("trial")) == int(trial_num):
            return trial
    raise ValueError(f"trial not found: map={map_id}, trial={trial_num}")


def load_all_traps(map_id: str) -> set[int]:
    with TRAP_LISTS_FILE.open("r", encoding="utf-8") as f:
        trap_tables = json.load(f)
    return {int(x) for x in trap_tables.get(str(map_id), [])}


def calculate_initial_agent_height(pos2property: dict[tuple[int, int], tuple[str, str]], start_pos: list[int]) -> float:
    start_type, start_height_str = pos2property.get((start_pos[0], start_pos[1]), ("normal", "normal"))
    if start_type == "flat":
        return float(int(start_height_str))
    if start_type == "slope":
        parts = start_height_str.split("|") if "|" in start_height_str else start_height_str.split("-")
        if len(parts) == 2:
            return (int(parts[0]) + int(parts[1])) / 2
    parts = start_height_str.split("/") if "/" in start_height_str else start_height_str.split("\\")
    if len(parts) == 2:
        return float(max(int(parts[0]), int(parts[1])))
    return 0.0


def get_known_traps_from_visible(visible_cells: list[list[int]], trap_set: set[int]) -> tuple[set[int], list[list[int]]]:
    known_trap_indices = set()
    known_trap_positions = []
    for row, col in visible_cells:
        tile_idx = row * MAP_SIZE + col
        if tile_idx in trap_set:
            known_trap_indices.add(tile_idx)
            known_trap_positions.append([row, col])
    return known_trap_indices, known_trap_positions


def create_failure_history_entry(step_num: int, state: GameState, reason: str) -> dict[str, Any]:
    return {
        "step": step_num,
        "state": [(state.position[0], state.position[1]), state.facing],
        "agent_height": state.agent_height,
        "action": "Null",
        "info": reason,
    }


def create_history_entry(step_num: int, state: GameState, action: str = "ToBeDecided") -> dict[str, Any]:
    return {
        "step": step_num,
        "state": [(state.position[0], state.position[1]), state.facing],
        "agent_height": state.agent_height,
        "action": action,
    }


def fmt_list(value: list[int] | tuple[int, int]) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def convert_idx_to_pos(idx: int | str) -> tuple[int, int]:
    idx_int = int(float(idx))
    return idx_int // MAP_SIZE, idx_int % MAP_SIZE


def process_adj_format(adj: dict[str, list[str]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    new_adj = {}
    for key, values in adj.items():
        try:
            key_pos = convert_idx_to_pos(key)
        except Exception:
            continue
        value_pos = []
        for value in values:
            try:
                value_pos.append(convert_idx_to_pos(value))
            except Exception:
                continue
        new_adj[key_pos] = value_pos
    return new_adj


def add_node_property(
    nodes: list[tuple[int, int]],
    pos2property: dict[tuple[int, int], tuple[str, str]],
) -> list[list[Any]]:
    return [[pos, pos2property.get(pos, ("normal", "normal"))] for pos in nodes]


def prepare_template(template_text: str, max_steps: int, preserve_failure_placeholders: bool = False) -> str:
    prepared = template_text.replace("[MAX_STEPS]", str(max_steps))
    if preserve_failure_placeholders:
        return prepared
    return (
        prepared.replace("[MAX_STAY]", str(MAX_STAY))
        .replace("[MAX_INVALID]", str(MAX_INVALID))
        .replace("[MAX_VISIT]", str(MAX_VISIT))
    )


def render_action_prompt(
    template_text: str,
    *,
    step_num: int,
    max_steps: int,
    goal_known: bool,
    goal_position: list[int],
    trap_known: bool,
    known_trap_list: list[list[int]],
    state: GameState,
    visible_now: list[list[int]],
    neighbors_now: dict[str, list[str]],
    visible_accumulated: list[list[int]],
    neighbors_accumulated: dict[str, list[str]],
    accumulated_cost: float,
    history: list[dict[str, Any]],
    pos_to_property: dict[tuple[int, int], tuple[str, str]],
) -> str:
    prompt = template_text
    prompt = prompt.replace("[STEP_NUM]", str(step_num))
    prompt = prompt.replace("[MAX_STEPS]", str(max_steps))
    prompt = prompt.replace("[CURRENT_POS]", str(fmt_list(state.position)))
    prompt = prompt.replace("[CURRENT_ORIENTATION]", state.facing)
    prompt = prompt.replace("[GOAL_POS]", str(fmt_list(goal_position)) if goal_known else "invisible")
    visible_traps = [fmt_list(x) for x in known_trap_list] if trap_known else []
    prompt = prompt.replace("[VISIBLE_TRAPS]", str(visible_traps) if visible_traps else "invisible")
    prompt = prompt.replace("[CURRENT_VISIBLE_NODES]", str(add_node_property([fmt_list(x) for x in visible_now], pos_to_property)))
    prompt = prompt.replace("[ACCUM_VISIBLE_NODES]", str(add_node_property([fmt_list(x) for x in visible_accumulated], pos_to_property)))
    prompt = prompt.replace("[CURRENT_VISIBLE_ADJ]", str(process_adj_format(neighbors_now)))
    prompt = prompt.replace("[ACCUM_VISIBLE_ADJ]", str(process_adj_format(neighbors_accumulated)))
    prompt = prompt.replace("[ACCUM_COST]", str(accumulated_cost))
    prompt = prompt.replace("[HISTORY]", str(history))
    return prompt


def render_cm_cot_prompt(
    template_text: str,
    *,
    step_num: int,
    max_steps: int,
    goal_known: bool,
    goal_position: list[int],
    trap_known: bool,
    known_trap_list: list[list[int]],
    state: GameState,
    visible_now: list[list[int]],
    neighbors_now: dict[str, list[str]],
    accumulated_cost: float,
    history: list[dict[str, Any]],
    previous_memory: str,
    pos_to_property: dict[tuple[int, int], tuple[str, str]],
) -> str:
    prompt = template_text
    prompt = prompt.replace("[STEP_NUM]", str(step_num))
    prompt = prompt.replace("[MAX_STEPS]", str(max_steps))
    prompt = prompt.replace("[CURRENT_POS]", str(fmt_list(state.position)))
    prompt = prompt.replace("[CURRENT_ORIENTATION]", state.facing)
    prompt = prompt.replace("[GOAL_POS]", str(fmt_list(goal_position)) if goal_known else "invisible")
    visible_traps = [fmt_list(x) for x in known_trap_list] if trap_known else []
    prompt = prompt.replace("[CURRENT_VISIBLE_TRAPS]", str(visible_traps) if visible_traps else "invisible")
    prompt = prompt.replace("[CURRENT_VISIBLE_NODES]", str(add_node_property([fmt_list(x) for x in visible_now], pos_to_property)))
    prompt = prompt.replace("[CURRENT_VISIBLE_ADJ]", str(process_adj_format(neighbors_now)))
    prompt = prompt.replace("[ACCUM_COST]", str(accumulated_cost))
    prompt = prompt.replace("[HISTORY]", str(history))
    prompt = prompt.replace("[PREVIOUS_MEMORY]", previous_memory)
    return prompt


def infer_prompt_mode(experiment_type: str, condition: str) -> str:
    if experiment_type == "cognitive_map_prompt" or condition.endswith("/cm_cot"):
        return "cm_cot"
    return "action_only"


def infer_prompt_template(prompt_mode: str, experiment_type: str = "", condition: str = "") -> Path:
    if experiment_type in {"intervention", "intervention_comparison"}:
        if condition not in INTERVENTION_PROMPT_FILES:
            supported = ", ".join(INTERVENTION_PROMPT_FILES)
            raise ValueError(f"unsupported intervention condition: {condition}; expected one of: {supported}")
        return INTERVENTION_PROMPT_FILES[condition]
    return CM_COT_TEMPLATE_FILE if prompt_mode == "cm_cot" else PROMPT_TEMPLATE_FILE


def infer_initial_facing(experiment_type: str) -> str:
    if experiment_type in {"cognitive_map_prompt", "goal_trap_memory", "intervention", "intervention_comparison"}:
        return "SOUTH"
    return "DOWN"


def resolve_landmark_retention(experiment_type: str, condition: str) -> tuple[bool, bool]:
    if experiment_type == "goal_trap_memory":
        supported = {"none", "goal_only", "trap_only", "all"}
        if condition not in supported:
            raise ValueError(f"unsupported goal_trap_memory condition: {condition}; expected one of: {', '.join(sorted(supported))}")
        return condition in {"goal_only", "all"}, condition in {"trap_only", "all"}
    if experiment_type == "cognitive_map_prompt":
        landmark_condition = condition.split("/", 1)[0]
        supported = {"none", "goal_only"}
        if landmark_condition not in supported:
            raise ValueError(
                f"unsupported cognitive_map_prompt landmark condition: {landmark_condition}; "
                f"expected one of: {', '.join(sorted(supported))}"
            )
        return landmark_condition == "goal_only", False
    return False, False


def update_landmark_memory(
    *,
    visible_cells: list[list[int]],
    goal_position: list[int],
    all_traps: set[int],
    goal_seen_ever: bool,
    retained_trap_ids: set[int],
    retain_goal: bool,
    retain_traps: bool,
) -> tuple[bool, set[int], bool, set[int], list[list[int]]]:
    current_goal_visible = goal_position in visible_cells
    current_trap_ids, current_trap_positions = get_known_traps_from_visible(visible_cells, all_traps)

    goal_seen_ever = goal_seen_ever or current_goal_visible
    if retain_traps:
        retained_trap_ids = retained_trap_ids | current_trap_ids
        effective_trap_ids = retained_trap_ids
        effective_trap_positions = [list(convert_idx_to_pos(tile_id)) for tile_id in sorted(effective_trap_ids)]
    else:
        effective_trap_ids = current_trap_ids
        effective_trap_positions = current_trap_positions

    goal_known = current_goal_visible or (retain_goal and goal_seen_ever)
    return goal_seen_ever, retained_trap_ids, goal_known, effective_trap_ids, effective_trap_positions


def extract_action_and_memory(content: str, prompt_mode: str) -> tuple[str, str | None, bool]:
    stripped = (content or "").strip()
    simple = stripped.strip('"').strip("'").strip()
    if simple in ACTION_SPACE:
        return simple, None, True

    code_block = re.search(r"```(?:json|text|txt)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
    candidates = [stripped]
    if code_block:
        candidates.insert(0, code_block.group(1).strip())

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            action = str(obj.get("action") or obj.get("next_action") or "").strip()
            memory = obj.get("updated_memory")
            if action in ACTION_SPACE:
                return action, str(memory) if memory is not None else None, True

    if prompt_mode == "cm_cot":
        match = re.search(r'"action"\s*:\s*"([^"]+)"', stripped)
        if match and match.group(1) in ACTION_SPACE:
            memory_match = re.search(r'"updated_memory"\s*:\s*"([\s\S]*?)"\s*[,}]', stripped)
            return match.group(1), memory_match.group(1) if memory_match else None, True

    return simple, None, False


def choose_local_action(
    state: GameState,
    goal_position: list[int],
    neighbors_now: dict[str, list[str]],
    neighbors_accumulated: dict[str, list[str]],
    all_traps: set[int],
) -> str:
    candidates = []
    for action in ["MoveForward", "MoveBackward", "MoveLeftward", "MoveRightward"]:
        next_row, next_col, _ = predict_next_state(state.position[0], state.position[1], state.facing, action)
        if not is_position_in_bounds(next_row, next_col):
            continue
        next_tile_id = next_row * MAP_SIZE + next_col
        if next_tile_id in all_traps:
            continue
        if not is_move_reachable(state.get_tile_id(), next_tile_id, neighbors_now, neighbors_accumulated):
            continue
        distance = abs(next_row - goal_position[0]) + abs(next_col - goal_position[1])
        candidates.append((distance, action))
    if candidates:
        return sorted(candidates)[0][1]
    return "TurnRight"


def validate_trial_setup(map_id: str, trial_num: int, prompt_template: Path) -> dict[str, Any]:
    trial_info = load_trial_info(map_id, trial_num)
    if not prompt_template.exists():
        raise FileNotFoundError(prompt_template)
    annotations_dir = resolve_annotations_dir(map_id)
    ann_file = annotations_dir / f"{map_id}.json"
    if not ann_file.exists():
        raise FileNotFoundError(ann_file)
    return {
        "map_id": str(map_id),
        "trial": int(trial_num),
        "start": trial_info["start_coordinates"],
        "goal": trial_info["end_coordinates"],
        "prompt_template": str(prompt_template),
        "annotation": str(ann_file),
    }


def resolve_annotations_dir(map_id: str) -> Path:
    env_dir = os.environ.get("LLM_ANNOTATIONS_DIR", "").strip()
    candidates = [Path(env_dir).expanduser()] if env_dir else []
    candidates.extend(ANNOTATION_DIR_CANDIDATES)
    for candidate in candidates:
        if candidate and (candidate / f"{map_id}.json").exists():
            return candidate
    raise FileNotFoundError(
        "No annotation JSON found for "
        f"map {map_id}. Set LLM_ANNOTATIONS_DIR or restore organized_data/matched/annotations."
    )


def run_navigation_episode(
    *,
    map_id: str,
    trial_num: int,
    repeat_idx: int,
    model: str,
    experiment_type: str,
    condition: str,
    output_dir: Path,
    call_model: Callable[[str, Path], LLMResponse] | None,
    prompt_mode: str,
    prompt_template: Path,
    memory_type: str,
    memory_size: int,
    max_steps: int = MAX_STEPS_DEFAULT,
    sleep_seconds: float = 0.0,
    initial_facing: str = "DOWN",
    local_policy: bool = False,
    visualize: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_text = prepare_template(
        prompt_template.read_text(encoding="utf-8"),
        max_steps,
        preserve_failure_placeholders=prompt_template == CM_COT_TEMPLATE_FILE,
    )

    trial_info = load_trial_info(map_id, trial_num)
    start_rc = [int(x) for x in trial_info["start_coordinates"]]
    goal_rc = [int(x) for x in trial_info["end_coordinates"]]
    all_traps = load_all_traps(map_id)
    annotations_dir = resolve_annotations_dir(map_id)
    graph = build_graph(map_id, annotations_dir)
    pos_to_property = map_pos_to_property(map_id, annotations_dir)
    initial_agent_height = calculate_initial_agent_height(pos_to_property, start_rc)

    state = GameState(position=start_rc.copy(), facing=initial_facing, agent_height=initial_agent_height)
    visible_cells_sequence = []
    visible_neighbors_sequence = []
    visible_now, neighbors_now = compute_visible_now(
        graph,
        (state.position[0], state.position[1]),
        state.facing,
        trap_set=set(),
        pos2property=pos_to_property,
        agent_height=state.agent_height,
    )
    visible_cells_sequence.append(visible_now)
    visible_neighbors_sequence.append(neighbors_now)
    visible_accumulated, neighbors_accumulated = accumulate_visibility(
        visible_cells_sequence,
        visible_neighbors_sequence,
        memory_type=memory_type,
        memory_size=memory_size,
    )
    retain_goal, retain_traps = resolve_landmark_retention(experiment_type, condition)
    goal_seen_ever = False
    retained_trap_ids: set[int] = set()
    goal_seen_ever, retained_trap_ids, goal_known, known_trap_set, known_trap_list = update_landmark_memory(
        visible_cells=visible_accumulated,
        goal_position=goal_rc,
        all_traps=all_traps,
        goal_seen_ever=goal_seen_ever,
        retained_trap_ids=retained_trap_ids,
        retain_goal=retain_goal,
        retain_traps=retain_traps,
    )

    accumulated_cost = 0.0
    history: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    token_usage: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    previous_memory = ""
    invalid_model_outputs: list[dict[str, Any]] = []

    for step_num in range(1, max_steps + 1):
        trap_known = len(known_trap_list) > 0
        observations.append(
            {
                "step": step_num,
                "current": state.position.copy(),
                "facing": state.facing,
                "visible_now": copy.deepcopy(visible_now),
                "adj_now": copy.deepcopy(neighbors_now),
                "visible_accum": copy.deepcopy(visible_accumulated),
                "adj_accum": copy.deepcopy(neighbors_accumulated),
                "goal_visible_now": goal_rc in visible_now,
                "goal_known": goal_known,
                "known_trap_list": copy.deepcopy(known_trap_list),
                "retain_goal": retain_goal,
                "retain_traps": retain_traps,
                "accum_cost": accumulated_cost,
            }
        )

        history.append(create_history_entry(step_num, state))
        if prompt_mode == "cm_cot":
            prompt_text = render_cm_cot_prompt(
                template_text,
                step_num=step_num,
                max_steps=max_steps,
                goal_known=goal_known,
                goal_position=goal_rc,
                trap_known=trap_known,
                known_trap_list=known_trap_list,
                state=state,
                visible_now=visible_now,
                neighbors_now=neighbors_now,
                accumulated_cost=accumulated_cost,
                history=history,
                previous_memory=previous_memory,
                pos_to_property=pos_to_property,
            )
        else:
            prompt_text = render_action_prompt(
                template_text,
                step_num=step_num,
                max_steps=max_steps,
                goal_known=goal_known,
                goal_position=goal_rc,
                trap_known=trap_known,
                known_trap_list=known_trap_list,
                state=state,
                visible_now=visible_now,
                neighbors_now=neighbors_now,
                visible_accumulated=visible_accumulated,
                neighbors_accumulated=neighbors_accumulated,
                accumulated_cost=accumulated_cost,
                history=history,
                pos_to_property=pos_to_property,
            )

        step_dir = output_dir / f"step_{step_num:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")

        if local_policy:
            raw_content = choose_local_action(state, goal_rc, neighbors_now, neighbors_accumulated, all_traps)
            llm_response = LLMResponse(content=raw_content)
        else:
            if call_model is None:
                raise NonModelRunFailure(
                    "call_model is missing",
                    failure_type="runner_missing_call_model",
                    step=step_num,
                    retryable=False,
                )
            try:
                llm_response = call_model(prompt_text, step_dir)
            except Exception as exc:
                failure_type = getattr(exc, "failure_type", "api_failure")
                retryable = bool(getattr(exc, "retryable", True))
                raise NonModelRunFailure(str(exc), failure_type=failure_type, step=step_num, retryable=retryable) from exc
            raw_content = llm_response.content
            token_usage.append(
                {
                    "step": step_num,
                    "prompt_tokens": llm_response.prompt_tokens,
                    "completion_tokens": llm_response.completion_tokens,
                    "reasoning_tokens": llm_response.reasoning_tokens,
                }
            )

        (step_dir / "raw.txt").write_text(raw_content or "", encoding="utf-8")
        action, updated_memory, parsed_ok = extract_action_and_memory(raw_content, prompt_mode)
        if updated_memory is not None:
            previous_memory = updated_memory

        if action not in ACTION_SPACE:
            history[-1]["info"] = "Invalid Action"
            history[-1]["action"] = action
            state.increment_invalid()
            invalid_model_outputs.append({"step": step_num, "raw": raw_content, "parsed_ok": parsed_ok})
            if state.invalid_counter >= MAX_INVALID:
                history.append(create_failure_history_entry(step_num + 1, state, f"Failure: TooManyInvalidActions ({state.invalid_counter})"))
                break
            continue

        next_row, next_col, next_facing = predict_next_state(state.position[0], state.position[1], state.facing, action)
        action_cost = calculate_action_cost(action)
        accumulated_cost = round(accumulated_cost + action_cost, 2)

        if not is_position_in_bounds(next_row, next_col):
            history[-1]["info"] = "Invalid Action"
            history[-1]["action"] = action
            state.increment_invalid()
            if state.invalid_counter >= MAX_INVALID:
                history.append(create_failure_history_entry(step_num + 1, state, f"Failure: TooManyInvalidActions ({state.invalid_counter})"))
                break
            continue

        next_tile_id = next_row * MAP_SIZE + next_col
        if next_tile_id in all_traps:
            history[-1]["info"] = "Falling into Trap"
            history[-1]["action"] = action
            history.append(
                {
                    "step": step_num + 1,
                    "state": [(next_row, next_col), next_facing],
                    "agent_height": -1,
                    "action": "Null",
                    "info": "Failure: FallenIntoTrap",
                }
            )
            break

        if action.startswith("Move") and not is_move_reachable(state.get_tile_id(), next_tile_id, neighbors_now, neighbors_accumulated):
            history[-1]["info"] = "Invalid Action"
            history[-1]["action"] = action
            state.increment_invalid()
            if state.invalid_counter >= MAX_INVALID:
                history.append(create_failure_history_entry(step_num + 1, state, f"Failure: TooManyInvalidActions ({state.invalid_counter})"))
                break
            continue

        start_tile_id = state.get_tile_id()
        steps.append({"step": step_num, "from": start_tile_id, "to": next_tile_id, "direction": next_facing, "action": action})

        if next_row != state.position[0] or next_col != state.position[1]:
            new_agent_height = get_agent_height(pos_to_property, state.position, [next_row, next_col], state.agent_height)
            if new_agent_height == -1:
                history[-1]["info"] = "Invalid Action"
                history[-1]["action"] = action
                state.increment_invalid()
                if state.invalid_counter >= MAX_INVALID:
                    history.append(create_failure_history_entry(step_num + 1, state, f"Failure: TooManyInvalidActions ({state.invalid_counter})"))
                    break
                continue
        else:
            new_agent_height = state.agent_height

        state.update_position([next_row, next_col], next_facing, new_agent_height)
        history[-1]["info"] = "Valid Action"
        history[-1]["action"] = action

        if state.stay_counter >= MAX_STAY:
            history.append(create_failure_history_entry(step_num + 1, state, f"Failure: StayedTooLongInOneTile {state.last_position} ({state.stay_counter})"))
            break

        if state.position == goal_rc:
            history.append(
                {
                    "step": step_num + 1,
                    "state": [(state.position[0], state.position[1]), state.facing],
                    "agent_height": state.agent_height,
                    "action": "Null",
                    "info": "ReachedGoal",
                }
            )
            break

        current_tile_id = state.get_tile_id()
        if state.visit_counter.get(current_tile_id, 0) > MAX_VISIT:
            history.append(
                create_failure_history_entry(
                    step_num + 1,
                    state,
                    f"Failure: TooManyVisitsToTile ({state.position}, count={state.visit_counter[current_tile_id]})",
                )
            )
            break

        visible_now, neighbors_now = compute_visible_now(
            graph,
            (state.position[0], state.position[1]),
            state.facing,
            trap_set=known_trap_set,
            pos2property=pos_to_property,
            agent_height=state.agent_height,
        )
        visible_cells_sequence.append(visible_now)
        visible_neighbors_sequence.append(neighbors_now)
        visible_accumulated, neighbors_accumulated = accumulate_visibility(
            visible_cells_sequence,
            visible_neighbors_sequence,
            memory_type=memory_type,
            memory_size=memory_size,
        )
        goal_seen_ever, retained_trap_ids, goal_known, known_trap_set, known_trap_list = update_landmark_memory(
            visible_cells=visible_accumulated,
            goal_position=goal_rc,
            all_traps=all_traps,
            goal_seen_ever=goal_seen_ever,
            retained_trap_ids=retained_trap_ids,
            retain_goal=retain_goal,
            retain_traps=retain_traps,
        )

        if sleep_seconds > 0 and not local_policy:
            time.sleep(sleep_seconds)

    reached_goal = state.position == goal_rc
    result = {
        "map_id": str(map_id),
        "trial": int(trial_num),
        "repeat_idx": int(repeat_idx),
        "model": "local-policy" if local_policy else model,
        "experiment_type": experiment_type,
        "condition": condition,
        "mode": prompt_mode,
        "start": start_rc,
        "goal": goal_rc,
        "known_traps": known_trap_list,
        "history_nodes": history,
        "token_usage": token_usage,
        "observations": observations,
        "memory_type": memory_type,
        "memory_size": memory_size,
        "retain_goal": retain_goal,
        "retain_traps": retain_traps,
        "steps": steps,
        "reached_goal": reached_goal,
        "num_steps": len(history),
        "accumulated_cost": accumulated_cost,
        "invalid_model_outputs": invalid_model_outputs,
        "runner_version": "publication-release-20260722",
    }
    result_path = output_dir / "result.json"
    tmp_path = output_dir / "result.json.tmp"
    tmp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(result_path)

    if visualize:
        visualize_path(map_id, trial_num, history, steps, known_trap_list, start_rc, goal_rc, accumulated_cost, result, output_dir, annotations_dir, XML_FILE)

    return result
