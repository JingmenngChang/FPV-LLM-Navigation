import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import ast
from matplotlib.collections import LineCollection

from map_setting.map2graph import map2digraph, nodeid2tileid
from map_setting.to2DGraph import Map3Dto2DConverter


ROTATE_CW = {
    'Up': 'Right',
    'Right': 'Down',
    'Down': 'Left',
    'Left': 'Up',
}


# 依赖函数（复用鲁棒 tiles->data）
def convert_tiles_to_data_robust(tiles_id, size=81):
    data = np.array(tiles_id)
    max_tile_id = data[:, 0].max()
    sum_tiles = int(max_tile_id) + 1
    ndata = data.reshape(size, 2)
    sum_rotations = 4
    tiledict = {}
    count = 0
    for tile_idx in range(sum_tiles):
        for rot_idx in range(sum_rotations):
            tiledict[str([tile_idx + 1, rot_idx])] = count
            count += 1
    newdata = []
    for item in ndata.tolist():
        tile_idx, rot_idx = int(item[0]), int(item[1])
        key = str([tile_idx + 1, rot_idx])
        if key in tiledict:
            newdata.append(tiledict[key])
        else:
            raise KeyError(f"Tile combination not found in dict: {key}")
    return newdata


def build_graph(map_id: str, anno_dir: str) -> nx.DiGraph:
    ann_file = Path(anno_dir) / f'{map_id}.json'
    data = json.load(open(ann_file, 'r'))['data']
    pairs = [(int(data[i]), int(data[i+1])) for i in range(0, len(data), 2)]
    # print('pairs:', pairs)

    processed = convert_tiles_to_data_robust(pairs, size=81)
    # print('processed:', processed)

    G = map2digraph(processed, size=81, corner_extra_node=True)
    # print('Graph nodes:', G.nodes)
    # print('Graph edges:', G.edges)

    return G


# 本地策略：允许后退与仅旋转（英文方向）

def local_policy_step(current: Tuple[int,int], goal: Tuple[int,int], adj_now: Dict[str, List[str]], adj_accum: Dict[str, List[str]], history: List[int], facing: str) -> Tuple[List[int], str]:
    from collections import Counter
    cr, cc = current
    gr, gc = goal
    a = cr * 9 + cc
    neigh_now = adj_now.get(str(a), [])
    neigh_acc = adj_accum.get(str(a), [])
    neigh_str = set(neigh_now) | set(neigh_acc)
    candidates: List[Tuple[int,int,str,int,int]] = []  # (nr,nc,dir,h,visited)
    visited_count = Counter(history)

    for ns in neigh_str:
        try:
            b = int(float(ns))
        except Exception:
            continue
        nr, nc = b // 9, b % 9
        dr, dc = nr - cr, nc - cc
        if abs(dr) + abs(dc) != 1:
            continue
        if dr == -1 and dc == 0:
            dname = 'Up'
        elif dr == 1 and dc == 0:
            dname = 'Down'
        elif dr == 0 and dc == -1:
            dname = 'Left'
        else:
            dname = 'Right'
        h = abs(nr - gr) + abs(nc - gc)
        candidates.append((nr, nc, dname, h, visited_count.get(b, 0)))

    if candidates:
        min_h = min(h for *_, h, _ in candidates)
        best_h = [c for c in candidates if c[3] == min_h]
        min_v = min(v for *_, v in best_h)
        nr, nc, dname, _, _ = [c for c in best_h if c[4] == min_v][0]
        return [nr, nc], dname

    # 无可行移动 => 仅旋转（顺时针）
    new_dir = ROTATE_CW.get(facing, 'Right')
    return [cr, cc], new_dir


def _make_segments(points: List[Tuple[float, float]]):
    segs = []
    for i in range(len(points) - 1):
        segs.append([points[i], points[i+1]])
    return np.array(segs)

def _plot_gradient_path(ax, path_rc: List[Tuple[int,int]], label: str):
    if len(path_rc) < 2:
        return
    # 转为中心坐标
    pts = [(c + 0.5, r + 0.5) for r, c in path_rc]
    segments = _make_segments(pts)
    # 渐变色
    values = np.linspace(0, 1, len(segments))
    lc = LineCollection(segments, cmap='viridis', array=values, linewidths=3, alpha=0.9)
    ax.add_collection(lc)
    # 起点终点圆点
    ax.plot(pts[0][0], pts[0][1], marker='o', color='green', markersize=10)
    ax.plot(pts[-1][0], pts[-1][1], marker='*', color='red', markersize=14)
    # 图例占位线
    ax.plot([], [], color='black', linewidth=3, label=label)

def _overlay_backtracks(ax, steps: List[Dict[str, Any]]):
    # 统计方向性边
    dir_counts: Dict[Tuple[int,int], int] = {}
    undirected_seen: Dict[frozenset, set] = {}
    for s in steps:
        if s.get('action') != 'move':
            continue
        a = int(s['from']); b = int(s['to'])
        dir_counts[(a,b)] = dir_counts.get((a,b), 0) + 1
        key = frozenset({a,b})
        if key not in undirected_seen:
            undirected_seen[key] = set()
        undirected_seen[key].add((a,b))
    # 找到存在双向经过的无向边
    back_edges = set()
    for key, dirs in undirected_seen.items():
        if len(dirs) >= 2:
            # 同一无向边两个方向都出现，标记为折返边
            for u,v in dirs:
                back_edges.add((u,v))
    # 叠加绘制折返段
    used_legend = False
    for s in steps:
        if s.get('action') != 'move':
            continue
        a = int(s['from']); b = int(s['to'])
        if (a,b) not in back_edges:
            continue
        ra, ca = a // 9, a % 9
        rb, cb = b // 9, b % 9
        x1, y1 = ca + 0.5, ra + 0.5
        x2, y2 = cb + 0.5, rb + 0.5
        ax.plot([x1, x2], [y1, y2], color='orange', linestyle='--', linewidth=4, alpha=0.9,
                label='Backtrack' if not used_legend else None)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, 'X', color='orange', fontsize=12, ha='center', va='center', fontweight='bold')
        used_legend = True


def visualize_path(map_id, trial_num, history, steps, known_trap_list, start_rc, goal_rc, accum_cost, result, run_dir, anno_dir, xml_file):
    # 绘图
    ann_data = json.load(open(anno_dir / f'{map_id}.json', 'r'))
    processed_grid = Map3Dto2DConverter(str(xml_file))._create_processed_grid(ann_data)
    fig, ax = plt.subplots(figsize=(10,10))
    converter = Map3Dto2DConverter(str(xml_file))
    converter.visualize_2d_map(processed_grid, ax=ax)

    path_rc = [i['state'][0] for i in history]
    if len(path_rc) >= 2:
        _plot_gradient_path(ax, path_rc, label='Stepwise Path')
        _overlay_backtracks(ax, steps)

    # 画已知陷阱（红色方块边框）
    for r, c in known_trap_list:
        rect = plt.Rectangle((c, r), 1, 1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)

    sr, sc = start_rc
    gr, gc = goal_rc
    ax.plot(sc + 0.5, sr + 0.5, 'go', markersize=15, label='Start')
    ax.plot(gc + 0.5, gr + 0.5, 'r*', markersize=20, label='Goal')

    # total steps从history中提取出所有的移动和转身动作的总数
    total_steps = sum(1 for entry in history if entry['action'].startswith(('Move', 'Turn')))

    ax.set_title(f'Map {map_id} (trial {trial_num}) | steps={total_steps} | costs={accum_cost} | reached={result["reached_goal"]}')
    ax.legend()
    out_png = run_dir / 'result.png'
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'结果已保存: {out_png}')



# ==================== Height Calculation ====================

def calculate_height_for_flat(prev_height: float, tile_height: int) -> float:
    """计算在平地上的高度"""
    if prev_height < tile_height and abs(prev_height - tile_height) >= 1:
        return -1
    return float(tile_height)


def calculate_height_for_slope(height_str: str) -> float:
    """计算在斜坡上的高度（取平均值）"""
    parts = height_str.split('|') if '|' in height_str else height_str.split('-')
    if len(parts) == 2:
        h1, h2 = int(parts[0]), int(parts[1])
        avg_height = (h1 + h2) / 2
        return avg_height
    return -1


def calculate_height_for_corner(prev_pos: List[int], next_pos: List[int], 
                                prev_height: float, height_str: str) -> float:
    """计算在拐角上的高度"""
    # 确认移动方向（从prev到next）
    delta_row = prev_pos[0] - next_pos[0]
    delta_col = prev_pos[1] - next_pos[1]
    delta_to_direction = {
        (-1, 0): 'UP', (1, 0): 'DOWN',
        (0, -1): 'LEFT', (0, 1): 'RIGHT'
    }
    approach_direction = delta_to_direction.get((delta_row, delta_col), None)
    
    parts = height_str.split('/') if '/' in height_str else height_str.split('\\')
    
    # 根据接近方向确定可能的高度
    if approach_direction == 'LEFT':
        possible_height = int(parts[0])
    elif approach_direction == 'RIGHT':
        possible_height = int(parts[1])
    elif approach_direction == 'UP':
        possible_height = int(parts[1]) if '\\' in height_str else int(parts[0])
    else:  # DOWN
        possible_height = int(parts[0]) if '\\' in height_str else int(parts[1])
    
    # 检查高度是否可达
    if prev_height >= possible_height:
        return float(possible_height)
    elif abs(prev_height - possible_height) < 1:
        return float(possible_height)
    else:
        return -1


def get_agent_height(pos2property: Dict[Tuple[int, int], Tuple[str, str]], 
                    prev_pos: List[int], next_pos: List[int], prev_height: float) -> float:
    """
    计算agent移动到新位置后的高度
    Returns: 新高度，如果不可达返回-1
    """
    next_type, next_height_str = pos2property.get(
        (next_pos[0], next_pos[1]), ('normal', 'normal')
    )
    
    if next_type == 'flat':
        return calculate_height_for_flat(prev_height, int(next_height_str))
    elif next_type == 'slope':
        return calculate_height_for_slope(next_height_str)
    else:  # 'corner'
        return calculate_height_for_corner(prev_pos, next_pos, prev_height, next_height_str)


# ==================== Map Property Parsing ====================

# 高度映射表
MAP_SIZE = 9
SLOPE_HEIGHTS = {21: [0, 1], 22: [1, 2], 23: [2, 3], 24: [3, 4], 25: [4, 5]}
CORNER_HEIGHTS = {
    6: [0, 1], 7: [1, 2], 8: [2, 3], 9: [3, 4], 10: [4, 5],
    11: [0, 2], 12: [1, 3], 13: [2, 4], 14: [3, 5],
    15: [0, 3], 16: [1, 4], 17: [2, 5],
    18: [0, 4], 19: [1, 5], 20: [0, 5]
}

def parse_tile_property(key_1: int, key_2: int) -> Tuple[str, str]:
    """
    解析瓦片属性
    Returns: (tile_type, height_str)
    """
    if key_1 in [0, 1, 2, 3, 4, 5]:
        return 'flat', f'{key_1}'
    
    elif key_1 in [21, 22, 23, 24, 25]:
        heights = SLOPE_HEIGHTS[key_1]
        # key_2 determines the direction of the slope
        if key_2 == 0:
            height_str = f'{heights[1]}|{heights[0]}'  # UP > DOWN
        elif key_2 == 1:
            height_str = f'{heights[1]}-{heights[0]}'  # LEFT > RIGHT
        elif key_2 == 2:
            height_str = f'{heights[0]}|{heights[1]}'  # UP < DOWN
        else:  # key_2 == 3
            height_str = f'{heights[0]}-{heights[1]}'  # LEFT < RIGHT
        return 'slope', height_str
    
    else:  # corner
        heights = CORNER_HEIGHTS[key_1]
        if key_2 == 0:
            height_str = f'{heights[1]}\\{heights[0]}'  # high \ low
        elif key_2 == 1:
            height_str = f'{heights[0]}/{heights[1]}'  # low / high
        elif key_2 == 2:
            height_str = f'{heights[0]}\\{heights[1]}'  # low \ high
        else:  # key_2 == 3
            height_str = f'{heights[1]}/{heights[0]}'  # high / low
        return 'corner', height_str


def map_pos_to_property(map_id: str, anno_dir: str) -> Dict[Tuple[int, int], Tuple[str, str]]:
    """
    构建地图位置到属性的映射
    Returns: dict mapping (row, col) -> (tile_type, height_str)
    """
    ann_file = Path(anno_dir) / f'{map_id}.json'
    with open(ann_file, 'r') as f:
        data = json.load(f)['data']
    
    pairs = [(int(data[i]), int(data[i+1])) for i in range(0, len(data), 2)]
    pos_to_property: Dict[Tuple[int, int], Tuple[str, str]] = {}
    
    for row in range(MAP_SIZE):
        for col in range(MAP_SIZE):
            tile_idx = row * MAP_SIZE + col
            key_1, key_2 = pairs[tile_idx]
            tile_type, height_str = parse_tile_property(key_1, key_2)
            pos_to_property[(row, col)] = (tile_type, height_str)
    
    return pos_to_property



# Parse contents in json format

def _find_code_block_json(text: str) -> Optional[str]:
    # 先尝试常见的 code fence（优先 json 指定）
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return m.group(1)
    # 有些输出可能把 JSON 放在单行的 ``` ... ```
    return None

def _extract_first_brace_json(text: str) -> Optional[str]:
    # 找到第一个 '{'，然后用栈寻找匹配的 '}'，以确保提取完整 json 块
    start = text.find('{')
    if start == -1:
        return None
    stack = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            stack += 1
        elif ch == '}':
            stack -= 1
            if stack == 0:
                return text[start:i+1]
    return None  # 未找到配对大括号

def _try_load_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def parse_json_action_memory(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    尝试从 LLM 的 output content 中解析出 {'action':..., 'updated_memory':...}
    返回 (action, updated_memory)；解析失败返回 (None, None)
    
    注意：此函数已被 parse_json_action_memory_reasoning 替代，保留用于向后兼容
    """
    reasoning, action, updated_memory = parse_json_action_memory_reasoning(content)
    return action, updated_memory


def parse_json_action_reasoning(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    尝试从 LLM 的 output content 中解析出 {'reasoning':..., 'action':...}
    返回 (reasoning, action)；解析失败返回 (None, None)
    """
    reasoning, action, updated_memory = parse_json_action_memory_reasoning(content)
    return reasoning, action


def parse_json_action_memory_reasoning(content: str) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    """
    尝试从 LLM 的 output content 中解析出 {'reasoning':..., 'action':..., 'updated_memory':...}
    返回 (reasoning, action, updated_memory)；解析失败返回 (None, None, None)
    
    reasoning 是一个字典，包含：
    - situation_analysis: 当前状况分析
    - goal_direction: 目标方向判断
    - path_planning: 路径规划
    - risks_consideration: 风险评估
    - decision: 决策依据
    """
    if not content:
        return None, None, None

    # 基本清理
    # 去除 BOM & 两端空白
    content = content.lstrip('\ufeff').strip()

    # DEBUG: 如果解析常常失败，建议把下面这行打开到日志中，帮助排查实际输出。
    # print("DEBUG content repr:", repr(content[:400]))

    # 1) 先查找 code block
    candidate = _find_code_block_json(content)
    if candidate:
        data = _try_load_json(candidate)
        if data:
            return data.get('reasoning'), data.get('action'), data.get('updated_memory')

    # 2) 然后尝试提取第一个成对的大括号 JSON 子串
    candidate = _extract_first_brace_json(content)
    if candidate:
        data = _try_load_json(candidate)
        if data:
            return data.get('reasoning'), data.get('action'), data.get('updated_memory')

        # 3) 回退尝试：用 ast.literal_eval（可处理单引号等 Python-literal 风格）
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed.get('reasoning'), parsed.get('action'), parsed.get('updated_memory')
        except Exception:
            pass

        # 4) 再回退：把外层的单引号换成双引号再试（谨慎）
        try:
            candidate2 = candidate.replace("'", '"')
            data = json.loads(candidate2)
            return data.get('reasoning'), data.get('action'), data.get('updated_memory')
        except Exception:
            pass

    # 5) 最后回退：如果 LLM 输出不是 JSON，但用了类似 "Action: ...\nUpdated Memory: ..." 的格式，用 regex 抽取
    # 兼容中英文冒号
    m_action = re.search(r"Action[:：]\s*([A-Za-z0-9_]+)", content)
    # Updated memory 可能是很长的段落，尝试捕获它直到文本结束或到另一个标签
    m_mem = re.search(r"(?:Updated Memory|Updated_Memory|updated_memory|Updated memory)[:：]\s*(.+)$", content, re.S | re.I)
    if m_action or m_mem:
        action_val = m_action.group(1) if m_action else None
        mem_val = m_mem.group(1).strip() if m_mem else None
        # reasoning 在非JSON格式中无法提取，返回None
        return None, action_val, mem_val

    # 如果都没有解析出有效结果，返回 None 并把 content 的 repr 提供为 hint
    # print("WARN: failed to parse JSON. content repr:", repr(content[:400]))
    return None, None, None
