import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import networkx as nx
import random

# 常量参数
GRID_ROWS = 9
GRID_COLS = 9
CELL_SIZE = 2.0  # 米
LAYER_HEIGHT = 1.0
CAMERA_HEIGHT = 1.7
FOV_DEG = 90.0  # 视野角度

FACING_TO_YAW_DEG = {
    'UP': -90.0,
    'RIGHT': 0.0,
    'DOWN': 90.0,
    'LEFT': 180.0,
    'NORTH': -90.0,
    'EAST': 0.0,
    'SOUTH': 90.0,
    'WEST': 180.0,
}

def grid_center_world(row: int, col: int) -> Tuple[float, float, float]:
    x = (col - 4) * CELL_SIZE
    z = (row - 4) * CELL_SIZE
    y = CAMERA_HEIGHT
    return x, y, z

# Bresenham 栅格穿越（行列空间）

def bresenham_line(r0, c0, r1, c1):
    r0 = int(r0); c0 = int(c0); r1 = int(r1); c1 = int(c1)
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = (dr - dc)
    r, c = r0, c0
    cells = []
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells

# 获取某个起点节点的所有邻居（统一为字符串比较）

def _neighbor_strs(G: nx.DiGraph, node_any) -> set:
    neighbor_set = set()
    candidates = []
    candidates.append(node_any)
    try:
        candidates.append(str(node_any))
    except Exception:
        pass
    try:
        candidates.append(float(node_any))
    except Exception:
        pass
    for cand in candidates:
        if G.has_node(cand):
            try:
                for nb in G.neighbors(cand):
                    neighbor_set.add(str(nb))
            except Exception:
                continue
    return neighbor_set

# 基于 G 的边判断两格连通（考虑整数/字符串/角点）

def cells_connected(G: nx.DiGraph, r0: int, c0: int, r1: int, c1: int) -> bool: # (r0,c0) -> (r1,c1)
    a = r0 * 9 + c0
    b = r1 * 9 + c1
    neigh_strs = _neighbor_strs(G, a)
    b_candidates = [str(b), f"{b}.1", f"{b}.2"]
    for bc in b_candidates:
        if bc in neigh_strs:
            return True
    return False

def corner_cells_connected(G: nx.DiGraph, nr: int, nc: int, corner_id: str) -> bool:
    corner_neighbors = _neighbor_strs(G, corner_id)
    b = nr * 9 + nc
    b_candidates = [str(b), f"{b}.1", f"{b}.2"]
    for bc in b_candidates:
        if bc in corner_neighbors:
            return True
    return False

# 角度判断：面向方向 yaw 与目标方向夹角

def within_fov(current_row: int, current_col: int, facing: str, target_row: int, target_col: int, fov_deg: float) -> bool:
    cx, cy, cz = grid_center_world(current_row, current_col)
    tx, ty, tz = grid_center_world(target_row, target_col)
    vx = tx - cx
    vz = tz - cz
    yaw = FACING_TO_YAW_DEG[facing]
    ang = math.degrees(math.atan2(vz, vx))
    delta = (ang - yaw + 180) % 360 - 180
    return abs(delta) <= fov_deg / 2.0

# 视线是否被阻挡：沿直线遍历格子，如果相邻格之间无边，则被挡

def line_of_sight_clear(G: nx.DiGraph, r0: int, c0: int, r1: int, c1: int, pos2property: Dict[Tuple[int, int], Tuple[str, str]]) -> bool:
    cells = bresenham_line(r0, c0, r1, c1)  ##=TODO：目前看到的可能过多了，考虑在视线上多取一些格子；或者设置最远观察距离
    # print('LOS cells:', cells)
    heights_in_line = []
    for (r, c) in cells:
        cur_type, cur_height = pos2property.get((r, c), ('normal', 'normal'))
        if cur_type == 'flat':
            h = int(cur_height)
        elif cur_type == 'slope':
            # cur_height 形如 "4|5" 或 "4-5", 取较高的值
            if '|' in cur_height:
                parts = cur_height.split('|')
            else:
                parts = cur_height.split('-')
            h = max(int(p) for p in parts)
        else: # 'corner'
            # cur_height 形如 "4/5" or "4\5", 取较高的值
            if '/' in cur_height:
                parts = cur_height.split('/')
            else:
                parts = cur_height.split('\\')
            h = max(int(p) for p in parts)
        heights_in_line.append(h)
    # print('heights_in_line:', heights_in_line)

    # 分析heights_in_line，判断视线是否被阻挡：（1）起点与终点之间的高度单调递减；（2）在起点和终点之间没有比终点更高的格子；（3）从起点出发一直减高度，当高度开始上升之后就一直单调递增直到终点（U型）
    visible = False
    if all(heights_in_line[i] >= heights_in_line[i+1] for i in range(len(heights_in_line)-1)):
        # print('1')
        visible = True
    if not visible:
        end_height = heights_in_line[-1]
        if all(h < end_height for h in heights_in_line[:-1]):
            # print('2')
            visible = True
    if not visible:
        # print('3')
        # print('heights_in_line:', heights_in_line)
        min_height = heights_in_line[0]
        started_increasing = False
        cur_visible = True
        for h in heights_in_line[1:]:
            if not started_increasing:
                if h <= min_height:
                    min_height = h
                elif h > min_height:
                    min_height = h
                    started_increasing = True
            else:
                if h < min_height:
                    cur_visible = False
                    break
                else:
                    min_height = h
        visible = cur_visible
    return visible


# 计算本轮可见域：返回可见格集合与它们之间的可见边（邻接表）

def compute_visible_now(G: nx.DiGraph, current: Tuple[int,int], facing: str, trap_set: set[int], 
                        pos2property: Dict[Tuple[int, int], Tuple[str, str]], agent_height: int) -> Tuple[List[List[int]], Dict[str, List[str]]]:
    cr, cc = current
    visible_cells = []

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # 当前位置总是可见
            if r == cr and c == cc:
                visible_cells.append([r, c])
                continue
            # 左右两边总是可见，根据facing方向确定左右两边tile的坐标
            delta_by_facing = {
                'UP': [(0, -1), (0, 1)],
                'DOWN': [(0, 1), (0, -1)],
                'LEFT': [(-1, 0), (1, 0)],
                'RIGHT': [(1, 0), (-1, 0)],
                'NORTH': [(0, -1), (0, 1)],
                'SOUTH': [(0, 1), (0, -1)],
                'WEST': [(-1, 0), (1, 0)],
                'EAST': [(1, 0), (-1, 0)],
            }
            adjacent_pos = [(cr + dr, cc + dc) for dr, dc in delta_by_facing[facing]]
            if (r, c) in adjacent_pos:
                visible_cells.append([r, c])
                continue
            # 其他格子需要判断是否在视野范围内且视线清晰
            if not within_fov(cr, cc, facing, r, c, FOV_DEG):
                continue  # 不在FOV_DEG度视野角内，跳过
            if line_of_sight_clear(G, cr, cc, r, c, pos2property):
                visible_cells.append([r, c])  # 视线清晰且在FOV内
    # print("visible_cells:", visible_cells)

    visible_set = set((r, c) for r, c in visible_cells)
    # print("visible_set:", visible_set)

    adj: Dict[str, List[str]] = {}
    for (r, c) in visible_set:
        cur_type, cur_height = pos2property.get((r, c), ('normal', 'normal'))
        if cur_type == 'flat' or cur_type == 'slope':
            # 正常从邻接表中获得adjacency
            a = r * 9 + c
            key = str(a)
            adj[key] = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and (nr, nc) in visible_set:
                    if cells_connected(G, r, c, nr, nc):
                        b = nr * 9 + nc
                        adj[key].append(str(b))
        else: # 'corner'
            # 假如agent处于corner上，则根据agent的位置高度判断可达性；假如agent不在corner上，取并集
            parts = cur_height.split('/') if '/' in cur_height else cur_height.split('\\')
            h1, h2 = int(parts[0]), int(parts[1]) # h1对应0.1, h2对应0.2
            cur_tile_id = r * 9 + c
            if (r, c) == (cr, cc):  # agent在corner上 
                if agent_height == h1:
                    cur_tile_id = f"{cur_tile_id}.1"
                else:
                    cur_tile_id = f"{cur_tile_id}.2"
                key = str(r*9 + c)
                adj[key] = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and (nr, nc) in visible_set:
                        if corner_cells_connected(G, nr, nc, cur_tile_id):
                            b = nr * 9 + nc
                            adj[key].append(str(b))

            else:  # corner在可见域内，但是agent不在corner上，取并集
                key = str(r*9 + c)
                adj[key] = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and (nr, nc) in visible_set:
                        if corner_cells_connected(G, nr, nc, f"{cur_tile_id}.1") or corner_cells_connected(G, nr, nc, f"{cur_tile_id}.2"):
                            b = nr * 9 + nc
                            adj[key].append(str(b))

    # print("adj:", adj)
    return visible_cells, adj



def merge_visibility(cells_seq: List[List[List[int]]], neighbors_seq: List[Dict[str, List[str]]]):
    """将多个时间步的可见cells和neighbors合并、去重"""
    accum_visible_cells = []
    accum_visible_neighbors = {}

    for visible_cells, visible_neighbors in zip(cells_seq, neighbors_seq):
        # 累积 cells
        accum_visible_cells.extend(visible_cells)
        # 累积 neighbors
        for key, values in visible_neighbors.items():
            if key not in accum_visible_neighbors:
                accum_visible_neighbors[key] = set(values)
            else:
                accum_visible_neighbors[key].update(values)

    # 去重 cells 并恢复为 list
    accum_visible_cells = [list(cell) for cell in {tuple(c) for c in accum_visible_cells}]
    # 将 neighbor 的 set 转回 list
    accum_visible_neighbors = {k: list(v) for k, v in accum_visible_neighbors.items()}

    return accum_visible_cells, accum_visible_neighbors


# 累计可见域聚合

def accumulate_visibility(visible_cells_sequence: List[List[List[int]]], visible_neighbors_sequence: List[Dict[str, List[str]]],
                            memory_type: str, memory_size: int,
                            decay_lambda: float = 0.2):

    total_steps = len(visible_cells_sequence)

    if memory_type == 'omniscient':
        # 全知全能：合并所有
        cells_seq = visible_cells_sequence
        neighbors_seq = visible_neighbors_sequence

    elif memory_type == 'recent':
        # 最近 memory_size 步
        cells_seq = visible_cells_sequence[-memory_size:]
        neighbors_seq = visible_neighbors_sequence[-memory_size:]

    elif memory_type == 'random_forget':
        # 随机遗忘 memory_size 步
        sampled_indices = random.sample(range(total_steps), k=min(memory_size, total_steps))
        sampled_indices.sort()  # 保持时间顺序
        cells_seq = [visible_cells_sequence[i] for i in sampled_indices]
        neighbors_seq = [visible_neighbors_sequence[i] for i in sampled_indices]

    elif memory_type == 'recall_by_recency':
        # 越靠近当前时间步，被保留概率越高
        probs = [math.exp(-decay_lambda * (total_steps - 1 - i)) for i in range(total_steps)]
        # 归一化为最大概率=1（非必须，但更直观）
        max_p = max(probs)
        probs = [p / max_p for p in probs]

        sampled_indices = [
            i for i, p in enumerate(probs)
            if random.random() < p
        ]
        # 如果选得太少，确保至少保留 memory_size 步
        if len(sampled_indices) < memory_size:
            extra_needed = memory_size - len(sampled_indices)
            all_indices = set(range(total_steps))
            remaining = list(all_indices - set(sampled_indices))
            sampled_indices.extend(random.sample(remaining, k=min(extra_needed, len(remaining))))

        sampled_indices.sort()
        cells_seq = [visible_cells_sequence[i] for i in sampled_indices]
        neighbors_seq = [visible_neighbors_sequence[i] for i in sampled_indices]

    elif memory_type == 'recall_by_block':
        return


    elif memory_type == 'recall_by_saliency':
        return


    elif memory_type == 'recall_by_saliency_and_block':
        return

    else:
        raise ValueError(f"Unknown memory type: {memory_type}")

    return merge_visibility(cells_seq, neighbors_seq)
