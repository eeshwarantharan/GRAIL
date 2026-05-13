from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import numpy as np

from gnss_vim_sim.planning.mission import Waypoint
from gnss_vim_sim.world.scene import MeshScene


@dataclass(frozen=True)
class RoutePlan:
    waypoints: list[Waypoint]
    planner_used: str
    obstacle_count: int


def plan_safe_route(
    requested: list[Waypoint],
    scene: MeshScene,
    *,
    planner: str = "astar",
    grid_m: float = 6.0,
    clearance_m: float = 10.0,
) -> RoutePlan:
    if planner.lower() in {"none", "direct", "off"} or len(requested) < 2:
        return RoutePlan(requested, "direct", 0)
    if len(scene.vertices) == 0:
        return RoutePlan(requested, "direct_no_mesh", 0)

    obstacle_xy = _obstacle_points(scene)
    if len(obstacle_xy) == 0:
        return RoutePlan(requested, "direct_no_obstacles", 0)

    planned: list[Waypoint] = [requested[0]]
    used = False
    for a, b in zip(requested[:-1], requested[1:]):
        segment = _astar_segment(a.position, b.position, obstacle_xy, grid_m, clearance_m)
        if segment is None:
            planned.append(b)
            continue
        used = True
        for idx, pos in enumerate(segment[1:], start=1):
            is_last = idx == len(segment) - 1
            name = b.name if is_last else f"route_{len(planned):03d}"
            planned.append(Waypoint(name=name, position=pos))

    return RoutePlan(planned, "astar" if used else "direct_fallback", int(len(obstacle_xy)))


def _obstacle_points(scene: MeshScene) -> np.ndarray:
    verts = np.asarray(scene.vertices)
    if len(verts) == 0:
        return np.empty((0, 2))
    z = verts[:, 2]
    high = verts[z > 1.5]
    if len(high) == 0:
        return np.empty((0, 2))
    return high[:, :2]


def _astar_segment(
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_xy: np.ndarray,
    grid_m: float,
    clearance_m: float,
) -> list[np.ndarray] | None:
    margin = max(80.0, clearance_m * 4.0)
    min_xy = np.minimum(start[:2], goal[:2]) - margin
    max_xy = np.maximum(start[:2], goal[:2]) + margin
    nearby = obstacle_xy[
        (obstacle_xy[:, 0] >= min_xy[0])
        & (obstacle_xy[:, 0] <= max_xy[0])
        & (obstacle_xy[:, 1] >= min_xy[1])
        & (obstacle_xy[:, 1] <= max_xy[1])
    ]
    if len(nearby) == 0 or _line_clear(start[:2], goal[:2], nearby, clearance_m):
        return [start, goal]

    origin = min_xy
    shape = np.ceil((max_xy - min_xy) / grid_m).astype(int) + 1
    if shape[0] * shape[1] > 90_000:
        return None

    blocked = np.zeros((shape[0], shape[1]), dtype=bool)
    obstacle_cells = np.round((nearby - origin) / grid_m).astype(int)
    inflate = max(1, int(math.ceil(clearance_m / grid_m)))
    for cx, cy in obstacle_cells:
        x0, x1 = max(0, cx - inflate), min(shape[0], cx + inflate + 1)
        y0, y1 = max(0, cy - inflate), min(shape[1], cy + inflate + 1)
        blocked[x0:x1, y0:y1] = True

    start_cell = _cell(start[:2], origin, grid_m, shape)
    goal_cell = _cell(goal[:2], origin, grid_m, shape)
    blocked[start_cell] = False
    blocked[goal_cell] = False
    path_cells = _astar_cells(start_cell, goal_cell, blocked)
    if path_cells is None:
        return None

    simple = _simplify_cells(path_cells)
    points = []
    for i, cell in enumerate(simple):
        xy = origin + np.array(cell, dtype=float) * grid_m
        frac = i / max(len(simple) - 1, 1)
        z = (1.0 - frac) * start[2] + frac * goal[2]
        points.append(np.array([xy[0], xy[1], z], dtype=float))
    points[0] = start
    points[-1] = goal
    return points


def _cell(xy: np.ndarray, origin: np.ndarray, grid_m: float, shape: np.ndarray) -> tuple[int, int]:
    c = np.round((xy - origin) / grid_m).astype(int)
    return int(np.clip(c[0], 0, shape[0] - 1)), int(np.clip(c[1], 0, shape[1] - 1))


def _astar_cells(start: tuple[int, int], goal: tuple[int, int], blocked: np.ndarray) -> list[tuple[int, int]] | None:
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    pq = [(0.0, start)]
    came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost = {start: 0.0}
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return list(reversed(path))
        for dx, dy in moves:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt[0] < 0 or nxt[1] < 0 or nxt[0] >= blocked.shape[0] or nxt[1] >= blocked.shape[1]:
                continue
            if blocked[nxt]:
                continue
            step = math.sqrt(2.0) if dx and dy else 1.0
            new_cost = cost[cur] + step
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                priority = new_cost + math.dist(nxt, goal)
                came[nxt] = cur
                heapq.heappush(pq, (priority, nxt))
    return None


def _simplify_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(cells) <= 2:
        return cells
    keep = [cells[0]]
    prev_dir = (cells[1][0] - cells[0][0], cells[1][1] - cells[0][1])
    for a, b in zip(cells[1:-1], cells[2:]):
        cur_dir = (b[0] - a[0], b[1] - a[1])
        if cur_dir != prev_dir:
            keep.append(a)
        prev_dir = cur_dir
    keep.append(cells[-1])
    return keep


def _line_clear(a: np.ndarray, b: np.ndarray, obstacle_xy: np.ndarray, clearance_m: float) -> bool:
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        return True
    t = np.clip(((obstacle_xy - a) @ ab) / denom, 0.0, 1.0)
    closest = a + t[:, None] * ab
    dist = np.linalg.norm(obstacle_xy - closest, axis=1)
    return bool(np.min(dist) > clearance_m)
