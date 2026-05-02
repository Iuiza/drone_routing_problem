from __future__ import annotations

import random
from collections import deque
from typing import List, Optional, Sequence, Tuple

from .models import EnvironmentConfig, NoFlyZone, Position

ORTHO_MOVES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def generate_random_instance(seed: int) -> EnvironmentConfig:
    rng = random.Random(seed)

    width = rng.randint(4, 6)
    height = rng.randint(4, 6)
    depth = rng.randint(2, 3)
    start = (0, 0, 0)
    goal = (width - 1, height - 1, depth - 1)

    all_positions = [(x, y, z) for x in range(width) for y in range(height) for z in range(depth)]
    free_candidates = [p for p in all_positions if p not in {start, goal}]

    obstacle_count = max(1, int(len(all_positions) * rng.uniform(0.08, 0.16)))
    obstacles = set(rng.sample(free_candidates, k=min(obstacle_count, len(free_candidates))))

    # Garante caminho básico removendo obstáculos se necessário.
    while not _has_spatial_path(width, height, depth, start, goal, obstacles):
        obstacles.pop()

    station_candidates = [p for p in free_candidates if p not in obstacles]
    recharge_count = min(2, len(station_candidates))
    recharge_stations = set(rng.sample(station_candidates, k=recharge_count)) if recharge_count > 0 else set()

    wind = {}
    for pos in all_positions:
        if pos in obstacles:
            continue
        if rng.random() < 0.30:
            wind[pos] = rng.choice(ORTHO_MOVES)

    no_fly_zones = []
    zone_candidates = [p for p in station_candidates if p not in recharge_stations]
    if len(zone_candidates) >= 2:
        zone_cells = frozenset(rng.sample(zone_candidates, k=rng.randint(1, min(3, len(zone_candidates)))))
        start_time = rng.randint(3, 8)
        end_time = start_time + rng.randint(3, 6)
        no_fly_zones.append(NoFlyZone(cells=zone_cells, start_time=start_time, end_time=end_time))

    return EnvironmentConfig(
        width=width,
        height=height,
        depth=depth,
        start=start,
        goal=goal,
        max_battery=rng.randint(8, 14),
        move_time=1,
        move_energy=1,
        recharge_time=rng.randint(2, 4),
        recharge_amount=rng.randint(4, 8),
        weight_time=1.0,
        weight_energy=1.0,
        obstacles=frozenset(obstacles),
        recharge_stations=frozenset(recharge_stations),
        wind=wind,
        no_fly_zones=tuple(no_fly_zones),
        max_time=50,
    )


def generate_n_instances(n: int, seed_start: int = 1) -> List[EnvironmentConfig]:
    return [generate_random_instance(seed_start + i) for i in range(n)]


def _has_spatial_path(
    width: int,
    height: int,
    depth: int,
    start: Position,
    goal: Position,
    obstacles: Sequence[Position],
) -> bool:
    blocked = set(obstacles)
    queue = deque([start])
    visited = {start}

    while queue:
        pos = queue.popleft()
        if pos == goal:
            return True
        for dx, dy, dz in ORTHO_MOVES:
            nxt = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height and 0 <= nxt[2] < depth):
                continue
            if nxt in blocked or nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)

    return False
