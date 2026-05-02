from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Tuple

Position = Tuple[int, int, int]
WindField = Dict[Position, Tuple[int, int, int]]


@dataclass(frozen=True)
class DroneState:
    """Estado imutável do drone para uso com graph_search=True."""

    x: int
    y: int
    z: int
    battery: int
    time_step: int

    @property
    def pos(self) -> Position:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class NoFlyZone:
    cells: FrozenSet[Position]
    start_time: int
    end_time: int

    def is_active(self, time_step: int) -> bool:
        return self.start_time <= time_step < self.end_time

    def blocks(self, position: Position, time_step: int) -> bool:
        return self.is_active(time_step) and position in self.cells


@dataclass(frozen=True)
class EnvironmentConfig:
    width: int
    height: int
    depth: int
    start: Position
    goal: Position
    max_battery: int = 18
    move_time: int = 1
    move_energy: int = 1
    recharge_time: int = 2
    recharge_amount: int = 6
    weight_time: float = 1.0
    weight_energy: float = 1.0
    obstacles: FrozenSet[Position] = field(default_factory=frozenset)
    recharge_stations: FrozenSet[Position] = field(default_factory=frozenset)
    wind: WindField = field(default_factory=dict)
    no_fly_zones: Tuple[NoFlyZone, ...] = field(default_factory=tuple)
    max_time: int = 60

    def in_bounds(self, position: Position) -> bool:
        x, y, z = position
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth

    def is_obstacle(self, position: Position) -> bool:
        return position in self.obstacles

    def is_recharge_station(self, position: Position) -> bool:
        return position in self.recharge_stations

    def is_blocked(self, position: Position, time_step: int) -> bool:
        if not self.in_bounds(position):
            return True
        if self.is_obstacle(position):
            return True
        return any(zone.blocks(position, time_step) for zone in self.no_fly_zones)

    def wind_vector(self, position: Position) -> Tuple[int, int, int]:
        return self.wind.get(position, (0, 0, 0))

    def iter_positions(self) -> Iterable[Position]:
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    yield (x, y, z)
