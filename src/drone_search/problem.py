from __future__ import annotations

from typing import List, Tuple

from simpleai.search import SearchProblem

from .models import DroneState, EnvironmentConfig

MOVE_DELTAS = {
    "UP_X": (1, 0, 0),
    "DOWN_X": (-1, 0, 0),
    "UP_Y": (0, 1, 0),
    "DOWN_Y": (0, -1, 0),
    "UP_Z": (0, 0, 1),
    "DOWN_Z": (0, 0, -1),
}


class DroneDeliveryProblem(SearchProblem):
    """Problema de busca para rota de drone em grid 3D dinâmico."""

    def __init__(self, env: EnvironmentConfig):
        self.env = env
        initial_state = DroneState(
            x=env.start[0],
            y=env.start[1],
            z=env.start[2],
            battery=env.max_battery,
            time_step=0,
        )
        super().__init__(initial_state=initial_state)

    def actions(self, state: DroneState) -> List[str]:
        if state.time_step >= self.env.max_time:
            return []

        possible: List[str] = []

        for action, delta in MOVE_DELTAS.items():
            if state.battery < self._energy_for_move(state.pos, delta):
                continue
            next_pos = self._add(state.pos, delta)
            next_time = state.time_step + self.env.move_time
            if self.env.is_blocked(next_pos, next_time):
                continue
            possible.append(action)

        if self.env.is_recharge_station(state.pos) and state.battery < self.env.max_battery:
            possible.append("RECHARGE")

        if state.time_step + 1 <= self.env.max_time:
            possible.append("WAIT")

        return possible

    def result(self, state: DroneState, action: str) -> DroneState:
        if action in MOVE_DELTAS:
            delta = MOVE_DELTAS[action]
            next_pos = self._add(state.pos, delta)
            energy = self._energy_for_move(state.pos, delta)
            return DroneState(
                x=next_pos[0],
                y=next_pos[1],
                z=next_pos[2],
                battery=max(0, state.battery - energy),
                time_step=state.time_step + self.env.move_time,
            )

        if action == "RECHARGE":
            return DroneState(
                x=state.x,
                y=state.y,
                z=state.z,
                battery=min(self.env.max_battery, state.battery + self.env.recharge_amount),
                time_step=state.time_step + self.env.recharge_time,
            )

        if action == "WAIT":
            return DroneState(
                x=state.x,
                y=state.y,
                z=state.z,
                battery=state.battery - self._energy_for_waiting(state.pos),
                time_step=state.time_step + 1,
            )

        raise ValueError(f"Ação desconhecida: {action}")

    def is_goal(self, state: DroneState) -> bool:
        return state.pos == self.env.goal

    def cost(self, state: DroneState, action: str, state2: DroneState) -> float:
        if action == "RECHARGE":
            time_cost = self.env.recharge_time
            energy_cost = 0
            return self.env.weight_time * time_cost + self.env.weight_energy * energy_cost

        if action == "WAIT":
            return self.env.weight_time * 1 + self.env.weight_energy * self._energy_for_waiting(state.pos)

        delta = MOVE_DELTAS[action]
        energy = self._energy_for_move(state.pos, delta)
        time_cost = self.env.move_time + self._time_penalty_from_wind(state.pos, delta)
        return self.env.weight_time * time_cost + self.env.weight_energy * energy

    def heuristic(self, state: DroneState) -> float:
        # Heurística admissível:
        # usa a distância Manhattan até o objetivo multiplicada pelo menor custo
        # possível por movimento, ignorando obstáculos, vento adverso e esperas.
        goal = self.env.goal
        distance = abs(state.x - goal[0]) + abs(state.y - goal[1]) + abs(state.z - goal[2])
        min_step_cost = self.env.weight_time * self.env.move_time + self.env.weight_energy * self.env.move_energy
        return distance * min_step_cost

    def _energy_for_move(self, position: Tuple[int, int, int], delta: Tuple[int, int, int]) -> int:
        base = self.env.move_energy
        wind = self.env.wind_vector(position)
        alignment = -(wind[0] * delta[0] + wind[1] * delta[1] + wind[2] * delta[2])
        return max(1, base + max(0, alignment))

    def _energy_for_waiting(self, position: Tuple[int, int, int]) -> int:
        wind = self.env.wind_vector(position)
        alignment = -(wind[0] * 0 + wind[1] * 0 + wind[2] * 0)
        return max(1, max(0, alignment))

    def _time_penalty_from_wind(self, position: Tuple[int, int, int], delta: Tuple[int, int, int]) -> int:
        wind = self.env.wind_vector(position)
        alignment = -(wind[0] * delta[0] + wind[1] * delta[1] + wind[2] * delta[2])
        return max(0, alignment)

    @staticmethod
    def _add(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
