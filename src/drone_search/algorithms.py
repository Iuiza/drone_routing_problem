from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional

from simpleai.search import astar, breadth_first, depth_first, greedy, uniform_cost

from .problem import DroneDeliveryProblem
from .stats import StatsViewer


@dataclass
class SearchRunResult:
    algorithm: str
    success: bool
    total_cost: Optional[float]
    solution_depth: Optional[int]
    execution_time_sec: float
    expanded_nodes: int
    chosen_nodes: int
    iterations: int
    max_fringe_size: int
    path_length: int
    actions: List[str]
    final_state: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


ALGORITHMS: Dict[str, Callable] = {
    "breadth_first": breadth_first,
    "depth_first": depth_first,
    "uniform_cost": uniform_cost,
    "greedy": greedy,
    "astar": astar,
}


def run_algorithm(problem: DroneDeliveryProblem, algorithm_name: str, graph_search: bool = True) -> SearchRunResult:
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Algoritmo inválido: {algorithm_name}")

    viewer = StatsViewer()
    algorithm = ALGORITHMS[algorithm_name]

    start = time.perf_counter()
    node = algorithm(problem, graph_search=graph_search, viewer=viewer)
    elapsed = time.perf_counter() - start

    if node is None:
        return SearchRunResult(
            algorithm=algorithm_name,
            success=False,
            total_cost=None,
            solution_depth=None,
            execution_time_sec=elapsed,
            expanded_nodes=viewer.stats.expanded_nodes,
            chosen_nodes=viewer.stats.chosen_nodes,
            iterations=viewer.stats.iterations,
            max_fringe_size=viewer.stats.max_fringe_size,
            path_length=0,
            actions=[],
            final_state=None,
        )

    path = node.path()
    actions = [action for action, _state in path[1:]]

    return SearchRunResult(
        algorithm=algorithm_name,
        success=True,
        total_cost=float(getattr(node, "cost", 0.0)),
        solution_depth=int(getattr(node, "depth", len(actions))),
        execution_time_sec=elapsed,
        expanded_nodes=viewer.stats.expanded_nodes,
        chosen_nodes=viewer.stats.chosen_nodes,
        iterations=viewer.stats.iterations,
        max_fringe_size=viewer.stats.max_fringe_size,
        path_length=len(actions),
        actions=actions,
        final_state=str(getattr(node, "state", None)),
    )


def run_many(problem: DroneDeliveryProblem, algorithms: Iterable[str]) -> List[SearchRunResult]:
    return [run_algorithm(problem, algorithm_name=name) for name in algorithms]
