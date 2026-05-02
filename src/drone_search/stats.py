from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchStats:
    started: bool = False
    finished: bool = False
    expanded_nodes: int = 0
    chosen_nodes: int = 0
    iterations: int = 0
    max_fringe_size: int = 0
    goal_found: bool = False
    finish_reason: str = ""


class StatsViewer:
    """Viewer simples para coletar métricas de execução da SimpleAI."""

    def __init__(self) -> None:
        self.stats = SearchStats()

    def event(self, name: str, *params: Any) -> None:
        if name == "started":
            self.stats.started = True
        elif name == "new_iteration":
            self.stats.iterations += 1
            fringe = params[0] if params else []
            try:
                self.stats.max_fringe_size = max(self.stats.max_fringe_size, len(fringe))
            except TypeError:
                pass
        elif name == "chosen_node":
            self.stats.chosen_nodes += 1
            is_goal = params[1] if len(params) > 1 else False
            if is_goal:
                self.stats.goal_found = True
        elif name == "expanded":
            expanded_from = params[0] if params else []
            self.stats.expanded_nodes += len(expanded_from)
        elif name == "finished":
            self.stats.finished = True
            self.stats.finish_reason = params[2] if len(params) > 2 else ""
        elif name == "no_more_runs":
            self.stats.finished = True
