from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from .algorithms import run_many
from .generator import generate_n_instances
from .problem import DroneDeliveryProblem


def run_experiments(
    num_instances: int,
    algorithms: Iterable[str],
    output_csv: str,
    seed_start: int = 1,
) -> List[dict]:
    rows: List[dict] = []
    instances = generate_n_instances(num_instances, seed_start=seed_start)

    for i, env in enumerate(instances, start=1):
        problem = DroneDeliveryProblem(env)
        results = run_many(problem, algorithms)

        for result in results:
            row = result.to_dict()
            row.update(
                {
                    "instance_id": i,
                    "grid": f"{env.width}x{env.height}x{env.depth}",
                    "start": env.start,
                    "goal": env.goal,
                    "max_battery": env.max_battery,
                    "recharge_stations": len(env.recharge_stations),
                    "obstacles": len(env.obstacles),
                    "no_fly_zones": len(env.no_fly_zones),
                }
            )
            rows.append(row)

    _write_csv(rows, output_csv)
    return rows


def _write_csv(rows: List[dict], output_csv: str) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
