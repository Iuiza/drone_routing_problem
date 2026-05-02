from __future__ import annotations

from pathlib import Path
from typing import Iterable

from drone_search.algorithms import run_algorithm
from drone_search.generator import generate_n_instances, generate_random_instance
from drone_search.problem import DroneDeliveryProblem


def _extract_route_details(
    problem: DroneDeliveryProblem,
    actions: list[str],
) -> list[dict]:
    state = problem.initial_state
    route = [
        {
            "step": 0,
            "pos": state.pos,
            "time": state.time_step,
            "action": "START",
        }
    ]

    for idx, action in enumerate(actions, start=1):
        state = problem.result(state, action)
        route.append(
            {
                "step": idx,
                "pos": state.pos,
                "time": state.time_step,
                "action": action,
            }
        )

    return route


def _collect_no_fly_cells(env) -> set[tuple[int, int, int]]:
    cells = set()
    for zone in env.no_fly_zones:
        cells.update(zone.cells)
    return cells


def _zones_for_cell(env, pos: tuple[int, int, int]) -> list:
    zones = []
    for zone in env.no_fly_zones:
        if pos in zone.cells:
            zones.append(zone)
    return zones


def _group_route_by_layer(
    route_details: Iterable[dict] | None,
    depth: int,
) -> dict[int, list[dict]]:
    grouped = {z: [] for z in range(depth)}

    if route_details is None:
        return grouped

    for item in route_details:
        x, y, z = item["pos"]
        grouped.setdefault(z, []).append(item)

    return grouped


def _draw_single_instance(
    env,
    output_path: str,
    route_details: list[dict] | None = None,
    title: str | None = None,
    show_step_numbers: bool = False,
    show_time_labels: bool = True,
    show_no_fly_time_windows: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch, Rectangle
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib não está instalado. Rode: pip install matplotlib"
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    no_fly_cells = _collect_no_fly_cells(env)
    route_by_z = _group_route_by_layer(route_details, env.depth)

    fig, axes = plt.subplots(1, env.depth, figsize=(7 * env.depth, 7))
    if env.depth == 1:
        axes = [axes]

    for z in range(env.depth):
        ax = axes[z]
        ax.set_title(f"Camada z={z}", fontsize=14, pad=12)
        ax.set_xlim(-0.5, env.width - 0.5)
        ax.set_ylim(-0.5, env.height - 0.5)
        ax.set_xticks(range(env.width))
        ax.set_yticks(range(env.height))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.6)
        ax.invert_yaxis()

        for x in range(env.width):
            for y in range(env.height):
                pos = (x, y, z)

                if pos in env.obstacles:
                    rect = Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="dimgray",
                        edgecolor="black",
                        linewidth=1.0,
                        alpha=0.95,
                    )
                    ax.add_patch(rect)

                elif pos in no_fly_cells:
                    rect = Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="#ffb3b3",
                        edgecolor="red",
                        linewidth=1.2,
                        alpha=0.55,
                    )
                    ax.add_patch(rect)

                    if show_no_fly_time_windows:
                        zones_here = _zones_for_cell(env, pos)
                        label = "NF"
                        if zones_here:
                            zone = zones_here[0]
                            start_time = getattr(zone, "start_time", "?")
                            end_time = getattr(zone, "end_time", "?")
                            label = f"NF\n[{start_time},{end_time})"

                        ax.text(
                            x,
                            y,
                            label,
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="darkred",
                            fontweight="bold",
                        )

                elif pos in env.recharge_stations:
                    rect = Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="#bde0fe",
                        edgecolor="#1d4ed8",
                        linewidth=1.0,
                        alpha=0.85,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x,
                        y,
                        "R",
                        ha="center",
                        va="center",
                        fontsize=11,
                        fontweight="bold",
                        color="#1d4ed8",
                    )

                if pos in env.wind:
                    wx, wy, _wz = env.wind[pos]
                    if wx != 0 or wy != 0:
                        ax.arrow(
                            x,
                            y,
                            wx * 0.22,
                            wy * 0.22,
                            head_width=0.10,
                            head_length=0.10,
                            linewidth=1.0,
                            color="#2a9d8f",
                            length_includes_head=True,
                            alpha=0.9,
                        )

        sx, sy, sz = env.start
        gx, gy, gz = env.goal

        if sz == z:
            ax.scatter(
                sx,
                sy,
                marker="o",
                s=220,
                color="#198754",
                edgecolors="black",
                zorder=6,
            )
            ax.text(
                sx,
                sy - 0.30,
                "Start",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="#198754",
            )

        if gz == z:
            ax.scatter(
                gx,
                gy,
                marker="*",
                s=320,
                color="#f4a261",
                edgecolors="black",
                zorder=6,
            )
            ax.text(
                gx,
                gy - 0.30,
                "Goal",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="#b45309",
            )

        layer_route = route_by_z.get(z, [])
        if layer_route:
            xs = [item["pos"][0] for item in layer_route]
            ys = [item["pos"][1] for item in layer_route]

            ax.plot(
                xs,
                ys,
                linewidth=2.8,
                marker="o",
                markersize=5,
                color="#7b2cbf",
                zorder=5,
            )

            for item in layer_route:
                rx, ry, rz = item["pos"]
                step = item["step"]
                time_val = item["time"]

                active_violation = env.is_blocked((rx, ry, rz), time_val)
                in_no_fly_space = (rx, ry, rz) in no_fly_cells

                if active_violation:
                    ax.scatter(
                        rx,
                        ry,
                        s=220,
                        facecolors="none",
                        edgecolors="crimson",
                        linewidths=2.0,
                        zorder=7,
                    )

                elif in_no_fly_space:
                    ax.scatter(
                        rx,
                        ry,
                        s=180,
                        facecolors="none",
                        edgecolors="orange",
                        linewidths=1.8,
                        zorder=7,
                    )

                labels = []
                if show_step_numbers:
                    labels.append(f"p{step}")
                if show_time_labels:
                    labels.append(f"t={time_val}")

                if labels:
                    ax.text(
                        rx + 0.10,
                        ry + 0.12,
                        "\n".join(labels),
                        fontsize=8,
                        color="#4c1d95",
                        fontweight="bold",
                        zorder=8,
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.75,
                        ),
                    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Start",
            markerfacecolor="#198754",
            markeredgecolor="black",
            markersize=11,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Goal",
            markerfacecolor="#f4a261",
            markeredgecolor="black",
            markersize=15,
        ),
        Patch(facecolor="dimgray", edgecolor="black", label="Obstáculo"),
        Patch(facecolor="#ffb3b3", edgecolor="red", label="No-fly zone"),
        Patch(facecolor="#bde0fe", edgecolor="#1d4ed8", label="Recarga"),
        Line2D([0], [0], color="#2a9d8f", linewidth=2, label="Vento"),
        Line2D(
            [0],
            [0],
            color="#7b2cbf",
            linewidth=2.8,
            marker="o",
            markersize=5,
            label="Rota",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="orange",
            markersize=10,
            label="Passou no espaço NF fora da janela ativa",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="crimson",
            markersize=10,
            label="Violação real de no-fly",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.suptitle(title or "Mapa da instância", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def visualize_instance(
    seed: int,
    output: str,
    algorithm: str | None = None,
    show_step_numbers: bool = False,
    show_time_labels: bool = True,
    show_no_fly_time_windows: bool = True,
) -> None:
    env = generate_random_instance(seed)
    route_details = None
    title = f"Instância seed={seed}"

    if algorithm is not None:
        problem = DroneDeliveryProblem(env)
        result = run_algorithm(problem, algorithm_name=algorithm)
        title += f" | algoritmo={algorithm} | sucesso={result.success}"

        if result.success:
            route_details = _extract_route_details(problem, result.actions)
            title += f" | custo={result.total_cost:.2f}"
        else:
            title += " | sem rota"

    _draw_single_instance(
        env=env,
        output_path=output,
        route_details=route_details,
        title=title,
        show_step_numbers=show_step_numbers,
        show_time_labels=show_time_labels,
        show_no_fly_time_windows=show_no_fly_time_windows,
    )


def visualize_instance_batch(
    instances: int,
    seed_start: int,
    output_dir: str,
    algorithm: str | None = None,
    show_step_numbers: bool = False,
    show_time_labels: bool = True,
    show_no_fly_time_windows: bool = True,
) -> None:
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    envs = generate_n_instances(instances, seed_start=seed_start)

    for idx, env in enumerate(envs, start=1):
        seed = seed_start + idx - 1
        route_details = None
        suffix = ""
        title = f"Instância {idx} | seed={seed}"

        if algorithm is not None:
            problem = DroneDeliveryProblem(env)
            result = run_algorithm(problem, algorithm_name=algorithm)
            suffix = f"_{algorithm}"
            title += f" | algoritmo={algorithm} | sucesso={result.success}"

            if result.success:
                route_details = _extract_route_details(problem, result.actions)
                title += f" | custo={result.total_cost:.2f}"
            else:
                title += " | sem rota"

        output_path = output_base / f"instance_{idx:03d}_seed_{seed}{suffix}.png"

        _draw_single_instance(
            env=env,
            output_path=str(output_path),
            route_details=route_details,
            title=title,
            show_step_numbers=show_step_numbers,
            show_time_labels=show_time_labels,
            show_no_fly_time_windows=show_no_fly_time_windows,
        )

        print(f"[{idx}/{instances}] {output_path}")
        