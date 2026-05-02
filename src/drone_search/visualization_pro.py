from __future__ import annotations

from pathlib import Path
from typing import Iterable


def extract_route_trace(problem, actions: list[str]) -> list[dict]:
    state = problem.initial_state
    trace = [
        {
            "step": 0,
            "action": "START",
            "pos": state.pos,
            "time": state.time_step,
            "battery": state.battery,
        }
    ]

    for idx, action in enumerate(actions, start=1):
        state = problem.result(state, action)
        trace.append(
            {
                "step": idx,
                "action": action,
                "pos": state.pos,
                "time": state.time_step,
                "battery": state.battery,
            }
        )

    return trace


def _collect_no_fly_cells(env) -> set[tuple[int, int, int]]:
    cells = set()
    for zone in env.no_fly_zones:
        cells.update(zone.cells)
    return cells


def _zones_for_cell(env, pos: tuple[int, int, int]) -> list:
    return [zone for zone in env.no_fly_zones if pos in zone.cells]


def _group_trace_by_layer(trace: Iterable[dict], depth: int) -> dict[int, list[dict]]:
    grouped = {z: [] for z in range(depth)}
    for item in trace:
        _, _, z = item["pos"]
        grouped.setdefault(z, []).append(item)
    return grouped


def _label_offset(index: int) -> tuple[float, float]:
    offsets = [
        (0.08, 0.10),
        (0.08, -0.22),
        (-0.30, 0.10),
        (-0.30, -0.22),
        (0.18, 0.24),
    ]
    return offsets[index % len(offsets)]


def draw_professional_visualization(
    env,
    trace: list[dict],
    output_path: str,
    title: str = "Visualização da rota",
    subtitle: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch, Rectangle
    except ImportError as exc:
        raise RuntimeError("Instale matplotlib: pip install matplotlib") from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    no_fly_cells = _collect_no_fly_cells(env)
    trace_by_layer = _group_trace_by_layer(trace, env.depth)

    fig = plt.figure(figsize=(8 * env.depth + 5, 8))
    gs = GridSpec(1, env.depth + 1, width_ratios=[1] * env.depth + [1.15], figure=fig)

    axes = [fig.add_subplot(gs[0, i]) for i in range(env.depth)]
    table_ax = fig.add_subplot(gs[0, env.depth])
    table_ax.axis("off")

    for z, ax in enumerate(axes):
        ax.set_title(f"Camada z={z}", fontsize=15, pad=14)
        ax.set_xlim(-0.5, env.width - 0.5)
        ax.set_ylim(-0.5, env.height - 0.5)
        ax.set_xticks(range(env.width))
        ax.set_yticks(range(env.height))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.30, linestyle="--", linewidth=0.7)
        ax.invert_yaxis()

        for x in range(env.width):
            for y in range(env.height):
                pos = (x, y, z)

                if pos in env.obstacles:
                    ax.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5), 1, 1,
                            facecolor="#666666",
                            edgecolor="black",
                            linewidth=1.1,
                            alpha=0.95,
                        )
                    )

                elif pos in no_fly_cells:
                    ax.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5), 1, 1,
                            facecolor="#f8c8c8",
                            edgecolor="red",
                            linewidth=1.2,
                            alpha=0.65,
                        )
                    )
                    zones = _zones_for_cell(env, pos)
                    if zones:
                        zone = zones[0]
                        start_t = getattr(zone, "start_time", "?")
                        end_t = getattr(zone, "end_time", "?")
                        ax.text(
                            x, y, f"NF\n[{start_t},{end_t})",
                            ha="center", va="center",
                            fontsize=7, color="#8b0000", fontweight="bold",
                        )

                elif pos in env.recharge_stations:
                    ax.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5), 1, 1,
                            facecolor="#cfe8ff",
                            edgecolor="#2563eb",
                            linewidth=1.1,
                            alpha=0.90,
                        )
                    )
                    ax.text(
                        x, y, "R",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold", color="#1d4ed8",
                    )

                if pos in env.wind:
                    wx, wy, _ = env.wind[pos]
                    if wx != 0 or wy != 0:
                        ax.arrow(
                            x, y,
                            wx * 0.20, wy * 0.20,
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
            ax.scatter(sx, sy, marker="o", s=240, color="#198754", edgecolors="black", zorder=7)
            ax.text(sx, sy - 0.28, "Start", ha="center", va="top", fontsize=11, fontweight="bold", color="#198754")

        if gz == z:
            ax.scatter(gx, gy, marker="*", s=340, color="#f4a261", edgecolors="black", zorder=7)
            ax.text(gx, gy - 0.28, "Goal", ha="center", va="top", fontsize=11, fontweight="bold", color="#b45309")

        layer_trace = trace_by_layer.get(z, [])
        if layer_trace:
            xs = [item["pos"][0] for item in layer_trace]
            ys = [item["pos"][1] for item in layer_trace]

            ax.plot(xs, ys, linewidth=3.0, marker="o", markersize=5, color="#7b2cbf", zorder=5)

            occupancy_counter: dict[tuple[int, int], int] = {}
            for item in layer_trace:
                rx, ry, rz = item["pos"]
                key = (rx, ry)
                idx = occupancy_counter.get(key, 0)
                occupancy_counter[key] = idx + 1

                dx, dy = _label_offset(idx)

                blocked_now = env.is_blocked((rx, ry, rz), item["time"])
                inside_nf_space = (rx, ry, rz) in no_fly_cells

                if blocked_now:
                    ax.scatter(
                        rx, ry, s=230,
                        facecolors="none", edgecolors="crimson",
                        linewidths=2.2, zorder=8,
                    )
                elif inside_nf_space:
                    ax.scatter(
                        rx, ry, s=190,
                        facecolors="none", edgecolors="orange",
                        linewidths=1.8, zorder=8,
                    )

                ax.text(
                    rx + dx,
                    ry + dy,
                    f"p{item['step']}\nt={item['time']}\nb={item['battery']}",
                    fontsize=8,
                    color="#4c1d95",
                    fontweight="bold",
                    zorder=9,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.82,
                    ),
                )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Start", markerfacecolor="#198754", markeredgecolor="black", markersize=11),
        Line2D([0], [0], marker="*", color="w", label="Goal", markerfacecolor="#f4a261", markeredgecolor="black", markersize=15),
        Patch(facecolor="#666666", edgecolor="black", label="Obstáculo"),
        Patch(facecolor="#f8c8c8", edgecolor="red", label="No-fly zone"),
        Patch(facecolor="#cfe8ff", edgecolor="#2563eb", label="Recarga"),
        Line2D([0], [0], color="#2a9d8f", linewidth=2, label="Vento"),
        Line2D([0], [0], color="#7b2cbf", linewidth=3, marker="o", markersize=5, label="Rota"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="orange", markersize=10, label="Passagem em NF fora da janela"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="crimson", markersize=10, label="Violação real"),
    ]

    fig.legend(handles=legend_elements, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))

    table_ax.set_title("Traço da execução", fontsize=14, fontweight="bold", pad=10)

    lines = []
    for item in trace:
        x, y, z = item["pos"]
        lines.append(
            f"p{item['step']:>2} | {item['action']:<10} | "
            f"({x},{y},{z}) | t={item['time']:<2} | b={item['battery']}"
        )

    max_lines = 24
    shown = lines[:max_lines]
    if len(lines) > max_lines:
        shown.append("...")

    table_ax.text(
        0.0, 1.0,
        "\n".join(shown),
        va="top", ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#dddddd"),
    )

    final_title = title
    if subtitle:
        final_title += f"\n{subtitle}"

    fig.suptitle(final_title, fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)