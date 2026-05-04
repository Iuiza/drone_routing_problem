from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


NUMERIC_COLUMNS = [
    "total_cost",
    "solution_depth",
    "execution_time_sec",
    "expanded_nodes",
    "chosen_nodes",
    "iterations",
    "max_fringe_size",
    "path_length",
    "max_battery",
    "recharge_stations",
    "obstacles",
    "no_fly_zones",
]


def analyze_csv(input_csv: str, output_dir: str = "analysis") -> Path:
    input_path = Path(input_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("O CSV está vazio. Execute os experimentos antes da análise.")

    _prepare_dataframe(df)

    summary = build_summary_table(df)
    summary.to_csv(out_dir / "summary_by_algorithm.csv", index=False)

    instance_summary = build_instance_summary(df)
    instance_summary.to_csv(out_dir / "summary_by_instance.csv", index=False)

    _write_text_report(df, summary, out_dir / "analysis_report.txt")

    _plot_success_rate(df, out_dir / "success_rate.png")
    _plot_average_cost(df, out_dir / "average_cost_success_only.png")
    _plot_average_time(df, out_dir / "average_time.png")
    _plot_average_expanded_nodes(df, out_dir / "average_expanded_nodes.png")
    _plot_box_time(df, out_dir / "boxplot_execution_time.png")
    _plot_box_cost(df, out_dir / "boxplot_total_cost_success_only.png")

    return out_dir



def _prepare_dataframe(df: pd.DataFrame) -> None:
    df["success"] = df["success"].astype(str).str.lower().map({"true": True, "false": False})
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")



def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("algorithm", dropna=False)
    summary = grouped.apply(_summarize_group).reset_index()
    return summary.sort_values(["success_rate", "avg_total_cost_success"], ascending=[False, True], na_position="last")



def _summarize_group(group: pd.DataFrame) -> pd.Series:
    success_only = group[group["success"] == True]
    return pd.Series(
        {
            "runs": int(len(group)),
            "successful_runs": int(group["success"].sum()),
            "failed_runs": int((~group["success"]).sum()),
            "success_rate": float(group["success"].mean() * 100.0),
            "avg_total_cost_success": _safe_mean(success_only.get("total_cost")),
            "avg_execution_time_sec": _safe_mean(group.get("execution_time_sec")),
            "avg_expanded_nodes": _safe_mean(group.get("expanded_nodes")),
            "avg_chosen_nodes": _safe_mean(group.get("chosen_nodes")),
            "avg_max_fringe_size": _safe_mean(group.get("max_fringe_size")),
            "avg_path_length_success": _safe_mean(success_only.get("path_length")),
            "median_execution_time_sec": _safe_median(group.get("execution_time_sec")),
            "median_total_cost_success": _safe_median(success_only.get("total_cost")),
            "best_total_cost_success": _safe_min(success_only.get("total_cost")),
            "worst_total_cost_success": _safe_max(success_only.get("total_cost")),
        }
    )



def build_instance_summary(df: pd.DataFrame) -> pd.DataFrame:
    by_instance = (
        df.groupby(["instance_id", "algorithm"], dropna=False)
        .agg(
            success=("success", "max"),
            total_cost=("total_cost", "mean"),
            execution_time_sec=("execution_time_sec", "mean"),
            expanded_nodes=("expanded_nodes", "mean"),
            obstacles=("obstacles", "max"),
            no_fly_zones=("no_fly_zones", "max"),
            recharge_stations=("recharge_stations", "max"),
        )
        .reset_index()
    )
    return by_instance



def _safe_mean(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    series = series.dropna()
    return None if series.empty else float(series.mean())



def _safe_median(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    series = series.dropna()
    return None if series.empty else float(series.median())



def _safe_min(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    series = series.dropna()
    return None if series.empty else float(series.min())



def _safe_max(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    series = series.dropna()
    return None if series.empty else float(series.max())



def _plot_success_rate(df: pd.DataFrame, output: Path) -> None:
    data = df.groupby("algorithm")["success"].mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(kind="bar", ax=ax)
    ax.set_title("Taxa de sucesso por algoritmo")
    ax.set_ylabel("Sucesso (%)")
    ax.set_xlabel("Algoritmo")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)



def _plot_average_cost(df: pd.DataFrame, output: Path) -> None:
    success_df = df[df["success"] == True]
    data = success_df.groupby("algorithm")["total_cost"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(kind="bar", ax=ax)
    ax.set_title("Custo médio da solução por algoritmo")
    ax.set_ylabel("Custo total médio")
    ax.set_xlabel("Algoritmo")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)



def _plot_average_time(df: pd.DataFrame, output: Path) -> None:
    data = df.groupby("algorithm")["execution_time_sec"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(kind="bar", ax=ax)
    ax.set_title("Tempo médio de execução por algoritmo")
    ax.set_ylabel("Tempo médio (s)")
    ax.set_xlabel("Algoritmo")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)



def _plot_average_expanded_nodes(df: pd.DataFrame, output: Path) -> None:
    data = df.groupby("algorithm")["expanded_nodes"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(kind="bar", ax=ax)
    ax.set_title("Média de nós expandidos por algoritmo")
    ax.set_ylabel("Nós expandidos")
    ax.set_xlabel("Algoritmo")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)



def _plot_box_time(df: pd.DataFrame, output: Path) -> None:
    algorithms = sorted(df["algorithm"].dropna().unique())
    data = [df.loc[df["algorithm"] == alg, "execution_time_sec"].dropna().values for alg in algorithms]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=algorithms, vert=True)
    ax.set_title("Distribuição do tempo de execução")
    ax.set_ylabel("Tempo (s)")
    ax.set_xlabel("Algoritmo")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)



def _plot_box_cost(df: pd.DataFrame, output: Path) -> None:
    success_df = df[df["success"] == True]
    algorithms = sorted(success_df["algorithm"].dropna().unique())
    data = [success_df.loc[success_df["algorithm"] == alg, "total_cost"].dropna().values for alg in algorithms]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=algorithms, vert=True)
    ax.set_title("Distribuição do custo das soluções")
    ax.set_ylabel("Custo total")
    ax.set_xlabel("Algoritmo")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _write_text_report(df: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    best_success = summary.sort_values(["success_rate", "avg_total_cost_success"], ascending=[False, True]).head(1)
    fastest = summary.sort_values("avg_execution_time_sec", ascending=True).head(1)
    lowest_cost = summary.sort_values("avg_total_cost_success", ascending=True, na_position="last").head(1)
    least_expanded = summary.sort_values("avg_expanded_nodes", ascending=True).head(1)

    lines: list[str] = []
    lines.append("ANÁLISE AUTOMÁTICA DOS EXPERIMENTOS\n")
    lines.append(f"Total de execuções analisadas: {len(df)}")
    lines.append(f"Quantidade de instâncias: {df['instance_id'].nunique()}")
    lines.append(f"Algoritmos comparados: {', '.join(sorted(df['algorithm'].dropna().unique()))}\n")

    if not best_success.empty:
        row = best_success.iloc[0]
        lines.append(
            f"Maior taxa de sucesso: {row['algorithm']} ({row['success_rate']:.2f}%)."
        )
    if not fastest.empty:
        row = fastest.iloc[0]
        lines.append(
            f"Menor tempo médio de execução: {row['algorithm']} ({row['avg_execution_time_sec']:.6f} s)."
        )
    if not lowest_cost.empty and pd.notna(lowest_cost.iloc[0]["avg_total_cost_success"]):
        row = lowest_cost.iloc[0]
        lines.append(
            f"Menor custo médio entre soluções válidas: {row['algorithm']} ({row['avg_total_cost_success']:.3f})."
        )
    if not least_expanded.empty:
        row = least_expanded.iloc[0]
        lines.append(
            f"Menor média de nós expandidos: {row['algorithm']} ({row['avg_expanded_nodes']:.2f})."
        )

    lines.append("\nResumo por algoritmo:\n")
    for _, row in summary.iterrows():
        lines.append(
            "- "
            f"{row['algorithm']}: sucesso={row['success_rate']:.2f}% | "
            f"custo médio={_fmt(row['avg_total_cost_success'])} | "
            f"tempo médio={_fmt(row['avg_execution_time_sec'])} s | "
            f"nós expandidos={_fmt(row['avg_expanded_nodes'])}"
        )

    output.write_text("\n".join(lines), encoding="utf-8")



def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.3f}"
