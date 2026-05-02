from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.drone_search.algorithms import ALGORITHMS, run_algorithm
from src.drone_search.analyze_results import analyze_csv
from src.drone_search.experiments import run_experiments
from src.drone_search.generator import generate_random_instance
from src.drone_search.problem import DroneDeliveryProblem
from src.drone_search.visualization import visualize_instance, visualize_instance_batch
from src.drone_search.visualization_pro import draw_professional_visualization, extract_route_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Projeto de drones com simpleai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Executa uma instância demonstrativa")
    demo.add_argument("--algorithm", choices=sorted(ALGORITHMS.keys()), default="astar")
    demo.add_argument("--seed", type=int, default=1)

    exp = subparsers.add_parser("experiments", help="Executa várias instâncias e salva CSV")
    exp.add_argument("--instances", type=int, default=50)
    exp.add_argument(
        "--algorithms",
        nargs="+",
        default=["breadth_first", "depth_first", "uniform_cost", "greedy", "astar"],
        choices=sorted(ALGORITHMS.keys()),
    )
    exp.add_argument("--output", default="results/results.csv")
    exp.add_argument("--seed-start", type=int, default=1)

    analyze = subparsers.add_parser(
        "analyze",
        help="Analisa o CSV dos experimentos e gera gráficos",
    )
    analyze.add_argument("--input", required=True, help="CSV gerado pelos experimentos")
    analyze.add_argument(
        "--output-dir",
        default="analysis",
        help="Pasta de saída dos gráficos e resumos",
    )

    visualize = subparsers.add_parser(
        "visualize",
        help="Gera a visualização de uma instância",
    )
    visualize.add_argument("--seed", type=int, default=1)
    visualize.add_argument("--algorithm", choices=sorted(ALGORITHMS.keys()), default=None)
    visualize.add_argument("--output", required=True, help="Arquivo PNG de saída")
    visualize.add_argument(
        "--show-step-numbers",
        action="store_true",
        help="Mostra a numeração dos passos da rota",
    )
    visualize.add_argument(
        "--hide-time-labels",
        action="store_true",
        help="Oculta os tempos dos passos da rota",
    )
    visualize.add_argument(
        "--hide-no-fly-windows",
        action="store_true",
        help="Oculta as janelas temporais das no-fly zones",
    )

    visualize_batch = subparsers.add_parser(
        "visualize-batch",
        help="Gera visualizações para várias instâncias",
    )
    visualize_batch.add_argument("--instances", type=int, default=50)
    visualize_batch.add_argument("--seed-start", type=int, default=1)
    visualize_batch.add_argument("--algorithm", choices=sorted(ALGORITHMS.keys()), default=None)
    visualize_batch.add_argument("--output-dir", default="visualizations")
    visualize_batch.add_argument(
        "--show-step-numbers",
        action="store_true",
        help="Mostra a numeração dos passos da rota",
    )
    visualize_batch.add_argument(
        "--hide-time-labels",
        action="store_true",
        help="Oculta os tempos dos passos da rota",
    )
    visualize_batch.add_argument(
        "--hide-no-fly-windows",
        action="store_true",
        help="Oculta as janelas temporais das no-fly zones",
    )

    inspect_cmd = subparsers.add_parser(
    "inspect",
    help="Gera visualização detalhada com tabela lateral e bateria",
)
    inspect_cmd.add_argument("--seed", type=int, default=1)
    inspect_cmd.add_argument("--algorithm", choices=sorted(ALGORITHMS.keys()), default="astar")
    inspect_cmd.add_argument("--output", required=True)

    return parser


def run_demo(algorithm: str, seed: int) -> None:
    env = generate_random_instance(seed)
    problem = DroneDeliveryProblem(env)
    result = run_algorithm(problem, algorithm_name=algorithm)

    print("=== INSTÂNCIA DEMO ===")
    print(f"Grid: {env.width}x{env.height}x{env.depth}")
    print(f"Start: {env.start} | Goal: {env.goal}")
    print(f"Obstáculos: {len(env.obstacles)}")
    print(f"Recarga: {len(env.recharge_stations)}")
    print(f"No-fly: {len(env.no_fly_zones)}")
    print(f"Algoritmo: {result.algorithm}")
    print(f"Sucesso: {result.success}")
    print(f"Custo total: {result.total_cost}")
    print(f"Profundidade: {result.solution_depth}")
    print(f"Tempo (s): {result.execution_time_sec:.6f}")
    print(f"Nós expandidos: {result.expanded_nodes}")
    print(f"Nós escolhidos: {result.chosen_nodes}")
    print(f"Ações: {result.actions}")
    print(f"Estado final: {result.final_state}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "demo":
        run_demo(algorithm=args.algorithm, seed=args.seed)

    elif args.command == "experiments":
        rows = run_experiments(
            num_instances=args.instances,
            algorithms=args.algorithms,
            output_csv=args.output,
            seed_start=args.seed_start,
        )
        print(f"Experimentos concluídos: {len(rows)} linhas salvas em {args.output}")

    elif args.command == "analyze":
        out_dir = analyze_csv(input_csv=args.input, output_dir=args.output_dir)
        print(f"Análise concluída. Arquivos gerados em: {out_dir}")

    elif args.command == "visualize":
        if args.algorithm:
            env = generate_random_instance(args.seed)
            problem = DroneDeliveryProblem(env)

            result = run_algorithm(problem, algorithm_name=args.algorithm)

            print("\n=== RESULTADO DA BUSCA ===")
            print(f"Algoritmo: {result.algorithm}")
            print(f"Sucesso: {result.success}")
            print(f"Custo total: {result.total_cost}")
            print(f"Tempo execução: {result.execution_time_sec:.6f}s")
            print(f"Nós expandidos: {result.expanded_nodes}")
            print(f"Nós escolhidos: {result.chosen_nodes}")
            print(f"Profundidade solução: {result.solution_depth}")
            print(f"Ações:")
            for i, action in enumerate(result.actions, 1):
                print(f" {i:02d}. {action}")

        visualize_instance(
            seed=args.seed,
            output=args.output,
            algorithm=args.algorithm,
            show_step_numbers=args.show_step_numbers,
            show_time_labels=not args.hide_time_labels,
            show_no_fly_time_windows=not args.hide_no_fly_windows,
        )
        print(f"Visualização salva em: {args.output}")

    elif args.command == "visualize-batch":
        visualize_instance_batch(
            instances=args.instances,
            seed_start=args.seed_start,
            output_dir=args.output_dir,
            algorithm=args.algorithm,
            show_step_numbers=args.show_step_numbers,
            show_time_labels=not args.hide_time_labels,
            show_no_fly_time_windows=not args.hide_no_fly_windows,
        )

    elif args.command == "inspect":
        env = generate_random_instance(args.seed)
        problem = DroneDeliveryProblem(env)
        result = run_algorithm(problem, algorithm_name=args.algorithm)

        print("\n=== RESULTADO DA BUSCA ===")
        print(f"Algoritmo: {result.algorithm}")
        print(f"Sucesso: {result.success}")
        print(f"Custo total: {result.total_cost}")
        print(f"Tempo execução: {result.execution_time_sec:.6f}s")
        print(f"Nós expandidos: {result.expanded_nodes}")
        print(f"Nós escolhidos: {result.chosen_nodes}")
        print(f"Profundidade solução: {result.solution_depth}")
        print("Ações:")
        for i, action in enumerate(result.actions, 1):
            print(f" {i:02d}. {action}")

        trace = extract_route_trace(problem, result.actions)

        draw_professional_visualization(
            env=env,
            trace=trace,
            output_path=args.output,
            title=f"Instância seed={args.seed} | algoritmo={args.algorithm}",
            subtitle=f"sucesso={result.success} | custo={result.total_cost:.2f}",
        )

        print(f"\nVisualização detalhada salva em: {args.output}")


if __name__ == "__main__":
    main()
