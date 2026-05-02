# Projeto 1 — Drones com SimpleAI

Código-base em Python para modelar e resolver o problema de rota de drones em ambiente urbano dinâmico usando a biblioteca **simpleai**.

## O que está incluído

- modelagem do problema em grid 3D
- custo combinando **tempo + energia**
- obstáculos fixos
- zonas de restrição temporárias
- estações de recarga
- campo de vento simplificado
- algoritmos: **BFS, DFS, custo uniforme, gulosa e A\***
- gerador de instâncias
- executor de experimentos para **50 ou mais instâncias**
- exportação dos resultados em CSV

- análise automática do CSV com gráficos e resumos estatísticos

## Estrutura

```text
src/drone_search/
  __init__.py
  algorithms.py
  experiments.py
  generator.py
  models.py
  problem.py
  stats.py
main.py
requirements.txt
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Execução rápida

Rodar uma instância demonstrativa:

```bash
python main.py demo --algorithm astar
```

Gerar e executar 50 instâncias:

```bash
python main.py experiments --instances 50 --output results.csv
```

Rodar um conjunto específico de algoritmos:

```bash
python main.py experiments --instances 50 --algorithms depth_first breadth_first uniform_cost greedy astar
```

## Observações importantes

1. O estado foi mantido **imutável** para funcionar bem com `graph_search=True`.
2. A heurística foi construída para ser **admissível**: usa a distância Manhattan, a distância entre dois pontos somando as diferenças absolutas de suas coordenadas até o objetivo (`|x1 - x2| + |y1 - y2| + |z1 - z2|`), multiplicada pelo menor custo possível por movimento, ignorando obstáculos, vento adverso e restrições temporárias.
3. BFS e DFS tratam o problema como busca cega. Como não usam custo na priorização, podem retornar soluções piores que custo uniforme ou A\*.
4. Para fins didáticos, a dinâmica temporal foi discretizada em passos inteiros.

## Saída dos experimentos

O arquivo CSV inclui, para cada instância e algoritmo:

- sucesso ou fracasso
- custo total encontrado
- profundidade da solução
- tempo de execução
- nós expandidos
- nós escolhidos/visitados
- tamanho máximo da fronteira
- quantidade de ações na rota

## Análise automática dos resultados

Depois de gerar o CSV, você pode criar gráficos e resumos automáticos:

```bash
python main.py analyze --input results.csv --output-dir analysis
```

Arquivos gerados automaticamente:

- `summary_by_algorithm.csv`: resumo estatístico por algoritmo
- `summary_by_instance.csv`: visão agregada por instância
- `analysis_report.txt`: resumo textual para ajudar no relatório
- `success_rate.png`: taxa de sucesso por algoritmo
- `average_cost_success_only.png`: custo médio das soluções válidas
- `average_time.png`: tempo médio de execução
- `average_expanded_nodes.png`: média de nós expandidos
- `boxplot_execution_time.png`: distribuição do tempo
- `boxplot_total_cost_success_only.png`: distribuição dos custos válidos
- `scatter_cost_vs_time.png`: relação entre custo e tempo
- `scalability_expanded_nodes_vs_obstacles.png`: escalabilidade com obstáculos

Fluxo sugerido:

```bash
python main.py experiments --instances 50 --output results/results.csv
python main.py analyze --input results/results.csv --output-dir analysis
python main.py inspect --seed 1 --algorithm astar --output mapa.png
```
