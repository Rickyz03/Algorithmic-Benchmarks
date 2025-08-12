# TSP Algorithm Comparison Project

A comprehensive implementation and comparison of various algorithms for solving the Traveling Salesman Problem (TSP), featuring exact algorithms, approximation algorithms, and heuristic approaches.

## 🎯 Project Overview

This project implements and analyzes multiple approaches to solve the TSP:

- **Exact Algorithms**: Brute force, Held-Karp dynamic programming, Branch & Bound
- **Approximation Algorithm**: MST-based 2-approximation with theoretical guarantee
- **Heuristic Algorithms**: Nearest Neighbor, 2-opt local search, multi-start approaches

The goal is to understand the trade-offs between solution quality, computational time, and scalability across different algorithm types.

## 🏗️ Project Structure

```
travel-salesman-problem-project/
├── algorithms/
│   ├── mst_approximation.py      # MST-based 2-approximation algorithm
│   ├── exact_algorithms.py       # Brute force, Held-Karp, Branch & Bound
│   └── heuristic_algorithms.py   # Nearest Neighbor, 2-opt, multi-start heuristics
├── utils/
│   └── graph_generator.py        # Graph generation utilities for TSP instances
├── tests/
│   └── test_tsp.py              # Unit tests and algorithm verification
├── benchmarks/
│   └── run_benchmarks.py        # Comprehensive benchmarking suite
├── main.py                      # Main execution script with interactive demo
├── requirements.txt             # Python package dependencies
├── README.md                    # This file
└── report.typ                   # Detailed technical report (Typst format)
```

## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Interactive Demo

```bash
python main.py
```

This launches an interactive demo where you can:
- Compare algorithms on single instances
- Run scaling experiments
- Test custom city coordinates
- Visualize solutions

### Command Line Usage

```bash
# Compare algorithms on a single 10-city instance
python main.py --mode compare --size 10 --type euclidean

# Run scaling experiment
python main.py --mode scale --sizes "5,6,7,8,10,12" --instances 3

# Run without visualizations
python main.py --mode compare --size 8 --no-viz
```

### Running Comprehensive Benchmarks

```bash
cd benchmarks
python run_benchmarks.py
```

This generates detailed performance analysis, plots, and CSV/JSON result files.

### Running Tests

```bash
python tests/test_tsp.py
```

## 📊 Key Features

### Algorithm Implementations

1. **MST 2-Approximation**: Guaranteed solution within 2× optimal for metric TSP
2. **Held-Karp DP**: Optimal solution in O(n²2ⁿ) time
3. **Branch & Bound**: Exact algorithm with intelligent pruning
4. **Nearest Neighbor + 2-opt**: Fast heuristic with local improvement
5. **Multi-start approaches**: Multiple restarts for better solutions

### Visualization

- 2D tour visualization for Euclidean instances
- Performance comparison plots
- Scaling analysis charts
- Approximation ratio analysis

### Instance Generation

- Random Euclidean instances (cities in 2D plane)
- Random metric instances (satisfying triangle inequality)
- Custom coordinate input support

## 🧪 Experimental Results

The project includes comprehensive benchmarking showing:

- **Approximation Quality**: How close heuristics get to optimal solutions
- **Scalability**: How algorithms scale with problem size
- **Time vs Quality Trade-offs**: Pareto frontier analysis
- **Success Rates**: Algorithm reliability across different instances

## 🔧 Technical Details

### Algorithms Implemented

| Algorithm | Type | Time Complexity | Space | Max Practical Size |
|-----------|------|-----------------|-------|-------------------|
| Brute Force | Exact | O(n!) | O(1) | ~8 cities |
| Held-Karp | Exact | O(n²2ⁿ) | O(n2ⁿ) | ~15 cities |
| Branch & Bound | Exact | O(n!) worst case | O(n) | ~12 cities |
| MST 2-Approx | Approximation | O(n²) | O(n) | 1000+ cities |
| Nearest Neighbor | Heuristic | O(n²) | O(n) | 1000+ cities |
| 2-opt | Heuristic | O(n²) per iter | O(n) | 1000+ cities |

### Instance Types

- **Euclidean**: Cities with random 2D coordinates, distances calculated using Euclidean metric
- **Metric**: Random symmetric matrices satisfying triangle inequality

## 📈 Usage Examples

### Comparing Algorithms

```python
from main import run_algorithm_comparison

# Compare algorithms on 8-city Euclidean instance
results = run_algorithm_comparison(size=8, instance_type='euclidean', seed=42)

# Results contain tour, length, time, and status for each algorithm
for algorithm, result in results.items():
    if result['status'] == 'success':
        print(f"{algorithm}: {result['length']:.2f} (time: {result['time']:.3f}s)")
```

### Custom Benchmarking

```python
from benchmarks.run_benchmarks import TSPBenchmark

benchmark = TSPBenchmark()
results_df = benchmark.run_size_scaling_benchmark(
    sizes=[5, 6, 7, 8, 10],
    instance_type='euclidean',
    num_instances=5
)
```

## 🏆 Key Insights

The project reveals important trade-offs in TSP algorithm selection:

- **For small instances (≤8 cities)**: Exact algorithms are feasible
- **For medium instances (8-20 cities)**: Held-Karp DP provides optimal solutions
- **For large instances (>20 cities)**: Heuristics with 2-opt improvement work well
- **MST 2-approximation**: Provides theoretical guarantees but may be outperformed by good heuristics in practice

## 📝 Contributing

This is an educational project demonstrating TSP algorithms. The code is structured for clarity and learning, with comprehensive documentation and testing.

## 📄 License

This project is for educational purposes. Feel free to use and modify for learning and research.

---

For detailed algorithm descriptions, theoretical analysis, and experimental results, see the full technical report in `report.typ`.
