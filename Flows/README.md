# Maximum Flow Algorithms Project

## Overview

This project implements and compares two fundamental algorithms for solving the Maximum Flow problem in graph theory:

- **Ford-Fulkerson Algorithm** (1956) - Generic maximum flow algorithm using DFS-based augmenting paths
- **Edmonds-Karp Algorithm** (1972) - Optimized Ford-Fulkerson variant using BFS for shortest augmenting paths

## Project Goals

The primary objectives of this project are to:

1. **Implement** both algorithms with clean, well-documented code
2. **Compare** their theoretical and empirical performance characteristics
3. **Visualize** flow networks and algorithm results using NetworkX and Matplotlib
4. **Analyze** algorithmic behavior across different graph topologies
5. **Benchmark** performance on various graph types and sizes

## Key Features

- **Multiple Graph Generators**: Create random, linear, dense, bottleneck, bipartite, and grid graphs
- **Interactive Visualization**: See flow networks with capacity/flow labels and algorithm comparisons
- **Comprehensive Testing**: Unit tests ensuring algorithm correctness
- **Performance Benchmarking**: Detailed timing and iteration analysis
- **Educational Focus**: Clear documentation and theoretical explanations

## Project Structure

```
maxflow-project/
├── algorithms/           # Core algorithm implementations
│   ├── ford_fulkerson.py    # Ford-Fulkerson with DFS
│   └── edmonds_karp.py      # Edmonds-Karp with BFS
├── utils/               # Graph generation utilities
│   └── graph_generator.py   # Various graph type generators
├── tests/               # Unit tests
│   └── test_flows.py        # Algorithm correctness tests
├── benchmarks/          # Performance analysis
│   └── run_benchmarks.py    # Comprehensive benchmarking suite
├── main.py              # Interactive demonstration script
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── report.typ          # Detailed technical report
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Interactive Demo**:
   ```bash
   python main.py
   ```

3. **Run Tests**:
   ```bash
   python -m pytest tests/
   ```

4. **Generate Benchmarks**:
   ```bash
   python benchmarks/run_benchmarks.py
   ```

## Algorithm Comparison

| Algorithm | Time Complexity | Space Complexity | Path Strategy |
|-----------|-----------------|------------------|---------------|
| Ford-Fulkerson | O(E · f) | O(V) | DFS (any path) |
| Edmonds-Karp | O(V · E²) | O(V) | BFS (shortest path) |

*Where V = vertices, E = edges, f = maximum flow value*

## Applications

Maximum flow algorithms have numerous real-world applications:
- Network routing and bandwidth allocation
- Transportation and logistics optimization  
- Image segmentation and computer vision
- Bipartite matching problems
- Project scheduling and resource allocation

## Educational Value

This project serves as a comprehensive study of fundamental graph algorithms, demonstrating:
- Algorithm design paradigms (greedy, augmenting paths)
- The importance of path selection strategies
- Empirical vs theoretical complexity analysis
- Graph visualization and algorithm animation techniques

## Technical Report

For detailed theoretical analysis, implementation details, and experimental results, see the comprehensive technical report in `report.typ`.
