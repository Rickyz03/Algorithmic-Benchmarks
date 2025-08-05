# Advanced Algorithms: Shortest Path Comparison

This project implements and compares three fundamental shortest path algorithms: Dijkstra's algorithm, Bellman-Ford algorithm, and A* search. The goal is to analyze their theoretical properties and practical performance on various graph types.

## Algorithms Implemented

- **Dijkstra's Algorithm**: Optimal for graphs with non-negative edge weights
- **Bellman-Ford Algorithm**: Handles graphs with negative edge weights
- **A* Search**: Informed search using heuristics, ideal for grid-based pathfinding

## Project Structure

```
├── src/
│   ├── graph.py          # Graph representation and utilities
│   ├── dijkstra.py       # Dijkstra's algorithm implementation
│   ├── bellman_ford.py   # Bellman-Ford algorithm implementation
│   ├── astar.py          # A* search implementation
│   └── main.py           # Main benchmarking and analysis script
├── report.typ            # Detailed analysis report (Typst format)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Features

- Modular implementation of all three algorithms
- Automatic graph generation (sparse, dense, grids)
- Performance benchmarking and visualization
- Comprehensive theoretical analysis and comparison

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the comparison: `python src/main.py`
3. View results and visualizations
4. Compile the report: `typst compile report.typ`

## Dependencies

- Python 3.8+
- NetworkX for graph utilities
- Matplotlib for visualization
- NumPy for numerical computations
