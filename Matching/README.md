# Bipartite Graph Matching Algorithms

A comprehensive implementation and comparison study of two fundamental algorithms for bipartite graph matching: the **Hungarian Algorithm** and the **Hopcroft-Karp Algorithm**.

## Overview

This project implements and analyzes two classical algorithms for solving different variants of the bipartite matching problem:

- **Hungarian Algorithm**: Solves the maximum weight perfect matching problem in O(n³) time
- **Hopcroft-Karp Algorithm**: Solves the maximum cardinality matching problem in O(√V · E) time

## Algorithms Implemented

### Hungarian Algorithm (Kuhn-Munkres)
- **Purpose**: Find maximum weight matching in weighted bipartite graphs
- **Input**: Square cost/weight matrix 
- **Output**: Perfect matching with maximum total weight
- **Complexity**: O(n³)
- **Applications**: Job assignment, resource allocation, tracking problems

### Hopcroft-Karp Algorithm  
- **Purpose**: Find maximum cardinality matching in unweighted bipartite graphs
- **Input**: Bipartite graph as edge list
- **Output**: Maximum number of matched vertex pairs
- **Complexity**: O(√V · E)  
- **Applications**: Marriage problem, task assignment, network flows

## Project Structure

```
matching-project/
│
├── src/
│   ├── hungarian.py           # Hungarian Algorithm implementation
│   ├── hopcroft_karp.py      # Hopcroft-Karp Algorithm implementation
│   └── main.py               # Main comparison and visualization script
│
├── utils/
│   └── graph_generator.py    # Bipartite graph generation utilities
│
├── tests/
│   └── test_matching.py      # Unit tests for both algorithms
│
├── benchmarks/
│   └── run_benchmarks.py     # Performance benchmarking suite
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── report.typ               # Detailed technical report (Typst)
```

## Features

- **Clean Implementations**: Both algorithms implemented from scratch without external graph libraries
- **Comprehensive Testing**: Unit tests covering edge cases and correctness verification
- **Performance Analysis**: Detailed benchmarking across different graph types and sizes
- **Visualization**: Matplotlib-based visualization of matching results
- **Graph Generation**: Utilities for creating various types of test graphs (sparse, dense, regular, etc.)

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd matching-project

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Hungarian Algorithm example
from src.hungarian import solve_maximum_weight_matching

cost_matrix = [
    [4, 1, 3],
    [2, 0, 5],
    [3, 2, 2]
]

matching, total_weight = solve_maximum_weight_matching(cost_matrix)
print(f"Maximum matching: {matching}")
print(f"Total weight: {total_weight}")

# Hopcroft-Karp Algorithm example  
from src.hopcroft_karp import create_bipartite_graph_from_edges

edges = [(0, 0), (0, 1), (1, 1), (2, 2)]
algorithm = create_bipartite_graph_from_edges(edges, 3, 3)
matching, size = algorithm.solve()
print(f"Maximum matching: {matching}")
print(f"Matching size: {size}")
```

### Running Comparisons

```bash
# Run main comparison with visualizations
cd src
python main.py

# Run comprehensive benchmarks
cd benchmarks  
python run_benchmarks.py

# Run unit tests
cd tests
python test_matching.py
```

## Key Results

The project demonstrates:

1. **Algorithm Trade-offs**: Hungarian algorithm excels for weighted problems while Hopcroft-Karp is optimal for cardinality matching
2. **Scalability**: Hopcroft-Karp generally outperforms Hungarian on sparse graphs, while Hungarian handles dense weighted graphs efficiently
3. **Practical Applications**: Both algorithms solve complementary problems in network optimization

## Technical Report

A comprehensive technical report is available in `report.typ` (Typst format) covering:
- Theoretical foundations and complexity analysis
- Implementation details and algorithmic choices  
- Experimental methodology and performance evaluation
- Comparative analysis and practical recommendations

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Standard library modules (collections, typing, etc.)

## License

This project is available for educational and research purposes.

## Author

Created as an educational implementation for understanding classical graph algorithms and their practical applications in computer science.
