"""
Main benchmarking and analysis script for shortest path algorithms.

This script generates various types of graphs, runs all three algorithms
(Dijkstra, Bellman-Ford, A*), measures their performance, and creates
visualizations of the results.
"""

OUTPUT_DIR = "output/"

import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any

from graph import Graph, GridGraph, generate_random_graph, generate_grid_with_obstacles
from dijkstra import dijkstra, DijkstraResult
from bellman_ford import bellman_ford, BellmanFordResult
from astar import astar_graph, astar_grid, zero_heuristic, AStarResult


class BenchmarkResult:
    """Container for benchmark results across all algorithms."""
    
    def __init__(self):
        self.graph_type: str = ""
        self.graph_size: int = 0
        self.edge_count: int = 0
        self.has_negative_edges: bool = False
        
        # Algorithm results
        self.dijkstra_time: float = 0.0
        self.dijkstra_operations: int = 0
        self.dijkstra_path_length: int = 0
        self.dijkstra_cost: float = float('inf')
        
        self.bellman_ford_time: float = 0.0
        self.bellman_ford_operations: int = 0
        self.bellman_ford_path_length: int = 0
        self.bellman_ford_cost: float = float('inf')
        self.has_negative_cycle: bool = False
        
        self.astar_time: float = 0.0
        self.astar_operations: int = 0
        self.astar_path_length: int = 0
        self.astar_cost: float = float('inf')


def benchmark_graph_algorithms(graph: Graph, start: int, end: int) -> BenchmarkResult:
    """
    Benchmark all three algorithms on a given graph.
    
    Args:
        graph: The graph to benchmark
        start: Source vertex
        end: Target vertex
        
    Returns:
        BenchmarkResult containing performance metrics
    """
    result = BenchmarkResult()
    result.graph_size = len(graph.get_vertices())
    result.edge_count = graph.get_edge_count()
    result.has_negative_edges = graph.has_negative_edges()
    
    # Benchmark Dijkstra's algorithm (only if no negative edges)
    if not graph.has_negative_edges():
        try:
            start_time = time.perf_counter()
            dijkstra_result = dijkstra(graph, start, end)
            end_time = time.perf_counter()
            
            result.dijkstra_time = end_time - start_time
            result.dijkstra_operations = dijkstra_result.operations_count
            result.dijkstra_path_length = len(dijkstra_result.path)
            result.dijkstra_cost = dijkstra_result.path_cost
        except Exception as e:
            print(f"Dijkstra failed: {e}")
    
    # Benchmark Bellman-Ford algorithm
    try:
        start_time = time.perf_counter()
        bf_result = bellman_ford(graph, start, end)
        end_time = time.perf_counter()
        
        result.bellman_ford_time = end_time - start_time
        result.bellman_ford_operations = bf_result.operations_count
        result.bellman_ford_path_length = len(bf_result.path)
        result.bellman_ford_cost = bf_result.path_cost
        result.has_negative_cycle = bf_result.has_negative_cycle
    except Exception as e:
        print(f"Bellman-Ford failed: {e}")
    
    # Benchmark A* algorithm with zero heuristic (equivalent to Dijkstra)
    if not graph.has_negative_edges():
        try:
            start_time = time.perf_counter()
            astar_result = astar_graph(graph, start, end, zero_heuristic)
            end_time = time.perf_counter()
            
            result.astar_time = end_time - start_time
            result.astar_operations = astar_result.operations_count
            result.astar_path_length = len(astar_result.path)
            result.astar_cost = astar_result.path_cost
        except Exception as e:
            print(f"A* failed: {e}")
    
    return result


def benchmark_grid_algorithms(grid: GridGraph, start: Tuple[int, int], 
                            end: Tuple[int, int]) -> Dict[str, Any]:
    """
    Benchmark A* algorithm on a grid graph.
    
    Args:
        grid: The grid graph
        start: Starting position
        end: Target position
        
    Returns:
        Dictionary containing benchmark results
    """
    result = {
        'grid_size': grid.width * grid.height,
        'obstacles': len(grid.obstacles),
        'astar_time': 0.0,
        'astar_operations': 0,
        'astar_path_length': 0,
        'astar_cost': float('inf')
    }
    
    try:
        start_time = time.perf_counter()
        astar_result = astar_grid(grid, start, end)
        end_time = time.perf_counter()
        
        result['astar_time'] = end_time - start_time
        result['astar_operations'] = astar_result.operations_count
        result['astar_path_length'] = len(astar_result.path)
        result['astar_cost'] = astar_result.path_cost
    except Exception as e:
        print(f"A* on grid failed: {e}")
    
    return result


def generate_test_graphs() -> List[Tuple[Graph, int, int, str]]:
    """
    Generate various types of test graphs for benchmarking.
    
    Returns:
        List of (graph, start, end, description) tuples
    """
    test_graphs = []
    random.seed(42)  # For reproducible results
    
    # Small sparse graphs
    for size in [10, 20, 50]:
        graph = generate_random_graph(size, 0.1, 1.0, 10.0, directed=False)
        vertices = list(graph.get_vertices())
        start, end = random.sample(vertices, 2)
        test_graphs.append((graph, start, end, f"sparse_{size}"))
    
    # Small dense graphs
    for size in [10, 20, 30]:
        graph = generate_random_graph(size, 0.5, 1.0, 10.0, directed=False)
        vertices = list(graph.get_vertices())
        start, end = random.sample(vertices, 2)
        test_graphs.append((graph, start, end, f"dense_{size}"))
    
    # Graphs with negative edges (smaller sizes due to Bellman-Ford complexity)
    for size in [10, 15, 20]:
        graph = generate_random_graph(size, 0.3, 1.0, 10.0, 
                                    allow_negative=True, directed=False)
        vertices = list(graph.get_vertices())
        start, end = random.sample(vertices, 2)
        test_graphs.append((graph, start, end, f"negative_{size}"))
    
    return test_graphs


def run_graph_benchmarks() -> pd.DataFrame:
    """
    Run comprehensive benchmarks on various graph types.
    
    Returns:
        DataFrame containing all benchmark results
    """
    print("Generating test graphs...")
    test_graphs = generate_test_graphs()
    
    results = []
    
    print(f"Running benchmarks on {len(test_graphs)} graphs...")
    for i, (graph, start, end, description) in enumerate(test_graphs):
        print(f"  Processing graph {i+1}/{len(test_graphs)}: {description}")
        
        benchmark_result = benchmark_graph_algorithms(graph, start, end)
        benchmark_result.graph_type = description
        
        # Convert to dictionary for DataFrame
        result_dict = {
            'graph_type': benchmark_result.graph_type,
            'graph_size': benchmark_result.graph_size,
            'edge_count': benchmark_result.edge_count,
            'has_negative_edges': benchmark_result.has_negative_edges,
            'dijkstra_time': benchmark_result.dijkstra_time,
            'dijkstra_operations': benchmark_result.dijkstra_operations,
            'dijkstra_path_length': benchmark_result.dijkstra_path_length,
            'dijkstra_cost': benchmark_result.dijkstra_cost,
            'bellman_ford_time': benchmark_result.bellman_ford_time,
            'bellman_ford_operations': benchmark_result.bellman_ford_operations,
            'bellman_ford_path_length': benchmark_result.bellman_ford_path_length,
            'bellman_ford_cost': benchmark_result.bellman_ford_cost,
            'has_negative_cycle': benchmark_result.has_negative_cycle,
            'astar_time': benchmark_result.astar_time,
            'astar_operations': benchmark_result.astar_operations,
            'astar_path_length': benchmark_result.astar_path_length,
            'astar_cost': benchmark_result.astar_cost
        }
        
        results.append(result_dict)
    
    return pd.DataFrame(results)


def run_grid_benchmarks() -> pd.DataFrame:
    """
    Run benchmarks on grid graphs for A* algorithm.
    
    Returns:
        DataFrame containing grid benchmark results
    """
    print("Running grid benchmarks...")
    results = []
    random.seed(42)
    
    # Various grid sizes and obstacle densities
    grid_configs = [
        (10, 10, 0.1), (10, 10, 0.2), (10, 10, 0.3),
        (20, 20, 0.1), (20, 20, 0.2), (20, 20, 0.3),
        (30, 30, 0.1), (30, 30, 0.2)
    ]
    
    for width, height, obstacle_prob in grid_configs:
        print(f"  Processing {width}x{height} grid with {obstacle_prob:.1%} obstacles")
        
        grid = generate_grid_with_obstacles(width, height, obstacle_prob)
        
        # Find valid start and end positions
        valid_positions = [(x, y) for x in range(width) for y in range(height)
                          if (x, y) not in grid.obstacles]
        
        if len(valid_positions) < 2:
            continue
        
        start, end = random.sample(valid_positions, 2)
        
        result = benchmark_grid_algorithms(grid, start, end)
        result['width'] = width
        result['height'] = height
        result['obstacle_prob'] = obstacle_prob
        
        results.append(result)
    
    return pd.DataFrame(results)


def create_performance_visualizations(df: pd.DataFrame) -> None:
    """
    Create visualizations of algorithm performance.
    
    Args:
        df: DataFrame containing benchmark results
    """
    # Filter out results with infinite times or costs
    df_clean = df[(df['dijkstra_time'] > 0) | (df['bellman_ford_time'] > 0) | (df['astar_time'] > 0)]
    df_clean = df_clean[df_clean['dijkstra_cost'] != float('inf')]
    
    if df_clean.empty:
        print("No valid results to visualize")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Shortest Path Algorithms Performance Comparison', fontsize=16)
    
    # Plot 1: Execution Time vs Graph Size
    ax1 = axes[0, 0]
    non_negative_graphs = df_clean[~df_clean['has_negative_edges']]
    
    if not non_negative_graphs.empty:
        ax1.scatter(non_negative_graphs['graph_size'], non_negative_graphs['dijkstra_time'], 
                   label='Dijkstra', alpha=0.7, color='blue')
        ax1.scatter(non_negative_graphs['graph_size'], non_negative_graphs['astar_time'], 
                   label='A* (zero heuristic)', alpha=0.7, color='green')
    
    ax1.scatter(df_clean['graph_size'], df_clean['bellman_ford_time'], 
               label='Bellman-Ford', alpha=0.7, color='red')
    
    ax1.set_xlabel('Graph Size (vertices)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time vs Graph Size')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot 2: Operations Count vs Graph Size
    ax2 = axes[0, 1]
    if not non_negative_graphs.empty:
        ax2.scatter(non_negative_graphs['graph_size'], non_negative_graphs['dijkstra_operations'], 
                   label='Dijkstra', alpha=0.7, color='blue')
        ax2.scatter(non_negative_graphs['graph_size'], non_negative_graphs['astar_operations'], 
                   label='A*', alpha=0.7, color='green')
    
    ax2.scatter(df_clean['graph_size'], df_clean['bellman_ford_operations'], 
               label='Bellman-Ford', alpha=0.7, color='red')
    
    ax2.set_xlabel('Graph Size (vertices)')
    ax2.set_ylabel('Operations Count')
    ax2.set_title('Operations Count vs Graph Size')
    ax2.legend()
    ax2.set_yscale('log')
    
    # Plot 3: Algorithm Comparison by Graph Type
    ax3 = axes[1, 0]
    graph_types = df_clean['graph_type'].unique()
    x_pos = np.arange(len(graph_types))
    
    dijkstra_times = []
    bellman_times = []
    astar_times = []
    
    for gt in graph_types:
        subset = df_clean[df_clean['graph_type'] == gt]
        dijkstra_times.append(subset['dijkstra_time'].mean() if subset['dijkstra_time'].max() > 0 else 0)
        bellman_times.append(subset['bellman_ford_time'].mean())
        astar_times.append(subset['astar_time'].mean() if subset['astar_time'].max() > 0 else 0)
    
    width = 0.25
    ax3.bar(x_pos - width, dijkstra_times, width, label='Dijkstra', alpha=0.8)
    ax3.bar(x_pos, bellman_times, width, label='Bellman-Ford', alpha=0.8)
    ax3.bar(x_pos + width, astar_times, width, label='A*', alpha=0.8)
    
    ax3.set_xlabel('Graph Type')
    ax3.set_ylabel('Average Execution Time (seconds)')
    ax3.set_title('Average Performance by Graph Type')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(graph_types, rotation=45)
    ax3.legend()
    
    # Plot 4: Path Cost Verification
    ax4 = axes[1, 1]
    valid_paths = df_clean[(df_clean['dijkstra_cost'] != float('inf')) & 
                          (df_clean['bellman_ford_cost'] != float('inf'))]
    
    if not valid_paths.empty:
        ax4.scatter(valid_paths['dijkstra_cost'], valid_paths['bellman_ford_cost'], 
                   alpha=0.7, color='purple')
        
        # Add diagonal line for perfect correlation
        min_cost = min(valid_paths['dijkstra_cost'].min(), valid_paths['bellman_ford_cost'].min())
        max_cost = max(valid_paths['dijkstra_cost'].max(), valid_paths['bellman_ford_cost'].max())
        ax4.plot([min_cost, max_cost], [min_cost, max_cost], 'r--', alpha=0.5)
        
        ax4.set_xlabel('Dijkstra Path Cost')
        ax4.set_ylabel('Bellman-Ford Path Cost')
        ax4.set_title('Path Cost Correlation\n(Should be on diagonal)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'algorithm_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_grid_visualizations(df_grid: pd.DataFrame) -> None:
    """
    Create visualizations for grid-based A* performance.
    
    Args:
        df_grid: DataFrame containing grid benchmark results
    """
    if df_grid.empty:
        print("No grid results to visualize")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('A* Performance on Grid Graphs', fontsize=14)
    
    # Plot 1: Performance vs Grid Size
    ax1 = axes[0]
    ax1.scatter(df_grid['grid_size'], df_grid['astar_time'], 
               c=df_grid['obstacle_prob'], cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Grid Size (cells)')
    ax1.set_ylabel('A* Execution Time (seconds)')
    ax1.set_title('A* Performance vs Grid Size')
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label('Obstacle Probability')
    
    # Plot 2: Operations vs Obstacles
    ax2 = axes[1]
    ax2.scatter(df_grid['obstacles'], df_grid['astar_operations'], 
               c=df_grid['grid_size'], cmap='plasma', alpha=0.7)
    ax2.set_xlabel('Number of Obstacles')
    ax2.set_ylabel('A* Operations Count')
    ax2.set_title('A* Operations vs Obstacles')
    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar2.set_label('Grid Size')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'grid_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("=== Shortest Path Algorithms Benchmark ===\n")
    
    # Run graph benchmarks
    df_results = run_graph_benchmarks()
    
    # Run grid benchmarks
    df_grid = run_grid_benchmarks()
    
    # Save results to CSV
    df_results.to_csv(OUTPUT_DIR + 'graph_benchmark_results.csv', index=False)
    df_grid.to_csv(OUTPUT_DIR + 'grid_benchmark_results.csv', index=False)
    
    print(f"\nBenchmark completed!")
    print(f"Graph results: {len(df_results)} test cases")
    print(f"Grid results: {len(df_grid)} test cases")
    
    # Display summary statistics
    print("\n=== Summary Statistics ===")
    if not df_results.empty:
        non_neg = df_results[~df_results['has_negative_edges']]
        if not non_neg.empty:
            print(f"Average Dijkstra time: {non_neg['dijkstra_time'].mean():.6f}s")
            print(f"Average A* time: {non_neg['astar_time'].mean():.6f}s")
        print(f"Average Bellman-Ford time: {df_results['bellman_ford_time'].mean():.6f}s")
    
    if not df_grid.empty:
        print(f"Average A* time (grids): {df_grid['astar_time'].mean():.6f}s")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_performance_visualizations(df_results)
    create_grid_visualizations(df_grid)
    
    print("\nBenchmark complete! Check the generated CSV files and visualizations.")


if __name__ == "__main__":
    main()
