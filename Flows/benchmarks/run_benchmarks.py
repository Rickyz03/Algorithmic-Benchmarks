"""
Benchmark suite for comparing Ford-Fulkerson and Edmonds-Karp algorithms.

This module runs performance benchmarks on various graph types and sizes
to compare the efficiency of both maximum flow algorithms.
"""

OUTPUT_DIR = "output/"

import time
import sys
import os
from typing import Dict, List, Tuple, Any
import json

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithms.ford_fulkerson import FordFulkerson
from algorithms.edmonds_karp import EdmondsKarp
from utils.graph_generator import FlowNetworkGenerator


class MaxFlowBenchmark:
    """
    Benchmark suite for maximum flow algorithms.
    
    Runs performance tests on different graph types and collects
    timing and iteration statistics for analysis.
    """
    
    def __init__(self):
        """Initialize the benchmark suite."""
        self.results = []
        self.generator = FlowNetworkGenerator()
    
    def benchmark_algorithm(self, algorithm_class, graph: Dict[int, List[Tuple[int, int]]], 
                          source: int, sink: int, algorithm_name: str) -> Dict[str, Any]:
        """
        Benchmark a single algorithm on a given graph.
        
        Args:
            algorithm_class: Class of the algorithm to benchmark
            graph: Graph to run the algorithm on
            source: Source node
            sink: Sink node
            algorithm_name: Name of the algorithm for reporting
            
        Returns:
            Dictionary containing benchmark results
        """
        algorithm = algorithm_class(graph)
        
        # Measure execution time
        start_time = time.perf_counter()
        max_flow_value, iterations = algorithm.max_flow(source, sink)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        return {
            'algorithm': algorithm_name,
            'max_flow': max_flow_value,
            'iterations': iterations,
            'execution_time': execution_time,
            'source': source,
            'sink': sink
        }
    
    def benchmark_graph_type(self, graph_type: str, graph_params: Dict[str, Any], 
                           num_trials: int = 5) -> List[Dict[str, Any]]:
        """
        Benchmark both algorithms on a specific graph type.
        
        Args:
            graph_type: Type of graph to generate
            graph_params: Parameters for graph generation
            num_trials: Number of trials to run for averaging
            
        Returns:
            List of benchmark results for all trials
        """
        results = []
        
        for trial in range(num_trials):
            # Generate graph based on type
            if graph_type == 'random':
                graph = self.generator.random_graph(**graph_params, seed=trial)
            elif graph_type == 'linear':
                graph = self.generator.linear_chain(**graph_params, seed=trial)
            elif graph_type == 'dense':
                graph = self.generator.dense_graph(**graph_params, seed=trial)
            elif graph_type == 'bottleneck':
                graph = self.generator.bottleneck_graph(**graph_params, seed=trial)
            elif graph_type == 'bipartite':
                graph = self.generator.bipartite_graph(**graph_params, seed=trial)
            elif graph_type == 'grid':
                graph = self.generator.grid_graph(**graph_params, seed=trial)
            else:
                raise ValueError(f"Unknown graph type: {graph_type}")
            
            # Determine source and sink
            if graph_type == 'bipartite':
                source = 0
                left_size = graph_params.get('left_size', 3)
                right_size = graph_params.get('right_size', 3)
                sink = left_size + right_size + 1
            elif graph_type == 'grid':
                source = 0
                rows = graph_params.get('rows', 3)
                cols = graph_params.get('cols', 3)
                sink = rows * cols - 1
            else:
                nodes = list(graph.keys())
                if not nodes:
                    continue
                source = min(nodes)
                sink = max(nodes)
            
            # Skip if source equals sink or graph is empty
            if source == sink or not graph:
                continue
            
            # Benchmark both algorithms
            ff_result = self.benchmark_algorithm(FordFulkerson, graph, source, sink, 'Ford-Fulkerson')
            ek_result = self.benchmark_algorithm(EdmondsKarp, graph, source, sink, 'Edmonds-Karp')
            
            # Add graph metadata
            graph_info = {
                'graph_type': graph_type,
                'trial': trial,
                'graph_params': graph_params,
                'num_nodes': len(set(graph.keys()) | {v for neighbors in graph.values() for v, _ in neighbors}),
                'num_edges': sum(len(neighbors) for neighbors in graph.values())
            }
            
            ff_result.update(graph_info)
            ek_result.update(graph_info)
            
            results.extend([ff_result, ek_result])
        
        return results
    
    def run_comprehensive_benchmark(self) -> List[Dict[str, Any]]:
        """
        Run comprehensive benchmarks on various graph types and sizes.
        
        Returns:
            List of all benchmark results
        """
        print("Starting comprehensive maximum flow benchmarks...")
        
        benchmark_configs = [
            # Random graphs with increasing size
            ('random', {'num_nodes': 10, 'num_edges': 20, 'max_capacity': 50}),
            ('random', {'num_nodes': 20, 'num_edges': 50, 'max_capacity': 50}),
            ('random', {'num_nodes': 30, 'num_edges': 80, 'max_capacity': 50}),
            
            # Linear chains
            ('linear', {'num_nodes': 10, 'max_capacity': 50}),
            ('linear', {'num_nodes': 20, 'max_capacity': 50}),
            ('linear', {'num_nodes': 30, 'max_capacity': 50}),
            
            # Dense graphs
            ('dense', {'num_nodes': 8, 'connection_prob': 0.4, 'max_capacity': 50}),
            ('dense', {'num_nodes': 12, 'connection_prob': 0.3, 'max_capacity': 50}),
            ('dense', {'num_nodes': 16, 'connection_prob': 0.25, 'max_capacity': 50}),
            
            # Bottleneck graphs
            ('bottleneck', {'num_layers': 4, 'layer_size': 3, 'bottleneck_capacity': 2, 'other_capacity': 20}),
            ('bottleneck', {'num_layers': 5, 'layer_size': 4, 'bottleneck_capacity': 1, 'other_capacity': 30}),
            
            # Bipartite graphs
            ('bipartite', {'left_size': 5, 'right_size': 5, 'connection_prob': 0.4, 'max_capacity': 30}),
            ('bipartite', {'left_size': 8, 'right_size': 6, 'connection_prob': 0.3, 'max_capacity': 40}),
            
            # Grid graphs
            ('grid', {'rows': 4, 'cols': 4, 'max_capacity': 25}),
            ('grid', {'rows': 5, 'cols': 5, 'max_capacity': 30}),
        ]
        
        all_results = []
        
        for i, (graph_type, params) in enumerate(benchmark_configs):
            print(f"Benchmarking {graph_type} graphs ({i+1}/{len(benchmark_configs)})...")
            try:
                results = self.benchmark_graph_type(graph_type, params, num_trials=3)
                all_results.extend(results)
                print(f"  Completed {len(results)} benchmark runs")
            except Exception as e:
                print(f"  Error benchmarking {graph_type}: {e}")
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = OUTPUT_DIR + 'benchmark_results.json'):
        """
        Save benchmark results to a JSON file.
        
        Args:
            results: List of benchmark results
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """
        Print a summary of benchmark results.
        
        Args:
            results: List of benchmark results
        """
        if not results:
            print("No results to summarize")
            return
        
        # Group results by algorithm
        ff_results = [r for r in results if r['algorithm'] == 'Ford-Fulkerson']
        ek_results = [r for r in results if r['algorithm'] == 'Edmonds-Karp']
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"Total benchmark runs: {len(results)}")
        print(f"Ford-Fulkerson runs: {len(ff_results)}")
        print(f"Edmonds-Karp runs: {len(ek_results)}")
        
        if ff_results:
            avg_ff_time = sum(r['execution_time'] for r in ff_results) / len(ff_results)
            avg_ff_iter = sum(r['iterations'] for r in ff_results) / len(ff_results)
            print(f"\nFord-Fulkerson averages:")
            print(f"  Execution time: {avg_ff_time:.6f} seconds")
            print(f"  Iterations: {avg_ff_iter:.2f}")
        
        if ek_results:
            avg_ek_time = sum(r['execution_time'] for r in ek_results) / len(ek_results)
            avg_ek_iter = sum(r['iterations'] for r in ek_results) / len(ek_results)
            print(f"\nEdmonds-Karp averages:")
            print(f"  Execution time: {avg_ek_time:.6f} seconds")
            print(f"  Iterations: {avg_ek_iter:.2f}")
        
        # Performance comparison
        if ff_results and ek_results:
            time_ratio = avg_ff_time / avg_ek_time if avg_ek_time > 0 else float('inf')
            iter_ratio = avg_ff_iter / avg_ek_iter if avg_ek_iter > 0 else float('inf')
            
            print(f"\nPerformance comparison (FF/EK ratios):")
            print(f"  Time ratio: {time_ratio:.2f}")
            print(f"  Iteration ratio: {iter_ratio:.2f}")
        
        # Graph type breakdown
        graph_types = set(r['graph_type'] for r in results)
        print(f"\nGraph types tested: {', '.join(graph_types)}")
        
        for graph_type in graph_types:
            type_results = [r for r in results if r['graph_type'] == graph_type]
            if type_results:
                avg_nodes = sum(r['num_nodes'] for r in type_results) / len(type_results)
                avg_edges = sum(r['num_edges'] for r in type_results) / len(type_results)
                print(f"  {graph_type}: {len(type_results)} runs, avg {avg_nodes:.1f} nodes, {avg_edges:.1f} edges")


def main():
    """Main function to run benchmarks."""
    benchmark = MaxFlowBenchmark()
    
    # Run comprehensive benchmarks
    results = benchmark.run_comprehensive_benchmark()
    
    # Save and display results
    benchmark.save_results(results)
    benchmark.print_summary(results)
    
    print("\nBenchmarking complete!")


if __name__ == '__main__':
    main()
