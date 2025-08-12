"""
Comprehensive benchmarking suite for TSP algorithms.

This module provides extensive benchmarking capabilities to compare
different TSP algorithms across various problem sizes and instance types.
"""

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple, Callable, Optional
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_generator import generate_euclidean_tsp_instance, generate_metric_tsp_instance
from algorithms.mst_approximation import mst_2_approximation
from algorithms.exact_algorithms import brute_force_tsp, held_karp_tsp, BranchAndBound
from algorithms.heuristic_algorithms import (
    nearest_neighbor_tsp, multi_start_nearest_neighbor,
    nearest_neighbor_with_2opt, multi_start_nn_with_2opt,
    random_restart_2opt
)


class TSPBenchmark:
    """Comprehensive TSP algorithm benchmarking suite."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results = []
        self.algorithms = {}
        self.instance_generators = {}
        
        # Register algorithms
        self._register_algorithms()
        
        # Register instance generators
        self._register_generators()
    
    def _register_algorithms(self):
        """Register all TSP algorithms for benchmarking."""
        
        # Exact algorithms (only for small instances)
        self.algorithms['brute_force'] = {
            'name': 'Brute Force',
            'function': brute_force_tsp,
            'max_size': 8,
            'type': 'exact'
        }
        
        self.algorithms['held_karp'] = {
            'name': 'Held-Karp DP',
            'function': held_karp_tsp,
            'max_size': 15,
            'type': 'exact'
        }
        
        self.algorithms['branch_bound'] = {
            'name': 'Branch & Bound',
            'function': lambda dm: BranchAndBound(dm).solve(),
            'max_size': 12,
            'type': 'exact'
        }
        
        # Approximation algorithms
        self.algorithms['mst_2approx'] = {
            'name': 'MST 2-Approximation',
            'function': mst_2_approximation,
            'max_size': 1000,
            'type': 'approximation'
        }
        
        # Heuristic algorithms
        self.algorithms['nearest_neighbor'] = {
            'name': 'Nearest Neighbor',
            'function': lambda dm: nearest_neighbor_tsp(dm, 0),
            'max_size': 1000,
            'type': 'heuristic'
        }
        
        self.algorithms['multi_start_nn'] = {
            'name': 'Multi-start NN',
            'function': multi_start_nearest_neighbor,
            'max_size': 1000,
            'type': 'heuristic'
        }
        
        self.algorithms['nn_2opt'] = {
            'name': 'NN + 2-opt',
            'function': lambda dm: nearest_neighbor_with_2opt(dm, 0),
            'max_size': 1000,
            'type': 'heuristic'
        }
        
        self.algorithms['multi_start_nn_2opt'] = {
            'name': 'Multi-start NN + 2-opt',
            'function': multi_start_nn_with_2opt,
            'max_size': 1000,
            'type': 'heuristic'
        }
        
        self.algorithms['random_restart_2opt'] = {
            'name': 'Random Restart + 2-opt',
            'function': lambda dm: random_restart_2opt(dm, num_restarts=10, seed=42),
            'max_size': 1000,
            'type': 'heuristic'
        }
    
    def _register_generators(self):
        """Register instance generators."""
        self.instance_generators['euclidean'] = {
            'name': 'Euclidean',
            'function': generate_euclidean_tsp_instance,
            'has_coordinates': True
        }
        
        self.instance_generators['metric'] = {
            'name': 'Random Metric',
            'function': lambda n, seed: (generate_metric_tsp_instance(n, seed=seed), None),
            'has_coordinates': False
        }
    
    def benchmark_algorithm(self, algorithm_key: str, distance_matrix: np.ndarray, 
                           timeout: float = 60.0) -> Dict:
        """
        Benchmark a single algorithm on a given instance.
        
        Args:
            algorithm_key: Key identifying the algorithm
            distance_matrix: Distance matrix for the TSP instance
            timeout: Maximum time allowed in seconds
            
        Returns:
            Dictionary with benchmark results
        """
        if algorithm_key not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_key}")
        
        algorithm = self.algorithms[algorithm_key]
        n = distance_matrix.shape[0]
        
        # Check size limit
        if n > algorithm['max_size']:
            return {
                'algorithm': algorithm_key,
                'algorithm_name': algorithm['name'],
                'size': n,
                'tour': None,
                'length': None,
                'time': None,
                'status': 'skipped_too_large',
                'nodes_explored': None
            }
        
        try:
            start_time = time.time()
            
            # Special handling for Branch and Bound to get nodes explored
            if algorithm_key == 'branch_bound':
                bb_solver = BranchAndBound(distance_matrix)
                tour, length = bb_solver.solve()
                nodes_explored = bb_solver.nodes_explored
            else:
                tour, length = algorithm['function'](distance_matrix)
                nodes_explored = None
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Check timeout
            if elapsed_time > timeout:
                return {
                    'algorithm': algorithm_key,
                    'algorithm_name': algorithm['name'],
                    'size': n,
                    'tour': None,
                    'length': None,
                    'time': elapsed_time,
                    'status': 'timeout',
                    'nodes_explored': nodes_explored
                }
            
            return {
                'algorithm': algorithm_key,
                'algorithm_name': algorithm['name'],
                'size': n,
                'tour': tour,
                'length': length,
                'time': elapsed_time,
                'status': 'success',
                'nodes_explored': nodes_explored
            }
            
        except Exception as e:
            return {
                'algorithm': algorithm_key,
                'algorithm_name': algorithm['name'],
                'size': n,
                'tour': None,
                'length': None,
                'time': None,
                'status': f'error_{str(e)[:50]}',
                'nodes_explored': None
            }
    
    def run_size_scaling_benchmark(self, sizes: List[int], 
                                  instance_type: str = 'euclidean',
                                  algorithms: Optional[List[str]] = None,
                                  num_instances: int = 3,
                                  timeout: float = 60.0) -> pd.DataFrame:
        """
        Run benchmark across different problem sizes.
        
        Args:
            sizes: List of problem sizes to test
            instance_type: Type of instances to generate
            algorithms: List of algorithm keys to test (None for all applicable)
            num_instances: Number of instances per size
            timeout: Timeout per algorithm run
            
        Returns:
            DataFrame with benchmark results
        """
        results = []
        
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        for size in sizes:
            print(f"\nTesting size {size}...")
            
            for instance_idx in range(num_instances):
                print(f"  Instance {instance_idx + 1}/{num_instances}")
                
                # Generate instance
                seed = 42 + instance_idx
                generator = self.instance_generators[instance_type]
                
                if generator['has_coordinates']:
                    distance_matrix, coordinates = generator['function'](size, seed)
                else:
                    distance_matrix, coordinates = generator['function'](size, seed)
                
                # Test each algorithm
                for alg_key in algorithms:
                    if size <= self.algorithms[alg_key]['max_size']:
                        print(f"    Running {self.algorithms[alg_key]['name']}...")
                        result = self.benchmark_algorithm(alg_key, distance_matrix, timeout)
                        result['instance_type'] = instance_type
                        result['instance_idx'] = instance_idx
                        result['seed'] = seed
                        results.append(result)
        
        return pd.DataFrame(results)
    
    def run_algorithm_comparison(self, size: int, instance_type: str = 'euclidean',
                               num_instances: int = 10) -> pd.DataFrame:
        """
        Compare all applicable algorithms on instances of fixed size.
        
        Args:
            size: Problem size
            instance_type: Type of instances to generate
            num_instances: Number of instances to test
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        applicable_algorithms = [
            key for key, alg in self.algorithms.items() 
            if size <= alg['max_size']
        ]
        
        print(f"Comparing algorithms on {num_instances} instances of size {size}")
        print(f"Algorithms: {[self.algorithms[key]['name'] for key in applicable_algorithms]}")
        
        for instance_idx in range(num_instances):
            print(f"\nInstance {instance_idx + 1}/{num_instances}")
            
            # Generate instance
            seed = 42 + instance_idx
            generator = self.instance_generators[instance_type]
            
            if generator['has_coordinates']:
                distance_matrix, coordinates = generator['function'](size, seed)
            else:
                distance_matrix, coordinates = generator['function'](size, seed)
            
            # Test each algorithm
            for alg_key in applicable_algorithms:
                result = self.benchmark_algorithm(alg_key, distance_matrix)
                result['instance_type'] = instance_type
                result['instance_idx'] = instance_idx
                result['seed'] = seed
                results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_approximation_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze approximation ratios relative to best known solutions.
        
        Args:
            df: DataFrame with benchmark results
            
        Returns:
            DataFrame with approximation ratio analysis
        """
        analysis_results = []
        
        # Group by instance
        for (size, instance_type, instance_idx), group in df.groupby(['size', 'instance_type', 'instance_idx']):
            successful_results = group[group['status'] == 'success'].copy()
            
            if len(successful_results) == 0:
                continue
            
            # Find best (optimal) solution
            best_length = successful_results['length'].min()
            
            # Calculate approximation ratios
            for _, row in successful_results.iterrows():
                ratio = row['length'] / best_length if best_length > 0 else 1.0
                
                analysis_results.append({
                    'algorithm': row['algorithm'],
                    'algorithm_name': row['algorithm_name'],
                    'algorithm_type': self.algorithms[row['algorithm']]['type'],
                    'size': size,
                    'instance_type': instance_type,
                    'instance_idx': instance_idx,
                    'length': row['length'],
                    'best_length': best_length,
                    'approximation_ratio': ratio,
                    'time': row['time']
                })
        
        return pd.DataFrame(analysis_results)
    
    def plot_scaling_results(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot algorithm scaling results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Group by algorithm and size, calculate means
        grouped = df[df['status'] == 'success'].groupby(['algorithm_name', 'size']).agg({
            'time': 'mean',
            'length': 'mean'
        }).reset_index()
        
        # Plot 1: Execution time vs size
        for alg_name in grouped['algorithm_name'].unique():
            alg_data = grouped[grouped['algorithm_name'] == alg_name]
            ax1.plot(alg_data['size'], alg_data['time'], marker='o', label=alg_name)
        
        ax1.set_xlabel('Problem Size')
        ax1.set_ylabel('Average Execution Time (s)')
        ax1.set_title('Algorithm Scaling: Execution Time')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Solution quality vs size
        for alg_name in grouped['algorithm_name'].unique():
            alg_data = grouped[grouped['algorithm_name'] == alg_name]
            ax2.plot(alg_data['size'], alg_data['length'], marker='o', label=alg_name)
        
        ax2.set_xlabel('Problem Size')
        ax2.set_ylabel('Average Tour Length')
        ax2.set_title('Algorithm Scaling: Solution Quality')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Success rate by algorithm
        success_rates = df.groupby('algorithm_name').agg({
            'status': lambda x: (x == 'success').mean()
        }).reset_index()
        success_rates.columns = ['algorithm_name', 'success_rate']
        
        ax3.bar(success_rates['algorithm_name'], success_rates['success_rate'])
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Algorithm Success Rates')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Time distribution by algorithm type
        successful_df = df[df['status'] == 'success'].copy()
        alg_types = successful_df['algorithm'].map(lambda x: self.algorithms[x]['type'])
        successful_df['algorithm_type'] = alg_types
        
        type_times = successful_df.groupby('algorithm_type')['time'].apply(list).to_dict()
        ax4.boxplot(type_times.values(), labels=type_times.keys())
        ax4.set_ylabel('Execution Time (s)')
        ax4.set_title('Time Distribution by Algorithm Type')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_approximation_analysis(self, analysis_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot approximation ratio analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Approximation ratios by algorithm
        algorithms = analysis_df['algorithm_name'].unique()
        ratios_by_alg = [analysis_df[analysis_df['algorithm_name'] == alg]['approximation_ratio'].values 
                        for alg in algorithms]
        
        ax1.boxplot(ratios_by_alg, labels=algorithms)
        ax1.set_ylabel('Approximation Ratio')
        ax1.set_title('Approximation Ratios by Algorithm')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Optimal')
        ax1.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='2-approx bound')
        ax1.legend()
        
        # Plot 2: Approximation ratio vs problem size
        for alg_name in algorithms:
            alg_data = analysis_df[analysis_df['algorithm_name'] == alg_name]
            sizes = sorted(alg_data['size'].unique())
            avg_ratios = [alg_data[alg_data['size'] == s]['approximation_ratio'].mean() for s in sizes]
            ax2.plot(sizes, avg_ratios, marker='o', label=alg_name)
        
        ax2.set_xlabel('Problem Size')
        ax2.set_ylabel('Average Approximation Ratio')
        ax2.set_title('Approximation Quality vs Problem Size')
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Time vs quality trade-off
        for alg_type in analysis_df['algorithm_type'].unique():
            type_data = analysis_df[analysis_df['algorithm_type'] == alg_type]
            ax3.scatter(type_data['time'], type_data['approximation_ratio'], 
                       label=alg_type, alpha=0.6)
        
        ax3.set_xlabel('Execution Time (s)')
        ax3.set_ylabel('Approximation Ratio')
        ax3.set_title('Time vs Quality Trade-off')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Algorithm type performance comparison
        type_stats = analysis_df.groupby('algorithm_type').agg({
            'approximation_ratio': ['mean', 'std'],
            'time': ['mean', 'std']
        }).round(3)
        
        ax4.axis('tight')
        ax4.axis('off')
        table_data = []
        for alg_type in type_stats.index:
            row = [
                alg_type,
                f"{type_stats.loc[alg_type, ('approximation_ratio', 'mean')]:.3f} ± {type_stats.loc[alg_type, ('approximation_ratio', 'std')]:.3f}",
                f"{type_stats.loc[alg_type, ('time', 'mean')]:.4f} ± {type_stats.loc[alg_type, ('time', 'std')]:.4f}"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Algorithm Type', 'Avg Approximation Ratio', 'Avg Time (s)'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Performance Summary by Algorithm Type')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, df: pd.DataFrame, filename: str):
        """Save benchmark results to file."""
        # Convert to JSON-serializable format
        df_copy = df.copy()
        df_copy['tour'] = df_copy['tour'].apply(lambda x: x.tolist() if x is not None else None)
        
        # Save as CSV
        csv_filename = filename.replace('.json', '.csv')
        df_copy.drop('tour', axis=1).to_csv(csv_filename, index=False)
        
        # Save as JSON with tours
        json_data = df_copy.to_dict('records')
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Results saved to {csv_filename} and {filename}")


def main():
    """Run comprehensive benchmarks."""
    benchmark = TSPBenchmark()
    
    print("Starting TSP Algorithm Benchmarking Suite")
    print("=" * 50)
    
    # Configuration
    small_sizes = [4, 5, 6, 7, 8]
    medium_sizes = [10, 12, 15, 20]
    large_sizes = [25, 30, 40, 50]
    
    # Benchmark 1: Small instances with all algorithms
    print("\n1. Small Instance Benchmark (all algorithms)")
    small_results = benchmark.run_size_scaling_benchmark(
        sizes=small_sizes,
        instance_type='euclidean',
        num_instances=5,
        timeout=30.0
    )
    
    # Benchmark 2: Medium instances with fast algorithms
    print("\n2. Medium Instance Benchmark (fast algorithms only)")
    fast_algorithms = ['mst_2approx', 'nearest_neighbor', 'multi_start_nn', 'nn_2opt', 'multi_start_nn_2opt']
    medium_results = benchmark.run_size_scaling_benchmark(
        sizes=medium_sizes,
        instance_type='euclidean',
        algorithms=fast_algorithms,
        num_instances=3,
        timeout=60.0
    )
    
    # Benchmark 3: Large instances with heuristics only
    print("\n3. Large Instance Benchmark (heuristics only)")
    heuristic_algorithms = ['nearest_neighbor', 'multi_start_nn', 'nn_2opt', 'multi_start_nn_2opt', 'random_restart_2opt']
    large_results = benchmark.run_size_scaling_benchmark(
        sizes=large_sizes,
        instance_type='euclidean',
        algorithms=heuristic_algorithms,
        num_instances=3,
        timeout=120.0
    )
    
    # Combine results
    all_results = pd.concat([small_results, medium_results, large_results], ignore_index=True)
    
    # Analysis
    print("\n4. Analyzing Results...")
    approximation_analysis = benchmark.analyze_approximation_ratios(all_results)
    
    # Generate plots
    print("\n5. Generating Plots...")
    benchmark.plot_scaling_results(all_results, 'scaling_results.png')
    benchmark.plot_approximation_analysis(approximation_analysis, 'approximation_analysis.png')
    
    # Save results
    print("\n6. Saving Results...")
    benchmark.save_results(all_results, 'benchmark_results.json')
    benchmark.save_results(approximation_analysis, 'approximation_analysis.json')
    
    # Summary statistics
    print("\n7. Summary Statistics")
    print("=" * 30)
    
    successful_results = all_results[all_results['status'] == 'success']
    
    print(f"Total runs: {len(all_results)}")
    print(f"Successful runs: {len(successful_results)}")
    print(f"Success rate: {len(successful_results) / len(all_results):.2%}")
    
    print("\nAverage approximation ratios:")
    avg_ratios = approximation_analysis.groupby('algorithm_name')['approximation_ratio'].mean().sort_values()
    for alg, ratio in avg_ratios.items():
        print(f"  {alg:25s}: {ratio:.3f}")
    
    print("\nAverage execution times:")
    avg_times = successful_results.groupby('algorithm_name')['time'].mean().sort_values()
    for alg, time in avg_times.items():
        print(f"  {alg:25s}: {time:.4f}s")


if __name__ == '__main__':
    main()
