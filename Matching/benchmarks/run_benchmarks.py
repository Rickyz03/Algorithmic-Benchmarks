"""
Benchmarking script for bipartite graph matching algorithms.

This script runs comprehensive performance tests on both Hungarian and 
Hopcroft-Karp algorithms, measuring execution time, scalability, and
efficiency across different graph types and sizes.

Author: Your Name
Date: 2025
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import sys
import os
import json
from datetime import datetime

# Add src and utils directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from hungarian import HungarianAlgorithm, solve_maximum_weight_matching
from hopcroft_karp import HopcroftKarpAlgorithm, create_bipartite_graph_from_edges
from graph_generator import BipartiteGraphGenerator


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite for matching algorithms."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the benchmark suite.
        
        Args:
            seed: Random seed for reproducible results
        """
        self.generator = BipartiteGraphGenerator(seed=seed)
        self.results = {}
        self.benchmark_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def time_algorithm(self, algorithm_func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Time the execution of an algorithm function.
        
        Args:
            algorithm_func: Function to time
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Tuple of (result, execution_time_seconds)
        """
        start_time = time.perf_counter()
        result = algorithm_func(*args, **kwargs)
        end_time = time.perf_counter()
        
        return result, end_time - start_time
    
    def benchmark_scalability(self, sizes: List[int], densities: List[float] = None) -> Dict:
        """
        Benchmark algorithm scalability across different graph sizes.
        
        Args:
            sizes: List of graph sizes (n for nxn graphs)
            densities: List of edge densities to test (default: [0.3, 0.5, 0.7])
            
        Returns:
            Dictionary with benchmark results
        """
        if densities is None:
            densities = [0.3, 0.5, 0.7]
        
        results = {
            'sizes': sizes,
            'densities': densities,
            'hopcroft_karp_times': {density: [] for density in densities},
            'hungarian_times': {density: [] for density in densities},
            'hopcroft_karp_matching_sizes': {density: [] for density in densities},
            'hungarian_matching_weights': {density: [] for density in densities}
        }
        
        print("Running Scalability Benchmark...")
        print(f"Graph sizes: {sizes}")
        print(f"Densities: {densities}")
        print("-" * 50)
        
        for size in sizes:
            print(f"\nTesting size {size}x{size}:")
            
            for density in densities:
                print(f"  Density {density:.1f}:", end=" ")
                
                # Generate test graph
                edges = self.generator.generate_random_bipartite_graph(size, size, density)
                weights = self.generator.generate_random_weights(edges, 1, 10)
                weight_matrix = self.generator.edges_to_weight_matrix(edges, weights, size, size)
                
                # Benchmark Hopcroft-Karp
                try:
                    hk_alg = create_bipartite_graph_from_edges(edges, size, size)
                    (hk_matching, hk_size), hk_time = self.time_algorithm(hk_alg.solve)
                    
                    results['hopcroft_karp_times'][density].append(hk_time)
                    results['hopcroft_karp_matching_sizes'][density].append(hk_size)
                    
                    print(f"HK: {hk_time:.4f}s ({hk_size} matches)", end=", ")
                    
                except Exception as e:
                    print(f"HK: ERROR ({e})", end=", ")
                    results['hopcroft_karp_times'][density].append(float('inf'))
                    results['hopcroft_karp_matching_sizes'][density].append(0)
                
                # Benchmark Hungarian
                try:
                    (hungarian_matching, hungarian_weight), hungarian_time = self.time_algorithm(
                        solve_maximum_weight_matching, weight_matrix
                    )
                    
                    results['hungarian_times'][density].append(hungarian_time)
                    results['hungarian_matching_weights'][density].append(hungarian_weight)
                    
                    print(f"H: {hungarian_time:.4f}s (weight: {hungarian_weight:.1f})")
                    
                except Exception as e:
                    print(f"H: ERROR ({e})")
                    results['hungarian_times'][density].append(float('inf'))
                    results['hungarian_matching_weights'][density].append(0)
        
        return results
    
    def benchmark_density_impact(self, size: int = 20, densities: List[float] = None) -> Dict:
        """
        Benchmark the impact of graph density on algorithm performance.
        
        Args:
            size: Fixed graph size (nxn)
            densities: List of densities to test
            
        Returns:
            Dictionary with density benchmark results
        """
        if densities is None:
            densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = {
            'size': size,
            'densities': densities,
            'hopcroft_karp_times': [],
            'hungarian_times': [],
            'edge_counts': [],
            'hopcroft_karp_matching_sizes': [],
            'hungarian_matching_weights': []
        }
        
        print(f"\nRunning Density Impact Benchmark (size: {size}x{size})...")
        print("-" * 50)
        
        for density in densities:
            print(f"Density {density:.1f}:", end=" ")
            
            # Generate test graph
            edges = self.generator.generate_random_bipartite_graph(size, size, density)
            weights = self.generator.generate_random_weights(edges, 1, 10)
            weight_matrix = self.generator.edges_to_weight_matrix(edges, weights, size, size)
            
            results['edge_counts'].append(len(edges))
            
            # Benchmark Hopcroft-Karp
            hk_alg = create_bipartite_graph_from_edges(edges, size, size)
            (hk_matching, hk_size), hk_time = self.time_algorithm(hk_alg.solve)
            
            results['hopcroft_karp_times'].append(hk_time)
            results['hopcroft_karp_matching_sizes'].append(hk_size)
            
            # Benchmark Hungarian
            (hungarian_matching, hungarian_weight), hungarian_time = self.time_algorithm(
                solve_maximum_weight_matching, weight_matrix
            )
            
            results['hungarian_times'].append(hungarian_time)
            results['hungarian_matching_weights'].append(hungarian_weight)
            
            print(f"HK: {hk_time:.4f}s, H: {hungarian_time:.4f}s, "
                  f"Edges: {len(edges)}, HK_matches: {hk_size}")
        
        return results
    
    def benchmark_graph_types(self, size: int = 15) -> Dict:
        """
        Benchmark different graph types (sparse, dense, regular, etc.).
        
        Args:
            size: Graph size for testing
            
        Returns:
            Dictionary with graph type benchmark results
        """
        results = {
            'size': size,
            'graph_types': [],
            'hopcroft_karp_times': [],
            'hungarian_times': [],
            'edge_counts': [],
            'descriptions': []
        }
        
        print(f"\nRunning Graph Type Benchmark (size: {size}x{size})...")
        print("-" * 50)
        
        # Test different graph types
        test_configs = [
            {
                'name': 'Sparse (degree 1-2)',
                'type': 'sparse',
                'min_degree': 1,
                'max_degree': 2
            },
            {
                'name': 'Medium Sparse (degree 2-3)',
                'type': 'sparse',
                'min_degree': 2,
                'max_degree': 3
            },
            {
                'name': 'Regular (degree 3)',
                'type': 'regular',
                'degree': min(3, size-1)
            },
            {
                'name': 'Dense Random (0.7)',
                'type': 'random',
                'density': 0.7
            },
            {
                'name': 'Very Dense (0.9)',
                'type': 'random',
                'density': 0.9
            },
            {
                'name': 'Complete',
                'type': 'complete'
            }
        ]
        
        for config in test_configs:
            print(f"Testing {config['name']}:", end=" ")
            
            # Generate graph based on type
            if config['type'] == 'sparse':
                edges = self.generator.generate_sparse_bipartite_graph(
                    size, size, config['min_degree'], config['max_degree']
                )
            elif config['type'] == 'regular':
                edges = self.generator.generate_regular_bipartite_graph(
                    size, size, config['degree']
                )
            elif config['type'] == 'random':
                edges = self.generator.generate_random_bipartite_graph(
                    size, size, config['density']
                )
            elif config['type'] == 'complete':
                edges = self.generator.generate_complete_bipartite_graph(size, size)
            
            weights = self.generator.generate_random_weights(edges, 1, 10)
            weight_matrix = self.generator.edges_to_weight_matrix(edges, weights, size, size)
            
            # Benchmark both algorithms
            hk_alg = create_bipartite_graph_from_edges(edges, size, size)
            (hk_matching, hk_size), hk_time = self.time_algorithm(hk_alg.solve)
            
            (hungarian_matching, hungarian_weight), hungarian_time = self.time_algorithm(
                solve_maximum_weight_matching, weight_matrix
            )
            
            # Store results
            results['graph_types'].append(config['name'])
            results['hopcroft_karp_times'].append(hk_time)
            results['hungarian_times'].append(hungarian_time)
            results['edge_counts'].append(len(edges))
            results['descriptions'].append(f"{len(edges)} edges, HK: {hk_size} matches")
            
            print(f"HK: {hk_time:.4f}s, H: {hungarian_time:.4f}s, "
                  f"Edges: {len(edges)}")
        
        return results
    
    def create_performance_plots(self, results: Dict, output_dir: str = "."):
        """
        Create performance visualization plots from benchmark results.
        
        Args:
            results: Dictionary containing benchmark results
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Scalability plot
        if 'scalability' in results:
            self._plot_scalability(results['scalability'], output_dir)
        
        # Density impact plot
        if 'density_impact' in results:
            self._plot_density_impact(results['density_impact'], output_dir)
        
        # Graph type comparison plot
        if 'graph_types' in results:
            self._plot_graph_types(results['graph_types'], output_dir)
    
    def _plot_scalability(self, data: Dict, output_dir: str):
        """Create scalability plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Scalability Analysis', fontsize=16, fontweight='bold')
        
        sizes = data['sizes']
        
        # Time comparison for different densities
        for density in data['densities']:
            hk_times = data['hopcroft_karp_times'][density]
            h_times = data['hungarian_times'][density]
            
            ax1.plot(sizes, hk_times, 'o-', label=f'HK (ρ={density})', alpha=0.7)
            ax2.plot(sizes, h_times, 's-', label=f'Hungarian (ρ={density})', alpha=0.7)
        
        ax1.set_xlabel('Graph Size (n×n)')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Hopcroft-Karp Scalability')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Graph Size (n×n)')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Hungarian Algorithm Scalability')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Direct time comparison (same density)
        main_density = data['densities'][1] if len(data['densities']) > 1 else data['densities'][0]
        hk_times = data['hopcroft_karp_times'][main_density]
        h_times = data['hungarian_times'][main_density]
        
        ax3.plot(sizes, hk_times, 'bo-', label='Hopcroft-Karp', linewidth=2, markersize=6)
        ax3.plot(sizes, h_times, 'rs-', label='Hungarian', linewidth=2, markersize=6)
        ax3.set_xlabel('Graph Size (n×n)')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title(f'Direct Time Comparison (ρ={main_density})')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Matching quality comparison
        hk_sizes = data['hopcroft_karp_matching_sizes'][main_density]
        h_weights = data['hungarian_matching_weights'][main_density]
        
        ax4_twin = ax4.twinx()
        ax4.plot(sizes, hk_sizes, 'bo-', label='HK Matching Size', linewidth=2)
        ax4_twin.plot(sizes, h_weights, 'rs-', label='Hungarian Weight', linewidth=2, color='red')
        
        ax4.set_xlabel('Graph Size (n×n)')
        ax4.set_ylabel('Matching Size', color='blue')
        ax4_twin.set_ylabel('Total Weight', color='red')
        ax4.set_title(f'Matching Quality Comparison (ρ={main_density})')
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/scalability_analysis_{self.benchmark_timestamp}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_density_impact(self, data: Dict, output_dir: str):
        """Create density impact plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Density Impact Analysis (Graph Size: {data["size"]}×{data["size"]})', 
                     fontsize=16, fontweight='bold')
        
        densities = data['densities']
        
        # Execution time vs density
        ax1.plot(densities, data['hopcroft_karp_times'], 'bo-', 
                label='Hopcroft-Karp', linewidth=2, markersize=6)
        ax1.plot(densities, data['hungarian_times'], 'rs-', 
                label='Hungarian', linewidth=2, markersize=6)
        ax1.set_xlabel('Graph Density')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time vs Graph Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Edge count vs density (verification)
        ax2.plot(densities, data['edge_counts'], 'go-', linewidth=2, markersize=6)
        ax2.set_xlabel('Graph Density')
        ax2.set_ylabel('Number of Edges')
        ax2.set_title('Edge Count vs Density')
        ax2.grid(True, alpha=0.3)
        
        # Matching quality vs density
        ax3.plot(densities, data['hopcroft_karp_matching_sizes'], 'bo-', 
                label='HK Matching Size', linewidth=2, markersize=6)
        ax3.set_xlabel('Graph Density')
        ax3.set_ylabel('Matching Size')
        ax3.set_title('Matching Size vs Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Speed ratio
        speed_ratios = [h_time / hk_time if hk_time > 0 else 0 
                       for hk_time, h_time in zip(data['hopcroft_karp_times'], data['hungarian_times'])]
        ax4.plot(densities, speed_ratios, 'mo-', linewidth=2, markersize=6)
        ax4.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Equal Performance')
        ax4.set_xlabel('Graph Density')
        ax4.set_ylabel('Time Ratio (Hungarian/Hopcroft-Karp)')
        ax4.set_title('Relative Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/density_impact_{self.benchmark_timestamp}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_graph_types(self, data: Dict, output_dir: str):
        """Create graph type comparison plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Graph Type Performance Comparison (Size: {data["size"]}×{data["size"]})', 
                     fontsize=16, fontweight='bold')
        
        graph_types = data['graph_types']
        y_pos = np.arange(len(graph_types))
        
        # Execution time comparison
        ax1.barh(y_pos - 0.2, data['hopcroft_karp_times'], 0.4, 
                label='Hopcroft-Karp', alpha=0.7)
        ax1.barh(y_pos + 0.2, data['hungarian_times'], 0.4, 
                label='Hungarian', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(graph_types)
        ax1.set_xlabel('Execution Time (seconds)')
        ax1.set_title('Execution Time by Graph Type')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Edge count by graph type
        ax2.bar(range(len(graph_types)), data['edge_counts'], alpha=0.7, color='green')
        ax2.set_xticks(range(len(graph_types)))
        ax2.set_xticklabels(graph_types, rotation=45, ha='right')
        ax2.set_ylabel('Number of Edges')
        ax2.set_title('Edge Count by Graph Type')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/graph_types_{self.benchmark_timestamp}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_benchmark_suite(self) -> Dict:
        """
        Run the complete benchmark suite.
        
        Returns:
            Complete results dictionary
        """
        print("=" * 60)
        print("COMPREHENSIVE MATCHING ALGORITHMS BENCHMARK")
        print("=" * 60)
        
        all_results = {}
        
        # Scalability benchmark
        scalability_sizes = [5, 10, 15, 20, 25, 30]
        scalability_densities = [0.3, 0.5, 0.7]
        all_results['scalability'] = self.benchmark_scalability(scalability_sizes, scalability_densities)
        
        # Density impact benchmark
        all_results['density_impact'] = self.benchmark_density_impact(size=20)
        
        # Graph types benchmark
        all_results['graph_types'] = self.benchmark_graph_types(size=15)
        
        # Save results to file
        self.save_results(all_results)
        
        # Create visualization plots
        self.create_performance_plots(all_results, "benchmark_results")
        
        return all_results
    
    def save_results(self, results: Dict, filename: str = None):
        """
        Save benchmark results to JSON file.
        
        Args:
            results: Results dictionary to save
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            filename = f"benchmark_results_{self.benchmark_timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-JSON types for serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


if __name__ == "__main__":
    # Run comprehensive benchmark
    benchmark = PerformanceBenchmark(seed=42)
    
    try:
        results = benchmark.run_full_benchmark_suite()
        
        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Generated files:")
        print(f"- benchmark_results_{benchmark.benchmark_timestamp}.json")
        print(f"- benchmark_results/scalability_analysis_{benchmark.benchmark_timestamp}.png")
        print(f"- benchmark_results/density_impact_{benchmark.benchmark_timestamp}.png")
        print(f"- benchmark_results/graph_types_{benchmark.benchmark_timestamp}.png")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
