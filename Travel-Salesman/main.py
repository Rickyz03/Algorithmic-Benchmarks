"""
Main execution script for TSP algorithm comparison and visualization.

This script provides an interactive interface for running different TSP algorithms,
comparing their performance, and visualizing the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import os
from typing import List, Dict, Tuple, Optional

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.graph_generator import generate_euclidean_tsp_instance, generate_metric_tsp_instance
from algorithms.mst_approximation import mst_2_approximation
from algorithms.exact_algorithms import brute_force_tsp, held_karp_tsp, BranchAndBound
from algorithms.heuristic_algorithms import (
    nearest_neighbor_tsp, multi_start_nearest_neighbor,
    nearest_neighbor_with_2opt, multi_start_nn_with_2opt,
    random_restart_2opt
)


def calculate_tour_length(tour: List[int], distance_matrix: np.ndarray) -> float:
    """Calculate total length of a tour."""
    if len(tour) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(tour)):
        current = tour[i]
        next_city = tour[(i + 1) % len(tour)]
        total_length += distance_matrix[current][next_city]
    
    return total_length


def visualize_tour(coordinates: List[Tuple[float, float]], tour: List[int], 
                   title: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Visualize a TSP tour on a 2D plot.
    
    Args:
        coordinates: List of (x, y) coordinates for each city
        tour: Tour as list of city indices
        title: Title for the plot
        ax: Matplotlib axes to plot on (creates new if None)
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Extract coordinates
    x_coords = [coordinates[i][0] for i in range(len(coordinates))]
    y_coords = [coordinates[i][1] for i in range(len(coordinates))]
    
    # Plot cities
    ax.scatter(x_coords, y_coords, c='red', s=100, zorder=3)
    
    # Label cities
    for i, (x, y) in enumerate(coordinates):
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    # Plot tour edges
    for i in range(len(tour)):
        start_city = tour[i]
        end_city = tour[(i + 1) % len(tour)]
        
        start_coord = coordinates[start_city]
        end_coord = coordinates[end_city]
        
        ax.plot([start_coord[0], end_coord[0]], [start_coord[1], end_coord[1]], 
               'b-', linewidth=2, alpha=0.7, zorder=2)
    
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    return ax


def run_algorithm_comparison(size: int, instance_type: str = 'euclidean', 
                           seed: int = 42, visualize: bool = True) -> Dict:
    """
    Run and compare multiple TSP algorithms on a single instance.
    
    Args:
        size: Number of cities
        instance_type: Type of instance ('euclidean' or 'metric')
        seed: Random seed for reproducibility
        visualize: Whether to create visualizations
        
    Returns:
        Dictionary with algorithm results
    """
    print(f"Generating {instance_type} TSP instance with {size} cities (seed={seed})")
    
    # Generate instance
    if instance_type == 'euclidean':
        distance_matrix, coordinates = generate_euclidean_tsp_instance(size, seed)
    else:
        distance_matrix = generate_metric_tsp_instance(size, seed)
        coordinates = None
    
    # Define algorithms to test
    algorithms = []
    
    # Exact algorithms (only for small instances)
    if size <= 8:
        algorithms.append(('Brute Force', brute_force_tsp))
    if size <= 15:
        algorithms.append(('Held-Karp DP', held_karp_tsp))
    if size <= 12:
        algorithms.append(('Branch & Bound', lambda dm: BranchAndBound(dm).solve()))
    
    # Approximation algorithm
    algorithms.append(('MST 2-Approximation', mst_2_approximation))
    
    # Heuristic algorithms
    algorithms.append(('Nearest Neighbor', lambda dm: nearest_neighbor_tsp(dm, 0)))
    algorithms.append(('Multi-start NN', multi_start_nearest_neighbor))
    algorithms.append(('NN + 2-opt', lambda dm: nearest_neighbor_with_2opt(dm, 0)))
    algorithms.append(('Multi-start NN + 2-opt', multi_start_nn_with_2opt))
    algorithms.append(('Random Restart + 2-opt', 
                      lambda dm: random_restart_2opt(dm, num_restarts=10, seed=seed)))
    
    results = {}
    
    print(f"\nRunning algorithms on {size}-city instance:")
    print("-" * 50)
    
    for name, algorithm in algorithms:
        try:
            import time
            start_time = time.time()
            tour, length = algorithm(distance_matrix)
            end_time = time.time()
            
            # Verify tour length calculation
            calculated_length = calculate_tour_length(tour, distance_matrix)
            
            results[name] = {
                'tour': tour,
                'length': length,
                'calculated_length': calculated_length,
                'time': end_time - start_time,
                'status': 'success'
            }
            
            print(f"{name:25s}: {length:8.2f} ({end_time - start_time:6.3f}s)")
            
        except Exception as e:
            results[name] = {
                'tour': None,
                'length': None,
                'calculated_length': None,
                'time': None,
                'status': f'error: {str(e)}'
            }
            print(f"{name:25s}: ERROR - {str(e)}")
    
    # Analysis
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}
    
    if successful_results:
        best_length = min(result['length'] for result in successful_results.values())
        
        print(f"\nResults Summary:")
        print("-" * 50)
        print(f"Best solution length: {best_length:.2f}")
        
        print(f"\nApproximation ratios:")
        for name, result in successful_results.items():
            ratio = result['length'] / best_length
            print(f"{name:25s}: {ratio:6.3f}")
    
    # Visualization
    if visualize and coordinates is not None and successful_results:
        # Select interesting algorithms to visualize
        algorithms_to_plot = []
        
        # Add best exact algorithm if available
        exact_algorithms = ['Brute Force', 'Held-Karp DP', 'Branch & Bound']
        for alg in exact_algorithms:
            if alg in successful_results:
                algorithms_to_plot.append(alg)
                break
        
        # Add approximation algorithm
        if 'MST 2-Approximation' in successful_results:
            algorithms_to_plot.append('MST 2-Approximation')
        
        # Add best heuristic
        heuristic_algorithms = ['Multi-start NN + 2-opt', 'NN + 2-opt', 'Multi-start NN', 'Nearest Neighbor']
        for alg in heuristic_algorithms:
            if alg in successful_results:
                algorithms_to_plot.append(alg)
                break
        
        # Create visualization
        n_plots = len(algorithms_to_plot)
        if n_plots > 0:
            fig, axes = plt.subplots(1, min(n_plots, 3), figsize=(5 * min(n_plots, 3), 5))
            if n_plots == 1:
                axes = [axes]
            elif n_plots == 2:
                axes = axes
            else:
                axes = axes[:3]  # Limit to 3 plots
            
            for i, alg_name in enumerate(algorithms_to_plot[:3]):
                result = successful_results[alg_name]
                tour = result['tour']
                length = result['length']
                time_taken = result['time']
                
                title = f"{alg_name}\nLength: {length:.2f}, Time: {time_taken:.3f}s"
                visualize_tour(coordinates, tour, title, axes[i])
            
            plt.tight_layout()
            plt.savefig(f'tsp_comparison_{size}_cities.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    return results


def run_scaling_experiment(sizes: List[int], instance_type: str = 'euclidean', 
                          num_instances: int = 3, timeout: float = 30.0) -> pd.DataFrame:
    """
    Run scaling experiment across different problem sizes.
    
    Args:
        sizes: List of problem sizes to test
        instance_type: Type of instances ('euclidean' or 'metric')
        num_instances: Number of instances per size
        timeout: Timeout per algorithm run in seconds
        
    Returns:
        DataFrame with scaling results
    """
    print(f"Running scaling experiment...")
    print(f"Sizes: {sizes}")
    print(f"Instances per size: {num_instances}")
    print(f"Instance type: {instance_type}")
    
    results = []
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Testing size: {size}")
        print(f"{'='*60}")
        
        for instance_idx in range(num_instances):
            print(f"\nInstance {instance_idx + 1}/{num_instances}")
            
            # Generate instance
            seed = 42 + instance_idx
            if instance_type == 'euclidean':
                distance_matrix, coordinates = generate_euclidean_tsp_instance(size, seed)
            else:
                distance_matrix = generate_metric_tsp_instance(size, seed)
                coordinates = None
            
            # Define algorithms based on size
            test_algorithms = []
            
            if size <= 8:
                test_algorithms.append(('Brute Force', brute_force_tsp))
            if size <= 15:
                test_algorithms.append(('Held-Karp DP', held_karp_tsp))
            if size <= 12:
                test_algorithms.append(('Branch & Bound', lambda dm: BranchAndBound(dm).solve()))
            
            # Always test these
            test_algorithms.extend([
                ('MST 2-Approximation', mst_2_approximation),
                ('Nearest Neighbor', lambda dm: nearest_neighbor_tsp(dm, 0)),
                ('Multi-start NN', multi_start_nearest_neighbor),
                ('NN + 2-opt', lambda dm: nearest_neighbor_with_2opt(dm, 0)),
                ('Multi-start NN + 2-opt', multi_start_nn_with_2opt)
            ])
            
            # Test each algorithm
            for alg_name, algorithm in test_algorithms:
                try:
                    import time
                    start_time = time.time()
                    tour, length = algorithm(distance_matrix)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    if elapsed_time > timeout:
                        status = 'timeout'
                        tour = None
                        length = None
                    else:
                        status = 'success'
                    
                    results.append({
                        'algorithm': alg_name,
                        'size': size,
                        'instance_idx': instance_idx,
                        'seed': seed,
                        'tour_length': length,
                        'execution_time': elapsed_time,
                        'status': status,
                        'instance_type': instance_type
                    })
                    
                    print(f"  {alg_name:20s}: {length:8.2f} ({elapsed_time:.3f}s)")
                    
                except Exception as e:
                    results.append({
                        'algorithm': alg_name,
                        'size': size,
                        'instance_idx': instance_idx,
                        'seed': seed,
                        'tour_length': None,
                        'execution_time': None,
                        'status': f'error_{str(e)[:20]}',
                        'instance_type': instance_type
                    })
                    print(f"  {alg_name:20s}: ERROR - {str(e)}")
    
    return pd.DataFrame(results)


def plot_scaling_results(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot scaling experiment results."""
    successful_df = df[df['status'] == 'success'].copy()
    
    if len(successful_df) == 0:
        print("No successful results to plot.")
        return
    
    # Group by algorithm and size, calculate means
    grouped = successful_df.groupby(['algorithm', 'size']).agg({
        'execution_time': 'mean',
        'tour_length': 'mean'
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Execution time vs size
    for algorithm in grouped['algorithm'].unique():
        alg_data = grouped[grouped['algorithm'] == algorithm]
        ax1.plot(alg_data['size'], alg_data['execution_time'], 
                marker='o', label=algorithm, linewidth=2)
    
    ax1.set_xlabel('Problem Size (number of cities)')
    ax1.set_ylabel('Average Execution Time (seconds)')
    ax1.set_title('Algorithm Scalability: Execution Time vs Problem Size')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Solution quality vs size
    for algorithm in grouped['algorithm'].unique():
        alg_data = grouped[grouped['algorithm'] == algorithm]
        ax2.plot(alg_data['size'], alg_data['tour_length'], 
                marker='o', label=algorithm, linewidth=2)
    
    ax2.set_xlabel('Problem Size (number of cities)')
    ax2.set_ylabel('Average Tour Length')
    ax2.set_title('Algorithm Scalability: Solution Quality vs Problem Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def interactive_demo():
    """Run interactive demo allowing user to test different configurations."""
    print("\n" + "="*60)
    print("TSP ALGORITHM COMPARISON - INTERACTIVE DEMO")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Compare algorithms on single instance")
        print("2. Run scaling experiment")
        print("3. Test with custom coordinates")
        print("4. Exit")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                # Single instance comparison
                size = int(input("Enter number of cities (4-50): "))
                if size < 4 or size > 50:
                    print("Size must be between 4 and 50")
                    continue
                
                instance_type = input("Instance type (euclidean/metric) [euclidean]: ").strip()
                if not instance_type:
                    instance_type = 'euclidean'
                
                seed = input("Random seed [42]: ").strip()
                seed = int(seed) if seed else 42
                
                visualize = input("Create visualizations? (y/n) [y]: ").strip()
                visualize = visualize.lower() != 'n'
                
                run_algorithm_comparison(size, instance_type, seed, visualize)
            
            elif choice == '2':
                # Scaling experiment
                sizes_input = input("Enter sizes (comma-separated) [5,6,7,8,10]: ").strip()
                if not sizes_input:
                    sizes = [5, 6, 7, 8, 10]
                else:
                    sizes = [int(x.strip()) for x in sizes_input.split(',')]
                
                instance_type = input("Instance type (euclidean/metric) [euclidean]: ").strip()
                if not instance_type:
                    instance_type = 'euclidean'
                
                num_instances = input("Instances per size [3]: ").strip()
                num_instances = int(num_instances) if num_instances else 3
                
                print(f"\nRunning scaling experiment...")
                results_df = run_scaling_experiment(sizes, instance_type, num_instances)
                
                # Save results
                results_df.to_csv('scaling_results.csv', index=False)
                print(f"Results saved to scaling_results.csv")
                
                # Plot results
                plot_scaling_results(results_df, 'scaling_plot.png')
                print(f"Plot saved to scaling_plot.png")
            
            elif choice == '3':
                # Custom coordinates
                print("Enter city coordinates (x,y). Empty line to finish:")
                coordinates = []
                i = 0
                
                while True:
                    coord_input = input(f"City {i} (x,y): ").strip()
                    if not coord_input:
                        break
                    
                    try:
                        x, y = map(float, coord_input.split(','))
                        coordinates.append((x, y))
                        i += 1
                    except ValueError:
                        print("Invalid format. Use: x,y")
                
                if len(coordinates) < 3:
                    print("Need at least 3 cities")
                    continue
                
                # Create distance matrix from coordinates
                from utils.graph_generator import coordinates_to_distance_matrix
                distance_matrix = coordinates_to_distance_matrix(coordinates)
                
                # Run comparison with custom coordinates
                size = len(coordinates)
                results = {}
                
                algorithms = [
                    ('MST 2-Approximation', mst_2_approximation),
                    ('Nearest Neighbor', lambda dm: nearest_neighbor_tsp(dm, 0)),
                    ('NN + 2-opt', lambda dm: nearest_neighbor_with_2opt(dm, 0)),
                    ('Multi-start NN + 2-opt', multi_start_nn_with_2opt)
                ]
                
                if size <= 8:
                    algorithms.insert(0, ('Brute Force', brute_force_tsp))
                if size <= 15:
                    algorithms.insert(0, ('Held-Karp DP', held_karp_tsp))
                
                print(f"\nRunning algorithms on {size} custom cities:")
                print("-" * 40)
                
                for name, algorithm in algorithms:
                    try:
                        import time
                        start_time = time.time()
                        tour, length = algorithm(distance_matrix)
                        end_time = time.time()
                        
                        print(f"{name:20s}: {length:8.2f} ({end_time - start_time:.3f}s)")
                        results[name] = (tour, length)
                        
                    except Exception as e:
                        print(f"{name:20s}: ERROR - {str(e)}")
                
                # Visualize best result
                if results:
                    best_alg = min(results.keys(), key=lambda k: results[k][1])
                    best_tour, best_length = results[best_alg]
                    
                    print(f"\nBest result: {best_alg} with length {best_length:.2f}")
                    
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                    title = f"Custom Instance - {best_alg}\nLength: {best_length:.2f}"
                    visualize_tour(coordinates, best_tour, title, ax)
                    plt.show()
            
            elif choice == '4':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please select 1-4.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TSP Algorithm Comparison Tool')
    
    parser.add_argument('--mode', choices=['compare', 'scale', 'interactive'], 
                       default='interactive',
                       help='Execution mode (default: interactive)')
    
    parser.add_argument('--size', type=int, default=8,
                       help='Number of cities for comparison mode (default: 8)')
    
    parser.add_argument('--sizes', type=str, default='5,6,7,8',
                       help='Comma-separated sizes for scaling mode (default: 5,6,7,8)')
    
    parser.add_argument('--type', choices=['euclidean', 'metric'], default='euclidean',
                       help='Instance type (default: euclidean)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    parser.add_argument('--instances', type=int, default=3,
                       help='Number of instances per size for scaling mode (default: 3)')
    
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualizations')
    
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Timeout for each algorithm run in seconds (default: 30.0)')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("TSP Algorithm Comparison Tool")
    print("="*50)
    
    if args.mode == 'compare':
        print(f"Mode: Single instance comparison")
        run_algorithm_comparison(
            size=args.size,
            instance_type=args.type,
            seed=args.seed,
            visualize=not args.no_viz
        )
    
    elif args.mode == 'scale':
        print(f"Mode: Scaling experiment")
        sizes = [int(x.strip()) for x in args.sizes.split(',')]
        
        results_df = run_scaling_experiment(
            sizes=sizes,
            instance_type=args.type,
            num_instances=args.instances,
            timeout=args.timeout
        )
        
        # Save and plot results
        results_df.to_csv('scaling_results.csv', index=False)
        print(f"Results saved to scaling_results.csv")
        
        if not args.no_viz:
            plot_scaling_results(results_df, 'scaling_plot.png')
            print(f"Plot saved to scaling_plot.png")
        
        # Print summary statistics
        successful_df = results_df[results_df['status'] == 'success']
        if len(successful_df) > 0:
            print(f"\nSummary Statistics:")
            print(f"Total runs: {len(results_df)}")
            print(f"Successful runs: {len(successful_df)}")
            print(f"Success rate: {len(successful_df) / len(results_df):.1%}")
            
            print(f"\nAverage execution times by algorithm:")
            time_stats = successful_df.groupby('algorithm')['execution_time'].mean().sort_values()
            for alg, avg_time in time_stats.items():
                print(f"  {alg:25s}: {avg_time:.4f}s")
    
    elif args.mode == 'interactive':
        interactive_demo()


if __name__ == '__main__':
    main()
