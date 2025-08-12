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
