"""
Main script for comparing Hungarian and Hopcroft-Karp algorithms.

This script demonstrates both algorithms on sample data, compares their
performance, and generates visualizations of the matchings found.

Author: Your Name
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any
import time
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from hungarian import HungarianAlgorithm, solve_maximum_weight_matching
from hopcroft_karp import HopcroftKarpAlgorithm, create_bipartite_graph_from_edges
from graph_generator import BipartiteGraphGenerator


class MatchingVisualizer:
    """Class for visualizing bipartite graph matchings."""
    
    @staticmethod
    def visualize_matching(matching_pairs: List[Tuple[int, int]], 
                          left_size: int, 
                          right_size: int,
                          all_edges: List[Tuple[int, int]] = None,
                          title: str = "Bipartite Graph Matching",
                          weights: Dict[Tuple[int, int], float] = None) -> plt.Figure:
        """
        Visualize a bipartite graph matching.
        
        Args:
            matching_pairs: List of matched (left, right) vertex pairs
            left_size: Number of vertices in left partition
            right_size: Number of vertices in right partition  
            all_edges: All edges in the graph (for visualization)
            title: Title for the plot
            weights: Edge weights dictionary (optional)
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Position vertices
        left_positions = [(0, i * 2) for i in range(left_size)]
        right_positions = [(4, i * 2) for i in range(right_size)]
        
        # Draw all edges in light gray
        if all_edges:
            for left_v, right_v in all_edges:
                if left_v < len(left_positions) and right_v < len(right_positions):
                    x1, y1 = left_positions[left_v]
                    x2, y2 = right_positions[right_v]
                    ax.plot([x1, x2], [y1, y2], 'lightgray', alpha=0.3, linewidth=1)
                    
                    # Add weight labels if provided
                    if weights and (left_v, right_v) in weights:
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        ax.text(mid_x, mid_y + 0.1, f'{weights[(left_v, right_v)]:.1f}', 
                               ha='center', va='bottom', fontsize=8, alpha=0.7)
        
        # Draw matching edges in red
        total_weight = 0
        for left_v, right_v in matching_pairs:
            if left_v < len(left_positions) and right_v < len(right_positions):
                x1, y1 = left_positions[left_v]
                x2, y2 = right_positions[right_v]
                ax.plot([x1, x2], [y1, y2], 'red', linewidth=3, alpha=0.8)
                
                if weights and (left_v, right_v) in weights:
                    total_weight += weights[(left_v, right_v)]
        
        # Draw vertices
        for i, (x, y) in enumerate(left_positions):
            color = 'lightblue' if any(left_v == i for left_v, _ in matching_pairs) else 'lightgray'
            circle = patches.Circle((x, y), 0.2, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, str(i), ha='center', va='center', fontweight='bold')
        
        for i, (x, y) in enumerate(right_positions):
            color = 'lightcoral' if any(right_v == i for _, right_v in matching_pairs) else 'lightgray'
            circle = patches.Circle((x, y), 0.2, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, str(i), ha='center', va='center', fontweight='bold')
        
        # Labels and formatting
        ax.text(-0.5, max(left_size, right_size), "Left\nVertices", ha='center', va='top', 
               fontweight='bold', fontsize=12)
        ax.text(4.5, max(left_size, right_size), "Right\nVertices", ha='center', va='top', 
               fontweight='bold', fontsize=12)
        
        # Title with statistics
        if weights:
            title += f"\nMatching Size: {len(matching_pairs)}, Total Weight: {total_weight:.2f}"
        else:
            title += f"\nMatching Size: {len(matching_pairs)}"
            
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(-1, 5)
        ax.set_ylim(-0.5, max(left_size, right_size) * 2 + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        return fig


def compare_algorithms_on_sample():
    """Compare both algorithms on sample data with visualization."""
    print("=" * 60)
    print("BIPARTITE GRAPH MATCHING ALGORITHMS COMPARISON")
    print("=" * 60)
    
    # Sample 1: Small unweighted graph for Hopcroft-Karp
    print("\n1. HOPCROFT-KARP ALGORITHM (Maximum Cardinality Matching)")
    print("-" * 50)
    
    edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2), (3, 2), (3, 3)]
    left_size, right_size = 4, 4
    
    print(f"Graph: {left_size} left vertices, {right_size} right vertices")
    print(f"Edges: {edges}")
    
    start_time = time.perf_counter()
    hk_algorithm = create_bipartite_graph_from_edges(edges, left_size, right_size)
    hk_matching, hk_size = hk_algorithm.solve()
    hk_time = time.perf_counter() - start_time
    
    print(f"Maximum Matching: {hk_matching}")
    print(f"Matching Size: {hk_size}")
    print(f"Execution Time: {hk_time:.6f} seconds")
    
    # Visualize Hopcroft-Karp result
    fig1 = MatchingVisualizer.visualize_matching(
        hk_matching, left_size, right_size, edges,
        title="Hopcroft-Karp Algorithm Result"
    )
    fig1.savefig('hopcroft_karp_result.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'hopcroft_karp_result.png'")
    
    # Sample 2: Small weighted graph for Hungarian Algorithm
    print("\n2. HUNGARIAN ALGORITHM (Maximum Weight Matching)")
    print("-" * 50)
    
    cost_matrix = [
        [4, 1, 3, 0],
        [2, 0, 5, 4],
        [3, 2, 2, 1],
        [1, 3, 4, 2]
    ]
    
    print("Cost Matrix:")
    for i, row in enumerate(cost_matrix):
        print(f"  {i}: {row}")
    
    start_time = time.perf_counter()
    hungarian_algorithm = HungarianAlgorithm(cost_matrix)
    hungarian_matching, hungarian_weight = hungarian_algorithm.solve()
    hungarian_time = time.perf_counter() - start_time
    
    print(f"Maximum Weight Matching: {hungarian_matching}")
    print(f"Total Weight: {hungarian_weight}")
    print(f"Execution Time: {hungarian_time:.6f} seconds")
    
    # Create all edges with weights for visualization
    all_edges_weighted = []
    weights_dict = {}
    for i in range(len(cost_matrix)):
        for j in range(len(cost_matrix[0])):
            all_edges_weighted.append((i, j))
            weights_dict[(i, j)] = cost_matrix[i][j]
    
    # Visualize Hungarian result
    fig2 = MatchingVisualizer.visualize_matching(
        hungarian_matching, len(cost_matrix), len(cost_matrix[0]),
        all_edges_weighted, title="Hungarian Algorithm Result",
        weights=weights_dict
    )
    fig2.savefig('hungarian_result.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'hungarian_result.png'")
    
    # Summary comparison
    print("\n3. ALGORITHM COMPARISON SUMMARY")
    print("-" * 50)
    print(f"{'Algorithm':<20} {'Problem Type':<25} {'Result':<20} {'Time (s)':<15}")
    print("-" * 80)
    print(f"{'Hopcroft-Karp':<20} {'Max Cardinality':<25} {'Size: ' + str(hk_size):<20} {hk_time:<15.6f}")
    print(f"{'Hungarian':<20} {'Max Weight':<25} {'Weight: ' + str(hungarian_weight):<20} {hungarian_time:<15.6f}")
    
    plt.show()
    
    return {
        'hopcroft_karp': {
            'matching': hk_matching,
            'size': hk_size,
            'time': hk_time
        },
        'hungarian': {
            'matching': hungarian_matching,
            'weight': hungarian_weight,
            'time': hungarian_time
        }
    }


def generate_random_comparison():
    """Compare algorithms on randomly generated graphs."""
    print("\n4. RANDOM GRAPH COMPARISON")
    print("-" * 50)
    
    generator = BipartiteGraphGenerator()
    
    # Generate random graphs of different sizes
    sizes = [5, 10, 15, 20]
    
    for n in sizes:
        print(f"\nGraph size: {n}x{n}")
        print("-" * 25)
        
        # Generate random bipartite graph
        edges = generator.generate_random_bipartite_graph(n, n, density=0.4)
        weights = generator.generate_random_weights(edges, min_weight=1, max_weight=10)
        
        # Hopcroft-Karp on unweighted version
        start_time = time.perf_counter()
        hk_alg = create_bipartite_graph_from_edges(edges, n, n)
        hk_matching, hk_size = hk_alg.solve()
        hk_time = time.perf_counter() - start_time
        
        # Hungarian on weighted complete bipartite graph
        weight_matrix = generator.edges_to_weight_matrix(edges, weights, n, n)
        start_time = time.perf_counter()
        hungarian_matching, hungarian_weight = solve_maximum_weight_matching(weight_matrix)
        hungarian_time = time.perf_counter() - start_time
        
        print(f"Hopcroft-Karp: {hk_size} edges matched in {hk_time:.6f}s")
        print(f"Hungarian: {len(hungarian_matching)} edges matched (weight: {hungarian_weight:.2f}) in {hungarian_time:.6f}s")


if __name__ == "__main__":
    try:
        # Run main comparison
        results = compare_algorithms_on_sample()
        
        # Run random graph comparison
        generate_random_comparison()
        
        print("\n" + "=" * 60)
        print("COMPARISON COMPLETED SUCCESSFULLY")
        print("Check the generated PNG files for visualizations!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
