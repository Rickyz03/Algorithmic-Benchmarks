"""
Heuristic algorithms for the Traveling Salesman Problem.

This module implements various heuristic approaches including:
- Nearest Neighbor algorithm
- 2-opt local search improvement
- Combination of constructive heuristics with local search
"""

import numpy as np
from typing import List, Tuple, Set, Optional
import random


def nearest_neighbor_tsp(distance_matrix: np.ndarray, start_city: int = 0) -> Tuple[List[int], float]:
    """
    Solve TSP using Nearest Neighbor heuristic.
    
    This greedy algorithm starts at a given city and repeatedly
    visits the nearest unvisited city until all cities are visited.
    
    Time complexity: O(n²)
    Space complexity: O(n)
    
    Args:
        distance_matrix: Symmetric distance matrix
        start_city: Starting city index
        
    Returns:
        Tuple containing:
        - Tour as list of vertex indices
        - Total tour length
    """
    n = distance_matrix.shape[0]
    
    if n <= 1:
        return list(range(n)), 0.0
    
    visited = set([start_city])
    tour = [start_city]
    current_city = start_city
    total_length = 0.0
    
    # Visit nearest unvisited city at each step
    for _ in range(n - 1):
        nearest_city = None
        nearest_distance = float('inf')
        
        for next_city in range(n):
            if next_city not in visited:
                distance = distance_matrix[current_city][next_city]
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_city = next_city
        
        # Move to nearest city
        visited.add(nearest_city)
        tour.append(nearest_city)
        total_length += nearest_distance
        current_city = nearest_city
    
    # Return to starting city
    total_length += distance_matrix[current_city][start_city]
    
    return tour, total_length


def multi_start_nearest_neighbor(distance_matrix: np.ndarray, num_starts: Optional[int] = None) -> Tuple[List[int], float]:
    """
    Solve TSP using Nearest Neighbor with multiple starting points.
    
    Tries nearest neighbor from different starting cities and returns the best result.
    
    Args:
        distance_matrix: Symmetric distance matrix
        num_starts: Number of different starting cities to try (default: all cities)
        
    Returns:
        Tuple containing:
        - Best tour found
        - Best tour length
    """
    n = distance_matrix.shape[0]
    
    if num_starts is None:
        num_starts = n
    
    best_tour = None
    best_length = float('inf')
    
    start_cities = list(range(n)) if num_starts >= n else random.sample(range(n), num_starts)
    
    for start_city in start_cities:
        tour, length = nearest_neighbor_tsp(distance_matrix, start_city)
        if length < best_length:
            best_length = length
            best_tour = tour.copy()
    
    return best_tour, best_length


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


def two_opt_improvement(tour: List[int], distance_matrix: np.ndarray, max_iterations: Optional[int] = None) -> Tuple[List[int], float]:
    """
    Improve a TSP tour using 2-opt local search.
    
    The 2-opt algorithm iteratively removes two edges from the tour
    and reconnects the two paths in a different way if it improves the tour length.
    
    Args:
        tour: Initial tour as list of vertex indices
        distance_matrix: Symmetric distance matrix
        max_iterations: Maximum number of iterations (default: unlimited until no improvement)
        
    Returns:
        Tuple containing:
        - Improved tour
        - Improved tour length
    """
    if len(tour) < 4:  # 2-opt requires at least 4 cities
        return tour.copy(), calculate_tour_length(tour, distance_matrix)
    
    current_tour = tour.copy()
    current_length = calculate_tour_length(current_tour, distance_matrix)
    n = len(current_tour)
    
    improved = True
    iterations = 0
    
    while improved and (max_iterations is None or iterations < max_iterations):
        improved = False
        iterations += 1
        
        for i in range(n):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:  # Skip if it would just reverse the entire tour
                    continue
                
                # Calculate change in tour length if we perform 2-opt swap
                # Current edges: (i, i+1) and (j, j+1)
                # New edges: (i, j) and (i+1, j+1)
                current_dist = (distance_matrix[current_tour[i]][current_tour[(i + 1) % n]] +
                               distance_matrix[current_tour[j]][current_tour[(j + 1) % n]])
                
                new_dist = (distance_matrix[current_tour[i]][current_tour[j]] +
                           distance_matrix[current_tour[(i + 1) % n]][current_tour[(j + 1) % n]])
                
                if new_dist < current_dist:
                    # Perform 2-opt swap: reverse the segment between i+1 and j
                    new_tour = current_tour.copy()
                    new_tour[(i + 1):(j + 1)] = reversed(new_tour[(i + 1):(j + 1)])
                    
                    current_tour = new_tour
                    current_length = calculate_tour_length(current_tour, distance_matrix)
                    improved = True
                    break
            
            if improved:
                break
    
    return current_tour, current_length


def nearest_neighbor_with_2opt(distance_matrix: np.ndarray, start_city: int = 0, 
                               max_2opt_iterations: Optional[int] = None) -> Tuple[List[int], float]:
    """
    Solve TSP using Nearest Neighbor followed by 2-opt improvement.
    
    Args:
        distance_matrix: Symmetric distance matrix
        start_city: Starting city for nearest neighbor
        max_2opt_iterations: Maximum 2-opt iterations
        
    Returns:
        Tuple containing:
        - Improved tour
        - Tour length after improvement
    """
    # Get initial tour using nearest neighbor
    initial_tour, _ = nearest_neighbor_tsp(distance_matrix, start_city)
    
    # Improve using 2-opt
    improved_tour, improved_length = two_opt_improvement(initial_tour, distance_matrix, max_2opt_iterations)
    
    return improved_tour, improved_length


def multi_start_nn_with_2opt(distance_matrix: np.ndarray, num_starts: Optional[int] = None,
                             max_2opt_iterations: Optional[int] = None) -> Tuple[List[int], float]:
    """
    Solve TSP using multi-start Nearest Neighbor with 2-opt improvement.
    
    This combines the multi-start nearest neighbor with 2-opt local search
    to get better solutions.
    
    Args:
        distance_matrix: Symmetric distance matrix
        num_starts: Number of different starting cities to try
        max_2opt_iterations: Maximum 2-opt iterations per start
        
    Returns:
        Tuple containing:
        - Best improved tour found
        - Best tour length
    """
    n = distance_matrix.shape[0]
    
    if num_starts is None:
        num_starts = n
    
    best_tour = None
    best_length = float('inf')
    
    start_cities = list(range(n)) if num_starts >= n else random.sample(range(n), num_starts)
    
    for start_city in start_cities:
        tour, length = nearest_neighbor_with_2opt(distance_matrix, start_city, max_2opt_iterations)
        if length < best_length:
            best_length = length
            best_tour = tour.copy()
    
    return best_tour, best_length


def random_tour(n: int, seed: Optional[int] = None) -> List[int]:
    """
    Generate a random tour visiting all cities.
    
    Args:
        n: Number of cities
        seed: Random seed for reproducibility
        
    Returns:
        Random permutation representing a tour
    """
    if seed is not None:
        random.seed(seed)
    
    tour = list(range(n))
    random.shuffle(tour)
    return tour


def random_restart_2opt(distance_matrix: np.ndarray, num_restarts: int = 10, 
                       max_2opt_iterations: Optional[int] = None, seed: Optional[int] = None) -> Tuple[List[int], float]:
    """
    Solve TSP using random restarts with 2-opt improvement.
    
    Args:
        distance_matrix: Symmetric distance matrix
        num_restarts: Number of random starting tours to try
        max_2opt_iterations: Maximum 2-opt iterations per restart
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - Best tour found
        - Best tour length
    """
    n = distance_matrix.shape[0]
    best_tour = None
    best_length = float('inf')
    
    if seed is not None:
        random.seed(seed)
    
    for restart in range(num_restarts):
        # Generate random starting tour
        initial_tour = random_tour(n, seed=None)  # Don't set seed here to get different tours
        
        # Improve with 2-opt
        improved_tour, improved_length = two_opt_improvement(initial_tour, distance_matrix, max_2opt_iterations)
        
        if improved_length < best_length:
            best_length = improved_length
            best_tour = improved_tour.copy()
    
    return best_tour, best_length


if __name__ == "__main__":
    # Example usage and testing
    from utils.graph_generator import generate_euclidean_tsp_instance
    
    print("Testing heuristic TSP algorithms...")
    
    # Generate test instance
    n = 8
    distance_matrix, coords = generate_euclidean_tsp_instance(n, seed=42)
    
    print(f"Distance matrix for {n} cities:")
    print(f"Matrix shape: {distance_matrix.shape}")
    
    # Test Nearest Neighbor
    print("\nNearest Neighbor (from city 0):")
    tour_nn, length_nn = nearest_neighbor_tsp(distance_matrix, 0)
    print(f"Tour: {tour_nn}")
    print(f"Length: {length_nn:.2f}")
    
    # Test Multi-start Nearest Neighbor
    print("\nMulti-start Nearest Neighbor:")
    tour_msnn, length_msnn = multi_start_nearest_neighbor(distance_matrix)
    print(f"Tour: {tour_msnn}")
    print(f"Length: {length_msnn:.2f}")
    print(f"Improvement over single NN: {((length_nn - length_msnn) / length_nn * 100):.1f}%")
    
    # Test 2-opt improvement
    print("\n2-opt improvement on NN tour:")
    tour_2opt, length_2opt = two_opt_improvement(tour_nn, distance_matrix)
    print(f"Original tour: {tour_nn}")
    print(f"Improved tour: {tour_2opt}")
    print(f"Original length: {length_nn:.2f}")
    print(f"Improved length: {length_2opt:.2f}")
    print(f"Improvement: {((length_nn - length_2opt) / length_nn * 100):.1f}%")
    
    # Test NN + 2-opt combination
    print("\nNearest Neighbor + 2-opt:")
    tour_nn2opt, length_nn2opt = nearest_neighbor_with_2opt(distance_matrix)
    print(f"Tour: {tour_nn2opt}")
    print(f"Length: {length_nn2opt:.2f}")
    
    # Test multi-start NN + 2-opt
    print("\nMulti-start NN + 2-opt:")
    tour_best, length_best = multi_start_nn_with_2opt(distance_matrix)
    print(f"Tour: {tour_best}")
    print(f"Length: {length_best:.2f}")
    
    # Test random restart 2-opt
    print("\nRandom restart + 2-opt:")
    tour_rr, length_rr = random_restart_2opt(distance_matrix, num_restarts=20, seed=42)
    print(f"Tour: {tour_rr}")
    print(f"Length: {length_rr:.2f}")
    
    print(f"\nSummary of results:")
    print(f"Nearest Neighbor: {length_nn:.2f}")
    print(f"Multi-start NN: {length_msnn:.2f}")
    print(f"NN + 2-opt: {length_nn2opt:.2f}")
    print(f"Multi-start NN + 2-opt: {length_best:.2f}")
    print(f"Random restart + 2-opt: {length_rr:.2f}")
