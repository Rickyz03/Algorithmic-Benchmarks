"""
Exact algorithms for the Traveling Salesman Problem.

This module implements exact TSP algorithms including:
- Brute force enumeration
- Branch and Bound
- Held-Karp dynamic programming algorithm
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import itertools
import math


def brute_force_tsp(distance_matrix: np.ndarray) -> Tuple[List[int], float]:
    """
    Solve TSP using brute force enumeration of all possible tours.
    
    Time complexity: O(n!)
    Space complexity: O(1)
    
    Args:
        distance_matrix: Symmetric distance matrix
        
    Returns:
        Tuple containing:
        - Optimal tour as list of vertex indices
        - Optimal tour length
    """
    n = distance_matrix.shape[0]
    
    if n <= 1:
        return list(range(n)), 0.0
    
    cities = list(range(1, n))  # Fix city 0 as starting point
    best_tour = None
    best_length = float('inf')
    
    # Generate all permutations of cities (excluding fixed starting city)
    for perm in itertools.permutations(cities):
        tour = [0] + list(perm)
        length = calculate_tour_length(tour, distance_matrix)
        
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


class BranchAndBound:
    """Branch and Bound solver for TSP."""
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize Branch and Bound solver.
        
        Args:
            distance_matrix: Symmetric distance matrix
        """
        self.distance_matrix = distance_matrix
        self.n = distance_matrix.shape[0]
        self.best_tour = None
        self.best_length = float('inf')
        self.nodes_explored = 0
    
    def solve(self) -> Tuple[List[int], float]:
        """
        Solve TSP using Branch and Bound.
        
        Returns:
            Tuple containing:
            - Optimal tour as list of vertex indices
            - Optimal tour length
        """
        if self.n <= 1:
            return list(range(self.n)), 0.0
        
        # Initialize with starting city 0
        initial_path = [0]
        initial_visited = {0}
        initial_bound = self._calculate_lower_bound(initial_path, initial_visited)
        
        self._branch_and_bound(initial_path, initial_visited, 0.0, initial_bound)
        
        return self.best_tour, self.best_length
    
    def _branch_and_bound(self, path: List[int], visited: Set[int], 
                         current_cost: float, lower_bound: float):
        """Recursive branch and bound implementation."""
        self.nodes_explored += 1
        
        # Pruning: if lower bound exceeds best known solution, prune
        if lower_bound >= self.best_length:
            return
        
        # Base case: all cities visited
        if len(path) == self.n:
            # Complete the tour by returning to start
            total_cost = current_cost + self.distance_matrix[path[-1]][path[0]]
            if total_cost < self.best_length:
                self.best_length = total_cost
                self.best_tour = path.copy()
            return
        
        # Branch: try all unvisited cities
        current_city = path[-1]
        for next_city in range(self.n):
            if next_city not in visited:
                new_path = path + [next_city]
                new_visited = visited | {next_city}
                new_cost = current_cost + self.distance_matrix[current_city][next_city]
                new_bound = self._calculate_lower_bound(new_path, new_visited)
                
                self._branch_and_bound(new_path, new_visited, new_cost, new_bound)
    
    def _calculate_lower_bound(self, path: List[int], visited: Set[int]) -> float:
        """
        Calculate lower bound for current partial path using MST heuristic.
        
        The lower bound consists of:
        1. Cost of the current path
        2. Minimum cost to connect remaining unvisited cities (using MST)
        3. Minimum cost to return to starting city
        """
        if len(path) == self.n:
            return calculate_tour_length(path, self.distance_matrix)
        
        current_cost = 0.0
        for i in range(len(path) - 1):
            current_cost += self.distance_matrix[path[i]][path[i + 1]]
        
        unvisited = [i for i in range(self.n) if i not in visited]
        
        if not unvisited:
            return current_cost + self.distance_matrix[path[-1]][path[0]]
        
        # MST of unvisited cities
        mst_cost = self._mst_cost(unvisited)
        
        # Minimum cost from last city in path to any unvisited city
        min_to_unvisited = min(self.distance_matrix[path[-1]][city] for city in unvisited)
        
        # Minimum cost from any unvisited city back to start
        min_to_start = min(self.distance_matrix[city][path[0]] for city in unvisited)
        
        return current_cost + mst_cost + min_to_unvisited + min_to_start
    
    def _mst_cost(self, vertices: List[int]) -> float:
        """Calculate MST cost for given vertices using Prim's algorithm."""
        if len(vertices) <= 1:
            return 0.0
        
        visited = {vertices[0]}
        total_cost = 0.0
        
        while len(visited) < len(vertices):
            min_edge_cost = float('inf')
            next_vertex = None
            
            for v in visited:
                for u in vertices:
                    if u not in visited and self.distance_matrix[v][u] < min_edge_cost:
                        min_edge_cost = self.distance_matrix[v][u]
                        next_vertex = u
            
            visited.add(next_vertex)
            total_cost += min_edge_cost
        
        return total_cost


def held_karp_tsp(distance_matrix: np.ndarray) -> Tuple[List[int], float]:
    """
    Solve TSP using Held-Karp dynamic programming algorithm.
    
    Time complexity: O(n² × 2ⁿ)
    Space complexity: O(n × 2ⁿ)
    
    Args:
        distance_matrix: Symmetric distance matrix
        
    Returns:
        Tuple containing:
        - Optimal tour as list of vertex indices
        - Optimal tour length
    """
    n = distance_matrix.shape[0]
    
    if n <= 1:
        return list(range(n)), 0.0
    
    if n == 2:
        return [0, 1], distance_matrix[0][1] + distance_matrix[1][0]
    
    # DP table: dp[mask][i] = minimum cost to visit cities in mask ending at city i
    # mask is a bitmask representing visited cities
    dp = {}
    parent = {}
    
    # Base case: start at city 0, visit only city 0
    dp[(1, 0)] = 0  # mask=1 (binary 001) means only city 0 visited
    
    # Fill DP table for all subset sizes
    for subset_size in range(2, n + 1):
        for mask in range(1, 1 << n):
            if bin(mask).count('1') != subset_size:
                continue
            if not (mask & 1):  # City 0 must be in the subset
                continue
            
            for i in range(1, n):
                if not (mask & (1 << i)):  # City i not in current subset
                    continue
                
                prev_mask = mask ^ (1 << i)  # Remove city i from mask
                min_cost = float('inf')
                best_prev = None
                
                for j in range(n):
                    if (prev_mask & (1 << j)) and (prev_mask, j) in dp:
                        cost = dp[(prev_mask, j)] + distance_matrix[j][i]
                        if cost < min_cost:
                            min_cost = cost
                            best_prev = j
                
                if min_cost < float('inf'):
                    dp[(mask, i)] = min_cost
                    parent[(mask, i)] = best_prev
    
    # Find minimum cost to complete the tour
    final_mask = (1 << n) - 1  # All cities visited
    min_cost = float('inf')
    last_city = None
    
    for i in range(1, n):
        if (final_mask, i) in dp:
            cost = dp[(final_mask, i)] + distance_matrix[i][0]
            if cost < min_cost:
                min_cost = cost
                last_city = i
    
    if min_cost == float('inf'):
        return None, float('inf')
    
    # Reconstruct the tour
    tour = [0]
    mask = final_mask
    current = last_city
    
    while current != 0:
        tour.append(current)
        next_current = parent[(mask, current)]
        mask ^= (1 << current)
        current = next_current
    
    tour.reverse()
    
    return tour, min_cost


if __name__ == "__main__":
    # Example usage and testing
    from utils.graph_generator import generate_euclidean_tsp_instance
    
    print("Testing exact TSP algorithms...")
    
    # Generate small test instance
    n = 5
    distance_matrix, coords = generate_euclidean_tsp_instance(n, seed=42)
    
    print(f"Distance matrix for {n} cities:")
    print(distance_matrix)
    
    # Test brute force
    print("\nBrute Force:")
    tour_bf, length_bf = brute_force_tsp(distance_matrix)
    print(f"Tour: {tour_bf}")
    print(f"Length: {length_bf:.2f}")
    
    # Test Branch and Bound
    print("\nBranch and Bound:")
    bb_solver = BranchAndBound(distance_matrix)
    tour_bb, length_bb = bb_solver.solve()
    print(f"Tour: {tour_bb}")
    print(f"Length: {length_bb:.2f}")
    print(f"Nodes explored: {bb_solver.nodes_explored}")
    
    # Test Held-Karp
    print("\nHeld-Karp Dynamic Programming:")
    tour_hk, length_hk = held_karp_tsp(distance_matrix)
    print(f"Tour: {tour_hk}")
    print(f"Length: {length_hk:.2f}")
    
    # Verify all algorithms give same result
    print(f"\nVerification:")
    print(f"Brute force length: {length_bf:.2f}")
    print(f"Branch and bound length: {length_bb:.2f}")
    print(f"Held-Karp length: {length_hk:.2f}")
    print(f"All equal: {abs(length_bf - length_bb) < 1e-10 and abs(length_bf - length_hk) < 1e-10}")
