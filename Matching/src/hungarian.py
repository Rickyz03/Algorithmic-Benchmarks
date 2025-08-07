"""
Hungarian Algorithm Implementation for Maximum Weight Matching in Bipartite Graphs.

The Hungarian algorithm (also known as Kuhn-Munkres algorithm) solves the assignment
problem in polynomial time. It finds a maximum weight matching in a weighted bipartite
graph, or equivalently, a minimum cost perfect matching.

Author: Your Name
Date: 2025
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import copy


class HungarianAlgorithm:
    """
    Implementation of the Hungarian Algorithm for solving the assignment problem.
    
    This implementation finds the maximum weight perfect matching in a complete
    bipartite graph represented as a cost matrix.
    """
    
    def __init__(self, cost_matrix: Union[List[List[float]], np.ndarray]):
        """
        Initialize the Hungarian algorithm with a cost matrix.
        
        Args:
            cost_matrix: Square matrix where cost_matrix[i][j] represents the weight
                        of the edge between left vertex i and right vertex j.
                        For maximum weight matching, use positive weights.
        """
        self.original_matrix = np.array(cost_matrix, dtype=float)
        self.n = len(self.original_matrix)
        
        # Validate input
        if self.original_matrix.shape != (self.n, self.n):
            raise ValueError("Cost matrix must be square")
        
        # For maximum weight matching, we convert to minimum cost by negating
        self.cost_matrix = -self.original_matrix.copy()
        
        # Algorithm state
        self.u = np.zeros(self.n)  # Dual variables for left vertices
        self.v = np.zeros(self.n)  # Dual variables for right vertices
        self.matching_left = [-1] * self.n  # matching[i] = j means left i matched to right j
        self.matching_right = [-1] * self.n  # matching[j] = i means right j matched to left i
        
    def _find_augmenting_path(self, start: int) -> bool:
        """
        Find an augmenting path starting from an unmatched left vertex.
        
        Args:
            start: Index of unmatched left vertex to start from
            
        Returns:
            True if an augmenting path was found, False otherwise
        """
        visited_left = [False] * self.n
        visited_right = [False] * self.n
        slack = [float('inf')] * self.n
        slack_left = [-1] * self.n
        parent = [-1] * self.n
        
        # Initialize slack values
        for j in range(self.n):
            slack[j] = self.cost_matrix[start][j] - self.u[start] - self.v[j]
            slack_left[j] = start
        
        visited_left[start] = True
        
        while True:
            # Find the minimum slack among unvisited right vertices
            min_slack = float('inf')
            min_j = -1
            
            for j in range(self.n):
                if not visited_right[j] and slack[j] < min_slack:
                    min_slack = slack[j]
                    min_j = j
            
            if min_slack > 0:
                # Update dual variables
                for i in range(self.n):
                    if visited_left[i]:
                        self.u[i] += min_slack
                for j in range(self.n):
                    if visited_right[j]:
                        self.v[j] -= min_slack
                    else:
                        slack[j] -= min_slack
            
            visited_right[min_j] = True
            
            if self.matching_right[min_j] == -1:
                # Found augmenting path, update matching
                current = min_j
                while current != -1:
                    prev = slack_left[current]
                    next_vertex = self.matching_left[prev]
                    
                    self.matching_left[prev] = current
                    self.matching_right[current] = prev
                    
                    current = next_vertex
                return True
            
            # Add the matched left vertex to the alternating tree
            matched_left = self.matching_right[min_j]
            visited_left[matched_left] = True
            
            # Update slack values
            for j in range(self.n):
                if not visited_right[j]:
                    new_slack = self.cost_matrix[matched_left][j] - self.u[matched_left] - self.v[j]
                    if new_slack < slack[j]:
                        slack[j] = new_slack
                        slack_left[j] = matched_left
    
    def solve(self) -> Tuple[List[Tuple[int, int]], float]:
        """
        Solve the maximum weight matching problem using the Hungarian algorithm.
        
        Returns:
            Tuple containing:
            - List of matched pairs (left_vertex, right_vertex)
            - Total weight of the matching
        """
        # Initialize dual variables
        for i in range(self.n):
            self.u[i] = min(self.cost_matrix[i])
        
        for j in range(self.n):
            self.v[j] = 0
        
        # Find maximum matching using augmenting paths
        for i in range(self.n):
            if self.matching_left[i] == -1:
                self._find_augmenting_path(i)
        
        # Extract matching pairs and calculate total weight
        matching_pairs = []
        total_weight = 0.0
        
        for i in range(self.n):
            if self.matching_left[i] != -1:
                j = self.matching_left[i]
                matching_pairs.append((i, j))
                total_weight += self.original_matrix[i][j]
        
        return matching_pairs, total_weight
    
    def get_matching_matrix(self) -> np.ndarray:
        """
        Get the matching as a binary matrix.
        
        Returns:
            Binary matrix where entry (i,j) is 1 if vertex i is matched to vertex j
        """
        matching_matrix = np.zeros((self.n, self.n), dtype=int)
        for i in range(self.n):
            if self.matching_left[i] != -1:
                matching_matrix[i][self.matching_left[i]] = 1
        return matching_matrix


def solve_maximum_weight_matching(cost_matrix: Union[List[List[float]], np.ndarray]) -> Tuple[List[Tuple[int, int]], float]:
    """
    Convenience function to solve maximum weight matching problem.
    
    Args:
        cost_matrix: Square matrix of edge weights
        
    Returns:
        Tuple of (matching_pairs, total_weight)
    """
    hungarian = HungarianAlgorithm(cost_matrix)
    return hungarian.solve()


if __name__ == "__main__":
    # Example usage
    cost_matrix = [
        [4, 1, 3],
        [2, 0, 5],
        [3, 2, 2]
    ]
    
    hungarian = HungarianAlgorithm(cost_matrix)
    matching, weight = hungarian.solve()
    
    print("Cost Matrix:")
    print(np.array(cost_matrix))
    print(f"\nMaximum Weight Matching: {matching}")
    print(f"Total Weight: {weight}")
    print(f"Matching Matrix:")
    print(hungarian.get_matching_matrix())
