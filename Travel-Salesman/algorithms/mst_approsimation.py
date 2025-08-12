"""
MST-based 2-approximation algorithm for the Traveling Salesman Problem.

This module implements the classical 2-approximation algorithm for TSP
on metric graphs using Minimum Spanning Tree (MST) and Depth-First Search (DFS).
"""

import numpy as np
from typing import List, Tuple, Set
import heapq


class UnionFind:
    """Union-Find data structure for Kruskal's algorithm."""
    
    def __init__(self, n: int):
        """
        Initialize Union-Find structure.
        
        Args:
            n: Number of elements
        """
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find root of element x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Union two sets containing x and y.
        
        Returns:
            True if union was performed, False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        return True


def kruskal_mst(distance_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Find Minimum Spanning Tree using Kruskal's algorithm.
    
    Args:
        distance_matrix: Symmetric distance matrix
        
    Returns:
        List of edges in MST as (u, v, weight) tuples
    """
    n = distance_matrix.shape[0]
    edges = []
    
    # Create list of all edges
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((distance_matrix[i][j], i, j))
    
    # Sort edges by weight
    edges.sort()
    
    # Apply Kruskal's algorithm
    uf = UnionFind(n)
    mst_edges = []
    
    for weight, u, v in edges:
        if uf.union(u, v):
            mst_edges.append((u, v, weight))
            if len(mst_edges) == n - 1:
                break
    
    return mst_edges


def prim_mst(distance_matrix: np.ndarray, start: int = 0) -> List[Tuple[int, int, float]]:
    """
    Find Minimum Spanning Tree using Prim's algorithm.
    
    Args:
        distance_matrix: Symmetric distance matrix
        start: Starting vertex for Prim's algorithm
        
    Returns:
        List of edges in MST as (u, v, weight) tuples
    """
    n = distance_matrix.shape[0]
    visited = [False] * n
    min_heap = [(0, start, -1)]  # (weight, vertex, parent)
    mst_edges = []
    
    while min_heap:
        weight, u, parent = heapq.heappop(min_heap)
        
        if visited[u]:
            continue
        
        visited[u] = True
        
        if parent != -1:
            mst_edges.append((parent, u, weight))
        
        # Add adjacent vertices to heap
        for v in range(n):
            if not visited[v] and v != u:
                heapq.heappush(min_heap, (distance_matrix[u][v], v, u))
    
    return mst_edges


def build_adjacency_list(mst_edges: List[Tuple[int, int, float]], n: int) -> List[List[int]]:
    """
    Build adjacency list representation from MST edges.
    
    Args:
        mst_edges: List of MST edges as (u, v, weight) tuples
        n: Number of vertices
        
    Returns:
        Adjacency list representation of the MST
    """
    adj_list = [[] for _ in range(n)]
    
    for u, v, _ in mst_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    return adj_list


def dfs_preorder(adj_list: List[List[int]], start: int = 0) -> List[int]:
    """
    Perform DFS traversal and return preorder sequence.
    
    Args:
        adj_list: Adjacency list representation of the tree
        start: Starting vertex for DFS
        
    Returns:
        List of vertices in DFS preorder
    """
    visited = set()
    preorder = []
    
    def dfs(v):
        visited.add(v)
        preorder.append(v)
        
        for neighbor in adj_list[v]:
            if neighbor not in visited:
                dfs(neighbor)
    
    dfs(start)
    return preorder


def calculate_tour_length(tour: List[int], distance_matrix: np.ndarray) -> float:
    """
    Calculate total length of a tour.
    
    Args:
        tour: List of vertices representing the tour
        distance_matrix: Distance matrix
        
    Returns:
        Total tour length
    """
    if len(tour) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(tour)):
        current = tour[i]
        next_city = tour[(i + 1) % len(tour)]
        total_length += distance_matrix[current][next_city]
    
    return total_length


def mst_2_approximation(distance_matrix: np.ndarray, use_prim: bool = False) -> Tuple[List[int], float]:
    """
    Solve TSP using MST-based 2-approximation algorithm.
    
    This algorithm:
    1. Finds the MST of the complete graph
    2. Performs DFS preorder traversal on the MST
    3. Returns the resulting Hamiltonian cycle
    
    For metric TSP instances, this guarantees a solution with cost ≤ 2 × OPT.
    
    Args:
        distance_matrix: Symmetric distance matrix
        use_prim: If True, use Prim's algorithm; otherwise use Kruskal's
        
    Returns:
        Tuple containing:
        - Tour as list of vertex indices
        - Total tour length
    """
    n = distance_matrix.shape[0]
    
    if n < 2:
        return list(range(n)), 0.0
    
    # Step 1: Find MST
    if use_prim:
        mst_edges = prim_mst(distance_matrix)
    else:
        mst_edges = kruskal_mst(distance_matrix)
    
    # Step 2: Build adjacency list from MST
    adj_list = build_adjacency_list(mst_edges, n)
    
    # Step 3: Perform DFS preorder traversal
    tour = dfs_preorder(adj_list, start=0)
    
    # Step 4: Calculate tour length
    tour_length = calculate_tour_length(tour, distance_matrix)
    
    return tour, tour_length


def get_mst_weight(mst_edges: List[Tuple[int, int, float]]) -> float:
    """
    Calculate total weight of MST.
    
    Args:
        mst_edges: List of MST edges
        
    Returns:
        Total MST weight
    """
    return sum(weight for _, _, weight in mst_edges)


if __name__ == "__main__":
    # Example usage and testing
    from utils.graph_generator import generate_euclidean_tsp_instance
    
    print("Testing MST 2-approximation algorithm...")
    
    # Generate test instance
    n = 6
    distance_matrix, coords = generate_euclidean_tsp_instance(n, seed=42)
    
    print(f"Distance matrix for {n} cities:")
    print(distance_matrix)
    
    # Test with Kruskal's MST
    tour_kruskal, length_kruskal = mst_2_approximation(distance_matrix, use_prim=False)
    print(f"\nKruskal-based 2-approximation:")
    print(f"Tour: {tour_kruskal}")
    print(f"Length: {length_kruskal:.2f}")
    
    # Test with Prim's MST
    tour_prim, length_prim = mst_2_approximation(distance_matrix, use_prim=True)
    print(f"\nPrim-based 2-approximation:")
    print(f"Tour: {tour_prim}")
    print(f"Length: {length_prim:.2f}")
    
    # Calculate MST weight for comparison
    mst_edges = kruskal_mst(distance_matrix)
    mst_weight = get_mst_weight(mst_edges)
    print(f"\nMST weight: {mst_weight:.2f}")
    print(f"Approximation ratio: {length_kruskal / mst_weight:.2f}")
