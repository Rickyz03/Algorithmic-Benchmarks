"""
Hopcroft-Karp Algorithm Implementation for Maximum Cardinality Matching in Bipartite Graphs.

The Hopcroft-Karp algorithm finds a maximum cardinality matching in an unweighted
bipartite graph in O(√V · E) time complexity, which is optimal for dense graphs.

The algorithm works by finding maximal sets of vertex-disjoint augmenting paths
in each iteration using BFS and DFS.

Author: Your Name
Date: 2025
"""

from collections import deque, defaultdict
from typing import List, Tuple, Set, Dict, Optional
import sys


class HopcroftKarpAlgorithm:
    """
    Implementation of the Hopcroft-Karp algorithm for maximum cardinality matching
    in bipartite graphs.
    """
    
    def __init__(self, left_vertices: int, right_vertices: int):
        """
        Initialize the Hopcroft-Karp algorithm.
        
        Args:
            left_vertices: Number of vertices in the left partition
            right_vertices: Number of vertices in the right partition
        """
        self.left_size = left_vertices
        self.right_size = right_vertices
        self.graph = defaultdict(list)  # Adjacency list representation
        
        # Matching arrays: -1 means unmatched
        self.match_left = [-1] * self.left_size
        self.match_right = [-1] * self.right_size
        
        # Distance array for BFS
        self.dist = [0] * self.left_size
        self.NIL = sys.maxsize  # Special value for unmatched vertices
        
    def add_edge(self, left_vertex: int, right_vertex: int):
        """
        Add an edge between a left vertex and a right vertex.
        
        Args:
            left_vertex: Index of vertex in left partition (0-indexed)
            right_vertex: Index of vertex in right partition (0-indexed)
        """
        if left_vertex >= self.left_size or left_vertex < 0:
            raise ValueError(f"Left vertex {left_vertex} out of bounds")
        if right_vertex >= self.right_size or right_vertex < 0:
            raise ValueError(f"Right vertex {right_vertex} out of bounds")
            
        self.graph[left_vertex].append(right_vertex)
    
    def _bfs(self) -> bool:
        """
        Breadth-first search to find augmenting paths.
        
        This BFS builds a layered graph and finds the shortest augmenting paths.
        It returns True if at least one augmenting path exists.
        
        Returns:
            True if augmenting paths exist, False otherwise
        """
        queue = deque()
        
        # Initialize distances
        for u in range(self.left_size):
            if self.match_left[u] == -1:
                # Unmatched vertex in left partition
                self.dist[u] = 0
                queue.append(u)
            else:
                self.dist[u] = self.NIL
        
        # Distance to NIL (represents unmatched vertices)
        dist_nil = self.NIL
        
        # BFS
        while queue:
            u = queue.popleft()
            
            if self.dist[u] < dist_nil:
                for v in self.graph[u]:
                    # v is a vertex in right partition
                    matched_u = self.match_right[v]
                    
                    if matched_u == -1:
                        # Found an augmenting path ending at unmatched vertex
                        dist_nil = self.dist[u] + 1
                    elif self.dist[matched_u] == self.NIL:
                        # Continue building the layered graph
                        self.dist[matched_u] = self.dist[u] + 1
                        queue.append(matched_u)
        
        return dist_nil != self.NIL
    
    def _dfs(self, u: int) -> bool:
        """
        Depth-first search to find and augment along vertex-disjoint paths.
        
        Args:
            u: Current left vertex being processed
            
        Returns:
            True if an augmenting path was found and augmented, False otherwise
        """
        if u != -1:  # u is not NIL
            for v in self.graph[u]:
                matched_u = self.match_right[v]
                
                # Check if this edge can be part of an augmenting path
                if matched_u == -1 or (self.dist[matched_u] == self.dist[u] + 1 and self._dfs(matched_u)):
                    # Augment the matching along this path
                    self.match_right[v] = u
                    self.match_left[u] = v
                    return True
            
            # No augmenting path found through u
            self.dist[u] = self.NIL
            return False
        
        return True  # NIL vertex always returns True
    
    def solve(self) -> Tuple[List[Tuple[int, int]], int]:
        """
        Find the maximum cardinality matching using the Hopcroft-Karp algorithm.
        
        Returns:
            Tuple containing:
            - List of matched pairs (left_vertex, right_vertex)
            - Size of the maximum matching
        """
        matching_size = 0
        
        # Keep finding augmenting paths until none exist
        while self._bfs():
            # Try to find vertex-disjoint augmenting paths using DFS
            for u in range(self.left_size):
                if self.match_left[u] == -1 and self._dfs(u):
                    matching_size += 1
        
        # Extract matching pairs
        matching_pairs = []
        for u in range(self.left_size):
            if self.match_left[u] != -1:
                matching_pairs.append((u, self.match_left[u]))
        
        return matching_pairs, matching_size
    
    def get_matching_info(self) -> Dict:
        """
        Get detailed information about the current matching.
        
        Returns:
            Dictionary with matching statistics and details
        """
        matched_left = sum(1 for match in self.match_left if match != -1)
        matched_right = sum(1 for match in self.match_right if match != -1)
        
        unmatched_left = [i for i in range(self.left_size) if self.match_left[i] == -1]
        unmatched_right = [i for i in range(self.right_size) if self.match_right[i] == -1]
        
        return {
            'matching_size': matched_left,
            'matched_left_vertices': matched_left,
            'matched_right_vertices': matched_right,
            'unmatched_left': unmatched_left,
            'unmatched_right': unmatched_right,
            'total_edges': sum(len(adj_list) for adj_list in self.graph.values())
        }


def create_bipartite_graph_from_edges(edges: List[Tuple[int, int]], 
                                     left_size: Optional[int] = None, 
                                     right_size: Optional[int] = None) -> HopcroftKarpAlgorithm:
    """
    Create a HopcroftKarpAlgorithm instance from a list of edges.
    
    Args:
        edges: List of (left_vertex, right_vertex) tuples
        left_size: Number of left vertices (inferred if None)
        right_size: Number of right vertices (inferred if None)
        
    Returns:
        Configured HopcroftKarpAlgorithm instance
    """
    if not edges:
        return HopcroftKarpAlgorithm(0, 0)
    
    if left_size is None:
        left_size = max(edge[0] for edge in edges) + 1
    if right_size is None:
        right_size = max(edge[1] for edge in edges) + 1
    
    algorithm = HopcroftKarpAlgorithm(left_size, right_size)
    
    for left_vertex, right_vertex in edges:
        algorithm.add_edge(left_vertex, right_vertex)
    
    return algorithm


if __name__ == "__main__":
    # Example usage
    edges = [
        (0, 0), (0, 1),
        (1, 1), (1, 2),
        (2, 0), (2, 2),
        (3, 2)
    ]
    
    algorithm = create_bipartite_graph_from_edges(edges)
    matching, size = algorithm.solve()
    
    print("Bipartite Graph Edges:", edges)
    print(f"Maximum Matching: {matching}")
    print(f"Matching Size: {size}")
    print("Matching Info:", algorithm.get_matching_info())
