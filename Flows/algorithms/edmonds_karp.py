"""
Edmonds-Karp algorithm implementation for maximum flow problem.

This module implements the Edmonds-Karp algorithm, which is a specific
implementation of Ford-Fulkerson that uses breadth-first search (BFS)
to find the shortest augmenting paths (in terms of number of edges).
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import copy


class EdmondsKarp:
    """
    Implementation of the Edmonds-Karp algorithm for maximum flow.
    
    This algorithm improves upon Ford-Fulkerson by using BFS to find
    the shortest augmenting paths, guaranteeing polynomial time complexity.
    
    Attributes:
        graph: Adjacency list representation of the flow network
        residual_graph: Residual graph used during computation
        iterations: Number of iterations (augmenting paths found)
    """
    
    def __init__(self, graph: Dict[int, List[Tuple[int, int]]]):
        """
        Initialize the Edmonds-Karp algorithm.
        
        Args:
            graph: Dictionary where keys are node IDs and values are lists
                  of tuples (neighbor, capacity)
        """
        self.graph = graph
        self.residual_graph = None
        self.iterations = 0
        
    def _build_residual_graph(self) -> Dict[int, Dict[int, int]]:
        """
        Build the residual graph from the original graph.
        
        Returns:
            Dictionary representing residual graph with capacities
        """
        residual = defaultdict(lambda: defaultdict(int))
        
        # Add forward edges with original capacities
        for u in self.graph:
            for v, capacity in self.graph[u]:
                residual[u][v] = capacity
                # Ensure backward edge exists (initially 0)
                if residual[v][u] == 0:
                    residual[v][u] = 0
                    
        return residual
    
    def _bfs_find_path(self, source: int, sink: int) -> Optional[List[int]]:
        """
        Find shortest augmenting path using breadth-first search.
        
        Args:
            source: Source node
            sink: Sink node
            
        Returns:
            List of nodes representing the shortest path, or None if not found
        """
        # Keep track of parent to reconstruct path
        parent = {}
        visited = {source}
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            
            # Explore all neighbors with available capacity
            for neighbor in self.residual_graph[current]:
                if neighbor not in visited and self.residual_graph[current][neighbor] > 0:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current
                    
                    # If we reached the sink, reconstruct path
                    if neighbor == sink:
                        path = []
                        node = sink
                        while node != source:
                            path.append(node)
                            node = parent[node]
                        path.append(source)
                        return path[::-1]  # Reverse to get source -> sink order
                        
        return None
    
    def _get_path_capacity(self, path: List[int]) -> int:
        """
        Calculate the minimum capacity along a path (bottleneck).
        
        Args:
            path: List of nodes representing the path
            
        Returns:
            Minimum capacity along the path
        """
        min_capacity = float('inf')
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            min_capacity = min(min_capacity, self.residual_graph[u][v])
            
        return int(min_capacity)
    
    def _update_residual_graph(self, path: List[int], flow: int):
        """
        Update residual graph after sending flow along a path.
        
        Args:
            path: Path along which flow is sent
            flow: Amount of flow to send
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Decrease forward edge capacity
            self.residual_graph[u][v] -= flow
            # Increase backward edge capacity
            self.residual_graph[v][u] += flow
    
    def max_flow(self, source: int, sink: int) -> Tuple[int, int]:
        """
        Compute maximum flow from source to sink using Edmonds-Karp algorithm.
        
        Args:
            source: Source node ID
            sink: Sink node ID
            
        Returns:
            Tuple of (maximum_flow_value, number_of_iterations)
        """
        self.residual_graph = self._build_residual_graph()
        self.iterations = 0
        total_flow = 0
        
        while True:
            # Find shortest augmenting path using BFS
            path = self._bfs_find_path(source, sink)
            
            if not path:
                break
                
            # Calculate bottleneck capacity
            path_flow = self._get_path_capacity(path)
            
            # Update residual graph
            self._update_residual_graph(path, path_flow)
            
            # Add to total flow
            total_flow += path_flow
            self.iterations += 1
            
        return total_flow, self.iterations
    
    def get_flow_edges(self) -> List[Tuple[int, int, int]]:
        """
        Get the flow on each edge in the original graph.
        
        Returns:
            List of tuples (u, v, flow) representing flow on each edge
        """
        if not self.residual_graph:
            return []
            
        flow_edges = []
        
        for u in self.graph:
            for v, original_capacity in self.graph[u]:
                # Flow on edge (u,v) is original_capacity - residual_capacity
                flow = original_capacity - self.residual_graph[u][v]
                if flow > 0:
                    flow_edges.append((u, v, flow))
                    
        return flow_edges
    
    def get_shortest_path_length_distribution(self, source: int, sink: int) -> List[int]:
        """
        Get distribution of shortest path lengths found during execution.
        
        This method re-runs the algorithm to collect path length statistics,
        which is useful for analysis purposes.
        
        Args:
            source: Source node ID
            sink: Sink node ID
            
        Returns:
            List of path lengths for each iteration
        """
        residual = self._build_residual_graph()
        path_lengths = []
        
        while True:
            # Temporarily set residual graph for path finding
            temp_residual = self.residual_graph
            self.residual_graph = residual
            
            path = self._bfs_find_path(source, sink)
            
            if not path:
                self.residual_graph = temp_residual
                break
                
            path_lengths.append(len(path) - 1)  # Number of edges in path
            path_flow = self._get_path_capacity(path)
            self._update_residual_graph(path, path_flow)
            
            self.residual_graph = temp_residual
            
        return path_lengths
