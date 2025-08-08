"""
Ford-Fulkerson algorithm implementation for maximum flow problem.

This module implements the generic Ford-Fulkerson algorithm that finds
augmenting paths using depth-first search (DFS). The algorithm repeatedly
finds augmenting paths and increases flow until no more paths exist.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import copy


class FordFulkerson:
    """
    Implementation of the Ford-Fulkerson algorithm for maximum flow.
    
    The algorithm uses DFS to find augmenting paths in the residual graph
    and continues until no more augmenting paths can be found.
    
    Attributes:
        graph: Adjacency list representation of the flow network
        residual_graph: Residual graph used during computation
        iterations: Number of iterations (augmenting paths found)
    """
    
    def __init__(self, graph: Dict[int, List[Tuple[int, int]]]):
        """
        Initialize the Ford-Fulkerson algorithm.
        
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
    
    def _dfs_find_path(self, source: int, sink: int, visited: set, 
                      path: List[int]) -> Optional[List[int]]:
        """
        Find an augmenting path using depth-first search.
        
        Args:
            source: Current node in DFS
            sink: Target sink node
            visited: Set of visited nodes
            path: Current path being explored
            
        Returns:
            List of nodes representing augmenting path, or None if not found
        """
        if source == sink:
            return path + [sink]
            
        visited.add(source)
        
        for neighbor in self.residual_graph[source]:
            if neighbor not in visited and self.residual_graph[source][neighbor] > 0:
                result = self._dfs_find_path(neighbor, sink, visited, path + [source])
                if result:
                    return result
                    
        return None
    
    def _find_augmenting_path(self, source: int, sink: int) -> Optional[List[int]]:
        """
        Find an augmenting path from source to sink.
        
        Args:
            source: Source node
            sink: Sink node
            
        Returns:
            List of nodes representing the path, or None if no path exists
        """
        visited = set()
        return self._dfs_find_path(source, sink, visited, [])
    
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
        Compute maximum flow from source to sink.
        
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
            # Find augmenting path
            path = self._find_augmenting_path(source, sink)
            
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
