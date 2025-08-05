"""
Dijkstra's shortest path algorithm implementation.

This module implements Dijkstra's algorithm for finding shortest paths in
graphs with non-negative edge weights. It uses a binary heap for efficient
priority queue operations.
"""

import heapq
from typing import Dict, List, Optional, Tuple
from graph import Graph


class DijkstraResult:
    """Container for Dijkstra's algorithm results and statistics."""
    
    def __init__(self):
        self.distances: Dict[int, float] = {}
        self.predecessors: Dict[int, Optional[int]] = {}
        self.path: List[int] = []
        self.path_cost: float = float('inf')
        self.operations_count: int = 0
        self.vertices_visited: int = 0


def dijkstra(graph: Graph, start: int, end: Optional[int] = None) -> DijkstraResult:
    """
    Execute Dijkstra's algorithm to find shortest paths.
    
    Dijkstra's algorithm finds the shortest path from a source vertex to all
    other vertices in a weighted graph with non-negative edge weights. The
    algorithm maintains a priority queue of vertices ordered by their current
    shortest distance from the source.
    
    Time Complexity: O((V + E) log V) where V is vertices and E is edges
    Space Complexity: O(V)
    
    Args:
        graph: The input graph (must have non-negative edge weights)
        start: Source vertex
        end: Target vertex (optional, if None computes all shortest paths)
        
    Returns:
        DijkstraResult containing distances, paths, and performance metrics
        
    Raises:
        ValueError: If the graph contains negative edge weights
    """
    if graph.has_negative_edges():
        raise ValueError("Dijkstra's algorithm cannot handle negative edge weights")
    
    if start not in graph.get_vertices():
        raise ValueError(f"Start vertex {start} not found in graph")
    
    result = DijkstraResult()
    
    # Initialize distances and predecessors
    vertices = graph.get_vertices()
    for vertex in vertices:
        result.distances[vertex] = float('inf')
        result.predecessors[vertex] = None
    
    result.distances[start] = 0.0
    
    # Priority queue: (distance, vertex)
    pq = [(0.0, start)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        result.operations_count += 1
        
        # Skip if we've already processed this vertex with a better distance
        if current in visited:
            continue
        
        visited.add(current)
        result.vertices_visited += 1
        
        # Early termination if we've reached the target
        if end is not None and current == end:
            break
        
        # Skip if current distance is worse than recorded
        if current_dist > result.distances[current]:
            continue
        
        # Relax all neighboring edges
        for neighbor, weight in graph.get_neighbors(current):
            if neighbor in visited:
                continue
            
            new_distance = result.distances[current] + weight
            
            # Relaxation step
            if new_distance < result.distances[neighbor]:
                result.distances[neighbor] = new_distance
                result.predecessors[neighbor] = current
                heapq.heappush(pq, (new_distance, neighbor))
    
    # Reconstruct path if end vertex was specified
    if end is not None:
        if end in graph.get_vertices() and result.distances[end] != float('inf'):
            result.path = _reconstruct_path(result.predecessors, start, end)
            result.path_cost = result.distances[end]
        else:
            result.path = []
            result.path_cost = float('inf')
    
    return result


def dijkstra_all_pairs(graph: Graph) -> Dict[int, DijkstraResult]:
    """
    Compute shortest paths between all pairs of vertices.
    
    This function runs Dijkstra's algorithm from each vertex as a source,
    effectively computing the all-pairs shortest path problem for graphs
    with non-negative edge weights.
    
    Time Complexity: O(V * (V + E) log V)
    
    Args:
        graph: The input graph
        
    Returns:
        Dictionary mapping each vertex to its DijkstraResult
    """
    results = {}
    
    for vertex in graph.get_vertices():
        results[vertex] = dijkstra(graph, vertex)
    
    return results


def _reconstruct_path(predecessors: Dict[int, Optional[int]], 
                     start: int, end: int) -> List[int]:
    """
    Reconstruct the shortest path from predecessors dictionary.
    
    Args:
        predecessors: Dictionary mapping each vertex to its predecessor
        start: Source vertex
        end: Target vertex
        
    Returns:
        List of vertices representing the shortest path
    """
    path = []
    current = end
    
    while current is not None:
        path.append(current)
        current = predecessors[current]
    
    path.reverse()
    
    # Verify the path starts with the correct vertex
    if path and path[0] == start:
        return path
    else:
        return []  # No path exists


def get_shortest_path(graph: Graph, start: int, end: int) -> Tuple[List[int], float]:
    """
    Convenience function to get the shortest path between two vertices.
    
    Args:
        graph: The input graph
        start: Source vertex
        end: Target vertex
        
    Returns:
        Tuple of (path, cost) where path is a list of vertices
    """
    result = dijkstra(graph, start, end)
    return result.path, result.path_cost
