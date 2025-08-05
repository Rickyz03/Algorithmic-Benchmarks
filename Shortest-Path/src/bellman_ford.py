"""
Bellman-Ford shortest path algorithm implementation.

This module implements the Bellman-Ford algorithm for finding shortest paths
in graphs that may contain negative edge weights. The algorithm can also
detect negative cycles.
"""

from typing import Dict, List, Optional, Tuple
from graph import Graph


class BellmanFordResult:
    """Container for Bellman-Ford algorithm results and statistics."""
    
    def __init__(self):
        self.distances: Dict[int, float] = {}
        self.predecessors: Dict[int, Optional[int]] = {}
        self.path: List[int] = []
        self.path_cost: float = float('inf')
        self.has_negative_cycle: bool = False
        self.operations_count: int = 0
        self.relaxations_performed: int = 0


def bellman_ford(graph: Graph, start: int, end: Optional[int] = None) -> BellmanFordResult:
    """
    Execute the Bellman-Ford algorithm to find shortest paths.
    
    The Bellman-Ford algorithm finds shortest paths from a source vertex to all
    other vertices in a weighted graph. Unlike Dijkstra's algorithm, it can handle
    graphs with negative edge weights and detect negative cycles.
    
    The algorithm works by relaxing all edges repeatedly. After V-1 iterations,
    if no further improvements can be made, the algorithm has found the shortest
    paths. If improvements can still be made in the V-th iteration, there exists
    a negative cycle.
    
    Time Complexity: O(VE) where V is vertices and E is edges
    Space Complexity: O(V)
    
    Args:
        graph: The input graph
        start: Source vertex  
        end: Target vertex (optional, if None computes all shortest paths)
        
    Returns:
        BellmanFordResult containing distances, paths, and cycle detection info
        
    Raises:
        ValueError: If start vertex is not in the graph
    """
    if start not in graph.get_vertices():
        raise ValueError(f"Start vertex {start} not found in graph")
    
    result = BellmanFordResult()
    vertices = list(graph.get_vertices())
    
    # Step 1: Initialize distances and predecessors
    for vertex in vertices:
        result.distances[vertex] = float('inf')
        result.predecessors[vertex] = None
    
    result.distances[start] = 0.0
    
    # Step 2: Relax edges repeatedly (V-1 times)
    for iteration in range(len(vertices) - 1):
        improved = False
        
        for vertex in vertices:
            if result.distances[vertex] == float('inf'):
                continue
            
            for neighbor, weight in graph.get_neighbors(vertex):
                result.operations_count += 1
                new_distance = result.distances[vertex] + weight
                
                # Relaxation step
                if new_distance < result.distances[neighbor]:
                    result.distances[neighbor] = new_distance
                    result.predecessors[neighbor] = vertex
                    result.relaxations_performed += 1
                    improved = True
        
        # Early termination if no improvements were made
        if not improved:
            break
    
    # Step 3: Check for negative cycles
    for vertex in vertices:
        if result.distances[vertex] == float('inf'):
            continue
        
        for neighbor, weight in graph.get_neighbors(vertex):
            result.operations_count += 1
            new_distance = result.distances[vertex] + weight
            
            if new_distance < result.distances[neighbor]:
                result.has_negative_cycle = True
                break
        
        if result.has_negative_cycle:
            break
    
    # Step 4: Reconstruct path if end vertex was specified
    if end is not None and not result.has_negative_cycle:
        if end in graph.get_vertices() and result.distances[end] != float('inf'):
            result.path = _reconstruct_path(result.predecessors, start, end)
            result.path_cost = result.distances[end]
        else:
            result.path = []
            result.path_cost = float('inf')
    
    return result


def detect_negative_cycle(graph: Graph) -> bool:
    """
    Detect if the graph contains any negative cycles.
    
    This function runs Bellman-Ford from an arbitrary vertex to detect
    negative cycles. If the graph is disconnected, it may not detect
    all negative cycles.
    
    Args:
        graph: The input graph
        
    Returns:
        True if a negative cycle is detected, False otherwise
    """
    vertices = graph.get_vertices()
    if not vertices:
        return False
    
    # Run Bellman-Ford from an arbitrary starting vertex
    start = next(iter(vertices))
    result = bellman_ford(graph, start)
    
    return result.has_negative_cycle


def bellman_ford_all_pairs(graph: Graph) -> Dict[int, BellmanFordResult]:
    """
    Compute shortest paths between all pairs of vertices using Bellman-Ford.
    
    This function runs Bellman-Ford from each vertex as a source.
    Note that this is less efficient than using Floyd-Warshall for
    all-pairs shortest paths, but it can handle negative edges.
    
    Time Complexity: O(V²E)
    
    Args:
        graph: The input graph
        
    Returns:
        Dictionary mapping each vertex to its BellmanFordResult
    """
    results = {}
    
    for vertex in graph.get_vertices():
        results[vertex] = bellman_ford(graph, vertex)
        
        # If we detect a negative cycle, we can stop early
        if results[vertex].has_negative_cycle:
            break
    
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


def get_shortest_path(graph: Graph, start: int, end: int) -> Tuple[List[int], float, bool]:
    """
    Convenience function to get the shortest path between two vertices.
    
    Args:
        graph: The input graph
        start: Source vertex
        end: Target vertex
        
    Returns:
        Tuple of (path, cost, has_negative_cycle)
    """
    result = bellman_ford(graph, start, end)
    return result.path, result.path_cost, result.has_negative_cycle
