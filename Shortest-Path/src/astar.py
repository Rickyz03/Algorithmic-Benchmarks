"""
A* search algorithm implementation.

This module implements the A* search algorithm for finding optimal paths
in graphs using heuristic functions. It's particularly effective for
pathfinding in grids and spatial graphs.
"""

import heapq
from typing import Dict, List, Optional, Tuple, Callable, Union
from graph import Graph, GridGraph


class AStarResult:
    """Container for A* algorithm results and statistics."""
    
    def __init__(self):
        self.path: List[Union[int, Tuple[int, int]]] = []
        self.path_cost: float = float('inf')
        self.operations_count: int = 0
        self.vertices_explored: int = 0
        self.vertices_expanded: int = 0


def astar_graph(graph: Graph, start: int, end: int, 
                heuristic: Callable[[int, int], float]) -> AStarResult:
    """
    Execute A* search on a general graph.
    
    A* is an informed search algorithm that uses a heuristic function to guide
    the search towards the goal. It maintains a priority queue ordered by
    f(n) = g(n) + h(n), where g(n) is the cost from start to n, and h(n) is
    the heuristic estimate from n to the goal.
    
    For A* to find optimal solutions, the heuristic must be admissible
    (never overestimate the true cost) and consistent (satisfy the triangle
    inequality).
    
    Time Complexity: O(b^d) where b is branching factor and d is depth
    Space Complexity: O(b^d)
    
    Args:
        graph: The input graph
        start: Source vertex
        end: Target vertex
        heuristic: Heuristic function h(current, goal) -> estimate
        
    Returns:
        AStarResult containing the optimal path and performance metrics
        
    Raises:
        ValueError: If start or end vertices are not in the graph
    """
    if start not in graph.get_vertices():
        raise ValueError(f"Start vertex {start} not found in graph")
    if end not in graph.get_vertices():
        raise ValueError(f"End vertex {end} not found in graph")
    
    result = AStarResult()
    
    # Priority queue: (f_score, g_score, vertex)
    open_set = [(heuristic(start, end), 0.0, start)]
    
    # Keep track of the best g_score for each vertex
    g_score: Dict[int, float] = {start: 0.0}
    
    # Keep track of the path
    came_from: Dict[int, int] = {}
    
    # Set of vertices already evaluated
    closed_set = set()
    
    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        result.operations_count += 1
        
        # Skip if we've already processed this vertex
        if current in closed_set:
            continue
        
        closed_set.add(current)
        result.vertices_expanded += 1
        
        # Check if we've reached the goal
        if current == end:
            result.path = _reconstruct_path_graph(came_from, start, end)
            result.path_cost = current_g
            return result
        
        # Explore neighbors
        for neighbor, edge_weight in graph.get_neighbors(current):
            if neighbor in closed_set:
                continue
            
            result.vertices_explored += 1
            tentative_g = current_g + edge_weight
            
            # If this path to neighbor is better than any previous one
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    
    # No path found
    return result


def astar_grid(grid: GridGraph, start: Tuple[int, int], 
               end: Tuple[int, int]) -> AStarResult:
    """
    Execute A* search on a 2D grid using Manhattan distance heuristic.
    
    This specialized version of A* is optimized for grid-based pathfinding.
    It uses the Manhattan distance as the heuristic function, which is
    admissible for grid worlds with 4-directional movement.
    
    Args:
        grid: The grid graph
        start: Starting position as (x, y) tuple
        end: Target position as (x, y) tuple
        
    Returns:
        AStarResult containing the optimal path and performance metrics
        
    Raises:
        ValueError: If start or end positions are invalid
    """
    if (start[0] < 0 or start[0] >= grid.width or 
        start[1] < 0 or start[1] >= grid.height or
        start in grid.obstacles):
        raise ValueError(f"Invalid start position: {start}")
    
    if (end[0] < 0 or end[0] >= grid.width or 
        end[1] < 0 or end[1] >= grid.height or
        end in grid.obstacles):
        raise ValueError(f"Invalid end position: {end}")
    
    result = AStarResult()
    
    # Priority queue: (f_score, g_score, position)
    open_set = [(grid.manhattan_distance(start, end), 0.0, start)]
    
    # Keep track of the best g_score for each position
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    
    # Keep track of the path
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    
    # Set of positions already evaluated
    closed_set = set()
    
    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        result.operations_count += 1
        
        # Skip if we've already processed this position
        if current in closed_set:
            continue
        
        closed_set.add(current)
        result.vertices_expanded += 1
        
        # Check if we've reached the goal
        if current == end:
            result.path = _reconstruct_path_grid(came_from, start, end)
            result.path_cost = current_g
            return result
        
        # Explore neighbors
        for neighbor, move_cost in grid.get_neighbors(current):
            if neighbor in closed_set:
                continue
            
            result.vertices_explored += 1
            tentative_g = current_g + move_cost
            
            # If this path to neighbor is better than any previous one
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + grid.manhattan_distance(neighbor, end)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    
    # No path found
    return result


def _reconstruct_path_graph(came_from: Dict[int, int], 
                           start: int, end: int) -> List[int]:
    """
    Reconstruct the path from came_from dictionary for graph search.
    
    Args:
        came_from: Dictionary mapping each vertex to its predecessor
        start: Source vertex
        end: Target vertex
        
    Returns:
        List of vertices representing the optimal path
    """
    path = []
    current = end
    
    while current in came_from:
        path.append(current)
        current = came_from[current]
    
    path.append(start)
    path.reverse()
    
    return path


def _reconstruct_path_grid(came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                          start: Tuple[int, int], 
                          end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from came_from dictionary for grid search.
    
    Args:
        came_from: Dictionary mapping each position to its predecessor
        start: Source position
        end: Target position
        
    Returns:
        List of positions representing the optimal path
    """
    path = []
    current = end
    
    while current in came_from:
        path.append(current)
        current = came_from[current]
    
    path.append(start)
    path.reverse()
    
    return path


# Heuristic functions for general graphs

def euclidean_distance_2d(pos1: Tuple[float, float], 
                         pos2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    This heuristic is admissible for graphs where vertices represent
    2D coordinates and edge weights represent Euclidean distances.
    
    Args:
        pos1: First position as (x, y) tuple
        pos2: Second position as (x, y) tuple
        
    Returns:
        Euclidean distance between the points
    """
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5


def manhattan_distance_2d(pos1: Tuple[float, float], 
                         pos2: Tuple[float, float]) -> float:
    """
    Calculate Manhattan distance between two 2D points.
    
    This heuristic is admissible for grid-like graphs where movement
    is restricted to orthogonal directions.
    
    Args:
        pos1: First position as (x, y) tuple
        pos2: Second position as (x, y) tuple
        
    Returns:
        Manhattan distance between the points
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def zero_heuristic(current: int, goal: int) -> float:
    """
    Zero heuristic function that always returns 0.
    
    Using this heuristic makes A* equivalent to Dijkstra's algorithm.
    It's useful for comparison purposes or when no domain knowledge
    is available for creating a meaningful heuristic.
    
    Args:
        current: Current vertex
        goal: Goal vertex
        
    Returns:
        Always returns 0.0
    """
    return 0.0


def get_shortest_path_graph(graph: Graph, start: int, end: int,
                           heuristic: Callable[[int, int], float]) -> Tuple[List[int], float]:
    """
    Convenience function to get the shortest path between two vertices in a graph.
    
    Args:
        graph: The input graph
        start: Source vertex
        end: Target vertex
        heuristic: Heuristic function
        
    Returns:
        Tuple of (path, cost) where path is a list of vertices
    """
    result = astar_graph(graph, start, end, heuristic)
    return result.path, result.path_cost


def get_shortest_path_grid(grid: GridGraph, start: Tuple[int, int], 
                          end: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
    """
    Convenience function to get the shortest path between two positions in a grid.
    
    Args:
        grid: The grid graph
        start: Starting position
        end: Target position
        
    Returns:
        Tuple of (path, cost) where path is a list of positions
    """
    result = astar_grid(grid, start, end)
    return result.path, result.path_cost
