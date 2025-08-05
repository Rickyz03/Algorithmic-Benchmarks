"""
Graph representation and utility functions for shortest path algorithms.

This module provides a flexible Graph class that supports both directed and
undirected graphs with weighted edges, along with utilities for generating
test instances and validating paths.
"""

from typing import Dict, List, Tuple, Set, Optional
import random
import math


class Graph:
    """
    A weighted graph implementation supporting both directed and undirected graphs.
    
    The graph is represented using an adjacency list where each vertex maps to
    a list of (neighbor, weight) tuples.
    """
    
    def __init__(self, directed: bool = False):
        """
        Initialize an empty graph.
        
        Args:
            directed: If True, creates a directed graph. Otherwise, undirected.
        """
        self.directed = directed
        self.vertices: Set[int] = set()
        self.edges: Dict[int, List[Tuple[int, float]]] = {}
        
    def add_vertex(self, vertex: int) -> None:
        """Add a vertex to the graph."""
        self.vertices.add(vertex)
        if vertex not in self.edges:
            self.edges[vertex] = []
    
    def add_edge(self, source: int, target: int, weight: float) -> None:
        """
        Add a weighted edge to the graph.
        
        Args:
            source: Source vertex
            target: Target vertex  
            weight: Edge weight
        """
        self.add_vertex(source)
        self.add_vertex(target)
        
        self.edges[source].append((target, weight))
        
        # For undirected graphs, add the reverse edge
        if not self.directed:
            self.edges[target].append((source, weight))
    
    def get_neighbors(self, vertex: int) -> List[Tuple[int, float]]:
        """Get all neighbors of a vertex with their edge weights."""
        return self.edges.get(vertex, [])
    
    def get_vertices(self) -> Set[int]:
        """Get all vertices in the graph."""
        return self.vertices.copy()
    
    def get_edge_count(self) -> int:
        """Get the total number of edges in the graph."""
        total_edges = sum(len(neighbors) for neighbors in self.edges.values())
        return total_edges if self.directed else total_edges // 2
    
    def has_negative_edges(self) -> bool:
        """Check if the graph contains any negative weight edges."""
        for neighbors in self.edges.values():
            for _, weight in neighbors:
                if weight < 0:
                    return True
        return False


class GridGraph:
    """
    A specialized graph class for 2D grid-based pathfinding.
    
    This class is optimized for A* search and provides Manhattan distance
    heuristic calculations.
    """
    
    def __init__(self, width: int, height: int, obstacles: Set[Tuple[int, int]] = None):
        """
        Initialize a grid graph.
        
        Args:
            width: Grid width
            height: Grid height
            obstacles: Set of (x, y) coordinates that are blocked
        """
        self.width = width
        self.height = height
        self.obstacles = obstacles or set()
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get valid neighboring positions with their movement costs.
        
        Args:
            pos: Current position as (x, y) tuple
            
        Returns:
            List of ((x, y), cost) tuples for valid neighbors
        """
        x, y = pos
        neighbors = []
        
        # Define 4-directional movement (up, down, left, right)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            if (0 <= new_x < self.width and 
                0 <= new_y < self.height and 
                (new_x, new_y) not in self.obstacles):
                neighbors.append(((new_x, new_y), 1.0))
        
        return neighbors
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def generate_random_graph(num_vertices: int, edge_probability: float, 
                         min_weight: float = 1.0, max_weight: float = 10.0,
                         allow_negative: bool = False, directed: bool = False) -> Graph:
    """
    Generate a random graph with specified parameters.
    
    Args:
        num_vertices: Number of vertices in the graph
        edge_probability: Probability of creating an edge between any two vertices
        min_weight: Minimum edge weight
        max_weight: Maximum edge weight
        allow_negative: If True, allows negative edge weights
        directed: If True, creates a directed graph
        
    Returns:
        Generated Graph instance
    """
    graph = Graph(directed=directed)
    
    # Add all vertices
    for i in range(num_vertices):
        graph.add_vertex(i)
    
    # Add random edges
    for i in range(num_vertices):
        for j in range(i + 1 if not directed else 0, num_vertices):
            if i != j and random.random() < edge_probability:
                weight = random.uniform(min_weight, max_weight)
                if allow_negative and random.random() < 0.2:  # 20% chance for negative weight
                    weight = -weight
                
                graph.add_edge(i, j, weight)
    
    return graph


def generate_grid_with_obstacles(width: int, height: int, 
                               obstacle_probability: float = 0.2) -> GridGraph:
    """
    Generate a grid graph with random obstacles.
    
    Args:
        width: Grid width
        height: Grid height
        obstacle_probability: Probability of each cell being an obstacle
        
    Returns:
        Generated GridGraph instance
    """
    obstacles = set()
    
    for x in range(width):
        for y in range(height):
            if random.random() < obstacle_probability:
                obstacles.add((x, y))
    
    return GridGraph(width, height, obstacles)


def validate_path(graph: Graph, path: List[int], start: int, end: int) -> Tuple[bool, float]:
    """
    Validate a path in the graph and calculate its total cost.
    
    Args:
        graph: The graph to validate against
        path: List of vertices representing the path
        start: Expected start vertex
        end: Expected end vertex
        
    Returns:
        Tuple of (is_valid, total_cost)
    """
    if not path:
        return False, float('inf')
    
    if path[0] != start or path[-1] != end:
        return False, float('inf')
    
    total_cost = 0.0
    
    for i in range(len(path) - 1):
        current = path[i]
        next_vertex = path[i + 1]
        
        # Find the edge weight
        neighbors = graph.get_neighbors(current)
        edge_found = False
        
        for neighbor, weight in neighbors:
            if neighbor == next_vertex:
                total_cost += weight
                edge_found = True
                break
        
        if not edge_found:
            return False, float('inf')
    
    return True, total_cost
