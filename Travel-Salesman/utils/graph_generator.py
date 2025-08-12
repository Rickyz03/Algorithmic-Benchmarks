"""
Graph generator utilities for creating TSP instances.

This module provides functions to generate various types of graphs
suitable for testing TSP algorithms, including random coordinate-based
graphs and metric graphs that satisfy the triangle inequality.
"""

import numpy as np
import random
from typing import Tuple, List
import math


def generate_random_coordinates(n: int, seed: int = None) -> List[Tuple[float, float]]:
    """
    Generate n random coordinates in a 2D plane.
    
    Args:
        n: Number of cities/coordinates to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of (x, y) coordinate tuples
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    coordinates = []
    for _ in range(n):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        coordinates.append((x, y))
    
    return coordinates


def euclidean_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two coordinates.
    
    Args:
        coord1: First coordinate (x1, y1)
        coord2: Second coordinate (x2, y2)
        
    Returns:
        Euclidean distance between the coordinates
    """
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def coordinates_to_distance_matrix(coordinates: List[Tuple[float, float]]) -> np.ndarray:
    """
    Convert coordinate list to distance matrix using Euclidean distance.
    
    Args:
        coordinates: List of (x, y) coordinate tuples
        
    Returns:
        Symmetric distance matrix where matrix[i][j] is the distance
        between city i and city j
    """
    n = len(coordinates)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = euclidean_distance(coordinates[i], coordinates[j])
    
    return matrix


def generate_random_metric_matrix(n: int, max_distance: float = 100.0, seed: int = None) -> np.ndarray:
    """
    Generate a random symmetric distance matrix that satisfies the triangle inequality.
    
    This function generates a metric space by using the Floyd-Warshall algorithm
    to ensure the triangle inequality is satisfied.
    
    Args:
        n: Number of cities
        max_distance: Maximum edge weight in the initial random graph
        seed: Random seed for reproducibility
        
    Returns:
        Symmetric distance matrix satisfying triangle inequality
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate initial random symmetric matrix
    matrix = np.random.uniform(1, max_distance, (n, n))
    
    # Make symmetric
    matrix = (matrix + matrix.T) / 2
    
    # Set diagonal to zero
    np.fill_diagonal(matrix, 0)
    
    # Apply Floyd-Warshall to ensure triangle inequality
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if matrix[i][k] + matrix[k][j] < matrix[i][j]:
                    matrix[i][j] = matrix[i][k] + matrix[k][j]
    
    return matrix


def generate_euclidean_tsp_instance(n: int, seed: int = None) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Generate a complete TSP instance based on Euclidean distances.
    
    Args:
        n: Number of cities
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - Distance matrix (numpy array)
        - List of coordinates for visualization
    """
    coordinates = generate_random_coordinates(n, seed)
    distance_matrix = coordinates_to_distance_matrix(coordinates)
    
    return distance_matrix, coordinates


def generate_metric_tsp_instance(n: int, max_distance: float = 100.0, seed: int = None) -> np.ndarray:
    """
    Generate a complete TSP instance with random metric distances.
    
    Args:
        n: Number of cities
        max_distance: Maximum distance between any two cities
        seed: Random seed for reproducibility
        
    Returns:
        Distance matrix satisfying triangle inequality
    """
    return generate_random_metric_matrix(n, max_distance, seed)


def validate_triangle_inequality(distance_matrix: np.ndarray) -> bool:
    """
    Validate that a distance matrix satisfies the triangle inequality.
    
    Args:
        distance_matrix: Square distance matrix to validate
        
    Returns:
        True if triangle inequality is satisfied, False otherwise
    """
    n = distance_matrix.shape[0]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j] + 1e-10:
                    return False
    
    return True


if __name__ == "__main__":
    # Example usage and testing
    print("Testing graph generation utilities...")
    
    # Test Euclidean instance generation
    n = 5
    dist_matrix, coords = generate_euclidean_tsp_instance(n, seed=42)
    print(f"Generated {n}x{n} Euclidean distance matrix:")
    print(dist_matrix)
    print(f"Coordinates: {coords}")
    print(f"Triangle inequality satisfied: {validate_triangle_inequality(dist_matrix)}")
    
    # Test metric instance generation
    metric_matrix = generate_metric_tsp_instance(n, seed=42)
    print(f"\nGenerated {n}x{n} metric distance matrix:")
    print(metric_matrix)
    print(f"Triangle inequality satisfied: {validate_triangle_inequality(metric_matrix)}")
