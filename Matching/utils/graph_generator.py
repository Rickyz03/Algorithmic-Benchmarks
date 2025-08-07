"""
Bipartite Graph Generator for Testing Matching Algorithms.

This module provides utilities to generate random bipartite graphs with various
properties for testing and benchmarking matching algorithms.

Author: Your Name  
Date: 2025
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import itertools


class BipartiteGraphGenerator:
    """Generator for bipartite graphs with various configurations."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator with optional random seed.
        
        Args:
            seed: Random seed for reproducible results
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_bipartite_graph(self, 
                                      left_size: int, 
                                      right_size: int,
                                      density: float = 0.5) -> List[Tuple[int, int]]:
        """
        Generate a random bipartite graph with specified density.
        
        Args:
            left_size: Number of vertices in left partition
            right_size: Number of vertices in right partition
            density: Edge density (probability of edge existence, 0.0 to 1.0)
            
        Returns:
            List of edges as (left_vertex, right_vertex) tuples
        """
        if not (0.0 <= density <= 1.0):
            raise ValueError("Density must be between 0.0 and 1.0")
        
        edges = []
        for left_v in range(left_size):
            for right_v in range(right_size):
                if random.random() < density:
                    edges.append((left_v, right_v))
        
        return edges
    
    def generate_complete_bipartite_graph(self, 
                                        left_size: int, 
                                        right_size: int) -> List[Tuple[int, int]]:
        """
        Generate a complete bipartite graph (all possible edges).
        
        Args:
            left_size: Number of vertices in left partition
            right_size: Number of vertices in right partition
            
        Returns:
            List of all possible edges
        """
        return [(left_v, right_v) for left_v in range(left_size) 
                for right_v in range(right_size)]
    
    def generate_regular_bipartite_graph(self,
                                       left_size: int,
                                       right_size: int,
                                       degree: int) -> List[Tuple[int, int]]:
        """
        Generate a regular bipartite graph where each vertex has the same degree.
        
        Args:
            left_size: Number of vertices in left partition
            right_size: Number of vertices in right partition
            degree: Degree of each vertex
            
        Returns:
            List of edges forming a regular bipartite graph
        """
        if degree > min(left_size, right_size):
            raise ValueError("Degree cannot exceed minimum partition size")
        
        edges = []
        
        # For each left vertex, randomly select 'degree' right vertices
        for left_v in range(left_size):
            right_neighbors = random.sample(range(right_size), degree)
            for right_v in right_neighbors:
                edges.append((left_v, right_v))
        
        return edges
    
    def generate_sparse_bipartite_graph(self,
                                      left_size: int,
                                      right_size: int,
                                      min_degree: int = 1,
                                      max_degree: int = 3) -> List[Tuple[int, int]]:
        """
        Generate a sparse bipartite graph with controlled degree distribution.
        
        Args:
            left_size: Number of vertices in left partition
            right_size: Number of vertices in right partition
            min_degree: Minimum degree for left vertices
            max_degree: Maximum degree for left vertices
            
        Returns:
            List of edges forming a sparse bipartite graph
        """
        edges = []
        
        for left_v in range(left_size):
            degree = random.randint(min_degree, min(max_degree, right_size))
            right_neighbors = random.sample(range(right_size), degree)
            for right_v in right_neighbors:
                edges.append((left_v, right_v))
        
        return edges
    
    def generate_random_weights(self, 
                              edges: List[Tuple[int, int]],
                              min_weight: float = 1.0,
                              max_weight: float = 10.0,
                              distribution: str = 'uniform') -> Dict[Tuple[int, int], float]:
        """
        Generate random weights for a list of edges.
        
        Args:
            edges: List of edges to assign weights to
            min_weight: Minimum weight value
            max_weight: Maximum weight value
            distribution: Weight distribution ('uniform', 'normal', 'exponential')
            
        Returns:
            Dictionary mapping edges to their weights
        """
        weights = {}
        
        for edge in edges:
            if distribution == 'uniform':
                weight = random.uniform(min_weight, max_weight)
            elif distribution == 'normal':
                mean = (min_weight + max_weight) / 2
                std = (max_weight - min_weight) / 4
                weight = max(min_weight, min(max_weight, random.normalvariate(mean, std)))
            elif distribution == 'exponential':
                # Exponential distribution scaled to range
                weight = min_weight + random.expovariate(1.0) * (max_weight - min_weight) / 3
                weight = min(max_weight, weight)
            else:
                raise ValueError(f"Unknown distribution: {distribution}")
            
            weights[edge] = weight
        
        return weights
    
    def edges_to_weight_matrix(self, 
                             edges: List[Tuple[int, int]],
                             weights: Dict[Tuple[int, int], float],
                             left_size: int,
                             right_size: int,
                             default_weight: float = 0.0) -> np.ndarray:
        """
        Convert edge list with weights to a weight matrix.
        
        Args:
            edges: List of edges
            weights: Dictionary of edge weights
            left_size: Number of left vertices
            right_size: Number of right vertices
            default_weight: Weight for non-existing edges
            
        Returns:
            Weight matrix as numpy array
        """
        if left_size != right_size:
            # For Hungarian algorithm, we need a square matrix
            # Pad with zeros to make it square
            size = max(left_size, right_size)
            matrix = np.full((size, size), default_weight, dtype=float)
        else:
            matrix = np.full((left_size, right_size), default_weight, dtype=float)
        
        for edge in edges:
            left_v, right_v = edge
            if edge in weights:
                matrix[left_v][right_v] = weights[edge]
        
        return matrix
    
    def generate_test_cases(self) -> List[Dict]:
        """
        Generate a suite of test cases for algorithm benchmarking.
        
        Returns:
            List of test case dictionaries with graph parameters
        """
        test_cases = []
        
        # Small test cases
        test_cases.append({
            'name': 'Small Dense',
            'left_size': 5,
            'right_size': 5,
            'type': 'random',
            'density': 0.8
        })
        
        test_cases.append({
            'name': 'Small Sparse',
            'left_size': 5,
            'right_size': 5,
            'type': 'sparse',
            'min_degree': 1,
            'max_degree': 2
        })
        
        # Medium test cases
        for size in [10, 20, 30]:
            test_cases.append({
                'name': f'Medium {size}x{size} Dense',
                'left_size': size,
                'right_size': size,
                'type': 'random',
                'density': 0.6
            })
            
            test_cases.append({
                'name': f'Medium {size}x{size} Sparse',
                'left_size': size,
                'right_size': size,
                'type': 'sparse',
                'min_degree': 2,
                'max_degree': 4
            })
        
        # Large test cases
        for size in [50, 100]:
            test_cases.append({
                'name': f'Large {size}x{size} Medium',
                'left_size': size,
                'right_size': size,
                'type': 'random',
                'density': 0.3
            })
        
        # Asymmetric cases
        test_cases.append({
            'name': 'Asymmetric 10x20',
            'left_size': 10,
            'right_size': 20,
            'type': 'random',
            'density': 0.4
        })
        
        test_cases.append({
            'name': 'Asymmetric 20x10',
            'left_size': 20,
            'right_size': 10,
            'type': 'random',
            'density': 0.4
        })
        
        return test_cases
    
    def generate_graph_from_test_case(self, test_case: Dict) -> Tuple[List[Tuple[int, int]], Dict]:
        """
        Generate a graph from a test case specification.
        
        Args:
            test_case: Test case dictionary with graph parameters
            
        Returns:
            Tuple of (edges, weights) where weights may be empty for unweighted graphs
        """
        if test_case['type'] == 'random':
            edges = self.generate_random_bipartite_graph(
                test_case['left_size'],
                test_case['right_size'],
                test_case['density']
            )
        elif test_case['type'] == 'sparse':
            edges = self.generate_sparse_bipartite_graph(
                test_case['left_size'],
                test_case['right_size'],
                test_case.get('min_degree', 1),
                test_case.get('max_degree', 3)
            )
        elif test_case['type'] == 'complete':
            edges = self.generate_complete_bipartite_graph(
                test_case['left_size'],
                test_case['right_size']
            )
        elif test_case['type'] == 'regular':
            edges = self.generate_regular_bipartite_graph(
                test_case['left_size'],
                test_case['right_size'],
                test_case.get('degree', 2)
            )
        else:
            raise ValueError(f"Unknown test case type: {test_case['type']}")
        
        # Generate weights if requested
        weights = {}
        if test_case.get('weighted', True):
            weights = self.generate_random_weights(edges)
        
        return edges, weights


def create_example_graphs() -> Dict[str, Tuple[List[Tuple[int, int]], Dict]]:
    """
    Create a collection of example graphs for demonstration purposes.
    
    Returns:
        Dictionary mapping graph names to (edges, weights) tuples
    """
    generator = BipartiteGraphGenerator(seed=42)  # Fixed seed for reproducibility
    examples = {}
    
    # Perfect matching example
    perfect_edges = [(0, 0), (1, 1), (2, 2)]
    perfect_weights = generator.generate_random_weights(perfect_edges, 1, 5)
    examples['perfect_matching'] = (perfect_edges, perfect_weights)
    
    # No perfect matching example  
    no_perfect_edges = [(0, 0), (1, 0), (2, 0)]
    no_perfect_weights = generator.generate_random_weights(no_perfect_edges, 1, 5)
    examples['no_perfect_matching'] = (no_perfect_edges, no_perfect_weights)
    
    # Dense graph
    dense_edges = generator.generate_random_bipartite_graph(4, 4, 0.75)
    dense_weights = generator.generate_random_weights(dense_edges, 1, 10)
    examples['dense_graph'] = (dense_edges, dense_weights)
    
    # Sparse graph
    sparse_edges = generator.generate_sparse_bipartite_graph(6, 6, 1, 2)
    sparse_weights = generator.generate_random_weights(sparse_edges, 1, 10)
    examples['sparse_graph'] = (sparse_edges, sparse_weights)
    
    return examples


if __name__ == "__main__":
    # Demonstration of graph generation capabilities
    generator = BipartiteGraphGenerator(seed=123)
    
    print("Bipartite Graph Generator Demo")
    print("=" * 40)
    
    # Generate different types of graphs
    print("\n1. Random Bipartite Graph (5x5, density=0.4):")
    random_edges = generator.generate_random_bipartite_graph(5, 5, 0.4)
    print(f"Edges: {random_edges}")
    print(f"Number of edges: {len(random_edges)}")
    
    print("\n2. Sparse Bipartite Graph (4x4, degree 1-2):")
    sparse_edges = generator.generate_sparse_bipartite_graph(4, 4, 1, 2)
    print(f"Edges: {sparse_edges}")
    
    print("\n3. Random Weights for Edges:")
    weights = generator.generate_random_weights(sparse_edges, 1.0, 10.0)
    for edge, weight in weights.items():
        print(f"Edge {edge}: weight {weight:.2f}")
    
    print("\n4. Weight Matrix:")
    weight_matrix = generator.edges_to_weight_matrix(sparse_edges, weights, 4, 4)
    print(weight_matrix)
    
    print("\n5. Test Cases Available:")
    test_cases = generator.generate_test_cases()
    for i, test_case in enumerate(test_cases[:5]):  # Show first 5
        print(f"{i+1}. {test_case['name']}: {test_case['left_size']}x{test_case['right_size']} "
              f"({test_case['type']})")
    print(f"... and {len(test_cases)-5} more test cases.")
