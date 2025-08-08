"""
Graph generator utilities for creating test instances for maximum flow algorithms.

This module provides functions to generate various types of flow networks
including random graphs, linear chains, dense graphs, and bottleneck scenarios.
"""

import random
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class FlowNetworkGenerator:
    """
    Generator for flow network test instances.
    
    Provides methods to create different types of flow networks
    suitable for testing maximum flow algorithms.
    """
    
    @staticmethod
    def random_graph(num_nodes: int, num_edges: int, max_capacity: int = 100, 
                    seed: int = None) -> Dict[int, List[Tuple[int, int]]]:
        """
        Generate a random directed graph with random capacities.
        
        Args:
            num_nodes: Number of nodes in the graph
            num_edges: Number of edges to generate
            max_capacity: Maximum capacity for any edge
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary representing the graph with adjacency lists
        """
        if seed is not None:
            random.seed(seed)
            
        if num_edges > num_nodes * (num_nodes - 1):
            raise ValueError("Too many edges for the given number of nodes")
            
        graph = defaultdict(list)
        edges_added = set()
        
        # Ensure connectivity by creating a path from 0 to num_nodes-1
        for i in range(num_nodes - 1):
            capacity = random.randint(1, max_capacity)
            graph[i].append((i + 1, capacity))
            edges_added.add((i, i + 1))
            
        # Add remaining edges randomly
        edges_to_add = num_edges - (num_nodes - 1)
        
        while len(edges_added) < num_edges and edges_to_add > 0:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            
            # Avoid self-loops and duplicate edges
            if u != v and (u, v) not in edges_added:
                capacity = random.randint(1, max_capacity)
                graph[u].append((v, capacity))
                edges_added.add((u, v))
                edges_to_add -= 1
                
        return dict(graph)
    
    @staticmethod
    def linear_chain(num_nodes: int, min_capacity: int = 1, 
                    max_capacity: int = 100, seed: int = None) -> Dict[int, List[Tuple[int, int]]]:
        """
        Generate a linear chain graph (path graph).
        
        Creates a simple path from node 0 to node num_nodes-1 with
        random capacities on each edge.
        
        Args:
            num_nodes: Number of nodes in the chain
            min_capacity: Minimum capacity for edges
            max_capacity: Maximum capacity for edges  
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary representing the linear chain graph
        """
        if seed is not None:
            random.seed(seed)
            
        graph = defaultdict(list)
        
        for i in range(num_nodes - 1):
            capacity = random.randint(min_capacity, max_capacity)
            graph[i].append((i + 1, capacity))
            
        return dict(graph)
    
    @staticmethod
    def dense_graph(num_nodes: int, connection_prob: float = 0.3, 
                   max_capacity: int = 100, seed: int = None) -> Dict[int, List[Tuple[int, int]]]:
        """
        Generate a dense random graph using connection probability.
        
        Args:
            num_nodes: Number of nodes in the graph
            connection_prob: Probability of connection between any two nodes
            max_capacity: Maximum capacity for any edge
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary representing the dense graph
        """
        if seed is not None:
            random.seed(seed)
            
        graph = defaultdict(list)
        
        # First ensure connectivity with a spanning tree
        nodes = list(range(num_nodes))
        random.shuffle(nodes)
        
        for i in range(num_nodes - 1):
            capacity = random.randint(1, max_capacity)
            graph[nodes[i]].append((nodes[i + 1], capacity))
        
        # Add additional edges based on probability
        for u in range(num_nodes):
            for v in range(num_nodes):
                if u != v and random.random() < connection_prob:
                    # Check if edge already exists
                    existing_neighbors = [neighbor for neighbor, _ in graph[u]]
                    if v not in existing_neighbors:
                        capacity = random.randint(1, max_capacity)
                        graph[u].append((v, capacity))
                        
        return dict(graph)
    
    @staticmethod
    def bottleneck_graph(num_layers: int, layer_size: int, 
                        bottleneck_capacity: int = 1, 
                        other_capacity: int = 100, seed: int = None) -> Dict[int, List[Tuple[int, int]]]:
        """
        Generate a graph with a bottleneck structure.
        
        Creates layers of nodes where one layer acts as a bottleneck
        with limited capacity edges.
        
        Args:
            num_layers: Number of layers in the graph
            layer_size: Number of nodes per layer
            bottleneck_capacity: Capacity for bottleneck edges
            other_capacity: Capacity for non-bottleneck edges
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary representing the bottleneck graph
        """
        if seed is not None:
            random.seed(seed)
            
        graph = defaultdict(list)
        bottleneck_layer = num_layers // 2  # Middle layer as bottleneck
        
        for layer in range(num_layers - 1):
            for i in range(layer_size):
                current_node = layer * layer_size + i
                
                # Connect to all nodes in next layer
                for j in range(layer_size):
                    next_node = (layer + 1) * layer_size + j
                    
                    # Use bottleneck capacity for middle layer
                    if layer == bottleneck_layer:
                        capacity = bottleneck_capacity
                    else:
                        capacity = other_capacity
                        
                    graph[current_node].append((next_node, capacity))
                    
        return dict(graph)
    
    @staticmethod
    def bipartite_graph(left_size: int, right_size: int, 
                       connection_prob: float = 0.5, max_capacity: int = 100,
                       seed: int = None) -> Dict[int, List[Tuple[int, int]]]:
        """
        Generate a bipartite graph with source and sink connections.
        
        Creates a bipartite graph with additional source (node 0) and sink
        (node left_size + right_size + 1) for maximum flow testing.
        
        Args:
            left_size: Number of nodes in left partition
            right_size: Number of nodes in right partition  
            connection_prob: Probability of edge between left and right nodes
            max_capacity: Maximum capacity for any edge
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary representing the bipartite flow network
        """
        if seed is not None:
            random.seed(seed)
            
        graph = defaultdict(list)
        
        # Node 0 is source, nodes 1 to left_size are left partition
        # Nodes left_size+1 to left_size+right_size are right partition
        # Node left_size+right_size+1 is sink
        
        source = 0
        sink = left_size + right_size + 1
        
        # Connect source to all left nodes
        for i in range(1, left_size + 1):
            capacity = random.randint(1, max_capacity)
            graph[source].append((i, capacity))
            
        # Connect left nodes to right nodes based on probability
        for i in range(1, left_size + 1):
            for j in range(left_size + 1, left_size + right_size + 1):
                if random.random() < connection_prob:
                    capacity = random.randint(1, max_capacity)
                    graph[i].append((j, capacity))
                    
        # Connect all right nodes to sink
        for j in range(left_size + 1, left_size + right_size + 1):
            capacity = random.randint(1, max_capacity)
            graph[j].append((sink, capacity))
            
        return dict(graph)
    
    @staticmethod
    def grid_graph(rows: int, cols: int, max_capacity: int = 100, 
                  seed: int = None) -> Dict[int, List[Tuple[int, int]]]:
        """
        Generate a grid graph with flow from top-left to bottom-right.
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            max_capacity: Maximum capacity for any edge
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary representing the grid graph
        """
        if seed is not None:
            random.seed(seed)
            
        graph = defaultdict(list)
        
        def node_id(r: int, c: int) -> int:
            return r * cols + c
            
        # Add horizontal and vertical edges
        for r in range(rows):
            for c in range(cols):
                current = node_id(r, c)
                
                # Right edge
                if c + 1 < cols:
                    right = node_id(r, c + 1)
                    capacity = random.randint(1, max_capacity)
                    graph[current].append((right, capacity))
                    
                # Down edge
                if r + 1 < rows:
                    down = node_id(r + 1, c)
                    capacity = random.randint(1, max_capacity)
                    graph[current].append((down, capacity))
                    
        return dict(graph)
