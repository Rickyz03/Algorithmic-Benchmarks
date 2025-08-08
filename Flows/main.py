"""
Main script for maximum flow algorithms demonstration and comparison.

This script demonstrates the Ford-Fulkerson and Edmonds-Karp algorithms
on various graph types, with visualization capabilities using NetworkX
and Matplotlib.
"""

import sys
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from algorithms.ford_fulkerson import FordFulkerson
from algorithms.edmonds_karp import EdmondsKarp
from utils.graph_generator import FlowNetworkGenerator


class FlowNetworkVisualizer:
    """
    Visualizer for flow networks using NetworkX and Matplotlib.
    
    Provides methods to visualize graphs, flows, and algorithm comparisons.
    """
    
    @staticmethod
    def graph_to_networkx(graph: Dict[int, List[Tuple[int, int]]]) -> nx.DiGraph:
        """
        Convert adjacency list representation to NetworkX directed graph.
        
        Args:
            graph: Dictionary representation of the graph
            
        Returns:
            NetworkX DirectedGraph object
        """
        G = nx.DiGraph()
        
        for u in graph:
            for v, capacity in graph[u]:
                G.add_edge(u, v, capacity=capacity, flow=0)
        
        return G
    
    @staticmethod
    def visualize_graph_with_flow(graph: Dict[int, List[Tuple[int, int]]], 
                                 flow_edges: List[Tuple[int, int, int]], 
                                 source: int, sink: int, title: str = "Flow Network",
                                 figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize a flow network with flow values displayed.
        
        Args:
            graph: Original graph structure
            flow_edges: List of edges with flow values
            source: Source node
            sink: Sink node
            title: Plot title
            figsize: Figure size tuple
        """
        # Create NetworkX graph
        G = FlowNetworkVisualizer.graph_to_networkx(graph)
        
        # Add flow information to edges
        flow_dict = {(u, v): flow for u, v, flow in flow_edges}
        
        for u, v, data in G.edges(data=True):
            data['flow'] = flow_dict.get((u, v), 0)
        
        # Create figure and axis
        plt.figure(figsize=figsize)
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        
        # Draw nodes with different colors for source, sink, and intermediate
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            if node == source:
                node_colors.append('lightgreen')
                node_sizes.append(800)
            elif node == sink:
                node_colors.append('lightcoral')
                node_sizes.append(800)
            else:
                node_colors.append('lightblue')
                node_sizes.append(600)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Draw edges with different styles based on flow
        edge_colors = []
        edge_widths = []
        
        for u, v, data in G.edges(data=True):
            flow = data['flow']
            capacity = data['capacity']
            
            if flow > 0:
                # Scale edge width based on flow
                edge_widths.append(1 + 3 * (flow / max(1, capacity)))
                edge_colors.append('red')
            else:
                edge_widths.append(1)
                edge_colors.append('gray')
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                              width=edge_widths, alpha=0.7,
                              arrowsize=20, arrowstyle='->')
        
        # Draw edge labels (capacity/flow)
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            capacity = data['capacity']
            flow = data['flow']
            edge_labels[(u, v)] = f"{flow}/{capacity}"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                      markersize=12, label='Source'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                      markersize=12, label='Sink'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=12, label='Intermediate'),
            plt.Line2D([0], [0], color='red', linewidth=3, label='Flow > 0'),
            plt.Line2D([0], [0], color='gray', linewidth=1, label='No Flow')
        ]
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compare_algorithms(graph: Dict[int, List[Tuple[int, int]]], 
                          source: int, sink: int):
        """
        Compare Ford-Fulkerson and Edmonds-Karp algorithms visually.
        
        Args:
            graph: Graph to analyze
            source: Source node
            sink: Sink node
        """
        # Run both algorithms
        ff = FordFulkerson(graph)
        ek = EdmondsKarp(graph)
        
        ff_flow, ff_iter = ff.max_flow(source, sink)
        ek_flow, ek_iter = ek.max_flow(source, sink)
        
        ff_edges = ff.get_flow_edges()
        ek_edges = ek.get_flow_edges()
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Ford-Fulkerson visualization
        plt.sca(ax1)
        FlowNetworkVisualizer.visualize_graph_with_flow(
            graph, ff_edges, source, sink,
            title=f"Ford-Fulkerson\nMax Flow: {ff_flow}, Iterations: {ff_iter}"
        )
        
        # Edmonds-Karp visualization
        plt.sca(ax2)
        FlowNetworkVisualizer.visualize_graph_with_flow(
            graph, ek_edges, source, sink,
            title=f"Edmonds-Karp\nMax Flow: {ek_flow}, Iterations: {ek_iter}"
        )
        
        plt.suptitle("Algorithm Comparison", fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return ff_flow, ff_iter, ek_flow, ek_iter


def demonstrate_algorithms():
    """Demonstrate both algorithms on various graph types."""
    generator = FlowNetworkGenerator()
    visualizer = FlowNetworkVisualizer()
    
    print("Maximum Flow Algorithms Demonstration")
    print("=" * 50)
    
    # Example 1: Simple demonstration graph
    print("\n1. Simple Example Graph")
    simple_graph = {
        0: [(1, 10), (2, 8)],
        1: [(2, 5), (3, 5)],
        2: [(3, 10)],
        3: []
    }
    
    print("Running algorithms on simple graph...")
    ff = FordFulkerson(simple_graph)
    ek = EdmondsKarp(simple_graph)
    
    ff_flow, ff_iter = ff.max_flow(0, 3)
    ek_flow, ek_iter = ek.max_flow(0, 3)
    
    print(f"Ford-Fulkerson: Max Flow = {ff_flow}, Iterations = {ff_iter}")
    print(f"Edmonds-Karp: Max Flow = {ek_flow}, Iterations = {ek_iter}")
    
    # Visualize the simple graph
    visualizer.visualize_graph_with_flow(
        simple_graph, ff.get_flow_edges(), 0, 3,
        title="Simple Graph - Ford-Fulkerson Result"
    )
    
    # Example 2: Random graph
    print("\n2. Random Graph Example")
    random_graph = generator.random_graph(8, 15, max_capacity=20, seed=42)
    source, sink = 0, 7
    
    print(f"Generated random graph with {len(random_graph)} nodes")
    print("Comparing algorithms...")
    
    ff_flow, ff_iter, ek_flow, ek_iter = visualizer.compare_algorithms(
        random_graph, source, sink
    )
    
    print(f"Ford-Fulkerson: Max Flow = {ff_flow}, Iterations = {ff_iter}")
    print(f"Edmonds-Karp: Max Flow = {ek_flow}, Iterations = {ek_iter}")
    
    # Example 3: Bottleneck graph
    print("\n3. Bottleneck Graph Example")
    bottleneck_graph = generator.bottleneck_graph(
        num_layers=4, layer_size=3, bottleneck_capacity=2, other_capacity=15, seed=42
    )
    
    # Find source and sink for bottleneck graph
    all_nodes = set(bottleneck_graph.keys())
    for neighbors in bottleneck_graph.values():
        for neighbor, _ in neighbors:
            all_nodes.add(neighbor)
    
    source, sink = min(all_nodes), max(all_nodes)
    
    print(f"Bottleneck graph: source={source}, sink={sink}")
    
    ff_flow, ff_iter, ek_flow, ek_iter = visualizer.compare_algorithms(
        bottleneck_graph, source, sink
    )
    
    print(f"Ford-Fulkerson: Max Flow = {ff_flow}, Iterations = {ff_iter}")
    print(f"Edmonds-Karp: Max Flow = {ek_flow}, Iterations = {ek_iter}")
    
    # Example 4: Performance comparison on different graph sizes
    print("\n4. Performance Analysis")
    graph_sizes = [10, 15, 20]
    
    print("Graph Size | FF Time | FF Iter | EK Time | EK Iter")
    print("-" * 55)
    
    for size in graph_sizes:
        # Generate test graph
        test_graph = generator.random_graph(size, size * 2, seed=42)
        
        if not test_graph:
            continue
            
        nodes = list(test_graph.keys())
        source, sink = min(nodes), max(nodes)
        
        # Time Ford-Fulkerson
        import time
        
        ff = FordFulkerson(test_graph)
        start_time = time.perf_counter()
        ff_flow, ff_iter = ff.max_flow(source, sink)
        ff_time = time.perf_counter() - start_time
        
        # Time Edmonds-Karp
        ek = EdmondsKarp(test_graph)
        start_time = time.perf_counter()
        ek_flow, ek_iter = ek.max_flow(source, sink)
        ek_time = time.perf_counter() - start_time
        
        print(f"{size:^10} | {ff_time:7.4f} | {ff_iter:7d} | {ek_time:7.4f} | {ek_iter:7d}")


def interactive_demo():
    """Interactive demo allowing user to choose graph type."""
    generator = FlowNetworkGenerator()
    visualizer = FlowNetworkVisualizer()
    
    while True:
        print("\n" + "="*50)
        print("Interactive Maximum Flow Demo")
        print("="*50)
        print("Choose a graph type:")
        print("1. Random Graph")
        print("2. Linear Chain")
        print("3. Dense Graph")
        print("4. Bottleneck Graph")
        print("5. Bipartite Graph")
        print("6. Grid Graph")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '7':
            print("Goodbye!")
            break
        
        try:
            if choice == '1':
                # Random Graph
                nodes = int(input("Number of nodes (5-20): ") or "10")
                edges = int(input("Number of edges: ") or str(nodes * 2))
                graph = generator.random_graph(nodes, edges, seed=42)
                source, sink = 0, nodes - 1
                
            elif choice == '2':
                # Linear Chain
                nodes = int(input("Number of nodes (3-15): ") or "8")
                graph = generator.linear_chain(nodes, seed=42)
                source, sink = 0, nodes - 1
                
            elif choice == '3':
                # Dense Graph
                nodes = int(input("Number of nodes (5-15): ") or "8")
                prob = float(input("Connection probability (0.1-0.8): ") or "0.4")
                graph = generator.dense_graph(nodes, prob, seed=42)
                source, sink = 0, nodes - 1
                
            elif choice == '4':
                # Bottleneck Graph
                layers = int(input("Number of layers (3-6): ") or "4")
                layer_size = int(input("Layer size (2-5): ") or "3")
                graph = generator.bottleneck_graph(layers, layer_size, seed=42)
                source = 0
                sink = layers * layer_size - 1
                
            elif choice == '5':
                # Bipartite Graph
                left = int(input("Left partition size (2-8): ") or "4")
                right = int(input("Right partition size (2-8): ") or "4")
                graph = generator.bipartite_graph(left, right, seed=42)
                source = 0
                sink = left + right + 1
                
            elif choice == '6':
                # Grid Graph
                rows = int(input("Number of rows (2-6): ") or "4")
                cols = int(input("Number of columns (2-6): ") or "4")
                graph = generator.grid_graph(rows, cols, seed=42)
                source = 0
                sink = rows * cols - 1
                
            else:
                print("Invalid choice!")
                continue
            
            if not graph:
                print("Failed to generate graph!")
                continue
            
            print(f"\nGenerated graph with {len(graph)} source nodes")
            print("Running algorithm comparison...")
            
            # Compare algorithms and visualize
            ff_flow, ff_iter, ek_flow, ek_iter = visualizer.compare_algorithms(
                graph, source, sink
            )
            
            print(f"\nResults:")
            print(f"Ford-Fulkerson: Max Flow = {ff_flow}, Iterations = {ff_iter}")
            print(f"Edmonds-Karp: Max Flow = {ek_flow}, Iterations = {ek_iter}")
            
            if ff_iter != ek_iter:
                efficiency = "Edmonds-Karp" if ek_iter < ff_iter else "Ford-Fulkerson"
                print(f"More efficient (fewer iterations): {efficiency}")
            else:
                print("Both algorithms used the same number of iterations")
                
        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function with menu options."""
    print("Maximum Flow Algorithms - Ford-Fulkerson vs Edmonds-Karp")
    print("="*60)
    
    while True:
        print("\nMain Menu:")
        print("1. Run Algorithm Demonstration")
        print("2. Interactive Demo")
        print("3. Run Benchmarks")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            demonstrate_algorithms()
        elif choice == '2':
            interactive_demo()
        elif choice == '3':
            print("Running benchmarks... (this may take a while)")
            from benchmarks.run_benchmarks import main as run_benchmarks
            run_benchmarks()
        elif choice == '4':
            print("Thank you for using the Maximum Flow demonstration!")
            break
        else:
            print("Invalid choice! Please enter 1-4.")


if __name__ == '__main__':
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        main()
    except ImportError as e:
        print("Error: Required packages not found.")
        print("Please install required packages:")
        print("pip install matplotlib networkx")
        print(f"Missing: {e}")
        sys.exit(1)
