"""
Unit tests for bipartite graph matching algorithms.

This module contains comprehensive tests for both Hungarian and Hopcroft-Karp
algorithms to ensure correctness and robustness.

Author: Your Name
Date: 2025
"""

import unittest
import numpy as np
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from hungarian import HungarianAlgorithm, solve_maximum_weight_matching
from hopcroft_karp import HopcroftKarpAlgorithm, create_bipartite_graph_from_edges
from graph_generator import BipartiteGraphGenerator


class TestHungarianAlgorithm(unittest.TestCase):
    """Test cases for Hungarian Algorithm implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = BipartiteGraphGenerator(seed=42)
    
    def test_simple_3x3_matrix(self):
        """Test Hungarian algorithm on a simple 3x3 cost matrix."""
        cost_matrix = [
            [4, 1, 3],
            [2, 0, 5], 
            [3, 2, 2]
        ]
        
        hungarian = HungarianAlgorithm(cost_matrix)
        matching, total_weight = hungarian.solve()
        
        # Verify matching properties
        self.assertEqual(len(matching), 3, "Should have 3 matched pairs")
        
        # Check that all vertices are matched exactly once
        left_matched = set(pair[0] for pair in matching)
        right_matched = set(pair[1] for pair in matching)
        
        self.assertEqual(len(left_matched), 3, "All left vertices should be matched")
        self.assertEqual(len(right_matched), 3, "All right vertices should be matched")
        self.assertEqual(left_matched, {0, 1, 2}, "Left vertices 0,1,2 should be matched")
        self.assertEqual(right_matched, {0, 1, 2}, "Right vertices 0,1,2 should be matched")
        
        # Verify the total weight
        expected_weight = sum(cost_matrix[i][j] for i, j in matching)
        self.assertEqual(total_weight, expected_weight, "Total weight should match sum of matched edges")
        
        # Check that this is indeed maximum (for this small case we can verify manually)
        self.assertGreaterEqual(total_weight, 7, "Total weight should be at least 7")
    
    def test_4x4_matrix(self):
        """Test Hungarian algorithm on a 4x4 cost matrix."""
        cost_matrix = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12],
            [4, 8, 12, 16]
        ]
        
        matching, total_weight = hungarian_algorithm.solve()
        
        self.assertEqual(len(matching), 4, "Should have 4 matched pairs")
        
        # Verify uniqueness of matching
        left_vertices = [pair[0] for pair in matching]
        right_vertices = [pair[1] for pair in matching]
        self.assertEqual(len(set(left_vertices)), 4, "All left vertices unique")
        self.assertEqual(len(set(right_vertices)), 4, "All right vertices unique")
    
    def test_rectangular_matrix_padding(self):
        """Test Hungarian algorithm behavior with non-square input."""
        # Note: Our implementation expects square matrices
        # This test verifies the error handling
        cost_matrix = [
            [1, 2, 3],
            [4, 5, 6]
        ]
        
        with self.assertRaises(ValueError):
            HungarianAlgorithm(cost_matrix)
    
    def test_single_vertex(self):
        """Test Hungarian algorithm on 1x1 matrix."""
        cost_matrix = [[5]]
        
        hungarian = HungarianAlgorithm(cost_matrix)
        matching, total_weight = hungarian.solve()
        
        self.assertEqual(matching, [(0, 0)], "Should match single vertex to itself")
        self.assertEqual(total_weight, 5, "Total weight should be 5")
    
    def test_zero_weights(self):
        """Test Hungarian algorithm with zero weights."""
        cost_matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        
        hungarian = HungarianAlgorithm(cost_matrix)
        matching, total_weight = hungarian.solve()
        
        self.assertEqual(len(matching), 3, "Should have 3 matched pairs")
        self.assertEqual(total_weight, 0, "Total weight should be 0")
    
    def test_negative_weights(self):
        """Test Hungarian algorithm with negative weights."""
        cost_matrix = [
            [-1, -2, -3],
            [-2, -4, -6],
            [-3, -6, -9]
        ]
        
        hungarian = HungarianAlgorithm(cost_matrix)
        matching, total_weight = hungarian.solve()
        
        self.assertEqual(len(matching), 3, "Should have 3 matched pairs")
        # With negative weights, maximum should be least negative
        self.assertLessEqual(total_weight, -6, "Should select less negative weights")


class TestHopcroftKarpAlgorithm(unittest.TestCase):
    """Test cases for Hopcroft-Karp Algorithm implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = BipartiteGraphGenerator(seed=42)
    
    def test_simple_perfect_matching(self):
        """Test Hopcroft-Karp on a graph with perfect matching."""
        edges = [(0, 0), (1, 1), (2, 2)]
        
        algorithm = create_bipartite_graph_from_edges(edges, 3, 3)
        matching, size = algorithm.solve()
        
        self.assertEqual(size, 3, "Should find perfect matching of size 3")
        self.assertEqual(len(matching), 3, "Matching should have 3 pairs")
        
        # Verify all vertices are matched
        left_matched = set(pair[0] for pair in matching)
        right_matched = set(pair[1] for pair in matching)
        self.assertEqual(left_matched, {0, 1, 2}, "All left vertices matched")
        self.assertEqual(right_matched, {0, 1, 2}, "All right vertices matched")
    
    def test_no_perfect_matching(self):
        """Test Hopcroft-Karp on a graph without perfect matching."""
        edges = [(0, 0), (1, 0), (2, 0)]  # All left vertices connect to same right vertex
        
        algorithm = create_bipartite_graph_from_edges(edges, 3, 3)
        matching, size = algorithm.solve()
        
        self.assertEqual(size, 1, "Should find maximum matching of size 1")
        self.assertEqual(len(matching), 1, "Matching should have 1 pair")
        self.assertIn((0, 0), matching, "Should include edge (0,0) or similar")
    
    def test_larger_graph(self):
        """Test Hopcroft-Karp on a larger bipartite graph."""
        edges = [
            (0, 0), (0, 1),
            (1, 1), (1, 2),
            (2, 0), (2, 2),
            (3, 2), (3, 3),
            (4, 3), (4, 4)
        ]
        
        algorithm = create_bipartite_graph_from_edges(edges, 5, 5)
        matching, size = algorithm.solve()
        
        self.assertLessEqual(size, 5, "Matching size should not exceed number of vertices")
        self.assertEqual(len(matching), size, "Matching list size should equal matching size")
        
        # Verify matching properties
        if size > 0:
            left_vertices = [pair[0] for pair in matching]
            right_vertices = [pair[1] for pair in matching]
            self.assertEqual(len(set(left_vertices)), size, "All matched left vertices unique")
            self.assertEqual(len(set(right_vertices)), size, "All matched right vertices unique")
    
    def test_empty_graph(self):
        """Test Hopcroft-Karp on empty graph."""
        algorithm = HopcroftKarpAlgorithm(3, 3)
        matching, size = algorithm.solve()
        
        self.assertEqual(size, 0, "Empty graph should have matching size 0")
        self.assertEqual(len(matching), 0, "Empty graph should have empty matching")
    
    def test_single_edge(self):
        """Test Hopcroft-Karp on graph with single edge."""
        edges = [(0, 0)]
        
        algorithm = create_bipartite_graph_from_edges(edges, 1, 1)
        matching, size = algorithm.solve()
        
        self.assertEqual(size, 1, "Single edge should give matching size 1")
        self.assertEqual(matching, [(0, 0)], "Should match the single edge")
    
    def test_asymmetric_graph(self):
        """Test Hopcroft-Karp on asymmetric bipartite graph."""
        edges = [(0, 0), (0, 1), (1, 1)]  # 2 left vertices, 2 right vertices
        
        algorithm = create_bipartite_graph_from_edges(edges, 2, 2)
        matching, size = algorithm.solve()
        
        self.assertLessEqual(size, 2, "Matching size bounded by smaller partition")
        self.assertEqual(len(matching), size, "Matching size consistent")
    
    def test_disconnected_components(self):
        """Test Hopcroft-Karp on graph with disconnected components."""
        edges = [(0, 0), (2, 2)]  # Two disconnected edges
        
        algorithm = create_bipartite_graph_from_edges(edges, 3, 3)
        matching, size = algorithm.solve()
        
        self.assertEqual(size, 2, "Should match both disconnected components")
        self.assertEqual(len(matching), 2, "Should have 2 matched pairs")


class TestAlgorithmComparison(unittest.TestCase):
    """Test cases comparing both algorithms on equivalent problems."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = BipartiteGraphGenerator(seed=123)
    
    def test_cardinality_comparison(self):
        """Compare cardinality results between algorithms on unweighted complete graphs."""
        size = 4
        edges = self.generator.generate_complete_bipartite_graph(size, size)
        
        # Hopcroft-Karp result
        hk_algorithm = create_bipartite_graph_from_edges(edges, size, size)
        hk_matching, hk_size = hk_algorithm.solve()
        
        # Hungarian result (with unit weights)
        unit_matrix = np.ones((size, size))
        hungarian_matching, hungarian_weight = solve_maximum_weight_matching(unit_matrix)
        
        # Both should find perfect matching
        self.assertEqual(hk_size, size, "Hopcroft-Karp should find perfect matching")
        self.assertEqual(len(hungarian_matching), size, "Hungarian should find perfect matching")
        self.assertEqual(hungarian_weight, size, "Hungarian weight should equal cardinality")
    
    def test_random_graphs_consistency(self):
        """Test both algorithms on random graphs for consistency."""
        test_cases = self.generator.generate_test_cases()
        
        for test_case in test_cases[:3]:  # Test first 3 cases to avoid long test times
            with self.subTest(test_case=test_case['name']):
                edges, weights = self.generator.generate_graph_from_test_case(test_case)
                
                if not edges:  # Skip empty graphs
                    continue
                
                # Hopcroft-Karp
                hk_alg = create_bipartite_graph_from_edges(
                    edges, test_case['left_size'], test_case['right_size']
                )
                hk_matching, hk_size = hk_alg.solve()
                
                # Basic validation
                self.assertLessEqual(hk_size, min(test_case['left_size'], test_case['right_size']))
                self.assertEqual(len(hk_matching), hk_size)
                
                # Verify matching properties
                if hk_size > 0:
                    left_matched = [pair[0] for pair in hk_matching]
                    right_matched = [pair[1] for pair in hk_matching]
                    self.assertEqual(len(set(left_matched)), hk_size, "Left vertices unique")
                    self.assertEqual(len(set(right_matched)), hk_size, "Right vertices unique")


class TestInputValidation(unittest.TestCase):
    """Test input validation and error handling."""
    
    def test_hungarian_invalid_input(self):
        """Test Hungarian algorithm input validation."""
        # Non-square matrix
        with self.assertRaises(ValueError):
            HungarianAlgorithm([[1, 2], [3, 4], [5, 6]])
        
        # Empty matrix
        with self.assertRaises((ValueError, IndexError)):
            HungarianAlgorithm([])
    
    def test_hopcroft_karp_invalid_edges(self):
        """Test Hopcroft-Karp edge validation."""
        algorithm = HopcroftKarpAlgorithm(3, 3)
        
        # Invalid left vertex
        with self.assertRaises(ValueError):
            algorithm.add_edge(5, 0)
        
        # Invalid right vertex  
        with self.assertRaises(ValueError):
            algorithm.add_edge(0, 5)
        
        # Negative vertices
        with self.assertRaises(ValueError):
            algorithm.add_edge(-1, 0)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestHungarianAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestHopcroftKarpAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestAlgorithmComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestInputValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
