"""
Unit tests for TSP algorithms and utilities.

This module provides comprehensive tests for all TSP implementations
including correctness verification, algorithm comparison, and performance testing.
"""

import numpy as np
import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_generator import (
    generate_euclidean_tsp_instance, 
    generate_metric_tsp_instance,
    validate_triangle_inequality
)
from algorithms.mst_approximation import mst_2_approximation, kruskal_mst, prim_mst
from algorithms.exact_algorithms import brute_force_tsp, held_karp_tsp, BranchAndBound
from algorithms.heuristic_algorithms import (
    nearest_neighbor_tsp, 
    two_opt_improvement,
    multi_start_nearest_neighbor
)


class TestGraphGenerator(unittest.TestCase):
    """Test graph generation utilities."""
    
    def test_euclidean_instance_generation(self):
        """Test Euclidean TSP instance generation."""
        n = 5
        distance_matrix, coordinates = generate_euclidean_tsp_instance(n, seed=42)
        
        # Check dimensions
        self.assertEqual(distance_matrix.shape, (n, n))
        self.assertEqual(len(coordinates), n)
        
        # Check symmetry
        np.testing.assert_array_almost_equal(distance_matrix, distance_matrix.T)
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(distance_matrix), np.zeros(n))
        
        # Check triangle inequality
        self.assertTrue(validate_triangle_inequality(distance_matrix))
    
    def test_metric_instance_generation(self):
        """Test metric TSP instance generation."""
        n = 6
        distance_matrix = generate_metric_tsp_instance(n, seed=42)
        
        # Check dimensions
        self.assertEqual(distance_matrix.shape, (n, n))
        
        # Check symmetry
        np.testing.assert_array_almost_equal(distance_matrix, distance_matrix.T)
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(distance_matrix), np.zeros(n))
        
        # Check triangle inequality
        self.assertTrue(validate_triangle_inequality(distance_matrix))
    
    def test_triangle_inequality_validation(self):
        """Test triangle inequality validation."""
        # Valid metric matrix
        valid_matrix = np.array([
            [0, 1, 2],
            [1, 0, 1.5],
            [2, 1.5, 0]
        ])
        self.assertTrue(validate_triangle_inequality(valid_matrix))
        
        # Invalid matrix (violates triangle inequality)
        invalid_matrix = np.array([
            [0, 1, 10],
            [1, 0, 1],
            [10, 1, 0]
        ])
        self.assertFalse(validate_triangle_inequality(invalid_matrix))


class TestMSTApproximation(unittest.TestCase):
    """Test MST-based 2-approximation algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.small_matrix = np.array([
            [0, 2, 3, 4],
            [2, 0, 4, 3],
            [3, 4, 0, 2],
            [4, 3, 2, 0]
        ])
    
    def test_kruskal_mst(self):
        """Test Kruskal's MST algorithm."""
        mst_edges = kruskal_mst(self.small_matrix)
        
        # Should have n-1 edges
        self.assertEqual(len(mst_edges), 3)
        
        # Check total MST weight
        total_weight = sum(weight for _, _, weight in mst_edges)
        self.assertEqual(total_weight, 6)  # Expected MST weight for this matrix
    
    def test_prim_mst(self):
        """Test Prim's MST algorithm."""
        mst_edges = prim_mst(self.small_matrix)
        
        # Should have n-1 edges
        self.assertEqual(len(mst_edges), 3)
        
        # Check total MST weight
        total_weight = sum(weight for _, _, weight in mst_edges)
        self.assertEqual(total_weight, 6)  # Same as Kruskal's
    
    def test_mst_2_approximation(self):
        """Test MST 2-approximation algorithm."""
        tour, length = mst_2_approximation(self.small_matrix)
        
        # Check tour visits all cities
        self.assertEqual(len(tour), 4)
        self.assertEqual(set(tour), {0, 1, 2, 3})
        
        # Check tour length is positive
        self.assertGreater(length, 0)
        
        # For this specific matrix, check approximation quality
        # Optimal tour has length 10, MST weight is 6, so approximation should be ≤ 12
        self.assertLessEqual(length, 12)
    
    def test_approximation_ratio_bound(self):
        """Test that 2-approximation bound holds for metric instances."""
        # Generate several random metric instances
        for n in [5, 6, 7]:
            with self.subTest(n=n):
                distance_matrix = generate_metric_tsp_instance(n, seed=42)
                
                # Get MST-based approximation
                tour, approx_length = mst_2_approximation(distance_matrix)
                
                # Get optimal solution for small instances
                if n <= 7:  # Only test for small instances due to computational cost
                    opt_tour, opt_length = brute_force_tsp(distance_matrix)
                    
                    # Check approximation ratio
                    ratio = approx_length / opt_length
                    self.assertLessEqual(ratio, 2.01)  # Allow small numerical tolerance


class TestExactAlgorithms(unittest.TestCase):
    """Test exact TSP algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Small test instance with known optimal solution
        self.test_matrix = np.array([
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0]
        ])
        # Make symmetric
        self.test_matrix = (self.test_matrix + self.test_matrix.T) / 2
    
    def test_brute_force_small_instance(self):
        """Test brute force on small instance."""
        tour, length = brute_force_tsp(self.test_matrix)
        
        # Check tour visits all cities
        self.assertEqual(len(tour), 4)
        self.assertEqual(set(tour), {0, 1, 2, 3})
        
        # Check length is positive
        self.assertGreater(length, 0)
    
    def test_held_karp_small_instance(self):
        """Test Held-Karp DP on small instance."""
        tour, length = held_karp_tsp(self.test_matrix)
        
        # Check tour visits all cities
        self.assertEqual(len(tour), 4)
        self.assertEqual(set(tour), {0, 1, 2, 3})
        
        # Check length is positive
        self.assertGreater(length, 0)
    
    def test_branch_and_bound_small_instance(self):
        """Test Branch and Bound on small instance."""
        bb_solver = BranchAndBound(self.test_matrix)
        tour, length = bb_solver.solve()
        
        # Check tour visits all cities
        self.assertEqual(len(tour), 4)
        self.assertEqual(set(tour), {0, 1, 2, 3})
        
        # Check length is positive
        self.assertGreater(length, 0)
        
        # Check that some nodes were explored
        self.assertGreater(bb_solver.nodes_explored, 0)
    
    def test_exact_algorithms_consistency(self):
        """Test that all exact algorithms give the same result."""
        # Test on small Euclidean instance
        distance_matrix, _ = generate_euclidean_tsp_instance(5, seed=42)
        
        # Get results from all exact algorithms
        tour_bf, length_bf = brute_force_tsp(distance_matrix)
        tour_hk, length_hk = held_karp_tsp(distance_matrix)
        
        bb_solver = BranchAndBound(distance_matrix)
        tour_bb, length_bb = bb_solver.solve()
        
        # All should give the same optimal length
        self.assertAlmostEqual(length_bf, length_hk, places=8)
        self.assertAlmostEqual(length_bf, length_bb, places=8)


class TestHeuristicAlgorithms(unittest.TestCase):
    """Test heuristic TSP algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.distance_matrix, self.coords = generate_euclidean_tsp_instance(6, seed=42)
    
    def test_nearest_neighbor(self):
        """Test nearest neighbor heuristic."""
        tour, length = nearest_neighbor_tsp(self.distance_matrix, start_city=0)
        
        # Check tour visits all cities
        self.assertEqual(len(tour), 6)
        self.assertEqual(set(tour), set(range(6)))
        
        # Check tour starts at specified city
        self.assertEqual(tour[0], 0)
        
        # Check length is positive
        self.assertGreater(length, 0)
    
    def test_multi_start_nearest_neighbor(self):
        """Test multi-start nearest neighbor."""
        tour, length = multi_start_nearest_neighbor(self.distance_matrix)
        
        # Check tour visits all cities
        self.assertEqual(len(tour), 6)
        self.assertEqual(set(tour), set(range(6)))
        
        # Check length is positive
        self.assertGreater(length, 0)
        
        # Compare with single-start NN
        single_tour, single_length = nearest_neighbor_tsp(self.distance_matrix, 0)
        
        # Multi-start should be at least as good
        self.assertLessEqual(length, single_length + 1e-10)  # Allow numerical tolerance
    
    def test_2opt_improvement(self):
        """Test 2-opt local search improvement."""
        # Start with nearest neighbor tour
        initial_tour, initial_length = nearest_neighbor_tsp(self.distance_matrix, 0)
        
        # Apply 2-opt improvement
        improved_tour, improved_length = two_opt_improvement(initial_tour, self.distance_matrix)
        
        # Check tour visits all cities
        self.assertEqual(len(improved_tour), 6)
        self.assertEqual(set(improved_tour), set(range(6)))
        
        # 2-opt should not make the tour worse
        self.assertLessEqual(improved_length, initial_length + 1e-10)
    
    def test_tour_length_calculation(self):
        """Test tour length calculation utility."""
        from algorithms.heuristic_algorithms import calculate_tour_length
        
        # Simple 3-city tour
        simple_matrix = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ])
        
        tour = [0, 1, 2]
        expected_length = 1 + 3 + 2  # 0->1->2->0
        calculated_length = calculate_tour_length(tour, simple_matrix)
        
        self.assertEqual(calculated_length, expected_length)


class TestAlgorithmComparison(unittest.TestCase):
    """Test comparison between different algorithm types."""
    
    def test_approximation_vs_optimal(self):
        """Test approximation algorithm against optimal for small instances."""
        # Generate small metric instance
        distance_matrix = generate_metric_tsp_instance(6, seed=42)
        
        # Get optimal solution
        opt_tour, opt_length = held_karp_tsp(distance_matrix)
        
        # Get MST approximation
        approx_tour, approx_length = mst_2_approximation(distance_matrix)
        
        # Check approximation ratio
        ratio = approx_length / opt_length
        self.assertLessEqual(ratio, 2.01)  # Should satisfy 2-approximation bound
        self.assertGreaterEqual(ratio, 1.0)  # Should be at least optimal
    
    def test_heuristic_vs_approximation(self):
        """Compare heuristic algorithms with approximation algorithm."""
        distance_matrix, coords = generate_euclidean_tsp_instance(8, seed=42)
        
        # Get results from different algorithms
        mst_tour, mst_length = mst_2_approximation(distance_matrix)
        nn_tour, nn_length = nearest_neighbor_tsp(distance_matrix, 0)
        nn_2opt_tour, nn_2opt_length = two_opt_improvement(nn_tour, distance_matrix)
        
        # All should give valid tours
        for tour in [mst_tour, nn_tour, nn_2opt_tour]:
            self.assertEqual(len(tour), 8)
            self.assertEqual(set(tour), set(range(8)))
        
        # 2-opt should improve or maintain nearest neighbor result
        self.assertLessEqual(nn_2opt_length, nn_length + 1e-10)


def run_performance_comparison():
    """Run a simple performance comparison between algorithms."""
    print("\nRunning performance comparison...")
    
    sizes = [5, 6, 7, 8]
    
    for n in sizes:
        print(f"\nTesting with {n} cities:")
        distance_matrix, coords = generate_euclidean_tsp_instance(n, seed=42)
        
        # Test different algorithms
        algorithms = [
            ("MST 2-approx", lambda: mst_2_approximation(distance_matrix)),
            ("Nearest Neighbor", lambda: nearest_neighbor_tsp(distance_matrix, 0)),
            ("Multi-start NN", lambda: multi_start_nearest_neighbor(distance_matrix)),
        ]
        
        if n <= 7:  # Only test exact algorithms for small instances
            algorithms.append(("Held-Karp DP", lambda: held_karp_tsp(distance_matrix)))
        
        results = []
        for name, algorithm in algorithms:
            try:
                import time
                start_time = time.time()
                tour, length = algorithm()
                end_time = time.time()
                
                results.append((name, length, end_time - start_time))
                print(f"  {name:15s}: {length:8.2f} (time: {end_time - start_time:.4f}s)")
            except Exception as e:
                print(f"  {name:15s}: ERROR - {str(e)}")
        
        # Find best result
        if results:
            best_length = min(length for _, length, _ in results)
            print(f"  Best length: {best_length:.2f}")


if __name__ == '__main__':
    # Run unit tests
    print("Running TSP algorithm tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance comparison
    run_performance_comparison()
