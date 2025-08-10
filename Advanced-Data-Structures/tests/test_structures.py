"""
Comprehensive test suite for all data structures.

This module contains unit tests for:
- Basic implementations of Segment Tree, Fenwick Tree, Union-Find
- Advanced variants with optimizations
- Edge cases and performance characteristics
- Correctness verification against naive implementations
"""

import pytest
import random
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-structures'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'variants'))

# Import all data structures
try:
    from segment_tree import SegmentTree, SumSegmentTree, MinSegmentTree, MaxSegmentTree
    from fenwick_tree import FenwickTree, FenwickTree2D
    from union_find import UnionFind, WeightedUnionFind, UnionFindWithRollback
    
    # Variants
    from segment_tree_variant import LazySegmentTree, RangeSumLazySegmentTree, SegmentTree2D, PersistentSegmentTree
    from fenwick_tree_variant import RangeUpdateFenwickTree, FenwickTree2DAdvanced, CoordinateCompressedFenwick
    from union_finds_variant import UnionFindOptimized, PersistentUnionFind, DynamicConnectivity
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are in the correct directories")


class TestSegmentTree:
    """Test cases for Segment Tree implementations."""
    
    def test_sum_segment_tree_basic(self):
        """Test basic sum segment tree operations."""
        arr = [1, 3, 5, 7, 9, 11]
        seg_tree = SumSegmentTree(arr)
        
        # Test range queries
        assert seg_tree.query(0, 2) == 9  # 1 + 3 + 5
        assert seg_tree.query(1, 4) == 24  # 3 + 5 + 7 + 9
        assert seg_tree.query(0, 5) == 36  # Sum of all elements
        
        # Test point updates
        seg_tree.update(2, 10)  # Change 5 to 10
        assert seg_tree.query(0, 2) == 14  # 1 + 3 + 10
        assert seg_tree.query(1, 4) == 29  # 3 + 10 + 7 + 9
    
    def test_min_segment_tree(self):
        """Test minimum segment tree."""
        arr = [4, 2, 6, 1, 8, 3]
        seg_tree = MinSegmentTree(arr)
        
        assert seg_tree.query(0, 2) == 2
        assert seg_tree.query(1, 4) == 1
        assert seg_tree.query(0, 5) == 1
        
        seg_tree.update(3, 0)  # Change 1 to 0
        assert seg_tree.query(0, 5) == 0
        assert seg_tree.query(4, 5) == 3
    
    def test_max_segment_tree(self):
        """Test maximum segment tree."""
        arr = [4, 2, 6, 1, 8, 3]
        seg_tree = MaxSegmentTree(arr)
        
        assert seg_tree.query(0, 2) == 6
        assert seg_tree.query(1, 4) == 8
        assert seg_tree.query(0, 5) == 8
        
        seg_tree.update(1, 10)  # Change 2 to 10
        assert seg_tree.query(0, 2) == 10
        assert seg_tree.query(3, 5) == 8
    
    def test_segment_tree_edge_cases(self):
        """Test edge cases for segment tree."""
        # Single element
        seg_tree = SumSegmentTree([42])
        assert seg_tree.query(0, 0) == 42
        
        seg_tree.update(0, 24)
        assert seg_tree.query(0, 0) == 24
        
        # Empty ranges should raise exceptions
        with pytest.raises(IndexError):
            seg_tree.query(-1, 0)
        
        with pytest.raises(IndexError):
            seg_tree.query(0, 1)  # Out of bounds
    
    def test_generic_segment_tree(self):
        """Test generic segment tree with custom operations."""
        # GCD segment tree
        import math
        arr = [12, 18, 24, 36]
        gcd_tree = SegmentTree(arr, math.gcd, 0)
        
        assert gcd_tree.query(0, 1) == 6  # gcd(12, 18)
        assert gcd_tree.query(0, 3) == 6  # gcd(12, 18, 24, 36)
        
        gcd_tree.update(0, 30)
        assert gcd_tree.query(0, 3) == 6  # gcd(30, 18, 24, 36)


class TestFenwickTree:
    """Test cases for Fenwick Tree implementations."""
    
    def test_fenwick_tree_basic(self):
        """Test basic Fenwick Tree operations."""
        arr = [1, 3, 5, 7, 9, 11]
        ft = FenwickTree(arr)
        
        # Test prefix sums
        assert ft.prefix_sum(0) == 1
        assert ft.prefix_sum(2) == 9  # 1 + 3 + 5
        assert ft.prefix_sum(5) == 36  # Sum of all
        
        # Test range sums
        assert ft.range_sum(1, 3) == 15  # 3 + 5 + 7
        assert ft.range_sum(0, 2) == 9   # 1 + 3 + 5
        
        # Test updates
        ft.update(2, 5)  # Add 5 to index 2 (was 5, now 10)
        assert ft.prefix_sum(2) == 14  # 1 + 3 + 10
        assert ft.range_sum(1, 3) == 20  # 3 + 10 + 7
    
    def test_fenwick_tree_from_size(self):
        """Test Fenwick Tree initialized with zeros."""
        ft = FenwickTree.from_size(5)
        
        # All sums should be 0 initially
        assert ft.prefix_sum(4) == 0
        assert ft.range_sum(1, 3) == 0
        
        # Add some values
        ft.update(0, 10)
        ft.update(2, 20)
        ft.update(4, 30)
        
        assert ft.prefix_sum(0) == 10
        assert ft.prefix_sum(2) == 30
        assert ft.prefix_sum(4) == 60
        assert ft.range_sum(1, 3) == 20
    
    def test_fenwick_tree_set_value(self):
        """Test setting absolute values in Fenwick Tree."""
        arr = [1, 2, 3, 4, 5]
        ft = FenwickTree(arr)
        
        # Set value at index 2 to 10
        ft.set_value(2, 10)
        assert ft.range_sum(2, 2) == 10
        assert ft.prefix_sum(4) == 22  # 1 + 2 + 10 + 4 + 5
        
        # Set value at index 0 to 0
        ft.set_value(0, 0)
        assert ft.prefix_sum(0) == 0
        assert ft.prefix_sum(4) == 21  # 0 + 2 + 10 + 4 + 5
    
    def test_fenwick_tree_find_kth(self):
        """Test finding k-th element in Fenwick Tree."""
        # Create array with positive values
        arr = [1, 2, 3, 4, 5]
        ft = FenwickTree(arr)
        
        # 1st element (cumsum=1) is at index 0
        assert ft.find_kth_element(1) == 0
        # 3rd element (cumsum=6, first >=3 at index 1) 
        assert ft.find_kth_element(3) == 1
        # 6th element (cumsum=6, first >=6 at index 2)
        assert ft.find_kth_element(6) == 2
    
    def test_fenwick_tree_2d(self):
        """Test 2D Fenwick Tree."""
        ft_2d = FenwickTree2D(3, 3)
        
        # Add some values
        ft_2d.update(0, 0, 1)
        ft_2d.update(1, 1, 2)
        ft_2d.update(2, 2, 3)
        
        # Test prefix sums
        assert ft_2d.prefix_sum(0, 0) == 1
        assert ft_2d.prefix_sum(1, 1) == 3  # 1 + 2
        assert ft_2d.prefix_sum(2, 2) == 6  # 1 + 2 + 3
        
        # Test range sums
        assert ft_2d.range_sum(0, 0, 1, 1) == 3  # Rectangle sum
        assert ft_2d.range_sum(1, 1, 2, 2) == 5  # 2 + 3


class TestUnionFind:
    """Test cases for Union-Find implementations."""
    
    def test_union_find_basic(self):
        """Test basic Union-Find operations."""
        uf = UnionFind(5)
        
        # Initially all separate
        assert uf.num_components == 5
        assert not uf.connected(0, 1)
        
        # Union some elements
        assert uf.union(0, 1) == True  # New union
        assert uf.union(0, 1) == False  # Already connected
        assert uf.connected(0, 1) == True
        assert uf.num_components == 4
        
        # Union more elements
        uf.union(2, 3)
        assert uf.num_components == 3
        
        uf.union(1, 2)  # Connect components
        assert uf.connected(0, 3) == True
        assert uf.num_components == 2
    
    def test_union_find_component_sizes(self):
        """Test component size tracking."""
        uf = UnionFind(6)
        
        # All components size 1 initially
        for i in range(6):
            assert uf.component_size(i) == 1
        
        # Union operations
        uf.union(0, 1)
        assert uf.component_size(0) == 2
        assert uf.component_size(1) == 2
        
        uf.union(2, 3)
        uf.union(3, 4)
        assert uf.component_size(2) == 3
        assert uf.component_size(4) == 3
        
        # Connect large components
        uf.union(1, 4)
        assert uf.component_size(0) == 5
        assert uf.component_size(3) == 5
    
    def test_union_find_get_components(self):
        """Test getting all components."""
        uf = UnionFind(6)
        
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(4, 5)
        
        components = uf.get_components()
        assert len(components) == 3
        
        # Check that each component has correct elements
        component_sizes = [len(comp) for comp in components.values()]
        assert sorted(component_sizes) == [2, 2, 2]
    
    def test_weighted_union_find(self):
        """Test Weighted Union-Find."""
        wuf = WeightedUnionFind(4)
        
        # Connect with weights: 0-1 with diff 5, 1-2 with diff 3
        wuf.union(0, 1, 5)
        wuf.union(1, 2, 3)
        
        # Check weight differences
        assert wuf.get_weight_difference(0, 1) == 5
        assert wuf.get_weight_difference(1, 2) == 3
        assert wuf.get_weight_difference(0, 2) == 8  # Transitive
        
        # Try to connect with inconsistent weight
        # This should work if consistent, fail if not
        assert wuf.connected(0, 2) == True
    
    def test_union_find_with_rollback(self):
        """Test Union-Find with rollback functionality."""
        uf_rb = UnionFindWithRollback(5)
        
        # Perform unions
        uf_rb.union(0, 1)
        uf_rb.union(2, 3)
        assert uf_rb.num_components == 3
        
        uf_rb.union(1, 2)
        assert uf_rb.num_components == 2
        assert uf_rb.connected(0, 3) == True
        
        # Rollback one operation
        uf_rb.rollback()
        assert uf_rb.num_components == 3
        assert uf_rb.connected(0, 3) == False
        assert uf_rb.connected(0, 1) == True  # Still connected
        assert uf_rb.connected(2, 3) == True  # Still connected


class TestAdvancedVariants:
    """Test cases for advanced variants."""
    
    def test_lazy_segment_tree(self):
        """Test Lazy Propagation Segment Tree."""
        arr = [1, 2, 3, 4, 5]
        lazy_st = RangeSumLazySegmentTree(arr)
        
        # Initial range query
        assert lazy_st.range_query(1, 3) == 9  # 2 + 3 + 4
        
        # Range update: add 10 to range [1, 3]
        lazy_st.range_update(1, 3, 10)
        assert lazy_st.range_query(1, 3) == 39  # (2+10) + (3+10) + (4+10)
        assert lazy_st.range_query(0, 4) == 45  # 1 + 12 + 13 + 14 + 5
        
        # Another range update
        lazy_st.range_update(0, 1, 5)
        assert lazy_st.range_query(0, 1) == 23  # (1+5) + (2+10+5)
    
    def test_coordinate_compressed_fenwick(self):
        """Test Coordinate Compressed Fenwick Tree."""
        ccf = CoordinateCompressedFenwick()
        
        # Add coordinates (can be large and sparse)
        coords = [1, 1000000, 999999999, 5]
        for coord in coords:
            ccf.add_coordinate(coord)
        
        ccf.finalize()
        
        # Perform updates
        ccf.update(1, 10)
        ccf.update(1000000, 20)
        ccf.update(5, 30)
        
        # Test prefix sums
        assert ccf.prefix_sum(1) == 10
        assert ccf.prefix_sum(100) == 40  # 1 and 5 are <= 100
        assert ccf.prefix_sum(999999999) == 60  # All coordinates
        
        # Test range sums
        assert ccf.range_sum(2, 1000000) == 50  # 5 and 1000000
    
    def test_2d_segment_tree(self):
        """Test 2D Segment Tree."""
        matrix = [
            [1, 2, 3],
            [4, 5, 6], 
            [7, 8, 9]
        ]
        st_2d = SegmentTree2D(matrix)
        
        # Test queries
        assert st_2d.query(0, 0, 1, 1) == 12  # Top-left 2x2: 1+2+4+5
        assert st_2d.query(1, 1, 2, 2) == 28  # Bottom-right 2x2: 5+6+8+9
        assert st_2d.query(0, 0, 2, 2) == 45  # Entire matrix
        
        # Test updates
        st_2d.update(1, 1, 10)  # Change 5 to 10
        assert st_2d.query(0, 0, 1, 1) == 17  # 1+2+4+10
        assert st_2d.query(1, 1, 2, 2) == 33  # 10+6+8+9
    
    def test_persistent_segment_tree(self):
        """Test Persistent Segment Tree."""
        arr = [1, 2, 3, 4, 5]
        pst = PersistentSegmentTree(arr)
        
        # Query initial version
        assert pst.query(0, 1, 3) == 9  # 2 + 3 + 4
        
        # Create new version with update
        v1 = pst.update(0, 2, 10)  # Change index 2 from 3 to 10
        
        # Old version unchanged
        assert pst.query(0, 1, 3) == 9  # Still 2 + 3 + 4
        
        # New version has update
        assert pst.query(v1, 1, 3) == 16  # 2 + 10 + 4
        
        # Create another version
        v2 = pst.update(v1, 0, 100)  # Change index 0 from 1 to 100
        
        # Check all versions
        assert pst.query(0, 0, 4) == 15    # Original: 1+2+3+4+5
        assert pst.query(v1, 0, 4) == 22   # v1: 1+2+10+4+5  
        assert pst.query(v2, 0, 4) == 121  # v2: 100+2+10+4+5


class TestPerformanceAndStress:
    """Performance and stress tests."""
    
    def test_segment_tree_stress(self):
        """Stress test for Segment Tree with random operations."""
        n = 1000
        arr = [random.randint(1, 100) for _ in range(n)]
        seg_tree = SumSegmentTree(arr)
        
        # Verify against naive implementation
        for _ in range(100):
            # Random range query
            left = random.randint(0, n-1)
            right = random.randint(left, n-1)
            
            expected = sum(arr[left:right+1])
            actual = seg_tree.query(left, right)
            assert actual == expected, f"Query failed: expected {expected}, got {actual}"
            
            # Random update
            index = random.randint(0, n-1)
            new_value = random.randint(1, 100)
            old_value = arr[index]
            
            arr[index] = new_value
            seg_tree.update(index, new_value)
    
    def test_fenwick_tree_stress(self):
        """Stress test for Fenwick Tree with random operations."""
        n = 1000
        arr = [random.randint(1, 100) for _ in range(n)]
        ft = FenwickTree(arr)
        
        for _ in range(100):
            # Random prefix sum query
            index = random.randint(0, n-1)
            expected = sum(arr[:index+1])
            actual = ft.prefix_sum(index)
            assert actual == expected
            
            # Random range sum query
            left = random.randint(0, n-1)
            right = random.randint(left, n-1)
            expected = sum(arr[left:right+1])
            actual = ft.range_sum(left, right)
            assert actual == expected
            
            # Random update
            index = random.randint(0, n-1)
            delta = random.randint(-50, 50)
            arr[index] += delta
            ft.update(index, delta)
    
    def test_union_find_stress(self):
        """Stress test for Union-Find with random operations."""
        n = 1000
        uf = UnionFind(n)
        
        # Perform random unions
        unions_performed = []
        for _ in range(500):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            
            was_connected = uf.connected(x, y)
            result = uf.union(x, y)
            
            if not was_connected:
                unions_performed.append((x, y))
                assert result == True
            else:
                assert result == False
            
            # Verify connectivity
            assert uf.connected(x, y) == True
        
        # Verify all performed unions are still connected
        for x, y in unions_performed:
            assert uf.connected(x, y) == True
    
    def test_comparison_segment_vs_fenwick(self):
        """Compare Segment Tree vs Fenwick Tree for sum operations."""
        n = 500
        arr = [random.randint(1, 100) for _ in range(n)]
        
        seg_tree = SumSegmentTree(arr)
        fenwick_tree = FenwickTree(arr)
        
        # Perform same operations on both and verify results match
        for _ in range(100):
            # Range sum query
            left = random.randint(0, n-1)
            right = random.randint(left, n-1)
            
            seg_result = seg_tree.query(left, right)
            fenwick_result = fenwick_tree.range_sum(left, right)
            assert seg_result == fenwick_result
            
            # Point update
            index = random.randint(0, n-1)
            new_value = random.randint(1, 100)
            
            seg_tree.update(index, new_value)
            
            # For Fenwick tree, we need to calculate delta
            current_value = fenwick_tree.range_sum(index, index)
            delta = new_value - current_value
            fenwick_tree.update(index, delta)
            
            # Verify they still match
            test_left = random.randint(0, n-1)
            test_right = random.randint(test_left, n-1)
            assert (seg_tree.query(test_left, test_right) == 
                   fenwick_tree.range_sum(test_left, test_right))


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        # Segment Tree with empty array
        with pytest.raises((ValueError, IndexError)):
            seg_tree = SumSegmentTree([])
        
        # Fenwick Tree with empty array
        ft = FenwickTree([])
        assert ft.n == 0
    
    def test_single_element_arrays(self):
        """Test with single element arrays."""
        # Segment Tree
        seg_tree = SumSegmentTree([42])
        assert seg_tree.query(0, 0) == 42
        
        seg_tree.update(0, 24)
        assert seg_tree.query(0, 0) == 24
        
        # Fenwick Tree
        ft = FenwickTree([42])
        assert ft.prefix_sum(0) == 42
        assert ft.range_sum(0, 0) == 42
        
        ft.update(0, 10)
        assert ft.prefix_sum(0) == 52
    
    def test_boundary_conditions(self):
        """Test boundary conditions and error handling."""
        arr = [1, 2, 3, 4, 5]
        seg_tree = SumSegmentTree(arr)
        ft = FenwickTree(arr)
        uf = UnionFind(5)
        
        # Out of bounds queries
        with pytest.raises(IndexError):
            seg_tree.query(-1, 2)
        
        with pytest.raises(IndexError):
            seg_tree.query(0, 5)
        
        with pytest.raises(IndexError):
            ft.prefix_sum(-1)
        
        with pytest.raises(IndexError):
            uf.find(5)
        
        # Invalid ranges
        with pytest.raises((IndexError, ValueError)):
            seg_tree.query(3, 1)  # start > end
        
        with pytest.raises(ValueError):
            ft.range_sum(3, 1)  # start > end


def run_all_tests():
    """Run all tests manually if pytest is not available."""
    test_classes = [
        TestSegmentTree,
        TestFenwickTree, 
        TestUnionFind,
        TestAdvancedVariants,
        TestPerformanceAndStress,
        TestEdgeCases
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]
        
        print(f"\nRunning {test_class.__name__}:")
        
        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
    
    print(f"\nTest Results: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    success = run_all_tests()
    if not success:
        print("Some tests failed!")
        sys.exit(1)
    else:
        print("All tests passed!")
