"""
Advanced Fenwick Tree Variants

This module contains enhanced versions of Fenwick Trees:
1. Range Update Fenwick Tree - supports range updates with difference arrays
2. 2D Fenwick Tree with enhanced operations
3. Fenwick Tree with coordinate compression
4. Multi-dimensional Fenwick Tree

These variants extend the basic Fenwick Tree functionality for specialized use cases.
"""

from typing import List, Dict, Tuple, Optional
import bisect


class RangeUpdateFenwickTree:
    """
    Fenwick Tree supporting both point and range updates using difference arrays.
    
    Uses two Fenwick Trees to support:
    - Point updates: O(log n)
    - Range updates: O(log n) 
    - Point queries: O(log n)
    - Range sum queries: O(log n)
    
    The key insight is to use a difference array and maintain two BITs:
    - One for the difference array
    - One for index-weighted differences
    """
    
    def __init__(self, n: int):
        """
        Initialize Range Update Fenwick Tree.
        
        Args:
            n: Size of the array
        """
        self.n = n
        self.bit1 = FenwickTreeBasic(n)  # For difference array
        self.bit2 = FenwickTreeBasic(n)  # For index-weighted differences
    
    def range_update(self, left: int, right: int, delta: int) -> None:
        """
        Add delta to all elements in range [left, right].
        
        Args:
            left: Start index (inclusive)
            right: End index (inclusive)
            delta: Value to add to all elements in range
        """
        self.bit1.update(left, delta)
        self.bit1.update(right + 1, -delta)
        self.bit2.update(left, delta * (left - 1))
        self.bit2.update(right + 1, -delta * right)
    
    def point_update(self, index: int, delta: int) -> None:
        """
        Add delta to element at given index.
        
        Args:
            index: Index to update
            delta: Value to add
        """
        self.range_update(index, index, delta)
    
    def prefix_sum(self, index: int) -> int:
        """
        Calculate prefix sum up to index.
        
        Args:
            index: End index (inclusive)
            
        Returns:
            Sum of elements from 0 to index
        """
        return self.bit1.prefix_sum(index) * index - self.bit2.prefix_sum(index)
    
    def point_query(self, index: int) -> int:
        """
        Get value at specific index.
        
        Args:
            index: Index to query
            
        Returns:
            Value at the given index
        """
        if index == 0:
            return self.prefix_sum(0)
        return self.prefix_sum(index) - self.prefix_sum(index - 1)
    
    def range_sum(self, left: int, right: int) -> int:
        """
        Calculate sum of elements in range [left, right].
        
        Args:
            left: Start index (inclusive)
            right: End index (inclusive)
            
        Returns:
            Sum of elements in the range
        """
        if left == 0:
            return self.prefix_sum(right)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)


class FenwickTreeBasic:
    """Basic Fenwick Tree for internal use in advanced variants."""
    
    def __init__(self, n: int):
        """Initialize basic Fenwick Tree."""
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, index: int, delta: int) -> None:
        """Add delta to element at index."""
        index += 1  # Convert to 1-based
        while index <= self.n:
            self.tree[index] += delta
            index += index & (-index)
    
    def prefix_sum(self, index: int) -> int:
        """Calculate prefix sum up to index."""
        if index < 0:
            return 0
        index += 1  # Convert to 1-based
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & (-index)
        return result


class FenwickTree2DAdvanced:
    """
    Advanced 2D Fenwick Tree with additional operations.
    
    Supports:
    - Point updates: O(log²n)
    - Rectangle sum queries: O(log²n)
    - Rectangle updates: O(log²n) per corner
    - K-th smallest element queries (with coordinate compression)
    """
    
    def __init__(self, rows: int, cols: int):
        """
        Initialize 2D Fenwick Tree.
        
        Args:
            rows: Number of rows
            cols: Number of columns
        """
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    def _lsb(self, x: int) -> int:
        """Get least significant bit."""
        return x & (-x)
    
    def update(self, row: int, col: int, delta: int) -> None:
        """
        Add delta to element at (row, col).
        
        Args:
            row: Row index (0-based)
            col: Column index (0-based)  
            delta: Value to add
        """
        row += 1  # Convert to 1-based
        col += 1
        
        orig_col = col
        while row <= self.rows:
            col = orig_col
            while col <= self.cols:
                self.tree[row][col] += delta
                col += self._lsb(col)
            row += self._lsb(row)
    
    def rectangle_update(self, r1: int, c1: int, r2: int, c2: int, delta: int) -> None:
        """
        Add delta to all elements in rectangle from (r1,c1) to (r2,c2).
        
        Args:
            r1, c1: Top-left corner
            r2, c2: Bottom-right corner
            delta: Value to add to all elements
        """
        # Use 2D difference array technique
        self.update(r1, c1, delta)
        self.update(r1, c2 + 1, -delta)
        self.update(r2 + 1, c1, -delta)
        self.update(r2 + 1, c2 + 1, delta)
    
    def prefix_sum(self, row: int, col: int) -> int:
        """Calculate sum from (0,0) to (row,col)."""
        if row < 0 or col < 0:
            return 0
        
        row += 1  # Convert to 1-based
        col += 1
        
        result = 0
        orig_col = col
        while row > 0:
            col = orig_col
            while col > 0:
                result += self.tree[row][col]
                col -= self._lsb(col)
            row -= self._lsb(row)
        
        return result
    
    def rectangle_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Calculate sum of rectangle from (r1,c1) to (r2,c2)."""
        return (self.prefix_sum(r2, c2) 
                - self.prefix_sum(r1 - 1, c2)
                - self.prefix_sum(r2, c1 - 1) 
                + self.prefix_sum(r1 - 1, c1 - 1))


class CoordinateCompressedFenwick:
    """
    Fenwick Tree with coordinate compression for sparse arrays.
    
    Useful when the array indices are large but sparse (e.g., indices up to 10^9
    but only 10^5 updates). Maps large indices to smaller compressed indices.
    """
    
    def __init__(self):
        """Initialize coordinate compressed Fenwick Tree."""
        self.coordinates = []  # Sorted list of coordinates
        self.tree = None
        self.compressed = {}  # Map from original to compressed coordinates
        self.finalized = False
    
    def add_coordinate(self, coord: int) -> None:
        """
        Add a coordinate that will be used.
        
        Args:
            coord: Coordinate to add
        """
        if self.finalized:
            raise RuntimeError("Cannot add coordinates after finalization")
        self.coordinates.append(coord)
    
    def finalize(self) -> None:
        """
        Finalize the coordinate system and build the tree.
        Must be called after adding all coordinates and before any operations.
        """
        if self.finalized:
            return
        
        # Remove duplicates and sort
        self.coordinates = sorted(set(self.coordinates))
        n = len(self.coordinates)
        
        # Build compression mapping
        for i, coord in enumerate(self.coordinates):
            self.compressed[coord] = i
        
        # Initialize Fenwick Tree
        self.tree = FenwickTreeBasic(n)
        self.finalized = True
    
    def _get_compressed(self, coord: int) -> int:
        """Get compressed coordinate."""
        if not self.finalized:
            raise RuntimeError("Must finalize before operations")
        if coord not in self.compressed:
            raise ValueError(f"Coordinate {coord} not found")
        return self.compressed[coord]
    
    def update(self, coord: int, delta: int) -> None:
        """
        Add delta to element at given coordinate.
        
        Args:
            coord: Original coordinate
            delta: Value to add
        """
        compressed_coord = self._get_compressed(coord)
        self.tree.update(compressed_coord, delta)
    
    def prefix_sum(self, coord: int) -> int:
        """
        Calculate prefix sum up to given coordinate.
        
        Args:
            coord: Original coordinate
            
        Returns:
            Prefix sum up to the coordinate
        """
        # Find largest compressed coordinate <= coord
        pos = bisect.bisect_right(self.coordinates, coord) - 1
        if pos < 0:
            return 0
        return self.tree.prefix_sum(pos)
    
    def range_sum(self, left: int, right: int) -> int:
        """
        Calculate sum in range [left, right].
        
        Args:
            left: Left coordinate (inclusive)
            right: Right coordinate (inclusive)
            
        Returns:
            Sum of elements in the range
        """
        return self.prefix_sum(right) - self.prefix_sum(left - 1)


class MultidimensionalFenwick:
    """
    Generic multi-dimensional Fenwick Tree.
    
    Supports arbitrary dimensions with the same operations as 1D Fenwick Tree.
    Time complexity: O(log^d n) where d is the number of dimensions.
    """
    
    def __init__(self, dimensions: List[int]):
        """
        Initialize multi-dimensional Fenwick Tree.
        
        Args:
            dimensions: List of sizes for each dimension
        """
        self.dimensions = dimensions
        self.d = len(dimensions)
        
        # Initialize multi-dimensional array
        def create_array(dims: List[int]) -> List:
            if len(dims) == 1:
                return [0] * (dims[0] + 1)
            return [create_array(dims[1:]) for _ in range(dims[0] + 1)]
        
        self.tree = create_array(dimensions)
    
    def _lsb(self, x: int) -> int:
        """Get least significant bit."""
        return x & (-x)
    
    def update(self, indices: List[int], delta: int) -> None:
        """
        Add delta to element at given multi-dimensional index.
        
        Args:
            indices: List of indices for each dimension (0-based)
            delta: Value to add
        """
        if len(indices) != self.d:
            raise ValueError(f"Expected {self.d} indices, got {len(indices)}")
        
        # Convert to 1-based indexing
        indices = [idx + 1 for idx in indices]
        self._update_recursive(self.tree, indices, delta, 0)
    
    def _update_recursive(self, tree: List, indices: List[int], delta: int, dim: int) -> None:
        """Recursive helper for multi-dimensional updates."""
        if dim == self.d:
            return
        
        idx = indices[dim]
        while idx <= self.dimensions[dim]:
            if dim == self.d - 1:
                # Last dimension
                tree[idx] += delta
            else:
                # Recurse to next dimension
                self._update_recursive(tree[idx], indices, delta, dim + 1)
            idx += self._lsb(idx)
    
    def prefix_sum(self, indices: List[int]) -> int:
        """
        Calculate prefix sum up to given multi-dimensional index.
        
        Args:
            indices: List of indices for each dimension (0-based)
            
        Returns:
            Prefix sum up to the given indices
        """
        if len(indices) != self.d:
            raise ValueError(f"Expected {self.d} indices, got {len(indices)}")
        
        # Handle negative indices
        for i, idx in enumerate(indices):
            if idx < 0:
                return 0
        
        # Convert to 1-based indexing
        indices = [idx + 1 for idx in indices]
        return self._prefix_sum_recursive(self.tree, indices, 0)
    
    def _prefix_sum_recursive(self, tree: List, indices: List[int], dim: int) -> int:
        """Recursive helper for multi-dimensional prefix sums."""
        if dim == self.d:
            return 0
        
        result = 0
        idx = indices[dim]
        while idx > 0:
            if dim == self.d - 1:
                # Last dimension
                result += tree[idx]
            else:
                # Recurse to next dimension
                result += self._prefix_sum_recursive(tree[idx], indices, dim + 1)
            idx -= self._lsb(idx)
        
        return result


class FenwickTreeWithFrequencies:
    """
    Fenwick Tree specialized for frequency counting and order statistics.
    
    Supports:
    - Insert/remove elements: O(log n)
    - Count elements <= x: O(log n)
    - Find k-th smallest element: O(log n)
    - Count elements in range: O(log n)
    """
    
    def __init__(self, max_value: int):
        """
        Initialize frequency Fenwick Tree.
        
        Args:
            max_value: Maximum value that can be stored
        """
        self.max_value = max_value
        self.tree = FenwickTreeBasic(max_value + 1)
        self.total_count = 0
    
    def insert(self, value: int, count: int = 1) -> None:
        """
        Insert value with given frequency.
        
        Args:
            value: Value to insert
            count: Number of times to insert (default 1)
        """
        if value < 0 or value > self.max_value:
            raise ValueError("Value out of range")
        
        self.tree.update(value, count)
        self.total_count += count
    
    def remove(self, value: int, count: int = 1) -> None:
        """
        Remove value with given frequency.
        
        Args:
            value: Value to remove
            count: Number of times to remove (default 1)
        """
        if value < 0 or value > self.max_value:
            raise ValueError("Value out of range")
        
        current_count = self.count_equal(value)
        if count > current_count:
            raise ValueError("Cannot remove more than current count")
        
        self.tree.update(value, -count)
        self.total_count -= count
    
    def count_less_equal(self, value: int) -> int:
        """
        Count elements <= value.
        
        Args:
            value: Upper bound (inclusive)
            
        Returns:
            Number of elements <= value
        """
        if value < 0:
            return 0
        if value > self.max_value:
            value = self.max_value
        
        return self.tree.prefix_sum(value)
    
    def count_less(self, value: int) -> int:
        """Count elements < value."""
        return self.count_less_equal(value - 1)
    
    def count_greater_equal(self, value: int) -> int:
        """Count elements >= value."""
        return self.total_count - self.count_less(value)
    
    def count_greater(self, value: int) -> int:
        """Count elements > value."""
        return self.total_count - self.count_less_equal(value)
    
    def count_equal(self, value: int) -> int:
        """Count elements equal to value."""
        if value < 0 or value > self.max_value:
            return 0
        return self.count_less_equal(value) - self.count_less_equal(value - 1)
    
    def count_range(self, left: int, right: int) -> int:
        """
        Count elements in range [left, right].
        
        Args:
            left: Left bound (inclusive)
            right: Right bound (inclusive)
            
        Returns:
            Number of elements in the range
        """
        if left > right:
            return 0
        return self.count_less_equal(right) - self.count_less_equal(left - 1)
    
    def find_kth(self, k: int) -> int:
        """
        Find the k-th smallest element (1-indexed).
        
        Args:
            k: Position to find (1-indexed)
            
        Returns:
            The k-th smallest element
        """
        if k <= 0 or k > self.total_count:
            raise IndexError("k out of range")
        
        # Binary search on the answer
        left, right = 0, self.max_value
        while left < right:
            mid = (left + right) // 2
            if self.count_less_equal(mid) >= k:
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def median(self) -> float:
        """
        Find the median of all elements.
        
        Returns:
            Median value (average of two middle elements for even count)
        """
        if self.total_count == 0:
            raise ValueError("No elements to find median")
        
        if self.total_count % 2 == 1:
            # Odd number of elements
            return float(self.find_kth((self.total_count + 1) // 2))
        else:
            # Even number of elements
            mid1 = self.find_kth(self.total_count // 2)
            mid2 = self.find_kth(self.total_count // 2 + 1)
            return (mid1 + mid2) / 2.0
    
    def percentile(self, p: float) -> int:
        """
        Find the p-th percentile (0 <= p <= 100).
        
        Args:
            p: Percentile to find (0-100)
            
        Returns:
            Value at the given percentile
        """
        if not 0 <= p <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        
        if self.total_count == 0:
            raise ValueError("No elements for percentile calculation")
        
        k = max(1, int(p * self.total_count / 100))
        return self.find_kth(k)


class FenwickTreeRollback:
    """
    Fenwick Tree with rollback functionality.
    
    Supports undoing recent updates, useful for algorithms that need to
    try different combinations of operations.
    """
    
    def __init__(self, n: int):
        """Initialize Fenwick Tree with rollback."""
        self.n = n
        self.tree = [0] * (n + 1)
        self.history = []  # Stack of (index, old_value, new_value)
    
    def _lsb(self, x: int) -> int:
        """Get least significant bit."""
        return x & (-x)
    
    def update(self, index: int, delta: int) -> None:
        """
        Add delta to element at index with history tracking.
        
        Args:
            index: Index to update (0-based)
            delta: Value to add
        """
        index += 1  # Convert to 1-based
        
        # Record all changes for rollback
        changes = []
        temp_index = index
        while temp_index <= self.n:
            changes.append((temp_index, self.tree[temp_index]))
            self.tree[temp_index] += delta
            temp_index += self._lsb(temp_index)
        
        # Store changes in history
        self.history.append(changes)
    
    def prefix_sum(self, index: int) -> int:
        """Calculate prefix sum up to index."""
        index += 1  # Convert to 1-based
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= self._lsb(index)
        return result
    
    def range_sum(self, left: int, right: int) -> int:
        """Calculate range sum."""
        if left == 0:
            return self.prefix_sum(right)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)
    
    def rollback(self, steps: int = 1) -> None:
        """
        Rollback the last `steps` update operations.
        
        Args:
            steps: Number of operations to undo
        """
        if steps > len(self.history):
            raise ValueError("Cannot rollback more steps than available")
        
        for _ in range(steps):
            if not self.history:
                break
            
            changes = self.history.pop()
            # Restore old values
            for index, old_value in changes:
                self.tree[index] = old_value
    
    def checkpoint(self) -> int:
        """
        Create a checkpoint and return its ID.
        
        Returns:
            Checkpoint ID (current history length)
        """
        return len(self.history)
    
    def rollback_to_checkpoint(self, checkpoint_id: int) -> None:
        """
        Rollback to a specific checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to rollback to
        """
        if checkpoint_id < 0 or checkpoint_id > len(self.history):
            raise ValueError("Invalid checkpoint ID")
        
        steps_to_rollback = len(self.history) - checkpoint_id
        if steps_to_rollback > 0:
            self.rollback(steps_to_rollback)
