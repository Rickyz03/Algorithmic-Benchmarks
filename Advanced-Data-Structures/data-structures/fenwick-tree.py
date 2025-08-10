"""
Fenwick Tree (Binary Indexed Tree) Implementation

A Fenwick Tree is a data structure that can efficiently calculate prefix sums
in O(log n) time, and update values in O(log n) time. It uses the binary
representation of indices to achieve this efficiency with minimal space overhead.

The key insight is that any positive integer can be decomposed into powers of 2,
and we can use this to build a tree-like structure in an array.

Time Complexity:
- Build: O(n)
- Prefix Sum Query: O(log n)
- Range Sum Query: O(log n)
- Update: O(log n)

Space Complexity: O(n)
"""

from typing import List


class FenwickTree:
    """
    Fenwick Tree (Binary Indexed Tree) for efficient prefix sum operations.
    
    The tree is 1-indexed internally for easier bit manipulation, but provides
    0-indexed interface to match Python conventions.
    
    Attributes:
        n: Size of the original array
        tree: Internal BIT array (1-indexed)
    """
    
    def __init__(self, arr: List[int]):
        """
        Initialize the Fenwick Tree from an array.
        
        Args:
            arr: Input array to build the tree from
        """
        self.n = len(arr)
        self.tree = [0] * (self.n + 1)  # 1-indexed
        
        # Build the tree by adding each element
        for i in range(self.n):
            self.update(i, arr[i])
    
    @classmethod
    def from_size(cls, size: int) -> 'FenwickTree':
        """
        Create a Fenwick Tree of given size initialized with zeros.
        
        Args:
            size: Size of the tree
            
        Returns:
            New FenwickTree instance
        """
        return cls([0] * size)
    
    def _lsb(self, x: int) -> int:
        """
        Get the least significant bit (rightmost set bit) of x.
        This is achieved using the identity: x & (-x)
        
        Args:
            x: Input number
            
        Returns:
            The value of the least significant bit
        """
        return x & (-x)
    
    def update(self, index: int, delta: int) -> None:
        """
        Add delta to the element at given index.
        
        Args:
            index: 0-based index to update
            delta: Value to add to the element
        """
        if index < 0 or index >= self.n:
            raise IndexError("Index out of bounds")
        
        # Convert to 1-based indexing
        index += 1
        
        while index <= self.n:
            self.tree[index] += delta
            index += self._lsb(index)
    
    def set_value(self, index: int, value: int) -> None:
        """
        Set the element at given index to a specific value.
        
        Args:
            index: 0-based index to set
            value: New value for the element
        """
        # To set a value, we need to know the current value
        current = self.range_sum(index, index)
        self.update(index, value - current)
    
    def prefix_sum(self, index: int) -> int:
        """
        Calculate the sum of elements from 0 to index (inclusive).
        
        Args:
            index: 0-based end index (inclusive)
            
        Returns:
            Sum of elements from 0 to index
        """
        if index < 0:
            return 0
        if index >= self.n:
            raise IndexError("Index out of bounds")
        
        # Convert to 1-based indexing
        index += 1
        result = 0
        
        while index > 0:
            result += self.tree[index]
            index -= self._lsb(index)
        
        return result
    
    def range_sum(self, left: int, right: int) -> int:
        """
        Calculate the sum of elements in range [left, right] (both inclusive).
        
        Args:
            left: 0-based start index (inclusive)
            right: 0-based end index (inclusive)
            
        Returns:
            Sum of elements in the given range
        """
        if left > right:
            raise ValueError("Invalid range: left > right")
        
        if left == 0:
            return self.prefix_sum(right)
        else:
            return self.prefix_sum(right) - self.prefix_sum(left - 1)
    
    def find_kth_element(self, k: int) -> int:
        """
        Find the index of the k-th element in the cumulative sum.
        This assumes all elements are positive.
        
        Args:
            k: Target cumulative sum value (1-indexed)
            
        Returns:
            0-based index where cumulative sum first reaches k
        """
        if k <= 0:
            return -1
        
        position = 0
        bit_mask = 1
        
        # Find the highest power of 2 <= n
        while bit_mask <= self.n:
            bit_mask <<= 1
        bit_mask >>= 1
        
        while bit_mask > 0:
            next_pos = position + bit_mask
            if next_pos <= self.n and self.tree[next_pos] < k:
                k -= self.tree[next_pos]
                position = next_pos
            bit_mask >>= 1
        
        return position  # Convert back to 0-based indexing
    
    def __str__(self) -> str:
        """String representation showing the internal tree structure."""
        return f"FenwickTree(size={self.n}, tree={self.tree[1:]})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


class FenwickTree2D:
    """
    2D Fenwick Tree for efficient 2D range sum queries.
    
    Allows updating single points and querying rectangular ranges in O(log²n) time.
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
        """Get the least significant bit of x."""
        return x & (-x)
    
    def update(self, row: int, col: int, delta: int) -> None:
        """
        Add delta to element at (row, col).
        
        Args:
            row: 0-based row index
            col: 0-based column index
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
    
    def prefix_sum(self, row: int, col: int) -> int:
        """
        Calculate sum of rectangle from (0,0) to (row,col) inclusive.
        
        Args:
            row: 0-based row index
            col: 0-based column index
            
        Returns:
            Sum of elements in the rectangle
        """
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
    
    def range_sum(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """
        Calculate sum of rectangle from (row1,col1) to (row2,col2) inclusive.
        
        Args:
            row1, col1: Top-left corner (inclusive)
            row2, col2: Bottom-right corner (inclusive)
            
        Returns:
            Sum of elements in the rectangle
        """
        return (self.prefix_sum(row2, col2) 
                - self.prefix_sum(row1 - 1, col2)
                - self.prefix_sum(row2, col1 - 1) 
                + self.prefix_sum(row1 - 1, col1 - 1))
