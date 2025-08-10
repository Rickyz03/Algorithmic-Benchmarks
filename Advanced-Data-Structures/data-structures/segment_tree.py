"""
Segment Tree Implementation

A Segment Tree is a binary tree data structure that allows efficient range queries
and point updates on an array. Each node represents an interval, and the value
stored at each node is the result of applying some operation (like sum, min, max)
on the elements in that interval.

Time Complexity:
- Build: O(n)
- Query: O(log n)
- Update: O(log n)

Space Complexity: O(4n) = O(n)
"""

from typing import List, Callable, TypeVar, Generic

T = TypeVar('T')

class SegmentTree(Generic[T]):
    """
    Generic Segment Tree implementation supporting any associative operation.
    
    Attributes:
        n: Size of the original array
        tree: Internal tree representation (4 times the size for safety)
        operation: Binary operation to combine values (must be associative)
        identity: Identity element for the operation
    """
    
    def __init__(self, arr: List[T], operation: Callable[[T, T], T], identity: T):
        """
        Initialize the segment tree.
        
        Args:
            arr: Input array to build the tree from
            operation: Binary operation (e.g., lambda x, y: x + y for sum)
            identity: Identity element (e.g., 0 for sum, float('inf') for min)
        """
        self.n = len(arr)
        self.tree = [identity] * (4 * self.n)
        self.operation = operation
        self.identity = identity
        
        if arr:
            self._build(arr, 0, 0, self.n - 1)
    
    def _build(self, arr: List[T], node: int, start: int, end: int) -> None:
        """
        Build the segment tree recursively.
        
        Args:
            arr: Input array
            node: Current node index in the tree
            start: Start index of current segment
            end: End index of current segment
        """
        if start == end:
            # Leaf node
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self._build(arr, left_child, start, mid)
            self._build(arr, right_child, mid + 1, end)
            
            self.tree[node] = self.operation(self.tree[left_child], self.tree[right_child])
    
    def query(self, query_start: int, query_end: int) -> T:
        """
        Query the result of operation on range [query_start, query_end].
        
        Args:
            query_start: Start index of query range (inclusive)
            query_end: End index of query range (inclusive)
            
        Returns:
            Result of operation applied to elements in the range
        """
        if query_start < 0 or query_end >= self.n or query_start > query_end:
            raise IndexError("Invalid query range")
            
        return self._query(0, 0, self.n - 1, query_start, query_end)
    
    def _query(self, node: int, start: int, end: int, query_start: int, query_end: int) -> T:
        """
        Helper method for range queries.
        
        Args:
            node: Current node index
            start: Start of current segment
            end: End of current segment  
            query_start: Start of query range
            query_end: End of query range
            
        Returns:
            Result of operation on the queried range
        """
        if query_start > end or query_end < start:
            # No overlap
            return self.identity
        
        if query_start <= start and end <= query_end:
            # Complete overlap
            return self.tree[node]
        
        # Partial overlap
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_result = self._query(left_child, start, mid, query_start, query_end)
        right_result = self._query(right_child, mid + 1, end, query_start, query_end)
        
        return self.operation(left_result, right_result)
    
    def update(self, index: int, value: T) -> None:
        """
        Update element at given index to new value.
        
        Args:
            index: Index to update
            value: New value
        """
        if index < 0 or index >= self.n:
            raise IndexError("Index out of bounds")
            
        self._update(0, 0, self.n - 1, index, value)
    
    def _update(self, node: int, start: int, end: int, index: int, value: T) -> None:
        """
        Helper method for point updates.
        
        Args:
            node: Current node index
            start: Start of current segment
            end: End of current segment
            index: Index to update
            value: New value
        """
        if start == end:
            # Leaf node
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            if index <= mid:
                self._update(left_child, start, mid, index, value)
            else:
                self._update(right_child, mid + 1, end, index, value)
            
            self.tree[node] = self.operation(self.tree[left_child], self.tree[right_child])


# Convenience classes for common operations
class SumSegmentTree(SegmentTree[int]):
    """Segment Tree specialized for sum operations."""
    
    def __init__(self, arr: List[int]):
        super().__init__(arr, lambda x, y: x + y, 0)


class MinSegmentTree(SegmentTree[int]):
    """Segment Tree specialized for minimum operations."""
    
    def __init__(self, arr: List[int]):
        super().__init__(arr, lambda x, y: min(x, y), float('inf'))


class MaxSegmentTree(SegmentTree[int]):
    """Segment Tree specialized for maximum operations."""
    
    def __init__(self, arr: List[int]):
        super().__init__(arr, lambda x, y: max(x, y), float('-inf'))
