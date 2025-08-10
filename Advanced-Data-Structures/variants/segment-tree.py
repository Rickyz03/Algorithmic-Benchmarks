"""
Advanced Segment Tree Variants

This module contains enhanced versions of Segment Trees with advanced features:
1. Lazy Propagation Segment Tree - for efficient range updates
2. Persistent Segment Tree - maintains multiple versions
3. 2D Segment Tree - for 2D range queries

These variants solve more complex problems while maintaining logarithmic complexity.
"""

from typing import List, Callable, TypeVar, Generic, Optional, Tuple
import copy

T = TypeVar('T')

class LazySegmentTree(Generic[T]):
    """
    Segment Tree with Lazy Propagation for efficient range updates.
    
    Supports both range queries and range updates in O(log n) time.
    The lazy propagation technique delays updates until they are actually needed,
    which dramatically improves performance for range update operations.
    
    Time Complexity:
    - Range Query: O(log n)
    - Range Update: O(log n)
    - Point Update: O(log n)
    """
    
    def __init__(self, arr: List[T], operation: Callable[[T, T], T], identity: T, 
                 update_operation: Callable[[T, T, int], T]):
        """
        Initialize Lazy Segment Tree.
        
        Args:
            arr: Input array
            operation: Binary operation for combining values
            identity: Identity element for the operation
            update_operation: Function to apply lazy updates (value, lazy_val, segment_length)
        """
        self.n = len(arr)
        self.tree = [identity] * (4 * self.n)
        self.lazy = [identity] * (4 * self.n)
        self.operation = operation
        self.identity = identity
        self.update_operation = update_operation
        
        if arr:
            self._build(arr, 0, 0, self.n - 1)
    
    def _build(self, arr: List[T], node: int, start: int, end: int) -> None:
        """Build the segment tree."""
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            
            self._build(arr, left_child, start, mid)
            self._build(arr, right_child, mid + 1, end)
            
            self.tree[node] = self.operation(self.tree[left_child], self.tree[right_child])
    
    def _push(self, node: int, start: int, end: int) -> None:
        """Push lazy updates down to children."""
        if self.lazy[node] != self.identity:
            # Apply lazy update to current node
            self.tree[node] = self.update_operation(self.tree[node], self.lazy[node], end - start + 1)
            
            if start != end:  # Not a leaf node
                left_child = 2 * node + 1
                right_child = 2 * node + 2
                
                # Propagate lazy value to children
                self.lazy[left_child] = self.operation(self.lazy[left_child], self.lazy[node])
                self.lazy[right_child] = self.operation(self.lazy[right_child], self.lazy[node])
            
            self.lazy[node] = self.identity
    
    def range_update(self, update_start: int, update_end: int, value: T) -> None:
        """
        Update range [update_start, update_end] with given value.
        
        Args:
            update_start: Start of update range (inclusive)
            update_end: End of update range (inclusive)
            value: Value to apply to the range
        """
        if update_start < 0 or update_end >= self.n or update_start > update_end:
            raise IndexError("Invalid update range")
            
        self._range_update(0, 0, self.n - 1, update_start, update_end, value)
    
    def _range_update(self, node: int, start: int, end: int, 
                     update_start: int, update_end: int, value: T) -> None:
        """Helper method for range updates."""
        self._push(node, start, end)
        
        if update_start > end or update_end < start:
            return  # No overlap
        
        if update_start <= start and end <= update_end:
            # Complete overlap - apply lazy update
            self.lazy[node] = self.operation(self.lazy[node], value)
            self._push(node, start, end)
            return
        
        # Partial overlap
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        self._range_update(left_child, start, mid, update_start, update_end, value)
        self._range_update(right_child, mid + 1, end, update_start, update_end, value)
        
        # Update current node after children are updated
        self._push(left_child, start, mid)
        self._push(right_child, mid + 1, end)
        self.tree[node] = self.operation(self.tree[left_child], self.tree[right_child])
    
    def range_query(self, query_start: int, query_end: int) -> T:
        """
        Query range [query_start, query_end].
        
        Args:
            query_start: Start of query range (inclusive)
            query_end: End of query range (inclusive)
            
        Returns:
            Result of operation on the queried range
        """
        if query_start < 0 or query_end >= self.n or query_start > query_end:
            raise IndexError("Invalid query range")
            
        return self._range_query(0, 0, self.n - 1, query_start, query_end)
    
    def _range_query(self, node: int, start: int, end: int, 
                    query_start: int, query_end: int) -> T:
        """Helper method for range queries."""
        if query_start > end or query_end < start:
            return self.identity
        
        self._push(node, start, end)
        
        if query_start <= start and end <= query_end:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_result = self._range_query(left_child, start, mid, query_start, query_end)
        right_result = self._range_query(right_child, mid + 1, end, query_start, query_end)
        
        return self.operation(left_result, right_result)


class RangeSumLazySegmentTree(LazySegmentTree[int]):
    """Lazy Segment Tree specialized for range sum with range addition updates."""
    
    def __init__(self, arr: List[int]):
        super().__init__(
            arr,
            lambda x, y: x + y,  # Sum operation
            0,  # Identity
            lambda tree_val, lazy_val, length: tree_val + lazy_val * length  # Range addition
        )


class RangeMinLazySegmentTree(LazySegmentTree[int]):
    """Lazy Segment Tree specialized for range minimum with range assignment updates."""
    
    def __init__(self, arr: List[int]):
        super().__init__(
            arr,
            lambda x, y: min(x, y),  # Min operation
            float('inf'),  # Identity
            lambda tree_val, lazy_val, length: lazy_val if lazy_val != float('inf') else tree_val
        )


class SegmentTree2D:
    """
    2D Segment Tree for efficient 2D range queries.
    
    Supports point updates and rectangular range queries in O(log²n) time.
    Each node in the outer segment tree contains an inner segment tree.
    
    Time Complexity:
    - Build: O(n²)
    - Point Update: O(log²n)  
    - Range Query: O(log²n)
    
    Space Complexity: O(n²)
    """
    
    def __init__(self, matrix: List[List[int]]):
        """
        Initialize 2D Segment Tree from a matrix.
        
        Args:
            matrix: 2D input matrix
        """
        if not matrix or not matrix[0]:
            raise ValueError("Matrix cannot be empty")
        
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        self.tree = {}
        self._build_y(matrix, 1, 0, self.rows - 1)
    
    def _build_y(self, matrix: List[List[int]], node_x: int, start_x: int, end_x: int) -> None:
        """Build segment tree for y-coordinates at each x-node."""
        if start_x == end_x:
            self._build_x(matrix[start_x], node_x, 1, 0, self.cols - 1)
        else:
            mid_x = (start_x + end_x) // 2
            self._build_y(matrix, 2 * node_x, start_x, mid_x)
            self._build_y(matrix, 2 * node_x + 1, mid_x + 1, end_x)
            
            self._merge_trees(node_x, 2 * node_x, 2 * node_x + 1, 1, 0, self.cols - 1)
    
    def _build_x(self, row: List[int], node_x: int, node_y: int, start_y: int, end_y: int) -> None:
        """Build segment tree for a single row."""
        if (node_x, node_y) not in self.tree:
            self.tree[(node_x, node_y)] = 0
            
        if start_y == end_y:
            self.tree[(node_x, node_y)] = row[start_y]
        else:
            mid_y = (start_y + end_y) // 2
            self._build_x(row, node_x, 2 * node_y, start_y, mid_y)
            self._build_x(row, node_x, 2 * node_y + 1, mid_y + 1, end_y)
            
            left_val = self.tree.get((node_x, 2 * node_y), 0)
            right_val = self.tree.get((node_x, 2 * node_y + 1), 0)
            self.tree[(node_x, node_y)] = left_val + right_val
    
    def _merge_trees(self, node_x: int, left_child_x: int, right_child_x: int,
                    node_y: int, start_y: int, end_y: int) -> None:
        """Merge two trees by summing corresponding nodes."""
        if (node_x, node_y) not in self.tree:
            self.tree[(node_x, node_y)] = 0
            
        if start_y == end_y:
            left_val = self.tree.get((left_child_x, node_y), 0)
            right_val = self.tree.get((right_child_x, node_y), 0)
            self.tree[(node_x, node_y)] = left_val + right_val
        else:
            mid_y = (start_y + end_y) // 2
            self._merge_trees(node_x, left_child_x, right_child_x, 2 * node_y, start_y, mid_y)
            self._merge_trees(node_x, left_child_x, right_child_x, 2 * node_y + 1, mid_y + 1, end_y)
            
            left_val = self.tree.get((node_x, 2 * node_y), 0)
            right_val = self.tree.get((node_x, 2 * node_y + 1), 0)
            self.tree[(node_x, node_y)] = left_val + right_val
    
    def update(self, x: int, y: int, value: int) -> None:
        """
        Update element at position (x, y) to new value.
        
        Args:
            x, y: Coordinates to update
            value: New value
        """
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            raise IndexError("Coordinates out of bounds")
            
        self._update_y(1, 0, self.rows - 1, x, y, value)
    
    def _update_y(self, node_x: int, start_x: int, end_x: int, x: int, y: int, value: int) -> None:
        """Update in the y-direction."""
        if start_x == end_x:
            self._update_x(node_x, 1, 0, self.cols - 1, y, value)
        else:
            mid_x = (start_x + end_x) // 2
            if x <= mid_x:
                self._update_y(2 * node_x, start_x, mid_x, x, y, value)
            else:
                self._update_y(2 * node_x + 1, mid_x + 1, end_x, x, y, value)
            
            self._update_x(node_x, 1, 0, self.cols - 1, y, value)
    
    def _update_x(self, node_x: int, node_y: int, start_y: int, end_y: int, y: int, value: int) -> None:
        """Update in the x-direction."""
        if start_y == end_y:
            self.tree[(node_x, node_y)] = value
        else:
            mid_y = (start_y + end_y) // 2
            if y <= mid_y:
                self._update_x(node_x, 2 * node_y, start_y, mid_y, y, value)
            else:
                self._update_x(node_x, 2 * node_y + 1, mid_y + 1, end_y, y, value)
            
            left_val = self.tree.get((node_x, 2 * node_y), 0)
            right_val = self.tree.get((node_x, 2 * node_y + 1), 0)
            self.tree[(node_x, node_y)] = left_val + right_val
    
    def query(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """
        Query sum of rectangle from (x1,y1) to (x2,y2) inclusive.
        
        Args:
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner
            
        Returns:
            Sum of elements in the rectangle
        """
        if (x1 < 0 or x2 >= self.rows or y1 < 0 or y2 >= self.cols or 
            x1 > x2 or y1 > y2):
            raise IndexError("Invalid query rectangle")
            
        return self._query_y(1, 0, self.rows - 1, x1, x2, y1, y2)
    
    def _query_y(self, node_x: int, start_x: int, end_x: int, 
                x1: int, x2: int, y1: int, y2: int) -> int:
        """Query in the y-direction."""
        if x1 > end_x or x2 < start_x:
            return 0
        
        if x1 <= start_x and end_x <= x2:
            return self._query_x(node_x, 1, 0, self.cols - 1, y1, y2)
        
        mid_x = (start_x + end_x) // 2
        left_result = self._query_y(2 * node_x, start_x, mid_x, x1, x2, y1, y2)
        right_result = self._query_y(2 * node_x + 1, mid_x + 1, end_x, x1, x2, y1, y2)
        
        return left_result + right_result
    
    def _query_x(self, node_x: int, node_y: int, start_y: int, end_y: int, 
                y1: int, y2: int) -> int:
        """Query in the x-direction."""
        if y1 > end_y or y2 < start_y:
            return 0
        
        if y1 <= start_y and end_y <= y2:
            return self.tree.get((node_x, node_y), 0)
        
        mid_y = (start_y + end_y) // 2
        left_result = self._query_x(node_x, 2 * node_y, start_y, mid_y, y1, y2)
        right_result = self._query_x(node_x, 2 * node_y + 1, mid_y + 1, end_y, y1, y2)
        
        return left_result + right_result


class PersistentSegmentTree:
    """
    Persistent Segment Tree that maintains multiple versions.
    
    Each update creates a new version while keeping old versions accessible.
    Only the changed path from root to leaf is copied, making it space-efficient.
    
    Time Complexity:
    - Update: O(log n)
    - Query: O(log n)
    
    Space Complexity: O(log n) per update
    """
    
    class Node:
        """Node in the persistent segment tree."""
        
        def __init__(self, value: int = 0, left: 'Node' = None, right: 'Node' = None):
            self.value = value
            self.left = left
            self.right = right
    
    def __init__(self, arr: List[int]):
        """Initialize persistent segment tree."""
        self.n = len(arr)
        self.versions = []  # List of root nodes for each version
        
        if arr:
            # Build initial version
            root = self._build(arr, 0, self.n - 1)
            self.versions.append(root)
    
    def _build(self, arr: List[int], start: int, end: int) -> 'Node':
        """Build the initial segment tree."""
        if start == end:
            return self.Node(arr[start])
        
        mid = (start + end) // 2
        left_child = self._build(arr, start, mid)
        right_child = self._build(arr, mid + 1, end)
        
        return self.Node(left_child.value + right_child.value, left_child, right_child)
    
    def update(self, version: int, index: int, value: int) -> int:
        """
        Create new version with updated value at index.
        
        Args:
            version: Version to base the update on
            index: Index to update
            value: New value
            
        Returns:
            Index of the new version created
        """
        if version < 0 or version >= len(self.versions):
            raise IndexError("Invalid version")
        
        old_root = self.versions[version]
        new_root = self._update(old_root, 0, self.n - 1, index, value)
        self.versions.append(new_root)
        
        return len(self.versions) - 1
    
    def _update(self, node: 'Node', start: int, end: int, index: int, value: int) -> 'Node':
        """Helper method for persistent updates."""
        if start == end:
            return self.Node(value)
        
        mid = (start + end) // 2
        if index <= mid:
            new_left = self._update(node.left, start, mid, index, value)
            return self.Node(new_left.value + node.right.value, new_left, node.right)
        else:
            new_right = self._update(node.right, mid + 1, end, index, value)
            return self.Node(node.left.value + new_right.value, node.left, new_right)
    
    def query(self, version: int, query_start: int, query_end: int) -> int:
        """
        Query range sum in a specific version.
        
        Args:
            version: Version to query
            query_start: Start of query range (inclusive)
            query_end: End of query range (inclusive)
            
        Returns:
            Sum of elements in the range for the given version
        """
        if version < 0 or version >= len(self.versions):
            raise IndexError("Invalid version")
        
        root = self.versions[version]
        return self._query(root, 0, self.n - 1, query_start, query_end)
    
    def _query(self, node: 'Node', start: int, end: int, 
              query_start: int, query_end: int) -> int:
        """Helper method for queries."""
        if query_start > end or query_end < start or not node:
            return 0
        
        if query_start <= start and end <= query_end:
            return node.value
        
        mid = (start + end) // 2
        left_result = self._query(node.left, start, mid, query_start, query_end)
        right_result = self._query(node.right, mid + 1, end, query_start, query_end)
        
        return left_result + right_result
    
    def get_version_count(self) -> int:
        """Get the number of versions available."""
        return len(self.versions)
