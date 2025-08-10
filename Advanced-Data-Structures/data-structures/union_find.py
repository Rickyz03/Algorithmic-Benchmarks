"""
Union-Find (Disjoint Set Union) Implementation

Union-Find is a data structure that maintains a collection of disjoint sets,
supporting two main operations:
- Find: Determine which set an element belongs to
- Union: Merge two sets into one

With path compression and union by rank optimizations, both operations
run in nearly O(1) amortized time.

Time Complexity (with optimizations):
- Find: O(α(n)) amortized, where α is the inverse Ackermann function
- Union: O(α(n)) amortized
- Both are effectively O(1) for practical purposes

Space Complexity: O(n)
"""

from typing import List, Dict, Set


class UnionFind:
    """
    Union-Find data structure with path compression and union by rank.
    
    Attributes:
        parent: Parent pointer for each element
        rank: Rank (approximate depth) of each tree
        size: Size of each component
        n: Number of elements
        num_components: Current number of disjoint components
    """
    
    def __init__(self, n: int):
        """
        Initialize Union-Find structure with n elements.
        
        Args:
            n: Number of elements (numbered 0 to n-1)
        """
        self.n = n
        self.parent = list(range(n))  # Each element is its own parent initially
        self.rank = [0] * n           # All trees have rank 0 initially
        self.size = [1] * n           # Each component has size 1 initially
        self.num_components = n       # Initially n components
    
    def find(self, x: int) -> int:
        """
        Find the root of the set containing element x.
        Uses path compression for optimization.
        
        Args:
            x: Element to find the root of
            
        Returns:
            Root element of the set containing x
        """
        if x < 0 or x >= self.n:
            raise IndexError("Element out of bounds")
        
        if self.parent[x] != x:
            # Path compression: make parent[x] point directly to root
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Union the sets containing elements x and y.
        Uses union by rank for optimization.
        
        Args:
            x, y: Elements whose sets should be unioned
            
        Returns:
            True if union was performed, False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in the same set
        
        # Union by rank: attach smaller rank tree under root of higher rank tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            # Same rank: make one root and increase its rank
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            self.rank[root_x] += 1
        
        self.num_components -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """
        Check if elements x and y are in the same set.
        
        Args:
            x, y: Elements to check
            
        Returns:
            True if x and y are connected, False otherwise
        """
        return self.find(x) == self.find(y)
    
    def component_size(self, x: int) -> int:
        """
        Get the size of the component containing element x.
        
        Args:
            x: Element to check
            
        Returns:
            Size of the component containing x
        """
        return self.size[self.find(x)]
    
    def get_components(self) -> Dict[int, List[int]]:
        """
        Get all components as a dictionary mapping root -> list of elements.
        
        Returns:
            Dictionary with roots as keys and lists of elements as values
        """
        components = {}
        for i in range(self.n):
            root = self.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
        
        return components
    
    def get_component_sizes(self) -> List[int]:
        """
        Get sizes of all components.
        
        Returns:
            List of component sizes
        """
        roots = set()
        for i in range(self.n):
            roots.add(self.find(i))
        
        return [self.size[root] for root in roots]
    
    def __str__(self) -> str:
        """String representation showing components."""
        components = self.get_components()
        return f"UnionFind({self.num_components} components: {components})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"UnionFind(n={self.n}, components={self.num_components}, "
                f"parent={self.parent[:10]}{'...' if self.n > 10 else ''})")


class WeightedUnionFind:
    """
    Weighted Union-Find supporting weighted edges between elements.
    
    Maintains relative weights/distances between elements in the same component.
    Useful for problems involving relative positions or differences.
    """
    
    def __init__(self, n: int):
        """
        Initialize Weighted Union-Find structure.
        
        Args:
            n: Number of elements
        """
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
        self.weight = [0] * n  # Weight relative to parent
        self.num_components = n
    
    def find(self, x: int) -> int:
        """
        Find root with path compression, updating weights along the path.
        
        Args:
            x: Element to find root of
            
        Returns:
            Root of the component containing x
        """
        if self.parent[x] != x:
            original_parent = self.parent[x]
            self.parent[x] = self.find(self.parent[x])
            # Update weight to be relative to new parent (root)
            self.weight[x] += self.weight[original_parent]
        
        return self.parent[x]
    
    def union(self, x: int, y: int, w: int) -> bool:
        """
        Union sets containing x and y with weight difference w.
        This means weight(y) - weight(x) = w after union.
        
        Args:
            x, y: Elements to union
            w: Weight difference (y's weight - x's weight)
            
        Returns:
            True if union was performed, False if already connected
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already connected
        
        # Calculate the weight that should be assigned to root_y
        # weight(y) - weight(x) = w
        # weight(root_y) + weight[y] - (weight(root_x) + weight[x]) = w
        # weight(root_y) - weight(root_x) = w - weight[y] + weight[x]
        weight_diff = w - self.weight[y] + self.weight[x]
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.weight[root_x] = -weight_diff
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.weight[root_y] = weight_diff
        else:
            self.parent[root_y] = root_x
            self.weight[root_y] = weight_diff
            self.rank[root_x] += 1
        
        self.num_components -= 1
        return True
    
    def get_weight_difference(self, x: int, y: int) -> int:
        """
        Get weight difference between x and y if they're connected.
        
        Args:
            x, y: Elements to compare
            
        Returns:
            Weight difference (y's weight - x's weight)
            
        Raises:
            ValueError: If x and y are not in the same component
        """
        if self.find(x) != self.find(y):
            raise ValueError("Elements are not in the same component")
        
        return self.weight[y] - self.weight[x]
    
    def connected(self, x: int, y: int) -> bool:
        """Check if elements x and y are connected."""
        return self.find(x) == self.find(y)


class UnionFindWithRollback:
    """
    Union-Find with rollback functionality.
    
    Supports undoing the last few union operations, useful for algorithms
    that need to try different combinations of unions.
    """
    
    def __init__(self, n: int):
        """Initialize Union-Find with rollback capability."""
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
        self.num_components = n
        self.history = []  # Stack of operations for rollback
    
    def find(self, x: int) -> int:
        """Find without path compression to enable rollback."""
        while self.parent[x] != x:
            x = self.parent[x]
        return x
    
    def union(self, x: int, y: int) -> bool:
        """Union with history tracking for rollback."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            # Save null operation for consistent history
            self.history.append(None)
            return False
        
        # Ensure root_x has smaller or equal rank
        if self.rank[root_x] > self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        # Save state before modification
        self.history.append((root_x, self.parent[root_x], self.rank[root_y]))
        
        self.parent[root_x] = root_y
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_y] += 1
        
        self.num_components -= 1
        return True
    
    def rollback(self) -> None:
        """Undo the last union operation."""
        if not self.history:
            raise RuntimeError("No operations to rollback")
        
        last_op = self.history.pop()
        
        if last_op is None:
            # Was a null operation (elements already connected)
            return
        
        node, old_parent, old_rank = last_op
        self.parent[node] = old_parent
        
        # Find the root that had its rank potentially increased
        root = node if old_parent == node else self.find(old_parent)
        if self.rank[root] != old_rank:
            self.rank[root] = old_rank
        
        self.num_components += 1
    
    def connected(self, x: int, y: int) -> bool:
        """Check if elements are connected."""
        return self.find(x) == self.find(y)
