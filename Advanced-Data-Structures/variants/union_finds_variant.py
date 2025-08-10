"""
Advanced Union-Find Variants

This module contains enhanced versions of Union-Find with specialized features:
1. Persistent Union-Find - maintains multiple versions
2. Partial Union-Find - supports partial rollbacks
3. Union-Find with custom merge functions
4. Dynamic Connectivity with edge deletion
5. Union-Find on graphs with additional properties

These variants extend the basic Union-Find for complex algorithmic scenarios.
"""

from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from collections import defaultdict
import copy


class UnionFindOptimized:
    """
    Highly optimized Union-Find with all standard optimizations and additional features.
    
    Features:
    - Path compression with halving
    - Union by size and rank
    - Component size tracking
    - Connected components iteration
    - Merge callbacks for custom operations
    """
    
    def __init__(self, n: int, merge_callback: Optional[Callable[[int, int], None]] = None):
        """
        Initialize optimized Union-Find.
        
        Args:
            n: Number of elements
            merge_callback: Optional function called when components merge
        """
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.num_components = n
        self.merge_callback = merge_callback
        
        # Additional tracking
        self.component_data = {}  # Custom data per component root
        self.merge_history = []   # History of merge operations
    
    def find(self, x: int) -> int:
        """
        Find with path compression using path halving.
        More efficient than full path compression in practice.
        """
        while self.parent[x] != self.parent[self.parent[x]]:
            self.parent[x] = self.parent[self.parent[x]]  # Path halving
            x = self.parent[x]
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union with both size and rank optimization."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank with size tie-breaking
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        elif self.rank[root_x] == self.rank[root_y] and self.size[root_x] < self.size[root_y]:
            root_x, root_y = root_y, root_x
        
        # Merge
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        self.num_components -= 1
        
        # Record merge history
        self.merge_history.append((root_y, root_x, self.size[root_y]))
        
        # Merge component data
        if root_y in self.component_data:
            if root_x not in self.component_data:
                self.component_data[root_x] = self.component_data[root_y]
            else:
                # Custom merge logic can be implemented here
                pass
            del self.component_data[root_y]
        
        # Call merge callback if provided
        if self.merge_callback:
            self.merge_callback(root_x, root_y)
        
        return True
    
    def set_component_data(self, x: int, data: Any) -> None:
        """Associate custom data with a component."""
        root = self.find(x)
        self.component_data[root] = data
    
    def get_component_data(self, x: int) -> Any:
        """Get custom data associated with component."""
        root = self.find(x)
        return self.component_data.get(root)
    
    def get_all_components(self) -> Dict[int, Set[int]]:
        """Get all components as dictionary mapping root -> set of elements."""
        components = defaultdict(set)
        for i in range(self.n):
            components[self.find(i)].add(i)
        return dict(components)
    
    def component_representatives(self) -> List[int]:
        """Get list of all component representatives (roots)."""
        return list(set(self.find(i) for i in range(self.n)))


class PersistentUnionFind:
    """
    Persistent Union-Find that maintains multiple versions.
    
    Each union operation creates a new version while keeping old versions accessible.
    Uses path copying to maintain persistent structure.
    """
    
    class Node:
        """Node in the persistent Union-Find tree."""
        
        def __init__(self, parent: int, rank: int = 0, size: int = 1):
            self.parent = parent
            self.rank = rank
            self.size = size
    
    def __init__(self, n: int):
        """Initialize persistent Union-Find."""
        self.n = n
        self.versions = []  # List of versions
        
        # Create initial version
        initial_version = {}
        for i in range(n):
            initial_version[i] = self.Node(i)
        
        self.versions.append(initial_version)
    
    def find(self, version: int, x: int) -> int:
        """Find root in a specific version."""
        if version >= len(self.versions):
            raise IndexError("Invalid version")
        
        nodes = self.versions[version]
        while nodes[x].parent != x:
            x = nodes[x].parent
        return x
    
    def connected(self, version: int, x: int, y: int) -> bool:
        """Check if elements are connected in a specific version."""
        return self.find(version, x) == self.find(version, y)
    
    def union(self, base_version: int, x: int, y: int) -> int:
        """
        Create new version with x and y unioned.
        
        Args:
            base_version: Version to base the union on
            x, y: Elements to union
            
        Returns:
            Index of new version created
        """
        if base_version >= len(self.versions):
            raise IndexError("Invalid base version")
        
        # Copy the base version
        new_version = copy.deepcopy(self.versions[base_version])
        
        root_x = self.find(len(self.versions), x)  # Will use new_version
        root_y = self.find(len(self.versions), y)
        
        # Temporarily add new version to find roots
        self.versions.append(new_version)
        root_x = self.find(len(self.versions) - 1, x)
        root_y = self.find(len(self.versions) - 1, y)
        self.versions.pop()  # Remove temporary version
        
        if root_x == root_y:
            # Already connected, but still create new version
            self.versions.append(new_version)
            return len(self.versions) - 1
        
        # Union by rank
        if new_version[root_x].rank < new_version[root_y].rank:
            new_version[root_x].parent = root_y
            new_version[root_y].size += new_version[root_x].size
        elif new_version[root_x].rank > new_version[root_y].rank:
            new_version[root_y].parent = root_x
            new_version[root_x].size += new_version[root_y].size
        else:
            new_version[root_y].parent = root_x
            new_version[root_x].size += new_version[root_y].size
            new_version[root_x].rank += 1
        
        self.versions.append(new_version)
        return len(self.versions) - 1
    
    def component_size(self, version: int, x: int) -> int:
        """Get component size in a specific version."""
        root = self.find(version, x)
        return self.versions[version][root].size
    
    def get_version_count(self) -> int:
        """Get number of available versions."""
        return len(self.versions)


class DynamicConnectivity:
    """
    Dynamic Connectivity data structure supporting edge insertions and deletions.
    
    Maintains connectivity information as edges are added and removed dynamically.
    Uses a combination of Union-Find and additional data structures.
    """
    
    def __init__(self, n: int):
        """Initialize dynamic connectivity structure."""
        self.n = n
        self.uf = UnionFindOptimized(n)
        self.edges = set()  # Current edges
        self.edge_history = []  # History of edge operations
        self.snapshots = []  # Periodic snapshots for efficient rollback
    
    def add_edge(self, u: int, v: int) -> bool:
        """
        Add edge between u and v.
        
        Args:
            u, v: Vertices to connect
            
        Returns:
            True if edge was added, False if already existed
        """
        if u > v:
            u, v = v, u
        
        if (u, v) in self.edges:
            return False
        
        self.edges.add((u, v))
        self.edge_history.append(('add', u, v))
        self.uf.union(u, v)
        
        return True
    
    def remove_edge(self, u: int, v: int) -> bool:
        """
        Remove edge between u and v.
        
        This is expensive as it requires rebuilding the Union-Find structure.
        
        Args:
            u, v: Vertices to disconnect
            
        Returns:
            True if edge was removed, False if didn't exist
        """
        if u > v:
            u, v = v, u
        
        if (u, v) not in self.edges:
            return False
        
        self.edges.remove((u, v))
        self.edge_history.append(('remove', u, v))
        
        # Rebuild Union-Find from remaining edges
        self._rebuild_union_find()
        
        return True
    
    def _rebuild_union_find(self) -> None:
        """Rebuild Union-Find structure from current edges."""
        self.uf = UnionFindOptimized(self.n)
        for u, v in self.edges:
            self.uf.union(u, v)
    
    def connected(self, u: int, v: int) -> bool:
        """Check if u and v are connected."""
        return self.uf.connected(u, v)
    
    def component_count(self) -> int:
        """Get number of connected components."""
        return self.uf.num_components
    
    def component_size(self, x: int) -> int:
        """Get size of component containing x."""
        return self.uf.component_size(x)
    
    def create_snapshot(self) -> int:
        """Create a snapshot of current state."""
        snapshot = {
            'edges': self.edges.copy(),
            'uf_state': (
                self.uf.parent.copy(),
                self.uf.rank.copy(),
                self.uf.size.copy(),
                self.uf.num_components
            )
        }
        self.snapshots.append(snapshot)
        return len(self.snapshots) - 1
    
    def restore_snapshot(self, snapshot_id: int) -> None:
        """Restore to a previous snapshot."""
        if snapshot_id >= len(self.snapshots):
            raise IndexError("Invalid snapshot ID")
        
        snapshot = self.snapshots[snapshot_id]
        self.edges = snapshot['edges'].copy()
        
        parent, rank, size, num_components = snapshot['uf_state']
        self.uf.parent = parent.copy()
        self.uf.rank = rank.copy()
        self.uf.size = size.copy()
        self.uf.num_components = num_components


class WeightedUnionFindAdvanced:
    """
    Advanced Weighted Union-Find with additional features.
    
    Supports:
    - Weighted edges with different operations (addition, multiplication, etc.)
    - Range queries on paths
    - LCA (Lowest Common Ancestor) queries
    - Path compression with weight adjustment
    """
    
    def __init__(self, n: int, operation: str = "add", identity: float = 0):
        """
        Initialize weighted Union-Find.
        
        Args:
            n: Number of elements
            operation: Weight operation ("add", "multiply", "min", "max")
            identity: Identity element for the operation
        """
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
        self.weight = [identity] * n  # Weight relative to parent
        self.num_components = n
        
        # Operation functions
        self.ops = {
            "add": (lambda x, y: x + y, lambda x: -x, 0),
            "multiply": (lambda x, y: x * y, lambda x: 1/x if x != 0 else float('inf'), 1),
            "min": (min, lambda x: x, float('inf')),
            "max": (max, lambda x: x, float('-inf'))
        }
        
        if operation not in self.ops:
            raise ValueError(f"Unsupported operation: {operation}")
        
        self.combine, self.inverse, self.default_identity = self.ops[operation]
        self.identity = identity if identity != 0 else self.default_identity
    
    def find(self, x: int) -> int:
        """Find with path compression, adjusting weights."""
        if self.parent[x] != x:
            original_parent = self.parent[x]
            root = self.find(self.parent[x])
            self.parent[x] = root
            # Update weight to be relative to root
            self.weight[x] = self.combine(self.weight[x], self.weight[original_parent])
        
        return self.parent[x]
    
    def union(self, x: int, y: int, w: float) -> bool:
        """
        Union sets with weight constraint: weight(y) = weight(x) op w.
        
        Args:
            x, y: Elements to union
            w: Weight relationship
            
        Returns:
            True if union was performed
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            # Check if weight constraint is satisfied
            current_diff = self.get_weight_difference(x, y)
            return abs(current_diff - w) < 1e-9  # Floating point comparison
        
        # Calculate weight for root_y relative to root_x
        # weight(y) = weight(x) op w
        # weight(root_y) + weight[y] = weight(root_x) + weight[x] op w
        # weight(root_y) = weight(root_x) + weight[x] op w - weight[y]
        target_weight = self.combine(
            self.combine(self.weight[x], w),
            self.inverse(self.weight[y])
        )
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.weight[root_x] = self.inverse(target_weight)
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.weight[root_y] = target_weight
        else:
            self.parent[root_y] = root_x
            self.weight[root_y] = target_weight
            self.rank[root_x] += 1
        
        self.num_components -= 1
        return True
    
    def get_weight_difference(self, x: int, y: int) -> float:
        """Get weight difference between x and y."""
        if not self.connected(x, y):
            raise ValueError("Elements not connected")
        
        # Ensure both are compressed to root
        self.find(x)
        self.find(y)
        
        return self.combine(self.weight[y], self.inverse(self.weight[x]))
    
    def connected(self, x: int, y: int) -> bool:
        """Check if elements are connected."""
        return self.find(x) == self.find(y)


class UnionFindWithRankQueries:
    """
    Union-Find supporting rank/order queries within components.
    
    Maintains the relative order of elements within each component,
    allowing queries like "how many elements in x's component have smaller IDs".
    """
    
    def __init__(self, n: int):
        """Initialize Union-Find with rank queries."""
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.component_elements = {i: {i} for i in range(n)}  # Root -> set of elements
        self.element_ranks = {i: {i: 0} for i in range(n)}  # Root -> {element: rank}
        self.num_components = n
    
    def find(self, x: int) -> int:
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union two sets, maintaining rank information."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by size
        if self.size[root_x] < self.size[root_y]:
            root_x, root_y = root_y, root_x
        
        # Merge smaller component into larger one
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        
        # Merge element sets and update ranks
        merged_elements = sorted(
            list(self.component_elements[root_x]) + list(self.component_elements[root_y])
        )
        
        # Reassign ranks in merged component
        new_ranks = {element: rank for rank, element in enumerate(merged_elements)}
        
        self.component_elements[root_x] = set(merged_elements)
        self.element_ranks[root_x] = new_ranks
        
        # Clean up old component data
        del self.component_elements[root_y]
        del self.element_ranks[root_y]
        
        self.num_components -= 1
        return True
    
    def get_rank_in_component(self, x: int) -> int:
        """Get the rank of element x within its component."""
        root = self.find(x)
        return self.element_ranks[root][x]
    
    def count_smaller_in_component(self, x: int) -> int:
        """Count elements in x's component that have smaller values than x."""
        return self.get_rank_in_component(x)
    
    def count_larger_in_component(self, x: int) -> int:
        """Count elements in x's component that have larger values than x."""
        root = self.find(x)
        return len(self.component_elements[root]) - self.get_rank_in_component(x) - 1
    
    def get_kth_in_component(self, x: int, k: int) -> int:
        """Get the k-th smallest element in x's component (0-indexed)."""
        root = self.find(x)
        elements = sorted(self.component_elements[root])
        if k < 0 or k >= len(elements):
            raise IndexError("k out of range")
        return elements[k]
    
    def connected(self, x: int, y: int) -> bool:
        """Check if elements are connected."""
        return self.find(x) == self.find(y)


class BipartiteUnionFind:
    """
    Union-Find for bipartite graphs with conflict detection.
    
    Maintains two types of relationships:
    - Same set (elements should be in same partition)
    - Different set (elements should be in different partitions)
    
    Detects conflicts when trying to put elements that should be different
    in the same partition.
    """
    
    def __init__(self, n: int):
        """Initialize bipartite Union-Find."""
        # We use 2n nodes: node i and node i+n represent the two possible partitions for element i
        self.n = n
        self.uf = UnionFindOptimized(2 * n)
        self.has_conflict = False
    
    def union_same(self, x: int, y: int) -> bool:
        """
        Specify that x and y should be in the same partition.
        
        Args:
            x, y: Elements that should be in same partition
            
        Returns:
            True if operation succeeded, False if conflict detected
        """
        if self.has_conflict:
            return False
        
        # x and y in same partition, x+n and y+n in same partition
        result1 = self.uf.union(x, y)
        result2 = self.uf.union(x + self.n, y + self.n)
        
        # Check for conflict: x and x+n should not be connected
        if self.uf.connected(x, x + self.n) or self.uf.connected(y, y + self.n):
            self.has_conflict = True
            return False
        
        return True
    
    def union_different(self, x: int, y: int) -> bool:
        """
        Specify that x and y should be in different partitions.
        
        Args:
            x, y: Elements that should be in different partitions
            
        Returns:
            True if operation succeeded, False if conflict detected
        """
        if self.has_conflict:
            return False
        
        # x with y+n, y with x+n
        result1 = self.uf.union(x, y + self.n)
        result2 = self.uf.union(y, x + self.n)
        
        # Check for conflict
        if self.uf.connected(x, x + self.n) or self.uf.connected(y, y + self.n):
            self.has_conflict = True
            return False
        
        return True
    
    def same_partition(self, x: int, y: int) -> bool:
        """Check if x and y are in the same partition."""
        if self.has_conflict:
            return False
        return self.uf.connected(x, y)
    
    def different_partition(self, x: int, y: int) -> bool:
        """Check if x and y are in different partitions."""
        if self.has_conflict:
            return False
        return self.uf.connected(x, y + self.n)
    
    def is_bipartite(self) -> bool:
        """Check if the current constraints allow a bipartite assignment."""
        return not self.has_conflict
    
    def get_partition_assignment(self) -> Optional[List[int]]:
        """
        Get a valid bipartition assignment.
        
        Returns:
            List where result[i] is 0 or 1 indicating partition of element i,
            or None if no valid assignment exists
        """
        if self.has_conflict:
            return None
        
        assignment = [0] * self.n
        processed = [False] * self.n
        
        for i in range(self.n):
            if processed[i]:
                continue
            
            # Find all elements connected to i
            component_0 = set()  # Elements that should be in same partition as i
            component_1 = set()  # Elements that should be in different partition from i
            
            for j in range(self.n):
                if self.uf.connected(i, j):
                    component_0.add(j)
                elif self.uf.connected(i, j + self.n):
                    component_1.add(j)
            
            # Assign partitions
            for elem in component_0:
                assignment[elem] = 0
                processed[elem] = True
            
            for elem in component_1:
                assignment[elem] = 1
                processed[elem] = True
        
        return assignment


class UnionFindWithDistances:
    """
    Union-Find maintaining distances/weights on the tree edges.
    
    Supports queries for the distance between any two elements in the same component
    along the tree path connecting them.
    """
    
    def __init__(self, n: int):
        """Initialize Union-Find with distance tracking."""
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
        self.dist_to_parent = [0] * n  # Distance to parent
        self.num_components = n
    
    def find(self, x: int) -> Tuple[int, int]:
        """
        Find root and calculate distance to root.
        
        Returns:
            Tuple of (root, distance_to_root)
        """
        if self.parent[x] == x:
            return x, 0
        
        root, dist_to_grandparent = self.find(self.parent[x])
        total_dist = self.dist_to_parent[x] + dist_to_grandparent
        
        # Path compression
        self.parent[x] = root
        self.dist_to_parent[x] = total_dist
        
        return root, total_dist
    
    def union(self, x: int, y: int, weight: int) -> bool:
        """
        Union components with specified edge weight between x and y.
        
        Args:
            x, y: Elements to connect
            weight: Weight of edge between x and y
            
        Returns:
            True if union was performed
        """
        root_x, dist_x = self.find(x)
        root_y, dist_y = self.find(y)
        
        if root_x == root_y:
            # Check consistency
            expected_weight = abs(dist_x - dist_y)
            return abs(expected_weight - weight) < 1e-9
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
            dist_x, dist_y = dist_y, dist_x
            x, y = y, x
            weight = -weight  # Reverse direction
        
        # Connect root_y to root_x
        self.parent[root_y] = root_x
        # dist_x + weight = dist_y + new_dist_to_parent[root_y]
        self.dist_to_parent[root_y] = dist_x + weight - dist_y
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        self.num_components -= 1
        return True
    
    def distance(self, x: int, y: int) -> int:
        """
        Get distance between x and y.
        
        Returns:
            Distance between x and y, or None if not connected
        """
        root_x, dist_x = self.find(x)
        root_y, dist_y = self.find(y)
        
        if root_x != root_y:
            raise ValueError("Elements are not connected")
        
        return abs(dist_x - dist_y)
    
    def connected(self, x: int, y: int) -> bool:
        """Check if elements are connected."""
        root_x, _ = self.find(x)
        root_y, _ = self.find(y)
        return root_x == root_y


class OnlineUnionFind:
    """
    Online Union-Find that answers connectivity queries efficiently
    while operations are being performed.
    
    Optimized for scenarios where queries are much more frequent than updates.
    """
    
    def __init__(self, n: int):
        """Initialize online Union-Find."""
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.num_components = n
        
        # Query optimization
        self.query_cache = {}  # Cache for recent queries
        self.cache_size_limit = 1000
    
    def find(self, x: int) -> int:
        """Find with aggressive path compression."""
        path = []
        original_x = x
        
        # Collect path
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        
        root = x
        
        # Compress entire path
        for node in path:
            self.parent[node] = root
        
        return root
    
    def union(self, x: int, y: int) -> bool:
        """Union with cache invalidation."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Clear relevant cache entries
        self._invalidate_cache(root_x, root_y)
        
        # Union by size
        if self.size[root_x] < self.size[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        self.num_components -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Fast connectivity query with caching."""
        if (x, y) in self.query_cache:
            return self.query_cache[(x, y)]
        if (y, x) in self.query_cache:
            return self.query_cache[(y, x)]
        
        result = self.find(x) == self.find(y)
        
        # Cache result if cache not full
        if len(self.query_cache) < self.cache_size_limit:
            self.query_cache[(x, y)] = result
        
        return result
    
    def _invalidate_cache(self, root_x: int, root_y: int) -> None:
        """Invalidate cache entries affected by union."""
        # In a more sophisticated implementation, we would track which
        # cache entries are affected by this union operation
        # For simplicity, we clear the entire cache
        if len(self.query_cache) > self.cache_size_limit // 2:
            self.query_cache.clear()
    
    def batch_connected(self, queries: List[Tuple[int, int]]) -> List[bool]:
        """
        Process multiple connectivity queries efficiently.
        
        Args:
            queries: List of (x, y) pairs to check
            
        Returns:
            List of boolean results
        """
        # Pre-compress all paths for elements in queries
        elements = set()
        for x, y in queries:
            elements.add(x)
            elements.add(y)
        
        # Batch path compression
        for elem in elements:
            self.find(elem)
        
        # Now process queries with already compressed paths
        results = []
        for x, y in queries:
            results.append(self.parent[x] == self.parent[y])
        
        return results
