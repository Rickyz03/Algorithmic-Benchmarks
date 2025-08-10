#import "@preview/algorithmic:0.1.0": algorithm
#import "@preview/codelst:2.0.1": sourcecode

#set document(title: "Advanced Data Structures: A Comprehensive Analysis")
#set page(numbering: "1", margin: (x: 1.5cm, y: 2cm))
#set text(font: "Linux Libertine", size: 11pt)
#set heading(numbering: "1.1")

#show math.equation: set text(weight: 400)
#show raw: set text(font: "Fira Code")

#align(center)[
  #text(size: 20pt, weight: "bold")[
    Advanced Data Structures: A Comprehensive Analysis
  ]
  
  #v(1cm)
  
  #text(size: 14pt)[
    Segment Trees, Fenwick Trees, and Union-Find Structures
  ]
  
  #v(0.5cm)
  
  #text(size: 12pt)[
    Implementation, Analysis, and Performance Comparison
  ]
  
  #v(2cm)
  
  #text(size: 12pt)[
    #datetime.today().display()
  ]
]

#pagebreak()

#outline()

#pagebreak()

= Introduction

This report presents a comprehensive analysis of three fundamental advanced data structures: Segment Trees, Fenwick Trees (Binary Indexed Trees), and Union-Find (Disjoint Set Union) structures. These data structures are essential tools in competitive programming and algorithm optimization, enabling efficient solutions to problems involving range queries, dynamic connectivity, and aggregate computations.

The primary objectives of this analysis are:

1. *Implementation*: Develop robust, well-documented implementations of each data structure from first principles
2. *Theoretical Analysis*: Provide rigorous mathematical analysis of time and space complexity
3. *Practical Comparison*: Benchmark performance characteristics across different problem sizes and operation types  
4. *Advanced Variants*: Explore sophisticated extensions including lazy propagation, persistence, and specialized optimizations

Each structure addresses distinct algorithmic challenges:

- *Segment Trees* excel at range queries with arbitrary associative operations
- *Fenwick Trees* provide memory-efficient prefix sum computations  
- *Union-Find* enables near-constant time dynamic connectivity queries

Through systematic implementation and empirical analysis, this study aims to provide clear guidance on when and how to apply these powerful data structures effectively.

= Theoretical Foundations

== Segment Trees

=== Definition and Structure

A Segment Tree is a binary tree data structure that stores information about array segments in its nodes. For an array of size $n$, the segment tree is a complete binary tree with the following properties:

- Each leaf node represents a single array element
- Each internal node represents the union of its children's segments
- The root represents the entire array range $[0, n-1]$

Formally, for a node representing segment $[l, r]$:
- If $l = r$, it's a leaf storing $"arr"[l]$
- Otherwise, it has children representing $[l, m]$ and $[m+1, r]$ where $m = floor((l+r)/2)$

=== Mathematical Foundation

The segment tree supports any *associative* binary operation $circle.tiny$. For operation to be valid:

$ (a circle.tiny b) circle.tiny c = a circle.tiny (b circle.tiny c) $

Common operations include:
- *Sum*: $a + b$, identity: $0$
- *Minimum*: $min(a,b)$, identity: $+infinity$  
- *Maximum*: $max(a,b)$, identity: $-infinity$
- *GCD*: $gcd(a,b)$, identity: $0$
- *Bitwise XOR*: $a plus.circle b$, identity: $0$

For a segment $[l, r]$ and operation $circle.tiny$, the segment tree stores:
$ "node"[l,r] = "arr"[l] circle.tiny "arr"[l+1] circle.tiny dots circle.tiny "arr"[r] $

=== Time Complexity Analysis

*Tree Construction*:
The tree has height $ceil(log_2 n)$ and at most $4n$ nodes. Construction involves:

$ T(n) = 2T(n/2) + O(1) = O(n) $

*Range Query*:
For query $[q_l, q_r]$, we traverse at most $O(log n)$ levels, visiting $O(log n)$ nodes:

$ T_"query"(n) = O(log n) $

*Point Update*:
Updates propagate from leaf to root along a single path:

$ T_"update"(n) = O(log n) $

=== Space Complexity

The segment tree requires $O(4n) = O(n)$ space. More precisely:
- Complete binary tree with $n$ leaves has at most $2n-1$ internal nodes
- Total nodes ≤ $4n$ in array representation with 1-based indexing

== Fenwick Trees (Binary Indexed Trees)

=== Mathematical Principle

The Fenwick Tree exploits the binary representation of indices to achieve efficient prefix sum computation. For any positive integer $i$, define:

$ "LSB"(i) = i & (-i) $

This gives the value of the least significant bit of $i$.

=== Tree Structure

The Fenwick Tree is conceptually a forest of binary trees where:
- Node $i$ is responsible for interval $[i - "LSB"(i) + 1, i]$
- Parent of node $i$ is $i + "LSB"(i)$
- The tree implicitly represents prefix sums through clever indexing

For prefix sum computation:
$ "prefix_sum"(i) = sum_(j=1)^i "arr"[j] $

The key insight: any number can be expressed as a sum of powers of 2, corresponding to a path in the BIT structure.

=== Update Operation

To update index $i$ by delta $delta$:
```
while i ≤ n:
    tree[i] += delta
    i += LSB(i)
```

This propagates the change to all nodes responsible for ranges containing index $i$.

=== Query Operation  

To compute prefix sum up to index $i$:
```
result = 0
while i > 0:
    result += tree[i] 
    i -= LSB(i)
```

This traverses from index $i$ toward the root, accumulating partial sums.

=== Complexity Analysis

*Time Complexity*:
Both update and query operations traverse paths of length $O(log n)$:

$ T_"update"(n) = T_"query"(n) = O(log n) $

*Space Complexity*:
The BIT requires exactly $O(n)$ space - one array element per input element:

$ S(n) = O(n) $

This is more memory-efficient than segment trees.

=== Range Sum Formula

For range sum $[l, r]$:
$ "range_sum"(l, r) = "prefix_sum"(r) - "prefix_sum"(l-1) $

This reduces range queries to two prefix sum computations.

== Union-Find (Disjoint Set Union)

=== Abstract Data Type

Union-Find maintains a collection of disjoint sets $S_1, S_2, dots, S_k$ where:
- Each element belongs to exactly one set: $S_i sect S_j = emptyset$ for $i != j$  
- Union of all sets covers the universe: $union.big_(i=1)^k S_i = U$

=== Operations

The data structure supports two primary operations:
- $"Find"(x)$: Return representative of set containing $x$
- $"Union"(x, y)$: Merge sets containing $x$ and $y$

=== Forest Representation

Union-Find is implemented as a forest of rooted trees where:
- Each tree represents one disjoint set
- Root serves as the set representative
- $"parent"[i]$ points to parent of node $i$, with $"parent"["root"] = "root"$

=== Path Compression Optimization

During $"Find"(x)$, compress the path by making all nodes point directly to root:

```
function Find(x):
    if parent[x] ≠ x:
        parent[x] = Find(parent[x])  // Path compression
    return parent[x]
```

This flattens tree structure, reducing future query times.

=== Union by Rank Optimization

To maintain balanced trees, always attach smaller tree under root of larger tree:

```
function Union(x, y):
    rootX = Find(x)
    rootY = Find(y)
    if rank[rootX] < rank[rootY]:
        parent[rootX] = rootY
    else if rank[rootX] > rank[rootY]:  
        parent[rootY] = rootX
    else:
        parent[rootY] = rootX
        rank[rootX]++
```

=== Complexity Analysis with Optimizations

*Without optimizations*: $O(n)$ per operation in worst case

*With path compression only*: $O(log n)$ amortized

*With union by rank only*: $O(log n)$ worst case

*With both optimizations*: $O(alpha(n))$ amortized, where $alpha$ is the inverse Ackermann function

The inverse Ackermann function grows extremely slowly:
$ alpha(n) < 5 text(" for all practical values of ") n < 2^65536 $

This makes Union-Find operations effectively constant time in practice.

=== Amortized Analysis

Using the potential method, the amortized cost per operation with both optimizations is:

$ T_"amortized" = O(alpha(n)) $

The potential function accounts for the "savings" from path compression, which benefits future operations by flattening tree structures.

= Implementation Analysis

== Design Principles

Each data structure implementation follows consistent design principles:

*Generic Programming*: Segment trees support arbitrary associative operations through function parameters, enabling reuse across different problem domains.

*Memory Efficiency*: Implementations minimize memory overhead while maintaining performance. Fenwick trees achieve optimal $O(n)$ space usage.

*Error Handling*: Comprehensive bounds checking and input validation prevent runtime errors and provide clear error messages.

*Code Clarity*: Extensive documentation and meaningful variable names enhance readability and maintainability.

== Segment Tree Implementation Details

The generic segment tree implementation uses templates to support any associative operation:

```python
class SegmentTree(Generic[T]):
    def __init__(self, arr: List[T], operation: Callable[[T, T], T], identity: T):
        self.n = len(arr)
        self.tree = [identity] * (4 * self.n)
        self.operation = operation
        self.identity = identity
        self._build(arr, 0, 0, self.n - 1)
```

*Key implementation choices*:
- Array-based representation with 4n size for simplicity
- 0-based indexing for consistency with Python conventions  
- Recursive structure mirrors mathematical definition
- Identity elements enable proper handling of out-of-bounds queries

== Fenwick Tree Implementation Strategy

The Fenwick tree leverages bit manipulation for efficient index computation:

```python
def _lsb(self, x: int) -> int:
    """Get the least significant bit using bit manipulation."""
    return x & (-x)
```

*Critical implementation aspects*:
- 1-based internal indexing simplifies bit operations
- 0-based external interface maintains Python conventions
- Careful handling of boundary conditions prevents errors
- Support for both point updates and absolute value setting

== Union-Find Optimization Implementation

The optimized Union-Find combines path compression with union by rank:

```python
def find(self, x: int) -> int:
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])  # Path compression
    return self.parent[x]

def union(self, x: int, y: int) -> bool:
    root_x, root_y = self.find(x), self.find(y)
    if root_x == root_y:
        return False
    
    # Union by rank
    if self.rank[root_x] < self.rank[root_y]:
        root_x, root_y = root_y, root_x
    
    self.parent[root_y] = root_x
    if self.rank[root_x] == self.rank[root_y]:
        self.rank[root_x] += 1
    
    return True
```

*Optimization benefits*:
- Path compression reduces tree height over time
- Union by rank prevents degenerate linear chains
- Combined optimizations achieve near-constant amortized performance

== Advanced Variants

=== Lazy Propagation Segment Trees

For efficient range updates, lazy propagation defers updates until necessary:

*Lazy Update Mechanism*:
- Store pending updates in lazy array
- Push updates to children only when needed
- Reduces range update complexity from $O(n log n)$ to $O(log n)$

*Mathematical Foundation*:
For range update adding value $v$ to interval $[l, r]$:
$ "tree"[p] = "tree"[p] + v times ("segment_length") $

=== Persistent Data Structures

Persistent variants maintain multiple versions efficiently:

*Path Copying Strategy*:
- Only copy nodes along update path
- Share unchanged subtrees between versions
- Space complexity: $O(log n)$ per update

*Version Management*:
Each update creates new root while preserving old versions, enabling:
- Historical queries on previous states
- Efficient branching and merging of versions
- Undo operations without explicit rollback mechanisms

=== Weighted Union-Find

Extension supporting edge weights between connected components:

*Weight Maintenance*:
- Store relative weights to parent nodes
- Update weights during path compression
- Support queries for weight differences between connected elements

*Applications*:
- Bipartite graph checking
- Relative positioning problems
- Constraint satisfaction with linear relationships

= Results and Analysis

...

= Conclusions and Recommendations

...

#pagebreak()

= References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.

2. Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley Professional.

3. Fenwick, P. M. (1994). A new data structure for cumulative frequency tables. *Software: Practice and Experience*, 24(3), 327-336.

4. Tarjan, R. E. (1975). Efficiency of a good but not linear set union algorithm. *Journal of the ACM*, 22(2), 215-225.

5. Ackermann, W. (1928). Zum Hilbertschen Aufbau der reellen Zahlen. *Mathematische Annalen*, 99(1), 118-133.

= Appendices

== Appendix A: Complete Code Listings

[Code listings would be included here in the full report]

== Appendix B: Benchmark Data

[Detailed benchmark results and raw data would be included here]

== Appendix C: Mathematical Proofs

[Formal proofs of complexity bounds would be included here]
