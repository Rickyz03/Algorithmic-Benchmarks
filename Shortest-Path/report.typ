#set document(
  title: "Advanced Algorithms: Shortest Path Comparison Study",
  author: "Riccardo",
  date: datetime.today()
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2cm),
  numbering: "1",
  number-align: center,
)

#set text(
  font: "Linux Libertine",
  size: 11pt,
  lang: "en"
)

#set heading(numbering: "1.")

#set par(justify: true, leading: 0.55em)

#show link: underline

// Title page
#align(center)[
  #text(20pt, weight: "bold")[
    Advanced Algorithms: \
    Shortest Path Comparison Study
  ]
  
  #v(2cm)
  
  #text(16pt)[
    A Comprehensive Analysis of Dijkstra's Algorithm, \
    Bellman-Ford Algorithm, and A* Search
  ]
  
  #v(3cm)
  
  #text(14pt)[
    Author: Riccardo \
    Date: #datetime.today().display()
  ]
  
  #v(2cm)
  
  #text(12pt, style: "italic")[
    Implementation and Performance Comparison \
    of Three Fundamental Shortest Path Algorithms
  ]
]

#pagebreak()

// Table of Contents
#outline(
  title: "Table of Contents",
  indent: auto
)

#pagebreak()

= Introduction

The shortest path problem is one of the most fundamental and well-studied problems in computer science and graph theory. Given a weighted graph and two vertices, the goal is to find a path between them that minimizes the sum of edge weights. This problem has numerous practical applications, from GPS navigation systems to network routing protocols, from game AI pathfinding to social network analysis.

This study implements and analyzes three classical algorithms for solving shortest path problems:

- *Dijkstra's Algorithm*: The gold standard for graphs with non-negative edge weights
- *Bellman-Ford Algorithm*: A versatile algorithm that handles negative edge weights and detects negative cycles  
- *A* Search*: An informed search algorithm that uses heuristics to guide the search toward the goal

Each algorithm represents a different approach to the problem, with distinct strengths, limitations, and computational complexities. Through systematic implementation and empirical analysis, this project aims to provide insights into their theoretical properties and practical performance characteristics.

== Objectives

The primary objectives of this study are:

1. Implement all three algorithms with comprehensive documentation and performance monitoring
2. Generate diverse test instances including sparse graphs, dense graphs, grids, and graphs with negative edges
3. Conduct systematic performance benchmarking across different graph types and sizes
4. Analyze the relationship between theoretical complexity and practical performance
5. Visualize and interpret the results to gain deeper insights into algorithm behavior

== Project Structure

The implementation is organized into modular components:

- `graph.py`: Graph representation and test instance generation utilities
- `dijkstra.py`: Complete implementation of Dijkstra's algorithm with performance metrics
- `bellman_ford.py`: Bellman-Ford algorithm with negative cycle detection
- `astar.py`: A* search for both general graphs and grid-based pathfinding
- `main.py`: Comprehensive benchmarking framework with visualization capabilities

= Theoretical Background

== Dijkstra's Algorithm

Dijkstra's algorithm, proposed by Edsger W. Dijkstra in 1956, is a greedy algorithm that finds the shortest path from a source vertex to all other vertices in a weighted graph. The algorithm maintains the invariant that once a vertex is processed, its shortest distance from the source is known and will never change.

=== Algorithm Description

The algorithm maintains a set $S$ of vertices whose shortest distance from the source $s$ has been determined, and a priority queue $Q$ of vertices to be processed. Initially, $S = emptyset$ and $Q$ contains all vertices with distance $d[s] = 0$ and $d[v] = infinity$ for all $v != s$.

At each iteration, the algorithm:
1. Extracts the vertex $u$ with minimum distance from $Q$
2. Adds $u$ to $S$ 
3. For each neighbor $v$ of $u$, performs the *relaxation* operation:
   $d[v] = min(d[v], d[u] + w(u,v))$

The relaxation operation is the key insight: if we can reach vertex $v$ through $u$ with a shorter total distance, we update $v$'s distance estimate.

=== Mathematical Formulation

Let $G = (V, E)$ be a weighted directed graph with weight function $w: E -> RR^+$. For a source vertex $s in V$, Dijkstra's algorithm computes:

$ delta(s, v) = min{sum_(e in P) w(e) : P "is a path from" s "to" v} $

The algorithm's correctness relies on the *optimal substructure* property: if $P$ is a shortest path from $s$ to $v$, then any subpath of $P$ is also a shortest path between its endpoints.

=== Complexity Analysis

*Time Complexity*: The algorithm's performance depends on the implementation of the priority queue:
- With a binary heap: $O((|V| + |E|) log |V|)$
- With a Fibonacci heap: $O(|E| + |V| log |V|)$

*Space Complexity*: $O(|V|)$ for storing distances and predecessors

=== Limitations

Dijkstra's algorithm requires all edge weights to be non-negative. This restriction ensures that the greedy choice (always selecting the vertex with minimum current distance) leads to globally optimal solutions. With negative edges, this greedy property fails, and the algorithm may produce incorrect results.

== Bellman-Ford Algorithm

The Bellman-Ford algorithm, developed by Richard Bellman and Lester Ford Jr., solves the single-source shortest path problem for graphs that may contain negative edge weights. Unlike Dijkstra's algorithm, it can handle negative weights and detect the presence of negative cycles.

=== Algorithm Description

The Bellman-Ford algorithm uses dynamic programming principles and operates in two phases:

*Phase 1 - Relaxation*: The algorithm performs $|V| - 1$ iterations, where each iteration relaxes all edges in the graph. This systematic relaxation ensures that shortest paths of length $k$ are found in the $k$-th iteration.

*Phase 2 - Negative Cycle Detection*: After $|V| - 1$ iterations, the algorithm performs one additional iteration. If any distance can still be improved, a negative cycle exists.

The relaxation operation is identical to Dijkstra's:
$ d[v] = min(d[v], d[u] + w(u,v)) $

However, instead of using a priority queue, Bellman-Ford systematically examines all edges in each iteration.

=== Mathematical Foundation

For a graph $G = (V, E)$ with source $s$, if no negative cycles are reachable from $s$, then after $|V| - 1$ iterations:
$ d[v] = delta(s, v) quad forall v in V $

This correctness follows from the fact that any shortest path contains at most $|V| - 1$ edges (since it cannot repeat vertices without forming a cycle).

The negative cycle detection works because if a negative cycle exists, distances can be improved indefinitely, violating the convergence property.

=== Complexity Analysis

*Time Complexity*: $O(|V| times |E|)$
- $|V| - 1$ iterations of relaxing all $|E|$ edges
- One additional iteration for cycle detection

*Space Complexity*: $O(|V|)$ for distance and predecessor arrays

=== Advantages and Applications

- Handles negative edge weights correctly
- Detects negative cycles reachable from the source
- Simpler implementation than Dijkstra (no priority queue needed)
- Can be easily parallelized or distributed
- Works with any edge ordering

== A* Search Algorithm

A* (pronounced "A-star") is an informed search algorithm that uses heuristics to guide the search toward the goal more efficiently than uninformed algorithms like Dijkstra's algorithm.

=== Algorithm Description

A* maintains two functions for each vertex $v$:
- $g(v)$: The actual cost from the start vertex to $v$
- $h(v)$: The heuristic estimate of cost from $v$ to the goal
- $f(v) = g(v) + h(v)$: The estimated total cost of a path through $v$

The algorithm uses $f(v)$ to prioritize which vertices to explore, always selecting the vertex with the lowest $f$-value.

=== Heuristic Properties

For A* to guarantee optimal solutions, the heuristic function must be:

*Admissible*: $h(v) <= h^*(v)$ where $h^*(v)$ is the true cost from $v$ to goal
$ h(v) <= delta(v, "goal") quad forall v in V $

*Consistent (Monotonic)*: $h(u) <= w(u,v) + h(v)$ for every edge $(u,v)$

When these properties hold, A* is guaranteed to find optimal solutions while exploring fewer vertices than Dijkstra's algorithm.

=== Mathematical Formulation

A* maintains the invariant that for any vertex $v$ in the closed set:
$ g(v) = delta("start", v) $

The algorithm terminates when the goal vertex is selected for expansion, ensuring optimality under admissible heuristics.

=== Complexity Analysis

*Time Complexity*: $O(b^d)$ where $b$ is the branching factor and $d$ is the depth of the solution
- In practice, much better than this worst-case bound with good heuristics
- Reduces to Dijkstra's complexity with $h(v) = 0$

*Space Complexity*: $O(b^d)$ for storing the open and closed sets

=== Grid-Specific Optimizations

For grid-based pathfinding, A* commonly uses:
- *Manhattan Distance*: $h((x_1, y_1), (x_2, y_2)) = |x_1 - x_2| + |y_1 - y_2|$
- *Euclidean Distance*: $h((x_1, y_1), (x_2, y_2)) = sqrt((x_1 - x_2)^2 + (y_1 - y_2)^2)$

Manhattan distance is admissible for 4-directional movement, while Euclidean distance works for 8-directional or continuous movement.

= Implementation Details

== Graph Representation

The implementation uses an adjacency list representation stored in a Python dictionary:

```python
edges: Dict[int, List[Tuple[int, float]]] = {}
```

This representation provides:
- $O(1)$ vertex addition
- $O(1)$ edge addition  
- $O(d)$ neighbor enumeration where $d$ is vertex degree
- Memory efficiency for sparse graphs

== Priority Queue Implementation

Both Dijkstra's algorithm and A* use Python's `heapq` module, which implements a binary min-heap:
- `heappush()`: $O(log n)$
- `heappop()`: $O(log n)$ 
- Space: $O(n)$

== Performance Monitoring

Each algorithm implementation includes comprehensive performance metrics:
- Operation counting for algorithm-specific operations
- Vertex visitation tracking
- Path reconstruction with validation
- Memory usage monitoring

== Test Instance Generation

The benchmarking framework generates diverse test cases:

*Random Graphs*:
- Erdős–Rényi model with configurable edge probability
- Controllable weight distributions
- Optional negative edges with specified probability

*Grid Graphs*:
- 2D grids with configurable dimensions
- Random obstacle placement
- Specialized for A* pathfinding evaluation

= Experimental Results and Analysis

== Performance Comparison...

_[Results will be populated after running the benchmarks]_

== Algorithm-Specific Analysis...

_[Detailed analysis will be added after examining the experimental data]_

== Practical Implications...  

_[Discussion of when to use each algorithm will be included after data analysis]_

= Conclusions

## Summary of Findings...

_[Conclusions will be drawn from the experimental results]_

## Future Work...

_[Potential extensions and improvements will be discussed]_

= References

1. Dijkstra, E. W. (1959). "A note on two problems in connexion with graphs". _Numerische Mathematik_, 1(1), 269-271.

2. Bellman, R. (1958). "On a routing problem". _Quarterly of Applied Mathematics_, 16(1), 87-90.

3. Ford Jr., L. R. (1956). "Network Flow Theory". RAND Corporation.

4. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). "A formal basis for the heuristic determination of minimum cost paths". _IEEE Transactions on Systems Science and Cybernetics_, 4(2), 100-107.

5. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). _Introduction to Algorithms_ (3rd ed.). MIT Press.

6. Sedgewick, R., & Wayne, K. (2011). _Algorithms_ (4th ed.). Addison-Wesley Professional.
