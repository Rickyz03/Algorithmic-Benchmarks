#import "@preview/algorithmic:0.1.0": algorithm
#import "@preview/cetz:0.2.2": canvas, draw, tree
#import "@preview/fletcher:0.5.1" as fletcher: node, edge

#set document(
  title: "Maximum Flow Algorithms: Ford-Fulkerson vs Edmonds-Karp",
  author: "Advanced Algorithms Project",
  date: datetime.today()
)

#set page(
  paper: "a4",
  margin: (left: 25mm, right: 25mm, top: 30mm, bottom: 30mm),
  numbering: "1",
  number-align: center,
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "en"
)

#set heading(numbering: "1.")

#show math.equation: set text(weight: 400)

#set par(justify: true)

// Title page
#align(center)[
  #text(size: 20pt, weight: "bold")[
    Maximum Flow Algorithms: \
    Ford-Fulkerson vs Edmonds-Karp
  ]
  
  #v(2cm)
  
  #text(size: 16pt)[
    A Comparative Study of Classical Flow Network Algorithms
  ]
  
  #v(1cm)
  
  #text(size: 12pt)[
    Advanced Algorithms and Data Structures Project
  ]
  
  #v(2cm)
  
  #text(size: 12pt)[
    #datetime.today().display("[month repr:long] [day], [year]")
  ]
]

#pagebreak()

// Table of contents
#outline(indent: auto)

#pagebreak()

= Introduction and Objectives

The Maximum Flow problem stands as one of the fundamental challenges in graph theory and network optimization. Given a flow network—a directed graph where each edge has a capacity constraint—the problem seeks to determine the maximum amount of flow that can be sent from a designated source vertex to a sink vertex while respecting the capacity limitations and flow conservation constraints.

This project presents a comprehensive analysis and implementation of two seminal algorithms that solve the Maximum Flow problem: the Ford-Fulkerson algorithm (1956) and its refined variant, the Edmonds-Karp algorithm (1972). Through theoretical analysis, empirical evaluation, and visual demonstration, we explore the fundamental differences, performance characteristics, and practical implications of these algorithmic approaches.

== Project Objectives

The primary goals of this investigation are:

1. *Implementation and Verification*: Develop robust, well-documented implementations of both algorithms with comprehensive test suites ensuring correctness across diverse graph topologies.

2. *Theoretical Analysis*: Provide detailed mathematical analysis of algorithmic complexity, convergence properties, and theoretical performance bounds.

3. *Empirical Evaluation*: Conduct extensive benchmarking across various graph types, sizes, and structural characteristics to understand practical performance differences.

4. *Visualization and Education*: Create intuitive visualizations that demonstrate algorithm execution, flow assignments, and comparative behavior to enhance understanding of the underlying mechanisms.

5. *Practical Applications*: Explore real-world applications where maximum flow algorithms provide optimal solutions to network optimization problems.

== Significance and Applications

Maximum flow algorithms find applications across numerous domains:

- *Network Infrastructure*: Bandwidth allocation, routing optimization, and network reliability analysis
- *Transportation Systems*: Traffic flow optimization, supply chain management, and logistics planning  
- *Computer Vision*: Image segmentation, stereo matching, and object recognition
- *Operations Research*: Project scheduling, resource allocation, and capacity planning
- *Bioinformatics*: Sequence alignment, protein folding analysis, and phylogenetic reconstruction

The ubiquity of these applications underscores the fundamental importance of understanding and optimizing maximum flow algorithms.

= Problem Definition and Mathematical Foundations

== Flow Network Formal Definition

A *flow network* is formally defined as a directed graph $G = (V, E)$ equipped with the following components:

#align(center)[
  $G = (V, E, c, s, t)$
]

where:
- $V$ is the set of vertices (nodes)
- $E ⊆ V × V$ is the set of directed edges
- $c: E → ℝ^+$ is the *capacity function* assigning positive real capacities to edges
- $s ∈ V$ is the designated *source* vertex
- $t ∈ V$ is the designated *sink* vertex, with $s ≠ t$

== Flow Function Properties

A *flow* in the network is a function $f: E → ℝ^+$ that must satisfy two fundamental constraints:

=== Capacity Constraint
For every edge $(u,v) ∈ E$:
$ f(u,v) ≤ c(u,v) $

This ensures that the flow through any edge never exceeds its capacity limit.

=== Flow Conservation Constraint  
For every vertex $v ∈ V ∖ {s,t}$ (all vertices except source and sink):
$ sum_{u:(u,v) ∈ E} f(u,v) = sum_{w:(v,w) ∈ E} f(v,w) $

This constraint ensures that flow is neither created nor destroyed at intermediate vertices—the total flow entering a vertex must equal the total flow leaving it.

== Maximum Flow Problem Statement

The *Maximum Flow Problem* seeks to find a flow $f$ that maximizes the *value of the flow*, defined as:

$ |f| = sum_{v:(s,v) ∈ E} f(s,v) - sum_{u:(u,s) ∈ E} f(u,s) $

Equivalently, by flow conservation, this equals:
$ |f| = sum_{u:(u,t) ∈ E} f(u,t) - sum_{v:(t,v) ∈ E} f(t,v) $

== Residual Network and Augmenting Paths

Central to understanding maximum flow algorithms is the concept of the *residual network*.

=== Residual Capacity
For any edge $(u,v)$, the *residual capacity* is defined as:
$ c_f(u,v) = cases(
  c(u,v) - f(u,v) & "if " (u,v) ∈ E,
  f(v,u) & "if " (v,u) ∈ E,
  0 & "otherwise"
) $

The first case represents remaining forward capacity, while the second represents the possibility of reducing existing flow (creating backward capacity).

=== Residual Network
The *residual network* $G_f = (V, E_f)$ with respect to flow $f$ contains only edges with positive residual capacity:
$ E_f = {(u,v) ∈ V × V : c_f(u,v) > 0} $

=== Augmenting Path
An *augmenting path* is a simple path from $s$ to $t$ in the residual network $G_f$. The *residual capacity* of such a path $P$ is:
$ c_f(P) = min_{(u,v) ∈ P} c_f(u,v) $

== Max-Flow Min-Cut Theorem

The theoretical foundation for maximum flow algorithms rests on the celebrated Max-Flow Min-Cut Theorem:

*Theorem (Max-Flow Min-Cut):* In any flow network, the value of the maximum flow equals the capacity of the minimum cut.

Formally, if $f^*$ is a maximum flow and $(S,T)$ is a minimum cut where $S ∪ T = V$, $S ∩ T = ∅$, $s ∈ S$, and $t ∈ T$, then:
$ |f^*| = c(S,T) = sum_{u ∈ S, v ∈ T, (u,v) ∈ E} c(u,v) $

This theorem provides both an optimality condition and a certificate for maximum flow solutions.

= Algorithm Analysis and Implementation

== Ford-Fulkerson Algorithm

The Ford-Fulkerson method, introduced by L.R. Ford Jr. and D.R. Fulkerson in 1956, establishes the foundational approach for solving maximum flow problems through the iterative augmentation of flow along source-to-sink paths.

=== Algorithmic Framework

The Ford-Fulkerson method follows a generic framework that can be instantiated with different path-finding strategies:

#algorithm({
  import algorithmic: *
  Function("Ford-Fulkerson", args: ("G", "s", "t"), {
    Assign[$f(u,v)$][$0$ for all $(u,v) ∈ E$]
    While(cond: "there exists an augmenting path $P$ in $G_f$", {
      Assign[$c_f(P)$][minimum residual capacity along $P$]
      ForAll(cond: "edges $(u,v)$ in $P$", {
        If(cond: "$(u,v) ∈ E$", {
          Assign[$f(u,v)$][$f(u,v) + c_f(P)$]
        })
        Else({
          Assign[$f(v,u)$][$f(v,u) - c_f(P)$]
        })
      })
    })
    Return[$f$]
  })
})

=== Path Selection Strategy

The generic Ford-Fulkerson framework does not specify how augmenting paths should be discovered. In our implementation, we employ *depth-first search* (DFS) to locate augmenting paths, which provides a straightforward recursive approach:

#algorithm({
  import algorithmic: *
  Function("DFS-Find-Path", args: ("u", "t", "visited", "path"), {
    If(cond: "$u = t$", {
      Return[path $∪ \{t\}$]
    })
    Assign("visited")["visited" $∪ \{u\}$]
    ForAll(cond: "$v$ such that $c_f(u,v) > 0$ and $v ∉$ visited", {
      Assign["result"]["DFS-Find-Path($v$, $t$, visited, path $∪ \{u\}$)"]
      If(cond: "result $≠$ null", {
        Return["result"]
      })
    })
    Return["null"]
  })
})

=== Complexity Analysis

The time complexity of Ford-Fulkerson depends critically on the path selection method and the nature of the input:

*Time Complexity:* $O(E · |f^*|)$ where $|f^*|$ is the value of the maximum flow.

This bound arises because:
- Each augmenting path increases the flow by at least 1 unit (assuming integer capacities)
- Finding each path requires $O(E)$ time using DFS
- At most $|f^*|$ iterations are needed

*Space Complexity:* $O(V + E)$ for storing the residual graph and maintaining the DFS recursion stack.

=== Limitations

The Ford-Fulkerson algorithm exhibits several limitations:

1. *Non-polynomial Runtime:* With irrational capacities, the algorithm may not terminate
2. *Inefficient Path Selection:* DFS may select long paths when shorter alternatives exist
3. *Poor Performance on Dense Graphs:* The $O(E · |f^*|)$ bound becomes problematic when $|f^*|$ is large

== Edmonds-Karp Algorithm

Jack Edmonds and Richard Karp addressed the efficiency limitations of Ford-Fulkerson in 1972 by proposing a specific path selection strategy that guarantees polynomial-time performance.

=== Key Innovation: Breadth-First Search

The Edmonds-Karp algorithm modifies Ford-Fulkerson by using *breadth-first search* (BFS) to find the *shortest* augmenting paths (in terms of number of edges):

#algorithm({
  import algorithmic: *
  Function("BFS-Find-Path", args: ("s", "t"), {
    Assign["queue"]["Queue($s$)"]
    Assign["parent"]["empty map"]
    Assign["visited"][$\{s\}$]
    
    While(cond: "queue is not empty", {
      Assign["u"]["queue.dequeue()"]
      ForAll(cond: "$v$ such that $c_f(u,v) > 0$ and $v ∉$ visited", {
        State["queue.enqueue($v$)"]
        Assign["visited"]["visited" $∪ \{v\}$]
        Assign["parent[$v$]"][$u$]
        If(cond: "$v = t$", {
          Return["reconstruct path using parent map"]
        })
      })
    })
    Return["null"]
  })
})

=== Theoretical Advantages

The BFS-based path selection provides several theoretical guarantees:

*Theorem:* The Edmonds-Karp algorithm runs in $O(VE^2)$ time.

*Proof Sketch:*
1. Each BFS operation requires $O(E)$ time
2. The distance from $s$ to $t$ (in terms of edges) can increase at most $V$ times
3. Between distance increases, at most $E$ edges can become saturated
4. Therefore, at most $O(VE)$ iterations are needed
5. Total complexity: $O(VE) · O(E) = O(VE^2)$

*Key Insight:* By always choosing shortest paths, the algorithm ensures that the distance to the sink in the residual network is non-decreasing, leading to the polynomial bound.

=== Implementation Details

Our Edmonds-Karp implementation incorporates several optimizations:

1. *Efficient Residual Graph Representation:* Using adjacency lists with both forward and backward edges
2. *Path Reconstruction:* Parent pointer technique for efficient path recovery
3. *Early Termination:* BFS terminates immediately upon reaching the sink

=== Comparative Analysis

| Aspect | Ford-Fulkerson | Edmonds-Karp |
|--------|---------------|--------------|
| *Path Strategy* | Any augmenting path (DFS) | Shortest augmenting path (BFS) |
| *Time Complexity* | $O(E \cdot |f^*|)$ | $O(VE^2)$ |
| *Termination* | Guaranteed only for rational capacities | Always guaranteed |
| *Practical Performance* | Variable, can be poor | Consistently polynomial |

== Algorithm Correctness

Both algorithms rely on the fundamental correctness of the augmenting path method:

*Theorem (Augmenting Path Correctness):* A flow $f$ is maximum if and only if there are no augmenting paths in the residual network $G_f$.

*Proof:*
- *Necessity:* If an augmenting path exists, the flow can be increased, contradicting maximality
- *Sufficiency:* If no augmenting path exists, then by the Max-Flow Min-Cut theorem, the current flow is maximum

= Results and Analysis

...

= Conclusions and Future Work

...
