#import "@preview/algorithmic:0.1.0"
#import "@preview/cetz:0.2.2"

#set document(
  title: "Traveling Salesman Problem: Algorithm Comparison and Analysis",
  author: "TSP Algorithm Study",
  date: datetime.today(),
)

#set text(font: "New Computer Modern", size: 11pt)
#set page(margin: 1in)
#set par(justify: true)

#show heading.where(level: 1): it => [
  #set text(size: 16pt, weight: "bold")
  #block(below: 1em, above: 1.5em)[#it]
]

#show heading.where(level: 2): it => [
  #set text(size: 14pt, weight: "bold") 
  #block(below: 0.8em, above: 1.2em)[#it]
]

#show heading.where(level: 3): it => [
  #set text(size: 12pt, weight: "bold")
  #block(below: 0.6em, above: 1em)[#it]
]

// Title page
#align(center)[
  #text(size: 20pt, weight: "bold")[
    Traveling Salesman Problem:\
    Algorithm Comparison and Analysis
  ]
  
  #v(2em)
  
  #text(size: 14pt)[
    A Comprehensive Study of Exact, Approximation,\
    and Heuristic Approaches
  ]
  
  #v(3em)
  
  #text(size: 12pt)[
    #datetime.today().display("[month repr:long] [day], [year]")
  ]
]

#pagebreak()

// Table of Contents
#outline(
  title: "Table of Contents",
  indent: auto,
  depth: 3
)

#pagebreak()

= Introduction

The Traveling Salesman Problem (TSP) is one of the most well-studied problems in combinatorial optimization and computational complexity theory. Given a collection of cities and the distances between each pair of cities, the objective is to find the shortest possible route that visits each city exactly once and returns to the starting city.

== Problem Definition

Formally, the TSP can be defined as follows:

*Input:* A complete graph $G = (V, E)$ where $V$ is a set of $n$ vertices (cities) and $E$ is the set of edges connecting every pair of vertices. Each edge $(i,j) ∈ E$ has an associated weight $d_(i,j) ≥ 0$ representing the distance between cities $i$ and $j$.

*Output:* A Hamiltonian cycle (a cycle that visits each vertex exactly once) of minimum total weight.

Mathematically, we seek to find a permutation $π$ of ${1, 2, ..., n}$ that minimizes:

$ sum_(i=1)^n d_(π(i), π(i+1)) $

where $π(n+1) = π(1)$ to ensure the cycle returns to the starting city.

== Problem Significance and Applications

The TSP is significant for several reasons:

1. *Theoretical Importance:* It is NP-complete, making it unlikely that a polynomial-time exact algorithm exists for general instances.

2. *Practical Applications:* The TSP models numerous real-world optimization problems including:
   - Vehicle routing and logistics
   - Circuit board drilling optimization  
   - DNA sequencing
   - Telescope scheduling
   - Manufacturing processes

3. *Algorithmic Research:* It serves as a benchmark for testing optimization techniques and has driven advances in approximation algorithms, heuristics, and metaheuristics.

== Metric TSP and Triangle Inequality

An important special case is the *metric TSP*, where distances satisfy the triangle inequality:

$ d_(i,k) ≤ d_(i,j) + d_(j,k) quad ∀ i,j,k ∈ V $

This property is crucial because:
- It models realistic distance functions (Euclidean, Manhattan, etc.)
- It enables approximation algorithms with performance guarantees
- Many practical TSP instances satisfy this property

== Computational Complexity

The general TSP is NP-complete, which means:
- No known polynomial-time algorithm exists for finding optimal solutions
- The best exact algorithms have exponential time complexity
- Even the metric TSP remains NP-complete

However, the metric TSP admits polynomial-time approximation algorithms, whereas the general TSP (without triangle inequality) cannot be approximated within any constant factor unless P = NP.

= Algorithm Descriptions

This section provides detailed descriptions of the various algorithmic approaches implemented in this study, ranging from exact algorithms that guarantee optimal solutions to fast heuristics that provide good approximate solutions.

== Exact Algorithms

Exact algorithms guarantee finding the optimal solution but typically have exponential time complexity, limiting their applicability to small instances.

=== Brute Force Enumeration

The most straightforward approach enumerates all possible tours and selects the best one.

*Algorithm:* Fix one city (say city 0) as the starting point to break symmetry. Generate all $(n-1)!$ permutations of the remaining cities, calculate the tour length for each permutation, and return the minimum.

*Time Complexity:* $O(n!)$ worst case, but often much better in practice  
*Space Complexity:* $O(n)$

*Analysis:* The effectiveness depends heavily on the quality of the lower bound. Good bounds lead to significant pruning, while poor bounds result in exploring most of the search space. This algorithm often performs well on structured instances.

== Approximation Algorithms

Approximation algorithms provide solutions with theoretical quality guarantees. For metric TSP, we can achieve constant-factor approximations.

=== MST-Based 2-Approximation

This classical algorithm provides a solution guaranteed to be at most twice the optimal cost for metric TSP instances.

*Algorithm Steps:*

1. *Compute MST:* Find a minimum spanning tree $T$ of the complete graph using Kruskal's or Prim's algorithm.

2. *DFS Traversal:* Perform a depth-first search traversal of $T$ starting from an arbitrary vertex.

3. *Extract Tour:* The DFS preorder gives a Hamiltonian cycle by visiting vertices in the order they are first encountered.

*Theoretical Analysis:*

Let $OPT$ be the cost of an optimal TSP tour and $MST$ be the cost of the minimum spanning tree.

- *Lower Bound:* $MST ≤ OPT$ because removing any edge from an optimal tour yields a spanning tree.

- *Algorithm Cost:* The DFS traversal visits each edge of the MST exactly twice (once down, once up), so the "doubling" tour has cost $2 · MST$.

- *Shortcutting:* Using the triangle inequality, we can shortcut the doubled tour to visit each vertex exactly once without increasing the total cost.

- *Approximation Ratio:* $ALG ≤ 2 · MST ≤ 2 · OPT$

*Time Complexity:* $O(n^2)$ for dense graphs  
*Space Complexity:* $O(n)$

*Practical Performance:* While the theoretical bound is 2, empirical studies show that this algorithm typically achieves approximation ratios between 1.2 and 1.5 on random instances.

== Heuristic Algorithms

Heuristics provide fast approximate solutions without theoretical guarantees but often perform well in practice.

=== Nearest Neighbor Algorithm

This greedy heuristic constructs a tour by always moving to the nearest unvisited city.

*Algorithm:*
1. Start at an arbitrary city
2. Repeatedly move to the nearest unvisited city
3. Return to the starting city when all cities have been visited

*Time Complexity:* $O(n^2)$  
*Space Complexity:* $O(n)$

*Analysis:* Simple and fast, but can produce poor solutions when the greedy choice leads to expensive connections later. Performance varies significantly with the starting city.

=== Multi-Start Nearest Neighbor

*Improvement:* Run the nearest neighbor algorithm from multiple starting cities and return the best solution found.

*Benefits:* Often provides better solutions than single-start nearest neighbor with minimal additional computational cost.

=== 2-opt Local Search

2-opt is a local search algorithm that iteratively improves a given tour by making local modifications.

*Core Operation:* Given a tour, consider removing two edges and reconnecting the tour in a different way. If this "2-opt move" improves the tour length, accept the change.

*Algorithm:*
1. Start with an initial tour (e.g., from nearest neighbor)
2. For each pair of edges $(i, i+1)$ and $(j, j+1)$ in the tour:
   - Consider removing these edges and adding edges $(i, j)$ and $(i+1, j+1)$
   - If this reduces the tour length, make the change and restart
3. Stop when no improving 2-opt move exists (local optimum)

*Mathematical Formulation:* For edges $(i, i+1)$ and $(j, j+1)$, the change in tour length is:
$ Δ = d_(i,j) + d_(i+1,j+1) - d_(i,i+1) - d_(j,j+1) $

Accept the move if $Δ < 0$.

*Time Complexity:* $O(n^2)$ per iteration, typically converges in $O(n)$ iterations  
*Space Complexity:* $O(n)$

*Analysis:* 2-opt is highly effective at removing crossing edges in Euclidean instances and generally provides significant improvements over greedy construction heuristics.

=== Combined Approaches

*Nearest Neighbor + 2-opt:* Use nearest neighbor to construct an initial tour, then improve it with 2-opt local search.

*Multi-Start with 2-opt:* Combine multiple starting points with local search for robust performance.

*Random Restart 2-opt:* Start 2-opt from multiple random initial tours to escape poor local optima.

= Algorithm Comparison Framework

This section describes the experimental methodology used to evaluate and compare the implemented algorithms.

== Instance Generation

We generate two types of TSP instances to evaluate algorithm performance:

=== Euclidean Instances

*Generation:* Place $n$ cities at random coordinates $(x_i, y_i)$ uniformly distributed in a square $[0, 100] × [0, 100]$.

*Distance Function:* Euclidean distance $d_(i,j) = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)$

*Properties:*
- Automatically satisfies triangle inequality
- Geometric structure enables visualization
- Representative of many practical applications (logistics, PCB drilling)

=== Random Metric Instances

*Generation:* Create a random symmetric distance matrix that satisfies the triangle inequality.

*Method:*
1. Generate initial random symmetric matrix with positive entries
2. Apply Floyd-Warshall algorithm to enforce triangle inequality:
   $d_(i,j) ← min(d_(i,j), d_(i,k) + d_(k,j))$ for all $i, j, k$

*Properties:*
- Tests algorithmic behavior on non-geometric instances
- Represents scenarios where geographical intuition may not apply

== Evaluation Metrics

=== Solution Quality

*Approximation Ratio:* For each instance, we define the approximation ratio as:
$ r_(ALG) = (L_(ALG))/(L_(OPT)) $
where $L_(ALG)$ is the tour length found by algorithm ALG and $L_(OPT)$ is the optimal tour length.

*Relative Performance:* When optimal solutions are unknown (large instances), we compare against the best known solution across all tested algorithms.

=== Computational Efficiency

*Execution Time:* Wall-clock time to find a solution, averaged over multiple runs.

*Scalability:* How execution time grows with problem size $n$.

*Success Rate:* Percentage of instances solved within time and memory limits.

=== Robustness

*Variance:* Standard deviation of solution quality across multiple instances of the same size.

*Worst-Case Performance:* Maximum approximation ratio observed.

== Experimental Design

*Problem Sizes:* Test instances ranging from 4 cities (for verification) to 50+ cities (for scalability analysis).

*Instance Counts:* Multiple random instances per size to ensure statistical significance.

*Time Limits:* Reasonable cutoffs for each algorithm class:
- Exact algorithms: 60 seconds
- Approximation/Heuristics: 10 seconds

*Implementation Details:* All algorithms implemented in Python with consistent data structures and measurement techniques.

= Results and Analysis

*[Placeholder - Results to be generated after running experiments]*

This section will contain:

- Performance comparison tables showing solution quality and execution times
- Scalability analysis with time complexity validation  
- Approximation ratio distributions and statistical analysis
- Trade-off analysis between solution quality and computation time
- Algorithm recommendations for different problem sizes and requirements

Specific analyses will include:

== Solution Quality Analysis
...

== Scalability and Time Complexity
...  

== Trade-off Analysis
...

== Algorithm Selection Guidelines
...

= Reflections and Conclusions

*[Placeholder - Conclusions to be written after experimental analysis]*

This section will provide:

- Summary of key findings from the experimental study
- Insights into when different algorithm classes are most appropriate  
- Lessons learned about TSP algorithm design and implementation
- Directions for future research and improvements
- Practical recommendations for TSP practitioners

Specific topics will include:

== Key Insights
...

== Algorithm Selection Framework  
...

== Limitations and Future Work
...

== Practical Recommendations
...

= Bibliography

1. Christofides, N. (1976). Worst-case analysis of a new heuristic for the travelling salesman problem. *Operations Research*, 93, 113.

2. Held, M., & Karp, R. M. (1962). A dynamic programming approach to sequencing problems. *Journal of the Society for Industrial and Applied Mathematics*, 10(1), 196-210.

3. Lin, S., & Kernighan, B. W. (1973). An effective heuristic algorithm for the traveling-salesman problem. *Operations Research*, 21(2), 498-516.

4. Applegate, D. L., Bixby, R. E., Chvátal, V., & Cook, W. J. (2006). *The traveling salesman problem: a computational study*. Princeton University Press.

5. Gutin, G., & Punnen, A. P. (Eds.). (2006). *The traveling salesman problem and its variations*. Springer Science & Business Media.

6. Lawler, E. L., Lenstra, J. K., Rinnooy Kan, A. H. G., & Shmoys, D. B. (Eds.). (1985). *The traveling salesman problem: a guided tour of combinatorial optimization*. John Wiley & Sons.

7. Johnson, D. S., & McGeoch, L. A. (1997). The traveling salesman problem: A case study in local optimization. *Local search in combinatorial optimization*, 1, 215-310.

8. Arora, S. (1998). Polynomial time approximation schemes for Euclidean traveling salesman and other geometric problems. *Journal of the ACM*, 45(5), 753-782.

= Appendices

== Appendix A: Implementation Details

*[Technical implementation notes, code structure, and algorithmic optimizations]*

== Appendix B: Additional Experimental Results  

*[Supplementary tables, figures, and statistical analyses]*

== Appendix C: Mathematical Proofs

*[Detailed proofs of approximation ratios and complexity bounds]*  
*Space Complexity:* $O(1)$

*Analysis:* While conceptually simple, this approach becomes intractable for $n > 10$ due to the factorial growth. However, it serves as a baseline for verifying the correctness of other algorithms on small instances.

=== Held-Karp Dynamic Programming

The Held-Karp algorithm uses dynamic programming to solve TSP optimally with better complexity than brute force enumeration.

*Core Idea:* Use bitmasks to represent subsets of visited cities and dynamic programming to avoid recomputing solutions to overlapping subproblems.

*State Definition:* Let $C(S, i)$ be the minimum cost of visiting all cities in set $S ⊆ {1, 2, ..., n}$ exactly once, starting from city 0 and ending at city $i$, where $0 ∉ S$ and $i ∈ S$.

*Recurrence Relations:*

Base case: $C({i}, i) = d_(0,i)$ for $i ∈ {1, 2, ..., n-1}$

Recursive case: $C(S, i) = min_(j ∈ S, j ≠ i) {C(S ∖ {i}, j) + d_(j,i)}$

*Final Solution:* $min_(i ∈ {1,2,...,n-1}} {C({1,2,...,n-1}, i) + d_(i,0)}$

*Time Complexity:* $O(n^2 2^n)$  
*Space Complexity:* $O(n 2^n)$

*Analysis:* This represents a significant improvement over brute force for moderately sized instances. The algorithm is practical for instances with up to approximately 20 cities.

=== Branch and Bound

Branch and bound systematically explores the solution space while pruning branches that cannot lead to optimal solutions.

*Key Components:*

1. *Branching:* Systematically enumerate partial tours by adding cities one at a time.

2. *Bounding:* For each partial tour, compute a lower bound on the cost of any complete tour extending this partial tour.

3. *Pruning:* If the lower bound for a partial tour exceeds the cost of the best complete tour found so far, prune this branch.

*Lower Bound Calculation:* We use an MST-based lower bound that combines:
- Cost of the current partial tour
- Minimum spanning tree cost of unvisited cities  
- Minimum cost edges connecting the partial tour to unvisited cities
- Minimum cost edge from unvisited cities back to the starting city

*Time Complexity:* $O(n!)$
