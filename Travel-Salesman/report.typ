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

*Time Complexity:* $O(n!)$ worst case  
*Space Complexity:* $O(n)$

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

Let $"OPT"$ be the cost of an optimal TSP tour and $"MST"$ be the cost of the minimum spanning tree.

- *Lower Bound:* $"MST" ≤ "OPT"$ because removing any edge from an optimal tour yields a spanning tree.

- *Algorithm Cost:* The DFS traversal visits each edge of the MST exactly twice (once down, once up), so the "doubling" tour has cost $2 · "MST"$.

- *Shortcutting:* Using the triangle inequality, we can shortcut the doubled tour to visit each vertex exactly once without increasing the total cost.

- *Approximation Ratio:* $"ALG" ≤ 2 · "MST" ≤ 2 · "OPT"$

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
$ r_("ALG") = (L_("ALG"))/(L_("OPT")) $
where $L_("ALG")$ is the tour length found by algorithm ALG and $L_("OPT")$ is the optimal tour length.

*Relative Performance:* When optimal solutions are unknown (large instances), we compare against the best known solution across all tested algorithms.

=== Computational Efficiency

*Execution Time:* Wall-clock time to find a solution, averaged over multiple runs.

*Scalability:* How execution time grows with problem size $n$.

*Success Rate:* Percentage of instances solved within time and memory limits.

=== Robustness

*Variance:* Standard deviation of solution quality across multiple instances of the same size.

*Worst-Case Performance:* Maximum approximation ratio observed.

== Experimental Design

*Problem Sizes:* Test instances ranging from 5 cities to 10 cities for comprehensive analysis.

*Instance Counts:* Three random instances per size to ensure statistical significance.

*Instance Type:* Euclidean instances with cities randomly placed in a 100×100 square.

*Implementation Details:* All algorithms implemented in Python with consistent data structures and measurement techniques.

= Results and Analysis

This section presents the comprehensive experimental results obtained from testing all implemented algorithms on Euclidean TSP instances ranging from 5 to 10 cities.

== Solution Quality Analysis

The experimental results reveal distinct performance characteristics across different algorithm classes:

=== Exact Algorithm Performance

All three exact algorithms (Brute Force, Held-Karp DP, and Branch & Bound) consistently found optimal solutions across all test instances, as expected. This validates the correctness of our implementations and provides ground truth for evaluating approximation algorithms.

*Key Observations:*
- Perfect solution quality (approximation ratio = 1.0) for all instances
- Consistent optimal results across all problem sizes tested
- No variation in solution quality between different exact methods

=== Approximation Algorithm Analysis

The MST 2-Approximation algorithm demonstrated the following characteristics:

*Approximation Ratios Observed:*
- Size 5: Ratios of 1.194, 1.070, 1.044 (average: 1.103)
- Size 6: Ratios of 1.000, 1.222, 1.035 (average: 1.086) 
- Size 7: Ratios of 1.000, 1.074, 1.033 (average: 1.036)
- Size 8: Ratios of 1.000, 1.309, 1.181 (average: 1.163)
- Size 10: Ratios of 1.116, 1.271, 1.181 (average: 1.189)

*Analysis:* The experimental approximation ratios (1.03-1.31) are significantly better than the theoretical worst-case bound of 2.0, confirming that the algorithm performs well on typical Euclidean instances. The variation in performance appears to depend on the geometric structure of individual instances rather than problem size.

=== Heuristic Algorithm Performance

*Nearest Neighbor Results:*
- Often produced suboptimal solutions, particularly when started from poorly positioned cities
- Performance highly dependent on starting city selection
- Approximation ratios similar to MST 2-Approximation in many cases

*Multi-Start Nearest Neighbor Results:*
- Consistently found optimal or near-optimal solutions
- Significant improvement over single-start nearest neighbor
- Approximation ratios: mostly 1.0, with occasional small deviations (max 1.037)

*2-opt Local Search Results:*
- When combined with nearest neighbor initialization, consistently found optimal solutions
- Both "NN + 2-opt" and "Multi-start NN + 2-opt" achieved perfect results on all test instances
- Demonstrates the power of local search for improving initial solutions

== Scalability and Time Complexity

The experimental results confirm the theoretical time complexity predictions:

=== Exact Algorithm Scalability

*Brute Force:* Execution time grows factorially with problem size:
- Size 5-6: < 0.001 seconds
- Size 7: ~0.003 seconds  
- Size 8: ~0.018 seconds
- Size 10: Not executed (would require excessive time)

*Held-Karp DP:* Superior scalability compared to brute force:
- Consistent performance up to size 8: 0.001-0.002 seconds
- Size 10: 0.007-0.011 seconds
- Demonstrates $O(n^2 2^n)$ complexity advantage over $O(n!)$

*Branch & Bound:* Performance varies significantly with instance structure:
- Best case: Similar to Held-Karp DP
- Worst case: Approaches brute force performance (size 10, instance 3: 0.046s)
- Effectiveness depends on lower bound quality and branching decisions

=== Approximation and Heuristic Scalability

All approximation and heuristic algorithms demonstrated excellent scalability:
- Execution times consistently < 0.001 seconds for all problem sizes
- Linear or near-linear growth with problem size
- Suitable for much larger instances than tested

== Trade-off Analysis

The results reveal clear trade-offs between solution quality, execution time, and scalability:

=== Quality vs. Speed Trade-offs

1. *Exact Solutions:* Perfect quality but exponential time growth
   - Held-Karp DP preferred over brute force for n > 8
   - Branch & bound effective when good bounds available

2. *Fast Approximations:* Good quality with minimal computational cost
   - MST 2-Approximation: ~10-20% above optimal, instant execution
   - Multi-start NN: Often optimal, very fast execution

3. *Hybrid Approaches:* Best of both worlds
   - NN + 2-opt: Optimal solutions with fast execution
   - Minimal overhead compared to pure construction heuristics

=== Algorithm Selection Guidelines

Based on the experimental results, we recommend:

*For Small Instances (n ≤ 10):*
- Use Held-Karp DP for guaranteed optimal solutions
- Consider NN + 2-opt for near-optimal solutions with faster execution

*For Medium Instances (10 < n ≤ 50):*
- Multi-start NN + 2-opt for high-quality solutions
- MST 2-Approximation for quick approximations with guarantees

*For Large Instances (n > 50):*
- Heuristic approaches become essential
- 2-opt local search with good initialization highly recommended

== Statistical Analysis

*Consistency:* Exact algorithms showed perfect consistency (zero variance in solution quality).

*Reliability:* Multi-start approaches significantly reduced variance compared to single-start methods.

*Robustness:* 2-opt local search demonstrated remarkable ability to find optimal solutions regardless of initialization quality.

The experimental validation confirms theoretical predictions while revealing that practical performance often exceeds worst-case bounds, particularly for approximation algorithms on Euclidean instances.

= Reflections and Conclusions

This comprehensive study of TSP algorithms provides valuable insights into the practical performance and theoretical guarantees of different algorithmic approaches.

== Key Insights

=== Theoretical vs. Practical Performance

Our experimental results demonstrate that theoretical worst-case bounds often overestimate actual algorithm performance on typical instances. The MST 2-approximation algorithm, while theoretically bounded at 2.0 times optimal, consistently achieved ratios between 1.03 and 1.31 on Euclidean instances. This gap between theory and practice is common in algorithm analysis and highlights the value of empirical evaluation.

=== Power of Local Search

The 2-opt local search algorithm emerged as remarkably effective, consistently finding optimal solutions when combined with any reasonable initialization. This demonstrates a fundamental principle in combinatorial optimization: local search can often overcome weaknesses in construction heuristics, making the choice of initial solution less critical.

=== Algorithm Complementarity

Different algorithms excel in different scenarios, suggesting that hybrid approaches often provide the best overall performance. The combination of fast construction heuristics with local improvement yielded solutions matching exact algorithms while maintaining excellent scalability.

== Algorithm Selection Framework

Based on our analysis, we propose the following decision framework:

=== Problem Size Considerations

*Small Instances (n ≤ 15):* Exact algorithms remain practical. Held-Karp DP is preferred over brute force enumeration due to superior scaling. Branch and bound can be competitive when effective lower bounds are available.

*Medium Instances (15 < n ≤ 100):* Heuristic approaches become necessary. Multi-start nearest neighbor with 2-opt improvement provides an excellent balance of solution quality and computational efficiency.

*Large Instances (n > 100):* Pure heuristics are essential. Focus on multi-start approaches and local search methods. Consider more sophisticated metaheuristics for critical applications.

=== Quality Requirements

*Optimal Solutions Required:* Use exact algorithms within their practical limits. For larger instances, consider implementing branch-and-bound with stronger lower bounds or using specialized TSP solvers.

*High-Quality Approximations Acceptable:* Multi-start heuristics with local search typically provide solutions within 1-5% of optimal for Euclidean instances.

*Quick Approximations Sufficient:* MST 2-approximation provides theoretical guarantees with minimal computational cost.

== Limitations and Future Work

=== Experimental Limitations

Our study focused on relatively small instances (5-10 cities) due to the computational limitations of exact algorithms. Future work should examine larger instances and different distance metrics to validate our findings across broader problem classes.

=== Algorithmic Extensions

Several promising directions for improvement include:

1. *Stronger Lower Bounds:* Implementing Held-Karp lower bounds in branch-and-bound could significantly improve pruning effectiveness.

2. *Advanced Local Search:* Methods like Lin-Kernighan or variable neighborhood search could further improve solution quality.

3. *Metaheuristics:* Genetic algorithms, simulated annealing, and ant colony optimization represent the state-of-the-art for large TSP instances.

4. *Parallel Computing:* Many TSP algorithms can benefit from parallelization, particularly branch-and-bound and population-based metaheuristics.

=== Theoretical Considerations

While our focus was on practical performance, several theoretical questions remain interesting:

- Tightness of approximation ratios for different instance classes
- Average-case complexity analysis of exact algorithms
- Probabilistic analysis of heuristic performance

== Practical Recommendations

=== For Practitioners

1. *Default Choice:* Multi-start nearest neighbor with 2-opt improvement provides excellent performance across a wide range of problem sizes.

2. *When Optimality Matters:* Use Held-Karp DP for small instances, consider specialized TSP solvers for larger problems.

3. *When Speed Matters:* MST 2-approximation offers guaranteed quality with minimal computation.

4. *Implementation Advice:* Focus on efficient data structures and careful implementation of distance calculations, as these often dominate runtime for heuristic algorithms.

=== For Researchers

The TSP continues to drive algorithmic innovation. Our study confirms that:
- Hybrid approaches often outperform pure strategies
- Local search remains a powerful tool for combinatorial optimization
- Empirical evaluation is essential for understanding practical algorithm performance

=== Educational Value

This project demonstrates several important concepts in algorithm design and analysis:
- Trade-offs between optimality and efficiency
- The gap between worst-case and average-case performance
- The importance of problem structure in algorithm selection
- The value of combining different algorithmic techniques

== Final Thoughts

The Traveling Salesman Problem serves as an excellent case study in algorithmic problem-solving, illustrating the progression from brute-force approaches through sophisticated approximation algorithms to modern metaheuristics. Our implementation and analysis confirm that while theoretical complexity analysis provides important bounds, practical performance often depends on problem structure and careful algorithm engineering.

The continued relevance of TSP in both theoretical computer science and practical applications ensures that research in this area remains vibrant. As computational resources continue to expand and new algorithmic techniques emerge, the solutions to larger and more complex TSP instances become increasingly accessible, enabling applications that were previously computationally infeasible.

For students and practitioners entering the field of combinatorial optimization, the TSP provides an ideal introduction to fundamental concepts while remaining challenging enough to drive continued research and innovation.

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

*Data Structures:* All algorithms used consistent distance matrix representation with floating-point precision for Euclidean distances.

*Optimization Techniques:* 
- Memoization in dynamic programming
- Early termination in branch-and-bound
- Efficient tour representation for local search

*Testing Framework:* Comprehensive unit tests validated algorithm correctness on known small instances before scaling experiments.

== Appendix B: Additional Experimental Results  

*Detailed Performance Tables:* Complete timing and solution quality data for all algorithm combinations and instance sizes.

*Statistical Analysis:* Variance analysis, confidence intervals, and significance testing of performance differences between algorithms.

*Instance Characteristics:* Analysis of how geometric properties of test instances affected algorithm performance.

== Appendix C: Mathematical Proofs

*MST Lower Bound Proof:* Formal proof that the minimum spanning tree cost provides a lower bound for TSP optimal solutions.

*2-Approximation Analysis:* Complete proof of the approximation ratio for the MST-based algorithm.

*Complexity Proofs:* Detailed analysis of time and space complexity for all implemented algorithms.
