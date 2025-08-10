"""
Comprehensive benchmarking suite for data structures.

This module provides performance comparisons between:
- Segment Tree vs Fenwick Tree for range sum operations
- Different Union-Find optimizations
- Basic vs advanced variants
- Memory usage analysis
- Scalability testing across different input sizes

Results are saved and can be visualized using matplotlib.
"""

import time
import random
import sys
import os
from typing import List, Dict, Tuple, Callable, Any
import tracemalloc
from dataclasses import dataclass

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-structures'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'variants'))

# Import all data structures
from segment_tree import SegmentTree, SumSegmentTree
from fenwick_tree import FenwickTree
from union_find import UnionFind, UnionFindWithRollback

# Import variants
try:
    from segment_tree import LazySegmentTree, RangeSumLazySegmentTree
    from fenwick_tree import RangeUpdateFenwickTree, FenwickTreeWithFrequencies
    from union_find import UnionFindOptimized, DynamicConnectivity
except ImportError as e:
    print(f"Warning: Could not import some variants: {e}")


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    operation: str
    size: int
    time_taken: float
    memory_used: int
    operations_count: int
    
    @property
    def ops_per_second(self) -> float:
        """Calculate operations per second."""
        return self.operations_count / self.time_taken if self.time_taken > 0 else 0
    
    @property
    def time_per_op(self) -> float:
        """Calculate average time per operation in microseconds."""
        return (self.time_taken * 1_000_000) / self.operations_count if self.operations_count > 0 else 0


class DataStructureBenchmark:
    """Main benchmarking class."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results: List[BenchmarkResult] = []
        self.sizes = [100, 500, 1000, 5000, 10000, 50000]
        self.num_operations = 1000
    
    def benchmark_function(self, name: str, operation: str, size: int, 
                          setup_func: Callable, test_func: Callable, num_ops: int = None) -> BenchmarkResult:
        """
        Benchmark a specific function.
        
        Args:
            name: Name of the data structure
            operation: Type of operation being tested
            size: Input size
            setup_func: Function to create the data structure
            test_func: Function to perform the operations
            num_ops: Number of operations to perform
            
        Returns:
            BenchmarkResult with timing and memory information
        """
        if num_ops is None:
            num_ops = self.num_operations
        
        # Setup
        tracemalloc.start()
        data_structure = setup_func(size)
        
        # Benchmark
        start_time = time.perf_counter()
        test_func(data_structure, size, num_ops)
        end_time = time.perf_counter()
        
        # Memory measurement
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result = BenchmarkResult(
            name=name,
            operation=operation,
            size=size,
            time_taken=end_time - start_time,
            memory_used=peak,
            operations_count=num_ops
        )
        
        self.results.append(result)
        return result
    
    def benchmark_segment_tree_vs_fenwick(self):
        """Compare Segment Tree vs Fenwick Tree for range sum operations."""
        print("Benchmarking Segment Tree vs Fenwick Tree...")
        
        def setup_segment_tree(size: int) -> SumSegmentTree:
            arr = [random.randint(1, 100) for _ in range(size)]
            return SumSegmentTree(arr)
        
        def setup_fenwick_tree(size: int) -> FenwickTree:
            arr = [random.randint(1, 100) for _ in range(size)]
            return FenwickTree(arr)
        
        def test_range_queries(ds, size: int, num_ops: int):
            """Test random range queries."""
            for _ in range(num_ops):
                left = random.randint(0, size - 1)
                right = random.randint(left, size - 1)
                if hasattr(ds, 'query'):
                    ds.query(left, right)
                else:
                    ds.range_sum(left, right)
        
        def test_point_updates(ds, size: int, num_ops: int):
            """Test random point updates."""
            for _ in range(num_ops):
                index = random.randint(0, size - 1)
                value = random.randint(1, 100)
                if hasattr(ds, 'update'):
                    ds.update(index, value)
                else:
                    # For Fenwick, we need to use delta updates
                    current = ds.range_sum(index, index)
                    delta = value - current
                    ds.update(index, delta)
        
        for size in self.sizes:
            # Range query benchmarks
            self.benchmark_function("SegmentTree", "range_query", size, 
                                  setup_segment_tree, test_range_queries)
            self.benchmark_function("FenwickTree", "range_query", size,
                                  setup_fenwick_tree, test_range_queries)
            
            # Point update benchmarks
            self.benchmark_function("SegmentTree", "point_update", size,
                                  setup_segment_tree, test_point_updates)
            self.benchmark_function("FenwickTree", "point_update", size,
                                  setup_fenwick_tree, test_point_updates)
    
    def benchmark_union_find_variants(self):
        """Compare different Union-Find implementations."""
        print("Benchmarking Union-Find variants...")
        
        def setup_basic_uf(size: int) -> UnionFind:
            return UnionFind(size)
        
        def setup_optimized_uf(size: int) -> UnionFindOptimized:
            return UnionFindOptimized(size)
        
        def setup_rollback_uf(size: int) -> UnionFindWithRollback:
            return UnionFindWithRollback(size)
        
        def test_union_operations(uf, size: int, num_ops: int):
            """Test random union operations."""
            for _ in range(num_ops):
                x = random.randint(0, size - 1)
                y = random.randint(0, size - 1)
                uf.union(x, y)
        
        def test_find_operations(uf, size: int, num_ops: int):
            """Test random find operations."""
            # First perform some unions to create structure
            for _ in range(size // 4):
                x = random.randint(0, size - 1)
                y = random.randint(0, size - 1)
                uf.union(x, y)
            
            # Now test find operations
            for _ in range(num_ops):
                x = random.randint(0, size - 1)
                uf.find(x)
        
        def test_connectivity_queries(uf, size: int, num_ops: int):
            """Test random connectivity queries."""
            # First perform some unions
            for _ in range(size // 4):
                x = random.randint(0, size - 1)
                y = random.randint(0, size - 1)
                uf.union(x, y)
            
            # Now test connectivity
            for _ in range(num_ops):
                x = random.randint(0, size - 1)
                y = random.randint(0, size - 1)
                uf.connected(x, y)
        
        for size in self.sizes:
            # Union operations
            self.benchmark_function("UnionFind_Basic", "union", size,
                                  setup_basic_uf, test_union_operations)
            self.benchmark_function("UnionFind_Optimized", "union", size,
                                  setup_optimized_uf, test_union_operations)
            self.benchmark_function("UnionFind_Rollback", "union", size,
                                  setup_rollback_uf, test_union_operations)
            
            # Find operations
            self.benchmark_function("UnionFind_Basic", "find", size,
                                  setup_basic_uf, test_find_operations)
            self.benchmark_function("UnionFind_Optimized", "find", size,
                                  setup_optimized_uf, test_find_operations)
            
            # Connectivity queries
            self.benchmark_function("UnionFind_Basic", "connected", size,
                                  setup_basic_uf, test_connectivity_queries)
            self.benchmark_function("UnionFind_Optimized", "connected", size,
                                  setup_optimized_uf, test_connectivity_queries)
    
    def benchmark_lazy_propagation(self):
        """Compare regular vs lazy segment trees for range updates."""
        print("Benchmarking Lazy Propagation...")
        
        def setup_lazy_segment_tree(size: int) -> RangeSumLazySegmentTree:
            arr = [random.randint(1, 100) for _ in range(size)]
            return RangeSumLazySegmentTree(arr)
        
        def setup_range_update_fenwick(size: int) -> RangeUpdateFenwickTree:
            return RangeUpdateFenwickTree(size)
        
        def test_range_updates(ds, size: int, num_ops: int):
            """Test random range updates."""
            for _ in range(num_ops):
                left = random.randint(0, size - 1)
                right = random.randint(left, size - 1)
                delta = random.randint(1, 50)
                ds.range_update(left, right, delta)
        
        def test_mixed_operations(ds, size: int, num_ops: int):
            """Test mix of range updates and queries."""
            for _ in range(num_ops):
                if random.random() < 0.5:  # 50% updates, 50% queries
                    left = random.randint(0, size - 1)
                    right = random.randint(left, size - 1)
                    delta = random.randint(1, 50)
                    ds.range_update(left, right, delta)
                else:
                    left = random.randint(0, size - 1)
                    right = random.randint(left, size - 1)
                    if hasattr(ds, 'range_query'):
                        ds.range_query(left, right)
                    else:
                        ds.range_sum(left, right)
        
        for size in [1000, 5000, 10000]:  # Smaller sizes for expensive operations
            # Range update benchmarks
            self.benchmark_function("LazySegmentTree", "range_update", size,
                                  setup_lazy_segment_tree, test_range_updates, 200)
            self.benchmark_function("RangeUpdateFenwick", "range_update", size,
                                  setup_range_update_fenwick, test_range_updates, 200)
            
            # Mixed operations
            self.benchmark_function("LazySegmentTree", "mixed_ops", size,
                                  setup_lazy_segment_tree, test_mixed_operations, 200)
            self.benchmark_function("RangeUpdateFenwick", "mixed_ops", size,
                                  setup_range_update_fenwick, test_mixed_operations, 200)
    
    def benchmark_memory_usage(self):
        """Analyze memory usage of different data structures."""
        print("Benchmarking memory usage...")
        
        sizes = [1000, 5000, 10000, 50000]
        
        for size in sizes:
            arr = [random.randint(1, 100) for _ in range(size)]
            
            # Segment Tree memory
            tracemalloc.start()
            seg_tree = SumSegmentTree(arr)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.results.append(BenchmarkResult(
                name="SegmentTree",
                operation="memory_usage",
                size=size,
                time_taken=0,
                memory_used=peak,
                operations_count=1
            ))
            
            # Fenwick Tree memory
            tracemalloc.start()
            fenwick_tree = FenwickTree(arr)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.results.append(BenchmarkResult(
                name="FenwickTree", 
                operation="memory_usage",
                size=size,
                time_taken=0,
                memory_used=peak,
                operations_count=1
            ))
            
            # Union-Find memory
            tracemalloc.start()
            uf = UnionFind(size)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.results.append(BenchmarkResult(
                name="UnionFind",
                operation="memory_usage", 
                size=size,
                time_taken=0,
                memory_used=peak,
                operations_count=1
            ))
    
    def run_all_benchmarks(self):
        """Run all benchmark suites."""
        print("Starting comprehensive benchmarks...\n")
        
        self.benchmark_segment_tree_vs_fenwick()
        self.benchmark_union_find_variants()
        self.benchmark_lazy_propagation()
        self.benchmark_memory_usage()
        
        print(f"\nCompleted {len(self.results)} benchmark tests.")
    
    def print_results(self):
        """Print benchmark results in a formatted table."""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        # Group results by operation type
        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = []
            operations[result.operation].append(result)
        
        for op_name, op_results in operations.items():
            print(f"\n{op_name.upper()} BENCHMARK:")
            print("-" * 50)
            print(f"{'Structure':<20} {'Size':<8} {'Time(ms)':<10} {'Ops/sec':<12} {'Memory(KB)':<12}")
            print("-" * 50)
            
            for result in sorted(op_results, key=lambda x: (x.size, x.name)):
                memory_kb = result.memory_used / 1024 if result.memory_used > 0 else 0
                time_ms = result.time_taken * 1000
                
                print(f"{result.name:<20} {result.size:<8} {time_ms:<10.2f} "
                      f"{result.ops_per_second:<12.0f} {memory_kb:<12.1f}")
    
    def save_results_csv(self, filename: str = "benchmark_results.csv"):
        """Save results to CSV file."""
        try:
            import csv
            
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['name', 'operation', 'size', 'time_taken', 'memory_used', 
                             'operations_count', 'ops_per_second', 'time_per_op']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in self.results:
                    writer.writerow({
                        'name': result.name,
                        'operation': result.operation,
                        'size': result.size,
                        'time_taken': result.time_taken,
                        'memory_used': result.memory_used,
                        'operations_count': result.operations_count,
                        'ops_per_second': result.ops_per_second,
                        'time_per_op': result.time_per_op
                    })
            
            print(f"Results saved to {filename}")
        except ImportError:
            print("CSV module not available, skipping save")
    
    def plot_results(self):
        """Create plots comparing performance."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            sns.set_style("whitegrid")
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Range Query Performance
            plt.subplot(2, 3, 1)
            query_results = [r for r in self.results if r.operation == 'range_query']
            
            segment_data = [(r.size, r.ops_per_second) for r in query_results if 'Segment' in r.name]
            fenwick_data = [(r.size, r.ops_per_second) for r in query_results if 'Fenwick' in r.name]
            
            if segment_data:
                seg_sizes, seg_ops = zip(*segment_data)
                plt.plot(seg_sizes, seg_ops, 'o-', label='Segment Tree', linewidth=2)
            
            if fenwick_data:
                fen_sizes, fen_ops = zip(*fenwick_data)
                plt.plot(fen_sizes, fen_ops, 's-', label='Fenwick Tree', linewidth=2)
            
            plt.xlabel('Array Size')
            plt.ylabel('Operations/Second')
            plt.title('Range Query Performance')
            plt.legend()
            plt.yscale('log')
            
            # Plot 2: Update Performance
            plt.subplot(2, 3, 2)
            update_results = [r for r in self.results if r.operation == 'point_update']
            
            segment_data = [(r.size, r.ops_per_second) for r in update_results if 'Segment' in r.name]
            fenwick_data = [(r.size, r.ops_per_second) for r in update_results if 'Fenwick' in r.name]
            
            if segment_data:
                seg_sizes, seg_ops = zip(*segment_data)
                plt.plot(seg_sizes, seg_ops, 'o-', label='Segment Tree', linewidth=2)
            
            if fenwick_data:
                fen_sizes, fen_ops = zip(*fenwick_data)
                plt.plot(fen_sizes, fen_ops, 's-', label='Fenwick Tree', linewidth=2)
            
            plt.xlabel('Array Size')
            plt.ylabel('Operations/Second')
            plt.title('Point Update Performance')
            plt.legend()
            plt.yscale('log')
            
            # Plot 3: Memory Usage
            plt.subplot(2, 3, 3)
            memory_results = [r for r in self.results if r.operation == 'memory_usage']
            
            segment_mem = [(r.size, r.memory_used/1024) for r in memory_results if 'Segment' in r.name]
            fenwick_mem = [(r.size, r.memory_used/1024) for r in memory_results if 'Fenwick' in r.name]
            uf_mem = [(r.size, r.memory_used/1024) for r in memory_results if 'UnionFind' in r.name]
            
            if segment_mem:
                sizes, mem = zip(*segment_mem)
                plt.plot(sizes, mem, 'o-', label='Segment Tree')
            
            if fenwick_mem:
                sizes, mem = zip(*fenwick_mem)
                plt.plot(sizes, mem, 's-', label='Fenwick Tree')
            
            if uf_mem:
                sizes, mem = zip(*uf_mem)
                plt.plot(sizes, mem, '^-', label='Union-Find')
            
            plt.xlabel('Input Size')
            plt.ylabel('Memory Usage (KB)')
            plt.title('Memory Usage Comparison')
            plt.legend()
            
            # Plot 4: Union-Find Operations
            plt.subplot(2, 3, 4)
            uf_results = [r for r in self.results if 'UnionFind' in r.name and r.operation == 'union']
            
            basic_data = [(r.size, r.ops_per_second) for r in uf_results if 'Basic' in r.name]
            optimized_data = [(r.size, r.ops_per_second) for r in uf_results if 'Optimized' in r.name]
            rollback_data = [(r.size, r.ops_per_second) for r in uf_results if 'Rollback' in r.name]
            
            if basic_data:
                sizes, ops = zip(*basic_data)
                plt.plot(sizes, ops, 'o-', label='Basic UF')
            
            if optimized_data:
                sizes, ops = zip(*optimized_data)
                plt.plot(sizes, ops, 's-', label='Optimized UF')
            
            if rollback_data:
                sizes, ops = zip(*rollback_data)
                plt.plot(sizes, ops, '^-', label='Rollback UF')
            
            plt.xlabel('Input Size')
            plt.ylabel('Operations/Second')
            plt.title('Union-Find Union Performance')
            plt.legend()
            plt.yscale('log')
            
            # Plot 5: Range Update Comparison
            plt.subplot(2, 3, 5)
            range_update_results = [r for r in self.results if r.operation == 'range_update']
            
            lazy_data = [(r.size, r.ops_per_second) for r in range_update_results if 'Lazy' in r.name]
            range_fenwick_data = [(r.size, r.ops_per_second) for r in range_update_results if 'RangeUpdate' in r.name]
            
            if lazy_data:
                sizes, ops = zip(*lazy_data)
                plt.plot(sizes, ops, 'o-', label='Lazy Segment Tree')
            
            if range_fenwick_data:
                sizes, ops = zip(*range_fenwick_data)
                plt.plot(sizes, ops, 's-', label='Range Update Fenwick')
            
            plt.xlabel('Input Size')
            plt.ylabel('Operations/Second')
            plt.title('Range Update Performance')
            plt.legend()
            
            # Plot 6: Time Complexity Verification
            plt.subplot(2, 3, 6)
            
            # Show theoretical vs actual complexity
            query_results = [r for r in self.results if r.operation == 'range_query' and 'Segment' in r.name]
            if query_results:
                sizes = [r.size for r in query_results]
                times = [r.time_per_op for r in query_results]
                
                plt.plot(sizes, times, 'o-', label='Actual Performance')
                
                # Theoretical O(log n) line
                import math
                theoretical = [times[0] * math.log2(size) / math.log2(sizes[0]) for size in sizes]
                plt.plot(sizes, theoretical, '--', label='Theoretical O(log n)')
            
            plt.xlabel('Input Size')
            plt.ylabel('Time per Operation (μs)')
            plt.title('Time Complexity Verification')
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("Matplotlib/Seaborn not available, skipping plots")
    
    def analyze_scalability(self):
        """Analyze how performance scales with input size."""
        print("\nSCALABILITY ANALYSIS:")
        print("="*50)
        
        # Group by structure and operation
        structure_ops = {}
        for result in self.results:
            key = (result.name, result.operation)
            if key not in structure_ops:
                structure_ops[key] = []
            structure_ops[key].append(result)
        
        for (structure, operation), results in structure_ops.items():
            if len(results) < 2:
                continue
                
            results.sort(key=lambda x: x.size)
            
            print(f"\n{structure} - {operation}:")
            print(f"  Size Range: {results[0].size:,} to {results[-1].size:,}")
            
            # Calculate scaling factor
            if len(results) >= 2:
                size_ratio = results[-1].size / results[0].size
                time_ratio = results[-1].time_per_op / results[0].time_per_op
                
                import math
                if size_ratio > 1:
                    complexity_factor = math.log(time_ratio) / math.log(size_ratio)
                    print(f"  Complexity scaling: O(n^{complexity_factor:.2f})")
                    
                    if 0.8 <= complexity_factor <= 1.2:
                        print("  → Close to O(n) scaling")
                    elif 0.8 <= complexity_factor <= 1.2:
                        print("  → Close to O(log n) scaling")
                    elif 1.8 <= complexity_factor <= 2.2:
                        print("  → Close to O(n log n) scaling")
            
            # Performance trend
            best_size = min(results, key=lambda x: x.time_per_op).size
            worst_size = max(results, key=lambda x: x.time_per_op).size
            print(f"  Best performance at size: {best_size:,}")
            print(f"  Worst performance at size: {worst_size:,}")
    
    def generate_recommendations(self):
        """Generate recommendations based on benchmark results."""
        print("\nRECOMMENDATIONS:")
        print("="*50)
        
        # Find best performers for each operation
        operations = set(r.operation for r in self.results)
        
        for operation in operations:
            if operation == 'memory_usage':
                continue
                
            op_results = [r for r in self.results if r.operation == operation]
            if not op_results:
                continue
            
            # Group by size and find best performer
            sizes = set(r.size for r in op_results)
            
            print(f"\n{operation.replace('_', ' ').title()}:")
            
            for size in sorted(sizes):
                size_results = [r for r in op_results if r.size == size]
                if len(size_results) < 2:
                    continue
                
                best = max(size_results, key=lambda x: x.ops_per_second)
                worst = min(size_results, key=lambda x: x.ops_per_second)
                
                speedup = best.ops_per_second / worst.ops_per_second if worst.ops_per_second > 0 else 1
                
                print(f"  Size {size:,}: {best.name} is {speedup:.1f}x faster than {worst.name}")
        
        # Memory efficiency recommendations
        memory_results = [r for r in self.results if r.operation == 'memory_usage']
        if memory_results:
            print(f"\nMemory Efficiency:")
            for size in sorted(set(r.size for r in memory_results)):
                size_results = [r for r in memory_results if r.size == size]
                if len(size_results) < 2:
                    continue
                
                best_mem = min(size_results, key=lambda x: x.memory_used)
                worst_mem = max(size_results, key=lambda x: x.memory_used)
                
                ratio = worst_mem.memory_used / best_mem.memory_used if best_mem.memory_used > 0 else 1
                print(f"  Size {size:,}: {best_mem.name} uses {ratio:.1f}x less memory than {worst_mem.name}")


def run_quick_benchmark():
    """Run a quick benchmark for basic verification."""
    print("Running quick benchmark...")
    
    # Quick test: small arrays, few operations
    arr = [random.randint(1, 100) for _ in range(100)]
    
    # Segment Tree timing
    start = time.perf_counter()
    seg_tree = SumSegmentTree(arr)
    for _ in range(100):
        left = random.randint(0, 99)
        right = random.randint(left, 99)
        seg_tree.query(left, right)
    seg_time = time.perf_counter() - start
    
    # Fenwick Tree timing
    start = time.perf_counter()
    fenwick_tree = FenwickTree(arr)
    for _ in range(100):
        left = random.randint(0, 99)
        right = random.randint(left, 99)
        fenwick_tree.range_sum(left, right)
    fenwick_time = time.perf_counter() - start
    
    # Union-Find timing
    start = time.perf_counter()
    uf = UnionFind(100)
    for _ in range(100):
        x = random.randint(0, 99)
        y = random.randint(0, 99)
        uf.union(x, y)
    uf_time = time.perf_counter() - start
    
    print(f"Quick Benchmark Results (100 operations on size 100):")
    print(f"  Segment Tree queries: {seg_time*1000:.2f} ms")
    print(f"  Fenwick Tree queries: {fenwick_time*1000:.2f} ms")
    print(f"  Union-Find unions: {uf_time*1000:.2f} ms")
    
    # Determine winner
    if fenwick_time < seg_time:
        print(f"  → Fenwick Tree is {seg_time/fenwick_time:.1f}x faster for range sums")
    else:
        print(f"  → Segment Tree is {fenwick_time/seg_time:.1f}x faster for range sums")


def main():
    """Main benchmark execution."""
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_benchmark()
        return
    
    benchmark = DataStructureBenchmark()
    
    try:
        benchmark.run_all_benchmarks()
        benchmark.print_results()
        benchmark.analyze_scalability()
        benchmark.generate_recommendations()
        benchmark.save_results_csv()
        benchmark.plot_results()
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
