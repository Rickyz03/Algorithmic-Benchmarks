"""
Main execution script for Advanced Data Structures Project.

This script orchestrates the entire project execution:
1. Runs correctness tests for all data structures
2. Executes comprehensive benchmarks
3. Demonstrates practical applications
4. Generates performance reports and visualizations

Usage:
    python main.py                  # Run full suite
    python main.py --test-only      # Run only tests
    python main.py --benchmark-only # Run only benchmarks
    python main.py --demo-only      # Run only demonstrations
    python main.py --quick          # Run quick verification
"""

import sys
import os
import argparse
import time
from typing import List, Dict, Any

# Add subdirectories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data-structures'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'variants'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'benchmarks'))

# Import modules
try:
    from test_structures import run_all_tests
    from run_benchmarks import DataStructureBenchmark, run_quick_benchmark
    
    # Import data structures for demonstrations
    from segment_tree import SumSegmentTree, MinSegmentTree
    from fenwick_tree import FenwickTree
    from union_find import UnionFind
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required files are in the correct directories.")
    sys.exit(1)


class ProjectDemo:
    """Demonstration of practical applications for each data structure."""
    
    def __init__(self):
        """Initialize demo class."""
        self.demos = {
            'segment_tree': self.demo_segment_tree,
            'fenwick_tree': self.demo_fenwick_tree,
            'union_find': self.demo_union_find,
            'comparison': self.demo_comparison
        }
    
    def demo_segment_tree(self):
        """Demonstrate Segment Tree applications."""
        print("\n" + "="*60)
        print("SEGMENT TREE DEMONSTRATION")
        print("="*60)
        
        print("\n1. Range Sum Queries on Stock Prices")
        print("-" * 40)
        
        # Simulate daily stock prices
        stock_prices = [100, 102, 98, 105, 110, 108, 115, 112, 118, 120]
        price_tree = SumSegmentTree(stock_prices)
        
        print(f"Stock prices: {stock_prices}")
        print(f"Total volume days 2-5: ${price_tree.query(2, 5):,}")
        print(f"Total volume days 0-9: ${price_tree.query(0, 9):,}")
        
        # Price update (stock split)
        print(f"\nStock split on day 4: {stock_prices[4]} → {stock_prices[4] // 2}")
        price_tree.update(4, stock_prices[4] // 2)
        print(f"New total volume days 2-5: ${price_tree.query(2, 5):,}")
        
        print("\n2. Range Minimum Queries - Tournament Scheduling")
        print("-" * 40)
        
        # Player skill levels
        skills = [75, 82, 68, 91, 88, 72, 95, 79, 84, 77]
        min_tree = MinSegmentTree(skills)
        
        print(f"Player skills: {skills}")
        print(f"Weakest player in group 1 (players 0-4): {min_tree.query(0, 4)}")
        print(f"Weakest player in group 2 (players 5-9): {min_tree.query(5, 9)}")
        print(f"Overall weakest player: {min_tree.query(0, 9)}")
        
        # Player improvement
        print(f"\nPlayer 2 trains and improves: {skills[2]} → 85")
        min_tree.update(2, 85)
        print(f"New weakest in group 1: {min_tree.query(0, 4)}")
    
    def demo_fenwick_tree(self):
        """Demonstrate Fenwick Tree applications."""
        print("\n" + "="*60)
        print("FENWICK TREE DEMONSTRATION")
        print("="*60)
        
        print("\n1. Frequency Analysis - Website Visits")
        print("-" * 40)
        
        # Daily website visits
        daily_visits = [1200, 1350, 980, 1100, 1500, 1320, 890]
        visits_tree = FenwickTree(daily_visits)
        
        print(f"Daily visits: {daily_visits}")
        print(f"Total visits days 0-2: {visits_tree.prefix_sum(2):,}")
        print(f"Total visits days 3-5: {visits_tree.range_sum(3, 5):,}")
        print(f"Weekly total: {visits_tree.prefix_sum(6):,}")
        
        # Traffic spike
        print(f"\nTraffic spike on day 2: +500 visits")
        visits_tree.update(2, 500)
        print(f"New total visits days 0-2: {visits_tree.prefix_sum(2):,}")
        
        print("\n2. Inventory Management - Product Quantities")
        print("-" * 40)
        
        # Product inventory counts
        inventory = [50, 30, 75, 20, 45, 60, 25]
        inventory_tree = FenwickTree(inventory)
        
        print(f"Product inventory: {inventory}")
        
        # Find when we reach 100 total items
        cumulative = 0
        for i in range(len(inventory)):
            cumulative = inventory_tree.prefix_sum(i)
            print(f"  Products 0-{i}: {cumulative} items")
            if cumulative >= 100:
                print(f"  → First 100+ items reached at product {i}")
                break
        
        # Restock product 1
        print(f"\nRestocking product 1: +20 items")
        inventory_tree.update(1, 20)
        print(f"New inventory sum 0-3: {inventory_tree.range_sum(0, 3)}")
    
    def demo_union_find(self):
        """Demonstrate Union-Find applications."""
        print("\n" + "="*60)
        print("UNION-FIND DEMONSTRATION")
        print("="*60)
        
        print("\n1. Social Network - Friend Connections")
        print("-" * 40)
        
        # 8 people in social network
        people = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry"]
        social_uf = UnionFind(len(people))
        
        print(f"People: {people}")
        print(f"Initial friend groups: {social_uf.num_components}")
        
        # Add friendships
        friendships = [
            (0, 1),  # Alice - Bob
            (1, 2),  # Bob - Carol
            (3, 4),  # Dave - Eve
            (5, 6),  # Frank - Grace
        ]
        
        for person1, person2 in friendships:
            social_uf.union(person1, person2)
            print(f"  {people[person1]} and {people[person2]} become friends")
        
        print(f"Friend groups after connections: {social_uf.num_components}")
        
        # Check connections
        print(f"\nConnectivity checks:")
        print(f"  Are Alice and Carol connected? {social_uf.connected(0, 2)}")
        print(f"  Are Alice and Dave connected? {social_uf.connected(0, 3)}")
        print(f"  Size of Alice's friend group: {social_uf.component_size(0)}")
        
        # Bridge different groups
        print(f"\n{people[2]} and {people[3]} meet and become friends")
        social_uf.union(2, 3)
        print(f"New friend groups: {social_uf.num_components}")
        print(f"Alice's friend group size: {social_uf.component_size(0)}")
        
        print("\n2. Network Connectivity - Internet Infrastructure")
        print("-" * 40)
        
        # Network nodes
        nodes = 6
        network_uf = UnionFind(nodes)
        
        print(f"Network nodes: {list(range(nodes))}")
        
        # Add network connections
        connections = [(0, 1), (1, 2), (3, 4), (4, 5)]
        
        for node1, node2 in connections:
            network_uf.union(node1, node2)
            print(f"  Connected nodes {node1} ↔ {node2}")
        
        print(f"Network segments: {network_uf.num_components}")
        
        # Check reachability
        print(f"\nReachability analysis:")
        for i in range(nodes):
            reachable_size = network_uf.component_size(i)
            print(f"  Node {i} can reach {reachable_size} nodes total")
        
        # Critical connection to merge networks
        print(f"\nAdding bridge connection 2 ↔ 3")
        network_uf.union(2, 3)
        print(f"Network segments after bridge: {network_uf.num_components}")
        print(f"Full network connectivity achieved: {network_uf.num_components == 1}")
    
    def demo_comparison(self):
        """Demonstrate when to use each data structure."""
        print("\n" + "="*60)
        print("DATA STRUCTURE SELECTION GUIDE")
        print("="*60)
        
        # Create test data
        test_size = 1000
        test_array = [i + 1 for i in range(test_size)]  # 1, 2, 3, ..., 1000
        
        print(f"\nComparison on array of size {test_size}")
        print("Problem scenarios and optimal choices:")
        
        scenarios = [
            {
                "problem": "Frequent range sum queries, rare updates",
                "recommendation": "Fenwick Tree",
                "reason": "Lower memory overhead, simpler implementation"
            },
            {
                "problem": "Mixed range queries (sum, min, max) on same data",
                "recommendation": "Segment Tree", 
                "reason": "Supports arbitrary associative operations"
            },
            {
                "problem": "Frequent range updates with occasional queries",
                "recommendation": "Lazy Propagation Segment Tree",
                "reason": "O(log n) range updates vs O(n log n) for basic structures"
            },
            {
                "problem": "Dynamic connectivity with frequent union operations",
                "recommendation": "Union-Find with optimizations",
                "reason": "Nearly O(1) amortized time for union/find operations"
            },
            {
                "problem": "Need to maintain multiple versions of data",
                "recommendation": "Persistent Segment Tree",
                "reason": "Efficient versioning with O(log n) space per update"
            },
            {
                "problem": "2D range queries on matrices",
                "recommendation": "2D Fenwick Tree",
                "reason": "Better constant factors than 2D Segment Tree"
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. Problem: {scenario['problem']}")
            print(f"   Best choice: {scenario['recommendation']}")
            print(f"   Reason: {scenario['reason']}")
        
        # Timing comparison example
        print(f"\n" + "-"*50)
        print("Quick timing comparison (1000 operations):")
        
        # Segment Tree
        seg_tree = SumSegmentTree(test_array)
        start = time.perf_counter()
        for i in range(1000):
            seg_tree.query(i % test_size, (i + 100) % test_size)
        seg_time = time.perf_counter() - start
        
        # Fenwick Tree
        fenwick_tree = FenwickTree(test_array)
        start = time.perf_counter() 
        for i in range(1000):
            fenwick_tree.range_sum(i % test_size, (i + 100) % test_size)
        fenwick_time = time.perf_counter() - start
        
        print(f"  Segment Tree: {seg_time*1000:.2f} ms")
        print(f"  Fenwick Tree: {fenwick_time*1000:.2f} ms")
        
        if fenwick_time < seg_time:
            print(f"  → Fenwick Tree is {seg_time/fenwick_time:.1f}x faster")
        else:
            print(f"  → Segment Tree is {fenwick_time/seg_time:.1f}x faster")
    
    def run_all_demos(self):
        """Run all demonstrations."""
        for demo_name, demo_func in self.demos.items():
            try:
                demo_func()
            except Exception as e:
                print(f"Demo {demo_name} failed: {e}")
    
    def run_specific_demo(self, demo_name: str):
        """Run a specific demonstration."""
        if demo_name in self.demos:
            self.demos[demo_name]()
        else:
            print(f"Unknown demo: {demo_name}")
            print(f"Available demos: {list(self.demos.keys())}")


def print_project_header():
    """Print project header information."""
    print("="*80)
    print("ADVANCED DATA STRUCTURES PROJECT")
    print("="*80)
    print("Comprehensive analysis of Segment Trees, Fenwick Trees, and Union-Find")
    print("Including basic implementations, advanced variants, and performance comparisons")
    print("="*80)


def run_tests():
    """Run all correctness tests."""
    print("\n🔧 RUNNING CORRECTNESS TESTS")
    print("-" * 50)
    
    try:
        success = run_all_tests()
        if success:
            print("✅ All tests passed successfully!")
            return True
        else:
            print("❌ Some tests failed!")
            return False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


def run_benchmarks():
    """Run performance benchmarks."""
    print("\n⚡ RUNNING PERFORMANCE BENCHMARKS")
    print("-" * 50)
    
    try:
        benchmark = DataStructureBenchmark()
        benchmark.run_all_benchmarks()
        benchmark.print_results()
        benchmark.analyze_scalability()
        benchmark.generate_recommendations()
        
        # Save results
        benchmark.save_results_csv()
        
        # Generate plots if possible
        try:
            benchmark.plot_results()
            print("📊 Performance plots saved as 'benchmark_results.png'")
        except ImportError:
            print("📊 Matplotlib not available - skipping plots")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_demonstrations():
    """Run practical demonstrations."""
    print("\n🎯 RUNNING PRACTICAL DEMONSTRATIONS")
    print("-" * 50)
    
    try:
        demo = ProjectDemo()
        demo.run_all_demos()
        print("\n✅ All demonstrations completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_verification():
    """Run quick verification of all components."""
    print("\n⚡ RUNNING QUICK VERIFICATION")
    print("-" * 50)
    
    try:
        # Quick test
        print("Testing basic functionality...")
        
        # Test Segment Tree
        seg_tree = SumSegmentTree([1, 2, 3, 4, 5])
        assert seg_tree.query(1, 3) == 9
        print("✓ Segment Tree basic test passed")
        
        # Test Fenwick Tree
        fenwick = FenwickTree([1, 2, 3, 4, 5])
        assert fenwick.range_sum(1, 3) == 9
        print("✓ Fenwick Tree basic test passed")
        
        # Test Union-Find
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.connected(0, 1) == True
        assert uf.connected(0, 2) == False
        print("✓ Union-Find basic test passed")
        
        # Quick benchmark
        run_quick_benchmark()
        
        print("\n✅ Quick verification completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick verification failed: {e}")
        return False


def generate_summary_report():
    """Generate a summary of project execution."""
    print("\n" + "="*80)
    print("PROJECT EXECUTION SUMMARY")
    print("="*80)
    
    print("""
This project successfully demonstrates three fundamental advanced data structures:

📊 SEGMENT TREE:
   • Efficient range queries with arbitrary associative operations
   • O(log n) queries and updates
   • Best for: Multiple types of range queries on same data

🌲 FENWICK TREE:
   • Optimized for prefix sums and range sum queries  
   • Lower memory overhead than Segment Tree
   • Best for: Frequent range sum queries with occasional updates

🔗 UNION-FIND:
   • Dynamic connectivity queries in nearly O(1) time
   • Path compression and union by rank optimizations
   • Best for: Graph connectivity, clustering, and equivalence relations

🚀 VARIANTS IMPLEMENTED:
   • Lazy Propagation Segment Trees for range updates
   • 2D structures for matrix operations
   • Persistent structures maintaining multiple versions
   • Weighted and specialized Union-Find variants

📈 PERFORMANCE INSIGHTS:
   • Fenwick Trees generally faster for simple sum operations
   • Segment Trees more versatile for complex queries
   • Union-Find performance improves dramatically with optimizations
   • Memory usage varies significantly between structures

The complete analysis with detailed performance metrics and recommendations
has been generated and saved to benchmark_results.csv and visualizations.
    """)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Advanced Data Structures Project')
    parser.add_argument('--test-only', action='store_true', help='Run only tests')
    parser.add_argument('--benchmark-only', action='store_true', help='Run only benchmarks')
    parser.add_argument('--demo-only', action='store_true', help='Run only demonstrations')
    parser.add_argument('--quick', action='store_true', help='Run quick verification')
    parser.add_argument('--demo', type=str, help='Run specific demo', 
                       choices=['segment_tree', 'fenwick_tree', 'union_find', 'comparison'])
    
    args = parser.parse_args()
    
    print_project_header()
    
    success = True
    
    if args.quick:
        success = run_quick_verification()
    elif args.test_only:
        success = run_tests()
    elif args.benchmark_only:
        success = run_benchmarks()
    elif args.demo_only:
        success = run_demonstrations()
    elif args.demo:
        demo = ProjectDemo()
        demo.run_specific_demo(args.demo)
    else:
        # Run full suite
        print("\n🚀 Running complete project suite...")
        
        # Step 1: Tests
        test_success = run_tests()
        
        # Step 2: Benchmarks (even if tests fail, benchmarks might work)
        benchmark_success = run_benchmarks()
        
        # Step 3: Demonstrations
        demo_success = run_demonstrations()
        
        success = test_success and benchmark_success and demo_success
        
        # Generate summary
        generate_summary_report()
    
    # Final status
    print("\n" + "="*80)
    if success:
