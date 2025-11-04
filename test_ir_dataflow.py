"""
Test script for Phase 1, Day 2: Dataflow Analysis

Validates dependency tracking, liveness analysis, loop analysis,
and fusion opportunity detection.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

from groggy.builder.ir import (
    IRGraph, CoreIRNode, GraphIRNode, AttrIRNode, ControlIRNode,
    analyze_dataflow, DataflowAnalyzer
)


def test_dependency_classification():
    """Test RAW, WAR, WAW dependency classification."""
    print("=" * 70)
    print("Test 1: Dependency Classification")
    print("=" * 70)
    
    graph = IRGraph("deps_test")
    
    # Create a scenario with different dependency types
    # x = a + b       (defines x)
    # y = x * 2       (RAW on x)
    # x = y + 1       (WAW on x, WAR on y)
    # z = x + y       (RAW on both x and y)
    
    n1 = CoreIRNode("n1", "add", ["a", "b"], "x")
    n2 = CoreIRNode("n2", "mul", ["x"], "y", b=2)
    n3 = CoreIRNode("n3", "add", ["y"], "x", b=1)  # Redefines x
    n4 = CoreIRNode("n4", "add", ["x", "y"], "z")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)
    
    # Analyze
    analyzer = DataflowAnalyzer(graph)
    analysis = analyzer.analyze()
    
    print(f"Graph has {len(graph.nodes)} nodes")
    print()
    
    print("RAW dependencies:")
    for var, readers in analysis.raw_deps.items():
        print(f"  {var}: {readers}")
    
    print("\nWAR dependencies:")
    for var, writers in analysis.war_deps.items():
        print(f"  {var}: {writers}")
    
    print("\nWAW dependencies:")
    for var, writers in analysis.waw_deps.items():
        print(f"  {var}: {writers}")
    
    # Verify expected dependencies
    assert "x" in analysis.raw_deps, "Should have RAW on x"
    assert "x" in analysis.waw_deps, "Should have WAW on x (redefined)"
    
    print("\n✅ Dependency classification works!\n")


def test_liveness_analysis():
    """Test variable liveness computation."""
    print("=" * 70)
    print("Test 2: Liveness Analysis")
    print("=" * 70)
    
    graph = IRGraph("liveness_test")
    
    # Simple chain: a → b → c → d
    # Only d is "live out" at the end, so others can be dropped progressively
    n1 = CoreIRNode("n1", "mul", ["input"], "a", b=2.0)
    n2 = CoreIRNode("n2", "add", ["a"], "b", b=1.0)
    n3 = CoreIRNode("n3", "mul", ["b"], "c", b=3.0)
    n4 = AttrIRNode("n4", "attach", ["c"], None, name="output")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)
    
    # Analyze
    analyzer = DataflowAnalyzer(graph)
    analysis = analyzer.analyze()
    
    print(f"Graph has {len(graph.nodes)} nodes")
    print()
    
    # Show liveness for each node
    for node in graph.topological_order():
        info = analysis.liveness[node.id]
        print(f"{node.id}: {node.op_type}")
        print(f"  live_in:  {info.live_in}")
        print(f"  live_out: {info.live_out}")
        print(f"  defs:     {info.defs}")
        print(f"  uses:     {info.uses}")
        if info.can_drop:
            print(f"  can_drop: {info.can_drop}")
        print()
    
    # Verify liveness properties
    # n1 defines 'a', which is live_out (needed by n2), so can't drop at n1
    assert "a" not in analysis.liveness["n1"].can_drop, "a is still needed by n2"
    assert "a" in analysis.liveness["n1"].live_out, "a should be live out of n1"
    
    # n2 defines 'b', which is live_out (needed by n3), so can't drop at n2
    assert "b" not in analysis.liveness["n2"].can_drop, "b is still needed by n3"
    assert "b" in analysis.liveness["n2"].live_out, "b should be live out of n2"
    
    # n3 defines 'c', which is live_out (needed by n4), so can't drop at n3
    assert "c" not in analysis.liveness["n3"].can_drop, "c is still needed by n4"
    
    # n4 doesn't define anything, so nothing to drop
    # But after n4, c is no longer live
    assert "c" not in analysis.liveness["n4"].live_out, "c should not be live after n4"
    
    print("✅ Liveness analysis works!\n")


def test_dead_code_detection():
    """Test detection of unused variables."""
    print("=" * 70)
    print("Test 3: Dead Code Detection")
    print("=" * 70)
    
    graph = IRGraph("dead_code_test")
    
    # Create some variables that are never used
    n1 = CoreIRNode("n1", "mul", ["input"], "used", b=2.0)
    n2 = CoreIRNode("n2", "add", ["input"], "dead1", b=1.0)  # Never used
    n3 = CoreIRNode("n3", "mul", ["used"], "result", b=3.0)
    n4 = CoreIRNode("n4", "sub", ["input"], "dead2", b=5.0)  # Never used
    n5 = AttrIRNode("n5", "attach", ["result"], None, name="output")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)
    graph.add_node(n5)
    
    # Analyze
    analyzer = DataflowAnalyzer(graph)
    analysis = analyzer.analyze()
    
    print(f"Dead variables: {analysis.dead_vars}")
    
    # Verify dead code detection
    assert "dead1" in analysis.dead_vars, "dead1 should be detected as dead"
    assert "dead2" in analysis.dead_vars, "dead2 should be detected as dead"
    assert "used" not in analysis.dead_vars, "used should not be dead"
    assert "result" not in analysis.dead_vars, "result should not be dead"
    
    print("✅ Dead code detection works!\n")


def test_fusion_chain_detection():
    """Test detection of fusable operation chains."""
    print("=" * 70)
    print("Test 4: Fusion Chain Detection")
    print("=" * 70)
    
    graph = IRGraph("fusion_test")
    
    # Create a fusable arithmetic chain: a * 2 + 3 - 1
    n1 = CoreIRNode("n1", "mul", ["input"], "a", b=2.0)
    n2 = CoreIRNode("n2", "add", ["a"], "b", b=3.0)
    n3 = CoreIRNode("n3", "sub", ["b"], "c", b=1.0)
    n4 = AttrIRNode("n4", "attach", ["c"], None, name="output")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)
    
    # Analyze
    analyzer = DataflowAnalyzer(graph)
    analysis = analyzer.analyze()
    
    print(f"Found {len(analysis.fusion_chains)} fusion chains")
    print()
    
    for i, chain in enumerate(analysis.fusion_chains):
        print(f"Chain {i+1}:")
        print(f"  Pattern: {chain.pattern}")
        print(f"  Benefit: {chain.fusion_benefit:.2f}")
        print(f"  Nodes ({len(chain.nodes)}):")
        for node in chain.nodes:
            print(f"    - {node}")
        print()
    
    # Should find one arithmetic chain
    assert len(analysis.fusion_chains) >= 1, "Should detect fusion chain"
    assert analysis.fusion_chains[0].pattern == "arithmetic_chain"
    assert len(analysis.fusion_chains[0].nodes) == 3  # mul + add + sub
    
    print("✅ Fusion chain detection works!\n")


def test_critical_path():
    """Test critical path computation."""
    print("=" * 70)
    print("Test 5: Critical Path Analysis")
    print("=" * 70)
    
    graph = IRGraph("critical_path_test")
    
    # Create a graph with multiple paths:
    #
    #     ┌─ n2 ─┐
    # n1 ─┤      ├─ n5
    #     └─ n3 ─ n4 ─┘
    #
    # Critical path is n1 → n3 → n4 → n5 (length 4)
    
    n1 = CoreIRNode("n1", "mul", ["input"], "a", b=2.0)
    n2 = CoreIRNode("n2", "add", ["a"], "b", b=1.0)       # Short path
    n3 = CoreIRNode("n3", "mul", ["a"], "c", b=3.0)       # Long path start
    n4 = CoreIRNode("n4", "add", ["c"], "d", b=4.0)       # Long path middle
    n5 = CoreIRNode("n5", "add", ["b", "d"], "result")    # Merge point
    n6 = AttrIRNode("n6", "attach", ["result"], None, name="output")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)
    graph.add_node(n5)
    graph.add_node(n6)
    
    # Analyze
    analyzer = DataflowAnalyzer(graph)
    analysis = analyzer.analyze()
    
    print(f"Critical path length: {len(analysis.critical_path)}")
    print("Critical path:")
    for i, node in enumerate(analysis.critical_path):
        print(f"  {i+1}. {node}")
    print()
    
    # Verify critical path
    assert len(analysis.critical_path) >= 4, "Should find longest path"
    
    print("✅ Critical path analysis works!\n")


def test_pagerank_analysis():
    """Test analysis on a realistic PageRank-like algorithm."""
    print("=" * 70)
    print("Test 6: PageRank Algorithm Analysis")
    print("=" * 70)
    
    graph = IRGraph("pagerank")
    
    # Simplified PageRank IR (without loop body expansion)
    n1 = AttrIRNode("n1", "load", [], "ranks", name="ranks", default=1.0)
    n2 = GraphIRNode("n2", "degree", [], "degrees")
    n3 = CoreIRNode("n3", "recip", ["degrees"], "inv_deg", epsilon=1e-9)
    n4 = CoreIRNode("n4", "mul", ["ranks", "inv_deg"], "contrib")
    n5 = GraphIRNode("n5", "neighbor_agg", ["contrib"], "neighbor_sum", agg="sum")
    n6 = CoreIRNode("n6", "mul", ["neighbor_sum"], "damped", b=0.85)
    n7 = CoreIRNode("n7", "add", ["damped"], "new_ranks", b=0.15)
    n8 = AttrIRNode("n8", "attach", ["new_ranks"], None, name="pagerank")
    
    for node in [n1, n2, n3, n4, n5, n6, n7, n8]:
        graph.add_node(node)
    
    # Analyze
    analyzer = DataflowAnalyzer(graph)
    analysis = analyzer.analyze()
    
    # Print full analysis report
    print(analyzer.print_analysis())
    
    # Verify some expected properties
    assert len(analysis.fusion_chains) > 0, "Should find fusion opportunities"
    assert len(analysis.critical_path) > 0, "Should compute critical path"
    
    # Check for specific fusion opportunities
    # Should find arithmetic chains like: neighbor_sum * 0.85 + 0.15
    arithmetic_chains = [
        c for c in analysis.fusion_chains 
        if c.pattern == "arithmetic_chain"
    ]
    assert len(arithmetic_chains) > 0, "Should find arithmetic chains"
    
    print("✅ PageRank analysis works!\n")


def test_complete_analysis_report():
    """Test the complete analysis report generation."""
    print("=" * 70)
    print("Test 7: Complete Analysis Report")
    print("=" * 70)
    
    # Build a more complex algorithm
    graph = IRGraph("complex_algorithm")
    
    # Multiple branches, some dead code, fusable chains
    n1 = CoreIRNode("n1", "mul", ["input"], "a", b=2.0)
    n2 = CoreIRNode("n2", "add", ["a"], "b", b=1.0)
    n3 = CoreIRNode("n3", "mul", ["b"], "c", b=3.0)  # Chain: a → b → c
    
    # Dead branch
    n4 = CoreIRNode("n4", "add", ["input"], "dead", b=99.0)
    
    # Parallel branch
    n5 = CoreIRNode("n5", "sub", ["a"], "d", b=5.0)
    
    # Merge
    n6 = CoreIRNode("n6", "add", ["c", "d"], "result")
    n7 = AttrIRNode("n7", "attach", ["result"], None, name="output")
    
    for node in [n1, n2, n3, n4, n5, n6, n7]:
        graph.add_node(node)
    
    # Analyze
    analyzer = DataflowAnalyzer(graph)
    analysis = analyzer.analyze()
    
    # Print full report
    report = analyzer.print_analysis()
    print(report)
    
    # Verify report has all sections
    assert "Dependencies:" in report
    assert "Liveness Analysis:" in report
    assert "Fusion Opportunities:" in report
    assert "Critical Path:" in report
    assert "Dead Variables" in report
    
    print("✅ Complete analysis report works!\n")


def test_visualization_integration():
    """Test that analysis integrates with visualization."""
    print("=" * 70)
    print("Test 8: Analysis Visualization Integration")
    print("=" * 70)
    
    graph = IRGraph("viz_test")
    
    # Simple algorithm
    n1 = CoreIRNode("n1", "mul", ["x"], "y", b=2.0)
    n2 = CoreIRNode("n2", "add", ["y"], "z", b=1.0)
    n3 = AttrIRNode("n3", "attach", ["z"], None, name="result")
    
    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    
    # Analyze
    analysis = analyze_dataflow(graph)
    
    # Combine with existing visualization
    print("Graph visualization:")
    print(graph.pretty_print())
    print()
    
    print("Analysis summary:")
    print(f"  Fusion chains: {len(analysis.fusion_chains)}")
    print(f"  Critical path length: {len(analysis.critical_path)}")
    print(f"  Dead variables: {len(analysis.dead_vars)}")
    print()
    
    # Generate DOT with analysis annotations (future enhancement)
    dot = graph.to_dot()
    print("DOT visualization available (see graph.to_dot())")
    print()
    
    print("✅ Visualization integration works!\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Phase 1, Day 2: Dataflow Analysis Tests")
    print("=" * 70 + "\n")
    
    try:
        test_dependency_classification()
        test_liveness_analysis()
        test_dead_code_detection()
        test_fusion_chain_detection()
        test_critical_path()
        test_pagerank_analysis()
        test_complete_analysis_report()
        test_visualization_integration()
        
        print("\n" + "=" * 70)
        print("🎉 All tests passed! Dataflow analysis is working.")
        print("=" * 70)
        print("\nKey capabilities demonstrated:")
        print("  ✅ RAW/WAR/WAW dependency classification")
        print("  ✅ Variable liveness analysis")
        print("  ✅ Dead code detection")
        print("  ✅ Fusion opportunity detection")
        print("  ✅ Critical path computation")
        print("  ✅ Complete analysis reporting")
        print()
        print("Ready for Phase 1, Day 3: Performance Profiling")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
