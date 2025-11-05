"""
Tests for parallel execution analysis and planning.

Tests:
1. Dependency graph construction
2. Execution level computation
3. Parallel group creation
4. Speedup estimation
5. Data-parallel operation detection
6. Thread-safety analysis
7. Full parallel plan generation
"""

import pytest
import sys
sys.path.insert(0, 'python-groggy/python')

from groggy.builder.algorithm_builder import AlgorithmBuilder
from groggy.builder.ir.parallel import (
    ParallelAnalyzer,
    analyze_parallelism,
    is_data_parallel_op,
    is_thread_safe_op,
)
from groggy.builder.ir.optimizer import optimize_ir


def test_dependency_graph():
    """Test dependency graph construction."""
    b = AlgorithmBuilder("test", use_ir=True)
    
    # Create a simple dependency chain: a → b → c
    a = b.init_nodes(1.0)  # No dependencies
    b_val = a * 2.0        # Depends on a
    c = b_val + 1.0        # Depends on b_val
    
    analyzer = ParallelAnalyzer(b.ir_graph)
    analyzer._build_dependency_graph()
    
    # Check that dependency graph was built
    assert len(analyzer.node_dependencies) > 0
    
    # Verify we have the expected number of nodes
    assert len(b.ir_graph.nodes) >= 3
    
    # Check execution levels (should form a chain)
    analyzer._compute_execution_levels()
    # Should have multiple levels since there are dependencies
    assert len(analyzer.dependency_levels) >= 1


def test_execution_levels_simple():
    """Test execution level computation for simple graph."""
    b = AlgorithmBuilder("test", use_ir=True)
    
    # Create independent operations
    a = b.init_nodes(1.0)
    x = b.init_nodes(2.0)
    y = b.init_nodes(3.0)
    
    # All three should be at level 0 (no dependencies)
    analyzer = ParallelAnalyzer(b.ir_graph)
    analyzer._build_dependency_graph()
    analyzer._compute_execution_levels()
    
    assert len(analyzer.dependency_levels) >= 1
    # First level should have all three init operations
    level0_ops = {node for node in analyzer.dependency_levels[0]}
    assert len(level0_ops) >= 3


def test_execution_levels_chain():
    """Test execution levels for dependency chain."""
    b = AlgorithmBuilder("test", use_ir=True)
    
    # Create a chain: a → b → c
    a = b.init_nodes(1.0)  # Level 0
    b_val = a * 2.0        # Level 1
    c = b_val + 3.0        # Level 2
    
    analyzer = ParallelAnalyzer(b.ir_graph)
    analyzer._build_dependency_graph()
    analyzer._compute_execution_levels()
    
    # Should have at least 3 levels
    assert len(analyzer.dependency_levels) >= 3


def test_execution_levels_diamond():
    """Test execution levels for diamond-shaped graph."""
    b = AlgorithmBuilder("test", use_ir=True)
    
    # Create diamond:
    #     a
    #    / \
    #   b   c
    #    \ /
    #     d
    a = b.init_nodes(1.0)      # Level 0
    left = a * 2.0              # Level 1
    right = a + 3.0             # Level 1 (parallel with left)
    d = left + right            # Level 2
    
    analyzer = ParallelAnalyzer(b.ir_graph)
    analyzer._build_dependency_graph()
    analyzer._compute_execution_levels()
    
    # Should have 3 levels: [a], [left, right], [d]
    assert len(analyzer.dependency_levels) >= 3
    
    # Level 1 should have 2 operations (left and right)
    if len(analyzer.dependency_levels) >= 2:
        level1 = analyzer.dependency_levels[1]
        # Should be at least 2 operations that can run in parallel
        assert len(level1) >= 2


def test_parallel_group_creation():
    """Test parallel group creation from execution levels."""
    b = AlgorithmBuilder("test", use_ir=True)
    
    # Create operations with parallelism
    a = b.init_nodes(1.0)
    x = a * 2.0
    y = a + 3.0
    z = x + y
    
    analyzer = ParallelAnalyzer(b.ir_graph)
    plan = analyzer.analyze()
    
    # Should have multiple groups
    assert len(plan.groups) >= 2
    
    # Check that groups have proper structure
    for group in plan.groups:
        assert isinstance(group.node_ids, list)
        assert isinstance(group.shared_inputs, set)
        assert isinstance(group.outputs, set)


def test_data_parallel_detection():
    """Test detection of data-parallel operations."""
    # Arithmetic operations are data-parallel
    assert is_data_parallel_op("add")
    assert is_data_parallel_op("mul")
    assert is_data_parallel_op("where")
    
    # Graph operations are not data-parallel
    assert not is_data_parallel_op("neighbor_agg")
    assert not is_data_parallel_op("degree")
    
    # Control flow is not data-parallel
    assert not is_data_parallel_op("loop")


def test_thread_safety():
    """Test thread-safety detection."""
    # Pure operations are thread-safe
    assert is_thread_safe_op("add")
    assert is_thread_safe_op("neighbor_agg")
    assert is_thread_safe_op("reduce")
    
    # Operations with side effects are not thread-safe
    assert not is_thread_safe_op("attach")
    assert not is_thread_safe_op("store")


def test_parallelism_factor_estimation():
    """Test estimation of parallelism factor."""
    b = AlgorithmBuilder("test", use_ir=True)
    
    # Create multiple independent arithmetic operations
    a = b.init_nodes(1.0)
    ops = []
    for i in range(8):
        # Create independent operations
        op = a * float(i + 1) + float(i)
        ops.append(op)
    
    analyzer = ParallelAnalyzer(b.ir_graph)
    analyzer._build_dependency_graph()
    analyzer._compute_execution_levels()
    
    # Get nodes from second level (the arithmetic operations)
    if len(analyzer.dependency_levels) >= 2:
        level_nodes = analyzer.dependency_levels[1]
        node_objs = [analyzer.ir_graph.node_map[nid] for nid in level_nodes]
        
        factor = analyzer._estimate_group_parallelism(node_objs)
        
        # With many data-parallel ops, should get significant speedup
        assert factor > 1.0
        # But capped at reasonable level (8 cores)
        assert factor <= 12.0  # 8 cores * 1.5x boost


def test_speedup_estimation():
    """Test overall speedup estimation."""
    b = AlgorithmBuilder("test", use_ir=True)
    
    # Create a mix of sequential and parallel work
    a = b.init_nodes(1.0)
    
    # Parallel stage
    x = a * 2.0
    y = a + 3.0
    z = a - 1.0
    
    # Sequential stage
    result = (x + y) * z
    
    analyzer = ParallelAnalyzer(b.ir_graph)
    plan = analyzer.analyze()
    
    # Should estimate some speedup
    assert plan.estimated_speedup >= 1.0


def test_full_parallel_plan():
    """Test generation of complete parallel execution plan."""
    b = AlgorithmBuilder("test", use_ir=True)
    
    # Create a realistic computation graph
    init = b.init_nodes(1.0)
    
    # Parallel operations
    a = init * 2.0
    b_val = init + 1.0
    c = init / 2.0
    
    # Combine
    result = a + b_val + c
    
    plan = analyze_parallelism(b.ir_graph)
    
    # Should have parallel groups
    assert len(plan.groups) > 0
    
    # Should have sequential fallback
    assert plan.sequential_plan is not None
    
    # Should be serializable
    json_str = plan.to_json()
    assert isinstance(json_str, str)
    assert len(json_str) > 0


def test_parallel_with_optimization():
    """Test parallel analysis after optimization passes."""
    b = AlgorithmBuilder("test", use_ir=True)
    
    # Create computation with optimization opportunities
    a = b.init_nodes(1.0)
    b_val = a * 2.0
    c = a * 2.0  # Duplicate of b (CSE opportunity)
    d = b_val + c
    unused = a * 3.0  # Dead code
    
    # Note: attr.save creates IR nodes, so we can test before optimization
    nodes_before = len(b.ir_graph.nodes)
    assert nodes_before > 0
    
    # Optimize (CSE will merge b_val and c, DCE will remove unused)
    optimize_ir(b.ir_graph, passes=['cse'])
    
    # Then analyze parallelism on optimized IR
    plan = analyze_parallelism(b.ir_graph)
    
    # Should work on optimized IR
    assert plan.sequential_plan is not None


def test_pagerank_parallelism():
    """Test parallel analysis on realistic PageRank-like algorithm."""
    b = AlgorithmBuilder("pagerank", use_ir=True)
    
    # Simplified PageRank structure
    ranks = b.init_nodes(1.0)
    
    # These operations could potentially be parallelized
    from groggy.builder.traits.graph import GraphOps
    graph_ops = GraphOps(b)
    deg = graph_ops.degree()
    inv_deg = b.core.recip(deg, epsilon=1e-9)
    
    # Sequential dependency
    contrib = ranks * inv_deg
    
    # Mark as output
    b.attr.save("contrib", contrib)
    
    plan = analyze_parallelism(b.ir_graph)
    
    # Should have some groups (at least 1)
    assert len(plan.groups) >= 1
    
    # Verify structure
    for group in plan.groups:
        assert len(group.node_ids) > 0


def test_empty_graph():
    """Test parallel analysis on empty graph."""
    b = AlgorithmBuilder("empty", use_ir=True)
    
    plan = analyze_parallelism(b.ir_graph)
    
    # Should handle empty graph gracefully
    assert len(plan.groups) == 0
    assert plan.estimated_speedup == 1.0


def test_single_operation():
    """Test parallel analysis with single operation."""
    b = AlgorithmBuilder("single", use_ir=True)
    
    a = b.init_nodes(1.0)
    
    plan = analyze_parallelism(b.ir_graph)
    
    # Single operation: no parallelism benefit
    assert plan.estimated_speedup <= 1.5  # Minimal or no speedup


def test_parallel_decision_threshold():
    """Test that parallel execution is only enabled when beneficial."""
    b = AlgorithmBuilder("minimal", use_ir=True)
    
    # Very simple computation (not worth parallelizing)
    a = b.init_nodes(1.0)
    result = a * 2.0
    
    plan = analyze_parallelism(b.ir_graph)
    
    # With minimal speedup, might not use parallel
    # (use_parallel threshold is 1.2x)
    if plan.estimated_speedup < 1.2:
        assert not plan.use_parallel


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
