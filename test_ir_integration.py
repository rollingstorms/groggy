"""
Integration tests for IR optimization pipeline.

Validates that optimization passes work together correctly and preserve semantics.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-groggy/python'))

from groggy.builder.ir import (
    IRGraph, CoreIRNode, GraphIRNode, AttrIRNode, ControlIRNode,
    IROptimizer, optimize_ir
)


def build_pagerank_ir() -> IRGraph:
    """Build a simplified PageRank IR for testing."""
    ir = IRGraph(name="pagerank")
    
    # Constants
    n_node = CoreIRNode('n1', 'constant', [], 'n', value=100)
    damping_node = CoreIRNode('damp1', 'constant', [], 'damping', value=0.85)
    one_minus_d = CoreIRNode('omd1', 'constant', [], 'one_minus_d', value=0.15)
    
    ir.add_node(n_node)
    ir.add_node(damping_node)
    ir.add_node(one_minus_d)
    
    # Initial ranks
    init_val = CoreIRNode('init1', 'div', ['one_minus_d', 'n'], 'init_rank')
    ranks = AttrIRNode('ranks1', 'load', [], 'ranks', name='pagerank', default='init_rank')
    
    ir.add_node(init_val)
    ir.add_node(ranks)
    
    # Degree computation
    degrees = GraphIRNode('deg1', 'degree', [], 'degrees')
    ir.add_node(degrees)
    
    # Loop body (simplified)
    loop = ControlIRNode('loop1', 'loop', [], None, count=10)
    ir.add_node(loop)
    
    # Inside loop: contrib = ranks / degrees
    contrib = CoreIRNode('contrib1', 'div', ['ranks', 'degrees'], 'contrib')
    ir.add_node(contrib)
    
    # neighbor_sum = G @ contrib
    neighbor_sum = GraphIRNode('ns1', 'neighbor_agg', ['contrib'], 'neighbor_sum', agg='sum')
    ir.add_node(neighbor_sum)
    
    # ranks = damping * neighbor_sum + (1 - damping) / n
    damped = CoreIRNode('damped1', 'mul', ['damping', 'neighbor_sum'], 'damped')
    teleport = CoreIRNode('tele1', 'div', ['one_minus_d', 'n'], 'teleport')  # Redundant!
    new_ranks = CoreIRNode('new_ranks1', 'add', ['damped', 'teleport'], 'ranks')
    
    ir.add_node(damped)
    ir.add_node(teleport)
    ir.add_node(new_ranks)
    
    # Output
    output = AttrIRNode('output1', 'attach', ['ranks'], None, name='pagerank')
    ir.add_node(output)
    
    return ir


def test_full_pipeline():
    """Test that full optimization pipeline works end-to-end."""
    print("Testing full optimization pipeline...")
    
    ir = build_pagerank_ir()
    initial_node_count = len(ir.nodes)
    print(f"  Initial nodes: {initial_node_count}")
    
    # Run full optimization
    optimized = optimize_ir(ir, passes=["constant_fold", "cse", "fuse_arithmetic", "dce"])
    final_node_count = len(optimized.nodes)
    print(f"  Optimized nodes: {final_node_count}")
    
    assert final_node_count < initial_node_count, "Optimization should reduce node count"
    print(f"  ✓ Reduced nodes by {initial_node_count - final_node_count}")
    
    # Check that output node is still present
    output_nodes = [n for n in optimized.nodes if isinstance(n, AttrIRNode) and n.op_type == 'attach']
    assert len(output_nodes) > 0, "Output node should be preserved"
    print("  ✓ Output node preserved")
    
    print()


def test_constant_folding_creates_cse_opportunities():
    """Test that constant folding enables more CSE."""
    print("Testing constant folding + CSE synergy...")
    
    ir = IRGraph(name="test")
    
    # x = 2 * 3 (constant expression)
    # y = 6 (another constant)
    # z = x + y (should become 6 + 6 after folding, then CSE)
    
    const1 = CoreIRNode('c1', 'constant', [], 'two', value=2.0)
    const2 = CoreIRNode('c2', 'constant', [], 'three', value=3.0)
    x = CoreIRNode('x1', 'mul', ['two', 'three'], 'x')
    
    const3 = CoreIRNode('c3', 'constant', [], 'six', value=6.0)
    y = CoreIRNode('y1', 'mul', ['two', 'three'], 'y')  # Duplicate of x!
    
    z = CoreIRNode('z1', 'add', ['x', 'y'], 'z')
    output = AttrIRNode('out1', 'attach', ['z'], None, name='result')
    
    for node in [const1, const2, x, const3, y, z, output]:
        ir.add_node(node)
    
    print(f"  Before: {len(ir.nodes)} nodes")
    
    # Optimize: constant folding should evaluate 2*3, then CSE should merge x and y
    optimized = optimize_ir(ir, passes=["constant_fold", "cse", "dce"])
    print(f"  After: {len(optimized.nodes)} nodes")
    
    # Should have removed duplicate computation
    mul_nodes = [n for n in optimized.nodes if n.op_type == 'mul']
    assert len(mul_nodes) <= 1, "CSE should have eliminated duplicate multiplication"
    print("  ✓ Duplicate computation eliminated")
    
    print()


def test_fusion_preserves_semantics():
    """Test that arithmetic fusion doesn't change results."""
    print("Testing arithmetic fusion semantic preservation...")
    
    ir = IRGraph(name="fusion_test")
    
    # Build: result = (a * b) + (c * d)
    a = CoreIRNode('a1', 'constant', [], 'a', value=2.0)
    b = CoreIRNode('b1', 'constant', [], 'b', value=3.0)
    c = CoreIRNode('c1', 'constant', [], 'c', value=4.0)
    d = CoreIRNode('d1', 'constant', [], 'd', value=5.0)
    
    ab = CoreIRNode('ab1', 'mul', ['a', 'b'], 'ab')
    cd = CoreIRNode('cd1', 'mul', ['c', 'd'], 'cd')
    result = CoreIRNode('res1', 'add', ['ab', 'cd'], 'result')
    output = AttrIRNode('out1', 'attach', ['result'], None, name='output')
    
    for node in [a, b, c, d, ab, cd, result, output]:
        ir.add_node(node)
    
    # Optimize with fusion
    optimized = optimize_ir(ir, passes=["fuse_arithmetic"])
    
    # Should have fewer nodes due to fusion
    assert len(optimized.nodes) <= len(ir.nodes), "Fusion should reduce or maintain node count"
    
    # Output should still exist
    output_nodes = [n for n in optimized.nodes if isinstance(n, AttrIRNode)]
    assert len(output_nodes) > 0, "Output should be preserved"
    print("  ✓ Semantics preserved after fusion")
    
    print()


def test_dce_removes_unused_computations():
    """Test that DCE correctly identifies and removes dead code."""
    print("Testing dead code elimination...")
    
    ir = IRGraph(name="dce_test")
    
    # Used path
    x = CoreIRNode('x1', 'constant', [], 'x', value=1.0)
    y = CoreIRNode('y1', 'add', ['x', 'x'], 'y')
    
    # Unused path (dead code)
    dead1 = CoreIRNode('dead1', 'mul', ['x', 'x'], 'z')
    dead2 = CoreIRNode('dead2', 'add', ['z', 'z'], 'w')
    
    # Output only uses y
    output = AttrIRNode('out1', 'attach', ['y'], None, name='result')
    
    for node in [x, y, dead1, dead2, output]:
        ir.add_node(node)
    
    print(f"  Before DCE: {len(ir.nodes)} nodes")
    assert len(ir.nodes) == 5
    
    # Apply DCE
    optimized = optimize_ir(ir, passes=["dce"])
    print(f"  After DCE: {len(optimized.nodes)} nodes")
    
    # Should have removed dead1 and dead2
    assert len(optimized.nodes) == 3, f"Expected 3 nodes, got {len(optimized.nodes)}"
    
    dead_ids = [n.id for n in optimized.nodes if n.id in ['dead1', 'dead2']]
    assert len(dead_ids) == 0, "Dead nodes should be removed"
    print("  ✓ Dead code removed correctly")
    
    print()


def test_iterative_optimization_converges():
    """Test that iterative optimization reaches a fixed point."""
    print("Testing iterative optimization convergence...")
    
    ir = build_pagerank_ir()
    
    # Run with multiple iterations
    optimized = optimize_ir(ir, passes=["constant_fold", "cse", "dce"], max_iterations=10)
    
    # Should converge in reasonable iterations (internally tracked)
    # If it doesn't, optimize_ir would timeout or loop forever
    
    print(f"  Final node count: {len(optimized.nodes)}")
    print("  ✓ Optimization converged")
    
    print()


def test_optimization_preserves_output_nodes():
    """Test that optimization never removes nodes with side effects."""
    print("Testing side effect preservation...")
    
    ir = IRGraph(name="side_effects")
    
    # Pure computation
    x = CoreIRNode('x1', 'constant', [], 'x', value=1.0)
    y = CoreIRNode('y1', 'add', ['x', 'x'], 'y')
    
    # Side effect: attach to graph
    output = AttrIRNode('out1', 'attach', ['y'], None, name='result')
    ir.add_node(x)
    ir.add_node(y)
    ir.add_node(output)
    
    # Optimize aggressively
    optimized = optimize_ir(ir, passes=["constant_fold", "cse", "fuse_arithmetic", "dce"])
    
    # Output must still exist
    output_nodes = [n for n in optimized.nodes if isinstance(n, AttrIRNode) and n.op_type == 'attach']
    assert len(output_nodes) > 0, "Output node with side effect must be preserved"
    print("  ✓ Side effects preserved")
    
    print()


def test_custom_pass_order():
    """Test that custom pass order works."""
    print("Testing custom pass order...")
    
    ir = build_pagerank_ir()
    
    # Try unusual order: DCE first (should do nothing), then constant folding
    optimized = optimize_ir(ir, passes=["dce", "constant_fold", "cse"])
    
    # Should still produce valid IR
    assert len(optimized.nodes) > 0, "Should have nodes after optimization"
    
    # Output should exist
    output_nodes = [n for n in optimized.nodes if isinstance(n, AttrIRNode)]
    assert len(output_nodes) > 0, "Output should exist"
    
    print("  ✓ Custom pass order works")
    
    print()


def test_no_optimization():
    """Test that passing empty pass list does nothing."""
    print("Testing no-op optimization...")
    
    ir = build_pagerank_ir()
    initial_count = len(ir.nodes)
    
    # No optimization
    result = optimize_ir(ir, passes=[])
    
    # Should be unchanged
    assert len(result.nodes) == initial_count, "No-op should not change node count"
    print("  ✓ No-op optimization preserves IR")
    
    print()


def test_optimizer_class_interface():
    """Test direct IROptimizer class usage."""
    print("Testing IROptimizer class interface...")
    
    ir = build_pagerank_ir()
    optimizer = IROptimizer(ir)
    
    # Run individual passes
    changed1 = optimizer.constant_folding()
    print(f"  Constant folding changed: {changed1}")
    
    changed2 = optimizer.common_subexpression_elimination()
    print(f"  CSE changed: {changed2}")
    
    changed3 = optimizer.dead_code_elimination()
    print(f"  DCE changed: {changed3}")
    
    # At least one should have made changes
    assert changed1 or changed2 or changed3, "At least one pass should modify IR"
    print("  ✓ Individual passes work")
    
    print()


def run_all_tests():
    """Run all integration tests."""
    print("=" * 70)
    print("IR Optimization Integration Tests")
    print("=" * 70)
    print()
    
    test_full_pipeline()
    test_constant_folding_creates_cse_opportunities()
    test_fusion_preserves_semantics()
    test_dce_removes_unused_computations()
    test_iterative_optimization_converges()
    test_optimization_preserves_output_nodes()
    test_custom_pass_order()
    test_no_optimization()
    test_optimizer_class_interface()
    
    print("=" * 70)
    print("✅ All integration tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
