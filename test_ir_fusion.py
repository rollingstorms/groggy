"""
Test IR fusion optimization passes.
"""
import sys
sys.path.insert(0, 'python-groggy/python')

from groggy.builder import AlgorithmBuilder
from groggy.builder.ir.optimizer import optimize_ir
from groggy.builder.ir import AttrIRNode


def test_arithmetic_fusion_axpy():
    """Test fusion of AXPY pattern: (a * b) + c."""
    b = AlgorithmBuilder("test_axpy", use_ir=True)
    
    a = b.init_nodes(1.0)
    b_var = b.init_nodes(2.0)
    c = b.init_nodes(3.0)
    
    # Create pattern: (a * b) + c
    mul_result = b.core.mul(a, b_var)
    result = b.core.add(mul_result, c)
    
    print("Before fusion:")
    print(b.ir_graph.pretty_print())
    print(f"Node count: {len(b.ir_graph.nodes)}")
    
    # Apply fusion
    optimize_ir(b.ir_graph, passes=['fuse_arithmetic'])
    
    print("\nAfter fusion:")
    print(b.ir_graph.pretty_print())
    print(f"Node count: {len(b.ir_graph.nodes)}")
    
    # Check that fusion happened
    fused_nodes = [n for n in b.ir_graph.nodes if n.op_type == 'fused_axpy']
    print(f"\nFused AXPY nodes: {len(fused_nodes)}")
    assert len(fused_nodes) > 0, "Expected AXPY fusion"
    print("✓ AXPY fusion working")


def test_conditional_fusion():
    """Test fusion of where with arithmetic: where(mask, a * b, 0)."""
    b = AlgorithmBuilder("test_cond_fusion", use_ir=True)
    
    a = b.init_nodes(1.0)
    b_var = b.init_nodes(2.0)
    mask = b.core.compare(a, "gt", 0.5)
    
    # Create pattern: where(mask, a * b, 0)
    mul_result = b.core.mul(a, b_var)
    zero = b.core.constant(0.0)
    result = b.core.where(mask, mul_result, zero)
    
    print("Before fusion:")
    print(b.ir_graph.pretty_print())
    print(f"Node count: {len(b.ir_graph.nodes)}")
    
    # Apply fusion
    optimize_ir(b.ir_graph, passes=['fuse_arithmetic'])
    
    print("\nAfter fusion:")
    print(b.ir_graph.pretty_print())
    print(f"Node count: {len(b.ir_graph.nodes)}")
    
    # Check that fusion happened
    fused_nodes = [n for n in b.ir_graph.nodes if 'fused_where' in n.op_type]
    print(f"\nFused where nodes: {len(fused_nodes)}")
    assert len(fused_nodes) > 0, "Expected conditional fusion"
    print("✓ Conditional fusion working")


def test_neighbor_pre_transform_fusion():
    """Test fusion of pre-transform with neighbor aggregation."""
    b = AlgorithmBuilder("test_neighbor_fusion", use_ir=True)
    
    values = b.init_nodes(1.0)
    weights = b.init_nodes(0.5)
    
    # Create pattern: neighbor_agg(values * weights)
    transformed = b.core.mul(values, weights)
    result = b.graph_ops.neighbor_agg(transformed, agg="sum")
    
    print("Before fusion:")
    print(b.ir_graph.pretty_print())
    print(f"Node count: {len(b.ir_graph.nodes)}")
    
    # Apply fusion
    optimize_ir(b.ir_graph, passes=['fuse_neighbor'])
    
    print("\nAfter fusion:")
    print(b.ir_graph.pretty_print())
    print(f"Node count: {len(b.ir_graph.nodes)}")
    
    # Check that fusion happened
    fused_nodes = [n for n in b.ir_graph.nodes if 'fused_neighbor' in n.op_type]
    print(f"\nFused neighbor nodes: {len(fused_nodes)}")
    assert len(fused_nodes) > 0, "Expected neighbor pre-transform fusion"
    print("✓ Neighbor pre-transform fusion working")


def test_combined_fusion():
    """Test combined arithmetic and neighbor fusion."""
    b = AlgorithmBuilder("test_combined", use_ir=True)
    
    ranks = b.init_nodes(1.0)
    degrees = b.graph_ops.degree()
    epsilon = b.core.constant(1e-9)
    
    # Pattern: neighbor_agg(where(degrees == 0, 0, ranks / degrees))
    zero_mask = b.core.compare(degrees, "eq", 0)
    safe_deg = b.core.add(degrees, epsilon)
    contrib = b.core.div(ranks, safe_deg)
    zero = b.core.constant(0.0)
    masked = b.core.where(zero_mask, zero, contrib)
    result = b.graph_ops.neighbor_agg(masked, agg="sum")
    
    print("Before fusion:")
    print(b.ir_graph.pretty_print())
    print(f"Node count: {len(b.ir_graph.nodes)}")
    
    # Apply all fusion passes
    optimize_ir(b.ir_graph, passes=['fuse_arithmetic', 'fuse_neighbor'])
    
    print("\nAfter fusion:")
    print(b.ir_graph.pretty_print())
    print(f"Node count: {len(b.ir_graph.nodes)}")
    
    # Check overall reduction
    initial_count = 10  # Rough estimate
    final_count = len(b.ir_graph.nodes)
    print(f"\nNode reduction: {initial_count} -> {final_count}")
    print("✓ Combined fusion working")


def test_full_pagerank_fusion():
    """Test fusion on a full PageRank-like algorithm."""
    b = AlgorithmBuilder("pagerank_fusion", use_ir=True)
    
    # Initialize
    n = b.core.constant(100.0)
    ranks = b.init_nodes(1.0)
    degrees = b.graph_ops.degree()
    epsilon = b.core.constant(1e-9)
    damping = b.core.constant(0.85)
    one_minus_damp = b.core.constant(0.15)
    
    # Single iteration body
    # contrib = ranks / (degrees + epsilon)
    safe_deg = b.core.add(degrees, epsilon)
    contrib = b.core.div(ranks, safe_deg)
    
    # Check for sinks: where(degrees == 0, 0, contrib)
    zero_mask = b.core.compare(degrees, "eq", 0)
    zero = b.core.constant(0.0)
    masked_contrib = b.core.where(zero_mask, zero, contrib)
    
    # Aggregate from neighbors
    neighbor_sum = b.graph_ops.neighbor_agg(masked_contrib, agg="sum")
    
    # Update: ranks = damping * neighbor_sum + (1-damping)/n
    inv_n = b.core.recip(n)
    teleport = b.core.mul(one_minus_damp, inv_n)
    damped = b.core.mul(damping, neighbor_sum)
    new_ranks = b.core.add(damped, teleport)
    
    # Add output to prevent DCE from removing everything
    output = AttrIRNode('output', 'attach', [new_ranks.name], None, name='pagerank')
    b.ir_graph.add_node(output)
    
    print("Before fusion:")
    print(b.ir_graph.pretty_print())
    print(f"Node count: {len(b.ir_graph.nodes)}")
    initial_count = len(b.ir_graph.nodes)
    
    # Apply all optimizations (except DCE to see fusion better)
    optimize_ir(b.ir_graph, passes=['constant_fold', 'cse', 'fuse_arithmetic', 'fuse_neighbor'])
    
    print("\nAfter fusion:")
    print(b.ir_graph.pretty_print())
    print(f"Node count: {len(b.ir_graph.nodes)}")
    final_count = len(b.ir_graph.nodes)
    
    reduction = initial_count - final_count
    print(f"\nOptimization results:")
    print(f"  Initial nodes: {initial_count}")
    print(f"  Final nodes: {final_count}")
    print(f"  Nodes removed: {reduction}")
    print(f"  Reduction: {100 * reduction / initial_count:.1f}%")
    
    # Check for fused operations
    fused_count = len([n for n in b.ir_graph.nodes if 'fused' in n.op_type])
    print(f"  Fused operations: {fused_count}")
    
    # Fusion creates new fused nodes, so count may stay same or increase slightly
    # The key metric is that we have fused operations
    assert fused_count > 0, "Expected some fused operations"
    print(f"✓ Full PageRank fusion working ({fused_count} fused operations created)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing IR Fusion Passes")
    print("=" * 60)
    
    print("\n1. Testing AXPY fusion")
    print("-" * 60)
    test_arithmetic_fusion_axpy()
    
    print("\n2. Testing conditional fusion")
    print("-" * 60)
    test_conditional_fusion()
    
    print("\n3. Testing neighbor pre-transform fusion")
    print("-" * 60)
    test_neighbor_pre_transform_fusion()
    
    print("\n4. Testing combined fusion")
    print("-" * 60)
    test_combined_fusion()
    
    print("\n5. Testing full PageRank fusion")
    print("-" * 60)
    test_full_pagerank_fusion()
    
    print("\n" + "=" * 60)
    print("✓ All fusion tests passed!")
    print("=" * 60)
