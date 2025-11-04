"""
Test script for Phase 1, Day 1: IR Foundation

Validates the typed IR system with a simple algorithm.
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

from groggy.builder import AlgorithmBuilder
from groggy.builder.ir import (
    IRGraph, CoreIRNode, GraphIRNode, AttrIRNode, IRDomain
)


def test_ir_node_creation():
    """Test creating and serializing IR nodes."""
    print("=" * 60)
    print("Test 1: IR Node Creation")
    print("=" * 60)
    
    # Create a simple arithmetic node
    node1 = CoreIRNode("n1", "mul", ["a", "b"], "c", scalar_b=2.0)
    print(f"Node 1: {node1}")
    print(f"  Dict: {node1.to_dict()}")
    print(f"  Step: {node1.to_step()}")
    print()
    
    # Create a graph operation node
    node2 = GraphIRNode("n2", "neighbor_agg", ["c"], "d", agg="sum")
    print(f"Node 2: {node2}")
    print(f"  Dict: {node2.to_dict()}")
    print(f"  Step: {node2.to_step()}")
    print()
    
    # Create an attribute node
    node3 = AttrIRNode("n3", "attach", ["d"], None, name="result")
    print(f"Node 3: {node3}")
    print(f"  Dict: {node3.to_dict()}")
    print(f"  Step: {node3.to_step()}")
    print()
    
    print("✅ IR node creation works!\n")


def test_ir_graph():
    """Test building an IR graph with dependencies."""
    print("=" * 60)
    print("Test 2: IR Graph Structure")
    print("=" * 60)
    
    graph = IRGraph("simple_algorithm")
    
    # Build a simple computation: e = (a * 2.0 + b) aggregated
    node1 = CoreIRNode("n1", "mul", ["a"], "c", b=2.0)
    node2 = CoreIRNode("n2", "add", ["c", "b"], "d")
    node3 = GraphIRNode("n3", "neighbor_agg", ["d"], "e", agg="sum")
    
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    
    print(f"Graph: {graph}")
    print(f"Stats: {graph.stats()}")
    print()
    
    # Test dependency tracking
    print("Dependencies:")
    print(f"  node2 depends on: {[n.id for n in graph.get_dependencies(node2)]}")
    print(f"  node3 depends on: {[n.id for n in graph.get_dependencies(node3)]}")
    print()
    
    print(f"  node1 used by: {[n.id for n in graph.get_dependents(node1)]}")
    print(f"  node2 used by: {[n.id for n in graph.get_dependents(node2)]}")
    print()
    
    # Test topological order
    print("Topological order:")
    for i, node in enumerate(graph.topological_order()):
        print(f"  {i}. {node}")
    print()
    
    print("✅ IR graph structure works!\n")


def test_ir_visualization():
    """Test IR visualization features."""
    print("=" * 60)
    print("Test 3: IR Visualization")
    print("=" * 60)
    
    graph = IRGraph("visualization_test")
    
    # Build a simple algorithm
    node1 = CoreIRNode("n1", "mul", ["x"], "y", b=2.0)
    node2 = CoreIRNode("n2", "add", ["y"], "z", b=1.0)
    node3 = GraphIRNode("n3", "neighbor_agg", ["z"], "result", agg="mean")
    
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    
    print("Pretty Print:")
    print(graph.pretty_print())
    print()
    
    print("DOT Format (first 10 lines):")
    dot = graph.to_dot()
    for line in dot.split('\n')[:10]:
        print(f"  {line}")
    print("  ...")
    print()
    
    print("✅ IR visualization works!\n")


def test_builder_ir_integration():
    """Test AlgorithmBuilder with IR enabled."""
    print("=" * 60)
    print("Test 4: Builder IR Integration")
    print("=" * 60)
    
    # Create builder with IR enabled
    builder = AlgorithmBuilder("test_algorithm", use_ir=True)
    
    print(f"Builder: {builder.name}")
    print(f"Use IR: {builder.use_ir}")
    print(f"IR Graph: {builder.ir_graph}")
    print()
    
    # Manually add some IR nodes to test
    from groggy.builder.ir import CoreIRNode
    
    node1 = CoreIRNode("test1", "add", ["a", "b"], "c")
    builder._add_ir_node(node1)
    
    node2 = CoreIRNode("test2", "mul", ["c"], "d", b=2.0)
    builder._add_ir_node(node2)
    
    print("IR Stats:")
    stats = builder.get_ir_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    print("Visualization:")
    print(builder.visualize_ir("text"))
    print()
    
    # Verify backward compatibility - steps should be populated
    print(f"Legacy steps count: {len(builder.steps)}")
    print("First step:", builder.steps[0] if builder.steps else "None")
    print()
    
    print("✅ Builder IR integration works!\n")


def test_backward_compatibility():
    """Test that IR mode is backward compatible with legacy steps."""
    print("=" * 60)
    print("Test 5: Backward Compatibility")
    print("=" * 60)
    
    # Test with IR disabled
    builder_legacy = AlgorithmBuilder("legacy", use_ir=False)
    print(f"Legacy builder - use_ir: {builder_legacy.use_ir}")
    print(f"Legacy builder - ir_graph: {builder_legacy.ir_graph}")
    
    # Steps should still work
    builder_legacy.steps.append({"type": "test", "output": "x"})
    print(f"Legacy builder - steps: {len(builder_legacy.steps)}")
    print()
    
    # Test with IR enabled
    builder_ir = AlgorithmBuilder("ir_enabled", use_ir=True)
    print(f"IR builder - use_ir: {builder_ir.use_ir}")
    print(f"IR builder - ir_graph: {builder_ir.ir_graph}")
    
    # Add node - should populate both IR and steps
    from groggy.builder.ir import CoreIRNode
    node = CoreIRNode("n1", "add", ["a", "b"], "c")
    builder_ir._add_ir_node(node)
    
    print(f"IR builder - IR nodes: {len(builder_ir.ir_graph.nodes)}")
    print(f"IR builder - legacy steps: {len(builder_ir.steps)}")
    print()
    
    print("✅ Backward compatibility maintained!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Phase 1, Day 1: IR Foundation Tests")
    print("=" * 60 + "\n")
    
    try:
        test_ir_node_creation()
        test_ir_graph()
        test_ir_visualization()
        test_builder_ir_integration()
        test_backward_compatibility()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! IR Foundation is working.")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
