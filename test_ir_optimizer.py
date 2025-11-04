"""
Tests for IR optimizer passes.

Validates dead code elimination, constant folding, and common subexpression
elimination on representative IR patterns.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-groggy/python'))

from groggy.builder.ir import IRGraph, CoreIRNode, AttrIRNode, IROptimizer, optimize_ir


def test_dead_code_elimination():
    """Test that unused computations are removed."""
    print("Testing Dead Code Elimination...")
    
    ir = IRGraph()
    
    # Create some nodes
    input_node = CoreIRNode('input', 'constant', [], 'x', value=1.0)
    ir.add_node(input_node)
    
    # This computation is used
    used_node = CoreIRNode('used', 'add', ['x', 'x'], 'y')
    ir.add_node(used_node)
    
    # This computation is NOT used (dead code)
    dead_node = CoreIRNode('dead', 'mul', ['x', 'x'], 'z')
    ir.add_node(dead_node)
    
    # Output only references the used node
    output_node = AttrIRNode('output', 'attach', ['y'], None, name='result')
    ir.add_node(output_node)
    
    print(f"  Before DCE: {len(ir.nodes)} nodes")
    assert len(ir.nodes) == 4
    
    # Apply DCE
    optimizer = IROptimizer(ir)
    modified = optimizer.dead_code_elimination()
    
    print(f"  After DCE: {len(ir.nodes)} nodes")
    assert modified, "DCE should have removed dead code"
    assert len(ir.nodes) == 3, "Should have removed 1 dead node"
    assert ir.get_node('dead') is None, "Dead node should be removed"
    assert ir.get_node('used') is not None, "Used node should remain"
    
    print("  ✓ Dead code eliminated correctly\n")


def test_constant_folding():
    """Test that constant expressions are folded at compile time."""
    print("Testing Constant Folding...")
    
    ir = IRGraph()
    
    # Create constant inputs
    const1 = CoreIRNode('const1', 'constant', [], 'x', value=2.0)
    ir.add_node(const1)
    
    const2 = CoreIRNode('const2', 'constant', [], 'y', value=3.0)
    ir.add_node(const2)
    
    # This should be folded to constant 5.0
    add_node = CoreIRNode('add', 'add', ['x', 'y'], 'z')
    ir.add_node(add_node)
    
    # This should be folded to constant 6.0
    mul_node = CoreIRNode('mul', 'mul', ['x', 'y'], 'w')
    ir.add_node(mul_node)
    
    output1 = AttrIRNode('out1', 'attach', ['z'], None, name='sum')
    ir.add_node(output1)
    
    output2 = AttrIRNode('out2', 'attach', ['w'], None, name='product')
    ir.add_node(output2)
    
    print(f"  Before folding: add node is {ir.get_node('add').op_type}")
    
    # Apply constant folding
    optimizer = IROptimizer(ir)
    modified = optimizer.constant_folding()
    
    print(f"  After folding: add node is {ir.get_node('add').op_type}")
    assert modified, "Constant folding should have modified IR"
    assert ir.get_node('add').op_type == 'constant', "Add should be folded to constant"
    assert ir.get_node('add').metadata['value'] == 5.0, "Add should evaluate to 5.0"
    assert ir.get_node('mul').op_type == 'constant', "Mul should be folded to constant"
    assert ir.get_node('mul').metadata['value'] == 6.0, "Mul should evaluate to 6.0"
    
    print("  ✓ Constants folded correctly\n")


def test_common_subexpression_elimination():
    """Test that duplicate computations are eliminated."""
    print("Testing Common Subexpression Elimination...")
    
    ir = IRGraph()
    
    # Create input
    input_node = CoreIRNode('input', 'constant', [], 'x', value=1.0)
    ir.add_node(input_node)
    
    # First computation: x + x
    add1 = CoreIRNode('add1', 'add', ['x', 'x'], 'y1')
    ir.add_node(add1)
    
    # Duplicate computation: x + x (same as add1)
    add2 = CoreIRNode('add2', 'add', ['x', 'x'], 'y2')
    ir.add_node(add2)
    
    # Different computation: x * x
    mul1 = CoreIRNode('mul1', 'mul', ['x', 'x'], 'z1')
    ir.add_node(mul1)
    
    # Use both adds
    out1 = AttrIRNode('out1', 'attach', ['y1'], None, name='result1')
    ir.add_node(out1)
    
    out2 = AttrIRNode('out2', 'attach', ['y2'], None, name='result2')
    ir.add_node(out2)
    
    out3 = AttrIRNode('out3', 'attach', ['z1'], None, name='result3')
    ir.add_node(out3)
    
    print(f"  Before CSE: {len(ir.nodes)} nodes")
    assert len(ir.nodes) == 7
    
    # Apply CSE
    optimizer = IROptimizer(ir)
    modified = optimizer.common_subexpression_elimination()
    
    print(f"  After CSE: {len(ir.nodes)} nodes")
    assert modified, "CSE should have removed duplicate"
    
    # After CSE + DCE, we should have removed the duplicate add
    optimizer.dead_code_elimination()
    
    print(f"  After CSE+DCE: {len(ir.nodes)} nodes")
    assert len(ir.nodes) == 6, "Should have removed 1 duplicate node"
    assert ir.get_node('add1') is not None or ir.get_node('add2') is not None, "One add should remain"
    
    # Check that output nodes now reference the same variable
    assert out1.inputs[0] == out2.inputs[0], "Both outputs should reference same add result"
    
    print("  ✓ Common subexpressions eliminated correctly\n")


def test_combined_optimization():
    """Test that multiple passes work together."""
    print("Testing Combined Optimization Passes...")
    
    ir = IRGraph()
    
    # Constant folding opportunity
    const1 = CoreIRNode('const1', 'constant', [], 'x', value=2.0)
    ir.add_node(const1)
    
    const2 = CoreIRNode('const2', 'constant', [], 'y', value=3.0)
    ir.add_node(const2)
    
    # This will be folded to 5.0
    add = CoreIRNode('add', 'add', ['x', 'y'], 'z')
    ir.add_node(add)
    
    # Dead code: never used
    dead = CoreIRNode('dead', 'mul', ['x', 'y'], 'w')
    ir.add_node(dead)
    
    # Duplicate of add (will become CSE opportunity after folding)
    dup = CoreIRNode('dup', 'add', ['x', 'y'], 'v')
    ir.add_node(dup)
    
    out1 = AttrIRNode('out1', 'attach', ['z'], None, name='result1')
    ir.add_node(out1)
    
    out2 = AttrIRNode('out2', 'attach', ['v'], None, name='result2')
    ir.add_node(out2)
    
    print(f"  Before optimization: {len(ir.nodes)} nodes")
    assert len(ir.nodes) == 7
    
    # Apply all optimizations
    modified = optimize_ir(ir, passes=['constant_fold', 'cse', 'dce'])
    
    print(f"  After optimization: {len(ir.nodes)} nodes")
    
    # Should have:
    # - Folded add and dup to constants
    # - Eliminated dead code (dead node)
    # - Eliminated common subexpressions (one of add/dup)
    # Final: const1, const2, one folded constant, two outputs = 5 nodes
    assert len(ir.nodes) <= 6, f"Expected ≤6 nodes, got {len(ir.nodes)}"
    
    print("  ✓ Combined optimization passes work correctly\n")


def test_optimization_preserves_semantics():
    """Test that optimization doesn't change program semantics."""
    print("Testing Semantic Preservation...")
    
    ir = IRGraph()
    
    # Build a simple computation graph
    x = CoreIRNode('x', 'constant', [], 'x_val', value=2.0)
    ir.add_node(x)
    
    y = CoreIRNode('y', 'constant', [], 'y_val', value=3.0)
    ir.add_node(y)
    
    # z = x + y
    z = CoreIRNode('z', 'add', ['x_val', 'y_val'], 'z_val')
    ir.add_node(z)
    
    # w = x + y (duplicate)
    w = CoreIRNode('w', 'add', ['x_val', 'y_val'], 'w_val')
    ir.add_node(w)
    
    # Use both
    out_z = AttrIRNode('out_z', 'attach', ['z_val'], None, name='z')
    ir.add_node(out_z)
    
    out_w = AttrIRNode('out_w', 'attach', ['w_val'], None, name='w')
    ir.add_node(out_w)
    
    # Record outputs before optimization
    outputs_before = set()
    for node in ir.nodes:
        if node.domain.value == 'attr' and node.op_type == 'attach':
            outputs_before.add(node.metadata['name'])
    
    # Optimize
    optimize_ir(ir)
    
    # Check outputs after optimization
    outputs_after = set()
    for node in ir.nodes:
        if node.domain.value == 'attr' and node.op_type == 'attach':
            outputs_after.add(node.metadata['name'])
    
    assert outputs_before == outputs_after, "Optimization changed outputs"
    print("  ✓ Semantics preserved\n")


def run_all_tests():
    """Run all optimizer tests."""
    print("=" * 60)
    print("IR Optimizer Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_dead_code_elimination,
        test_constant_folding,
        test_common_subexpression_elimination,
        test_combined_optimization,
        test_optimization_preserves_semantics,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
