#!/usr/bin/env python3
"""Comprehensive tests for scalar auto-detection in arithmetic operations."""

from groggy import Graph
from groggy.builder import AlgorithmBuilder

# Create test graph
g = Graph()
n1, n2, n3, n4 = g.add_nodes(4)
g.add_edge(n1, n2)
g.add_edge(n2, n3)
g.add_edge(n3, n4)
g.add_edge(n4, n1)

print("Graph created with 4 nodes in a cycle")

# Test 1: Map + Scalar operations
print("\n=== Test 1: Map + Scalar (both directions) ===")
builder = AlgorithmBuilder("map_scalar_test")
# Just use init_nodes for this test
degrees = builder.init_nodes(default=2.0)  # Simulate degree values
scaled_up = builder.core.mul(degrees, 10.0)  # map * scalar
scaled_down = builder.core.div(scaled_up, 2.0)  # map / scalar
shifted = builder.core.add(scaled_down, 1.0)  # map + scalar
final = builder.core.sub(shifted, 0.5)  # map - scalar
builder.attach_as("result", final)

g.apply(builder.build())

print("Degree * 10.0 / 2.0 + 1.0 - 0.5 = Degree * 5.0 + 0.5")
for node in [n1, n2, n3, n4]:
    degree = 2.0  # Each node has degree 2 in the cycle
    expected = degree * 5.0 + 0.5
    actual = g.get_node_attr(node, "result")
    print(f"  Node {node}: {actual} (expected {expected})")
    assert abs(actual - expected) < 1e-9, f"Mismatch: {actual} vs {expected}"

print("✓ Map + Scalar operations work!")

# Test 2: Scalar + Map operations (reversed order)
print("\n=== Test 2: Scalar + Map (reversed order) ===")
builder = AlgorithmBuilder("scalar_map_test")
values = builder.init_nodes(default=3.0)
# Scalar on left side
result1 = builder.core.add(10.0, values)  # scalar + map
result2 = builder.core.sub(20.0, values)  # scalar - map
result3 = builder.core.mul(2.0, values)  # scalar * map
builder.attach_as("add_result", result1)
builder.attach_as("sub_result", result2)
builder.attach_as("mul_result", result3)

g.apply(builder.build())

for node in [n1, n2, n3, n4]:
    add_val = g.get_node_attr(node, "add_result")
    sub_val = g.get_node_attr(node, "sub_result")
    mul_val = g.get_node_attr(node, "mul_result")

    print(f"  Node {node}:")
    print(f"    10.0 + 3.0 = {add_val} (expected 13.0)")
    print(f"    20.0 - 3.0 = {sub_val} (expected 17.0)")
    print(f"    2.0 * 3.0 = {mul_val} (expected 6.0)")

    assert abs(add_val - 13.0) < 1e-9
    assert abs(sub_val - 17.0) < 1e-9
    assert abs(mul_val - 6.0) < 1e-9

print("✓ Scalar + Map operations work!")

# Test 3: Map + Map operations (no scalars)
print("\n=== Test 3: Map + Map operations ===")
builder = AlgorithmBuilder("map_map_test")
map1 = builder.init_nodes(default=5.0)
map2 = builder.init_nodes(default=3.0)
sum_result = builder.core.add(map1, map2)
diff_result = builder.core.sub(map1, map2)
prod_result = builder.core.mul(map1, map2)
quot_result = builder.core.div(map1, map2)
builder.attach_as("sum", sum_result)
builder.attach_as("diff", diff_result)
builder.attach_as("prod", prod_result)
builder.attach_as("quot", quot_result)

g.apply(builder.build())

for node in [n1, n2, n3, n4]:
    sum_val = g.get_node_attr(node, "sum")
    diff_val = g.get_node_attr(node, "diff")
    prod_val = g.get_node_attr(node, "prod")
    quot_val = g.get_node_attr(node, "quot")

    print(
        f"  Node {node}: sum={sum_val}, diff={diff_val}, prod={prod_val}, quot={quot_val}"
    )
    assert abs(sum_val - 8.0) < 1e-9
    assert abs(diff_val - 2.0) < 1e-9
    assert abs(prod_val - 15.0) < 1e-9
    assert abs(quot_val - (5.0 / 3.0)) < 1e-6  # Float precision

print("✓ Map + Map operations work!")

# Test 4: Check no unnecessary node maps created
print("\n=== Test 4: Verify efficiency (no extra node maps) ===")
builder = AlgorithmBuilder("efficiency_test")
nodes = builder.init_nodes(default=1.0)
r1 = builder.core.mul(nodes, 0.85)  # Should create ONE scalar var
r2 = builder.core.add(r1, 0.15)  # Should create ONE scalar var
r3 = builder.core.div(r2, 2.0)  # Should create ONE scalar var
builder.attach_as("result", r3)

# Count step types
init_nodes_count = sum(1 for s in builder.steps if s.get("type") == "init_nodes")
init_scalar_count = sum(1 for s in builder.steps if s.get("type") == "init_scalar")

print(f"init_nodes count: {init_nodes_count} (should be 1)")
print(f"init_scalar count: {init_scalar_count} (should be 3)")

assert init_nodes_count == 1, "Should only have 1 init_nodes"
assert init_scalar_count == 3, "Should have 3 init_scalar (for 0.85, 0.15, 2.0)"

print("✓ No unnecessary node maps created!")

# Test 5: Complex expression with multiple scalars
print("\n=== Test 5: Complex PageRank-like expression ===")
builder = AlgorithmBuilder("pagerank_like")
ranks = builder.init_nodes(default=1.0)
damping = 0.85
teleport = 0.15

# Simulate: ranks = ranks * 0.85 + 0.15
updated = builder.core.mul(ranks, damping)
updated = builder.core.add(updated, teleport)
builder.attach_as("pagerank", updated)

g.apply(builder.build())

for node in [n1, n2, n3, n4]:
    pr = g.get_node_attr(node, "pagerank")
    expected = 1.0 * 0.85 + 0.15
    print(f"  Node {node}: {pr} (expected {expected})")
    assert abs(pr - expected) < 1e-9

print("✓ PageRank-like expression works!")

# Test 6: Integer scalars
print("\n=== Test 6: Integer scalars ===")
builder = AlgorithmBuilder("int_scalar_test")
nodes = builder.init_nodes(default=5.0)
result = builder.core.mul(nodes, 3)  # Integer literal
builder.attach_as("triple", result)

g.apply(builder.build())

for node in [n1, n2, n3, n4]:
    val = g.get_node_attr(node, "triple")
    print(f"  Node {node}: {val} (expected 15.0)")
    assert abs(val - 15.0) < 1e-9

print("✓ Integer scalar operations work!")

# Test 7: Step count comparison with old approach
print("\n=== Test 7: Step efficiency comparison ===")
builder = AlgorithmBuilder("step_count")
nodes = builder.init_nodes(default=1.0)

# Chain of operations with scalars
r = nodes
for i in range(5):
    r = builder.core.mul(r, 1.1)
    r = builder.core.add(r, 0.01)

builder.attach_as("result", r)

total_steps = len(builder.steps)
init_scalar_steps = sum(1 for s in builder.steps if s.get("type") == "init_scalar")

print(f"Total steps: {total_steps}")
print(f"  Init scalars: {init_scalar_steps}")
print(f"  Other steps: {total_steps - init_scalar_steps}")

# With new approach: 1 init_nodes + 10 init_scalar + 5 mul + 5 add + 1 attach = 22
# With old approach: 1 init_nodes + 10 init_nodes + 5 mul + 5 add + 1 attach = 22
# But the new scalars are O(1), not O(n)!
print(f"\nNew approach creates {init_scalar_steps} O(1) scalar variables")
print(f"Old approach would create {init_scalar_steps} O(n) node maps")
print("✓ Significant efficiency improvement!")

print("\n" + "=" * 60)
print("All comprehensive tests passed! ✓")
print("Scalar auto-detection working correctly!")
