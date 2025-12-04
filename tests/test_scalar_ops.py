#!/usr/bin/env python3
"""Test scalar operations in builder."""

from groggy import Graph
from groggy.builder import AlgorithmBuilder

# Create a simple test graph
g = Graph()
n1, n2, n3 = g.add_nodes(3)
g.add_edge(n1, n2)
g.add_edge(n2, n3)

print("Graph created with 3 nodes")

# Test 1: Basic scalar multiplication
print("\n=== Test 1: Scalar Multiplication ===")
builder = AlgorithmBuilder("test_scalar_mul")
nodes = builder.init_nodes(default=5.0)
result = builder.core.mul(
    nodes, 2.0
)  # Should create scalar variable instead of node map
builder.attach_as("scaled", result)

algo = builder.build()
g.apply(algo)

print(f"Input: 5.0 for each node")
print(f"Operation: mul by 2.0 (scalar)")
for node in [n1, n2, n3]:
    val = g.get_node_attr(node, "scaled")
    print(f"  Node {node}: {val}")
    assert abs(val - 10.0) < 1e-9, f"Expected 10.0, got {val}"

print("✓ Scalar multiplication works!")

# Test 2: Scalar addition
print("\n=== Test 2: Scalar Addition ===")
builder = AlgorithmBuilder("test_scalar_add")
nodes = builder.init_nodes(default=1.0)
result = builder.core.add(nodes, 0.5)
builder.attach_as("added", result)

algo = builder.build()
g.apply(algo)

print(f"Input: 1.0 for each node")
print(f"Operation: add 0.5 (scalar)")
for node in [n1, n2, n3]:
    val = g.get_node_attr(node, "added")
    print(f"  Node {node}: {val}")
    assert abs(val - 1.5) < 1e-9, f"Expected 1.5, got {val}"

print("✓ Scalar addition works!")

# Test 3: Scalar division
print("\n=== Test 3: Scalar Division ===")
builder = AlgorithmBuilder("test_scalar_div")
nodes = builder.init_nodes(default=10.0)
result = builder.core.div(nodes, 2.0)
builder.attach_as("divided", result)

algo = builder.build()
g.apply(algo)

print(f"Input: 10.0 for each node")
print(f"Operation: div by 2.0 (scalar)")
for node in [n1, n2, n3]:
    val = g.get_node_attr(node, "divided")
    print(f"  Node {node}: {val}")
    assert abs(val - 5.0) < 1e-9, f"Expected 5.0, got {val}"

print("✓ Scalar division works!")

# Test 4: Check step efficiency
print("\n=== Test 4: Check Step Efficiency ===")
builder = AlgorithmBuilder("test_spec_size")
nodes = builder.init_nodes(default=1.0)
result = builder.core.mul(nodes, 0.85)
result = builder.core.add(result, 0.15)
builder.attach_as("result", result)

# Check the internal steps
print(f"Number of steps: {len(builder.steps)}")
for i, step in enumerate(builder.steps):
    step_type = step.get("type")
    print(f"  Step {i}: {step_type}")
    if step_type == "init_scalar":
        print(f"    -> scalar value: {step.get('value')}")

# Should have: init_nodes, init_scalar (0.85), mul, init_scalar (0.15), add, attach_attr
expected_types = [
    "init_nodes",
    "init_scalar",
    "core.mul",
    "init_scalar",
    "core.add",
    "attach_attr",
]
actual_types = [step.get("type") for step in builder.steps]
print(f"\nExpected types: {expected_types}")
print(f"Actual types: {actual_types}")
assert actual_types == expected_types, f"Step sequence mismatch"

print("✓ Builder uses scalar variables efficiently!")

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("Scalar operations are now handled efficiently.")
