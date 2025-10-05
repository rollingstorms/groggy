#!/usr/bin/env python3
"""
Compare __iter__ (implicit in for loops) vs explicit iteration methods
"""

import groggy as gr

g = gr.Graph()

# Create simple graph
for i in range(6):
    dept = 'Eng' if i < 3 else 'Sales'
    g.add_node(name=f'Person{i}', department=dept, value=float(i))

print("=" * 60)
print("SubgraphArray Iteration Comparison")
print("=" * 60)

groups = g.nodes.group_by('department')
print(f"\nCreated {len(groups)} groups")
print(f"Type: {type(groups)}")

# Test 1: Using __iter__ implicitly (for loop)
print("\n--- Test 1: Using for loop (implicit __iter__) ---")
try:
    for i, sg in enumerate(groups):
        print(f"  Group {i}: {sg.node_count()} nodes, type={type(sg)}")
    print("✓ For loop iteration works")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Check if there's an explicit iter() method
print("\n--- Test 2: Check for explicit iter() method ---")
if hasattr(groups, 'iter'):
    print(f"  groups.iter() exists: {groups.iter}")
    print(f"  Trying groups.iter()...")
    try:
        result = groups.iter()
        print(f"  Result: {result}, type={type(result)}")
    except Exception as e:
        print(f"  ✗ Error calling iter(): {e}")
else:
    print("  No explicit iter() method found")

# Test 3: Check what __iter__ returns
print("\n--- Test 3: Check __iter__ return type ---")
try:
    iterator = iter(groups)
    print(f"  iter(groups) returns: {type(iterator)}")
    print(f"  First item: {type(next(iterator))}")
    print("✓ __iter__ protocol works")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Compare with to_list()
print("\n--- Test 4: Compare with to_list() ---")
try:
    as_list = groups.to_list()
    print(f"  to_list() returns: {type(as_list)}, len={len(as_list)}")
    print(f"  First item: {type(as_list[0])}")
    print("✓ to_list() works")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: Show all available iteration-related methods
print("\n--- Test 5: Available iteration methods ---")
iter_methods = [m for m in dir(groups) if 'iter' in m.lower() or m == 'to_list']
for method in iter_methods:
    print(f"  - {method}")

print("\n" + "=" * 60)
print("Conclusion")
print("=" * 60)
