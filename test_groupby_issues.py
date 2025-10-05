#!/usr/bin/env python3
"""
Test the group_by issues reported
"""

import groggy as gr

g = gr.Graph()

# Create test data
for i in range(6):
    obj_name = 'ObjA' if i < 3 else 'ObjB'
    g.add_node(object_name=obj_name, successful_methods=i, value=float(i))

for i in range(4):
    obj_name = 'ObjA' if i < 2 else 'ObjB'
    success = True if i % 2 == 0 else False
    g.add_edge(i, i+1, object_name=obj_name, success=success, weight=float(i))

print("=" * 60)
print("Issue 1: g.nodes.table().group_by(['object_name'])['successful_methods']")
print("=" * 60)
try:
    result = g.nodes.table().group_by(['object_name'])['successful_methods']
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Issue 2: g.nodes.group_by(['object_name'])")
print("=" * 60)
try:
    result = g.nodes.group_by(['object_name'])
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Issue 3: g.edges.group_by('object_name')['success'] (boolean)")
print("=" * 60)
try:
    result = g.edges.group_by('object_name')['success']
    print(f"✓ Success: {result}")
    print(f"  Type: {type(result)}")
    # Try to sum booleans
    if hasattr(result, 'sum'):
        print(f"  Can sum: {result.sum()}")
    else:
        print(f"  No sum() method available")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
