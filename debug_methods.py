#!/usr/bin/env python3
"""Debug script to see what methods are available on Graph objects"""

import groggy as gr

g = gr.Graph()

print("Available methods on Graph:")
methods = [method for method in dir(g) if not method.startswith('_')]
for method in sorted(methods):
    print(f"  {method}")

print(f"\nTotal methods: {len(methods)}")

# Check specific missing methods
missing_methods = ['adjacency_matrix', 'adjacency', 'commit', 'create_branch', 'table']
print(f"\nChecking for specific methods:")
for method in missing_methods:
    has_method = hasattr(g, method)
    print(f"  {method}: {'✅' if has_method else '❌'}")