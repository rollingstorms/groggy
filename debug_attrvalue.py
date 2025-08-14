#!/usr/bin/env python3
"""
Debug AttrValue to understand how to extract the actual value.
"""

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy as gr

# Create a simple test
g = gr.Graph()
node_id = g.add_node(salary=75000, name="test")

node_view = g.nodes[node_id]
salary = node_view['salary']
name = node_view['name']

print(f"salary type: {type(salary)}")
print(f"salary value: {salary}")
print(f"salary attributes: {dir(salary)}")

print(f"name type: {type(name)}")
print(f"name value: {name}")
print(f"name attributes: {dir(name)}")

# Try different ways to extract value
print(f"\nTrying different extraction methods:")
try:
    print(f"salary.inner: {salary.inner}")
except AttributeError as e:
    print(f"salary.inner failed: {e}")

try:
    print(f"int(salary): {int(salary)}")
except Exception as e:
    print(f"int(salary) failed: {e}")

try:
    print(f"str(salary): {str(salary)}")
except Exception as e:
    print(f"str(salary) failed: {e}")

# Test direct comparison
try:
    result = salary > 50000
    print(f"salary > 50000: {result}")
except Exception as e:
    print(f"salary > 50000 failed: {e}")

# Check if it has methods
for attr in dir(salary):
    if not attr.startswith('_'):
        print(f"Method: {attr}")