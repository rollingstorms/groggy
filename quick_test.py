#!/usr/bin/env python3
import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy')

import groggy as gr

# Test the fixed matrix method
print("🧪 Quick Matrix Test")
print("=" * 20)

# Create numeric-only table (all Int type)
ages = gr.array([25, 30, 35, 40, 45])
heights = gr.array([175, 180, 165, 170, 185])  # Also Int type
table = gr.table([ages, heights], column_names=['age', 'height'])

print(f"Original table shape: {table.shape}")

# This should work now
matrix = table.matrix()
print(f"✅ Matrix conversion successful!")
print(f"Matrix shape: {matrix.shape}")
print(f"Matrix dtype: {matrix.dtype}")