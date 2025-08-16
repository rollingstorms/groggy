#!/usr/bin/env python3
"""
Demo script showing the rich display module in action.

Run this to see example outputs for GraphTable, GraphMatrix, and GraphArray.
"""

import sys
import os

# Add the python package to path so we can import groggy.display
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from groggy.display import format_table, format_matrix, format_array

def demo_table_display():
    """Demo GraphTable rich display."""
    print("=" * 60)
    print("GraphTable Display Demo")
    print("=" * 60)
    
    table_data = {
        'columns': ['name', 'city', 'age', 'score', 'joined'],
        'dtypes': {
            'name': 'string', 
            'city': 'category', 
            'age': 'int64', 
            'score': 'float32', 
            'joined': 'datetime'
        },
        'data': [
            ['Alice', 'NYC', 25, 91.5, '2024-02-15'],
            ['Bob', 'Paris', 30, 87.0, '2023-11-20'],
            ['Charlie', 'Tokyo', 35, None, '2024-06-10'],
            ['Diana', 'London', 28, 95.2, '2022-08-05'],
            ['Eve', 'Berlin', 32, 88.7, '2023-03-12'],
            ['Frank', 'Sydney', 29, 92.1, '2024-01-18'],
            ['Grace', 'Toronto', 31, 89.3, '2022-12-03'],
            ['Henry', 'Dubai', 27, 94.8, '2024-05-22'],
            ['Iris', 'Singapore', 33, 86.4, '2023-09-14'],
            ['Jack', 'Mumbai', 26, 93.7, '2024-03-08'],
            ['Kate', 'São Paulo', 34, 90.1, '2022-11-27'],
            ['Liam', 'Amsterdam', 30, 91.9, '2023-07-16'],
        ],
        'shape': (1000, 5),  # Simulating larger dataset
        'nulls': {'score': 12},
        'index_type': 'int64'
    }
    
    formatted = format_table(table_data)
    print(formatted)
    print()

def demo_matrix_display():
    """Demo GraphMatrix rich display."""
    print("=" * 60)
    print("GraphMatrix Display Demo")
    print("=" * 60)
    
    # Small matrix
    small_matrix_data = {
        'data': [
            [0.12, -1.50, 2.00, 0.00],
            [3.14, 0.00, float('nan'), 1.25],
            [-0.01, 4.50, -2.30, 8.00]
        ],
        'shape': (3, 4),
        'dtype': 'f32'
    }
    
    formatted_small = format_matrix(small_matrix_data)
    print(formatted_small)
    print()
    
    # Large matrix (will be truncated)
    large_data = []
    for i in range(500):
        row = []
        for j in range(200):
            if i == 0 and j in [0, 1, 2, 197, 198, 199]:
                # First row, specific values for demo
                values = {0: 0.12, 1: -1.50, 2: 2.00, 197: 7.77, 198: 8.88, 199: 9.99}
                row.append(values[j])
            elif i == 1 and j in [0, 1, 2, 197, 198, 199]:
                # Second row, specific values
                values = {0: 3.14, 1: 0.00, 2: float('nan'), 197: -2.10, 198: 1.25, 199: 2.35}
                row.append(values[j])
            elif i == 499 and j in [0, 1, 2, 197, 198, 199]:
                # Last row, specific values
                values = {0: -0.01, 1: 4.50, 2: -2.30, 197: 9.99, 198: 8.00, 199: 7.11}
                row.append(values[j])
            else:
                # Random-ish values
                row.append((i * 200 + j) % 100 / 10.0)
        large_data.append(row)
    
    large_matrix_data = {
        'data': large_data,
        'shape': (500, 200),
        'dtype': 'f32'
    }
    
    formatted_large = format_matrix(large_matrix_data)
    print(formatted_large)
    print()

def demo_array_display():
    """Demo GraphArray rich display."""
    print("=" * 60)
    print("GraphArray Display Demo")
    print("=" * 60)
    
    array_data = {
        'data': [0.125, 3.1416, float('nan'), -2.75, 8.0, 34],
        'dtype': 'f32',
        'shape': (6,),
        'name': 'col1'
    }
    
    formatted = format_array(array_data)
    print(formatted)
    print()

def main():
    """Run all display demos."""
    print("Groggy Rich Display Module Demo")
    print("===============================")
    print()
    
    demo_table_display()
    demo_matrix_display() 
    demo_array_display()
    
    print("Demo completed! 🎉")

if __name__ == '__main__':
    main()
