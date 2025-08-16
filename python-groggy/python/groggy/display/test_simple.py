#!/usr/bin/env python3
"""Simple test of the table formatter."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from groggy.display import format_table

def simple_test():
    table_data = {
        'columns': ['name', 'age'],
        'dtypes': {'name': 'string', 'age': 'int64'},
        'data': [
            ['Alice', 25],
            ['Bob', 30],
        ],
        'shape': (2, 2),
        'nulls': {},
        'index_type': 'int64'
    }
    
    formatted = format_table(table_data)
    print("Raw string:")
    print(repr(formatted))
    print("\nFormatted output:")
    print(formatted)

if __name__ == '__main__':
    simple_test()
