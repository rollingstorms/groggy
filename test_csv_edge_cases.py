#!/usr/bin/env python3
"""
Test edge cases for CSV import/export to reproduce the user's reported issue
"""

import sys
import os

# Add the python-groggy module to the path
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy')

def test_bad_column_names():
    """Test importing CSV files with wrong column names"""
    try:
        import groggy
        
        print("=== Testing NodesTable with 'node_ids' instead of 'node_id' ===")
        try:
            bad_nodes = groggy.NodesTable.from_csv("/tmp/test_bad_nodes.csv")
            print("SUCCESS: Imported nodes with 'node_ids' column")
        except Exception as e:
            print("ERROR: Failed to import nodes with 'node_ids' column:", str(e))
        
        print("\n=== Testing EdgesTable with wrong column names ===")
        try:
            bad_edges = groggy.EdgesTable.from_csv("/tmp/test_bad_edges.csv")
            print("SUCCESS: Imported edges with wrong columns")
        except Exception as e:
            print("ERROR: Failed to import edges with wrong columns:", str(e))
            
    except Exception as e:
        print("ERROR in test setup:", str(e))

if __name__ == "__main__":
    test_bad_column_names()