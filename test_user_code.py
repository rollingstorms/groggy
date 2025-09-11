#!/usr/bin/env python3
"""Test the exact code the user provided"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy')

import groggy as gr

try:
    print("Testing user's exact code...")
    nodes_table = gr.NodesTable.from_csv('groggy_architecture_nodes.csv')
    print(f"✅ SUCCESS: Loaded NodesTable with {nodes_table.nrows()} rows")
    
    edges_table = gr.EdgesTable.from_csv('groggy_architecture_edges.csv')  
    print(f"✅ SUCCESS: Loaded EdgesTable with {edges_table.nrows()} rows")
    
    print("\nNodes table columns:", list(nodes_table.column_names))
    print("Edges table columns:", list(edges_table.column_names))
    
except Exception as e:
    print(f"❌ ERROR: {str(e)}")