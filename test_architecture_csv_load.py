#!/usr/bin/env python3
"""Test loading the generated architecture CSV files"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy')

def test_load_architecture_csvs():
    """Test loading the architecture CSV files"""
    try:
        import groggy as gr
        
        print("=== Testing Architecture CSV Loading ===")
        
        # Test loading nodes table
        print("\n1. Loading NodesTable from groggy_architecture_nodes.csv")
        try:
            nodes_table = gr.NodesTable.from_csv('groggy_architecture_nodes.csv')
            print(f"✅ SUCCESS: Loaded NodesTable with {nodes_table.nrows()} rows")
            print(f"   Columns: {list(nodes_table.column_names)}")
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
        
        # Test loading edges table  
        print("\n2. Loading EdgesTable from groggy_architecture_edges.csv")
        try:
            edges_table = gr.EdgesTable.from_csv('groggy_architecture_edges.csv')
            print(f"✅ SUCCESS: Loaded EdgesTable with {edges_table.nrows()} rows")
            print(f"   Columns: {list(edges_table.column_names)}")
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            
    except Exception as e:
        print(f"ERROR in test setup: {str(e)}")

if __name__ == "__main__":
    test_load_architecture_csvs()