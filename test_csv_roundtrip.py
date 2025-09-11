#!/usr/bin/env python3
"""
Test script to reproduce the CSV import/export issue with NodesTables and EdgesTables
"""

import sys
import os

# Add the python-groggy module to the path
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy')

def test_nodes_table_csv_roundtrip():
    """Test exporting and importing a NodesTable to/from CSV"""
    try:
        # Import the module
        import groggy
        
        # Create a social network graph using generator
        g = groggy.generators.social_network(n=10)
        
        # Get nodes table
        nodes_table = g.nodes.table()
        print("Original nodes table columns:", nodes_table.column_names)
        
        # Export to CSV
        csv_path = "/tmp/test_nodes.csv"
        nodes_table.to_csv(csv_path)
        print("Exported to CSV:", csv_path)
        
        # Read the CSV file to see what was exported
        with open(csv_path, 'r') as f:
            content = f.read()
            print("CSV content:")
            print(content)
        
        # Try to import it back
        try:
            imported_table = groggy.NodesTable.from_csv(csv_path)
            print("SUCCESS: Successfully imported nodes table from CSV")
            print("Imported table columns:", imported_table.column_names)
        except Exception as e:
            print("ERROR: Failed to import nodes table from CSV:", str(e))
            return False
            
    except Exception as e:
        print("ERROR in test setup:", str(e))
        return False
    
    return True

def test_edges_table_csv_roundtrip():
    """Test exporting and importing an EdgesTable to/from CSV"""
    try:
        # Import the module
        import groggy
        
        # Create a social network graph using generator with more connections
        g = groggy.generators.social_network(n=10)
        
        # Add a few manual edges to ensure we have edge data
        node_ids = [0, 1, 2]  # Use simple node IDs
        if len(list(g.nodes)) >= 3:
            g.add_edge(node_ids[0], node_ids[1], relation="friend")
            g.add_edge(node_ids[1], node_ids[2], relation="colleague")
        
        # Get edges table
        edges_table = g.edges.table()
        print("Original edges table columns:", edges_table.column_names)
        
        # Export to CSV
        csv_path = "/tmp/test_edges.csv"
        edges_table.to_csv(csv_path)
        print("Exported to CSV:", csv_path)
        
        # Read the CSV file to see what was exported
        with open(csv_path, 'r') as f:
            content = f.read()
            print("CSV content:")
            print(content)
        
        # Try to import it back
        try:
            imported_table = groggy.EdgesTable.from_csv(csv_path)
            print("SUCCESS: Successfully imported edges table from CSV")
            print("Imported table columns:", imported_table.column_names)
        except Exception as e:
            print("ERROR: Failed to import edges table from CSV:", str(e))
            return False
            
    except Exception as e:
        print("ERROR in test setup:", str(e))
        return False
    
    return True

if __name__ == "__main__":
    print("=== Testing NodesTable CSV roundtrip ===")
    nodes_success = test_nodes_table_csv_roundtrip()
    
    print("\n=== Testing EdgesTable CSV roundtrip ===")
    edges_success = test_edges_table_csv_roundtrip()
    
    print(f"\n=== Results ===")
    print(f"Nodes table test: {'PASS' if nodes_success else 'FAIL'}")
    print(f"Edges table test: {'PASS' if edges_success else 'FAIL'}")