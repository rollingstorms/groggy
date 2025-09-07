#!/usr/bin/env python3
"""
Debug the data ordering and NaN issue
"""
import groggy

def debug_data_ordering():
    print("Creating test graph...")
    g = groggy.Graph()
    
    # Add nodes and edges with some missing attributes
    n1 = g.add_node(name='Alice', age=25)
    n2 = g.add_node(name='Bob')  # Missing age
    n3 = g.add_node(age=35)      # Missing name
    
    print(f"Added nodes: n1={n1}, n2={n2}, n3={n3}")
    
    # Check individual node attributes
    print("\n=== Individual Node Attributes ===")
    for node_id in [n1, n2, n3]:
        print(f"Node {node_id}:")
        try:
            attrs = g.get_node_attributes(node_id)
            print(f"  Attributes: {attrs}")
        except Exception as e:
            print(f"  Error getting attributes: {e}")
    
    e1 = g.add_edge(n1, n2, strength=5)        # Missing type
    e2 = g.add_edge(n2, n3, type='colleague')  # Missing strength
    
    print(f"\nAdded edges: e1={e1}, e2={e2}")
    
    # Check individual edge attributes
    print("\n=== Individual Edge Attributes ===")
    for edge_id in [e1, e2]:
        print(f"Edge {edge_id}:")
        try:
            attrs = g.get_edge_attributes(edge_id)
            print(f"  Attributes: {attrs}")
        except Exception as e:
            print(f"  Error getting attributes: {e}")
    
    # Check graph stats
    print(f"\n=== Graph Stats ===")
    print(f"Node count: {g.node_count()}")
    print(f"Edge count: {g.edge_count()}")
    
    # Check the table creation
    print(f"\n=== Table Debug ===")
    table = g.table()
    print(f"Table created successfully")
    print(f"Nodes table shape: {table.nodes.shape()}")
    print(f"Edges table shape: {table.edges.shape()}")
    
    # Show the problematic table again
    print(f"\n=== Detailed Table Output ===")
    print("Nodes:")
    print(table.nodes)
    print("\nEdges:")  
    print(table.edges)

if __name__ == '__main__':
    debug_data_ordering()
