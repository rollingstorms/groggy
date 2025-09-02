#!/usr/bin/env python3

import sys
sys.path.insert(0, './python')

import groggy

def debug_mysterious_subgraphs():
    print("🔍 Debugging mysterious PySubgraph::from_core_subgraph calls...")
    
    print("\n=== STEP 1: Creating graph ===")
    g = groggy.Graph()
    
    print("\n=== STEP 2: Adding nodes ===")
    nodes = g.add_nodes(50000)  # Match user's configuration
    
    print("\n=== STEP 3: Adding edges ===")
    edge_pairs = [(i, i+1) for i in range(49999)]
    edges = g.add_edges(edge_pairs)
    
    print("\n=== STEP 4: Just accessing graph object g ===")
    print("About to access 'g' (the graph object)...")
    print(g)  # This is what should trigger the 20+ calls
    
    print("\n=== STEP 5: Accessing it again ===")
    print("Accessing 'g' again...")
    print(g)
    
    print("\n=== STEP 6: Try accessing different properties ===")
    print(f"Node count: {g.node_count()}")
    print(f"Edge count: {g.edge_count()}")
    
    print("\n=== STEP 7: Try some operations that might trigger subgraph creation ===")
    print("Trying g.nodes()...")
    try:
        nodes_list = g.nodes()
        print(f"Got nodes list with {len(nodes_list)} nodes")
    except Exception as e:
        print(f"Error: {e}")
    
    print("Trying g.edges()...")
    try:
        edges_list = g.edges()
        print(f"Got edges list with {len(edges_list)} edges")
    except Exception as e:
        print(f"Error: {e}")
        
    print("Trying g.view()...")
    try:
        view = g.view()
        print(f"Got view: {type(view)}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== STEP 8: Try IPython-style representation ===")
    try:
        repr_str = repr(g)
        print(f"repr(g): {repr_str}")
    except Exception as e:
        print(f"Error in repr: {e}")
        
    try:
        if hasattr(g, '_repr_html_'):
            html = g._repr_html_()
            print(f"HTML repr length: {len(html) if html else 0}")
    except Exception as e:
        print(f"Error in HTML repr: {e}")

if __name__ == "__main__":
    debug_mysterious_subgraphs()