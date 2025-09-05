#!/usr/bin/env python3
"""
Minimal test for node_strategy parameter functionality.
"""

import groggy

def main():
    print("Creating graph...")
    g = groggy.Graph()
    
    # Add some nodes
    n1 = g.add_node()
    n2 = g.add_node()
    n3 = g.add_node()
    
    # Add some edges
    e1 = g.add_edge(n1, n2)
    e2 = g.add_edge(n2, n3)
    
    print(f"Original graph has {len(g.nodes)} nodes")
    print(f"Original graph has {len(g.edges)} edges")
    
    # Create subgraph
    print("Creating subgraph view...")
    subgraph = g.view()
    print(f"Subgraph type: {type(subgraph)}")
    
    # Test method exists and is callable
    if hasattr(subgraph, 'collapse'):
        collapse_method = getattr(subgraph, 'collapse')
        print(f"Collapse method type: {type(collapse_method)}")
        print(f"Collapse method is callable: {callable(collapse_method)}")
        
        try:
            print("Attempting collapse with extract strategy...")
            result = subgraph.collapse(node_strategy="extract")
            print(f"Success! Result type: {type(result)}")
            
        except Exception as e:
            print(f"Error during collapse: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Error: subgraph does not have a collapse method")

if __name__ == "__main__":
    main()
