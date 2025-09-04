#!/usr/bin/env python3
"""
Debug script to track exactly what happens to stored subgraphs during edge removal.
This will help us understand if the consistency fixes are being applied.
"""
import sys
import traceback

try:
    import groggy as gr
    print("✓ Groggy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import groggy: {e}")
    sys.exit(1)

def debug_stored_subgraphs():
    """Debug stored subgraph updates during edge removal"""
    print("\n=== Debugging Stored Subgraph Updates ===")
    
    g = gr.Graph()
    
    # Create a simple graph that will trigger the issue
    for i in range(8):
        g.add_node(name=f"Node_{i}")
    
    # Create edges that will make high-degree nodes
    edges_created = []
    for i in range(6):
        for j in range(i+1, min(i+4, 8)):
            g.add_edge(i, j, weight=1.0)
            edges_created.append((i, j))
    
    print(f"Created graph: {g.node_count()} nodes, {g.edge_count()} edges")
    print(f"Edges created: {edges_created}")
    
    # Find high-degree nodes
    high_degree = []
    for node_id in g.node_ids:
        degree = len(g.neighbors(node_id))
        print(f"Node {node_id}: degree = {degree}")
        if degree > 4:
            high_degree.append(node_id)
    
    if not high_degree:
        print("No high-degree nodes found, creating manually...")
        # Add more edges to create high-degree nodes
        g.add_edge(0, 7, weight=1.0)
        g.add_edge(1, 7, weight=1.0)
        g.add_edge(2, 7, weight=1.0)
        for node_id in g.node_ids:
            degree = len(g.neighbors(node_id))
            if degree > 4:
                high_degree.append(node_id)
    
    print(f"High-degree nodes: {high_degree}")
    
    if high_degree:
        # Create connected components from high-degree nodes
        high_degree_subgraph = g.nodes[high_degree]
        print(f"High-degree subgraph has {len(high_degree_subgraph.node_ids)} nodes")
        
        # Track what happens during collapse
        print("\n--- Before Meta-Node Creation ---")
        print(f"Graph: {g.node_count()} nodes, {g.edge_count()} edges")
        print(f"Edge IDs: {list(g.edge_ids)}")
        
        try:
            components = list(high_degree_subgraph.connected_components())
            print(f"Found {len(components)} connected components")
            
            meta_nodes_created = []
            for i, component in enumerate(components):
                if len(component.node_ids) >= 2:
                    print(f"\n--- Collapsing Component {i+1} ---")
                    print(f"Component nodes: {component.node_ids}")
                    print(f"Graph before collapse: {g.node_count()} nodes, {g.edge_count()} edges")
                    
                    meta_node = component.collapse(edge_strategy='keep_external')
                    meta_nodes_created.append(meta_node)
                    
                    print(f"Created meta-node: {meta_node}")
                    print(f"Graph after collapse: {g.node_count()} nodes, {g.edge_count()} edges")
                    print(f"Current edge IDs: {list(g.edge_ids)}")
            
            print(f"\n--- After All Meta-Node Creation ---")
            print(f"Graph: {g.node_count()} nodes, {g.edge_count()} edges")
            print(f"Meta-nodes created: {meta_nodes_created}")
            print(f"All meta-nodes: {list(g.nodes.meta.all().node_ids)}")
            
            # Now try to remove the meta-nodes - this is where the error occurs
            print(f"\n--- Attempting Meta-Node Removal ---")
            meta_node_ids = list(g.nodes.meta.all().node_ids)
            print(f"About to remove meta-nodes: {meta_node_ids}")
            
            try:
                # This should trigger the stale reference error if my fix isn't working
                g.remove_nodes(meta_node_ids)
                print("✓ All meta-nodes removed successfully - no stale reference errors!")
                return True
                
            except Exception as e:
                print(f"✗ Meta-node removal failed: {e}")
                print(f"Error type: {type(e)}")
                
                # Check if this is the stale edge reference error
                error_str = str(e)
                if "Edge" in error_str and "not found" in error_str and "remove edge" in error_str:
                    print("❌ This is the STALE EDGE REFERENCE ERROR!")
                    print("The stored subgraphs are not being properly updated.")
                else:
                    print("This is a different error.")
                
                traceback.print_exc()
                return False
            
        except Exception as e:
            print(f"✗ Error during meta-node operations: {e}")
            traceback.print_exc()
            return False
    else:
        print("No high-degree nodes found, cannot test")
        return False

def main():
    """Run the debug test"""
    print("Stored Subgraph Update Debugging")
    print("=" * 50)
    
    success = debug_stored_subgraphs()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Debug test completed successfully")
    else:
        print("❌ Debug test found issues")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
