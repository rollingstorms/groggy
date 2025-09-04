#!/usr/bin/env python3
"""
Debug the real issue with the user's exact scenario.
Since the user is still getting the error, let's create a minimal reproduction.
"""

import sys
import traceback

try:
    import groggy as gr
    print("✓ Groggy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import groggy: {e}")
    sys.exit(1)

def create_problematic_scenario():
    """Create a scenario that might trigger the real issue"""
    print("Creating potentially problematic scenario...")
    
    g = gr.Graph()
    
    # Create many nodes to get to higher IDs like 102
    for i in range(120):
        g.add_node(name=f"Node_{i}", value=i, category=i % 10)
    
    # Create lots of edges to ensure high degrees
    import random
    random.seed(42)  # Reproducible
    
    edges_created = 0
    for i in range(120):
        # Each node connects to ~8-12 others to ensure degree > 4
        targets = random.sample([j for j in range(120) if j != i], min(10, 120-1))
        for target in targets:
            try:
                g.add_edge(i, target, weight=random.random())
                edges_created += 1
            except:
                pass  # Edge might already exist
    
    print(f"Created graph: {g.node_count()} nodes, {edges_created} edges")
    
    # Create some initial meta-nodes to make the state more complex
    print("Creating initial meta-nodes to complicate graph state...")
    
    # Create several rounds of meta-nodes
    for round_num in range(3):
        print(f"  Meta-node round {round_num}")
        available_base_nodes = []
        for nid in g.node_ids:
            try:
                entity_type = g.get_node_attr(nid, "entity_type")
                if entity_type != "meta":  # Only use non-meta nodes
                    available_base_nodes.append(nid)
            except:
                available_base_nodes.append(nid)  # Assume it's a base node
        
        print(f"    Available base nodes: {len(available_base_nodes)}")
        
        if len(available_base_nodes) >= 6:
            # Take first 3 for meta-node
            selected = available_base_nodes[:3]
            print(f"    Creating meta-node from nodes: {selected}")
            
            try:
                subgraph = g.nodes[selected]
                meta_node = subgraph.collapse(
                    node_aggs={"size": "count", "avg_value": ("mean", "value")},
                    edge_strategy='keep_external'
                )
                print(f"    ✓ Created meta-node: {meta_node}")
            except Exception as e:
                print(f"    ✗ Failed to create meta-node: {e}")
                # Don't break, continue to next round
        else:
            print(f"    Not enough base nodes available: {len(available_base_nodes)}")
            break
    
    print(f"Final graph state: {g.node_count()} nodes, {g.edge_count()} edges")
    return g

def test_problematic_user_code():
    """Test the user's exact code that's failing"""
    print("\n=== Testing User's Exact Code ===")
    
    g = create_problematic_scenario()
    
    try:
        print("Getting high-degree nodes...")
        degrees = g.degree()
        high_degree_count = sum(1 for d in degrees if d > 4)
        print(f"Nodes with degree > 4: {high_degree_count}")
        
        high_degree_nodes = g.nodes[g.degree() > 4]
        print(f"High-degree nodes selected: {len(high_degree_nodes)}")
        
        print("Getting connected components...")
        components = high_degree_nodes.connected_components()
        print(f"Found {len(components)} components")
        
        print("About to execute user's problematic code...")
        for i, component in enumerate(components):
            print(f"\nProcessing component {i}...")
            print(f"  Component: {component.node_count()} nodes, {component.edge_count()} edges")
            
            # Show some node IDs to see if we have high IDs like 102
            comp_nodes = list(component.node_ids)
            high_id_nodes = [nid for nid in comp_nodes if nid > 100]
            print(f"  Nodes with ID > 100: {high_id_nodes}")
            
            try:
                # This is the exact line that's failing for the user
                result = component.collapse(edge_strategy='keep_external')
                print(f"  ✓ SUCCESS: {result}")
                
            except RuntimeError as e:
                error_msg = str(e)
                print(f"  ✗ CAUGHT THE ERROR: {error_msg}")
                
                # If this is our target error, let's analyze it
                if "not found while attempting to add edge" in error_msg:
                    print(f"  🎯 THIS IS THE ERROR WE'RE LOOKING FOR!")
                    
                    # Extract the problematic node ID
                    import re
                    match = re.search(r'Node (\d+)', error_msg)
                    if match:
                        problematic_node = int(match.group(1))
                        print(f"  Problematic node ID: {problematic_node}")
                        
                        # Check if this node actually exists
                        current_nodes = list(g.node_ids)
                        print(f"  Node {problematic_node} exists in graph: {problematic_node in current_nodes}")
                        print(f"  Node {problematic_node} in component: {problematic_node in comp_nodes}")
                        
                        # If it exists, try to understand why the error occurred
                        if problematic_node in current_nodes:
                            print(f"  🔍 Node exists but add_edge failed - this suggests space/pool inconsistency")
                            try:
                                node_obj = g.nodes[problematic_node]
                                print(f"    Node object: {node_obj}")
                                print(f"    Node type: {type(node_obj).__name__}")
                            except Exception as node_err:
                                print(f"    Error accessing node: {node_err}")
                        else:
                            print(f"  🔍 Node doesn't exist - component has stale reference")
                    
                    return False  # We reproduced the error!
                else:
                    print(f"  Different error: {error_msg}")
                    traceback.print_exc()
                    return False
            
            except Exception as e:
                print(f"  ✗ Unexpected error: {e}")
                traceback.print_exc()
                return False
        
        print(f"\n✓ All {len(components)} components processed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error in test: {e}")
        traceback.print_exc()
        return False

def main():
    """Run the real issue debugging"""
    print("Real Issue Debugging - User's Exact Scenario")
    print("=" * 60)
    
    result = test_problematic_user_code()
    
    if not result:
        print("\n🎯 SUCCESSFULLY REPRODUCED THE USER'S ERROR!")
        print("Now we can analyze and fix the real issue.")
    else:
        print("\n❓ Could not reproduce the error - may need different conditions")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())