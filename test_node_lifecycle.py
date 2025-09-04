#!/usr/bin/env python3
"""
Test to understand exactly what happens to nodes and edges during meta-node operations.
This should help identify when stale references are created.
"""

import sys
import traceback

try:
    import groggy as gr
    print("✓ Groggy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import groggy: {e}")
    sys.exit(1)

def track_graph_state(g, label):
    """Helper to track graph state at a specific point"""
    print(f"\n--- {label} ---")
    node_ids = list(g.node_ids)
    edge_data = []
    
    for edge in g.edges:
        try:
            edge_data.append(f"Edge {edge.id}: {edge.source} -> {edge.target}")
        except Exception as e:
            edge_data.append(f"Edge {edge.id}: ERROR - {e}")
    
    print(f"Nodes: {node_ids}")
    print(f"Edges: {edge_data}")
    print(f"Node count: {g.node_count()}, Edge count: {g.edge_count()}")
    return node_ids, edge_data

def test_node_absorption_lifecycle():
    """Track what happens to nodes during absorption"""
    print("\n=== Test: Node Absorption Lifecycle ===")
    
    g = gr.Graph()
    
    # Create simple graph
    g.add_node(name="A", value=10)
    g.add_node(name="B", value=20) 
    g.add_node(name="C", value=30)
    g.add_node(name="D", value=40)
    
    g.add_edge(0, 1, weight=1.0)  # A-B
    g.add_edge(1, 2, weight=2.0)  # B-C
    g.add_edge(2, 3, weight=3.0)  # C-D
    g.add_edge(0, 2, weight=4.0)  # A-C (external edge when [A,B] collapsed)
    
    initial_nodes, initial_edges = track_graph_state(g, "Initial State")
    
    # Create meta-node from A and B
    print("\nCollapsing nodes [0, 1] (A, B)...")
    try:
        subgraph = g.nodes[[0, 1]]
        print("Subgraph created successfully")
        
        # Track state right before collapse
        pre_collapse_nodes, pre_collapse_edges = track_graph_state(g, "Pre-Collapse")
        
        # Collapse
        meta_node = subgraph.collapse(
            node_aggs={"size": "count", "total_value": ("sum", "value")},
            edge_strategy='keep_external'
        )
        print(f"Meta-node created: {meta_node}")
        
        # Track state after collapse
        post_collapse_nodes, post_collapse_edges = track_graph_state(g, "Post-Collapse")
        
        # Check what happened to original nodes
        print(f"\nNode lifecycle analysis:")
        print(f"  Original nodes [0, 1] still exist: {0 in post_collapse_nodes and 1 in post_collapse_nodes}")
        print(f"  Meta-node ID: {meta_node.id}")
        print(f"  Meta-node in graph: {meta_node.id in post_collapse_nodes}")
        
        # Try to access original nodes
        try:
            node_0 = g.nodes[0]
            print(f"  Node 0 accessible: {node_0}")
            print(f"  Node 0 type: {type(node_0).__name__}")
        except Exception as e:
            print(f"  Node 0 access error: {e}")
            
        try:
            node_1 = g.nodes[1] 
            print(f"  Node 1 accessible: {node_1}")
            print(f"  Node 1 type: {type(node_1).__name__}")
        except Exception as e:
            print(f"  Node 1 access error: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during collapse: {e}")
        traceback.print_exc()
        return False

def test_edge_reference_tracking():
    """Track edge references during operations"""
    print("\n=== Test: Edge Reference Tracking ===")
    
    g = gr.Graph()
    
    # Create a more complex scenario
    for i in range(6):
        g.add_node(name=f"Node_{i}", group=i//2)
    
    # Create edges including some that cross groups
    g.add_edge(0, 1, weight=0.5, type="internal")    # Group 0 internal
    g.add_edge(2, 3, weight=0.6, type="internal")    # Group 1 internal  
    g.add_edge(4, 5, weight=0.7, type="internal")    # Group 2 internal
    g.add_edge(1, 2, weight=1.0, type="cross")       # Cross group 0->1
    g.add_edge(3, 4, weight=1.1, type="cross")       # Cross group 1->2
    g.add_edge(0, 4, weight=1.2, type="long")        # Long distance 0->2
    
    track_graph_state(g, "Initial Complex State")
    
    try:
        # First collapse: group 0 (nodes 0,1)
        print("\nFirst collapse: Group 0 [0, 1]")
        subgraph1 = g.nodes[[0, 1]]
        meta1 = subgraph1.collapse(
            node_aggs={"group_size": "count"},
            edge_strategy='keep_external'
        )
        
        state1_nodes, state1_edges = track_graph_state(g, "After First Collapse")
        print(f"Meta-node 1: {meta1}")
        
        # Second collapse: group 1 (nodes 2,3)
        print("\nSecond collapse: Group 1 [2, 3]")
        # Find which nodes are still available
        available_nodes = list(g.node_ids)
        print(f"Available nodes for second collapse: {available_nodes}")
        
        # Make sure nodes 2,3 are still available
        if 2 in available_nodes and 3 in available_nodes:
            subgraph2 = g.nodes[[2, 3]]
            meta2 = subgraph2.collapse(
                node_aggs={"group_size": "count"},
                edge_strategy='keep_external'
            )
            
            state2_nodes, state2_edges = track_graph_state(g, "After Second Collapse")
            print(f"Meta-node 2: {meta2}")
            
            # Try third collapse: group 2 (nodes 4,5) 
            print("\nThird collapse: Group 2 [4, 5]")
            available_nodes = list(g.node_ids)
            print(f"Available nodes for third collapse: {available_nodes}")
            
            if 4 in available_nodes and 5 in available_nodes:
                subgraph3 = g.nodes[[4, 5]]
                meta3 = subgraph3.collapse(
                    node_aggs={"group_size": "count"},
                    edge_strategy='keep_external'
                )
                
                final_nodes, final_edges = track_graph_state(g, "Final State")
                print(f"Meta-node 3: {meta3}")
                
                return True
            else:
                print("  Nodes 4,5 not available - this suggests a problem")
                return False
        else:
            print("  Nodes 2,3 not available - this suggests a problem")
            return False
            
    except Exception as e:
        print(f"✗ Error during edge tracking test: {e}")
        traceback.print_exc()
        return False

def test_concurrent_component_collapse():
    """Test the specific scenario: collapsing multiple components"""
    print("\n=== Test: Concurrent Component Collapse ===")
    
    g = gr.Graph()
    
    # Create a graph with multiple high-degree components
    # Component 1: nodes 0-3
    for i in range(4):
        g.add_node(name=f"Comp1_Node_{i}", comp=1, value=i*10)
        
    # Make them highly connected (each node degree > 4)
    # Connect each to every other within component (degree 3 within component)
    for i in range(4):
        for j in range(i+1, 4):
            g.add_edge(i, j, weight=0.1 * (i+j))
    
    # Component 2: nodes 4-7  
    for i in range(4, 8):
        g.add_node(name=f"Comp2_Node_{i}", comp=2, value=i*10)
        
    # Make them highly connected (each node degree > 4)
    # Connect each to every other within component (degree 3 within component)
    for i in range(4, 8):
        for j in range(i+1, 8):
            g.add_edge(i, j, weight=0.1 * (i+j))
    
    # Add MORE cross-component edges to increase degrees
    g.add_edge(0, 5, weight=2.0, type="cross")
    g.add_edge(0, 6, weight=2.1, type="cross")  
    g.add_edge(1, 7, weight=2.2, type="cross")
    g.add_edge(2, 4, weight=2.3, type="cross")
    g.add_edge(3, 5, weight=2.4, type="cross")
    
    # Remove duplicate cross-component edges (already added above)
    # g.add_edge(1, 4, weight=5.0, type="cross_comp")
    # g.add_edge(2, 6, weight=6.0, type="cross_comp")
    
    track_graph_state(g, "Multi-Component Initial")
    
    try:
        # Get high-degree nodes
        degrees = g.degree()
        print(f"Degrees: {degrees}")
        
        high_degree_nodes = g.nodes[g.degree() > 4]
        print(f"High-degree nodes: {len(high_degree_nodes)}")
        
        components = high_degree_nodes.connected_components()
        print(f"Components found: {len(components)}")
        
        # Try to collapse each component sequentially
        meta_nodes = []
        for i, component in enumerate(components):
            print(f"\nCollapsing component {i}...")
            
            # Check what nodes/edges are in this component
            comp_nodes = component.node_ids
            comp_edges = component.edge_ids
            print(f"  Component {i} nodes: {list(comp_nodes)}")
            print(f"  Component {i} edges: {list(comp_edges)}")
            
            # Verify all nodes still exist
            current_graph_nodes = list(g.node_ids)
            print(f"  Current graph nodes: {current_graph_nodes}")
            
            nodes_exist = all(nid in current_graph_nodes for nid in comp_nodes)
            print(f"  All component nodes exist in graph: {nodes_exist}")
            
            if not nodes_exist:
                print(f"  ✗ STALE REFERENCE DETECTED: Component contains non-existent nodes")
                missing = [nid for nid in comp_nodes if nid not in current_graph_nodes]
                print(f"  Missing nodes: {missing}")
                return False
            
            try:
                meta_node = component.collapse(
                    node_aggs={"comp_size": "count", "total_value": ("sum", "value")},
                    edge_strategy='keep_external'
                )
                
                meta_nodes.append(meta_node)
                print(f"  ✓ Successfully created meta-node {i}: {meta_node}")
                track_graph_state(g, f"After Component {i} Collapse")
                
            except Exception as e:
                print(f"  ✗ Failed to collapse component {i}: {e}")
                
                # Let's examine the specific error
                error_str = str(e)
                if "not found" in error_str:
                    print(f"  This looks like the stale reference error!")
                    
                    # Try to identify what's stale
                    import re
                    node_match = re.search(r'Node (\d+)', error_str)
                    edge_match = re.search(r'Edge (\d+)', error_str)
                    
                    if node_match:
                        problematic_node = int(node_match.group(1))
                        print(f"  Problematic node: {problematic_node}")
                        print(f"  Node in component: {problematic_node in comp_nodes}")
                        print(f"  Node in graph: {problematic_node in current_graph_nodes}")
                        
                    if edge_match:
                        problematic_edge = int(edge_match.group(1))
                        print(f"  Problematic edge: {problematic_edge}")
                        print(f"  Edge in component: {problematic_edge in comp_edges}")
                        
                        # Try to see if edge still exists
                        current_edges = [e.id for e in g.edges]
                        print(f"  Edge in graph: {problematic_edge in current_edges}")
                
                traceback.print_exc()
                return False
        
        print(f"\n✓ Successfully collapsed {len(meta_nodes)} components")
        return True
        
    except Exception as e:
        print(f"✗ Error in concurrent component collapse: {e}")
        traceback.print_exc()
        return False

def main():
    """Run node lifecycle tests"""
    print("Meta-Node Lifecycle Debug Tests")
    print("=" * 50)
    
    tests = [
        test_node_absorption_lifecycle,
        test_edge_reference_tracking,
        test_concurrent_component_collapse,
    ]
    
    for test in tests:
        try:
            print(f"\n{'='*20} {test.__name__} {'='*20}")
            result = test()
            if result:
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED - stopping here")
                break
        except Exception as e:
            print(f"💥 {test.__name__} CRASHED: {e}")
            traceback.print_exc()
            break
    
    return 0

if __name__ == "__main__":
    sys.exit(main())