#!/usr/bin/env python3
"""
More specific debug test for the exact user scenario:
for component in g.nodes[g.degree() > 4].connected_components():
    component.collapse(edge_strategy='keep_external')
"""

import sys
import traceback

try:
    import groggy as gr
    print("✓ Groggy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import groggy: {e}")
    sys.exit(1)

def create_complex_graph_with_existing_meta_nodes():
    """Create a graph that might already have meta-nodes"""
    print("Creating complex graph...")
    g = gr.Graph()
    
    # Create initial nodes
    for i in range(15):
        g.add_node(name=f"Node_{i}", value=i * 5, category=i % 3)
    
    # Create dense connections
    for i in range(15):
        for j in range(i+1, 15):
            if abs(i - j) <= 3:  # Connect nearby nodes
                g.add_edge(i, j, weight=0.1 * (i + j))
    
    print(f"Initial graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Create some meta-nodes first to make graph state more complex
    print("Creating initial meta-nodes...")
    
    # First meta-node
    subgraph1 = g.nodes[[0, 1, 2]]
    meta1 = subgraph1.collapse(node_aggs={"group_size": "count", "avg_value": ("mean", "value")})
    print(f"Created meta-node 1: {meta1}")
    
    # Second meta-node
    remaining = [nid for nid in g.node_ids if nid != meta1.id]
    subgraph2 = g.nodes[remaining[5:8]]  # Take some middle nodes
    meta2 = subgraph2.collapse(node_aggs={"group_size": "count"})
    print(f"Created meta-node 2: {meta2}")
    
    print(f"Graph after meta-nodes: {g.node_count()} nodes, {g.edge_count()} edges")
    
    return g

def test_exact_user_scenario():
    """Test the exact scenario the user described"""
    print("\n=== Test: Exact User Scenario ===")
    
    g = create_complex_graph_with_existing_meta_nodes()
    
    try:
        print("Executing: g.nodes[g.degree() > 4].connected_components()")
        
        # Get degrees
        degrees = g.degree()
        print(f"Degree vector: {degrees}")
        
        # Filter high-degree nodes
        high_degree_mask = [d > 4 for d in degrees]
        print(f"High-degree mask: {high_degree_mask}")
        print(f"High-degree node count: {sum(high_degree_mask)}")
        
        high_degree_nodes = g.nodes[high_degree_mask]
        print(f"High-degree nodes: {len(high_degree_nodes)}")
        
        # Get connected components
        print("Getting connected components of high-degree nodes...")
        components = high_degree_nodes.connected_components()
        print(f"Found {len(components)} components")
        
        # Show component details
        for i, component in enumerate(components):
            print(f"Component {i}: {component.node_count()} nodes, {component.edge_count()} edges")
            print(f"  Node IDs in component: {component.node_ids}")
            print(f"  Edge IDs in component: {component.edge_ids}")
        
        print("\nExecuting: for component in components: component.collapse(edge_strategy='keep_external')")
        
        # Now try to collapse each component
        for i, component in enumerate(components):
            print(f"\nCollapsing component {i}...")
            print(f"  Pre-collapse - Component nodes: {component.node_ids}")
            print(f"  Pre-collapse - Component edges: {component.edge_ids}")
            print(f"  Pre-collapse - Graph nodes: {list(g.node_ids)}")
            print(f"  Pre-collapse - Graph edges: {[e.id for e in g.edges]}")
            
            try:
                meta_node = component.collapse(
                    node_aggs={"size": "count"},
                    edge_strategy='keep_external'
                )
                print(f"  ✓ Successfully created meta-node: {meta_node}")
                print(f"  Post-collapse - Graph nodes: {list(g.node_ids)}")
                print(f"  Post-collapse - Graph edges: {[e.id for e in g.edges]}")
                
            except Exception as e:
                print(f"  ✗ Error collapsing component {i}: {e}")
                print(f"  Error type: {type(e).__name__}")
                
                # Show detailed state
                print(f"  Current graph nodes: {list(g.node_ids)}")
                print(f"  Current graph edges: {[e.id for e in g.edges]}")
                
                # Try to identify the problematic node/edge
                error_str = str(e)
                if "Node" in error_str and "not found" in error_str:
                    # Extract node ID from error
                    import re
                    match = re.search(r'Node (\d+)', error_str)
                    if match:
                        problematic_node = int(match.group(1))
                        print(f"  Problematic node ID: {problematic_node}")
                        print(f"  Is node {problematic_node} in graph? {problematic_node in g.node_ids}")
                        
                elif "Edge" in error_str and "not found" in error_str:
                    # Extract edge ID from error
                    import re
                    match = re.search(r'Edge (\d+)', error_str)
                    if match:
                        problematic_edge = int(match.group(1))
                        print(f"  Problematic edge ID: {problematic_edge}")
                        current_edge_ids = [e.id for e in g.edges]
                        print(f"  Is edge {problematic_edge} in graph? {problematic_edge in current_edge_ids}")
                
                traceback.print_exc()
                return False
                
    except Exception as e:
        print(f"✗ Error in user scenario: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_manual_step_by_step():
    """Manually step through to identify where stale refs happen"""
    print("\n=== Test: Manual Step-by-Step ===")
    
    g = gr.Graph()
    
    # Simple case first
    for i in range(8):
        g.add_node(name=f"Simple_{i}")
    
    # Create edges to ensure high degrees
    edges_to_create = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),  # Node 0 has degree 5
        (1, 2), (1, 3), (1, 6), (1, 7),          # Node 1 has degree 5  
        (2, 3), (2, 4), (2, 7),                  # Node 2 has degree 5
        (3, 4), (3, 5),                          # Node 3 has degree 5
        (4, 5), (4, 6),                          # Node 4 has degree 5
        (5, 6), (5, 7),                          # Node 5 has degree 5
        (6, 7)                                   # Nodes 6,7 have degree 4,4
    ]
    
    for src, dst in edges_to_create:
        g.add_edge(src, dst, weight=1.0)
    
    print(f"Created simple graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    degrees = g.degree()
    print(f"Degrees: {degrees}")
    
    # Step 1: Get high-degree nodes
    high_degree_nodes = g.nodes[g.degree() > 4]
    print(f"High-degree nodes: {len(high_degree_nodes)} nodes")
    print(f"High-degree node IDs: {[n.id for n in high_degree_nodes]}")
    
    # Step 2: Get their connected components  
    components = high_degree_nodes.connected_components()
    print(f"Components: {len(components)}")
    
    for i, comp in enumerate(components):
        print(f"  Component {i}: nodes={comp.node_ids}, edges={comp.edge_ids}")
    
    # Step 3: Try collapsing first component
    if len(components) > 0:
        comp = components[0]
        print(f"\nTrying to collapse first component:")
        print(f"  Component node IDs: {comp.node_ids}")
        print(f"  Component edge IDs: {comp.edge_ids}")
        print(f"  Graph state before collapse:")
        print(f"    All nodes: {list(g.node_ids)}")
        print(f"    All edges: {[e.id for e in g.edges]}")
        
        try:
            meta_node = comp.collapse(node_aggs={"size": "count"}, edge_strategy='keep_external')
            print(f"  ✓ Success: {meta_node}")
            
            print(f"  Graph state after collapse:")
            print(f"    All nodes: {list(g.node_ids)}")  
            print(f"    All edges: {[e.id for e in g.edges]}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            traceback.print_exc()
            return False
    
    return True

def main():
    """Run specific debug tests"""
    print("Specific Stale Reference Debug Tests")
    print("=" * 50)
    
    tests = [
        test_exact_user_scenario,
        test_manual_step_by_step,
    ]
    
    for test in tests:
        try:
            print(f"\n{'='*20} {test.__name__} {'='*20}")
            result = test()
            if result:
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
                break  # Stop on first failure to analyze
        except Exception as e:
            print(f"💥 {test.__name__} CRASHED: {e}")
            traceback.print_exc()
            break
    
    return 0

if __name__ == "__main__":
    sys.exit(main())