#!/usr/bin/env python3
"""
Test the exact user scenario that's causing stale references.
User reported: "for component in g.nodes[g.degree() > 4].connected_components(): component.collapse(edge_strategy='keep_external')"
Error: "RuntimeError: Failed to create meta-node: Node 103 not found while attempting to add edge"
"""

import sys
import traceback

try:
    import groggy as gr
    print("✓ Groggy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import groggy: {e}")
    sys.exit(1)

def create_graph_with_prior_meta_nodes():
    """Create a graph that already has meta-nodes, which might cause the stale reference issue"""
    print("Creating graph with existing meta-nodes...")
    
    g = gr.Graph()
    
    # Create a larger graph to ensure we get nodes with ID > 100
    for i in range(150):
        g.add_node(name=f"Node_{i}", value=i, category=i % 5)
    
    # Create dense connections to ensure high degrees
    import random
    random.seed(42)  # Reproducible
    
    # Add edges to create high-degree nodes
    for i in range(150):
        # Each node connects to ~6-8 others
        for j in range(6):
            target = random.randint(0, 149)
            if target != i:
                # Only add if edge doesn't exist
                try:
                    g.add_edge(i, target, weight=random.random())
                except:
                    pass  # Edge might already exist
    
    print(f"Initial graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    # Create some meta-nodes first (this might be the key to the issue)
    print("Creating initial meta-nodes...")
    
    # Create a few meta-nodes from random selections
    for meta_round in range(3):
        # Pick some random nodes to collapse
        available_nodes = [nid for nid in g.node_ids if g.get_node_attr(nid, "entity_type") != "meta"][:10]
        if len(available_nodes) >= 3:
            selected = available_nodes[:3]
            try:
                subgraph = g.nodes[selected]
                meta_node = subgraph.collapse(
                    node_aggs={"size": "count", "avg_value": ("mean", "value")},
                    edge_strategy='keep_external'
                )
                print(f"  Created initial meta-node {meta_round}: {meta_node}")
            except Exception as e:
                print(f"  Failed to create initial meta-node {meta_round}: {e}")
    
    print(f"Graph after initial meta-nodes: {g.node_count()} nodes, {g.edge_count()} edges")
    return g

def test_exact_user_code():
    """Test the exact code pattern the user used"""
    print("\n=== Test: Exact User Code Pattern ===")
    
    g = create_graph_with_prior_meta_nodes()
    
    try:
        print("Executing user code: for component in g.nodes[g.degree() > 4].connected_components(): component.collapse(edge_strategy='keep_external')")
        
        # Show degrees before filtering
        degrees = g.degree()
        high_degree_count = sum(1 for d in degrees if d > 4)
        print(f"Nodes with degree > 4: {high_degree_count}")
        
        if high_degree_count == 0:
            print("No high-degree nodes found - adjusting threshold")
            # Find what the max degree is
            max_degree = max(degrees)
            threshold = max(1, max_degree // 2)
            print(f"Using degree > {threshold} instead")
        else:
            threshold = 4
        
        # Filter for high-degree nodes
        high_degree_nodes = g.nodes[g.degree() > threshold]
        print(f"High-degree nodes selected: {len(high_degree_nodes)}")
        
        # Get connected components
        components = high_degree_nodes.connected_components()
        print(f"Connected components: {len(components)}")
        
        # The problematic loop
        for i, component in enumerate(components):
            print(f"\nProcessing component {i}...")
            print(f"  Component size: {component.node_count()} nodes, {component.edge_count()} edges")
            
            # Show some diagnostic info
            comp_node_ids = list(component.node_ids)
            print(f"  Component nodes: {comp_node_ids[:5]}{'...' if len(comp_node_ids) > 5 else ''}")
            
            # Check if any nodes are already meta-nodes
            meta_nodes_in_comp = []
            for nid in comp_node_ids:
                try:
                    entity_type = g.get_node_attr(nid, "entity_type")
                    if entity_type == "meta":
                        meta_nodes_in_comp.append(nid)
                except:
                    pass
            
            if meta_nodes_in_comp:
                print(f"  WARNING: Component contains meta-nodes: {meta_nodes_in_comp}")
            
            try:
                # This is the line that's failing for the user
                meta_node = component.collapse(edge_strategy='keep_external')
                print(f"  ✓ Successfully collapsed component {i}: {meta_node}")
                
            except Exception as e:
                print(f"  ✗ STALE REFERENCE ERROR on component {i}: {e}")
                
                # Detailed diagnosis
                error_str = str(e)
                print(f"    Error message: {error_str}")
                
                # Check if it's a node reference error
                import re
                node_match = re.search(r'Node (\d+)', error_str)
                edge_match = re.search(r'Edge (\d+)', error_str)
                
                if node_match:
                    problematic_node = int(node_match.group(1))
                    print(f"    Problematic node: {problematic_node}")
                    
                    # Check if this node exists
                    current_nodes = list(g.node_ids)
                    print(f"    Node exists in graph: {problematic_node in current_nodes}")
                    print(f"    Node in component: {problematic_node in comp_node_ids}")
                    
                    # If it doesn't exist, where did it go?
                    if problematic_node not in current_nodes:
                        print(f"    🔍 FOUND THE ISSUE: Node {problematic_node} was referenced but doesn't exist!")
                        print(f"    This suggests the component contains stale node references")
                        
                        # Check what the highest node ID in the graph is
                        max_node_id = max(current_nodes) if current_nodes else -1
                        print(f"    Current max node ID: {max_node_id}")
                        print(f"    Missing node ID: {problematic_node}")
                        
                        if problematic_node > max_node_id:
                            print(f"    🚨 DIAGNOSIS: Component references future/non-existent node ID!")
                        else:
                            print(f"    🚨 DIAGNOSIS: Component references deleted node ID!")
                
                if edge_match:
                    problematic_edge = int(edge_match.group(1))
                    print(f"    Problematic edge: {problematic_edge}")
                    
                    current_edge_ids = [e.id for e in g.edges]
                    print(f"    Edge exists in graph: {problematic_edge in current_edge_ids}")
                    
                    if problematic_edge not in current_edge_ids:
                        print(f"    🔍 FOUND THE ISSUE: Edge {problematic_edge} was referenced but doesn't exist!")
                        max_edge_id = max(current_edge_ids) if current_edge_ids else -1
                        print(f"    Current max edge ID: {max_edge_id}")
                        print(f"    Missing edge ID: {problematic_edge}")
                
                # Full traceback for debugging
                traceback.print_exc()
                
                # Stop on first error to analyze
                return False
        
        print(f"\n✓ Successfully processed all {len(components)} components")
        return True
        
    except Exception as e:
        print(f"✗ Error in user code pattern test: {e}")
        traceback.print_exc()
        return False

def test_rapid_multiple_collapses():
    """Test rapid multiple collapses that might cause race conditions"""
    print("\n=== Test: Rapid Multiple Collapses ===")
    
    g = gr.Graph()
    
    # Create multiple disconnected components
    node_counter = 0
    for comp in range(5):
        # Each component has 4 nodes
        comp_nodes = []
        for i in range(4):
            g.add_node(name=f"C{comp}_N{i}", comp=comp, value=comp*10 + i)
            comp_nodes.append(node_counter)
            node_counter += 1
        
        # Connect them in a cycle to ensure high degree
        for i in range(4):
            g.add_edge(comp_nodes[i], comp_nodes[(i + 1) % 4], weight=1.0)
            # Add extra edges for higher degree
            g.add_edge(comp_nodes[i], comp_nodes[(i + 2) % 4], weight=2.0)
    
    print(f"Created multi-component graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    try:
        # Collapse all components rapidly
        components = g.connected_components()
        print(f"Found {len(components)} components")
        
        for i, comp in enumerate(components):
            print(f"Collapsing component {i} (nodes: {list(comp.node_ids)})")
            
            try:
                meta_node = comp.collapse(
                    node_aggs={"comp_id": ("first", "comp"), "size": "count"},
                    edge_strategy='keep_external'
                )
                print(f"  ✓ Created meta-node: {meta_node}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                # Check if this is our stale reference error
                if "not found" in str(e):
                    print(f"    🚨 STALE REFERENCE ERROR: {e}")
                    return False
                traceback.print_exc()
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Error in rapid collapse test: {e}")
        traceback.print_exc()
        return False

def main():
    """Run exact user scenario tests"""
    print("Exact User Scenario Debug Tests")
    print("=" * 50)
    
    tests = [
        test_exact_user_code,
        test_rapid_multiple_collapses,
    ]
    
    results = []
    for test in tests:
        try:
            print(f"\n{'='*15} {test.__name__} {'='*15}")
            result = test()
            results.append(result)
            if result:
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
                # Continue testing to see if we can reproduce the issue
        except Exception as e:
            print(f"💥 {test.__name__} CRASHED: {e}")
            traceback.print_exc()
            results.append(False)
    
    # Summary
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"EXACT USER SCENARIO TEST RESULTS")
    print(f"{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed < total:
        print(f"⚠️ {total - passed} test(s) revealed potential issues")
    else:
        print("✅ Could not reproduce the stale reference issue")
        print("This suggests it's a specific sequence/timing problem")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())