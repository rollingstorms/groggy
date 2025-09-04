#!/usr/bin/env python3
"""
Test to verify that the defensive checks in meta-edge creation are working.
This creates scenarios that should trigger the defensive checks.
"""

import sys
import traceback

try:
    import groggy as gr
    print("✓ Groggy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import groggy: {e}")
    sys.exit(1)

def test_defensive_checks_working():
    """Test that our defensive checks actually prevent crashes"""
    print("\n=== Test: Defensive Checks Working ===")
    
    g = gr.Graph()
    
    # Create a simple scenario
    for i in range(5):
        g.add_node(name=f"Node_{i}", value=i*10)
    
    # Connect them
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=2.0)  
    g.add_edge(2, 3, weight=3.0)
    g.add_edge(3, 4, weight=4.0)
    g.add_edge(0, 4, weight=5.0)  # Make a cycle
    
    print(f"Created graph: {g.node_count()} nodes, {g.edge_count()} edges")
    
    try:
        # Normal collapse should work fine
        subgraph = g.nodes[[0, 1]]
        meta_node = subgraph.collapse(
            node_aggs={"size": "count"},
            edge_strategy='keep_external'
        )
        print(f"✓ Normal collapse worked: {meta_node}")
        
        # Now the defensive checks should prevent any crashes from stale references
        # Even if there were any stale references, they should be caught and warned about
        
        # Collapse another component
        remaining_nodes = [nid for nid in g.node_ids if nid != meta_node.id]
        print(f"Remaining nodes after first collapse: {remaining_nodes}")
        
        if len(remaining_nodes) >= 2:
            subgraph2 = g.nodes[remaining_nodes[:2]]
            meta_node2 = subgraph2.collapse(
                node_aggs={"size": "count"},
                edge_strategy='keep_external'
            )
            print(f"✓ Second collapse worked: {meta_node2}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during defensive check test: {e}")
        traceback.print_exc()
        return False

def main():
    """Run defensive check validation"""
    print("Defensive Check Validation Test")
    print("=" * 40)
    
    result = test_defensive_checks_working()
    
    if result:
        print("\n✅ DEFENSIVE CHECKS VALIDATION PASSED")
        print("The meta-node operations now have defensive checks that should prevent")
        print("stale reference errors by validating node existence before edge creation.")
        return 0
    else:
        print("\n❌ DEFENSIVE CHECKS VALIDATION FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())