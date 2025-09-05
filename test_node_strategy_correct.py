#!/usr/bin/env python3
"""
Test for node_strategy parameter functionality using the correct API.
"""

import sys
import groggy as gr

def test_extract_strategy():
    """Test that extract strategy keeps original nodes."""
    print("\n=== Testing Extract Strategy ===")
    
    # Create a graph with nodes and edges
    g = gr.Graph()
    g.add_node(name="Alice", age=25, salary=85000)
    g.add_node(name="Bob", age=30, salary=95000)
    g.add_node(name="Carol", age=28, salary=80000)
    
    # Add edges
    g.add_edge(0, 1, weight=0.9, type="collaboration")
    g.add_edge(1, 2, weight=0.7, type="mentoring")
    
    print(f"Original graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    original_node_count = len(g.nodes)
    
    # Create subgraph and collapse with extract strategy
    subgraph = g.nodes[[0, 1, 2]]
    print(f"Subgraph: {subgraph.node_count()} nodes, {subgraph.edge_count()} edges")
    
    try:
        meta_node = subgraph.collapse(
            node_aggs={
                "team_size": "count",
                "avg_age": ("mean", "age"),
                "total_salary": ("sum", "salary")
            },
            node_strategy="extract"  # Should keep original nodes
        )
        
        print(f"✓ Created meta-node with extract strategy: {meta_node}")
        print(f"  Meta-node ID: {meta_node.id}")
        print(f"  Graph now has: {len(g.nodes)} nodes, {len(g.edges)} edges")
        
        # With extract strategy, we should have original nodes + meta node
        expected_nodes = original_node_count + 1  # Original 3 + meta node
        if len(g.nodes) == expected_nodes:
            print("✅ Extract strategy working correctly - original nodes preserved")
            return True
        else:
            print(f"❌ Extract strategy failed - expected {expected_nodes} nodes, got {len(g.nodes)}")
            return False
            
    except Exception as e:
        print(f"❌ Extract strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_collapse_strategy():
    """Test that collapse strategy removes original nodes."""
    print("\n=== Testing Collapse Strategy ===")
    
    # Create a graph with nodes and edges
    g = gr.Graph()
    g.add_node(name="Dave", age=35, department="engineering")
    g.add_node(name="Eve", age=32, department="engineering")
    g.add_node(name="Frank", age=40, department="marketing")
    
    # Add edges
    g.add_edge(0, 1, weight=0.8, type="collaboration")
    g.add_edge(1, 2, weight=0.6, type="cross_dept")
    
    print(f"Original graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    
    # Create subgraph of first two nodes and collapse with collapse strategy
    subgraph = g.nodes[[0, 1]]
    print(f"Subgraph: {subgraph.node_count()} nodes, {subgraph.edge_count()} edges")
    
    try:
        meta_node = subgraph.collapse(
            node_aggs={
                "team_size": "count",
                "avg_age": ("mean", "age"),
                "department": ("first", "department")
            },
            node_strategy="collapse"  # Should remove original nodes
        )
        
        print(f"✓ Created meta-node with collapse strategy: {meta_node}")
        print(f"  Meta-node ID: {meta_node.id}")
        print(f"  Graph now has: {len(g.nodes)} nodes, {len(g.edges)} edges")
        
        # With collapse strategy, we should have: remaining nodes + meta node
        # Original: 3 nodes, collapsed 2 → should have 1 remaining + 1 meta = 2 total
        expected_nodes = 2  # 1 remaining + 1 meta node
        if len(g.nodes) == expected_nodes:
            print("✅ Collapse strategy working correctly - original nodes removed")
            return True
        else:
            print(f"❌ Collapse strategy failed - expected {expected_nodes} nodes, got {len(g.nodes)}")
            return False
            
    except Exception as e:
        print(f"❌ Collapse strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_default_behavior():
    """Test the default behavior (should be extract)."""
    print("\n=== Testing Default Behavior ===")
    
    # Create a graph
    g = gr.Graph()
    g.add_node(name="Grace", role="dev")
    g.add_node(name="Henry", role="dev")
    
    g.add_edge(0, 1, weight=0.9)
    
    print(f"Original graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    original_node_count = len(g.nodes)
    
    # Create subgraph and collapse without specifying node_strategy
    subgraph = g.nodes[[0, 1]]
    
    try:
        meta_node = subgraph.collapse(
            node_aggs={
                "team_size": "count",
                "role": ("first", "role")
            }
            # No node_strategy specified - should default to extract
        )
        
        print(f"✓ Created meta-node with default strategy: {meta_node}")
        print(f"  Graph now has: {len(g.nodes)} nodes, {len(g.edges)} edges")
        
        # Default should be extract (keep original nodes)
        expected_nodes = original_node_count + 1  # Original 2 + meta node
        if len(g.nodes) == expected_nodes:
            print("✅ Default behavior working correctly - behaves like extract")
            return True
        else:
            print(f"❌ Default behavior unexpected - expected {expected_nodes} nodes, got {len(g.nodes)}")
            return False
            
    except Exception as e:
        print(f"❌ Default behavior test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all node_strategy tests"""
    print("Node Strategy Parameter Test")
    print("=" * 50)
    
    tests = [
        test_extract_strategy,
        test_collapse_strategy,
        test_default_behavior,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"💥 {test.__name__} CRASHED: {e}")
            results.append(False)
    
    # Summary
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"NODE STRATEGY TEST RESULTS")
    print(f"{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Node strategy parameter working perfectly!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
