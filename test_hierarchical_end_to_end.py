#!/usr/bin/env python3
"""
Comprehensive end-to-end test of hierarchical subgraph functionality.
"""

import groggy

def test_hierarchical_end_to_end():
    print("🧪 Testing hierarchical subgraph functionality end-to-end...")
    
    # Create a larger graph with various attributes
    g = groggy.Graph()
    
    # Create nodes for different departments
    engineering = [
        g.add_node(name="Alice", age=25, salary=75000, department="engineering"),
        g.add_node(name="Bob", age=30, salary=85000, department="engineering"),
        g.add_node(name="Charlie", age=35, salary=95000, department="engineering")
    ]
    
    marketing = [
        g.add_node(name="David", age=28, salary=65000, department="marketing"),
        g.add_node(name="Eve", age=32, salary=70000, department="marketing")
    ]
    
    print(f"Created {len(engineering)} engineering nodes: {engineering}")
    print(f"Created {len(marketing)} marketing nodes: {marketing}")
    
    # Create some edges within departments
    g.add_edge(engineering[0], engineering[1])
    g.add_edge(engineering[1], engineering[2])
    g.add_edge(marketing[0], marketing[1])
    
    print(f"Total nodes: {len(list(g.node_ids))}")
    print(f"Total edges: {len(list(g.edge_ids))}")
    
    # Test 1: Collapse engineering department with multiple aggregations
    print(f"\n📊 Test 1: Collapse engineering department...")
    eng_subgraph = g.nodes[engineering]
    
    try:
        eng_meta_node = eng_subgraph.collapse_to_node({
            "head_count": "count",
            "salary": "sum",      # Sum the salary attribute
            "age": "mean"         # Mean of the age attribute  
        })
        
        print(f"✅ Engineering meta-node created: {eng_meta_node}")
        
        # Verify all aggregations worked
        all_nodes = g.nodes[list(g.node_ids)]
        head_count = all_nodes.get_node_attribute(eng_meta_node, "head_count")
        total_salary = all_nodes.get_node_attribute(eng_meta_node, "salary") 
        avg_age = all_nodes.get_node_attribute(eng_meta_node, "age")
        contained_subgraph = all_nodes.get_node_attribute(eng_meta_node, "contained_subgraph")
        
        print(f"  Head count: {head_count} (expected: 3)")
        print(f"  Total salary: {total_salary} (expected: 255000)")  # 75k + 85k + 95k
        print(f"  Average age: {avg_age} (expected: 30.0)")  # (25 + 30 + 35) / 3
        print(f"  Contained subgraph: {contained_subgraph is not None}")
        
        # Verify expected values
        assert head_count == 3, f"Expected head_count=3, got {head_count}"
        assert total_salary == 255000, f"Expected total_salary=255000, got {total_salary}"
        assert abs(avg_age - 30.0) < 0.1, f"Expected avg_age≈30.0, got {avg_age}"
        assert contained_subgraph is not None, "Expected contained_subgraph to be set"
        
        print("✅ All engineering aggregations correct!")
        
    except Exception as e:
        print(f"❌ Engineering collapse failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Collapse marketing department with different aggregations
    print(f"\n📊 Test 2: Collapse marketing department...")
    mkt_subgraph = g.nodes[marketing]
    
    try:
        mkt_meta_node = mkt_subgraph.collapse_to_node({
            "team_size": "count",
            "salary": "mean"  # Average salary for the team
        })
        
        print(f"✅ Marketing meta-node created: {mkt_meta_node}")
        
        # Verify aggregations
        all_nodes = g.nodes[list(g.node_ids)]
        team_size = all_nodes.get_node_attribute(mkt_meta_node, "team_size")
        avg_salary = all_nodes.get_node_attribute(mkt_meta_node, "salary")
        contained_subgraph = all_nodes.get_node_attribute(mkt_meta_node, "contained_subgraph")
        
        print(f"  Team size: {team_size} (expected: 2)")
        print(f"  Average salary: {avg_salary} (expected: 67500)")  # (65k + 70k) / 2
        print(f"  Contained subgraph: {contained_subgraph is not None}")
        
        # Verify expected values  
        assert team_size == 2, f"Expected team_size=2, got {team_size}"
        assert abs(avg_salary - 67500.0) < 0.1, f"Expected avg_salary≈67500.0, got {avg_salary}"
        assert contained_subgraph is not None, "Expected contained_subgraph to be set"
        
        print("✅ All marketing aggregations correct!")
        
    except Exception as e:
        print(f"❌ Marketing collapse failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Verify graph structure after hierarchical operations
    print(f"\n🏗️  Test 3: Verify final graph structure...")
    
    final_nodes = list(g.node_ids)
    final_edges = list(g.edge_ids)
    
    print(f"Final nodes: {len(final_nodes)} (original nodes + 2 meta-nodes)")
    print(f"Final edges: {len(final_edges)}")
    
    # Check that meta-nodes are properly integrated
    all_nodes = g.nodes[final_nodes]
    meta_nodes = []
    for node_id in final_nodes:
        if all_nodes.get_node_attribute(node_id, "contained_subgraph") is not None:
            meta_nodes.append(node_id)
    
    print(f"Meta-nodes found: {meta_nodes}")
    assert len(meta_nodes) == 2, f"Expected 2 meta-nodes, found {len(meta_nodes)}"
    
    print("✅ Graph structure verification passed!")
    
    print(f"\n🎉 All hierarchical functionality tests passed!")
    return True

if __name__ == "__main__":
    success = test_hierarchical_end_to_end()
    if success:
        print("\n✅ END-TO-END TEST PASSED!")
    else:
        print("\n❌ END-TO-END TEST FAILED!")
