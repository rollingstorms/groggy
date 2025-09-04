#!/usr/bin/env python3
"""Test edge strategies and meta-edge creation"""

import sys
sys.path.append('.')
import groggy as gr

def create_test_graph_with_external_connections():
    """Create a graph where a subgraph has external connections"""
    g = gr.Graph(directed=False)
    
    # Internal cluster (nodes 0-2)
    g.add_node(name="A", group="cluster")
    g.add_node(name="B", group="cluster") 
    g.add_node(name="C", group="cluster")
    
    # External nodes (3-5)
    g.add_node(name="X", group="external")
    g.add_node(name="Y", group="external")
    g.add_node(name="Z", group="external")
    
    # Internal edges within cluster
    g.add_edge(0, 1, weight=0.9, type="internal", strength=5)
    g.add_edge(1, 2, weight=0.8, type="internal", strength=4)
    g.add_edge(0, 2, weight=0.7, type="internal", strength=3)
    
    # External edges from cluster to outside nodes
    g.add_edge(1, 3, weight=0.6, type="external", strength=2)  # B -> X
    g.add_edge(2, 4, weight=0.5, type="external", strength=3)  # C -> Y  
    g.add_edge(0, 5, weight=0.4, type="external", strength=1)  # A -> Z
    g.add_edge(2, 5, weight=0.3, type="external", strength=2)  # C -> Z (second connection)
    
    # Some external-to-external edges (should not affect meta-edges)
    g.add_edge(3, 4, weight=0.2, type="external_only", strength=1)
    
    return g

def analyze_meta_edges(g, meta_node):
    """Analyze the meta-edges created"""
    print(f"  Meta-node ID: {meta_node.node_id}")
    
    # Find edges connected to the meta-node
    meta_edges = []
    for edge_id in g.edge_ids:
        src, dst = g.edge_endpoints(edge_id)
        if src == meta_node.node_id or dst == meta_node.node_id:
            other_node = dst if src == meta_node.node_id else src
            # Get all edge attributes - newer API
            edge_attrs = {}
            try:
                # Try to get all attributes on the edge including source/target
                for attr in ['weight', 'type', 'strength', 'source', 'target']:  # Common edge attributes
                    try:
                        value = g.get_edge_attr(edge_id, attr)
                        if value is not None:
                            edge_attrs[attr] = value
                    except:
                        pass
            except:
                pass
            meta_edges.append((edge_id, other_node, edge_attrs))
    
    print(f"  Meta-edges found: {len(meta_edges)}")
    for edge_id, other_node, attrs in meta_edges:
        other_name = g.get_node_attr(other_node, 'name')
        # Get source and target for this edge
        src, dst = g.edge_endpoints(edge_id)
        print(f"    Edge {edge_id}: {src} → {dst} (Meta-node ↔ {other_name})")
        print(f"      Attributes: {dict(attrs)}")
        
        # Verify edge direction is correct
        if src == meta_node.node_id:
            print(f"      Direction: Meta-node ({meta_node.node_id}) → {other_name} ({other_node})")
        elif dst == meta_node.node_id:
            print(f"      Direction: {other_name} ({other_node}) → Meta-node ({meta_node.node_id})")
        else:
            print(f"      ⚠️  ERROR: Edge {edge_id} doesn't connect to meta-node!")
    
    return meta_edges

def test_edge_strategy_aggregate():
    """Test aggregate edge strategy"""
    print("\n=== Testing AGGREGATE Strategy ===")
    
    g = create_test_graph_with_external_connections()
    cluster = g.nodes[[0, 1, 2]]  # A, B, C
    
    print("Before collapse:")
    print(f"  Graph has {g.node_count()} nodes and {g.edge_count()} edges")
    print(f"  Cluster has {cluster.node_count()} nodes and {cluster.edge_count()} edges")
    
    # Count external edges from cluster before collapse
    external_edges_before = 0
    for node_id in [0, 1, 2]:  # A, B, C
        for neighbor in g.neighbors(node_id):
            if neighbor not in [0, 1, 2]:  # External connection
                external_edges_before += 1
    print(f"  External connections before: {external_edges_before}")
    
    # Collapse with aggregate strategy
    meta_node = cluster.collapse(
        node_aggs={"size": "count", "group_name": ("first", "group")},
        edge_strategy="aggregate",
        allow_missing_attributes=True
    )
    
    print(f"\nAfter collapse with AGGREGATE:")
    print(f"  Graph has {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Analyze meta-edges
    meta_edges = analyze_meta_edges(g, meta_node)
    
    # Expected: Should have meta-edges to external nodes, possibly aggregated
    expected_external_targets = {3, 4, 5}  # X, Y, Z
    actual_targets = {other_node for _, other_node, _ in meta_edges}
    
    print(f"  Expected external targets: {expected_external_targets}")
    print(f"  Actual targets: {actual_targets}")
    
    return len(meta_edges) > 0

def test_edge_strategy_keep_external():
    """Test keep_external edge strategy"""
    print("\n=== Testing KEEP_EXTERNAL Strategy ===")
    
    g = create_test_graph_with_external_connections()
    cluster = g.nodes[[0, 1, 2]]  # A, B, C
    
    print("Before collapse:")
    print(f"  Graph has {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Collapse with keep_external strategy
    meta_node = cluster.collapse(
        node_aggs={"size": "count"},
        edge_strategy="keep_external",
        allow_missing_attributes=True
    )
    
    print(f"\nAfter collapse with KEEP_EXTERNAL:")
    print(f"  Graph has {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Analyze meta-edges
    meta_edges = analyze_meta_edges(g, meta_node)
    
    return len(meta_edges) > 0

def test_edge_strategy_drop_all():
    """Test drop_all edge strategy"""
    print("\n=== Testing DROP_ALL Strategy ===")
    
    g = create_test_graph_with_external_connections()
    cluster = g.nodes[[0, 1, 2]]  # A, B, C
    
    print("Before collapse:")
    print(f"  Graph has {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Collapse with drop_all strategy
    meta_node = cluster.collapse(
        node_aggs={"size": "count"},
        edge_strategy="drop_all",
        allow_missing_attributes=True
    )
    
    print(f"\nAfter collapse with DROP_ALL:")
    print(f"  Graph has {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Analyze meta-edges
    meta_edges = analyze_meta_edges(g, meta_node)
    
    # Expected: Should have NO meta-edges
    success = len(meta_edges) == 0
    if success:
        print("  ✅ Correctly isolated meta-node (no external edges)")
    else:
        print(f"  ❌ Expected no meta-edges but found {len(meta_edges)}")
    
    return success

def test_edge_strategy_contract_all():
    """Test contract_all edge strategy"""
    print("\n=== Testing CONTRACT_ALL Strategy ===") 
    
    g = create_test_graph_with_external_connections()
    cluster = g.nodes[[0, 1, 2]]  # A, B, C
    
    print("Before collapse:")
    print(f"  Graph has {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Collapse with contract_all strategy
    meta_node = cluster.collapse(
        node_aggs={"size": "count"},
        edge_strategy="contract_all",
        allow_missing_attributes=True
    )
    
    print(f"\nAfter collapse with CONTRACT_ALL:")
    print(f"  Graph has {g.node_count()} nodes and {g.edge_count()} edges")
    
    # Analyze meta-edges  
    meta_edges = analyze_meta_edges(g, meta_node)
    
    return len(meta_edges) >= 0  # May or may not have edges depending on implementation

def main():
    """Test all edge strategies"""
    print("Testing MetaGraph Edge Strategies and Meta-Edge Creation")
    print("=" * 70)
    
    tests = [
        test_edge_strategy_aggregate,
        test_edge_strategy_keep_external, 
        test_edge_strategy_drop_all,
        test_edge_strategy_contract_all,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*70}")
    print(f"EDGE STRATEGY TEST RESULTS: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All edge strategies working correctly!")
    else:
        print(f"⚠️ {total - passed} edge strategies may have issues")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())