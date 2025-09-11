#!/usr/bin/env python3
"""
Comprehensive Pipeline Testing - Build on crazy_delegation_examples.py
to systematically test the repository and identify weak points
"""

import sys
import traceback
import time
from typing import Any, List, Dict, Tuple

def safe_call(func_name: str, func_callable, *args, **kwargs) -> Tuple[bool, Any, str]:
    """
    Safely call a function and return (success, result, error_info)
    """
    try:
        start_time = time.time()
        result = func_callable(*args, **kwargs)
        elapsed = time.time() - start_time
        return True, result, f"✓ {func_name} ({elapsed:.3f}s)"
    except Exception as e:
        error_info = f"✗ {func_name}: {type(e).__name__}: {str(e)}"
        return False, None, error_info

def test_basic_functionality():
    """Test basic graph operations that should definitely work"""
    print("🔧 BASIC FUNCTIONALITY TESTS")
    print("=" * 60)
    
    results = []
    
    # Test 1: Can we import groggy?
    success, groggy, info = safe_call("import groggy", lambda: __import__('groggy'))
    print(f"  {info}")
    results.append(success)
    if not success:
        return results
    gr = groggy
    
    # Test 2: Can we create a basic graph?
    success, g, info = safe_call("create empty graph", gr.Graph)
    print(f"  {info}")
    results.append(success)
    if not success:
        return results
    
    # Test 3: Can we add nodes and edges?
    success, _, info = safe_call("add node", g.add_node, 0, {"name": "Alice"})
    print(f"  {info}")
    results.append(success)
    
    success, _, info = safe_call("add node", g.add_node, 1, {"name": "Bob"})
    print(f"  {info}")
    results.append(success)
    
    success, _, info = safe_call("add edge", g.add_edge, 0, 1, {"weight": 1.0})
    print(f"  {info}")
    results.append(success)
    
    # Test 4: Basic graph queries
    success, nodes, info = safe_call("get nodes", g.nodes)
    print(f"  {info}")
    results.append(success)
    
    success, edges, info = safe_call("get edges", g.edges)
    print(f"  {info}")
    results.append(success)
    
    # Test 5: Can we create karate club graph?
    success, karate, info = safe_call("karate_club graph", gr.karate_club)
    print(f"  {info}")
    results.append(success)
    
    if success:
        success, node_count, info = safe_call("karate node count", len, karate)
        print(f"  {info} - {node_count} nodes")
        results.append(success)
    
    return results

def test_accessor_operations():
    """Test node/edge accessor operations"""
    print("\n🎯 ACCESSOR OPERATIONS TESTS")
    print("=" * 60)
    
    results = []
    try:
        import groggy as gr
        g = gr.karate_club()
        
        # Test nodes accessor
        success, nodes_accessor, info = safe_call("g.nodes accessor", lambda: g.nodes)
        print(f"  {info}")
        results.append(success)
        
        if success:
            # Test node accessor indexing
            success, node0, info = safe_call("nodes[0] access", lambda: nodes_accessor[0])
            print(f"  {info}")
            results.append(success)
            
            # Test node accessor iteration
            success, node_list, info = safe_call("list(nodes)", list, nodes_accessor)
            print(f"  {info} - {len(node_list) if node_list else 0} nodes")
            results.append(success)
            
        # Test edges accessor
        success, edges_accessor, info = safe_call("g.edges accessor", lambda: g.edges)
        print(f"  {info}")
        results.append(success)
        
        if success:
            # Test edge accessor indexing  
            success, edge0, info = safe_call("edges[0] access", lambda: edges_accessor[0])
            print(f"  {info}")
            results.append(success)
            
            # Test edge accessor iteration
            success, edge_list, info = safe_call("list(edges)", list, edges_accessor)
            print(f"  {info} - {len(edge_list) if edge_list else 0} edges")
            results.append(success)
            
    except Exception as e:
        print(f"  ✗ Failed to set up accessor tests: {e}")
        results.append(False)
    
    return results

def test_subgraph_operations():
    """Test subgraph creation and operations"""
    print("\n🌐 SUBGRAPH OPERATIONS TESTS")
    print("=" * 60)
    
    results = []
    try:
        import groggy as gr
        g = gr.karate_club()
        
        # Test BFS traversal
        success, bfs_subgraph, info = safe_call("BFS from node 0", g.bfs, 0)
        print(f"  {info}")
        results.append(success)
        
        if success:
            success, bfs_len, info = safe_call("BFS subgraph length", len, bfs_subgraph)
            print(f"  {info} - {bfs_len} nodes")
            results.append(success)
            
            # Test subgraph methods
            success, _, info = safe_call("BFS subgraph density", bfs_subgraph.density)
            print(f"  {info}")
            results.append(success)
            
            success, _, info = safe_call("BFS subgraph node_count", bfs_subgraph.node_count)
            print(f"  {info}")
            results.append(success)
            
            success, _, info = safe_call("BFS subgraph edge_count", bfs_subgraph.edge_count)
            print(f"  {info}")
            results.append(success)
        
        # Test DFS traversal
        success, dfs_subgraph, info = safe_call("DFS from node 0", g.dfs, 0)
        print(f"  {info}")
        results.append(success)
        
        # Test neighborhood
        success, neighborhood, info = safe_call("neighborhood of node 0", g.neighborhood, 0)
        print(f"  {info}")
        results.append(success)
        
        if success:
            success, neigh_len, info = safe_call("neighborhood length", len, neighborhood)
            print(f"  {info} - {neigh_len} nodes")
            results.append(success)
        
    except Exception as e:
        print(f"  ✗ Failed to set up subgraph tests: {e}")
        results.append(False)
    
    return results

def test_table_operations():
    """Test table creation and operations"""
    print("\n📊 TABLE OPERATIONS TESTS")
    print("=" * 60)
    
    results = []
    try:
        import groggy as gr
        g = gr.karate_club()
        
        # Test nodes table
        success, nodes_table, info = safe_call("g.nodes.table()", lambda: g.nodes.table())
        print(f"  {info}")
        results.append(success)
        
        if success:
            success, table_len, info = safe_call("nodes table length", len, nodes_table)
            print(f"  {info} - {table_len} rows")
            results.append(success)
            
            # Test table methods
            success, _, info = safe_call("table.head(5)", nodes_table.head, 5)
            print(f"  {info}")
            results.append(success)
            
            success, _, info = safe_call("table columns", nodes_table.columns)
            print(f"  {info}")
            results.append(success)
        
        # Test edges table
        success, edges_table, info = safe_call("g.edges.table()", lambda: g.edges.table())
        print(f"  {info}")
        results.append(success)
        
        if success:
            success, table_len, info = safe_call("edges table length", len, edges_table)
            print(f"  {info} - {table_len} rows")
            results.append(success)
        
    except Exception as e:
        print(f"  ✗ Failed to set up table tests: {e}")
        results.append(False)
    
    return results

def test_matrix_operations():
    """Test matrix creation and operations"""
    print("\n🔢 MATRIX OPERATIONS TESTS")
    print("=" * 60)
    
    results = []
    try:
        import groggy as gr
        g = gr.karate_club()
        
        # Test adjacency matrix creation
        success, adj_matrix, info = safe_call("adjacency matrix", g.adjacency_matrix)
        print(f"  {info}")
        results.append(success)
        
        if success:
            success, shape, info = safe_call("matrix shape", adj_matrix.shape)
            print(f"  {info} - shape: {shape}")
            results.append(success)
            
            success, _, info = safe_call("matrix is_square", adj_matrix.is_square)
            print(f"  {info}")
            results.append(success)
        
        # Test dense adjacency matrix
        success, dense_matrix, info = safe_call("dense adjacency matrix", g.dense_adjacency_matrix)
        print(f"  {info}")
        results.append(success)
        
        if success:
            success, dense_shape, info = safe_call("dense matrix shape", dense_matrix.shape)
            print(f"  {info} - shape: {dense_shape}")
            results.append(success)
        
    except Exception as e:
        print(f"  ✗ Failed to set up matrix tests: {e}")
        results.append(False)
    
    return results

def test_pipeline_chains():
    """Test method chaining pipelines that should work"""
    print("\n⛓️ PIPELINE CHAINING TESTS")
    print("=" * 60)
    
    results = []
    try:
        import groggy as gr
        g = gr.karate_club()
        
        # Chain 1: Simple subgraph to table
        success, result, info = safe_call(
            "BFS → table", 
            lambda: g.bfs(0).table()
        )
        print(f"  {info}")
        results.append(success)
        
        # Chain 2: Table operations
        success, result, info = safe_call(
            "nodes → table → head(3)",
            lambda: g.nodes.table().head(3)
        )
        print(f"  {info}")
        results.append(success)
        
        # Chain 3: Subgraph operations
        success, result, info = safe_call(
            "neighborhood → density",
            lambda: g.neighborhood(0).density()
        )
        print(f"  {info}")
        results.append(success)
        
        # Chain 4: Matrix operations
        success, result, info = safe_call(
            "adjacency → shape",
            lambda: g.adjacency_matrix().shape()
        )
        print(f"  {info}")
        results.append(success)
        
    except Exception as e:
        print(f"  ✗ Failed to set up pipeline tests: {e}")
        results.append(False)
    
    return results

def identify_weak_points():
    """Systematically test advanced features to find weak points"""
    print("\n🔍 WEAK POINT IDENTIFICATION")
    print("=" * 60)
    
    weak_points = []
    
    try:
        import groggy as gr
        g = gr.karate_club()
        
        print("Testing advanced features...")
        
        # Test connected components
        success, _, info = safe_call("connected_components", g.connected_components)
        print(f"  {info}")
        if not success:
            weak_points.append("Connected Components algorithm")
            
        # Test shortest path
        success, _, info = safe_call("shortest_path(0, 5)", g.shortest_path, 0, 5)
        print(f"  {info}")
        if not success:
            weak_points.append("Shortest Path algorithm")
            
        # Test pagerank
        success, _, info = safe_call("pagerank", g.pagerank)
        print(f"  {info}")
        if not success:
            weak_points.append("PageRank algorithm")
            
        # Test graph filtering
        success, _, info = safe_call("filter_nodes", g.filter_nodes, "degree > 2")
        print(f"  {info}")
        if not success:
            weak_points.append("Node filtering")
            
        # Test graph persistence
        success, _, info = safe_call("save_to_path", g.save_to_path, "/tmp/test_graph.json")
        print(f"  {info}")
        if not success:
            weak_points.append("Graph persistence (save)")
            
        # Test subgraph sampling
        bfs_result = g.bfs(0)
        success, _, info = safe_call("subgraph.sample(10)", bfs_result.sample, 10)
        print(f"  {info}")
        if not success:
            weak_points.append("Subgraph sampling")
            
        # Test matrix operations
        try:
            matrix = g.adjacency_matrix()
            success, _, info = safe_call("matrix.transpose()", matrix.transpose)
            print(f"  {info}")
            if not success:
                weak_points.append("Matrix transpose")
                
            success, _, info = safe_call("matrix eigenvalues", getattr, matrix, 'eigenvalues', None)
            if success and info:
                success, _, info = safe_call("matrix.eigenvalues()", matrix.eigenvalues)
            print(f"  {info}")
            if not success:
                weak_points.append("Matrix eigenvalues")
        except:
            weak_points.append("Matrix operations setup")
            
        # Test table aggregation
        try:
            table = g.nodes.table()
            success, _, info = safe_call("table.group_by", table.group_by, ["degree"])
            print(f"  {info}")
            if not success:
                weak_points.append("Table group_by operations")
        except:
            weak_points.append("Table operations setup")
            
    except Exception as e:
        print(f"  ✗ Failed to set up weak point identification: {e}")
        weak_points.append("Basic graph setup for advanced features")
    
    return weak_points

def main():
    """Run comprehensive pipeline testing"""
    print("🚀 COMPREHENSIVE PIPELINE TESTING")
    print("Building on crazy_delegation_examples.py to find weak points")
    print("=" * 80)
    
    all_results = []
    
    # Run test suites
    test_suites = [
        ("Basic Functionality", test_basic_functionality),
        ("Accessor Operations", test_accessor_operations),
        ("Subgraph Operations", test_subgraph_operations),
        ("Table Operations", test_table_operations),
        ("Matrix Operations", test_matrix_operations),
        ("Pipeline Chains", test_pipeline_chains),
    ]
    
    for suite_name, suite_func in test_suites:
        print(f"\n{'='*20} {suite_name.upper()} {'='*20}")
        suite_results = suite_func()
        all_results.extend(suite_results)
        
        passed = sum(suite_results)
        total = len(suite_results)
        print(f"\n📊 {suite_name}: {passed}/{total} tests passed")
    
    # Identify weak points
    weak_points = identify_weak_points()
    
    # Final summary
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS")
    print("="*80)
    
    total_passed = sum(all_results)
    total_tests = len(all_results)
    
    print(f"📈 Overall: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
    
    if weak_points:
        print(f"\n🔴 WEAK POINTS IDENTIFIED ({len(weak_points)}):")
        for i, weak_point in enumerate(weak_points, 1):
            print(f"  {i}. {weak_point}")
            
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  • Focus development on the {len(weak_points)} identified weak areas")
        print(f"  • These represent the biggest gaps in functionality")
        print(f"  • Fixing these will unlock more sophisticated pipeline chains")
    else:
        print(f"\n🟢 NO MAJOR WEAK POINTS FOUND!")
        print(f"  • Core functionality is solid")
        print(f"  • Ready for more advanced delegation chains")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"  • Build more sophisticated pipeline examples")
    print(f"  • Test cross-type conversions extensively")  
    print(f"  • Benchmark performance on realistic workloads")

if __name__ == "__main__":
    main()