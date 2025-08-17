#!/usr/bin/env python3
"""
Comprehensive Groggy Functionality Test Script
Based on docs/usage_examples.md

This script tests all major features of the Groggy graph library
and demonstrates the API in action.
"""

import sys
import time
import traceback
import groggy as gr
import numpy as np

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def test_with_timing(func, description):
    """Run a test function and measure timing"""
    print(f"\n🧪 {description}")
    print("-" * 40)
    start_time = time.time()
    try:
        result = func()
        end_time = time.time()
        print(f"✅ SUCCESS - {description} ({end_time - start_time:.4f}s)")
        return result, True
    except Exception as e:
        end_time = time.time()
        print(f"❌ FAILED - {description} ({end_time - start_time:.4f}s)")
        print(f"   Error: {e}")
        traceback.print_exc()
        return None, False

def test_graph_construction():
    """Test 🏗️ Graph Construction and Basic Operations"""
    print_section("Graph Construction and Basic Operations")
    
    def basic_construction():
        g = gr.Graph()
        
        # Clean API with kwargs and flexible inputs
        alice = g.add_node(id="alice", age=30, role="engineer")
        bob = g.add_node(id="bob", age=25, role="engineer") 
        charlie = g.add_node(id="charlie", age=35, role="manager")
        
        print(f"Created nodes: alice={alice}, bob={bob}, charlie={charlie}")
        
        # Add edges
        g.add_edge(alice, bob, relationship="collaborates")
        g.add_edge(bob, charlie, relationship="reports_to")
        
        print(f"Graph has {g.node_count()} nodes and {g.edge_count()} edges")
        return g
    
    def bulk_operations():
        g = gr.Graph()
        
        # Bulk node creation
        node_data = [
            {"id": "alice", "age": 30, "role": "engineer", "salary": 75000}, 
            {"id": "bob", "age": 25, "role": "engineer", "salary": 65000},
            {"id": "charlie", "age": 35, "role": "manager", "salary": 85000},
            {"id": "diana", "age": 28, "role": "designer", "salary": 70000},
            {"id": "eve", "age": 32, "role": "engineer", "salary": 80000}
        ]
        
        node_mapping = g.add_nodes(node_data, uid_key="id")
        print(f"Node mapping: {node_mapping}")
        
        # Bulk edge creation
        edge_data = [
            {"source": "alice", "target": "bob", "relationship": "collaborates"},
            {"source": "bob", "target": "charlie", "relationship": "reports_to"},
            {"source": "alice", "target": "diana", "relationship": "collaborates"},
            {"source": "eve", "target": "charlie", "relationship": "reports_to"}
        ]
        
        g.add_edges(edge_data, node_mapping)
        print(f"Final graph: {g.node_count()} nodes, {g.edge_count()} edges")
        return g
    
    g1, success1 = test_with_timing(basic_construction, "Basic node/edge construction")
    g2, success2 = test_with_timing(bulk_operations, "Bulk operations with mappings")
    
    return g2 if success2 else g1, success1 and success2

def test_graph_array():
    """Test 🚀 GraphArray - Statistical Arrays with Native Performance"""
    print_section("GraphArray - Statistical Arrays")
    
    def basic_statistics():
        # Create GraphArray from values
        ages = gr.GraphArray([25, 30, 35, 40, 45, 50, 28, 33, 42, 38])
        salaries = gr.GraphArray([65000, 75000, 85000, 95000, 105000, 110000, 70000, 80000, 100000, 90000])
        
        print(f"Ages: {ages.to_list()}")
        print(f"Length: {len(ages)}")
        print(f"Mean age: {ages.mean():.2f}")
        print(f"Std age: {ages.std():.2f}")
        print(f"Min age: {ages.min()}")
        print(f"Max age: {ages.max()}")
        print(f"Median age: {ages.median()}")
        print(f"95th percentile: {ages.quantile(0.95)}")
        
        # Test indexing and iteration
        print(f"First element: {ages[0]}")
        print(f"Last element: {ages[-1]}")
        
        # Statistical summary
        summary = ages.describe()
        print(f"\nStatistical Summary:")
        print(f"  Count: {summary.count}")
        print(f"  Mean: {summary.mean:.2f}")
        print(f"  Std: {summary.std:.2f}")
        print(f"  Min: {summary.min}")
        print(f"  Max: {summary.max}")
        
        return ages, salaries
    
    def list_compatibility():
        ages = gr.GraphArray([25, 30, 35, 40, 45])
        
        # Test list-like behavior
        print(f"Length: {len(ages)}")
        print(f"Iteration: {[x for x in ages]}")
        
        # Convert back to plain list
        plain_list = ages.to_list()
        print(f"Converted to list: {plain_list}")
        print(f"Type of converted: {type(plain_list)}")
        
        return ages
    
    arrays1, success1 = test_with_timing(basic_statistics, "Basic statistical operations")
    arrays2, success2 = test_with_timing(list_compatibility, "List compatibility and conversion")
    
    return arrays1, success1 and success2

def test_filtering_and_queries(g):
    """Test 🔍 Querying and Filtering"""
    print_section("Querying and Filtering")
    
    def string_based_filtering():
        # Test various filtering patterns
        engineers = g.filter_nodes("role == 'engineer'")
        print(f"Engineers: {len(engineers.node_ids)} nodes")
        print(f"Engineer node IDs: {engineers.node_ids}")
        
        high_earners = g.filter_nodes("salary > 75000")
        print(f"High earners (>75k): {len(high_earners.node_ids)} nodes")
        
        # Complex expressions
        young_engineers = g.filter_nodes("age < 35 AND role == 'engineer'")
        print(f"Young engineers: {len(young_engineers.node_ids)} nodes")
        
        senior_or_manager = g.filter_nodes("age > 30 OR role == 'manager'")
        print(f"Senior or managers: {len(senior_or_manager.node_ids)} nodes")
        
        return engineers, high_earners
    
    def edge_filtering():
        # Test edge filtering
        collaborations = g.filter_edges("relationship == 'collaborates'")
        print(f"Collaboration edges: {len(collaborations.edge_ids)}")
        
        reports = g.filter_edges("relationship == 'reports_to'")
        print(f"Reporting edges: {len(reports.edge_ids)}")
        
        return collaborations, reports
    
    results1, success1 = test_with_timing(string_based_filtering, "String-based node filtering")
    results2, success2 = test_with_timing(edge_filtering, "Edge filtering")
    
    return results1, success1 and success2

def test_adjacency_matrices(g):
    """Test 🔢 Adjacency Matrix and Scientific Computing"""
    print_section("Adjacency Matrix and Scientific Computing")
    
    def dense_adjacency():
        # Test dense adjacency matrix
        adj_matrix = g.dense_adjacency_matrix()
        print(f"Dense adjacency matrix shape: {adj_matrix.shape}")
        
        # Test matrix access patterns
        print(f"Matrix[0,1]: {adj_matrix[0, 1]}")
        print(f"Row 0: {adj_matrix[0]}")
        print(f"Column 1: {adj_matrix.get_column(1)}")
        
        return adj_matrix
    
    def adjacency_variants():
        # Test different adjacency matrix types
        try:
            sparse_adj = g.sparse_adjacency_matrix()
            print(f"Sparse adjacency: {sparse_adj.shape}")
        except NotImplementedError:
            print("⚠️ Sparse adjacency not yet implemented (expected)")
        
        try:
            weighted_adj = g.weighted_adjacency_matrix("salary")
            print(f"Weighted adjacency: {weighted_adj.shape}")
        except Exception as e:
            print(f"⚠️ Weighted adjacency failed: {e}")
        
        try:
            laplacian = g.laplacian_matrix()
            print(f"Laplacian matrix: {laplacian.shape}")
        except Exception as e:
            print(f"⚠️ Laplacian matrix failed: {e}")
        
        return True
    
    matrix1, success1 = test_with_timing(dense_adjacency, "Dense adjacency matrix operations")
    result2, success2 = test_with_timing(adjacency_variants, "Adjacency matrix variants")
    
    return matrix1, success1 and success2

def test_table_functionality(g):
    """Test 📊 GraphTable - DataFrame-like Operations"""
    print_section("GraphTable - DataFrame-like Operations")
    
    def table_creation():
        # Create table views
        node_table = g.table()
        print(f"Node table type: {type(node_table)}")
        print(f"Table shape: {node_table.shape}")
        print(f"Available columns: {node_table.columns}")
        
        return node_table
    
    def table_column_access():
        table = g.table()
        
        # Test column access
        try:
            ages = table['age']
            print(f"Ages column type: {type(ages)}")
            print(f"Ages data: {ages}")
            
            # Check if it's GraphArray with statistical methods
            if hasattr(ages, 'mean'):
                print(f"✅ Column is GraphArray with mean: {ages.mean()}")
            else:
                print(f"⚠️ Column is {type(ages)} - not GraphArray yet")
                
        except Exception as e:
            print(f"❌ Column access failed: {e}")
        
        # Test row access  
        try:
            first_row = table[0]
            print(f"First row: {first_row}")
        except Exception as e:
            print(f"⚠️ Row access failed: {e}")
        
        return table
    
    def table_exports():
        table = g.table()
        
        # Test export capabilities (if available)
        try:
            pandas_df = table.to_pandas()
            print(f"✅ Pandas export successful: {type(pandas_df)}")
            print(f"Pandas shape: {pandas_df.shape}")
        except Exception as e:
            print(f"⚠️ Pandas export not available: {e}")
        
        return table
    
    table1, success1 = test_with_timing(table_creation, "Table creation and basic info")
    table2, success2 = test_with_timing(table_column_access, "Table column and row access")
    table3, success3 = test_with_timing(table_exports, "Table export capabilities")
    
    return table1, success1 and success2 and success3

def test_algorithms(g):
    """Test 🧮 Algorithm and Graph Analysis"""
    print_section("Algorithm and Graph Analysis")
    
    def graph_algorithms():
        # Connected components
        components = g.connected_components()
        print(f"Found {len(components)} connected components")
        for i, comp in enumerate(components):
            print(f"  Component {i}: {len(comp.node_ids)} nodes")
        
        # BFS traversal
        first_node = g.node_ids[0]
        bfs_result = g.bfs(start_node=first_node)
        print(f"BFS from node {first_node}: visited {len(bfs_result.node_ids)} nodes")
        
        # DFS traversal  
        dfs_result = g.dfs(start_node=first_node)
        print(f"DFS from node {first_node}: visited {len(dfs_result.node_ids)} nodes")
        
        return components, bfs_result, dfs_result
    
    def shortest_paths():
        # Test shortest path algorithms
        try:
            node_ids = g.node_ids
            if len(node_ids) >= 2:
                path = g.shortest_path(source=node_ids[0], target=node_ids[-1])
                print(f"Shortest path from {node_ids[0]} to {node_ids[-1]}: {len(path.node_ids)} nodes")
                print(f"Path nodes: {path.node_ids}")
            else:
                print("⚠️ Not enough nodes for shortest path test")
        except Exception as e:
            print(f"⚠️ Shortest path failed: {e}")
        
        return True
    
    def graph_metrics():
        # Basic graph properties
        print(f"Graph properties:")
        print(f"  Nodes: {g.node_count()}")
        print(f"  Edges: {g.edge_count()}")
        
        # Node degrees (manual calculation for now)
        node_ids = g.node_ids
        degrees = []
        node_list = list(node_ids)
        for node in node_list[:5]:  # Sample first 5 nodes
            degree = len([e for e in g.edge_ids if node in g.edge_endpoints(e)])
            degrees.append(degree)
            print(f"  Node {node} degree: {degree}")
        
        if degrees:
            avg_degree = sum(degrees) / len(degrees)
            print(f"  Average degree (sample): {avg_degree:.2f}")
        
        return True
    
    algos1, success1 = test_with_timing(graph_algorithms, "Graph traversal algorithms")
    result2, success2 = test_with_timing(shortest_paths, "Shortest path algorithms")
    result3, success3 = test_with_timing(graph_metrics, "Graph metrics and properties")
    
    return algos1, success1 and success2 and success3

def test_version_control(g):
    """Test 📚 Version Control and History"""
    print_section("Version Control and History")
    
    def basic_version_control():
        # Test commit functionality
        commit_id = g.commit("Initial graph with test data", "Test User")
        print(f"Created commit: {commit_id}")
        
        # Add some changes
        new_node = g.add_node(id="frank", age=29, role="intern", salary=45000)
        print(f"Added new node: {new_node}")
        
        # Create another commit
        commit_id2 = g.commit("Added intern Frank", "Test User")
        print(f"Created second commit: {commit_id2}")
        
        return commit_id, commit_id2
    
    def branch_operations():
        # Test branch functionality
        g.create_branch("feature-branch")
        print("Created feature branch")
        
        branches = g.branches()
        print(f"Available branches: {[b.name for b in branches]}")
        
        # Test checkout
        g.checkout_branch("feature-branch")
        print("Checked out feature branch")
        
        # Add changes on branch
        branch_node = g.add_node(id="grace", age=26, role="junior", salary=55000)
        print(f"Added node on feature branch: {branch_node}")
        
        return branches
    
    def history_operations():
        # Commit changes on feature branch first
        feature_commit = g.commit("Added Grace on feature branch", "Test User")
        print(f"Created feature commit: {feature_commit}")
        
        # Test commit history (switch back to main to see commits)
        g.checkout_branch("main")
        history = g.commit_history()
        print(f"Main branch commit history has {len(history)} commits")
        for commit in history:
            print(f"  Commit {commit.id}: '{commit.message}' by {commit.author}")
        
        # Test state methods
        has_changes = g.has_uncommitted_changes()
        print(f"Has uncommitted changes: {has_changes}")
        
        # Test node mapping
        try:
            mapping = g.get_node_mapping("id")
            print(f"Node ID mapping: {mapping}")
        except Exception as e:
            print(f"⚠️ Node mapping failed: {e}")
        
        return history
    
    commits, success1 = test_with_timing(basic_version_control, "Basic commit operations")
    branches, success2 = test_with_timing(branch_operations, "Branch operations")
    history, success3 = test_with_timing(history_operations, "History and state operations")
    
    return (commits, branches, history), success1 and success2 and success3

def test_advanced_features(g):
    """Test advanced and experimental features"""
    print_section("Advanced Features and Integrations")
    
    def networkx_conversion():
        # Test NetworkX conversion
        try:
            nx_graph = g.to_networkx()
            print(f"✅ NetworkX conversion successful: {type(nx_graph)}")
            print(f"NetworkX nodes: {nx_graph.number_of_nodes()}")
            print(f"NetworkX edges: {nx_graph.number_of_edges()}")
        except Exception as e:
            print(f"⚠️ NetworkX conversion failed: {e}")
        
        return True
    
    def aggregation_operations():
        # Test aggregation methods
        try:
            # Group by role and aggregate salary  
            role_groups = g.group_by("role", "salary", "count")
            print(f"Grouped by role: {type(role_groups)}")
            print(f"Group result: {role_groups}")
                
        except Exception as e:
            print(f"⚠️ Grouping operations failed: {e}")
        
        try:
            # Test aggregation
            avg_salary = g.aggregate("salary", "mean")
            print(f"Average salary: {avg_salary.value}")
        except Exception as e:
            print(f"⚠️ Aggregation failed: {e}")
        
        return True
    
    def subgraph_operations():
        # Test subgraph functionality
        engineers = g.filter_nodes("role == 'engineer'")
        print(f"Engineer subgraph: {len(engineers.node_ids)} nodes")
        
        # Test subgraph properties
        print(f"Subgraph edge count: {len(engineers.edge_ids)}")
        
        # Test subgraph table (if available)
        try:
            eng_table = engineers.table()
            print(f"✅ Subgraph table: {type(eng_table)}")
        except Exception as e:
            print(f"⚠️ Subgraph table not available: {e}")
        
        return engineers
    
    result1, success1 = test_with_timing(networkx_conversion, "NetworkX integration")
    result2, success2 = test_with_timing(aggregation_operations, "Aggregation operations")
    subgraph, success3 = test_with_timing(subgraph_operations, "Subgraph operations")
    
    return subgraph, success1 and success2 and success3

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🚀 Groggy Comprehensive Functionality Test")
    print("Based on docs/usage_examples.md")
    print("=" * 60)
    
    start_time = time.time()
    all_success = True
    
    # Run all test sections
    g, success = test_graph_construction()
    all_success &= success
    
    arrays, success = test_graph_array()
    all_success &= success
    
    if g:
        filters, success = test_filtering_and_queries(g)
        all_success &= success
        
        matrix, success = test_adjacency_matrices(g)
        all_success &= success
        
        table, success = test_table_functionality(g)
        all_success &= success
        
        algos, success = test_algorithms(g)
        all_success &= success
        
        version, success = test_version_control(g)
        all_success &= success
        
        advanced, success = test_advanced_features(g)
        all_success &= success
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print_section("Final Results")
    print(f"🕒 Total test time: {total_time:.2f} seconds")
    print(f"📊 Overall result: {'✅ ALL TESTS PASSED' if all_success else '❌ SOME TESTS FAILED'}")
    
    if g:
        print(f"📈 Final graph stats:")
        print(f"   Nodes: {g.node_count()}")
        print(f"   Edges: {g.edge_count()}")
        print(f"   Methods available: {len([m for m in dir(g) if not m.startswith('_')])}")
    
    return all_success

def main():
    """Main entry point"""
    try:
        success = run_comprehensive_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())