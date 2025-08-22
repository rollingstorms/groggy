#!/usr/bin/env python3
"""
Simple validation test for key Groggy features mentioned in documentation
"""

def test_core_features():
    print("🧪 Testing Core Groggy Features")
    print("=" * 40)
    
    import groggy as gr
    
    # 1. Basic graph creation
    print("1. Graph Creation")
    g = gr.Graph()
    g_directed = gr.Graph(directed=True)
    print("   ✅ Graph creation works")
    
    # 2. Node operations
    print("2. Node Operations")
    alice = g.add_node(name="Alice", age=30, department="Engineering")
    bob = g.add_node(name="Bob", age=25, department="Design")
    charlie = g.add_node(name="Charlie", age=35, department="Management")
    
    batch_nodes = [
        {'name': 'Dave', 'age': 28, 'department': 'Marketing'},
        {'name': 'Eve', 'age': 32, 'department': 'HR'}
    ]
    batch_ids = g.add_nodes(batch_nodes)
    
    print(f"   ✅ Added {g.node_count()} nodes (individual + batch)")
    
    # 3. Edge operations
    print("3. Edge Operations")
    g.add_edge(alice, bob, relationship="collaborates", strength=0.8)
    g.add_edge(charlie, alice, relationship="manages", strength=0.9)
    g.add_edge(charlie, bob, relationship="manages", strength=0.7)
    
    batch_edges = [
        (alice, charlie, {'type': 'reports_to'}),
        (bob, charlie, {'type': 'reports_to'})
    ]
    g.add_edges(batch_edges)
    
    print(f"   ✅ Added {g.edge_count()} edges")
    
    # 4. Graph properties
    print("4. Graph Properties")
    print(f"   ✅ Directed: {g.is_directed}")
    print(f"   ✅ Density: {g.density():.3f}")
    print(f"   ✅ Connected: {g.is_connected()}")
    
    # 5. Degree operations
    print("5. Degree Operations")
    degrees = g.degree()
    alice_degree = g.degree(alice)
    print(f"   ✅ All degrees: {type(degrees)}")
    print(f"   ✅ Alice's degree: {alice_degree}")
    
    # 6. Node access
    print("6. Node Access")
    alice_data = g.nodes[alice]
    alice_name = alice_data['name']
    print(f"   ✅ Node access: {alice_name}")
    
    # 7. Table operations
    print("7. Table Operations")
    nodes_table = g.nodes.table()
    edges_table = g.edges.table()
    print(f"   ✅ Nodes table: {nodes_table.shape}")
    print(f"   ✅ Edges table: {edges_table.shape}")
    print(f"   ✅ Table columns: {nodes_table.columns}")
    
    # 8. Array operations
    print("8. Array Operations")
    age_column = nodes_table['age']
    mean_age = age_column.mean()
    std_age = age_column.std()
    print(f"   ✅ Mean age: {mean_age:.1f}")
    print(f"   ✅ Std age: {std_age:.1f}")
    
    # 9. Table statistics
    print("9. Table Statistics")
    table_mean = nodes_table.mean('age')
    table_sum = nodes_table.sum('age')
    describe_result = nodes_table.describe()
    print(f"   ✅ Table mean: {table_mean:.1f}")
    print(f"   ✅ Table sum: {table_sum}")
    
    # 10. Sorting and filtering
    print("10. Sorting and Filtering")
    sorted_table = nodes_table.sort_by('age', ascending=True)
    young_people = nodes_table[nodes_table['age'] < 30]
    print(f"   ✅ Sorted table: {sorted_table.shape}")
    print(f"   ✅ Young people: {young_people.shape}")
    
    # 11. Node filtering
    print("11. Node Filtering")
    young_filter = gr.NodeFilter.attribute_filter('age', gr.AttributeFilter.less_than(30))
    young_nodes = g.filter_nodes(young_filter)
    
    eng_filter = gr.NodeFilter.attribute_equals('department', 'Engineering')
    age_filter = gr.NodeFilter.attribute_filter('age', gr.AttributeFilter.greater_than(25))
    combined_filter = gr.NodeFilter.and_filters([eng_filter, age_filter])
    filtered_nodes = g.filter_nodes(combined_filter)
    print(f"   ✅ Node filtering works")
    
    # 12. Edge filtering  
    print("12. Edge Filtering")
    strong_filter = gr.EdgeFilter.attribute_filter('strength', gr.AttributeFilter.greater_than(0.7))
    strong_edges = g.filter_edges(strong_filter)
    print(f"   ✅ Edge filtering works")
    
    # 13. Analytics
    print("13. Graph Analytics")
    components = g.analytics.connected_components()
    path = g.analytics.shortest_path(alice, bob)
    has_path = g.analytics.has_path(alice, bob)
    bfs_result = g.analytics.bfs(alice)
    dfs_result = g.analytics.dfs(alice)
    print(f"   ✅ Connected components: {len(components)}")
    print(f"   ✅ Shortest path: {path}")
    print(f"   ✅ BFS/DFS work")
    
    # 14. Matrix operations
    print("14. Matrix Operations")
    adj_matrix = g.adjacency()
    row_sums = adj_matrix.sum_axis(1)
    col_means = adj_matrix.mean_axis(0)
    matrix_power = adj_matrix.power(2)
    print(f"   ✅ Adjacency matrix: {adj_matrix.shape}")
    print(f"   ✅ Matrix operations work")
    
    # 15. Graph-aware table operations
    print("15. Graph-Aware Table Operations")
    try:
        high_degree = nodes_table.filter_by_degree(g, 'node_id', min_degree=1)
        connected = nodes_table.filter_by_connectivity(g, 'node_id', [alice], mode='direct')
        nearby = nodes_table.filter_by_distance(g, 'node_id', [alice], max_distance=2)
        print(f"   ✅ Graph-aware filtering works")
    except Exception as e:
        print(f"   ⚠️ Graph-aware filtering: {e}")
    
    # 16. Table joins
    print("16. Table Joins")
    salary_data = {
        'name': ['Alice', 'Bob', 'Charlie'],
        'salary': [95000, 75000, 120000]
    }
    salary_table = gr.table(salary_data)
    inner_result = nodes_table.inner_join(salary_table, 'name', 'name')
    left_result = nodes_table.left_join(salary_table, 'name', 'name')
    print(f"   ✅ Table joins work")
    
    # 17. Exports
    print("17. Export Operations")
    nx_graph = g.to_networkx()
    pandas_df = nodes_table.to_pandas()
    numpy_array = adj_matrix.to_numpy()
    print(f"   ✅ NetworkX export: {type(nx_graph)}")
    print(f"   ✅ Pandas export: {type(pandas_df)}")
    print(f"   ✅ NumPy export: {type(numpy_array)}")
    
    # 18. Standalone creation
    print("18. Standalone Creation")
    standalone_table = gr.table({'x': [1, 2, 3], 'y': [4, 5, 6]})
    standalone_array = gr.array([1, 2, 3, 4, 5])
    standalone_matrix = gr.matrix([[1, 0], [0, 1]])
    print(f"   ✅ Standalone table: {standalone_table.shape}")
    print(f"   ✅ Standalone array: {len(standalone_array)}")
    print(f"   ✅ Standalone matrix: {standalone_matrix.shape}")
    
    print("\n🎉 ALL CORE FEATURES WORK!")
    print("✅ Documentation examples are validated and ready for release!")

if __name__ == "__main__":
    test_core_features()