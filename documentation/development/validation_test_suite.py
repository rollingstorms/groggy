#!/usr/bin/env python3
"""
Groggy Documentation Validation Test Suite

This script tests all examples from our updated documentation to verify
they actually work with the current implementation.

Results will be saved to validation_results.md
"""

import sys
import traceback
from datetime import datetime

# Global results tracking
test_results = []
current_section = ""

def log_test(test_name, success=True, error_msg="", code_snippet=""):
    """Log a test result"""
    global test_results, current_section
    test_results.append({
        'section': current_section,
        'test': test_name,
        'success': success,
        'error': error_msg,
        'code': code_snippet
    })
    
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {test_name}")
    if not success and error_msg:
        print(f"    Error: {error_msg}")

def set_section(section_name):
    """Set the current section for test organization"""
    global current_section
    current_section = section_name
    print(f"\n🧪 Testing: {section_name}")

def test_basic_imports():
    """Test basic module imports"""
    set_section("Basic Imports")
    
    # Test main import
    try:
        import groggy as gr
        log_test("Import groggy as gr", success=True)
        return gr
    except Exception as e:
        log_test("Import groggy as gr", success=False, 
                error_msg=str(e), code_snippet="import groggy as gr")
        return None

def test_graph_creation(gr):
    """Test basic graph creation"""
    set_section("Graph Creation")
    
    # Test default graph creation
    try:
        g = gr.Graph()
        log_test("Create default graph", success=True)
    except Exception as e:
        log_test("Create default graph", success=False, 
                error_msg=str(e), code_snippet="g = gr.Graph()")
        return None
    
    # Test directed graph creation
    try:
        g_directed = gr.Graph(directed=True)
        log_test("Create directed graph", success=True)
    except Exception as e:
        log_test("Create directed graph", success=False, 
                error_msg=str(e), code_snippet="g = gr.Graph(directed=True)")
    
    # Test undirected graph creation
    try:
        g_undirected = gr.Graph(directed=False)
        log_test("Create undirected graph", success=True)
    except Exception as e:
        log_test("Create undirected graph", success=False, 
                error_msg=str(e), code_snippet="g = gr.Graph(directed=False)")
    
    return g

def test_node_operations(gr, g):
    """Test node addition and access"""
    if not g:
        return None, None, None
        
    set_section("Node Operations")
    
    alice, bob, charlie = None, None, None
    
    # Test single node addition
    try:
        alice = g.add_node(name="Alice", age=30, department="Engineering")
        log_test("Add single node with attributes", success=True)
    except Exception as e:
        log_test("Add single node with attributes", success=False,
                error_msg=str(e), 
                code_snippet='alice = g.add_node(name="Alice", age=30, department="Engineering")')
    
    try:
        bob = g.add_node(name="Bob", age=25, department="Design")
        charlie = g.add_node(name="Charlie", age=35, department="Management")
        log_test("Add multiple individual nodes", success=True)
    except Exception as e:
        log_test("Add multiple individual nodes", success=False,
                error_msg=str(e))
    
    # Test batch node addition
    try:
        nodes_data = [
            {'name': 'Dave', 'age': 28, 'department': 'Marketing'},
            {'name': 'Eve', 'age': 32, 'department': 'HR'}
        ]
        batch_node_ids = g.add_nodes(nodes_data)
        log_test("Batch node addition", success=True)
    except Exception as e:
        log_test("Batch node addition", success=False,
                error_msg=str(e),
                code_snippet="g.add_nodes([{'name': 'Dave', 'age': 28}])")
    
    # Test node count
    try:
        node_count = g.node_count()
        log_test(f"Get node count: {node_count}", success=True)
    except Exception as e:
        log_test("Get node count", success=False,
                error_msg=str(e), code_snippet="g.node_count()")
    
    # Test has_node
    try:
        if alice is not None:
            has_alice = g.has_node(alice)
            log_test(f"Check node exists: {has_alice}", success=True)
    except Exception as e:
        log_test("Check node exists", success=False,
                error_msg=str(e), code_snippet="g.has_node(alice)")
    
    return alice, bob, charlie

def test_edge_operations(gr, g, alice, bob, charlie):
    """Test edge addition and access"""
    if not g or not alice or not bob:
        return
        
    set_section("Edge Operations")
    
    # Test single edge addition
    try:
        g.add_edge(alice, bob, relationship="collaborates", strength=0.8)
        log_test("Add single edge with attributes", success=True)
    except Exception as e:
        log_test("Add single edge with attributes", success=False,
                error_msg=str(e),
                code_snippet='g.add_edge(alice, bob, relationship="collaborates", strength=0.8)')
    
    # Test additional edges
    try:
        if charlie:
            g.add_edge(charlie, alice, relationship="manages", strength=0.9)
            g.add_edge(charlie, bob, relationship="manages", strength=0.7)
        log_test("Add multiple edges", success=True)
    except Exception as e:
        log_test("Add multiple edges", success=False, error_msg=str(e))
    
    # Test batch edge addition
    try:
        if alice and bob and charlie:
            edges_data = [
                (alice, charlie, {'type': 'reports_to'}),
                (bob, charlie, {'type': 'reports_to'})
            ]
            g.add_edges(edges_data)
            log_test("Batch edge addition", success=True)
    except Exception as e:
        log_test("Batch edge addition", success=False,
                error_msg=str(e),
                code_snippet="g.add_edges([(alice, bob, {'weight': 0.8})])")
    
    # Test edge count
    try:
        edge_count = g.edge_count()
        log_test(f"Get edge count: {edge_count}", success=True)
    except Exception as e:
        log_test("Get edge count", success=False,
                error_msg=str(e), code_snippet="g.edge_count()")
    
    # Test has_edge
    try:
        if alice and bob:
            has_edge = g.has_edge(alice, bob)
            log_test(f"Check edge exists: {has_edge}", success=True)
    except Exception as e:
        log_test("Check edge exists", success=False,
                error_msg=str(e), code_snippet="g.has_edge(alice, bob)")

def test_graph_properties(gr, g):
    """Test graph property access"""
    if not g:
        return
        
    set_section("Graph Properties")
    
    # Test is_directed
    try:
        is_directed = g.is_directed
        log_test(f"Check is_directed: {is_directed}", success=True)
    except Exception as e:
        log_test("Check is_directed", success=False,
                error_msg=str(e), code_snippet="g.is_directed")
    
    # Test density
    try:
        density = g.density()
        log_test(f"Calculate density: {density:.4f}", success=True)
    except Exception as e:
        log_test("Calculate density", success=False,
                error_msg=str(e), code_snippet="g.density()")
    
    # Test is_connected
    try:
        is_connected = g.is_connected()
        log_test(f"Check is_connected: {is_connected}", success=True)
    except Exception as e:
        log_test("Check is_connected", success=False,
                error_msg=str(e), code_snippet="g.is_connected()")

def test_degree_operations(gr, g, alice, bob):
    """Test degree calculations"""
    if not g:
        return
        
    set_section("Degree Operations")
    
    # Test all degrees
    try:
        all_degrees = g.degree()
        log_test(f"Get all degrees: {type(all_degrees)}", success=True)
    except Exception as e:
        log_test("Get all degrees", success=False,
                error_msg=str(e), code_snippet="g.degree()")
    
    # Test specific node degree
    try:
        if alice:
            alice_degree = g.degree(alice)
            log_test(f"Get specific node degree: {alice_degree}", success=True)
    except Exception as e:
        log_test("Get specific node degree", success=False,
                error_msg=str(e), code_snippet="g.degree(alice)")
    
    # Test directed graph degrees
    try:
        in_degrees = g.in_degree()
        out_degrees = g.out_degree()
        log_test("Get in/out degrees", success=True)
    except Exception as e:
        log_test("Get in/out degrees", success=False,
                error_msg=str(e), code_snippet="g.in_degree(), g.out_degree()")

def test_node_access(gr, g, alice):
    """Test node attribute access"""
    if not g or not alice:
        return
        
    set_section("Node Access")
    
    # Test node views access
    try:
        nodes_view = g.nodes
        log_test("Access nodes view", success=True)
    except Exception as e:
        log_test("Access nodes view", success=False,
                error_msg=str(e), code_snippet="g.nodes")
    
    # Test individual node access
    try:
        if alice:
            alice_data = g.nodes[alice]
            log_test(f"Access node data: {type(alice_data)}", success=True)
    except Exception as e:
        log_test("Access node data", success=False,
                error_msg=str(e), code_snippet="g.nodes[alice]")
    
    # Test node attribute access
    try:
        if alice:
            alice_node = g.nodes[alice]
            name = alice_node['name']
            log_test(f"Access node attribute: {name}", success=True)
    except Exception as e:
        log_test("Access node attribute", success=False,
                error_msg=str(e), code_snippet="g.nodes[alice]['name']")

def test_table_operations(gr, g):
    """Test table creation and operations"""
    if not g:
        return
        
    set_section("Table Operations")
    
    # Test nodes table
    try:
        nodes_table = g.nodes.table()
        log_test(f"Get nodes table: {type(nodes_table)}", success=True)
    except Exception as e:
        log_test("Get nodes table", success=False,
                error_msg=str(e), code_snippet="g.nodes.table()")
        return None
    
    # Test edges table
    try:
        edges_table = g.edges.table()
        log_test(f"Get edges table: {type(edges_table)}", success=True)
    except Exception as e:
        log_test("Get edges table", success=False,
                error_msg=str(e), code_snippet="g.edges.table()")
    
    # Test table properties
    try:
        shape = nodes_table.shape
        log_test(f"Table shape: {shape}", success=True)
    except Exception as e:
        log_test("Table shape", success=False,
                error_msg=str(e), code_snippet="nodes_table.shape")
    
    try:
        columns = nodes_table.columns
        log_test(f"Table columns: {columns}", success=True)
    except Exception as e:
        log_test("Table columns", success=False,
                error_msg=str(e), code_snippet="nodes_table.columns")
    
    # Test head/tail
    try:
        head_result = nodes_table.head()
        log_test("Table head()", success=True)
    except Exception as e:
        log_test("Table head()", success=False,
                error_msg=str(e), code_snippet="nodes_table.head()")
    
    try:
        tail_result = nodes_table.tail()
        log_test("Table tail()", success=True)
    except Exception as e:
        log_test("Table tail()", success=False,
                error_msg=str(e), code_snippet="nodes_table.tail()")
    
    return nodes_table

def test_array_operations(gr, nodes_table):
    """Test array operations on table columns"""
    if not nodes_table:
        return
        
    set_section("Array Operations")
    
    # Test column access
    try:
        age_column = nodes_table['age']
        log_test(f"Access table column: {type(age_column)}", success=True)
    except Exception as e:
        log_test("Access table column", success=False,
                error_msg=str(e), code_snippet="nodes_table['age']")
        return
    
    # Test array statistics
    try:
        mean_age = age_column.mean()
        log_test(f"Array mean: {mean_age}", success=True)
    except Exception as e:
        log_test("Array mean", success=False,
                error_msg=str(e), code_snippet="age_column.mean()")
    
    try:
        std_age = age_column.std()
        log_test(f"Array std: {std_age}", success=True)
    except Exception as e:
        log_test("Array std", success=False,
                error_msg=str(e), code_snippet="age_column.std()")
    
    try:
        describe_result = age_column.describe()
        log_test(f"Array describe: {type(describe_result)}", success=True)
    except Exception as e:
        log_test("Array describe", success=False,
                error_msg=str(e), code_snippet="age_column.describe()")

def test_table_statistics(gr, nodes_table):
    """Test table-level statistics"""
    if not nodes_table:
        return
        
    set_section("Table Statistics")
    
    # Test table mean
    try:
        mean_result = nodes_table.mean('age')
        log_test(f"Table mean: {mean_result}", success=True)
    except Exception as e:
        log_test("Table mean", success=False,
                error_msg=str(e), code_snippet="nodes_table.mean('age')")
    
    # Test table sum
    try:
        sum_result = nodes_table.sum('age')
        log_test(f"Table sum: {sum_result}", success=True)
    except Exception as e:
        log_test("Table sum", success=False,
                error_msg=str(e), code_snippet="nodes_table.sum('age')")
    
    # Test table describe
    try:
        describe_result = nodes_table.describe()
        log_test(f"Table describe: {type(describe_result)}", success=True)
    except Exception as e:
        log_test("Table describe", success=False,
                error_msg=str(e), code_snippet="nodes_table.describe()")

def test_sorting_filtering(gr, nodes_table):
    """Test table sorting and filtering"""
    if not nodes_table:
        return
        
    set_section("Sorting and Filtering")
    
    # Test sorting
    try:
        sorted_table = nodes_table.sort_by('age', ascending=True)
        log_test(f"Sort table: {type(sorted_table)}", success=True)
    except Exception as e:
        log_test("Sort table", success=False,
                error_msg=str(e), code_snippet="nodes_table.sort_by('age', ascending=True)")
    
    # Test boolean indexing
    try:
        young_people = nodes_table[nodes_table['age'] < 30]
        log_test(f"Boolean indexing: {type(young_people)}", success=True)
    except Exception as e:
        log_test("Boolean indexing", success=False,
                error_msg=str(e), code_snippet="nodes_table[nodes_table['age'] < 30]")

def test_node_filtering(gr, g):
    """Test graph node filtering"""
    if not g:
        return
        
    set_section("Node Filtering")
    
    # Test NodeFilter creation and usage
    try:
        young_filter = gr.NodeFilter.attribute_filter('age', gr.AttributeFilter.less_than(30))
        young_nodes = g.filter_nodes(young_filter)
        log_test(f"Filter nodes: {type(young_nodes)}", success=True)
    except Exception as e:
        log_test("Filter nodes", success=False,
                error_msg=str(e), 
                code_snippet="gr.NodeFilter.attribute_filter('age', gr.AttributeFilter.less_than(30))")
    
    # Test complex filters
    try:
        eng_filter = gr.NodeFilter.attribute_equals('department', 'Engineering')
        age_filter = gr.NodeFilter.attribute_filter('age', gr.AttributeFilter.greater_than(25))
        combined_filter = gr.NodeFilter.and_filters([eng_filter, age_filter])
        filtered_nodes = g.filter_nodes(combined_filter)
        log_test("Complex node filtering", success=True)
    except Exception as e:
        log_test("Complex node filtering", success=False,
                error_msg=str(e), code_snippet="gr.NodeFilter.and_filters([...])")

def test_edge_filtering(gr, g):
    """Test graph edge filtering"""
    if not g:
        return
        
    set_section("Edge Filtering")
    
    # Test EdgeFilter
    try:
        strong_filter = gr.EdgeFilter.attribute_filter('strength', gr.AttributeFilter.greater_than(0.7))
        strong_edges = g.filter_edges(strong_filter)
        log_test(f"Filter edges: {type(strong_edges)}", success=True)
    except Exception as e:
        log_test("Filter edges", success=False,
                error_msg=str(e),
                code_snippet="gr.EdgeFilter.attribute_filter('strength', gr.AttributeFilter.greater_than(0.7))")

def test_analytics(gr, g, alice, bob):
    """Test graph analytics"""
    if not g:
        return
        
    set_section("Graph Analytics")
    
    # Test connected components
    try:
        components = g.analytics.connected_components()
        log_test(f"Connected components: {len(components)} components", success=True)
    except Exception as e:
        log_test("Connected components", success=False,
                error_msg=str(e), code_snippet="g.analytics.connected_components()")
    
    # Test shortest path
    try:
        if alice and bob:
            path = g.analytics.shortest_path(alice, bob)
            log_test(f"Shortest path: {path}", success=True)
    except Exception as e:
        log_test("Shortest path", success=False,
                error_msg=str(e), code_snippet="g.analytics.shortest_path(alice, bob)")
    
    # Test has_path
    try:
        if alice and bob:
            has_path = g.analytics.has_path(alice, bob)
            log_test(f"Has path: {has_path}", success=True)
    except Exception as e:
        log_test("Has path", success=False,
                error_msg=str(e), code_snippet="g.analytics.has_path(alice, bob)")
    
    # Test BFS
    try:
        if alice:
            bfs_result = g.analytics.bfs(alice)
            log_test(f"BFS: {type(bfs_result)}", success=True)
    except Exception as e:
        log_test("BFS", success=False,
                error_msg=str(e), code_snippet="g.analytics.bfs(alice)")
    
    # Test DFS
    try:
        if alice:
            dfs_result = g.analytics.dfs(alice)
            log_test(f"DFS: {type(dfs_result)}", success=True)
    except Exception as e:
        log_test("DFS", success=False,
                error_msg=str(e), code_snippet="g.analytics.dfs(alice)")

def test_matrix_operations(gr, g):
    """Test matrix operations"""
    if not g:
        return
        
    set_section("Matrix Operations")
    
    # Test adjacency matrix
    try:
        adj_matrix = g.adjacency()
        log_test(f"Adjacency matrix: {type(adj_matrix)}", success=True)
    except Exception as e:
        log_test("Adjacency matrix", success=False,
                error_msg=str(e), code_snippet="g.adjacency()")
        return
    
    # Test matrix properties
    try:
        shape = adj_matrix.shape
        log_test(f"Matrix shape: {shape}", success=True)
    except Exception as e:
        log_test("Matrix shape", success=False,
                error_msg=str(e), code_snippet="adj_matrix.shape")
    
    try:
        is_sparse = adj_matrix.is_sparse
        log_test(f"Matrix is_sparse: {is_sparse}", success=True)
    except Exception as e:
        log_test("Matrix is_sparse", success=False,
                error_msg=str(e), code_snippet="adj_matrix.is_sparse")
    
    # Test matrix operations
    try:
        row_sums = adj_matrix.sum_axis(1)
        log_test(f"Matrix row sums: {type(row_sums)}", success=True)
    except Exception as e:
        log_test("Matrix row sums", success=False,
                error_msg=str(e), code_snippet="adj_matrix.sum_axis(1)")
    
    try:
        col_means = adj_matrix.mean_axis(0)
        log_test(f"Matrix column means: {type(col_means)}", success=True)
    except Exception as e:
        log_test("Matrix column means", success=False,
                error_msg=str(e), code_snippet="adj_matrix.mean_axis(0)")
    
    try:
        matrix_power = adj_matrix.power(2)
        log_test(f"Matrix power: {type(matrix_power)}", success=True)
    except Exception as e:
        log_test("Matrix power", success=False,
                error_msg=str(e), code_snippet="adj_matrix.power(2)")

def test_graph_aware_table_ops(gr, g, nodes_table, alice):
    """Test graph-aware table operations"""
    if not g or not nodes_table:
        return
        
    set_section("Graph-Aware Table Operations")
    
    # Test filter by degree
    try:
        high_degree = nodes_table.filter_by_degree(g, 'node_id', min_degree=1)
        log_test(f"Filter by degree: {type(high_degree)}", success=True)
    except Exception as e:
        log_test("Filter by degree", success=False,
                error_msg=str(e), 
                code_snippet="nodes_table.filter_by_degree(g, 'node_id', min_degree=1)")
    
    # Test filter by connectivity
    try:
        if alice:
            connected = nodes_table.filter_by_connectivity(g, 'node_id', [alice], mode='direct')
            log_test(f"Filter by connectivity: {type(connected)}", success=True)
    except Exception as e:
        log_test("Filter by connectivity", success=False,
                error_msg=str(e),
                code_snippet="nodes_table.filter_by_connectivity(g, 'node_id', [alice], mode='direct')")
    
    # Test filter by distance
    try:
        if alice:
            nearby = nodes_table.filter_by_distance(g, 'node_id', [alice], max_distance=2)
            log_test(f"Filter by distance: {type(nearby)}", success=True)
    except Exception as e:
        log_test("Filter by distance", success=False,
                error_msg=str(e),
                code_snippet="nodes_table.filter_by_distance(g, 'node_id', [alice], max_distance=2)")

def test_table_joins(gr, nodes_table):
    """Test table join operations"""
    if not nodes_table:
        return
        
    set_section("Table Joins")
    
    # Create test table for joining
    try:
        salary_data = {
            'name': ['Alice', 'Bob', 'Charlie'],
            'salary': [95000, 75000, 120000]
        }
        salary_table = gr.table(salary_data)
        log_test("Create table for joining", success=True)
    except Exception as e:
        log_test("Create table for joining", success=False,
                error_msg=str(e), code_snippet="gr.table(salary_data)")
        return
    
    # Test inner join
    try:
        inner_result = nodes_table.inner_join(salary_table, 'name', 'name')
        log_test(f"Inner join: {type(inner_result)}", success=True)
    except Exception as e:
        log_test("Inner join", success=False,
                error_msg=str(e), 
                code_snippet="nodes_table.inner_join(salary_table, 'name', 'name')")
    
    # Test other joins
    try:
        left_result = nodes_table.left_join(salary_table, 'name', 'name')
        log_test("Left join", success=True)
    except Exception as e:
        log_test("Left join", success=False, error_msg=str(e))
    
    try:
        right_result = nodes_table.right_join(salary_table, 'name', 'name')
        log_test("Right join", success=True)
    except Exception as e:
        log_test("Right join", success=False, error_msg=str(e))
    
    try:
        outer_result = nodes_table.outer_join(salary_table, 'name', 'name')
        log_test("Outer join", success=True)
    except Exception as e:
        log_test("Outer join", success=False, error_msg=str(e))

def test_exports(gr, g, nodes_table):
    """Test export functionality"""
    if not g:
        return
        
    set_section("Export Operations")
    
    # Test NetworkX export
    try:
        nx_graph = g.to_networkx()
        log_test(f"NetworkX export: {type(nx_graph)}", success=True)
    except Exception as e:
        log_test("NetworkX export", success=False,
                error_msg=str(e), code_snippet="g.to_networkx()")
    
    # Test pandas export
    try:
        if nodes_table:
            pandas_df = nodes_table.to_pandas()
            log_test(f"Pandas export: {type(pandas_df)}", success=True)
    except Exception as e:
        log_test("Pandas export", success=False,
                error_msg=str(e), code_snippet="nodes_table.to_pandas()")
    
    # Test numpy export
    try:
        adj_matrix = g.adjacency()
        numpy_array = adj_matrix.to_numpy()
        log_test(f"NumPy export: {type(numpy_array)}", success=True)
    except Exception as e:
        log_test("NumPy export", success=False,
                error_msg=str(e), code_snippet="adj_matrix.to_numpy()")

def test_table_creation(gr):
    """Test standalone table creation"""
    set_section("Standalone Table Creation")
    
    # Test gr.table() creation
    try:
        data = {
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [30, 25, 35],
            'department': ['Engineering', 'Design', 'Management']
        }
        table = gr.table(data)
        log_test(f"Create table from dict: {type(table)}", success=True)
        return table
    except Exception as e:
        log_test("Create table from dict", success=False,
                error_msg=str(e), code_snippet="gr.table(data)")
        return None

def test_array_creation(gr):
    """Test standalone array creation"""
    set_section("Standalone Array Creation")
    
    # Test gr.array() creation
    try:
        arr = gr.array([25, 30, 35, 40, 45])
        log_test(f"Create array: {type(arr)}", success=True)
        return arr
    except Exception as e:
        log_test("Create array", success=False,
                error_msg=str(e), code_snippet="gr.array([25, 30, 35, 40, 45])")
        return None

def test_matrix_creation(gr):
    """Test standalone matrix creation"""
    set_section("Standalone Matrix Creation")
    
    # Test gr.matrix() creation
    try:
        matrix_data = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        matrix = gr.matrix(matrix_data)
        log_test(f"Create matrix: {type(matrix)}", success=True)
        return matrix
    except Exception as e:
        log_test("Create matrix", success=False,
                error_msg=str(e), code_snippet="gr.matrix([[1, 0, 1], [0, 1, 0], [1, 0, 1]])")
        return None

def generate_report():
    """Generate a comprehensive test report"""
    
    report = f"""# Groggy Documentation Validation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

"""
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result['success'])
    failed_tests = total_tests - passed_tests
    
    report += f"""- **Total Tests**: {total_tests}
- **Passed**: {passed_tests} ✅
- **Failed**: {failed_tests} ❌
- **Success Rate**: {(passed_tests/total_tests*100):.1f}%

"""
    
    # Group by section
    sections = {}
    for result in test_results:
        section = result['section']
        if section not in sections:
            sections[section] = {'passed': 0, 'failed': 0, 'tests': []}
        
        if result['success']:
            sections[section]['passed'] += 1
        else:
            sections[section]['failed'] += 1
        sections[section]['tests'].append(result)
    
    report += "## Results by Section\n\n"
    
    for section, data in sections.items():
        total = data['passed'] + data['failed']
        success_rate = (data['passed'] / total * 100) if total > 0 else 0
        
        status = "✅" if data['failed'] == 0 else "⚠️" if data['failed'] < data['passed'] else "❌"
        
        report += f"### {status} {section}\n"
        report += f"- Passed: {data['passed']}/{total} ({success_rate:.1f}%)\n"
        
        if data['failed'] > 0:
            report += f"- **Failed Tests:**\n"
            for test in data['tests']:
                if not test['success']:
                    report += f"  - `{test['test']}`: {test['error']}\n"
                    if test['code']:
                        report += f"    ```python\n    {test['code']}\n    ```\n"
        report += "\n"
    
    report += "## Failed Tests Details\n\n"
    
    failed_count = 0
    for result in test_results:
        if not result['success']:
            failed_count += 1
            report += f"### {failed_count}. {result['test']} ({result['section']})\n\n"
            report += f"**Error**: `{result['error']}`\n\n"
            if result['code']:
                report += f"**Code**:\n```python\n{result['code']}\n```\n\n"
            report += "**Status**: Needs documentation fix or implementation\n\n"
    
    if failed_count == 0:
        report += "*No failed tests - all documentation examples work!* 🎉\n\n"
    
    report += "## Recommendations\n\n"
    
    if failed_count > 0:
        report += f"1. **Fix {failed_count} failing examples** in documentation\n"
        report += "2. **Update method signatures** to match actual implementation\n"
        report += "3. **Remove non-existent features** from documentation\n"
        report += "4. **Re-run this test** after fixes to verify\n\n"
    else:
        report += "✅ **All documentation examples work correctly!**\n\n"
        report += "The documentation is ready for release.\n\n"
    
    report += "---\n*Generated by Groggy Documentation Validation Suite*\n"
    
    return report

def main():
    """Run all validation tests"""
    print("🧪 Groggy Documentation Validation Test Suite")
    print("=" * 50)
    
    # Test basic imports first
    gr = test_basic_imports()
    if not gr:
        print("❌ CRITICAL: Cannot import groggy - stopping tests")
        return
    
    # Test graph creation
    g = test_graph_creation(gr)
    
    # Test node operations
    alice, bob, charlie = test_node_operations(gr, g)
    
    # Test edge operations
    test_edge_operations(gr, g, alice, bob, charlie)
    
    # Test graph properties
    test_graph_properties(gr, g)
    
    # Test degree operations
    test_degree_operations(gr, g, alice, bob)
    
    # Test node access
    test_node_access(gr, g, alice)
    
    # Test table operations
    nodes_table = test_table_operations(gr, g)
    
    # Test array operations
    test_array_operations(gr, nodes_table)
    
    # Test table statistics
    test_table_statistics(gr, nodes_table)
    
    # Test sorting and filtering
    test_sorting_filtering(gr, nodes_table)
    
    # Test node filtering
    test_node_filtering(gr, g)
    
    # Test edge filtering
    test_edge_filtering(gr, g)
    
    # Test analytics
    test_analytics(gr, g, alice, bob)
    
    # Test matrix operations
    test_matrix_operations(gr, g)
    
    # Test graph-aware table operations
    test_graph_aware_table_ops(gr, g, nodes_table, alice)
    
    # Test table joins
    test_table_joins(gr, nodes_table)
    
    # Test exports
    test_exports(gr, g, nodes_table)
    
    # Test standalone creation
    test_table_creation(gr)
    test_array_creation(gr)
    test_matrix_creation(gr)
    
    # Generate and save report
    print("\n" + "=" * 50)
    print("📊 Generating validation report...")
    
    report = generate_report()
    
    # Save report
    report_file = "validation_results.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {report_file}")
    
    # Print summary
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result['success'])
    failed_tests = total_tests - passed_tests
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests} ✅")
    print(f"   Failed: {failed_tests} ❌")
    print(f"   Success Rate: {(passed_tests/total_tests*100):.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! Documentation is ready for release!")
    else:
        print(f"\n⚠️  {failed_tests} tests failed - documentation needs fixes")
        print(f"   See {report_file} for detailed failure analysis")

if __name__ == "__main__":
    main()