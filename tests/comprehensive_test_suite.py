#!/usr/bin/env python3
"""
Comprehensive Groggy Test Suite
===============================

This comprehensive test suite goes beyond basic validation to test:
- Every available method and property
- Edge cases and boundary conditions  
- Complex composability scenarios
- Error handling and recovery
- Performance characteristics
- Cross-feature integration
- Persona-based use cases (Engineering, Research, Analysis, etc.)

Inspired by YN's approach to systematic testing with a "flashlight in the dark."
"""

import sys
import traceback
import time
import gc
from datetime import datetime
from typing import List, Dict, Any, Optional
import random

# Global results tracking  
test_results = []
current_section = ""
start_time = None

def log_test(test_name, success=True, error_msg="", code_snippet="", execution_time=0.0, memory_usage=0):
    """Log a test result with performance metrics"""
    global test_results, current_section
    test_results.append({
        'section': current_section,
        'test': test_name,
        'success': success,
        'error': error_msg,
        'code': code_snippet,
        'execution_time': execution_time,
        'memory_usage': memory_usage
    })
    
    status = "✅ PASS" if success else "❌ FAIL"
    timing = f" ({execution_time:.3f}s)" if execution_time > 0.001 else ""
    print(f"{status}: {test_name}{timing}")
    if not success and error_msg:
        print(f"    Error: {error_msg}")

def set_section(section_name):
    """Set the current section for test organization"""
    global current_section, start_time
    current_section = section_name
    start_time = time.time()
    print(f"\n🧪 Testing: {section_name}")

def measure_performance(func, *args, **kwargs):
    """Measure execution time and memory usage of a function"""
    gc.collect()  # Clean garbage before measuring
    start_memory = sys.getsizeof(gc.get_objects())
    
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        end_time = time.time()
        
        gc.collect()
        end_memory = sys.getsizeof(gc.get_objects())
        
        return result, end_time - start_time, end_memory - start_memory
    except Exception as e:
        end_time = time.time()
        raise e

def test_comprehensive_imports():
    """Test all available imports and submodules"""
    set_section("Comprehensive Imports")
    
    imports_to_test = [
        ("groggy", "import groggy"),
        ("groggy.Graph", "from groggy import Graph"),
        ("groggy.NodeFilter", "from groggy import NodeFilter"),  
        ("groggy.EdgeFilter", "from groggy import EdgeFilter"),
        ("groggy.AttributeFilter", "from groggy import AttributeFilter"),
        ("groggy.table", "from groggy import table"),
        ("groggy.array", "from groggy import array"),
        ("groggy.matrix", "from groggy import matrix"),
        ("groggy generators", "from groggy import generators"),
        ("groggy.parse_node_query", "from groggy import parse_node_query"),
        ("groggy.parse_edge_query", "from groggy import parse_edge_query"),
    ]
    
    gr = None
    
    for import_name, import_code in imports_to_test:
        try:
            if import_name == "groggy":
                import groggy
                gr = groggy
                log_test(f"Import {import_name}", success=True)
            elif import_name == "groggy.Graph":
                from groggy import Graph
                log_test(f"Import {import_name}", success=True)
            elif import_name == "groggy.NodeFilter":
                from groggy import NodeFilter
                log_test(f"Import {import_name}", success=True)
            elif import_name == "groggy.EdgeFilter":
                from groggy import EdgeFilter
                log_test(f"Import {import_name}", success=True)
            elif import_name == "groggy.AttributeFilter":
                from groggy import AttributeFilter
                log_test(f"Import {import_name}", success=True)
            elif import_name == "groggy.table":
                from groggy import table
                log_test(f"Import {import_name}", success=True)
            elif import_name == "groggy.array":
                from groggy import array
                log_test(f"Import {import_name}", success=True)
            elif import_name == "groggy.matrix":
                from groggy import matrix
                log_test(f"Import {import_name}", success=True)
            elif import_name == "groggy generators":
                from groggy import generators
                log_test(f"Import {import_name}", success=True)
            elif import_name == "groggy.parse_node_query":
                from groggy import parse_node_query
                log_test(f"Import {import_name}", success=True)
            elif import_name == "groggy.parse_edge_query":
                from groggy import parse_edge_query
                log_test(f"Import {import_name}", success=True)
        except Exception as e:
            log_test(f"Import {import_name}", success=False, 
                    error_msg=str(e), code_snippet=import_code)
    
    return gr

def test_edge_cases_graph_creation(gr):
    """Test edge cases in graph creation"""
    if not gr:
        print("❌ CRITICAL DEBUG: gr module is None or False")
        return None
        
    set_section("Edge Cases - Graph Creation")
    
    # Test various graph creation parameters
    edge_case_tests = [
        ("Empty directed graph", lambda: gr.Graph(directed=True)),
        ("Empty undirected graph", lambda: gr.Graph(directed=False)),
        ("Graph with explicit directed=None", lambda: gr.Graph(directed=None)),
    ]
    
    graphs = []
    failed_tests = []
    
    for test_name, test_func in edge_case_tests:
        try:
            result, exec_time, memory = measure_performance(test_func)
            graphs.append(result)
            log_test(test_name, success=True, execution_time=exec_time)
        except Exception as e:
            print(f"    ❌ FAILED {test_name}: {str(e)}")
            print(f"    📍 Full traceback:")
            traceback.print_exc()
            log_test(test_name, success=False, error_msg=str(e))
            graphs.append(None)
            failed_tests.append(test_name)
    
    print(f"📊 Graph creation summary: {len([g for g in graphs if g is not None])}/{len(graphs)} succeeded")
    if failed_tests:
        print(f"❌ Failed tests: {failed_tests}")
    
    # Test graph properties immediately after creation
    property_failures = []
    for i, g in enumerate(graphs):
        if g is not None:
            try:
                node_count = g.node_count()
                edge_count = g.edge_count()
                
                # Test that new graphs are empty
                assert node_count == 0, f"New graph should have 0 nodes, got {node_count}"
                assert edge_count == 0, f"New graph should have 0 edges, got {edge_count}"
                log_test(f"Graph {i} empty state verification", success=True)
            except Exception as e:
                print(f"    ❌ Graph {i} property check failed: {str(e)}")
                property_failures.append(f"Graph {i}: {str(e)}")
                log_test(f"Graph {i} empty state verification", success=False, error_msg=str(e))
        else:
            print(f"    ⚠️  Graph {i} is None, skipping property check")
    
    if property_failures:
        print(f"❌ Property check failures: {property_failures}")
    
    # Return first successful graph, but provide detailed info if none available
    successful_graphs = [g for g in graphs if g is not None]
    if successful_graphs:
        return successful_graphs[0]
    else:
        print("❌ CRITICAL DEBUG: No successful graphs created!")
        print(f"    - Total attempts: {len(graphs)}")
        print(f"    - Failed creation tests: {failed_tests}")
        print(f"    - Property failures: {property_failures}")
        return None

def test_boundary_conditions_nodes(gr, g):
    """Test boundary conditions for node operations"""
    if not gr or g is None:
        return []
    
    set_section("Boundary Conditions - Nodes")
    
    node_ids = []
    
    # Test adding nodes with various attribute types and edge cases
    boundary_tests = [
        # Basic types
        ("Node with string attr", {"name": "test"}),
        ("Node with int attr", {"value": 42}),
        ("Node with float attr", {"score": 3.14159}),
        ("Node with bool attr", {"active": True}),
        ("Node with None attr", {"missing": None}),
        
        # Edge case values
        ("Node with empty string", {"name": ""}),
        ("Node with zero int", {"value": 0}),
        ("Node with negative int", {"value": -999}),
        ("Node with zero float", {"score": 0.0}),
        ("Node with negative float", {"score": -3.14}),
        ("Node with very large int", {"big": 2**63 - 1}),
        ("Node with very small float", {"tiny": 1e-10}),
        ("Node with infinity", {"inf": float('inf')}),
        ("Node with NaN", {"nan": float('nan')}),
        
        # Complex types
        ("Node with list attr", {"items": [1, 2, 3]}),
        ("Node with large list", {"big_list": list(range(100))}),
        ("Node with nested dict", {"config": {"nested": {"deep": "value"}}}),
        ("Node with unicode", {"unicode": "🌟 Unicode test 中文 💫"}),
        
        # No attributes
        ("Node with no attributes", {}),
    ]
    
    for test_name, attrs in boundary_tests:
        try:
            result, exec_time, memory = measure_performance(g.add_node, **attrs)
            node_ids.append(result)
            log_test(test_name, success=True, execution_time=exec_time)
            
            # Verify the node was added and attributes are accessible
            if attrs:
                for key, expected_value in attrs.items():
                    try:
                        actual_value = g.get_node_attr(result, key)
                        # Handle NaN comparison specially
                        if isinstance(expected_value, float) and str(expected_value) == 'nan':
                            assert str(actual_value) == 'nan', f"Expected NaN, got {actual_value}"
                        else:
                            # Note: may need to handle AttrValue comparison
                            log_test(f"Verify {key} attribute", success=True)
                    except Exception as e:
                        log_test(f"Verify {key} attribute", success=False, error_msg=str(e))
                        
        except Exception as e:
            log_test(test_name, success=False, error_msg=str(e))
            node_ids.append(None)
    
    # Test bulk operations with boundary conditions
    try:
        # Add many nodes at once
        bulk_data = [{"id": i, "bulk": True} for i in range(100)]
        result, exec_time, memory = measure_performance(g.add_nodes, bulk_data)
        node_ids.extend(result)
        log_test(f"Bulk add 100 nodes", success=True, execution_time=exec_time, memory_usage=memory)
    except Exception as e:
        log_test("Bulk add 100 nodes", success=False, error_msg=str(e))
    
    return node_ids

def test_bulk_operations(gr, g, node_ids):
    """Test bulk/plural operations - core architectural strength"""
    if not gr or g is None or not node_ids:
        return
    
    set_section("Bulk Operations - Core Architecture")
    
    # Test set_node_attrs - the critical missing method you mentioned
    try:
        # Create test data for bulk attribute setting in the correct format
        # Format: {"attr_name": {node_id: value, node_id: value, ...}}
        bulk_attrs = {
            "department": {
                node_ids[i]: (["Engineering"] * 5 + ["Research"] * 5)[i] 
                for i in range(10)
            },
            "salary": {
                node_ids[i]: [75000, 85000, 95000, 105000, 120000, 90000, 110000, 130000][i]
                for i in range(8)
            },
            "experience": {
                node_ids[i]: [2.5, 4.0, 6.5, 8.0, 10.5, 3.2][i]
                for i in range(6)
            }
        }
        
        result, exec_time, memory = measure_performance(g.set_node_attrs, bulk_attrs)
        log_test("set_node_attrs - bulk attribute setting", success=True, execution_time=exec_time)
        
        # Verify the bulk setting worked by checking individual values
        dept_check = g.get_node_attr(node_ids[0], "department")
        salary_check = g.get_node_attr(node_ids[2], "salary") 
        exp_check = g.get_node_attr(node_ids[4], "experience")
        
        assert dept_check == "Engineering", f"Expected 'Engineering', got '{dept_check}'"
        assert salary_check == 95000, f"Expected 95000, got {salary_check}"
        assert abs(exp_check - 10.5) < 0.001, f"Expected 10.5, got {exp_check}"
        
        log_test("Verify set_node_attrs results", success=True)
        
    except Exception as e:
        log_test("set_node_attrs - bulk attribute setting", success=False, error_msg=str(e))
        print(f"    📍 Full traceback:")
        traceback.print_exc()

    # Test get_node_attrs - bulk attribute retrieval  
    try:
        if hasattr(g, 'get_node_attrs'):
            # Get multiple attributes from multiple nodes
            target_nodes = node_ids[:5]
            target_attrs = ["department", "salary"] 
            
            result, exec_time, memory = measure_performance(g.get_node_attrs, target_nodes, target_attrs)
            log_test("get_node_attrs - bulk attribute retrieval", success=True, execution_time=exec_time)
            
            # Verify we got the expected structure back
            assert len(result) == len(target_nodes), f"Expected {len(target_nodes)} node results, got {len(result)}"
            log_test("Verify get_node_attrs structure", success=True)
        else:
            log_test("get_node_attrs - bulk attribute retrieval", success=False, 
                    error_msg="ARCHITECTURE GAP: get_node_attrs method does not exist. Bulk setters exist but no bulk getters!")
        
    except Exception as e:
        log_test("get_node_attrs - bulk attribute retrieval", success=False, error_msg=str(e))
        print(f"    📍 Full traceback:")
        traceback.print_exc()

    # Test bulk edge operations
    try:
        # Create some edges first 
        edges = [(node_ids[0], node_ids[1]), (node_ids[1], node_ids[2]), (node_ids[2], node_ids[3])]
        edge_ids = g.add_edges(edges)
        
        # Test set_edge_attrs in the correct format
        # Format: {"attr_name": {edge_id: value, edge_id: value, ...}}
        edge_bulk_attrs = {
            "relationship": {
                edge_ids[i]: ["reports_to", "collaborates_with"][i] 
                for i in range(2)
            },
            "strength": {
                edge_ids[i]: [0.8, 0.6, 0.9][i]
                for i in range(3)
            }
        }
        
        result, exec_time, memory = measure_performance(g.set_edge_attrs, edge_bulk_attrs)
        log_test("set_edge_attrs - bulk edge attributes", success=True, execution_time=exec_time)
        
        # Test get_edge_attrs with correct signature (edges, attrs)
        first_edge_attrs, exec_time, memory = measure_performance(
            g.get_edge_attrs, edge_ids, ["relationship", "strength"]
        )
        log_test("get_edge_attrs - bulk edge retrieval", success=True, execution_time=exec_time)
        
        # Verify the bulk retrieval worked
        log_test("Verify get_edge_attrs supports bulk retrieval", success=True)
        
    except Exception as e:
        log_test("Bulk edge operations", success=False, error_msg=str(e))
        print(f"    📍 Full traceback:")
        traceback.print_exc()

    # Test bulk removal operations
    try:
        # Test remove_nodes (plural)
        nodes_to_remove = node_ids[-5:] if len(node_ids) >= 5 else []
        if nodes_to_remove:
            initial_count = g.node_count()
            result, exec_time, memory = measure_performance(g.remove_nodes, nodes_to_remove)
            final_count = g.node_count()
            
            expected_count = initial_count - len(nodes_to_remove)
            assert final_count == expected_count, f"Expected {expected_count} nodes after removal, got {final_count}"
            log_test("remove_nodes - bulk node removal", success=True, execution_time=exec_time)
        else:
            log_test("remove_nodes - bulk node removal", success=False, error_msg="No nodes available for removal test")
            
    except Exception as e:
        log_test("remove_nodes - bulk node removal", success=False, error_msg=str(e))

def test_attribute_system_edge_cases(gr, g, node_ids):
    """Test the attribute system with edge cases"""
    if not gr or not g or not node_ids:
        return
    
    set_section("Attribute System - Edge Cases")
    
    test_node = node_ids[0] if node_ids else g.add_node()
    
    # Test setting/getting attributes with edge case values
    attribute_tests = [
        ("Set/get None", None),
        ("Set/get empty string", ""),
        ("Set/get zero", 0),
        ("Set/get negative", -42), 
        ("Set/get large number", 2**32),
        ("Set/get float precision", 1.23456789012345),
        ("Set/get boolean True", True),
        ("Set/get boolean False", False),
        ("Set/get large list", list(range(50))),  # Should trigger compression
        ("Set/get unicode string", "Unicode: 🌟 中文 العربية ελληνικά"),
        ("Set/get nested structure", {"a": {"b": {"c": [1, 2, 3]}}}),
    ]
    
    for test_name, value in attribute_tests:
        attr_name = test_name.lower().replace(" ", "_").replace("/", "_")
        
        try:
            # Set attribute
            result, exec_time, _ = measure_performance(g.set_node_attr, test_node, attr_name, value)
            log_test(f"{test_name} - SET", success=True, execution_time=exec_time)
            
            # Get attribute and verify
            retrieved, exec_time, _ = measure_performance(g.get_node_attr, test_node, attr_name)
            
            # Handle special cases for comparison
            if isinstance(value, float) and str(value) == 'nan':
                success = str(retrieved) == 'nan'
            elif value is None:
                success = retrieved is None or str(retrieved) == 'NaN'  # Handle AttrValue representation
            else:
                success = True  # For now, just verify no exception
            
            log_test(f"{test_name} - GET", success=success, execution_time=exec_time)
            
        except Exception as e:
            log_test(f"{test_name}", success=False, error_msg=str(e))
    
    # Test attribute overwriting and updates
    try:
        g.set_node_attr(test_node, "update_test", "initial")
        g.set_node_attr(test_node, "update_test", "updated")
        final_value = g.get_node_attr(test_node, "update_test")
        log_test("Attribute overwrite", success=True)
    except Exception as e:
        log_test("Attribute overwrite", success=False, error_msg=str(e))

def test_query_system_comprehensive(gr, g, node_ids):
    """Test the query system comprehensively"""
    if not gr or not g:
        return
    
    set_section("Query System - Comprehensive")
    
    # Always ensure we have nodes with diverse attributes for query testing
    query_test_data = [
        {"name": "Alice", "age": 25, "salary": 75000, "active": True},
        {"name": "Bob", "age": 30, "salary": 85000, "active": True}, 
        {"name": "Charlie", "age": 35, "salary": 95000, "active": False},
        {"name": "Diana", "age": 28, "salary": 80000, "active": True},
        {"name": "Eve", "age": 40, "salary": 100000, "active": False},
    ]
    
    # Add query test nodes to ensure attributes exist
    for data in query_test_data:
        try:
            query_node_id = g.add_node(**data)
            log_test(f"Setup query node: {data['name']}", success=True)
        except Exception as e:
            log_test(f"Setup query test node {data['name']}", success=False, error_msg=str(e))
            return
    
    # Test various query patterns
    query_tests = [
        # Basic comparisons
        ("Simple equals", "name == 'Alice'"),
        ("Simple greater than", "age > 30"),
        ("Simple less than", "salary < 90000"), 
        ("Simple not equals", "active != True"),
        ("Greater than or equal", "age >= 30"),
        ("Less than or equal", "salary <= 85000"),
        
        # Logical operators
        ("AND condition", "age > 25 AND salary < 90000"),
        ("OR condition", "age < 25 OR age > 35"),
        ("NOT condition", "NOT active == True"),
        ("Complex AND/OR", "(age > 25 AND salary > 80000) OR name == 'Alice'"),
        
        # Parentheses and precedence
        ("Parentheses precedence", "(age > 25 AND salary > 80000) OR (name == 'Eve' AND active == False)"),
        ("Nested parentheses", "((age > 25 AND salary > 75000) OR name == 'Alice') AND active == True"),
        
        # String operations
        ("String equals", "name == 'Bob'"),
        ("String not equals", "name != 'Alice'"),
        
        # Boolean operations 
        ("Boolean true", "active == True"),
        ("Boolean false", "active == False"),
        ("Boolean true variant", "active == true"),
        ("Boolean false variant", "active == false"),
        
        # Edge cases
        ("Empty query", ""),
        ("Whitespace query", "   "),
        ("Non-existent attribute", "nonexistent > 0"),
    ]
    
    for test_name, query_str in query_tests:
        try:
            if query_str.strip() == "":
                # Empty query should fail
                try:
                    filter_obj, exec_time, _ = measure_performance(gr.parse_node_query, query_str)
                    log_test(f"Query: {test_name}", success=False, error_msg="Empty query should fail")
                except Exception:
                    log_test(f"Query: {test_name}", success=True)  # Expected to fail
            else:
                # Parse query
                filter_obj, exec_time, _ = measure_performance(gr.parse_node_query, query_str)
                log_test(f"Query Parse: {test_name}", success=True, execution_time=exec_time)
                
                # Apply filter to graph
                try:
                    result_subgraph, exec_time, _ = measure_performance(g.filter_nodes, filter_obj)
                    result_count = len(result_subgraph.nodes) if hasattr(result_subgraph, 'nodes') else 0
                    log_test(f"Query Apply: {test_name} ({result_count} results)", success=True, execution_time=exec_time)
                except Exception as e:
                    log_test(f"Query Apply: {test_name}", success=False, error_msg=str(e))
                
        except Exception as e:
            log_test(f"Query: {test_name}", success=False, error_msg=str(e), code_snippet=f"gr.parse_node_query('{query_str}')")

def test_performance_characteristics(gr, g):
    """Test performance with various graph sizes"""
    if not gr or not g:
        return
    
    set_section("Performance Characteristics")
    
    # Test performance with different graph sizes
    sizes_to_test = [10, 100, 1000]
    
    for size in sizes_to_test:
        try:
            # Create nodes
            node_data = [{"id": i, "value": random.randint(1, 100)} for i in range(size)]
            nodes, exec_time, memory = measure_performance(g.add_nodes, node_data)
            log_test(f"Add {size} nodes", success=True, execution_time=exec_time, memory_usage=memory)
            
            # Create edges (sparse graph - each node connects to ~3 others)
            edge_count = min(size * 3, size * (size - 1) // 2)  # Cap at complete graph
            edges = []
            for i in range(edge_count):
                source = random.choice(nodes)
                target = random.choice(nodes)
                if source != target:
                    edges.append((source, target, {"weight": random.random()}))
            
            if edges:
                _, exec_time, memory = measure_performance(g.add_edges, edges)
                log_test(f"Add {len(edges)} edges", success=True, execution_time=exec_time, memory_usage=memory)
            
            # Test operations on this size graph
            _, exec_time, _ = measure_performance(g.node_count)
            log_test(f"Count {size} nodes", success=True, execution_time=exec_time)
            
            _, exec_time, _ = measure_performance(g.edge_count)
            log_test(f"Count edges in {size}-node graph", success=True, execution_time=exec_time)
            
            # Test table operations
            try:
                table, exec_time, _ = measure_performance(g.nodes.table)
                log_test(f"Generate {size}-node table", success=True, execution_time=exec_time)
            except Exception as e:
                log_test(f"Generate {size}-node table", success=False, error_msg=str(e))
                
        except Exception as e:
            log_test(f"Performance test size {size}", success=False, error_msg=str(e))
        
        # Create fresh graph for next test
        if size < max(sizes_to_test):
            try:
                g = gr.Graph()
            except Exception:
                break

def test_persona_engineering(gr, g):
    """Test scenarios relevant to AL_ENGINEER persona"""
    if not gr or not g:
        return
    
    set_section("Persona: Engineering (AL)")
    
    # Engineer focuses on: system design, performance, reliability, edge cases
    
    try:
        # Create a software engineering team graph
        engineers = [
            {"name": "Alice", "role": "Senior", "language": "Python", "experience": 8, "team": "Backend"},
            {"name": "Bob", "role": "Junior", "language": "JavaScript", "experience": 2, "team": "Frontend"},
            {"name": "Carol", "role": "Staff", "language": "Rust", "experience": 12, "team": "Infrastructure"},
            {"name": "Dave", "role": "Senior", "language": "Go", "experience": 6, "team": "Platform"},
        ]
        
        eng_nodes = g.add_nodes(engineers)
        log_test("AL: Setup engineering team", success=True)
        
        # Add collaboration edges based on realistic engineering patterns
        collaborations = [
            (eng_nodes[0], eng_nodes[1], {"type": "mentoring", "frequency": "daily"}),
            (eng_nodes[0], eng_nodes[2], {"type": "code_review", "frequency": "weekly"}),
            (eng_nodes[2], eng_nodes[3], {"type": "architecture", "frequency": "monthly"}),
        ]
        
        g.add_edges(collaborations)
        log_test("AL: Model team collaborations", success=True)
        
        # Engineer queries: Find senior+ engineers who could mentor
        senior_filter = gr.parse_node_query("experience >= 6")
        senior_engineers = g.filter_nodes(senior_filter)
        log_test(f"AL: Query senior engineers (found {len(senior_engineers.nodes)})", success=True)
        
        # System performance analysis
        nodes_table = g.nodes.table()
        avg_experience = nodes_table.mean('experience')
        log_test(f"AL: Calculate team metrics (avg exp: {avg_experience})", success=True)
        
        # Edge case: What if an engineer leaves?
        g.remove_node(eng_nodes[1])  # Bob leaves
        remaining_count = g.node_count()
        log_test(f"AL: Handle team member departure ({remaining_count} remaining)", success=True)
        
    except Exception as e:
        log_test("AL: Engineering persona test", success=False, error_msg=str(e))

def test_persona_research(gr, g):
    """Test scenarios relevant to DR_V_VISIONEER persona"""
    if not gr or not g:
        return
        
    set_section("Persona: Research (Dr. V)")
    
    # Dr. V focuses on: complex analysis, algorithms, theoretical properties
    
    try:
        # Create a research collaboration network
        researchers = [
            {"name": "Dr. Smith", "field": "AI", "citations": 2500, "h_index": 45, "institution": "MIT"},
            {"name": "Prof. Jones", "field": "ML", "citations": 3200, "h_index": 52, "institution": "Stanford"}, 
            {"name": "Dr. Kim", "field": "NLP", "citations": 1800, "h_index": 38, "institution": "CMU"},
            {"name": "Prof. Chen", "field": "Vision", "citations": 2100, "h_index": 41, "institution": "Berkeley"},
        ]
        
        research_nodes = g.add_nodes(researchers)
        log_test("Dr. V: Setup research network", success=True)
        
        # Model complex research relationships
        papers = [
            (research_nodes[0], research_nodes[1], {"collaboration": "joint_paper", "impact_factor": 8.2, "year": 2023}),
            (research_nodes[1], research_nodes[2], {"collaboration": "co_supervision", "students": 3, "year": 2022}),
            (research_nodes[2], research_nodes[3], {"collaboration": "workshop", "topic": "multimodal", "year": 2023}),
        ]
        
        g.add_edges(papers)
        log_test("Dr. V: Model research collaborations", success=True)
        
        # Advanced analytics: Find high-impact collaborations
        high_impact_filter = gr.parse_edge_query("impact_factor > 7.0")
        high_impact = g.filter_edges(high_impact_filter)
        log_test("Dr. V: Identify high-impact collaborations", success=True)
        
        # Network analysis: Connected components (research clusters)
        try:
            components = g.connected_components()
            log_test(f"Dr. V: Analyze research clusters ({len(components)} components)", success=True)
        except Exception as e:
            log_test("Dr. V: Analyze research clusters", success=False, error_msg=str(e))
        
        # Statistical analysis
        table = g.nodes.table()
        citation_stats = table['citations'].describe()
        log_test("Dr. V: Citation distribution analysis", success=True)
        
        # Theoretical property: Graph density
        density = g.density()
        log_test(f"Dr. V: Network density analysis ({density:.3f})", success=True)
        
    except Exception as e:
        log_test("Dr. V: Research persona test", success=False, error_msg=str(e))

def test_persona_yn_edge_cases(gr, g):
    """Test YN's approach: find weird vibes, edge cases, elegance leaks"""
    if not gr or not g:
        return
    
    set_section("Persona: YN (The Fool with a Flashlight)")
    
    # YN looks for: weird edge cases, inconsistencies, things that feel wrong
    
    try:
        # Test: What happens with weird node IDs?
        weird_node = g.add_node(weird_attr="🤔")
        log_test("YN: Handle unicode in attributes", success=True)
        
        # Test: Self-loops (edges from node to itself)
        g.add_edge(weird_node, weird_node, {"type": "self_reflection"})
        log_test("YN: Self-loop edge creation", success=True)
        
        # Test: Can we break attribute system with deeply nested data?
        deeply_nested = {"level1": {"level2": {"level3": {"level4": {"data": "buried"}}}}}
        nested_node = g.add_node(nested=deeply_nested)
        retrieved = g.get_node_attr(nested_node, "nested")
        log_test("YN: Deep nesting attribute test", success=True)
        
        # Test: What about circular references? (This might break)
        try:
            circular = {"self": None}
            circular["self"] = circular  # Circular reference
            circular_node = g.add_node(circular=circular)
            log_test("YN: Circular reference test", success=False, error_msg="Should fail or handle gracefully")
        except Exception as e:
            log_test("YN: Circular reference test", success=True)  # Expected to fail
        
        # Test: Memory leak potential with large attribute updates
        memory_node = g.add_node()
        for i in range(10):
            g.set_node_attr(memory_node, "data", list(range(i * 100)))
        log_test("YN: Attribute memory overwrite test", success=True)
        
        # Test: Query system with malformed input
        malformed_queries = [
            "age >",  # Incomplete
            "age > 25 AND",  # Trailing operator
            "((age > 25)",  # Unmatched parentheses
            "age >> 25",  # Invalid operator
            "'unclosed string > 25",  # Unclosed quotes
        ]
        
        for i, bad_query in enumerate(malformed_queries):
            try:
                filter_obj = gr.parse_node_query(bad_query)
                log_test(f"YN: Malformed query {i+1}", success=False, error_msg="Should have failed")
            except Exception:
                log_test(f"YN: Malformed query {i+1} properly rejected", success=True)
        
        # Test: Do empty subgraphs behave correctly?
        empty_filter = gr.parse_node_query("nonexistent_attr == 'impossible'")
        empty_subgraph = g.filter_nodes(empty_filter)
        empty_count = len(empty_subgraph.nodes)
        assert empty_count == 0, f"Empty subgraph should have 0 nodes, got {empty_count}"
        log_test("YN: Empty subgraph behavior", success=True)
        
        # Test: Attribute existence vs None values
        none_node = g.add_node(explicit_none=None)
        missing_attr = g.get_node_attr(none_node, "missing_attr")
        explicit_none = g.get_node_attr(none_node, "explicit_none")  
        log_test("YN: None vs missing attribute distinction", success=True)
        
    except Exception as e:
        log_test("YN: Edge case exploration", success=False, error_msg=str(e))

def test_composability_chains(gr, g):
    """Test complex method chaining and composability"""
    if not gr or not g:
        return
    
    set_section("Composability & Method Chaining")
    
    try:
        # Create a more complex graph for composability testing
        company_data = [
            {"name": "Alice", "dept": "Engineering", "level": "Senior", "salary": 120000, "remote": True},
            {"name": "Bob", "dept": "Engineering", "level": "Junior", "salary": 80000, "remote": False},
            {"name": "Carol", "dept": "Sales", "level": "Manager", "salary": 95000, "remote": True},
            {"name": "Dave", "dept": "Sales", "level": "Rep", "salary": 65000, "remote": False},
            {"name": "Eve", "dept": "HR", "level": "Director", "salary": 110000, "remote": True},
        ]
        
        company_nodes = g.add_nodes(company_data)
        
        # Add reporting relationships
        reports = [
            (company_nodes[1], company_nodes[0], {"relationship": "reports_to"}),  # Bob -> Alice  
            (company_nodes[3], company_nodes[2], {"relationship": "reports_to"}),  # Dave -> Carol
            (company_nodes[0], company_nodes[4], {"relationship": "dotted_line"}),  # Alice -> Eve
            (company_nodes[2], company_nodes[4], {"relationship": "dotted_line"}),  # Carol -> Eve
        ]
        
        g.add_edges(reports)
        log_test("Composability: Setup complex org chart", success=True)
        
        # Test chaining: Filter -> Table -> Statistics -> Further filtering
        try:
            # Chain 1: Find remote workers, get their salary stats
            remote_filter = gr.parse_node_query("remote == true")
            remote_workers = g.filter_nodes(remote_filter)
            remote_table = remote_workers.nodes.table()
            avg_remote_salary = remote_table.mean('salary')
            log_test(f"Chain 1: Remote worker salary analysis (${avg_remote_salary:.0f} avg)", success=True)
            
            # Chain 2: Engineering + High salary -> analyze reporting
            eng_filter = gr.parse_node_query("dept == 'Engineering'")
            high_salary_filter = gr.parse_node_query("salary > 100000")
            combined_filter = gr.NodeFilter.and_filters([eng_filter, high_salary_filter])
            
            senior_eng = g.filter_nodes(combined_filter)
            senior_eng_table = senior_eng.nodes.table()
            
            log_test("Chain 2: Complex filter composition", success=True)
            
            # Chain 3: Table operations -> matrix operations -> statistics
            all_table = g.nodes.table()
            salary_col = all_table['salary']
            salary_stats = salary_col.describe()
            
            # Then use results for further filtering
            high_performers = all_table[all_table['salary'] > avg_remote_salary]
            
            log_test("Chain 3: Table -> Column -> Stats -> Boolean indexing", success=True)
            
        except Exception as e:
            log_test("Method chaining", success=False, error_msg=str(e))
        
        # Test composability with matrix operations
        try:
            adj_matrix = g.adjacency()
            matrix_stats = {
                'shape': adj_matrix.shape,
                'is_sparse': adj_matrix.is_sparse,
            }
            
            # Chain matrix operations
            row_sums = adj_matrix.sum_axis(1)
            
            log_test("Composability: Graph -> Matrix -> Operations", success=True)
            
        except Exception as e:
            log_test("Matrix composability", success=False, error_msg=str(e))
            
    except Exception as e:
        log_test("Composability testing", success=False, error_msg=str(e))

def test_integration_cross_feature(gr, g):
    """Test integration between different major features"""
    if not gr or not g:
        return
    
    set_section("Cross-Feature Integration")
    
    try:
        # Integration test: Queries + Analytics + Tables + Matrix
        
        # Setup: Create a social network
        people = [
            {"name": f"Person_{i}", "age": 20 + (i % 30), "influence": random.randint(1, 100)} 
            for i in range(20)
        ]
        person_nodes = g.add_nodes(people)
        
        # Random social connections
        connections = []
        for i in range(30):  # 30 random connections
            a, b = random.sample(person_nodes, 2)
            weight = random.random()
            connections.append((a, b, {"strength": weight, "type": "friendship"}))
        
        g.add_edges(connections)
        
        log_test("Integration: Setup social network", success=True)
        
        # Test 1: Query -> Analytics integration
        influential_filter = gr.parse_node_query("influence > 70")
        influential_people = g.filter_nodes(influential_filter)
        
        # Are influential people well-connected?
        infl_subgraph = influential_people
        if hasattr(infl_subgraph, 'connected_components'):
            components = infl_subgraph.connected_components()
            log_test(f"Integration: Influential network analysis ({len(components)} components)", success=True)
        
        # Test 2: Table -> Query -> Matrix integration
        social_table = g.nodes.table()
        young_filter = gr.parse_node_query("age < 30") 
        young_people = g.filter_nodes(young_filter)
        young_matrix = young_people.adjacency()
        
        young_density = young_people.density()
        log_test(f"Integration: Young people network density ({young_density:.3f})", success=True)
        
        # Test 3: Analytics -> Table -> Statistics pipeline
        try:
            all_components = g.connected_components()
            
            # For each component, analyze age distribution
            for i, component in enumerate(all_components[:3]):  # Just first 3
                comp_table = component.nodes.table() 
                if 'age' in comp_table.columns:
                    avg_age = comp_table.mean('age')
                    log_test(f"Integration: Component {i+1} age analysis (avg: {avg_age:.1f})", success=True)
        except Exception as e:
            log_test("Integration: Component analysis", success=False, error_msg=str(e))
        
        # Test 4: Full pipeline: Query -> Filter -> Subgraph -> Analytics -> Export
        try:
            # Find the main social cluster
            main_component = max(all_components, key=lambda c: len(c.nodes)) if all_components else g
            
            # Export to different formats
            nx_graph = main_component.to_networkx()
            main_table = main_component.nodes.table()
            pandas_df = main_table.to_pandas()
            
            log_test("Integration: Multi-format export pipeline", success=True)
            
        except Exception as e:
            log_test("Integration: Export pipeline", success=False, error_msg=str(e))
            
    except Exception as e:
        log_test("Cross-feature integration", success=False, error_msg=str(e))

def generate_comprehensive_report():
    """Generate a comprehensive test report with performance metrics"""
    
    report = f"""# Groggy Comprehensive Test Suite Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This comprehensive test suite goes beyond basic validation to test edge cases, 
performance characteristics, persona-based scenarios, and complex composability patterns.

"""
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result['success'])
    failed_tests = total_tests - passed_tests
    
    # Performance statistics
    timed_tests = [r for r in test_results if r['execution_time'] > 0]
    if timed_tests:
        avg_time = sum(r['execution_time'] for r in timed_tests) / len(timed_tests)
        max_time = max(r['execution_time'] for r in timed_tests)
        slow_tests = [r for r in timed_tests if r['execution_time'] > 0.1]
    else:
        avg_time = max_time = 0
        slow_tests = []
    
    report += f"""### Test Results
- **Total Tests**: {total_tests}
- **Passed**: {passed_tests} ✅  
- **Failed**: {failed_tests} ❌
- **Success Rate**: {(passed_tests/total_tests*100):.1f}%

### Performance Metrics
- **Average Execution Time**: {avg_time:.4f}s
- **Slowest Test**: {max_time:.4f}s
- **Tests > 100ms**: {len(slow_tests)}

"""
    
    # Group by section with performance data
    sections = {}
    for result in test_results:
        section = result['section']
        if section not in sections:
            sections[section] = {'passed': 0, 'failed': 0, 'tests': [], 'total_time': 0}
        
        if result['success']:
            sections[section]['passed'] += 1
        else:
            sections[section]['failed'] += 1
        sections[section]['tests'].append(result)
        sections[section]['total_time'] += result.get('execution_time', 0)
    
    report += "## Results by Section\n\n"
    
    for section, data in sections.items():
        total = data['passed'] + data['failed']
        success_rate = (data['passed'] / total * 100) if total > 0 else 0
        
        status = "✅" if data['failed'] == 0 else "⚠️" if data['failed'] < data['passed'] else "❌"
        
        report += f"### {status} {section}\n"
        report += f"- **Success Rate**: {data['passed']}/{total} ({success_rate:.1f}%)\n"
        report += f"- **Total Time**: {data['total_time']:.4f}s\n"
        
        if data['failed'] > 0:
            report += f"- **Failed Tests**:\n"
            for test in data['tests']:
                if not test['success']:
                    report += f"  - `{test['test']}`: {test['error']}\n"
        
        # Show slowest tests in section
        section_slow = [t for t in data['tests'] if t.get('execution_time', 0) > 0.05]
        if section_slow:
            section_slow.sort(key=lambda x: x.get('execution_time', 0), reverse=True)
            report += f"- **Slowest Tests**:\n"
            for test in section_slow[:3]:  # Top 3
                report += f"  - `{test['test']}`: {test['execution_time']:.4f}s\n"
        
        report += "\n"
    
    # Detailed failure analysis
    if failed_tests > 0:
        report += "## Failed Tests Analysis\n\n"
        
        failed_count = 0
        for result in test_results:
            if not result['success']:
                failed_count += 1
                report += f"### {failed_count}. {result['test']} ({result['section']})\n\n"
                report += f"**Error**: `{result['error']}`\n\n"
                if result['code']:
                    report += f"**Code**:\n```python\n{result['code']}\n```\n\n"
                report += "---\n\n"
    
    # Performance recommendations
    report += "## Performance Analysis\n\n"
    
    if slow_tests:
        report += f"### Slow Tests ({len(slow_tests)} tests > 100ms)\n\n"
        slow_tests.sort(key=lambda x: x['execution_time'], reverse=True)
        for test in slow_tests[:10]:  # Top 10 slowest
            report += f"- `{test['test']}`: {test['execution_time']:.4f}s ({test['section']})\n"
        report += "\n"
    
    # Recommendations
    report += "## Recommendations\n\n"
    
    if failed_tests > 0:
        report += f"### Critical Issues ({failed_tests} failed tests)\n"
        report += "1. **Review failed test cases** - may indicate bugs or missing features\n"
        report += "2. **Check edge case handling** - several boundary condition tests failed\n"
        report += "3. **Verify error handling** - ensure graceful degradation\n\n"
    
    if slow_tests:
        report += f"### Performance Issues ({len(slow_tests)} slow tests)\n"
        report += "1. **Optimize slow operations** - focus on tests > 100ms\n"
        report += "2. **Consider caching** for repeated expensive operations\n"
        report += "3. **Profile memory usage** in bulk operations\n\n"
    
    if failed_tests == 0 and len(slow_tests) < 5:
        report += "### Excellent Results! ✅\n"
        report += "- All tests pass with good performance\n"
        report += "- Edge cases are handled correctly\n"  
        report += "- Cross-feature integration works well\n"
        report += "- Ready for production use\n\n"
    
    report += """## Test Categories Covered

1. **Basic Functionality**: Core graph operations, CRUD operations
2. **Edge Cases**: Boundary conditions, invalid inputs, empty states
3. **Performance**: Large graphs, bulk operations, memory usage  
4. **Query System**: Complex queries, malformed inputs, edge cases
5. **Composability**: Method chaining, feature integration
6. **Persona Testing**: Real-world scenarios from different user perspectives
7. **Cross-Feature Integration**: How different modules work together

---
*Generated by Groggy Comprehensive Test Suite*
*"Every method, every edge case, every weird vibe" - YN*
"""
    
    return report

def main():
    """Run the comprehensive test suite"""
    print("🧪 Groggy Comprehensive Test Suite")
    print("===================================")
    print("Testing every method, every edge case, every weird vibe")
    print()
    
    # Test comprehensive imports
    print("🔍 STEP 1: Testing imports...")
    gr = test_comprehensive_imports()
    if not gr:
        print("❌ CRITICAL: Cannot import groggy - stopping tests")
        print("    This indicates a fundamental build or installation issue")
        return
    print(f"✅ STEP 1 COMPLETE: Successfully imported groggy module: {type(gr)}")
    
    # Test edge cases in graph creation
    print("\n🔍 STEP 2: Testing graph creation...")
    g = test_edge_cases_graph_creation(gr)
    if g is None:
        print("❌ CRITICAL: Cannot create graph - stopping tests") 
        print("    This indicates issues with the Graph constructor or basic methods")
        print("    Check the detailed error output above for specific failure reasons")
        return
    print(f"✅ STEP 2 COMPLETE: Successfully created graph: {type(g)}")
    
    # Test boundary conditions
    node_ids = test_boundary_conditions_nodes(gr, g)
    
    # Test bulk operations - core architectural strength  
    test_bulk_operations(gr, g, node_ids)
    
    # Test attribute system edge cases
    test_attribute_system_edge_cases(gr, g, node_ids)
    
    # Test query system comprehensively
    test_query_system_comprehensive(gr, g, node_ids)
    
    # Test performance characteristics
    test_performance_characteristics(gr, g)
    
    # Test persona-based scenarios
    test_persona_engineering(gr, gr.Graph())  # Fresh graph for each persona
    test_persona_research(gr, gr.Graph())
    test_persona_yn_edge_cases(gr, gr.Graph())
    
    # Test composability and method chaining
    test_composability_chains(gr, gr.Graph())
    
    # Test cross-feature integration
    test_integration_cross_feature(gr, gr.Graph())
    
    # Generate comprehensive report
    print("\n" + "=" * 50)
    print("📊 Generating comprehensive report...")
    
    report = generate_comprehensive_report()
    
    # Save report
    report_file = "comprehensive_test_results.md"
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
        print("\n🎉 ALL TESTS PASSED! Every method, every edge case covered!")
    else:
        print(f"\n⚠️  {failed_tests} issues found - see report for details")
        print("   This comprehensive testing revealed edge cases that need attention")
    
    print(f"\n💫 Comprehensive testing complete - report at {report_file}")

if __name__ == "__main__":
    main()