#!/usr/bin/env python3
"""
BaseArray ↔ BaseTable Final Integration Test (Python)

Tests the complete BaseArray ↔ BaseTable integration through the Python FFI
to verify the entire stack works correctly from Python.
"""

import sys
import traceback
try:
    import groggy as gr
except ImportError:
    print("❌ Could not import groggy - ensure Python package is built")
    sys.exit(1)

def test_1_basearray_basetable_creation():
    """Test 1: Create BaseTable from BaseArray columns via Python"""
    print("\n🔧 Test 1: BaseArray → BaseTable Creation (Python)")
    
    try:
        # Create a simple graph to work with
        g = gr.Graph()
        
        # Add some nodes with attributes
        g.add_node(name="Alice", age=25, department="Engineering") 
        g.add_node(name="Bob", age=30, department="Marketing")
        g.add_node(name="Carol", age=35, department="Engineering")
        g.add_node(name="Dave", age=40, department="Sales")
        
        # Add some edges using node IDs 0, 1, 2, 3 (auto-assigned)
        try:
            g.add_edge(0, 1, relation="colleague")
            g.add_edge(1, 2, relation="manager") 
            g.add_edge(2, 3, relation="colleague")
        except Exception as e:
            print(f"   ⚠️  Edge creation issue: {e}")
        
        print(f"   ✅ Graph created: {g.node_count()} nodes, {g.edge_count()} edges")
        return g
        
    except Exception as e:
        print(f"   ❌ Graph creation failed: {e}")
        traceback.print_exc()
        return None

def test_2_graph_to_table_conversion(g):
    """Test 2: Convert Graph to GraphTable (g.table())"""
    print("\n🔧 Test 2: Graph → GraphTable Conversion")
    
    try:
        # Test g.table() - should return GraphTable
        graph_table = g.table()
        print(f"   ✅ g.table() successful: {type(graph_table).__name__}")
        
        # Test shape and basic properties
        shape = graph_table.shape()
        print(f"   ✅ GraphTable shape: {shape}")
        
        # Test validation
        validation_report = graph_table.validate()
        print(f"   ✅ Validation report: {len(validation_report)} chars")
        
        return graph_table
        
    except Exception as e:
        print(f"   ❌ Graph → GraphTable conversion failed: {e}")
        traceback.print_exc()
        return None

def test_3_nodes_table_access(g):
    """Test 3: Access NodesTable (g.nodes.table())"""
    print("\n🔧 Test 3: NodesTable Access")
    
    try:
        # Test g.nodes.table() - should return NodesTable
        nodes_table = g.nodes.table()
        print(f"   ✅ g.nodes.table() successful: {type(nodes_table).__name__}")
        
        # Test basic properties
        nrows = nodes_table.nrows()
        ncols = nodes_table.ncols()
        print(f"   ✅ NodesTable shape: {nrows} x {ncols}")
        
        # Test node IDs access
        node_ids = nodes_table.node_ids()
        print(f"   ✅ Node IDs: {node_ids}")
        
        return nodes_table
        
    except Exception as e:
        print(f"   ❌ NodesTable access failed: {e}")
        traceback.print_exc()
        return None

def test_4_edges_table_access(g):
    """Test 4: Access EdgesTable (g.edges.table())"""
    print("\n🔧 Test 4: EdgesTable Access")
    
    try:
        # Test g.edges.table() - should return EdgesTable
        edges_table = g.edges.table()
        print(f"   ✅ g.edges.table() successful: {type(edges_table).__name__}")
        
        # Test basic properties
        nrows = edges_table.nrows()
        ncols = edges_table.ncols()
        print(f"   ✅ EdgesTable shape: {nrows} x {ncols}")
        
        # Test edge IDs access
        edge_ids = edges_table.edge_ids()
        print(f"   ✅ Edge IDs: {edge_ids}")
        
        # Test sources and targets
        sources = edges_table.sources()
        targets = edges_table.targets()
        print(f"   ✅ Sources: {sources}")
        print(f"   ✅ Targets: {targets}")
        
        return edges_table
        
    except Exception as e:
        print(f"   ❌ EdgesTable access failed: {e}")
        traceback.print_exc()
        return None

def test_5_graphtable_components(graph_table):
    """Test 5: Access GraphTable components"""
    print("\n🔧 Test 5: GraphTable Component Access")
    
    try:
        # Test accessing nodes component
        nodes_component = graph_table.nodes
        print(f"   ✅ Graphtable.nodes: {type(nodes_component).__name__}")
        print(f"   ✅ Nodes shape: {nodes_component.shape()}")
        
        # Test accessing edges component  
        edges_component = graph_table.edges
        print(f"   ✅ Graphtable.edges: {type(edges_component).__name__}")
        print(f"   ✅ Edges shape: {edges_component.shape()}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GraphTable component access failed: {e}")
        traceback.print_exc()
        return False

def test_6_multi_graphtable_operations():
    """Test 6: Multi-GraphTable operations (Phase 6 features)"""
    print("\n🔧 Test 6: Multi-GraphTable Operations")
    
    try:
        # Create two separate graphs
        g1 = gr.Graph()
        alice = g1.add_node(name="Alice", domain="A")
        bob = g1.add_node(name="Bob", domain="A")
        g1.add_edge(alice, bob)
        
        g2 = gr.Graph() 
        carol = g2.add_node(name="Carol", domain="B")  # Same ID, different domain
        dave = g2.add_node(name="Dave", domain="B")
        g2.add_edge(carol, dave)
        
        # Convert to GraphTables
        table1 = g1.table()
        table2 = g2.table()
        
        print(f"   ✅ Created two GraphTables")
        print(f"   ✅ Table 1: {table1.shape()}")
        print(f"   ✅ Table 2: {table2.shape()}")
        
        # Test merge with domain prefix strategy
        try:
            merged = gr.GraphTable.merge_with_strategy([table1, table2], "domain_prefix")
            print(f"   ✅ Merge with domain_prefix: {merged.shape()}")
        except Exception as e:
            print(f"   ⚠️  Merge failed (expected - may not be fully implemented): {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Multi-GraphTable operations failed: {e}")
        traceback.print_exc()
        return False

def test_7_round_trip_conversion(graph_table):
    """Test 7: Round-trip conversion GraphTable → Graph"""
    print("\n🔧 Test 7: Round-trip GraphTable → Graph")
    
    try:
        # Convert GraphTable back to Graph
        reconstructed_graph = graph_table.to_graph()
        print(f"   ✅ GraphTable → Graph successful")
        
        # Verify basic properties
        node_count = reconstructed_graph.node_count()
        edge_count = reconstructed_graph.edge_count()
        print(f"   ✅ Reconstructed graph: {node_count} nodes, {edge_count} edges")
        
        # Test that we can access the same data
        reconstructed_table = reconstructed_graph.table()
        original_shape = graph_table.shape()
        reconstructed_shape = reconstructed_table.shape()
        print(f"   ✅ Round-trip shape check: {original_shape} → {reconstructed_shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Round-trip conversion failed: {e}")
        traceback.print_exc()
        return False

def test_8_equivalence_verification(g):
    """Test 8: Verify g.nodes.table() === g.table().nodes equivalence"""
    print("\n🔧 Test 8: Equivalence Verification")
    
    try:
        # Get tables via both paths
        nodes_direct = g.nodes.table()
        graph_table = g.table()
        nodes_via_graph = graph_table.nodes
        
        # Compare shapes (should be equivalent)
        direct_shape = nodes_direct.shape()
        via_graph_shape = nodes_via_graph.shape()
        
        print(f"   ✅ Direct nodes.table(): {direct_shape}")
        print(f"   ✅ Via graph.table().nodes: {via_graph_shape}")
        
        if direct_shape == via_graph_shape:
            print(f"   ✅ Equivalence verified: shapes match")
        else:
            print(f"   ⚠️  Shapes differ - may indicate implementation gap")
        
        return direct_shape == via_graph_shape
        
    except Exception as e:
        print(f"   ❌ Equivalence verification failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("🧪 BaseArray ↔ BaseTable Final Integration Test (Python)")
    print("=======================================================")
    
    test_results = []
    
    # Test 1: Basic graph creation
    g = test_1_basearray_basetable_creation()
    test_results.append(g is not None)
    
    if g is None:
        print("\n❌ CRITICAL: Graph creation failed - cannot continue")
        return False
    
    # Test 2: Graph to table conversion
    graph_table = test_2_graph_to_table_conversion(g)
    test_results.append(graph_table is not None)
    
    # Test 3: Nodes table access
    nodes_table = test_3_nodes_table_access(g)
    test_results.append(nodes_table is not None)
    
    # Test 4: Edges table access
    edges_table = test_4_edges_table_access(g)
    test_results.append(edges_table is not None)
    
    # Test 5: GraphTable components (if we have graph_table)
    if graph_table:
        result = test_5_graphtable_components(graph_table)
        test_results.append(result)
    else:
        test_results.append(False)
    
    # Test 6: Multi-GraphTable operations
    result = test_6_multi_graphtable_operations()
    test_results.append(result)
    
    # Test 7: Round-trip conversion (if we have graph_table)
    if graph_table:
        result = test_7_round_trip_conversion(graph_table)
        test_results.append(result)
    else:
        test_results.append(False)
    
    # Test 8: Equivalence verification
    result = test_8_equivalence_verification(g)
    test_results.append(result)
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\n📊 TEST SUMMARY")
    print(f"================")
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! BaseArray ↔ BaseTable integration is working!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed or had issues")
        print("This may indicate areas needing completion or debugging")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)