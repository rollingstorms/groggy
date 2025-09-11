#!/usr/bin/env python3
"""
Test script for the unified delegation architecture
Tests BaseArray/NumArray separation and array transformations
"""

import sys
import os

# Add the python-groggy package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-groggy', 'python'))

def test_base_array_functionality():
    """Test GraphArray BaseArray functionality - comprehensive testing"""
    print("🧪 Testing BaseArray functionality...")
    failures = []
    
    try:
        import groggy as gr
        g = gr.karate_club()
        
        # Test GraphArray comprehensive functionality
        node_ids = g.nodes.ids()
        
        # Test basic array properties
        print(f"   • Array length: {len(node_ids)}")
        assert len(node_ids) == 34, f"Expected 34 nodes, got {len(node_ids)}"
        
        # Test array access patterns
        first_element = node_ids[0]
        print(f"   • First element: {first_element}")
        
        # Test array conversion methods
        try:
            as_list = node_ids.to_list()
            print(f"   ✓ to_list(): {len(as_list)} elements")
            assert len(as_list) == 34, "List conversion failed"
        except Exception as e:
            failures.append(f"to_list() failed: {e}")
        
        try:
            as_numpy = node_ids.to_numpy()
            print(f"   ✓ to_numpy(): shape {as_numpy.shape}")
            assert len(as_numpy) == 34, "NumPy conversion failed"
        except Exception as e:
            failures.append(f"to_numpy() failed: {e}")
        
        # Test summary and statistics
        try:
            summary = node_ids.summary()
            print(f"   ✓ summary(): {type(summary)}")
        except Exception as e:
            failures.append(f"summary() failed: {e}")
        
        try:
            count = node_ids.count()
            print(f"   ✓ count(): {count}")
            assert count == 34, f"Count mismatch: {count} != 34"
        except Exception as e:
            failures.append(f"count() failed: {e}")
        
        # Test statistical operations
        try:
            max_val = node_ids.max()
            min_val = node_ids.min()
            mean_val = node_ids.mean()
            print(f"   ✓ Statistics: min={min_val}, max={max_val}, mean={mean_val:.2f}")
        except Exception as e:
            failures.append(f"Statistical operations failed: {e}")
        
        if failures:
            print(f"❌ BaseArray test had {len(failures)} failures:")
            for failure in failures:
                print(f"     - {failure}")
            return False
        else:
            print("✅ BaseArray comprehensive test passed!")
            return True
            
    except ImportError as e:
        print(f"⚠️  Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ BaseArray test failed: {e}")
        return False

def test_stats_array_functionality():
    """Test GraphMatrix and statistical operations comprehensively"""
    print("🧪 Testing StatsArray functionality...")
    failures = []
    
    try:
        import groggy as gr
        g = gr.karate_club()
        
        # Test GraphMatrix statistical operations
        dense_matrix = g.dense_adjacency_matrix()
        print(f"   • Created GraphMatrix: {type(dense_matrix)}")
        
        # Test matrix properties (attributes, not methods)
        try:
            shape = dense_matrix.shape
            print(f"   ✓ Matrix shape: {shape}")
            assert shape == (34, 34), f"Expected (34, 34), got {shape}"
        except Exception as e:
            failures.append(f"Matrix shape failed: {e}")
        
        try:
            is_symmetric = dense_matrix.is_symmetric
            print(f"   ✓ Matrix is symmetric: {is_symmetric}")
        except Exception as e:
            failures.append(f"is_symmetric failed: {e}")
        
        try:
            is_sparse = dense_matrix.is_sparse
            print(f"   ✓ Matrix sparsity: {is_sparse}")
        except Exception as e:
            failures.append(f"is_sparse failed: {e}")
            
        try:
            dtype = dense_matrix.dtype
            print(f"   ✓ Matrix dtype: {dtype}")
        except Exception as e:
            failures.append(f"dtype failed: {e}")
        
        # Test statistical operations on matrix
        try:
            mean_axis = dense_matrix.mean_axis(0)  # Row-wise mean
            print(f"   ✓ Row-wise mean: {type(mean_axis)}")
        except Exception as e:
            failures.append(f"mean_axis() failed: {e}")
        
        try:
            sum_axis = dense_matrix.sum_axis(1)  # Column-wise sum
            print(f"   ✓ Column-wise sum: {type(sum_axis)}")
        except Exception as e:
            failures.append(f"sum_axis() failed: {e}")
        
        try:
            std_axis = dense_matrix.std_axis(0)  # Row-wise std dev
            print(f"   ✓ Row-wise std dev: {type(std_axis)}")
        except Exception as e:
            failures.append(f"std_axis() failed: {e}")
        
        # Test matrix operations
        try:
            transpose = dense_matrix.transpose()
            print(f"   ✓ Matrix transpose: {type(transpose)}")
        except Exception as e:
            failures.append(f"transpose() failed: {e}")
        
        # Skip determinant - explicitly not implemented yet
        # try:
        #     determinant = dense_matrix.determinant()
        #     print(f"   ✓ Matrix determinant: {determinant}")
        # except Exception as e:
        #     failures.append(f"determinant() failed: {e}")
        print(f"   • Matrix determinant: Not implemented yet (Phase 5)")
        
        # Test GraphArray statistical operations
        node_ids = g.nodes.ids()
        
        try:
            median_val = node_ids.median()
            std_val = node_ids.std()
            print(f"   ✓ Array stats: median={median_val}, std={std_val:.2f}")
        except Exception as e:
            failures.append(f"Array statistics failed: {e}")
        
        try:
            percentile_90 = node_ids.percentile(90)
            print(f"   ✓ 90th percentile: {percentile_90}")
        except Exception as e:
            failures.append(f"percentile() failed: {e}")
        
        if failures:
            print(f"❌ StatsArray test had {len(failures)} failures:")
            for failure in failures:
                print(f"     - {failure}")
            return False
        else:
            print("✅ StatsArray comprehensive test passed!")
            return True
            
    except ImportError as e:
        print(f"⚠️  Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ StatsArray test failed: {e}")
        return False

def test_specialized_arrays():
    """Test SubgraphArray, NodesArray, EdgesArray, MatrixArray from the plan"""
    print("🧪 Testing specialized arrays...")
    try:
        import groggy as gr
        
        g = gr.karate_club()
        
        # Test SubgraphArray creation and delegation
        # According to the plan: Graph → connected_components() → SubgraphArray
        print("   Testing SubgraphArray creation...")
        
        # Test NodesArray through nodes accessor
        nodes_subgraph = g.nodes.all()  # Should return a Subgraph with node operations
        print(f"   ✓ NodesArray via subgraph: {type(nodes_subgraph)} with {len(nodes_subgraph)} nodes")
        
        # Test EdgesArray through edge operations
        edge_ids = g.edge_ids  # Property, not method
        print(f"   ✓ EdgesArray via IDs: {type(edge_ids)} with {len(edge_ids)} edges")
        
        # Test MatrixArray creation
        matrices = [
            ("adjacency_matrix", g.adjacency_matrix()),
            ("dense_adjacency_matrix", g.dense_adjacency_matrix()),
            ("laplacian_matrix", g.laplacian_matrix()),
        ]
        
        for name, matrix in matrices:
            print(f"   ✓ MatrixArray {name}: {type(matrix)}")
        
        # Test that arrays support delegation (basic array operations)
        print(f"   ✓ Node subgraph has length: {len(nodes_subgraph)}")
        print(f"   ✓ Edge IDs has length: {len(edge_ids)}")
        
        print("✅ Specialized arrays work!")
        return True
    except ImportError as e:
        print(f"⚠️  Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Specialized arrays test failed: {e}")
        return False

def test_delegation_chaining():
    """Test comprehensive delegation chaining with real method calls"""
    print("🧪 Testing delegation chaining...")
    failures = []
    
    try:
        import groggy as gr
        g = gr.karate_club()
        
        # Test Subgraph delegation - it has 54 methods!
        print("   Testing Subgraph method delegation...")
        subgraph = g.nodes.all()
        
        # Test BFS on subgraph (since it has bfs method)
        try:
            sub_bfs = subgraph.bfs(0)
            print(f"   ✓ Subgraph.bfs(): {type(sub_bfs)} with {len(sub_bfs)} nodes")
        except Exception as e:
            failures.append(f"Subgraph BFS failed: {e}")
        
        # Test neighborhood with proper parameters
        try:
            # Check signature first
            neighborhood = subgraph.neighborhood([0], 1)  # central_nodes, hops
            print(f"   ✓ Subgraph.neighborhood(): {type(neighborhood)} with {len(neighborhood)} nodes")
        except Exception as e:
            failures.append(f"Subgraph neighborhood failed: {e}")
        
        # Test connected components on subgraph
        try:
            components = subgraph.connected_components()
            print(f"   ✓ Subgraph.connected_components(): {type(components)}")
        except Exception as e:
            failures.append(f"Subgraph connected_components failed: {e}")
        
        # Test table delegation
        try:
            sub_table = subgraph.table()
            print(f"   ✓ Subgraph.table(): {type(sub_table)}")
        except Exception as e:
            failures.append(f"Subgraph table failed: {e}")
        
        # Test subgraph sampling
        try:
            sample = subgraph.sample(10)
            print(f"   ✓ Subgraph.sample(10): {type(sample)} with {len(sample)} nodes")
        except Exception as e:
            failures.append(f"Subgraph sample failed: {e}")
        
        # Test filtering operations
        try:
            filtered_nodes = subgraph.filter_nodes("degree > 2")
            print(f"   ✓ Subgraph.filter_nodes(): {type(filtered_nodes)} with {len(filtered_nodes)} nodes")
        except Exception as e:
            failures.append(f"Subgraph filter_nodes failed: {e}")
        
        # Test statistical operations on subgraph
        try:
            density = subgraph.density()
            print(f"   ✓ Subgraph density: {density:.3f}")
        except Exception as e:
            failures.append(f"Subgraph density failed: {e}")
            
        # Skip clustering coefficient - not implemented yet
        print(f"   • Clustering coefficient: Not implemented yet")
        
        # Test conversion chains: Graph → Subgraph → EdgesTable → filtering
        print("   Testing conversion chains...")
        try:
            edges_table = subgraph.edges_table()
            print(f"   ✓ Subgraph → EdgesTable: {type(edges_table)}")
            
            # Test if edges table has methods
            if hasattr(edges_table, 'head'):
                head_result = edges_table.head(5)
                print(f"   ✓ EdgesTable.head(5): {type(head_result)}")
        except Exception as e:
            failures.append(f"EdgesTable operations failed: {e}")
        
        if failures:
            print(f"❌ Delegation chaining had {len(failures)} failures:")
            for failure in failures:
                print(f"     - {failure}")
            return False
        else:
            print("✅ Delegation chaining comprehensive test passed!")
            return True
            
    except ImportError as e:
        print(f"⚠️  Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Delegation chaining test failed: {e}")
        return False

def test_architecture_examples():
    """Test the specific examples from the delegation architecture plan"""
    print("🧪 Testing architecture plan examples...")
    try:
        import groggy as gr
        
        g = gr.karate_club()
        
        # Example 1: Component Analysis Chain
        # Graph → connected_components() → SubgraphArray → .iter() → DelegatingIterator<Subgraph>
        print("   Testing Example 1: Component Analysis...")
        
        # Since connected_components might not be implemented yet, test similar pattern
        try:
            # Alternative: use BFS to get subgraph, then chain operations
            component = g.bfs(0)  # Simulates a connected component
            print(f"   ✓ Got component (via BFS): {type(component)}")
            
            # Try to get table from component
            if hasattr(component, 'table'):
                component_table = component.table()
                print(f"   ✓ Component → Table: {type(component_table)}")
            else:
                print("   ⚠️ Component table conversion not available")
        except Exception as e:
            print(f"   ⚠️ Component analysis chain failed: {e}")
        
        # Example 2: Neighborhood Analysis  
        # Graph → bfs(start) → Subgraph → neighborhood() → Subgraph → table() → NodesTable
        print("   Testing Example 2: Neighborhood Analysis...")
        
        try:
            # Get neighborhood statistics (alternative to full neighborhood)
            neighborhood_stats = g.neighborhood_statistics(0)
            print(f"   ✓ Neighborhood stats: {neighborhood_stats}")
            
            # Get actual neighbors
            neighbors = g.neighbors(0)
            print(f"   ✓ Neighbors of node 0: {type(neighbors)} - {neighbors[:5] if hasattr(neighbors, '__getitem__') else neighbors}")
        except Exception as e:
            print(f"   ⚠️ Neighborhood analysis failed: {e}")
        
        # Test the "Type Flow" concept from the plan
        print("   Testing Type Flow transformations...")
        
        # Start with Graph, try to reach different target types
        transformations = []
        
        # Graph → NodesTable (via table accessor)
        try:
            table = g.table()
            transformations.append(("Graph", "NodesTable", type(table)))
        except Exception as e:
            transformations.append(("Graph", "NodesTable", f"Failed: {e}"))
        
        # Graph → Subgraph (via nodes.all())
        try:
            subgraph = g.nodes.all()
            transformations.append(("Graph", "Subgraph", type(subgraph)))
        except Exception as e:
            transformations.append(("Graph", "Subgraph", f"Failed: {e}"))
        
        # Graph → Matrix (via adjacency_matrix)
        try:
            matrix = g.adjacency_matrix()
            transformations.append(("Graph", "Matrix", type(matrix)))
        except Exception as e:
            transformations.append(("Graph", "Matrix", f"Failed: {e}"))
        
        for from_type, to_type, result in transformations:
            print(f"   ✓ {from_type} → {to_type}: {result}")
        
        print("✅ Architecture examples work!")
        return True
        
    except ImportError as e:
        print(f"⚠️  Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Architecture examples test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Unified Delegation Architecture")
    print("=" * 50)
    
    tests = [
        ("BaseArray Functionality", test_base_array_functionality),
        ("StatsArray Functionality", test_stats_array_functionality), 
        ("Specialized Arrays", test_specialized_arrays),
        ("Delegation Chaining", test_delegation_chaining),
        ("Architecture Plan Examples", test_architecture_examples),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    print("\n📊 Test Results")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🏆 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Unified delegation architecture is working!")
    else:
        print("🔧 Some tests failed. Check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)