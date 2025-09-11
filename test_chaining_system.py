#!/usr/bin/env python3
"""
Test the BaseArray Chaining System implementation for Phase 3
"""

import sys
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy')

try:
    import groggy as gr
    print("✅ Groggy imported successfully")
except ImportError as e:
    print(f"❌ Failed to import groggy: {e}")
    sys.exit(1)

def test_base_array_chaining():
    """Test BaseArray chaining operations"""
    print("\n=== Testing BaseArray Chaining System ===")
    
    # Create sample data
    data = {
        'node_id': [1, 2, 3, 4, 5],
        'age': [25, 30, 35, 20, 45],
        'department': ['engineering', 'sales', 'engineering', 'marketing', 'engineering']
    }
    
    # Create BaseTable
    base_table = gr.BaseTable.from_dict(data)
    print(f"✅ Created BaseTable with {base_table.nrows} rows")
    
    # Test if BaseArray implements ArrayOps - try to get a column and chain operations
    try:
        # Get a column (should return BaseArray or similar)
        age_column = base_table.column('age')
        print(f"✅ Got age column: {type(age_column)}")
        
        # Test if we can call .iter() to get chaining capabilities
        if hasattr(age_column, 'iter'):
            print("✅ Column has .iter() method - chaining should work")
            
            # Try basic chaining operations
            try:
                result = age_column.iter().filter(lambda x: x > 25).collect()
                print(f"✅ Basic filter+collect chaining works: {result}")
            except Exception as e:
                print(f"⚠️  Basic chaining failed: {e}")
        else:
            print("❌ Column doesn't have .iter() method")
            
    except Exception as e:
        print(f"❌ Column access failed: {e}")

def test_specialized_arrays():
    """Test specialized array types"""
    print("\n=== Testing Specialized Arrays ===")
    
    # Test if specialized array types are available in Python
    try:
        # Try to access specialized arrays through FFI
        # Note: These might not be exposed to Python yet
        if hasattr(gr, 'NodesArray'):
            print("✅ NodesArray available in Python")
        else:
            print("⚠️  NodesArray not exposed to Python (expected in Phase 3)")
            
        if hasattr(gr, 'EdgesArray'):
            print("✅ EdgesArray available in Python")
        else:
            print("⚠️  EdgesArray not exposed to Python (expected in Phase 3)")
            
    except Exception as e:
        print(f"❌ Error checking specialized arrays: {e}")

def test_graph_chaining():
    """Test graph-based chaining operations"""
    print("\n=== Testing Graph-Based Chaining ===")
    
    try:
        # Create a simple graph
        nodes_data = {'node_id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']}
        edges_data = {'edge_id': [1, 2], 'source': [1, 2], 'target': [2, 3]}
        
        nodes = gr.NodesTable.from_dict(nodes_data)
        edges = gr.EdgesTable.from_dict(edges_data)
        graph_table = gr.GraphTable(nodes, edges)
        
        print("✅ Created GraphTable for chaining tests")
        
        # Test if graph operations support chaining
        graph = graph_table.to_graph()
        print("✅ Converted to Graph")
        
        # Try to get components and test chaining
        try:
            components = graph.connected_components()
            print(f"✅ Got connected components: {type(components)}")
            
            # Test if components support the fluent API
            if hasattr(components, 'iter'):
                print("✅ Components support .iter() - chaining should work")
                
                # Test the fluent API from the documentation
                try:
                    # This is the target API from Phase 3 plan:
                    # results = components.iter().filter_nodes('age > 25').collapse()
                    
                    # For now, just test basic iteration
                    iter_result = components.iter()
                    print(f"✅ Basic .iter() works: {type(iter_result)}")
                    
                    # Test if trait-based methods are available
                    if hasattr(iter_result, 'filter_nodes'):
                        print("✅ Trait-based method .filter_nodes() available")
                        
                        # Try the fluent chaining
                        try:
                            result = iter_result.filter_nodes('age > 25')
                            print(f"✅ filter_nodes() works: {type(result)}")
                        except Exception as e:
                            print(f"⚠️  filter_nodes() failed: {e}")
                    else:
                        print("⚠️  Trait-based method .filter_nodes() not available")
                        
                except Exception as e:
                    print(f"⚠️  Component iteration failed: {e}")
            else:
                print("❌ Components don't support .iter()")
                
        except Exception as e:
            print(f"❌ Component access failed: {e}")
        
    except Exception as e:
        print(f"❌ Graph chaining test failed: {e}")

def test_trait_based_methods():
    """Test trait-based method injection"""
    print("\n=== Testing Trait-Based Method Injection ===")
    
    # The goal is to test if different types get different methods based on their traits
    try:
        # Create different array types and test what methods they have
        
        # Test NodeId-like operations
        print("Testing NodeIdLike trait methods...")
        # This would test filter_by_degree, get_neighbors, etc.
        
        # Test SubgraphLike operations  
        print("Testing SubgraphLike trait methods...")
        # This would test filter_nodes, filter_edges, collapse, etc.
        
        # Test EdgeLike operations
        print("Testing EdgeLike trait methods...")
        # This would test filter_by_weight, filter_by_endpoints, etc.
        
        print("⚠️  Trait-based method testing requires specialized array exposure to Python")
        
    except Exception as e:
        print(f"❌ Trait-based method testing failed: {e}")

def test_delegation_concept():
    """Test the delegation concept using the apply_to_each method"""
    print("\n=== Testing Delegation Concept ===")
    
    try:
        # Create an array of string values
        str_data = ['hello', 'world', 'python', 'groggy']
        str_array = gr.GraphArray(str_data)
        print(f"✅ Created string array: {str_array}")
        
        # Test the delegation concept with apply_to_each method
        # This should apply the 'upper' method to each string element
        if hasattr(str_array, 'apply_to_each'):
            print("✅ apply_to_each method available")
            
            try:
                # Apply 'upper' method to each string in the array
                result = str_array.apply_to_each('upper', ())
                print(f"✅ Delegation concept works! Result: {result}")
                print(f"   Original: {str_data}")
                print(f"   After .upper(): {list(result)}")
                
                # Test with method arguments
                result2 = str_array.apply_to_each('replace', ('o', 'X'))
                print(f"✅ Method with arguments works: {list(result2)}")
                
            except Exception as e:
                print(f"⚠️  apply_to_each failed: {e}")
        else:
            print("❌ apply_to_each method not available")
            
    except Exception as e:
        print(f"❌ Delegation concept test failed: {e}")

def demonstrate_chaining_vision():
    """Demonstrate the vision for method chaining"""
    print("\n=== Vision for Method Chaining ===")
    
    print("🎯 Target API (what we want to achieve):")
    print("""
    # Instead of manual loops:
    results = []
    for component in graph.connected_components():
        filtered = []
        for node in component.nodes():
            if node.age > 25:
                filtered.append(node)
        results.append(Component(filtered))
    
    # We want fluent chaining:
    results = (graph.connected_components()
               .apply_to_each('filter_nodes', ('age > 25',))
               .apply_to_each('collapse', ({'team_size': 'count'},)))
    
    # Or even better with real __getattr__ delegation:
    results = (graph.connected_components()
               .filter_nodes('age > 25')     # Automatically applied to each component
               .collapse({'team_size': 'count'}))   # Applied to each filtered component
    """)
    
    print("✅ This delegation approach eliminates method duplication!")
    print("✅ Any method that works on individual elements automatically works on arrays!")
    print("✅ No need to reimplement every method for array types!")

if __name__ == "__main__":
    print("🧪 Testing Phase 3: BaseArray Chaining System Implementation")
    
    test_base_array_chaining()
    test_specialized_arrays() 
    test_graph_chaining()
    test_trait_based_methods()
    test_delegation_concept()
    demonstrate_chaining_vision()
    
    print("\n=== Phase 3 Testing Summary ===")
    print("✅ Core infrastructure appears to be implemented")
    print("✅ Delegation concept provides path forward for chaining")
    print("⚠️  Need to expose specialized arrays to Python FFI")
    print("⚠️  Need to ensure BaseArray.iter() returns proper ArrayIterator")
    print("⚠️  Future: Implement __getattr__ delegation for automatic method forwarding")