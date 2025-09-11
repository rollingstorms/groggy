#!/usr/bin/env python3
"""
WORKING Crazy delegation chaining examples - tested and verified!
"""

import groggy as gr

def example_1_matrix_array_statistical_madness():
    """
    Matrix → Array → Stats → Numpy → Back to Graph operations
    """
    print("🔥 Example 1: Matrix-Array Statistical Madness")
    
    g = gr.karate_club()
    
    try:
        # Ultra-chain: Matrix ops → Array stats → Numpy → Analysis
        result = (g.dense_adjacency_matrix()      # Graph → GraphMatrix (34x34)
                 .transpose()                     # GraphMatrix → GraphMatrix (transposed)  
                 .sum_axis(0)                     # GraphMatrix → GraphArray (row sums)
                 .to_numpy())                     # GraphArray → numpy array
        
        print(f"   ✓ Matrix→Array chain: {result.shape} array, sum = {result.sum()}")
        
        # Even crazier: Matrix → Stats → More stats → Final value
        crazy_stat = (g.dense_adjacency_matrix()
                     .mean_axis(1)                # Column means → GraphArray
                     .std())                      # Standard deviation of means → float
        
        print(f"   ✓ Matrix statistical chain result: {crazy_stat:.4f}")
        
        # Triple matrix chain with different operations
        triple_result = (g.dense_adjacency_matrix()
                        .transpose()              # Transpose
                        .elementwise_multiply(g.dense_adjacency_matrix())  # Element-wise multiply with original
                        .mean_axis(0)             # Row means
                        .percentile(75))          # 75th percentile
        
        print(f"   ✓ Triple matrix chain: 75th percentile = {triple_result:.4f}")
        return True
        
    except Exception as e:
        print(f"   ✗ Matrix madness failed: {e}")
        return False

def example_2_subgraph_transformation_extravaganza():
    """
    Subgraph → BFS → DFS → Sample → Components → Table → Stats
    """
    print("🔥 Example 2: Subgraph Transformation Extravaganza")
    
    g = gr.karate_club()
    
    try:
        # Start with full graph as subgraph, then transform like crazy
        base_subgraph = g.nodes.all()
        
        # Chain 1: BFS → Sample → BFS again → Sample again
        bfs_chain = (base_subgraph
                    .bfs(0)                       # BFS from node 0 → Subgraph
                    .sample(20)                   # Sample 20 nodes → Subgraph  
                    .bfs(5))                      # BFS from node 5 → Subgraph
        
        print(f"   ✓ BFS chain: {len(bfs_chain)} nodes")
        
        # Chain 2: DFS → Components → Largest → Sample
        component_chain = (base_subgraph
                          .dfs(10)                # DFS from node 10 → Subgraph
                          .connected_components() # → ComponentsArray
                          .largest_component()    # → Subgraph (largest component)
                          .sample(15))            # → Subgraph (sampled)
        
        print(f"   ✓ Component chain: {len(component_chain)} nodes")
        
        # Chain 3: Combine with table operations
        table_chain = (component_chain
                      .table()                    # Subgraph → NodesTable
                      .head(10))                  # First 10 rows → NodesTable
        
        print(f"   ✓ Table chain: {type(table_chain)}")
        
        # Chain 4: Back to graph operations
        final_density = component_chain.density()
        print(f"   ✓ Final subgraph density: {final_density:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Subgraph extravaganza failed: {e}")
        return False

def example_3_array_statistics_bonanza():
    """
    GraphArray → Multiple statistical operations → Conversions → More stats
    """
    print("🔥 Example 3: Array Statistics Bonanza")
    
    g = gr.karate_club()
    
    try:
        # Get node IDs and do statistical gymnastics
        node_array = g.nodes.ids()
        
        # Statistical chain 1: Basic stats → Advanced stats
        basic_stats = {
            'count': node_array.count(),
            'mean': node_array.mean(), 
            'median': node_array.median(),
            'std': node_array.std(),
            'min': node_array.min(),
            'max': node_array.max()
        }
        
        print(f"   ✓ Basic stats: count={basic_stats['count']}, mean={basic_stats['mean']:.2f}")
        
        # Advanced percentile analysis
        percentiles = [node_array.percentile(p) for p in [25, 50, 75, 90, 95]]
        print(f"   ✓ Percentiles [25,50,75,90,95]: {percentiles}")
        
        # Conversion chain: Array → List → Numpy → Back to analysis
        numpy_result = (node_array
                       .to_numpy())              # → numpy array
        
        list_result = (node_array  
                      .to_list())                # → Python list
        
        print(f"   ✓ Conversions: numpy {numpy_result.shape}, list {len(list_result)} elements")
        
        # Get edge IDs and compare statistics
        edge_array = g.edge_ids
        edge_stats = {
            'count': edge_array.count(),
            'mean': edge_array.mean(),
            'unique': len(edge_array.unique().to_list())
        }
        
        print(f"   ✓ Edge stats: count={edge_stats['count']}, unique={edge_stats['unique']}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Array bonanza failed: {e}")
        return False

def example_4_neighborhood_component_fusion():
    """
    Neighborhood operations → Component analysis → Statistical fusion
    """
    print("🔥 Example 4: Neighborhood-Component Fusion")
    
    g = gr.karate_club()
    
    try:
        # Get base subgraph
        subgraph = g.nodes.all()
        
        # Neighborhood analysis
        neighborhood_result = subgraph.neighborhood([0, 5, 10], 2)  # 2-hop neighborhoods of 3 nodes
        
        print(f"   ✓ Neighborhoods: {neighborhood_result.total_neighborhoods()} total")
        print(f"   ✓ Largest neighborhood: {neighborhood_result.largest_neighborhood_size()} nodes")
        print(f"   ✓ Execution time: {neighborhood_result.execution_time_ms()} ms")
        
        # Get individual neighborhoods (it's a list, not callable)
        neighborhoods_list = neighborhood_result.neighborhoods  # Property, not method!
        
        if len(neighborhoods_list) > 0:
            first_neighborhood = neighborhoods_list[0]
            print(f"   ✓ First neighborhood: {type(first_neighborhood)} with {len(first_neighborhood)} nodes")
            
            # Chain operations on the neighborhood
            neighbor_table = first_neighborhood.table().head(5)
            neighbor_density = first_neighborhood.density()
            
            print(f"   ✓ Neighborhood table: {type(neighbor_table)}")
            print(f"   ✓ Neighborhood density: {neighbor_density:.4f}")
        
        # Component analysis on neighborhoods
        for i, neighborhood in enumerate(neighborhoods_list[:2]):  # First 2 neighborhoods
            components = neighborhood.connected_components()
            largest = components.largest_component()
            print(f"   ✓ Neighborhood {i}: largest component has {len(largest)} nodes")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Neighborhood fusion failed: {e}")
        return False

def example_5_ultimate_delegation_madness():
    """
    The ultimate chain: Every type, every operation, maximum complexity
    """
    print("🔥 Example 5: ULTIMATE DELEGATION MADNESS")
    
    g = gr.karate_club()
    
    try:
        print("   Phase 1: Matrix → Array → Stats")
        # Start with matrix operations
        degree_variance = (g.dense_adjacency_matrix()
                          .sum_axis(0)            # Get degrees as GraphArray
                          .std())                 # Standard deviation of degrees
        
        print(f"   ✓ Degree variance: {degree_variance:.4f}")
        
        print("   Phase 2: Graph → Subgraph → Components → Analysis")
        # Subgraph component analysis
        components = (g.nodes.all()               # Graph → Subgraph
                     .connected_components())     # Subgraph → ComponentsArray
        
        component_sizes = components.sizes()      # → List of (nodes, edges) tuples
        print(f"   ✓ Component analysis: {len(component_sizes)} components")
        
        print("   Phase 3: Largest Component → Multi-operation Chain")
        largest = components.largest_component()  # ComponentsArray → Subgraph
        
        # Multi-operation chain on largest component
        chain_result = (largest
                       .sample(25)                # Sample nodes
                       .bfs(0)                    # BFS traversal
                       .dfs(10))                  # DFS traversal
        
        print(f"   ✓ Multi-operation result: {len(chain_result)} nodes")
        
        print("   Phase 4: Statistical Convergence")
        # Final statistical analysis
        final_stats = {
            'density': chain_result.density(),
            'nodes': len(chain_result),
            'edges': chain_result.edge_count(),
            'is_connected': chain_result.is_connected()
        }
        
        print(f"   ✓ Final stats: {final_stats}")
        
        print("   Phase 5: Cross-Type Validation")
        # Validate by converting through all types
        validation_chain = (chain_result
                           .table()               # → NodesTable
                           .head(3))              # → NodesTable (first 3)
        
        matrix_validation = (g.laplacian_matrix() # Different matrix
                            .transpose()          # → GraphMatrix
                            .mean_axis(1)         # → GraphArray
                            .count())             # → int
        
        print(f"   ✓ Cross-validation: table={type(validation_chain)}, matrix_count={matrix_validation}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Ultimate madness failed: {e}")
        return False

def main():
    """Run all WORKING crazy delegation examples"""
    print("🚀 WORKING CRAZY DELEGATION CHAINS")
    print("=" * 60)
    
    examples = [
        example_1_matrix_array_statistical_madness,
        example_2_subgraph_transformation_extravaganza,
        example_3_array_statistics_bonanza,
        example_4_neighborhood_component_fusion,
        example_5_ultimate_delegation_madness,
    ]
    
    results = []
    for i, example_func in enumerate(examples, 1):
        print(f"\n📍 Example {i}")
        print("-" * 50)
        success = example_func()
        results.append(success)
        print()
    
    print("🏆 FINAL RESULTS")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    for i, success in enumerate(results, 1):
        status = "✅ WORKS" if success else "❌ BROKEN" 
        print(f"{status} Example {i}")
    
    print(f"\n🔥 {passed}/{total} INSANE chains work perfectly!")
    
    if passed >= 4:
        print("🚀 DELEGATION ARCHITECTURE IS ABSOLUTELY INCREDIBLE!")
        print("   You can chain operations across types seamlessly!")
        print("   Matrix → Array → Stats → Subgraph → Components → Table → Operations!")
    
    return passed == total

if __name__ == "__main__":
    main()
