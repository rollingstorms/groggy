#!/usr/bin/env python3
"""
Crazy delegation chaining examples showing the power of the unified architecture
"""

import groggy as gr

def example_1_component_analysis_chain():
    """
    Ultra-long chain: Graph → Components → Largest → Sample → BFS → Neighborhood → Table → Head
    """
    print("🔥 Example 1: Component Analysis Super-Chain")
    
    g = gr.karate_club()
    
    try:
        result = (g.nodes.all()                    # Graph → Subgraph
                   .connected_components()         # Subgraph → ComponentsArray  
                   .largest_component()            # ComponentsArray → Subgraph
                   .sample(15)                     # Subgraph → Subgraph (sampled)
                   .bfs(0)                         # Subgraph → Subgraph (BFS traversal)
                   .neighborhood([5], 1)           # Subgraph → NeighborhoodResult
                   .neighborhoods()[0]             # NeighborhoodResult → Subgraph (first neighborhood)
                   .table()                        # Subgraph → NodesTable
                   .head(5))                       # NodesTable → NodesTable (first 5 rows)
        
        print(f"   ✓ Super-chain result: {type(result)}")
        return True
    except Exception as e:
        print(f"   ✗ Super-chain failed: {e}")
        return False

def example_2_matrix_statistical_chain():
    """
    Matrix operations chain: Graph → Matrix → Transpose → Stats → Array → Statistics
    """
    print("🔥 Example 2: Matrix Statistical Chain")
    
    g = gr.karate_club()
    
    try:
        # Chain matrix operations with array statistics
        mean_degrees = (g.dense_adjacency_matrix()    # Graph → GraphMatrix
                       .transpose()                   # GraphMatrix → GraphMatrix
                       .sum_axis(0)                   # GraphMatrix → GraphArray (row sums = degrees)
                       .to_numpy())                   # GraphArray → numpy array
        
        print(f"   ✓ Matrix stats chain: shape {mean_degrees.shape}, mean degree = {mean_degrees.mean():.2f}")
        
        # Even crazier: Matrix → Array → Statistics → Back to operations
        percentiles = (g.dense_adjacency_matrix()
                      .mean_axis(1)                   # Column-wise means
                      .percentile(90))                # 90th percentile of means
        
        print(f"   ✓ Matrix percentile chain: 90th percentile = {percentiles}")
        return True
        
    except Exception as e:
        print(f"   ✗ Matrix chain failed: {e}")
        return False

def example_3_multi_subgraph_operations():
    """
    Multiple subgraph transformations: BFS → DFS → Sample → Filter → Merge operations
    """
    print("🔥 Example 3: Multi-Subgraph Operations")
    
    g = gr.karate_club()
    
    try:
        # Create multiple subgraphs and operate on them
        bfs_subgraph = g.bfs(0)                      # Start with BFS from node 0
        dfs_subgraph = g.dfs(10)                     # DFS from node 10
        
        print(f"   • BFS subgraph: {len(bfs_subgraph)} nodes")
        print(f"   • DFS subgraph: {len(dfs_subgraph)} nodes")
        
        # Chain operations on BFS result
        bfs_chain = (bfs_subgraph
                    .sample(20)                       # Sample 20 nodes
                    .induced_subgraph()               # Get induced subgraph
                    .filter_nodes("degree > 2")       # Filter high-degree nodes
                    .edges_table())                   # Convert to edges table
        
        print(f"   ✓ BFS chain result: {type(bfs_chain)}")
        
        # Test intersection/merge operations  
        if hasattr(bfs_subgraph, 'intersect_with'):
            intersection = bfs_subgraph.intersect_with(dfs_subgraph)
            print(f"   ✓ Subgraph intersection: {len(intersection)} nodes")
            
        return True
        
    except Exception as e:
        print(f"   ✗ Multi-subgraph chain failed: {e}")
        return False

def example_4_component_neighborhood_analysis():
    """
    Advanced component and neighborhood analysis with statistical aggregation
    """
    print("🔥 Example 4: Component-Neighborhood Analysis")
    
    g = gr.karate_club()
    
    try:
        # Get components, analyze neighborhoods, aggregate statistics
        components = g.nodes.all().connected_components()
        
        # Get the largest component and analyze its neighborhood structure
        largest_comp = components.largest_component()
        
        # For each high-degree node, get neighborhood and analyze
        high_degree_nodes = largest_comp.filter_nodes("degree > 5")
        
        if len(high_degree_nodes) > 0:
            # Get neighborhoods of high-degree nodes
            central_nodes = high_degree_nodes.node_ids().to_list()[:3]  # Top 3 high-degree nodes
            neighborhoods = largest_comp.neighborhood(central_nodes, 2)  # 2-hop neighborhoods
            
            print(f"   ✓ Neighborhood analysis: {neighborhoods.total_neighborhoods()} neighborhoods")
            print(f"   ✓ Largest neighborhood: {neighborhoods.largest_neighborhood_size()} nodes")
            
            # Convert first neighborhood to table for analysis
            if neighborhoods.total_neighborhoods() > 0:
                first_neighborhood = neighborhoods.neighborhoods()[0]
                neighbor_table = first_neighborhood.table().head(10)
                print(f"   ✓ Neighborhood table: {type(neighbor_table)}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Component-neighborhood analysis failed: {e}")
        return False

def example_5_cross_type_statistical_fusion():
    """
    Ultimate cross-type chain: Graph → Matrix → Array → Stats → Subgraph → Table → Stats
    """
    print("🔥 Example 5: Cross-Type Statistical Fusion")
    
    g = gr.karate_club()
    
    try:
        # Start with graph, go through every major type with statistics at each step
        
        # Step 1: Graph → Matrix stats
        matrix_stats = g.dense_adjacency_matrix().std_axis(0)  # Standard deviation per row
        print(f"   • Matrix std stats: {type(matrix_stats)}")
        
        # Step 2: Matrix stats → Array stats
        high_variance_threshold = matrix_stats.percentile(75)  # 75th percentile of std devs
        print(f"   • High variance threshold: {high_variance_threshold}")
        
        # Step 3: Use stats to filter graph → Subgraph
        # Find nodes with high connectivity variance (indirect measure)
        high_degree_nodes = g.filter_nodes("degree > 5")  # Proxy for high variance
        print(f"   • High-degree subgraph: {len(high_degree_nodes)} nodes")
        
        # Step 4: Subgraph → BFS → Sample → Statistics
        if len(high_degree_nodes) > 0:
            sampled_bfs = (high_degree_nodes
                          .bfs(list(high_degree_nodes.node_ids().to_list())[0])
                          .sample(10))
            
            # Step 5: Final statistics - density and edge analysis  
            final_density = sampled_bfs.density()
            final_edges = sampled_bfs.edge_count()
            
            print(f"   ✓ Final fusion stats: density={final_density:.3f}, edges={final_edges}")
            
            # Step 6: Convert to table and get summary
            final_table = sampled_bfs.table().head(5)
            print(f"   ✓ Final table: {type(final_table)}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Cross-type fusion failed: {e}")
        return False

def main():
    """Run all crazy delegation examples"""
    print("🚀 CRAZY DELEGATION CHAINING EXAMPLES")
    print("=" * 60)
    
    examples = [
        example_1_component_analysis_chain,
        example_2_matrix_statistical_chain, 
        example_3_multi_subgraph_operations,
        example_4_component_neighborhood_analysis,
        example_5_cross_type_statistical_fusion,
    ]
    
    results = []
    for i, example_func in enumerate(examples, 1):
        print(f"\n📍 Running Example {i}")
        print("-" * 40)
        success = example_func()
        results.append(success)
        print()
    
    print("📊 RESULTS")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    for i, success in enumerate(results, 1):
        status = "✅ PASS" if success else "❌ FAIL" 
        print(f"{status} Example {i}")
    
    print(f"\n🏆 {passed}/{total} crazy chains worked!")
    
    if passed == total:
        print("🔥 INCREDIBLE! All crazy delegation chains work!")
        print("   The delegation architecture is absolutely INSANE! 🚀")
    else:
        print("🔧 Some chains need work, but the architecture is still amazing!")

if __name__ == "__main__":
    main()
