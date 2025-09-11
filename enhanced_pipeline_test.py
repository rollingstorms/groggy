#!/usr/bin/env python3
"""
Enhanced Pipeline Test - Building on comprehensive_pipeline_test.py results
Focus on working functionality and identify the most impactful fixes needed
"""

import groggy as gr
import traceback
import time

def test_working_pipelines():
    """Test the pipelines that we know work based on the previous analysis"""
    print("🟢 WORKING PIPELINE PATTERNS")
    print("=" * 60)
    
    g = gr.karate_club()
    
    # These work well - let's push them further
    successful_chains = []
    
    # 1. Subgraph → Table chains (WORK!)
    print("1. Subgraph → Table Chains:")
    
    bfs_table = g.bfs(0).table()
    print(f"   ✓ BFS(0) → table: {len(bfs_table)} rows")
    successful_chains.append("BFS → Table")
    
    dfs_table = g.dfs(0).table()
    print(f"   ✓ DFS(0) → table: {len(dfs_table)} rows")
    successful_chains.append("DFS → Table")
    
    bfs_head = g.bfs(0).table().head(5)
    print(f"   ✓ BFS(0) → table → head(5): {len(bfs_head)} rows")
    successful_chains.append("BFS → Table → Head")
    
    # 2. Accessor chains (WORK!)
    print("\n2. Accessor Chains:")
    
    node_list = list(g.nodes)
    print(f"   ✓ list(g.nodes): {len(node_list)} nodes")
    successful_chains.append("Graph → Nodes → List")
    
    edge_list = list(g.edges)
    print(f"   ✓ list(g.edges): {len(edge_list)} edges")
    successful_chains.append("Graph → Edges → List")
    
    nodes_table = g.nodes.table()
    print(f"   ✓ g.nodes.table(): {len(nodes_table)} rows")
    successful_chains.append("Nodes → Table")
    
    edges_table = g.edges.table()
    print(f"   ✓ g.edges.table(): {len(edges_table)} rows")
    successful_chains.append("Edges → Table")
    
    # 3. Table operations (MOSTLY WORK!)
    print("\n3. Table Operations:")
    
    nodes_head = g.nodes.table().head(3)
    print(f"   ✓ nodes.table().head(3): {len(nodes_head)} rows")
    successful_chains.append("Nodes → Table → Head")
    
    edges_head = g.edges.table().head(5)
    print(f"   ✓ edges.table().head(5): {len(edges_head)} rows")
    successful_chains.append("Edges → Table → Head")
    
    # 4. Subgraph property access (WORKS!)
    print("\n4. Subgraph Properties:")
    
    bfs_sg = g.bfs(0)
    print(f"   ✓ BFS density: {bfs_sg.density():.3f}")
    print(f"   ✓ BFS node count: {bfs_sg.node_count()}")
    print(f"   ✓ BFS edge count: {bfs_sg.edge_count()}")
    successful_chains.extend(["Subgraph → Density", "Subgraph → NodeCount", "Subgraph → EdgeCount"])
    
    # 5. Connected components (WORKS!)
    print("\n5. Connected Components:")
    
    components = g.connected_components()
    print(f"   ✓ Connected components: {type(components)}")
    successful_chains.append("Graph → ConnectedComponents")
    
    print(f"\n✅ {len(successful_chains)} working pipeline patterns identified!")
    return successful_chains

def test_broken_patterns_with_fixes():
    """Test broken patterns and show what needs to be fixed"""
    print("\n🔴 BROKEN PATTERNS & REQUIRED FIXES")
    print("=" * 60)
    
    g = gr.karate_club()
    fixes_needed = []
    
    # 1. Graph creation API issues
    print("1. Graph Creation API:")
    try:
        empty_graph = gr.Graph()
        empty_graph.add_node(0, {"name": "test"})
    except Exception as e:
        print(f"   ✗ Graph.add_node() API: {type(e).__name__}")
        fixes_needed.append("Fix Graph.add_node() to accept (node_id, attributes) signature")
    
    # 2. Neighborhood API issues  
    print("\n2. Neighborhood API:")
    try:
        neighborhood = g.neighborhood(0)
    except Exception as e:
        print(f"   ✗ g.neighborhood(single_node): {type(e).__name__}")
        fixes_needed.append("Fix neighborhood() to accept single node or sequence")
        
        # Try the correct API
        try:
            neighborhood = g.neighborhood([0])
            print(f"   ✓ g.neighborhood([0]) works: {len(neighborhood)} nodes")
        except Exception as e2:
            print(f"   ✗ g.neighborhood([0]): {type(e2).__name__}")
            fixes_needed.append("Fix neighborhood() sequence API")
    
    # 3. Matrix API issues
    print("\n3. Matrix Operations:")
    try:
        matrix = g.adjacency_matrix()
        shape = matrix.shape()
    except Exception as e:
        print(f"   ✗ matrix.shape() method: {type(e).__name__}")
        if hasattr(matrix, 'shape'):
            print(f"   ℹ️ matrix.shape property works: {matrix.shape}")
            fixes_needed.append("Ensure matrix.shape is a property, not method")
        else:
            fixes_needed.append("Add matrix.shape property or method")
    
    # 4. Table columns API
    print("\n4. Table API:")
    try:
        table = g.nodes.table()
        columns = table.columns()
    except Exception as e:
        print(f"   ✗ table.columns(): {type(e).__name__}")
        # Check if it's a property
        if hasattr(table, 'columns') and not callable(table.columns):
            print(f"   ℹ️ table.columns property might work")
            fixes_needed.append("Ensure table.columns is accessible as property")
        else:
            fixes_needed.append("Add table.columns() method or property")
    
    # 5. Advanced algorithms
    print("\n5. Advanced Algorithms:")
    advanced_missing = []
    for algo in ['pagerank', 'betweenness_centrality', 'closeness_centrality', 'clustering_coefficient']:
        if not hasattr(g, algo):
            advanced_missing.append(algo)
    
    if advanced_missing:
        print(f"   ✗ Missing algorithms: {', '.join(advanced_missing)}")
        fixes_needed.append(f"Implement missing algorithms: {', '.join(advanced_missing)}")
    
    print(f"\n❌ {len(fixes_needed)} fixes needed for full functionality")
    return fixes_needed

def test_delegation_potential():
    """Test what delegation patterns are possible with current functionality"""
    print("\n🔗 DELEGATION CHAIN POTENTIAL")
    print("=" * 60)
    
    g = gr.karate_club()
    
    print("Testing increasingly complex chains...")
    
    # Level 1: Basic chains (should work)
    chains_level1 = [
        ("Graph → BFS → Table", lambda: g.bfs(0).table()),
        ("Graph → Nodes → Table → Head", lambda: g.nodes.table().head(3)),
        ("Graph → Connected Components", lambda: g.connected_components()),
    ]
    
    working_level1 = 0
    for name, chain_func in chains_level1:
        try:
            result = chain_func()
            print(f"   ✓ Level 1: {name}")
            working_level1 += 1
        except Exception as e:
            print(f"   ✗ Level 1: {name} - {type(e).__name__}")
    
    # Level 2: Cross-type conversions
    print(f"\nLevel 1 Success Rate: {working_level1}/{len(chains_level1)}")
    
    chains_level2 = []
    
    # Try to find cross-type conversions that work
    try:
        bfs_sg = g.bfs(0)
        
        # Test if subgraph has conversion methods
        conversion_methods = []
        for method_name in ['nodes', 'edges', 'to_nodes', 'to_edges', 'to_matrix']:
            if hasattr(bfs_sg, method_name):
                conversion_methods.append(method_name)
        
        print(f"   ℹ️ Subgraph conversion methods available: {conversion_methods}")
        
        # Test working conversions
        for method in conversion_methods:
            try:
                method_func = getattr(bfs_sg, method)
                if callable(method_func):
                    result = method_func()
                    print(f"   ✓ Level 2: Subgraph.{method}() → {type(result).__name__}")
                else:
                    print(f"   ✓ Level 2: Subgraph.{method} → {type(result).__name__}")
            except Exception as e:
                print(f"   ✗ Level 2: Subgraph.{method} - {type(e).__name__}")
    
    except Exception as e:
        print(f"   ✗ Level 2 setup failed: {e}")
    
    # Level 3: Statistical operations
    print(f"\nTesting statistical operations...")
    try:
        # Create some arrays to test with
        node_degrees = []
        for node in g.nodes:
            node_degrees.append(node.degree())
        
        print(f"   ✓ Level 3: Manual degree calculation: {len(node_degrees)} values")
        print(f"   ℹ️ Degree stats: min={min(node_degrees)}, max={max(node_degrees)}, avg={sum(node_degrees)/len(node_degrees):.1f}")
        
    except Exception as e:
        print(f"   ✗ Level 3: Statistical operations - {type(e).__name__}")

def test_realistic_use_cases():
    """Test realistic graph analysis use cases"""
    print("\n🎯 REALISTIC USE CASES")
    print("=" * 60)
    
    g = gr.karate_club()
    
    use_cases = [
        ("Node Analysis", lambda: analyze_nodes(g)),
        ("Edge Analysis", lambda: analyze_edges(g)),
        ("Structural Analysis", lambda: analyze_structure(g)),
        ("Subgraph Exploration", lambda: explore_subgraphs(g)),
    ]
    
    for case_name, case_func in use_cases:
        print(f"\n{case_name}:")
        try:
            result = case_func()
            if result:
                print(f"   ✓ {case_name} successful")
            else:
                print(f"   ⚠️ {case_name} completed with limitations")
        except Exception as e:
            print(f"   ✗ {case_name} failed: {type(e).__name__}: {str(e)[:100]}...")

def analyze_nodes(g):
    """Analyze nodes in various ways"""
    try:
        # Basic node analysis
        nodes_table = g.nodes.table()
        print(f"     • Total nodes: {len(nodes_table)}")
        
        # Get degree information
        degrees = []
        for node in g.nodes:
            degrees.append(node.degree())
        
        print(f"     • Degree range: {min(degrees)}-{max(degrees)}")
        print(f"     • Average degree: {sum(degrees)/len(degrees):.1f}")
        
        # Find high-degree nodes
        high_degree_nodes = [i for i, deg in enumerate(degrees) if deg > sum(degrees)/len(degrees)]
        print(f"     • High-degree nodes: {len(high_degree_nodes)}")
        
        return True
    except Exception as e:
        print(f"     ✗ Node analysis error: {e}")
        return False

def analyze_edges(g):
    """Analyze edges in various ways"""
    try:
        # Basic edge analysis  
        edges_table = g.edges.table()
        print(f"     • Total edges: {len(edges_table)}")
        
        # Analyze edge distribution
        edge_count = 0
        for edge in g.edges:
            edge_count += 1
        
        print(f"     • Edge count verified: {edge_count}")
        return True
    except Exception as e:
        print(f"     ✗ Edge analysis error: {e}")
        return False

def analyze_structure(g):
    """Analyze graph structure"""
    try:
        # Connected components
        components = g.connected_components()
        print(f"     • Connected components: {type(components)}")
        
        # Graph density (if available on full graph)
        try:
            # Create a subgraph containing all nodes for density calculation
            full_bfs = g.bfs(0)  # This should give us the full connected component
            density = full_bfs.density()
            print(f"     • Graph density: {density:.3f}")
        except Exception as e:
            print(f"     • Density calculation unavailable: {type(e).__name__}")
        
        return True
    except Exception as e:
        print(f"     ✗ Structural analysis error: {e}")
        return False

def explore_subgraphs(g):
    """Explore different subgraph types"""
    try:
        # BFS exploration
        bfs_sg = g.bfs(0)
        print(f"     • BFS from 0: {len(bfs_sg)} nodes, {bfs_sg.edge_count()} edges")
        
        # DFS exploration  
        dfs_sg = g.dfs(0)
        print(f"     • DFS from 0: {len(dfs_sg)} nodes, {dfs_sg.edge_count()} edges")
        
        # Neighborhood exploration (with correct API)
        try:
            neigh_sg = g.neighborhood([0])  # Use list instead of single int
            print(f"     • Neighborhood of [0]: {len(neigh_sg)} nodes")
        except Exception as e:
            print(f"     • Neighborhood unavailable: {type(e).__name__}")
        
        return True
    except Exception as e:
        print(f"     ✗ Subgraph exploration error: {e}")
        return False

def main():
    """Run enhanced pipeline testing"""
    print("🚀 ENHANCED PIPELINE TESTING")
    print("Building on comprehensive analysis to focus on working patterns")
    print("="*80)
    
    # Test what works well
    working_patterns = test_working_pipelines()
    
    # Identify what needs fixing
    fixes_needed = test_broken_patterns_with_fixes()
    
    # Test delegation potential
    test_delegation_potential()
    
    # Test realistic use cases
    test_realistic_use_cases()
    
    # Final recommendations
    print("\n" + "="*80)
    print("🎯 STRATEGIC RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n🟢 STRENGTHS ({len(working_patterns)} working patterns):")
    print("  • Subgraph operations are solid")
    print("  • Table creation and basic operations work well")
    print("  • Accessor patterns are reliable")
    print("  • Basic graph traversal algorithms work")
    
    print(f"\n🔴 HIGH-IMPACT FIXES NEEDED ({len(fixes_needed)}):")
    for i, fix in enumerate(fixes_needed, 1):
        print(f"  {i}. {fix}")
    
    print(f"\n💡 DEVELOPMENT PRIORITIES:")
    print(f"  1. Fix API inconsistencies (add_node, neighborhood, matrix.shape)")
    print(f"  2. Add missing table methods (columns access)")
    print(f"  3. Implement core graph algorithms (pagerank, centrality)")
    print(f"  4. Add cross-type conversion methods")
    print(f"  5. Enable advanced delegation patterns")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"  • Focus on the {min(3, len(fixes_needed))} highest-impact fixes first")
    print(f"  • Build comprehensive delegation examples once APIs are fixed") 
    print(f"  • Create performance benchmarks for working patterns")
    print(f"  • Document working pipeline patterns for users")

if __name__ == "__main__":
    main()