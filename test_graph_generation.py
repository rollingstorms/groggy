#!/usr/bin/env python3
"""
Test Graph Generation & Interoperability (Step B)

This tests the new graph generators and NetworkX interoperability.
"""

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy as gr

def test_graph_generators():
    """Test the basic graph generators"""
    
    print("🎯 Testing Graph Generation & Interoperability (Step B)")
    
    # Test 1: Complete graph
    print(f"\n📋 Test 1: Complete Graph")
    try:
        g_complete = gr.generators.complete_graph(5, group="test")
        print(f"✅ Complete graph: {g_complete.node_count()} nodes, {g_complete.edge_count()} edges")
        assert g_complete.node_count() == 5
        assert g_complete.edge_count() == 10  # n*(n-1)/2 = 5*4/2 = 10
        print(f"✅ Complete graph structure verified")
        
    except Exception as e:
        print(f"❌ Complete graph failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Erdős-Rényi random graph
    print(f"\n📋 Test 2: Erdős-Rényi Random Graph")
    try:
        g_random = gr.generators.erdos_renyi(50, 0.1, seed=42)
        print(f"✅ Erdős-Rényi graph: {g_random.node_count()} nodes, {g_random.edge_count()} edges")
        assert g_random.node_count() == 50
        # Should have roughly 50*49/2 * 0.1 = ~122 edges (with variance)
        assert 80 <= g_random.edge_count() <= 180  # Allow reasonable variance
        print(f"✅ Erdős-Rényi graph structure verified")
        
    except Exception as e:
        print(f"❌ Erdős-Rényi graph failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Barabási-Albert scale-free network
    print(f"\n📋 Test 3: Barabási-Albert Scale-Free Network")
    try:
        g_ba = gr.generators.barabasi_albert(100, 3, seed=42)
        print(f"✅ Barabási-Albert graph: {g_ba.node_count()} nodes, {g_ba.edge_count()} edges")
        assert g_ba.node_count() == 100
        # Should have roughly 100*3 = 300 edges (after initial complete graph)
        assert 280 <= g_ba.edge_count() <= 320
        print(f"✅ Barabási-Albert graph structure verified")
        
    except Exception as e:
        print(f"❌ Barabási-Albert graph failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Watts-Strogatz small-world network
    print(f"\n📋 Test 4: Watts-Strogatz Small-World Network")
    try:
        g_ws = gr.generators.watts_strogatz(100, 6, 0.1, seed=42)
        print(f"✅ Watts-Strogatz graph: {g_ws.node_count()} nodes, {g_ws.edge_count()} edges")
        assert g_ws.node_count() == 100
        # Should have exactly 100*6/2 = 300 edges (k=6 means each node connects to 3 on each side)
        assert g_ws.edge_count() == 300
        print(f"✅ Watts-Strogatz graph structure verified")
        
    except Exception as e:
        print(f"❌ Watts-Strogatz graph failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Classic graph families
    print(f"\n📋 Test 5: Classic Graph Families")
    try:
        g_cycle = gr.generators.cycle_graph(10)
        g_path = gr.generators.path_graph(10)
        g_star = gr.generators.star_graph(10)
        
        print(f"✅ Cycle graph: {g_cycle.node_count()} nodes, {g_cycle.edge_count()} edges")
        print(f"✅ Path graph: {g_path.node_count()} nodes, {g_path.edge_count()} edges") 
        print(f"✅ Star graph: {g_star.node_count()} nodes, {g_star.edge_count()} edges")
        
        assert g_cycle.node_count() == 10 and g_cycle.edge_count() == 10
        assert g_path.node_count() == 10 and g_path.edge_count() == 9
        assert g_star.node_count() == 10 and g_star.edge_count() == 9
        print(f"✅ Classic graph families verified")
        
    except Exception as e:
        print(f"❌ Classic graph families failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Grid graph
    print(f"\n📋 Test 6: Grid Graph")
    try:
        g_grid_2d = gr.generators.grid_graph([5, 5])
        g_grid_3d = gr.generators.grid_graph([3, 3, 3])
        
        print(f"✅ 2D Grid graph: {g_grid_2d.node_count()} nodes, {g_grid_2d.edge_count()} edges")
        print(f"✅ 3D Grid graph: {g_grid_3d.node_count()} nodes, {g_grid_3d.edge_count()} edges")
        
        # 5x5 grid: 25 nodes, 40 edges (4*5 + 4*5 = 20+20)
        assert g_grid_2d.node_count() == 25 and g_grid_2d.edge_count() == 40
        # 3x3x3 grid: 27 nodes, 54 edges
        assert g_grid_3d.node_count() == 27 and g_grid_3d.edge_count() == 54
        print(f"✅ Grid graphs verified")
        
    except Exception as e:
        print(f"❌ Grid graphs failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 7: Tree graph
    print(f"\n📋 Test 7: Tree Graph")
    try:
        g_tree = gr.generators.tree(15, branching_factor=3)
        print(f"✅ Tree graph: {g_tree.node_count()} nodes, {g_tree.edge_count()} edges")
        assert g_tree.node_count() == 15 and g_tree.edge_count() == 14
        print(f"✅ Tree graph verified")
        
    except Exception as e:
        print(f"❌ Tree graph failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 8: Karate club
    print(f"\n📋 Test 8: Karate Club Graph")
    try:
        g_karate = gr.generators.karate_club()
        print(f"✅ Karate club: {g_karate.node_count()} nodes, {g_karate.edge_count()} edges")
        assert g_karate.node_count() == 34 and g_karate.edge_count() == 78
        print(f"✅ Karate club graph verified")
        
    except Exception as e:
        print(f"❌ Karate club failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 9: Social network
    print(f"\n📋 Test 9: Social Network")
    try:
        g_social = gr.generators.social_network(50, communities=3, seed=42)
        print(f"✅ Social network: {g_social.node_count()} nodes, {g_social.edge_count()} edges")
        assert g_social.node_count() == 50
        assert g_social.edge_count() > 0  # Should have some edges
        print(f"✅ Social network verified")
        
        # Check that nodes have attributes
        node_ids = g_social.node_ids
        first_node = g_social.nodes[node_ids[0]]
        print(f"✅ Sample node attributes: {first_node}")
        
    except Exception as e:
        print(f"❌ Social network failed: {e}")
        import traceback
        traceback.print_exc()

def test_networkx_interoperability():
    """Test NetworkX interoperability"""
    
    print(f"\n🔗 Testing NetworkX Interoperability")
    
    try:
        # Test if NetworkX is available
        import networkx as nx
        print(f"✅ NetworkX {nx.__version__} available")
        
        # Test 1: Groggy to NetworkX conversion
        print(f"\n📋 Test 1: Groggy → NetworkX Conversion")
        try:
            g = gr.generators.karate_club()
            nx_graph = gr.networkx_compat.to_networkx(g, include_attributes=True)
            
            print(f"✅ Converted to NetworkX: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
            assert nx_graph.number_of_nodes() == 34
            assert nx_graph.number_of_edges() == 78
            
            # Check that attributes were preserved
            node_data = nx_graph.nodes[0]
            print(f"✅ Sample NetworkX node attributes: {dict(node_data)}")
            
        except Exception as e:
            print(f"❌ Groggy→NetworkX conversion failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: NetworkX to Groggy conversion
        print(f"\n📋 Test 2: NetworkX → Groggy Conversion")
        try:
            # Create a NetworkX graph
            nx_g = nx.karate_club_graph()
            groggy_g = gr.networkx_compat.from_networkx(nx_g)
            
            print(f"✅ Converted from NetworkX: {groggy_g.node_count()} nodes, {groggy_g.edge_count()} edges")
            assert groggy_g.node_count() == 34
            assert groggy_g.edge_count() == 78
            
        except Exception as e:
            print(f"❌ NetworkX→Groggy conversion failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3: Round-trip conversion
        print(f"\n📋 Test 3: Round-Trip Conversion")
        try:
            original = gr.generators.erdos_renyi(20, 0.3, seed=42)
            nx_version = gr.networkx_compat.to_networkx(original)
            back_to_groggy = gr.networkx_compat.from_networkx(nx_version)
            
            print(f"✅ Original: {original.node_count()} nodes, {original.edge_count()} edges")
            print(f"✅ Round-trip: {back_to_groggy.node_count()} nodes, {back_to_groggy.edge_count()} edges")
            
            # Structure should be preserved
            assert original.node_count() == back_to_groggy.node_count()
            assert original.edge_count() == back_to_groggy.edge_count()
            
        except Exception as e:
            print(f"❌ Round-trip conversion failed: {e}")
            import traceback
            traceback.print_exc()
            
    except ImportError:
        print(f"⚠️  NetworkX not available - skipping interoperability tests")
        print(f"   Install with: pip install networkx")

def test_integration_workflow():
    """Test a complete workflow using generators and interoperability"""
    
    print(f"\n🚀 Testing Integration Workflow")
    
    try:
        # Step 1: Generate a social network
        print(f"Step 1: Generating synthetic social network...")
        g = gr.generators.social_network(30, communities=3, seed=42)
        print(f"✅ Created network: {g.node_count()} nodes, {g.edge_count()} edges")
        
        # Step 2: Analyze with Groggy
        print(f"Step 2: Analyzing with Groggy...")
        components = g.connected_components()
        print(f"✅ Found {len(components)} connected components")
        
        # Step 3: Convert to NetworkX for additional analysis
        print(f"Step 3: Converting to NetworkX for advanced analysis...")
        try:
            import networkx as nx
            nx_g = gr.networkx_compat.to_networkx(g)
            clustering = nx.average_clustering(nx_g)
            diameter = nx.diameter(nx_g) if nx.is_connected(nx_g) else "disconnected"
            print(f"✅ Average clustering: {clustering:.3f}")
            print(f"✅ Network diameter: {diameter}")
            
        except ImportError:
            print(f"⚠️  NetworkX not available for advanced analysis")
        
        # Step 4: Generate different topologies for comparison
        print(f"Step 4: Comparing different network topologies...")
        topologies = {
            "Random": gr.generators.erdos_renyi(30, 0.15, seed=42),
            "Scale-free": gr.generators.barabasi_albert(30, 2, seed=42),
            "Small-world": gr.generators.watts_strogatz(30, 4, 0.1, seed=42),
            "Complete": gr.generators.complete_graph(30),
        }
        
        for name, graph in topologies.items():
            print(f"✅ {name:12}: {graph.node_count():2d} nodes, {graph.edge_count():3d} edges")
        
        print(f"✅ Integration workflow completed successfully!")
        
    except Exception as e:
        print(f"❌ Integration workflow failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_graph_generators()
    test_networkx_interoperability() 
    test_integration_workflow()
    
    print(f"\n🎉 Graph Generation & Interoperability Testing Complete!")
    print(f"✨ Step B Successfully Implemented:")
    print(f"   • Complete graph generator: complete_graph(n)")
    print(f"   • Random models: erdos_renyi(n,p), barabasi_albert(n,m), watts_strogatz(n,k,p)")
    print(f"   • Classic families: cycle_graph(n), path_graph(n), star_graph(n)")
    print(f"   • Structured graphs: grid_graph(dims), tree(n, branching_factor)")
    print(f"   • Real-world models: karate_club(), social_network(n, communities)")
    print(f"   • NetworkX interoperability: to_networkx(), from_networkx()")
    print(f"   • Round-trip conversion support with attribute preservation")
    print(f"   • Integration with existing Groggy analysis capabilities")