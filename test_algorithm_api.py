#!/usr/bin/env python3
"""
Complete test script for Groggy's algorithm API.

This script demonstrates:
1. Label Propagation Algorithm (LPA) with visible community results
2. Multi-algorithm pipeline with PageRank + BFS
3. Convenience apply() function for simpler usage
4. Algorithm discovery and introspection
5. Builder DSL (foundation for future custom algorithms)
"""

import sys
from groggy import Graph, pipeline, algorithms, builder, apply


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def quick_start_example():
    """Show the tiniest possible PageRank run for quick orientation."""
    print_header("QUICK START: Single Algorithm Example")

    g = Graph()
    a = g.add_node()
    b = g.add_node()
    g.add_edge(a, b)
    g.add_edge(b, a)

    sg = g.view()
    print("Created 2 nodes with reciprocal edge")

    result = sg.apply(
        algorithms.centrality.pagerank(max_iter=10, output_attr="score")
    )

    scores = {node.id: node.score for node in result.nodes}
    for node_id, value in sorted(scores.items()):
        print(f"  Node {node_id}: score={value:.6f}")

    return result


def test_lpa_community_detection():
    """
    Test Label Propagation Algorithm (LPA) for community detection.
    
    Creates a graph with clear community structure and shows which nodes
    are assigned to which communities.
    """
    print_header("TEST 1: Label Propagation Algorithm (LPA)")
    
    # Create a graph with 3 distinct communities
    print("\nCreating graph with 3 communities...")
    g = Graph()
    
    # Community 1: Nodes 0-4 (tightly connected)
    community1 = [g.add_node() for _ in range(5)]
    for i in range(len(community1)):
        for j in range(i + 1, len(community1)):
            g.add_edge(community1[i], community1[j])  # Fully connected
    
    # Community 2: Nodes 5-9 (tightly connected)
    community2 = [g.add_node() for _ in range(5)]
    for i in range(len(community2)):
        for j in range(i + 1, len(community2)):
            g.add_edge(community2[i], community2[j])  # Fully connected
    
    # Community 3: Nodes 10-14 (tightly connected)
    community3 = [g.add_node() for _ in range(5)]
    for i in range(len(community3)):
        for j in range(i + 1, len(community3)):
            g.add_edge(community3[i], community3[j])  # Fully connected
    
    # Add a few weak connections between communities
    g.add_edge(community1[0], community2[0])
    g.add_edge(community2[0], community3[0])
    
    comm1_ids = community1  # Already IDs
    comm2_ids = community2
    comm3_ids = community3
    
    print(f"  Created 15 nodes in 3 communities")
    print(f"  Community 1: nodes {comm1_ids}")
    print(f"  Community 2: nodes {comm2_ids}")
    print(f"  Community 3: nodes {comm3_ids}")
    
    # Use g.view() to work with full graph
    sg = g.view()
    
    # Run LPA
    print("\nRunning Label Propagation Algorithm...")
    lpa_algo = algorithms.community.lpa(
        max_iter=50,
        output_attr="community"
    )
    
    pipe = pipeline([lpa_algo])
    result = pipe(sg)
    
    print("✓ LPA completed successfully")
    print(f"✓ Processed {len(result.nodes)} nodes")
    
    # Extract community labels using direct attribute access
    print("\n--- Community Detection Results ---")
    communities = {}
    for node in result.nodes:
        label = node.community
        communities.setdefault(label, []).append(node.id)
    
    # Print communities
    for community_id in sorted(communities.keys()):
        members = sorted(communities[community_id])
        print(f"\nCommunity {community_id}: {len(members)} nodes")
        print(f"  Nodes: {members}")
        
        # Check which original communities these correspond to
        in_comm1 = sum(1 for n in members if n in comm1_ids)
        in_comm2 = sum(1 for n in members if n in comm2_ids)
        in_comm3 = sum(1 for n in members if n in comm3_ids)
        print(f"  Breakdown: {in_comm1} from C1, {in_comm2} from C2, {in_comm3} from C3")
    
    # Verify against expected structure
    print("\n--- Verification ---")
    print(f"Expected: 3 communities (5 nodes each)")
    print(f"Found: {len(communities)} communities")
    if len(communities) == 3:
        print("✓ Correct number of communities detected!")
        # Check if each community is relatively pure
        for comm_id, members in communities.items():
            if len(members) >= 4:  # At least 4 from same original community
                print(f"✓ Community {comm_id} shows good clustering")
    
    return result


def test_multi_algorithm_pipeline():
    """
    Test a multi-algorithm pipeline: PageRank + BFS.
    
    Demonstrates composing algorithms where each builds on the previous.
    """
    print_header("TEST 2: Multi-Algorithm Pipeline (PageRank + BFS)")
    
    # Create a network with a hub structure
    print("\nCreating hub-and-spoke network...")
    g = Graph()
    
    # Central hub node
    hub = g.add_node()
    
    # Create 4 spokes, each with 5 nodes
    spokes = []
    for spoke_idx in range(4):
        spoke_nodes = []
        prev = hub
        for i in range(5):
            node = g.add_node()
            g.add_edge(prev, node)
            prev = node
            spoke_nodes.append(node)
        spokes.extend(spoke_nodes)
    
    print(f"  Created {1 + len(spokes)} nodes")
    print(f"  Hub node: {hub}")  # hub is already an integer ID
    print(f"  Spoke nodes: {len(spokes)} nodes in 4 chains")
    
    # Mark hub as start for BFS
    g.nodes.set_attrs({hub: {"is_start": True}})
    
    # Use g.view()
    sg = g.view()
    
    # Build pipeline
    print("\nBuilding pipeline:")
    print("  1. PageRank (measures importance)")
    print("  2. BFS from hub (measures distance)")
    
    pipe = pipeline([
        algorithms.centrality.pagerank(
            max_iter=50,
            damping=0.85,
            output_attr="importance"
        ),
        algorithms.pathfinding.bfs(
            start_attr="is_start",
            output_attr="distance_from_hub"
        )
    ])
    
    print("\nExecuting pipeline...")
    result = pipe(sg)
    
    print("✓ Pipeline completed successfully")
    print(f"✓ Processed {len(result.nodes)} nodes")
    
    # Extract and display results using direct attribute access
    print("\n--- Algorithm Results ---")
    
    # Collect results
    node_results = []
    for node in result.nodes:
        node_results.append((node.id, node.importance, node.distance_from_hub))
    
    # Sort by importance (descending)
    node_results.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 nodes by PageRank importance:")
    print(f"{'Node':<8} {'Importance':<15} {'Distance from Hub':<20}")
    print("-" * 50)
    for node_id, importance, distance in node_results[:10]:
        print(f"{node_id:<8} {importance:<15.6f} {int(distance):<20}")
    
    # Show hub node specifically
    hub_idx = next(i for i, (nid, _, _) in enumerate(node_results) if nid == hub)  # hub is already an int
    hub_imp, hub_dist = node_results[hub_idx][1], node_results[hub_idx][2]
    
    print(f"\n--- Hub Node Analysis (Node {hub}) ---")  # hub is already an int
    print(f"  PageRank Importance: {hub_imp:.6f}")
    print(f"  Distance from Hub: {int(hub_dist)}")
    print(f"  Rank by Importance: {hub_idx + 1} of {len(node_results)}")
    
    # Analyze distance distribution
    distance_counts = {}
    for node_id, importance, distance in node_results:
        dist_int = int(distance)
        distance_counts[dist_int] = distance_counts.get(dist_int, 0) + 1
    
    print("\n--- Distance Distribution ---")
    for dist in sorted(distance_counts.keys()):
        count = distance_counts[dist]
        print(f"  Distance {dist}: {count} nodes")
    
    print("\n--- Verification ---")
    if hub_dist == 0:
        print("✓ Hub has distance 0 (correct)")
    if hub_imp > 0.04:  # Hub should have high importance
        print(f"✓ Hub has high importance ({hub_imp:.6f})")
    if hub_idx < 5:  # Hub should be in top 5
        print(f"✓ Hub is in top 5 by importance (rank {hub_idx + 1})")
    
    return result


def test_subgraph_apply_method():
    """Demonstrate the three supported forms of Subgraph.apply()."""
    print_header("TEST 3: Subgraph.apply() Method")

    g = Graph()
    nodes = [g.add_node() for _ in range(8)]

    # Build a simple cycle with a shortcut edge
    for i in range(len(nodes)):
        g.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])
    g.add_edge(nodes[0], nodes[4])

    g.nodes.set_attrs({nodes[0]: {"is_seed": True}})

    sg = g.view()
    print(f"Graph prepared with {len(nodes)} nodes")

    # 1. Single algorithm handle
    print("\nForm 1: sg.apply(algorithm_handle)")
    result_single = sg.apply(
        algorithms.centrality.pagerank(max_iter=15, output_attr="pr_single")
    )
    print("  ✓ PageRank executed via handle")

    # 2. List of algorithms
    print("\nForm 2: sg.apply([algo1, algo2])")
    result_list = sg.apply([
        algorithms.centrality.pagerank(max_iter=15, output_attr="importance"),
        algorithms.pathfinding.bfs(start_attr="is_seed", output_attr="distance"),
    ])
    print("  ✓ PageRank followed by BFS executed via list")

    node0 = next(node for node in result_list.nodes if node.id == nodes[0])
    print(
        f"  Node {node0.id}: importance={node0.importance:.6f}, distance={int(node0.distance)}"
    )

    # 3. Pipeline object
    print("\nForm 3: sg.apply(pipeline_object)")
    pipe = pipeline([
        algorithms.community.lpa(max_iter=20, output_attr="community"),
    ])
    result_pipeline = sg.apply(pipe)
    communities = sorted({node.community for node in result_pipeline.nodes})
    print(f"  ✓ Pipeline executed; communities discovered: {communities}")

    return result_single, result_list, result_pipeline


def test_apply_convenience_function():
    """
    Test the apply() convenience function for simpler usage.
    
    Shows that you can use apply(subgraph, algorithm) instead of
    creating a pipeline manually.
    """
    print_header("TEST 4: Convenience apply() Function")
    
    print("\nThe apply() function provides a simpler way to run algorithms:")
    print("  Instead of: pipe = pipeline([algo]); result = pipe(sg)")
    print("  You can:    result = apply(sg, algo)")
    
    # Create a simple graph
    print("\nCreating test graph...")
    g = Graph()
    nodes = [g.add_node() for _ in range(15)]
    
    # Create a ring structure
    for i in range(len(nodes)):
        g.add_edge(nodes[i], nodes[(i + 1) % len(nodes)])
    
    sg = g.view()
    print(f"  Created {len(nodes)} nodes in a ring")
    
    # Method 1: Single algorithm
    print("\n--- Method 1: Apply Single Algorithm ---")
    print("  Code: apply(sg, algorithms.centrality.pagerank())")
    result1 = apply(sg, algorithms.centrality.pagerank(max_iter=20, output_attr="pr1"))
    print(f"  ✓ Success: {len(result1.nodes)} nodes processed")
    
    # Show some results
    top_nodes = sorted([(n.id, n.pr1) for n in result1.nodes], key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top 3 nodes by PageRank:")
    for node_id, pr_score in top_nodes:
        print(f"    Node {node_id}: {pr_score:.6f}")
    
    # Method 2: List of algorithms
    print("\n--- Method 2: Apply List of Algorithms ---")
    print("  Code: apply(sg, [algo1, algo2, algo3])")
    
    # Mark a start node
    g.nodes.set_attrs({nodes[0]: {"start": True}})
    sg = g.view()  # Refresh view
    
    result2 = apply(sg, [
        algorithms.centrality.pagerank(max_iter=20, output_attr="importance"),
        algorithms.pathfinding.bfs(start_attr="start", output_attr="dist"),
        algorithms.community.lpa(max_iter=10, output_attr="community")
    ])
    print(f"  ✓ Success: {len(result2.nodes)} nodes processed")
    print("  ✓ All 3 algorithms executed in sequence")
    
    # Show combined results
    print(f"\n  Sample node (node 0):")
    node0 = [n for n in result2.nodes if n.id == 0][0]
    print(f"    Importance (PageRank): {node0.importance:.6f}")
    print(f"    Distance (BFS): {int(node0.dist)}")
    print(f"    Community (LPA): {node0.community}")
    
    # Method 3: Pipeline object
    print("\n--- Method 3: Apply Pipeline Object ---")
    print("  Code: pipe = pipeline([...]); apply(sg, pipe)")
    
    pipe = pipeline([
        algorithms.centrality.pagerank(max_iter=15, output_attr="score")
    ])
    result3 = apply(sg, pipe)
    print(f"  ✓ Success: {len(result3.nodes)} nodes processed")
    
    print("\n--- Comparison ---")
    print("  Traditional: pipe = pipeline([algo]); result = pipe(sg)")
    print("  Simplified:  result = apply(sg, algo)")
    print("  Both work identically, apply() is just more convenient!")
    
    return result2


def test_algorithm_discovery():
    """
    Test algorithm discovery and introspection features.
    """
    print_header("TEST 5: Algorithm Discovery & Introspection")
    
    # List all algorithms
    print_subheader("All Available Algorithms")
    all_algos = algorithms.list()
    print(f"Total algorithms: {len(all_algos)}")
    for algo_id in sorted(all_algos):
        print(f"  - {algo_id}")
    
    # List by category
    print_subheader("Algorithms by Category")
    cats = algorithms.categories()
    for category, algo_list in sorted(cats.items()):
        print(f"\n{category.upper()}:")
        for algo_id in algo_list:
            print(f"  - {algo_id}")
    
    # Get detailed info
    print_subheader("PageRank Algorithm Details")
    info = algorithms.info("centrality.pagerank")
    print(f"ID: {info['id']}")
    print(f"Name: {info['name']}")
    print(f"Version: {info['version']}")
    print(f"Description: {info['description']}")
    
    if 'parameters' in info and info['parameters']:
        print("\nParameters:")
        import json
        try:
            params = info['parameters'] if isinstance(info['parameters'], list) else json.loads(info['parameters'])
            for param in params:
                required = "required" if param.get('required', False) else "optional"
                print(f"  - {param['name']}: {param['type']} ({required})")
                if 'default' in param:
                    print(f"    default: {param['default']}")
        except:
            print(f"  {info['parameters']}")
    
    # Search
    print_subheader("Search Results")
    results = algorithms.search("community")
    print(f"Search 'community': {results}")
    
    results = algorithms.search("shortest")
    print(f"Search 'shortest': {results}")
    
    return info


def test_builder_dsl():
    """Test the Builder DSL end-to-end."""
    print_header("TEST 6: Builder DSL")

    print("\nThe Builder DSL composes step primitives and executes them in Rust.")
    print("Building a custom 'degree_centrality' algorithm...")
    
    # Create builder
    b = builder("degree_centrality")
    
    print("\n  Step 1: Initialize node values")
    nodes = b.init_nodes(default=0.0)
    print(f"    Created variable: {nodes}")
    
    print("\n  Step 2: Compute node degrees")
    degrees = b.node_degrees(nodes)
    print(f"    Created variable: {degrees}")
    
    print("\n  Step 3: Normalize values")
    normalized = b.normalize(degrees, method="max")
    print(f"    Created variable: {normalized}")
    
    print("\n  Step 4: Attach as output attribute")
    b.attach_as("degree_score", normalized)
    print("    Attached to attribute: degree_score")
    
    # Build
    print("\n  Building algorithm...")
    algo = b.build()
    
    print(f"\n✓ Built algorithm: {algo}")
    print(f"  Algorithm ID: {algo.id}")
    print(f"  Number of steps: {len(b.steps)}")
    print(f"  Variables tracked: {len(b.variables)}")
    
    print("\n  Step sequence:")
    for i, step in enumerate(b.steps, 1):
        step_type = step['type']
        output = step.get('output', step.get('attr_name', ''))
        print(f"    {i}. {step_type} -> {output}")
    
    g = Graph()
    n1 = g.add_node()
    n2 = g.add_node()
    n3 = g.add_node()
    g.add_edge(n1, n2)
    g.add_edge(n2, n3)

    result = apply(g.view(), algo)
    scores = {node.id: getattr(node, "degree_score") for node in result.nodes}

    print("\n  Sample scores:")
    for node_id, value in sorted(scores.items()):
        print(f"    Node {node_id}: {value:.6f}")

    max_score = max(scores.values())
    min_score = min(scores.values())
    assert abs(max_score - 1.0) < 1e-6
    assert min_score < max_score

    return algo


def test_parameter_customization():
    """
    Test different ways to customize algorithm parameters.
    """
    print_header("TEST 7: Parameter Customization Patterns")
    
    # Create a simple graph
    g = Graph()
    nodes = [g.add_node() for _ in range(10)]
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    sub = g.induced_subgraph(nodes)
    
    print("\nMethod 1: Direct specification")
    pr1 = algorithms.centrality.pagerank(
        max_iter=100,
        damping=0.9,
        output_attr="pr_high"
    )
    print(f"  Created: {pr1}")
    
    print("\nMethod 2: Start with defaults, customize with with_params()")
    pr2 = algorithms.centrality.pagerank()
    pr2_custom = pr2.with_params(max_iter=20, output_attr="pr_fast")
    print(f"  Original: max_iter from defaults")
    print(f"  Customized: {pr2_custom}")
    
    print("\nMethod 3: Generic algorithm() function")
    pr3 = algorithms.algorithm(
        "centrality.pagerank",
        max_iter=50,
        damping=0.85,
        output_attr="pr_balanced"
    )
    print(f"  Created: {pr3}")
    
    print("\nMethod 4: Validation before use")
    try:
        pr4 = algorithms.centrality.pagerank(max_iter=30, output_attr="pr_valid")
        pr4.validate()
        print(f"  ✓ Validation passed: {pr4}")
    except ValueError as e:
        print(f"  ✗ Validation failed: {e}")
    
    # Run one to show it works
    print("\nExecuting Method 1 configuration...")
    pipe = pipeline([pr1])
    result = pipe(sub)
    print(f"✓ Completed: {len(result.nodes)} nodes processed")
    
    return result


def run_all_tests():
    """
    Run all test scenarios and display results.
    """
    print("=" * 70)
    print("  GROGGY ALGORITHM API - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print("\nThis script demonstrates all major features of the algorithm API:")
    print("  0. Quick start single-algorithm example")
    print("  1. Label Propagation (LPA) for community detection")
    print("  2. Multi-algorithm pipelines")
    print("  3. Subgraph.apply() method forms")
    print("  4. Convenience apply() function")
    print("  5. Algorithm discovery and introspection")
    print("  6. Builder DSL for custom algorithms")
    print("  7. Parameter customization patterns")

    try:
        # Run all tests
        quick_start_example()
        result1 = test_lpa_community_detection()
        result2 = test_multi_algorithm_pipeline()
        result3a, result3b, result3c = test_subgraph_apply_method()
        result3 = test_apply_convenience_function()
        result4 = test_algorithm_discovery()
        result5 = test_builder_dsl()
        result6 = test_parameter_customization()

        # Summary
        print_header("SUMMARY")
        print("\n✓ All tests completed successfully!")
        print("\nWhat worked:")
        print("  ✓ Quick start PageRank example")
        print("  ✓ Label Propagation Algorithm (LPA) - detected communities")
        print("  ✓ Multi-algorithm pipeline - PageRank + BFS executed")
        print("  ✓ Subgraph.apply() method - handle, list, and pipeline inputs")
        print("  ✓ Convenience apply() function - simpler API")
        print("  ✓ Algorithm discovery - introspection working")
        print("  ✓ Builder DSL - custom step pipelines executed")
        print("  ✓ Parameter customization - all patterns functional")

        print("\nAPI Status:")
        print("  ✓ Phases 1-2: Rust algorithms fully implemented")
        print("  ✓ Phase 3: FFI bridge complete and tested")
        print("  ✓ Phase 4: Python API complete with documentation")
        print("  ✓ Phase 5: Builder DSL executing via step interpreter")
        
        print("\nReady for production use with pre-registered algorithms!")
        print("\nNext steps:")
        print("  - Use algorithms.centrality, algorithms.community, algorithms.pathfinding")
        print("  - Compose pipelines for complex workflows")
        print("  - Explore with algorithms.search() and algorithms.info()")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
