"""
Examples demonstrating the Groggy algorithm API.

This module provides complete, runnable examples of how to use algorithms,
pipelines, and the builder DSL.
"""

from groggy import Graph, algorithms, pipeline


def example_single_algorithm():
    """
    Run a single algorithm on a graph.

    This is the simplest way to use algorithms.
    """
    # Create a graph
    g = Graph()
    nodes = [g.add_node() for _ in range(20)]

    # Add edges
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])

    # Create subgraph
    sub = g.induced_subgraph(nodes)

    # Create and configure algorithm
    pagerank = algorithms.centrality.pagerank(
        max_iter=50, damping=0.85, output_attr="pagerank"
    )

    # Create pipeline with single algorithm
    pipe = pipeline([pagerank])

    # Run the pipeline
    result = pipe(sub)

    print(f"Processed {len(result.nodes)} nodes")
    return result


def example_multi_algorithm_pipeline():
    """
    Compose multiple algorithms into a pipeline.

    Each algorithm's output becomes the input to the next.
    """
    # Create a graph
    g = Graph()
    nodes = [g.add_node() for _ in range(50)]

    # Create a connected graph
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    g.add_edge(nodes[-1], nodes[0])  # Make it cyclic

    # Mark a start node for pathfinding
    g.nodes.set_attrs({nodes[0]: {"is_start": True}})

    # Create subgraph
    sub = g.induced_subgraph(nodes)

    # Build a multi-algorithm pipeline
    pipe = pipeline(
        [
            # First, compute PageRank
            algorithms.centrality.pagerank(max_iter=50, output_attr="importance"),
            # Then, run BFS from start node
            algorithms.pathfinding.bfs(start_attr="is_start", output_attr="distance"),
            # Finally, detect communities
            algorithms.community.lpa(max_iter=20, output_attr="community"),
        ]
    )

    # Execute the pipeline
    result = pipe(sub)

    print(f"Pipeline executed {len(pipe)} algorithms")
    print(f"Processed {len(result.nodes)} nodes")
    return result


def example_algorithm_discovery():
    """
    Discover and explore available algorithms.

    Use the discovery API to find algorithms dynamically.
    """
    # List all algorithms
    all_algorithms = algorithms.list()
    print(f"Total algorithms available: {len(all_algorithms)}")

    # List by category
    centrality_algos = algorithms.list(category="centrality")
    print(f"Centrality algorithms: {centrality_algos}")

    # Get all categories
    cats = algorithms.categories()
    print(f"Categories: {list(cats.keys())}")

    # Search for algorithms
    results = algorithms.search("shortest path")
    print(f"Path-related algorithms: {results}")

    # Get detailed info about an algorithm
    info = algorithms.info("centrality.pagerank")
    print(f"\nPageRank info:")
    print(f"  Description: {info['description']}")
    print(f"  Version: {info['version']}")
    print(f"  Parameters: {len(info.get('parameters', []))} params")

    return info


def example_parameter_customization():
    """
    Customize algorithm parameters.

    Shows different ways to configure algorithms.
    """
    # Create graph
    g = Graph()
    nodes = [g.add_node() for _ in range(30)]
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    sub = g.induced_subgraph(nodes)

    # Method 1: Direct parameter specification
    pr1 = algorithms.centrality.pagerank(
        max_iter=100, damping=0.9, tolerance=1e-8, output_attr="pr_high_iter"
    )

    # Method 2: Start with defaults, then customize
    pr2 = algorithms.centrality.pagerank()
    pr2 = pr2.with_params(max_iter=50, output_attr="pr_custom")

    # Method 3: Use the generic algorithm function
    pr3 = algorithms.algorithm(
        "centrality.pagerank", max_iter=20, output_attr="pr_fast"
    )

    # Run each configuration
    results = []
    for algo in [pr1, pr2, pr3]:
        pipe = pipeline([algo])
        result = pipe(sub)
        results.append(result)

    print(f"Ran {len(results)} different configurations")
    return results


def example_error_handling():
    """
    Handle errors in algorithm execution.

    Shows proper error handling patterns.
    """
    from groggy import AttrValue

    try:
        # This will fail - algorithm doesn't exist
        bad_algo = algorithms.algorithm("nonexistent.algorithm")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    try:
        # This will fail - invalid parameter using generic algorithm function
        bad_params = algorithms.algorithm("centrality.pagerank", unknown_param=123)
        bad_params.validate()
    except ValueError as e:
        print(f"✓ Caught validation error: {e}")

    try:
        # This will fail - missing required parameter
        # Use generic algorithm function to bypass Python required args
        bfs_bad = algorithms.algorithm("pathfinding.bfs", output_attr="distance")
        # Missing start_attr
        bfs_bad.validate()
    except ValueError as e:
        print(f"✓ Caught missing parameter error: {e}")

    print("\n✓ All error handling tests passed")


def example_algorithm_reuse():
    """
    Reuse algorithm handles across multiple graphs.

    Algorithm handles are reusable and thread-safe.
    """
    # Create a reusable algorithm configuration
    my_pagerank = algorithms.centrality.pagerank(
        max_iter=30, damping=0.85, output_attr="rank"
    )

    # Apply to multiple graphs
    results = []
    for size in [10, 20, 30]:
        g = Graph()
        nodes = [g.add_node() for _ in range(size)]
        for i in range(size - 1):
            g.add_edge(nodes[i], nodes[i + 1])

        sub = g.induced_subgraph(nodes)
        pipe = pipeline([my_pagerank])
        result = pipe(sub)
        results.append(result)

    print(f"✓ Reused algorithm on {len(results)} graphs")
    return results


def run_all_examples():
    """
    Run all examples.

    Demonstrates the complete API surface.
    """
    print("=" * 60)
    print("Groggy Algorithm API Examples")
    print("=" * 60)

    print("\n1. Single Algorithm")
    print("-" * 60)
    example_single_algorithm()

    print("\n2. Multi-Algorithm Pipeline")
    print("-" * 60)
    example_multi_algorithm_pipeline()

    print("\n3. Algorithm Discovery")
    print("-" * 60)
    example_algorithm_discovery()

    print("\n4. Parameter Customization")
    print("-" * 60)
    example_parameter_customization()

    print("\n5. Error Handling")
    print("-" * 60)
    example_error_handling()

    print("\n6. Algorithm Reuse")
    print("-" * 60)
    example_algorithm_reuse()

    print("\n" + "=" * 60)
    print("✓ All examples completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
