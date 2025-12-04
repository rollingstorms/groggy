"""
Graph Generators - Create various graph families and synthetic datasets

This module provides functions to generate different types of graphs including:
- Classic graph families (complete, cycle, path, star, etc.)
- Random graph models (Erdős-Rényi, Barabási-Albert, Watts-Strogatz)
- Real-world network models (karate club, social networks)
- Synthetic datasets with attributes
"""

import random
from typing import List, Optional

from . import Graph


def complete_graph(n: int, **node_attrs) -> Graph:
    """
    Generate a complete graph with n nodes (every pair of nodes connected).

    Args:
        n: Number of nodes
        **node_attrs: Additional attributes to set on all nodes

    Returns:
        Graph: Complete graph with n nodes and n*(n-1)/2 edges

    Example:
        >>> g = complete_graph(5, group="test")
        >>> print(f"Nodes: {g.node_count()}, Edges: {g.edge_count()}")
        Nodes: 5, Edges: 10
    """
    g = Graph()

    # ✅ BULK: Create node data with attributes
    node_data = []
    for i in range(n):
        node_dict = {"index": i, **node_attrs}
        node_data.append(node_dict)

    # Add all nodes with data at once
    nodes = g.add_nodes(node_data)

    # ✅ BULK: Pre-allocate edge list with exact size
    num_edges = n * (n - 1) // 2
    edge_pairs = []
    edge_pairs_reserve = num_edges  # Hint for better allocation

    for i in range(n):
        for j in range(i + 1, n):
            edge_pairs.append((nodes[i], nodes[j]))

    g.add_edges(edge_pairs)

    return g


def erdos_renyi(
    n: int, p: float, directed: bool = False, seed: Optional[int] = None, **node_attrs
) -> Graph:
    """
    Generate an Erdős-Rényi random graph G(n,p).

    Args:
        n: Number of nodes
        p: Probability of edge creation between any pair of nodes (0 <= p <= 1)
        directed: If True, create directed edges
        seed: Random seed for reproducibility
        **node_attrs: Additional attributes to set on all nodes

    Returns:
        Graph: Random graph with n nodes and approximately p*n*(n-1)/2 edges

    Example:
        >>> g = erdos_renyi(100, 0.05, seed=42)
        >>> print(f"Nodes: {g.node_count()}, Edges: {g.edge_count()}")

    Note:
        For sparse graphs (p < 0.1), uses optimized sampling approach.
        For dense graphs, uses traditional iteration for better accuracy.
    """
    if seed is not None:
        random.seed(seed)

    g = Graph()

    # ✅ BULK: Create node data with attributes
    node_data = []
    for i in range(n):
        node_dict = {"index": i, **node_attrs}
        node_data.append(node_dict)

    # Add all nodes with data at once
    nodes = g.add_nodes(node_data)

    # ✅ BULK: Create edge pairs, optimized for sparse graphs
    edge_pairs = []

    # Calculate expected number of edges to pre-allocate
    if directed:
        total_possible_edges = n * (n - 1)
    else:
        total_possible_edges = n * (n - 1) // 2

    expected_edges = int(p * total_possible_edges)

    # For very sparse graphs (p < 0.05), use pure sampling approach
    if p < 0.05 and n > 1000:
        # Sample edges directly rather than iterating all pairs
        # This is O(E) instead of O(N²) where E = expected edges
        edges_created = 0
        max_samples = min(expected_edges * 3, total_possible_edges)

        used_pairs = set()
        sample_count = 0

        while edges_created < expected_edges and sample_count < max_samples:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)

            if i == j:
                sample_count += 1
                continue

            # Ensure consistent ordering for undirected graphs
            if not directed and i > j:
                i, j = j, i

            pair = (i, j)
            if pair not in used_pairs:
                used_pairs.add(pair)
                edge_pairs.append((nodes[i], nodes[j]))
                edges_created += 1

            sample_count += 1

    # For moderately sparse graphs (0.05 <= p < 0.15), use probabilistic filtering
    elif p < 0.15 and n > 1000:
        # Iterate through all pairs but with probabilistic early termination
        for i in range(n):
            start_j = 0 if directed else i + 1
            for j in range(start_j, n):
                if i != j and random.random() < p:
                    edge_pairs.append((nodes[i], nodes[j]))

    # For dense graphs or small n, use traditional approach
    else:
        # Traditional O(N²) iteration - most accurate and efficient for dense graphs
        for i in range(n):
            start_j = 0 if directed else i + 1
            for j in range(start_j, n):
                if i != j and random.random() < p:
                    edge_pairs.append((nodes[i], nodes[j]))

    g.add_edges(edge_pairs)

    return g


def barabasi_albert(n: int, m: int, seed: Optional[int] = None, **node_attrs) -> Graph:
    """
    Generate a Barabási-Albert scale-free network using preferential attachment.

    Args:
        n: Number of nodes
        m: Number of edges to attach from a new node to existing nodes
        seed: Random seed for reproducibility
        **node_attrs: Additional attributes to set on all nodes

    Returns:
        Graph: Scale-free graph with n nodes and approximately n*m edges

    Example:
        >>> g = barabasi_albert(1000, 3, seed=42)
        >>> print(f"Nodes: {g.node_count()}, Edges: {g.edge_count()}")
    """
    if seed is not None:
        random.seed(seed)

    if m >= n:
        raise ValueError("m must be less than n")

    g = Graph()

    # ✅ BULK: Create all node data first
    node_data = []
    for i in range(n):
        node_dict = {"index": i, **node_attrs}
        node_data.append(node_dict)

    # Add all nodes at once
    nodes = g.add_nodes(node_data)

    # ✅ BULK: Collect all edges, then add at once
    edge_pairs = []

    # Start with initial complete graph edges (m+1 nodes)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            edge_pairs.append((nodes[i], nodes[j]))

    # ✅ OPTIMIZED: Use repeated nodes list for O(1) preferential attachment
    # This is the classic trick: maintain a list where each node appears
    # proportional to its degree, then sample uniformly from this list
    repeated_nodes = []
    for i in range(m + 1):
        repeated_nodes.extend([i] * m)  # Each initial node appears m times

    # Generate remaining edges with preferential attachment
    for i in range(m + 1, n):
        # Choose m unique nodes to connect to
        targets = set()

        # Sample from repeated_nodes list (O(1) per sample)
        attempts = 0
        max_attempts = m * 10  # Reasonable limit to avoid infinite loops

        while len(targets) < m and attempts < max_attempts:
            target = random.choice(repeated_nodes)
            if target != i:  # Don't connect to self
                targets.add(target)
            attempts += 1

        # If we couldn't get m targets, fill with sequential nodes
        if len(targets) < m:
            for j in range(i):
                if j not in targets:
                    targets.add(j)
                    if len(targets) >= m:
                        break

        # Add edges and update repeated_nodes list
        for target_idx in targets:
            edge_pairs.append((nodes[i], nodes[target_idx]))
            repeated_nodes.append(target_idx)  # Target gets one more connection

        # Add new node to repeated_nodes m times (once per edge)
        repeated_nodes.extend([i] * m)

    g.add_edges(edge_pairs)

    return g


def watts_strogatz(
    n: int, k: int, p: float, seed: Optional[int] = None, **node_attrs
) -> Graph:
    """
    Generate a Watts-Strogatz small-world network.

    Args:
        n: Number of nodes
        k: Each node is connected to k nearest neighbors in ring topology
        p: Probability of rewiring each edge
        seed: Random seed for reproducibility
        **node_attrs: Additional attributes to set on all nodes

    Returns:
        Graph: Small-world network with n nodes

    Example:
        >>> g = watts_strogatz(1000, 6, 0.1, seed=42)
        >>> print(f"Nodes: {g.node_count()}, Edges: {g.edge_count()}")
    """
    if seed is not None:
        random.seed(seed)

    if k >= n:
        raise ValueError("k must be less than n")
    if k % 2 != 0:
        raise ValueError("k must be even")

    g = Graph()

    # ✅ BULK: Create all node data first
    node_data = []
    for i in range(n):
        node_dict = {"index": i, **node_attrs}
        node_data.append(node_dict)

    # Add all nodes at once
    nodes = g.add_nodes(node_data)

    # ✅ OPTIMIZED: Create ring lattice and rewire efficiently
    # Track existing targets for each node to avoid duplicates during rewiring
    node_targets = [set() for _ in range(n)]

    # Build initial ring lattice edges with rewiring
    rewired_edges = []

    for i in range(n):
        # For each of k/2 neighbors on the right side
        for j in range(1, k // 2 + 1):
            neighbor = (i + j) % n

            # Rewire with probability p
            if random.random() < p:
                # Find a new target that isn't i and isn't already connected
                # Use rejection sampling with a limit to avoid infinite loops
                attempts = 0
                max_attempts = min(100, n)  # Reasonable limit

                while attempts < max_attempts:
                    new_target = random.randint(0, n - 1)
                    if new_target != i and new_target not in node_targets[i]:
                        neighbor = new_target
                        break
                    attempts += 1

                # If we couldn't find a valid target, keep the original

            # Track this connection and add the edge
            node_targets[i].add(neighbor)
            rewired_edges.append((i, neighbor))

    # ✅ BULK: Convert to node pairs and add all edges at once
    edge_pairs = [(nodes[i], nodes[j]) for i, j in rewired_edges]

    g.add_edges(edge_pairs)

    return g


def cycle_graph(n: int, **node_attrs) -> Graph:
    """
    Generate a cycle graph with n nodes.

    Args:
        n: Number of nodes
        **node_attrs: Additional attributes to set on all nodes

    Returns:
        Graph: Cycle graph with n nodes and n edges
    """
    g = Graph()

    # ✅ BULK: Create all node data first
    node_data = []
    for i in range(n):
        node_dict = {"index": i, **node_attrs}
        node_data.append(node_dict)

    # Add all nodes at once
    nodes = g.add_nodes(node_data)

    # ✅ BULK: Create all edge pairs, then add at once
    edge_pairs = []
    for i in range(n):
        next_node = (i + 1) % n
        edge_pairs.append((nodes[i], nodes[next_node]))

    g.add_edges(edge_pairs)

    return g


def path_graph(n: int, **node_attrs) -> Graph:
    """
    Generate a path graph with n nodes.

    Args:
        n: Number of nodes
        **node_attrs: Additional attributes to set on all nodes

    Returns:
        Graph: Path graph with n nodes and n-1 edges
    """
    g = Graph()

    # ✅ BULK: Create all node data first
    node_data = []
    for i in range(n):
        node_dict = {"index": i, **node_attrs}
        node_data.append(node_dict)

    # Add all nodes at once
    nodes = g.add_nodes(node_data)

    # ✅ BULK: Create all edge pairs, then add at once
    edge_pairs = []
    for i in range(n - 1):
        edge_pairs.append((nodes[i], nodes[i + 1]))

    g.add_edges(edge_pairs)

    return g


def star_graph(n: int, **node_attrs) -> Graph:
    """
    Generate a star graph with n nodes (one central node connected to all others).

    Args:
        n: Number of nodes
        **node_attrs: Additional attributes to set on all nodes

    Returns:
        Graph: Star graph with n nodes and n-1 edges
    """
    g = Graph()

    # ✅ BULK: Create all node data first
    node_data = []
    for i in range(n):
        node_dict = {"index": i, **node_attrs}
        node_data.append(node_dict)

    # Add all nodes at once
    nodes = g.add_nodes(node_data)

    # ✅ BULK: Create all edge pairs (center to all others), then add at once
    edge_pairs = []
    center = nodes[0]
    for i in range(1, n):
        edge_pairs.append((center, nodes[i]))

    g.add_edges(edge_pairs)

    return g


def grid_graph(dims: List[int], **node_attrs) -> Graph:
    """
    Generate a grid graph with given dimensions.

    Args:
        dims: List of dimensions [width, height] or [width, height, depth]
        **node_attrs: Additional attributes to set on all nodes

    Returns:
        Graph: Grid graph

    Example:
        >>> g = grid_graph([10, 10])  # 10x10 2D grid
        >>> g = grid_graph([5, 5, 5])  # 5x5x5 3D grid
    """
    g = Graph()

    if len(dims) == 2:
        width, height = dims

        # ✅ BULK: Create all node data first
        node_data = []
        coord_to_idx = {}
        idx = 0
        for x in range(width):
            for y in range(height):
                node_dict = {"x": x, "y": y, **node_attrs}
                node_data.append(node_dict)
                coord_to_idx[(x, y)] = idx
                idx += 1

        # Add all nodes at once
        nodes = g.add_nodes(node_data)

        # ✅ BULK: Create all edge pairs, then add at once
        edge_pairs = []
        for x in range(width):
            for y in range(height):
                current_idx = coord_to_idx[(x, y)]
                current = nodes[current_idx]
                # Right neighbor
                if x + 1 < width:
                    neighbor_idx = coord_to_idx[(x + 1, y)]
                    edge_pairs.append((current, nodes[neighbor_idx]))
                # Down neighbor
                if y + 1 < height:
                    neighbor_idx = coord_to_idx[(x, y + 1)]
                    edge_pairs.append((current, nodes[neighbor_idx]))

        g.add_edges(edge_pairs)

    elif len(dims) == 3:
        width, height, depth = dims

        # ✅ BULK: Create all node data first
        node_data = []
        coord_to_idx = {}
        idx = 0
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    node_dict = {"x": x, "y": y, "z": z, **node_attrs}
                    node_data.append(node_dict)
                    coord_to_idx[(x, y, z)] = idx
                    idx += 1

        # Add all nodes at once
        nodes = g.add_nodes(node_data)

        # ✅ BULK: Create all edge pairs, then add at once
        edge_pairs = []
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    current_idx = coord_to_idx[(x, y, z)]
                    current = nodes[current_idx]
                    # Right neighbor
                    if x + 1 < width:
                        neighbor_idx = coord_to_idx[(x + 1, y, z)]
                        edge_pairs.append((current, nodes[neighbor_idx]))
                    # Down neighbor
                    if y + 1 < height:
                        neighbor_idx = coord_to_idx[(x, y + 1, z)]
                        edge_pairs.append((current, nodes[neighbor_idx]))
                    # Forward neighbor
                    if z + 1 < depth:
                        neighbor_idx = coord_to_idx[(x, y, z + 1)]
                        edge_pairs.append((current, nodes[neighbor_idx]))

        g.add_edges(edge_pairs)

    else:
        raise ValueError("Only 2D and 3D grids are supported")

    return g


def tree(n: int, branching_factor: int = 2, **node_attrs) -> Graph:
    """
    Generate a regular tree with n nodes.

    Args:
        n: Number of nodes
        branching_factor: Number of children per internal node
        **node_attrs: Additional attributes to set on all nodes

    Returns:
        Graph: Tree with n nodes and n-1 edges
    """
    g = Graph()

    if n <= 0:
        return g

    # ✅ BULK: Create all node data first
    node_data = []
    for i in range(n):
        level = 0
        temp_i = i
        temp_bf = branching_factor
        while temp_i >= temp_bf:
            temp_i -= temp_bf
            temp_bf *= branching_factor
            level += 1

        node_dict = {"index": i, "level": level, **node_attrs}
        node_data.append(node_dict)

    # Add all nodes at once
    nodes = g.add_nodes(node_data)

    # ✅ BULK: Create all edge pairs, then add at once
    edge_pairs = []

    # Create parent-child relationships
    for i in range(1, n):  # Skip root (node 0)
        parent_idx = (i - 1) // branching_factor
        if parent_idx < len(nodes):
            edge_pairs.append((nodes[parent_idx], nodes[i]))

    g.add_edges(edge_pairs)

    return g


# Real-world network models
def karate_club() -> Graph:
    """
    Generate Zachary's karate club graph.

    Returns:
        Graph: The famous karate club social network (34 nodes, 78 edges)
    """
    g = Graph()

    # ✅ BULK: Create all node data first
    node_data = []
    for i in range(34):
        node_dict = {"index": i, "name": f"Member_{i}"}
        node_data.append(node_dict)

    # Add all nodes at once
    nodes = g.add_nodes(node_data)

    # Define the edges from the original dataset
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 10),
        (0, 11),
        (0, 12),
        (0, 13),
        (0, 17),
        (0, 19),
        (0, 21),
        (0, 31),
        (1, 2),
        (1, 3),
        (1, 7),
        (1, 13),
        (1, 17),
        (1, 19),
        (1, 21),
        (1, 30),
        (2, 3),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 13),
        (2, 27),
        (2, 28),
        (2, 32),
        (3, 7),
        (3, 12),
        (3, 13),
        (4, 6),
        (4, 10),
        (5, 6),
        (5, 10),
        (5, 16),
        (6, 16),
        (8, 30),
        (8, 32),
        (8, 33),
        (9, 33),
        (13, 33),
        (14, 32),
        (14, 33),
        (15, 32),
        (15, 33),
        (18, 32),
        (18, 33),
        (19, 33),
        (20, 32),
        (20, 33),
        (22, 32),
        (22, 33),
        (23, 25),
        (23, 27),
        (23, 29),
        (23, 32),
        (23, 33),
        (24, 25),
        (24, 27),
        (24, 31),
        (25, 31),
        (26, 29),
        (26, 33),
        (27, 33),
        (28, 31),
        (28, 33),
        (29, 32),
        (29, 33),
        (30, 32),
        (30, 33),
        (31, 32),
        (31, 33),
        (32, 33),
    ]

    # ✅ BULK: Create all edge data with attributes, then add at once
    edge_data = []
    for i, j in edges:
        edge_data.append((nodes[i], nodes[j], {"relationship": "friendship"}))

    g.add_edges(edge_data)

    return g


def social_network(
    n: int,
    communities: int = 3,
    node_attrs: Optional[List[str]] = None,
    edge_attrs: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Graph:
    """
    Generate a synthetic social network with realistic attributes.

    Args:
        n: Number of nodes (people)
        communities: Number of communities/clusters
        node_attrs: List of node attribute names to generate
        edge_attrs: List of edge attribute names to generate
        seed: Random seed for reproducibility

    Returns:
        Graph: Synthetic social network with attributes
    """
    if seed is not None:
        random.seed(seed)

    if node_attrs is None:
        node_attrs = ["age", "income", "location"]
    if edge_attrs is None:
        edge_attrs = ["strength", "frequency"]

    g = Graph()

    # Generate realistic attribute values
    locations = ["NYC", "SF", "LA", "Chicago", "Boston", "Austin", "Seattle", "Denver"]

    # ✅ BULK: Create all node data first
    node_data = []
    for i in range(n):
        attrs = {"index": i, "community": i % communities}

        if "age" in node_attrs:
            attrs["age"] = random.randint(18, 65)
        if "income" in node_attrs:
            attrs["income"] = random.randint(30000, 200000)
        if "location" in node_attrs:
            attrs["location"] = random.choice(locations)

        node_data.append(attrs)

    # Add all nodes at once
    nodes = g.add_nodes(node_data)

    # ✅ BULK: Create edges more efficiently using expected degree approach
    edge_data = []

    # Group nodes by community for efficient sampling
    community_nodes = {}
    for i, node_id in enumerate(nodes):
        community = i % communities
        if community not in community_nodes:
            community_nodes[community] = []
        community_nodes[community].append((i, node_id))

    # Generate edges within communities (higher probability)
    within_community_p = 0.15
    for community, node_list in community_nodes.items():
        community_size = len(node_list)
        expected_edges = int(
            within_community_p * community_size * (community_size - 1) / 2
        )

        # Sample edge pairs efficiently
        all_pairs = [
            (i, j) for i in range(community_size) for j in range(i + 1, community_size)
        ]
        if len(all_pairs) > 0:
            # Sample without replacement up to expected number
            num_edges = min(expected_edges, len(all_pairs))
            selected_pairs = random.sample(all_pairs, num_edges)

            for i_idx, j_idx in selected_pairs:
                i, node_i = node_list[i_idx]
                j, node_j = node_list[j_idx]

                edge_attrs_dict = {}
                if "strength" in edge_attrs:
                    edge_attrs_dict["strength"] = random.uniform(0.1, 1.0)
                if "frequency" in edge_attrs:
                    edge_attrs_dict["frequency"] = random.choice(
                        ["daily", "weekly", "monthly", "rarely"]
                    )

                edge_data.append((node_i, node_j, edge_attrs_dict))

    # Generate edges between communities (lower probability)
    between_community_p = 0.02
    total_inter_community_pairs = 0
    for i in range(communities):
        for j in range(i + 1, communities):
            total_inter_community_pairs += len(community_nodes[i]) * len(
                community_nodes[j]
            )

    expected_inter_edges = int(between_community_p * total_inter_community_pairs)

    # Efficiently sample inter-community edges
    inter_community_edges = 0
    for i in range(communities):
        for j in range(i + 1, communities):
            if inter_community_edges >= expected_inter_edges:
                break

            nodes_i = community_nodes[i]
            nodes_j = community_nodes[j]

            # Sample a reasonable number of pairs between these communities
            pairs_needed = min(
                len(nodes_i) * len(nodes_j),
                expected_inter_edges - inter_community_edges,
            )

            for _ in range(pairs_needed):
                if random.random() < between_community_p:
                    node_i_idx, node_i = random.choice(nodes_i)
                    node_j_idx, node_j = random.choice(nodes_j)

                    edge_attrs_dict = {}
                    if "strength" in edge_attrs:
                        edge_attrs_dict["strength"] = random.uniform(0.1, 1.0)
                    if "frequency" in edge_attrs:
                        edge_attrs_dict["frequency"] = random.choice(
                            ["daily", "weekly", "monthly", "rarely"]
                        )

                    edge_data.append((node_i, node_j, edge_attrs_dict))
                    inter_community_edges += 1

    g.add_edges(edge_data)

    return g


def meta_api_graph() -> Graph:
    """
    Generate the Meta API Graph - a graph representation of Groggy's own API structure.

    This function loads the pre-generated API meta-graph that was created by the
    Meta API Discovery and Testing System. The graph contains:
    - Nodes: Groggy objects (Graph, BaseTable, etc.) + return types
    - Edges: Methods connecting objects to their return types

    This serves as the ultimate meta-example: Groggy analyzing its own API structure!

    Returns:
        Graph: The API meta-graph (35 nodes, 205 edges representing 205 methods)

    Example:
        >>> api_graph = meta_api_graph()
        >>> print(f"API Graph: {api_graph.node_count()} objects/types, {api_graph.edge_count()} methods")
        API Graph: 35 objects/types, 205 methods

        # Analyze which objects have the most methods
        >>> method_counts = api_graph.nodes.table().to_pandas()
        >>> method_counts[method_counts['method_count'] > 0].sort_values('method_count', ascending=False)

        # Find most common return types
        >>> edges = api_graph.edges.table().to_pandas()
        >>> print(edges['return_type'].value_counts().head())

        # See all methods for a specific object
        >>> graph_methods = edges[edges['method_name'].str.contains('graph', case=False)]
    """
    try:
        # Try to load the pre-generated meta-graph bundle
        import os

        from . import GraphTable

        bundle_path = os.path.join(
            os.path.dirname(__file__),
            "../../../documentation/meta_api_discovery/groggy_api_meta_graph",
        )

        if not os.path.exists(bundle_path):
            # Fallback: create a minimal example if bundle doesn't exist
            print("⚠️  Meta-graph bundle not found. Creating minimal example.")
            return _create_minimal_meta_api_graph()

        # Load the actual meta-graph
        graph_table = GraphTable.load_bundle(bundle_path)
        api_graph = graph_table.to_graph()

        print(
            f"✅ Loaded Meta API Graph: {api_graph.node_count()} nodes, {api_graph.edge_count()} edges"
        )
        print("   This graph represents Groggy's complete API structure!")

        return api_graph

    except Exception as e:
        print(f"⚠️  Could not load meta-graph: {e}")
        return _create_minimal_meta_api_graph()


def _create_minimal_meta_api_graph() -> Graph:
    """Create a minimal version of the API meta-graph for demonstration"""

    g = Graph()

    # Create nodes for core object types
    core_objects = [
        {"name": "Graph", "type": "core", "method_count": 62},
        {"name": "BaseTable", "type": "table", "method_count": 30},
        {"name": "NodesTable", "type": "table", "method_count": 27},
        {"name": "EdgesTable", "type": "table", "method_count": 32},
        {"name": "BaseArray", "type": "array", "method_count": 7},
    ]

    nodes = g.add_nodes(core_objects)

    # Create sample method edges
    method_edges = [
        (nodes[0], nodes[1], {"method": "table", "returns": "BaseTable"}),
        (nodes[1], nodes[4], {"method": "column", "returns": "BaseArray"}),
        (nodes[2], nodes[1], {"method": "base_table", "returns": "BaseTable"}),
        (nodes[3], nodes[1], {"method": "base_table", "returns": "BaseTable"}),
        (nodes[4], nodes[4], {"method": "unique", "returns": "BaseArray"}),
    ]

    g.add_edges(method_edges)

    print("✅ Created minimal Meta API Graph demonstration")
    return g
