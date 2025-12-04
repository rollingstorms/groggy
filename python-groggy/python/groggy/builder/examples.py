"""
Example algorithms using the new builder DSL syntax.

This module demonstrates how to write graph algorithms with:
- Operator overloading for natural mathematical expressions
- Domain-specific traits (GraphOps, AttrOps, IterOps)
- @algorithm decorator for clean definitions
"""

from groggy.builder.decorators import algorithm


@algorithm("pagerank_new")
def pagerank(sG, damping=0.85, max_iter=100, tol=1e-6):
    """
    PageRank algorithm using new DSL syntax.

    Computes the PageRank score for each node, representing the probability
    of arriving at that node via random walks with damping.

    Args:
        sG: Subgraph view to process (GraphHandle)
        damping: Damping factor (typically 0.85)
        max_iter: Maximum iterations
        tol: Convergence tolerance (not yet used)

    Returns:
        VarHandle with PageRank scores (normalized)

    Example:
        >>> pr = pagerank(damping=0.85, max_iter=100)
        >>> result = graph.view().apply(pr)
        >>> scores = result.nodes()["pagerank_new"]
    """
    # Initialize ranks uniformly
    ranks = sG.nodes(1.0)
    n_scalar = sG.N
    inv_n_scalar = sG.builder.core.recip(n_scalar, epsilon=1e-9)
    ranks = sG.builder.core.broadcast_scalar(inv_n_scalar, ranks)
    ranks = sG.var("ranks", ranks)  # Create logical variable for loop

    # Compute degrees and identify sinks
    deg = ranks.degrees()
    inv_deg = 1.0 / (deg + 1e-9)
    is_sink = deg == 0.0

    # Pre-broadcast inv_n to avoid scalar*scalar operations
    inv_n_map = sG.builder.core.broadcast_scalar(inv_n_scalar, ranks)

    # Iterative update
    with sG.iterate(max_iter):
        # Contribution from each node (sinks contribute 0)
        contrib = is_sink.where(0.0, ranks * inv_deg)

        # Aggregate neighbor contributions
        neighbor_sum = sG @ contrib

        # Redistribute sink mass uniformly
        sink_mass = is_sink.where(ranks, 0.0).reduce("sum")

        # PageRank formula components
        # 1. Damped neighbor contribution
        damped_neighbor = neighbor_sum * damping

        # 2. Teleport term: (1-damping)/N
        teleport = inv_n_map * (1 - damping)

        # 3. Sink redistribution: damping * sink_mass / N
        # Broadcast sink_mass, multiply by inv_n_map, then by damping
        sink_mass_map = sG.builder.core.broadcast_scalar(sink_mass, ranks)
        sink_contrib = sink_mass_map * inv_n_map * damping

        ranks = sG.var("ranks", damped_neighbor + teleport + sink_contrib)

    # Normalize to sum to 1
    return ranks.normalize()


@algorithm("pagerank_simple")
def pagerank_simple(sG, damping=0.85, max_iter=100):
    """
    Simplified PageRank without sink handling.

    This version assumes no sink nodes (all nodes have out-degree > 0).
    Simpler but less robust than full PageRank.

    Args:
        sG: Subgraph view to process (GraphHandle)
        damping: Damping factor
        max_iter: Maximum iterations

    Returns:
        VarHandle with PageRank scores
    """
    # Initialize ranks uniformly: 1/N per node
    ranks = sG.nodes(1.0)
    n_scalar = sG.N
    inv_n_scalar = sG.builder.core.recip(n_scalar, epsilon=1e-9)
    ranks = sG.builder.core.broadcast_scalar(inv_n_scalar, ranks)
    ranks = sG.var("ranks", ranks)

    deg = ranks.degrees()

    # Compute teleport value: (1-damping)/N as a map
    inv_n_map = sG.builder.core.broadcast_scalar(inv_n_scalar, ranks)
    teleport_map = inv_n_map * (1.0 - damping)

    with sG.iterate(max_iter):
        # Each node distributes rank evenly to neighbors
        contrib = ranks / (deg + 1e-9)
        neighbor_sum = sG @ contrib

        # Update with damping
        ranks = sG.var("ranks", damping * neighbor_sum + teleport_map)

    return ranks.normalize()


@algorithm("label_propagation")
def label_propagation(sG, max_iter=10, ordered=True):
    """
    Asynchronous Label Propagation Algorithm (LPA).

    Each node adopts the most common label among its neighbors.
    Uses asynchronous updates for faster convergence.

    Args:
        sG: Subgraph view to process (GraphHandle)
        max_iter: Maximum iterations
        ordered: Process nodes in sorted order for determinism

    Returns:
        VarHandle with community labels

    Example:
        >>> lpa = label_propagation(max_iter=20)
        >>> result = graph.view().apply(lpa)
        >>> labels = result.nodes()["label_propagation"]
    """
    # Initialize each node with unique label
    labels = sG.nodes(unique=True)

    # Iteratively update labels to match neighbor mode
    with sG.iterate(max_iter):
        labels = sG.builder.graph_ops.neighbor_mode_update(
            labels, include_self=True, ordered=ordered
        )

    return labels


@algorithm("label_propagation_sync")
def label_propagation_sync(sG, max_iter=10):
    """
    Synchronous Label Propagation Algorithm.

    Updates all nodes simultaneously (less efficient than async version).

    Args:
        sG: Subgraph view to process (GraphHandle)
        max_iter: Maximum iterations

    Returns:
        VarHandle with community labels
    """
    labels = sG.nodes(unique=True)

    with sG.iterate(max_iter):
        # Collect neighbor labels
        neighbor_labels = sG.builder.graph_ops.collect_neighbor_values(
            labels, include_self=True
        )

        # Take mode (most common)
        new_labels = sG.builder.core.mode(neighbor_labels)

        # Synchronous update
        labels = sG.var("labels", new_labels)

    return labels


@algorithm("degree_centrality")
def degree_centrality(sG, normalized=True):
    """
    Degree centrality - simplest centrality measure.

    Args:
        sG: Subgraph view to process (GraphHandle)
        normalized: If True, normalize by max possible degree

    Returns:
        VarHandle with degree centrality scores
    """
    # Get degrees
    degrees = sG.builder.graph_ops.degree()

    if normalized:
        # Normalize by (N-1) - max possible degree
        max_degree = sG.N - 1
        centrality = degrees / max_degree
    else:
        centrality = degrees

    return centrality


@algorithm("weighted_degree")
def weighted_degree(sG, weight_attr="weight", default_weight=1.0):
    """
    Weighted degree centrality.

    Args:
        sG: Subgraph view to process (GraphHandle)
        weight_attr: Name of edge weight attribute
        default_weight: Default weight for edges without attribute

    Returns:
        VarHandle with weighted degrees
    """
    # Load edge weights
    edge_weights = sG.builder.attr.load_edge(weight_attr, default=default_weight)

    # Initialize node values to 1 (to count edges)
    ones = sG.nodes(1.0)

    # Sum weighted edges per node
    weighted_deg = sG.builder.graph_ops.neighbor_agg(
        ones, agg="sum", weights=edge_weights
    )

    return weighted_deg


@algorithm("node_attribute_propagation")
def node_attribute_propagation(sG, attr_name, max_iter=10, damping=0.5):
    """
    Propagate node attributes through the graph.

    Useful for filling in missing values based on neighbors.

    Args:
        sG: Subgraph view to process (GraphHandle)
        attr_name: Attribute to propagate
        max_iter: Maximum iterations
        damping: How much to retain original value (0-1)

    Returns:
        VarHandle with propagated attributes
    """
    # Load initial attributes
    values = sG.builder.attr.load(attr_name, default=0.0)
    initial = values  # Keep original values

    with sG.iterate(max_iter):
        # Average neighbor values
        neighbor_avg = sG.builder.graph_ops.neighbor_agg(values, agg="mean")

        # Blend with original
        values = sG.var("values", (1 - damping) * neighbor_avg + damping * initial)

    return values


# Comparison: Old vs New Syntax
# ==============================


def pagerank_old_style():
    """
    PageRank using old verbose syntax (for comparison).

    This shows how algorithms looked before the refactor.
    """
    from groggy.builder import AlgorithmBuilder

    builder = AlgorithmBuilder("pagerank_old")

    # Initialize
    n = builder.graph_node_count()
    ranks = builder.init_nodes(1.0)
    inv_n = builder.core.recip(n, 1e-9)
    uniform = builder.core.broadcast_scalar(inv_n, ranks)
    ranks = builder.var("ranks", uniform)

    # Get degrees
    deg = builder.node_degrees(ranks)
    inv_deg = builder.core.recip(deg, 1e-9)
    is_sink = builder.core.compare(deg, "eq", 0.0)

    # Iterate
    with builder.iterate(100):
        # Contribution
        contrib = builder.core.mul(ranks, inv_deg)
        contrib = builder.core.where(is_sink, 0.0, contrib)

        # Aggregate
        neighbor_sum = builder.core.neighbor_agg(contrib, "sum")

        # Apply damping
        damped = builder.core.mul(neighbor_sum, 0.85)

        # Teleport
        teleport_val = builder.core.mul(inv_n, 0.15)
        teleport = builder.core.broadcast_scalar(teleport_val, deg)

        # Sink mass
        sink_mass_vals = builder.core.where(is_sink, ranks, 0.0)
        sink_mass = builder.core.reduce_scalar(sink_mass_vals, "sum")
        sink_contrib_val = builder.core.mul(builder.core.mul(inv_n, sink_mass), 0.85)
        sink_contrib = builder.core.broadcast_scalar(sink_contrib_val, deg)

        # Combine
        new_ranks = builder.core.add(builder.core.add(damped, teleport), sink_contrib)
        ranks = builder.var("ranks", new_ranks)

    # Normalize
    normalized = builder.core.normalize_sum(ranks)
    builder.attach_as("pagerank", normalized)

    return builder.build()


# Export all algorithms
__all__ = [
    "pagerank",
    "pagerank_simple",
    "label_propagation",
    "label_propagation_sync",
    "degree_centrality",
    "weighted_degree",
    "node_attribute_propagation",
    "pagerank_old_style",
]
