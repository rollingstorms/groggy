"""
Community detection algorithms for finding clusters in graphs.

This module provides algorithms for detecting communities and clusters,
including Label Propagation, Louvain, Leiden, and Connected Components.
"""

from typing import Literal, Optional

from groggy.algorithms.base import RustAlgorithmHandle


def lpa(
    max_iter: int = 100, seed: Optional[int] = None, output_attr: str = "community"
) -> RustAlgorithmHandle:
    """
    Label Propagation Algorithm for community detection.

    Detects communities by propagating labels through the network. Each node
    adopts the most common label among its neighbors.

    Args:
        max_iter: Maximum iterations (default: 100)
        seed: Random seed for reproducibility (optional)
        output_attr: Attribute name for community labels (default: "community")

    Returns:
        Algorithm handle configured for LPA

    Example:
        >>> from groggy.algorithms import community
        >>> lpa_algo = community.lpa(max_iter=50)
        >>> result = subgraph.apply(lpa_algo)
    """
    params = {
        "max_iter": max_iter,
        "output_attr": output_attr,
    }
    if seed is not None:
        params["seed"] = seed

    return RustAlgorithmHandle("community.lpa", params)


def louvain(
    resolution: float = 1.0,
    max_iter: int = 100,
    output_attr: str = "community",
    max_phases: int = 10,
) -> RustAlgorithmHandle:
    """
    Louvain modularity optimization for community detection.

    Finds communities by optimizing modularity, a measure of network
    structure quality. Supports undirected unweighted multigraphs, retaining
    parallel instances and self-loops. Local moves alternate with aggregation;
    internal coarse-edge multiplicities preserve the original objective.

    Args:
        resolution: Finite positive gamma applied to the modularity null-model term
        max_iter: Maximum node-move sweeps per level (default: 100)
        max_phases: Maximum levels (default: 10); one requests local moves only
        output_attr: Attribute name for community labels (default: "community")

    Returns:
        Algorithm handle configured for Louvain

    Example:
        >>> from groggy.algorithms import community
        >>> louv = community.louvain(resolution=1.5)
        >>> result = subgraph.apply(louv)
    """
    params = {
        "resolution": resolution,
        "max_iter": max_iter,
        "max_phases": max_phases,
        "output_attr": output_attr,
    }

    return RustAlgorithmHandle("community.louvain", params)


def leiden(
    resolution: float = 1.0,
    max_iter: int = 20,
    max_phases: int = 10,
    seed: Optional[int] = None,
    output_attr: str = "community",
    theta: float = 0.01,
) -> RustAlgorithmHandle:
    """
    Leiden community detection algorithm.

    Runs bounded fast moves, constrained randomized refinement and refined
    aggregation on undirected unweighted multigraphs. Parallel instances and
    self-loops are retained. The broader partition initializes each coarse graph.

    Args:
        resolution: Finite positive gamma applied to the modularity null-model term
        max_iter: Node-visit budget per level as a multiple of level size (default: 20)
        max_phases: Maximum aggregation levels (default: 10)
        seed: Nonnegative local RNG seed; omitted uses reproducible seed zero
        output_attr: Attribute name for community labels (default: "community")
        theta: Positive finite refinement temperature in normalized modularity units

    Returns:
        Algorithm handle configured for Leiden

    Example:
        >>> from groggy.algorithms import community
        >>> leiden_algo = community.leiden(resolution=1.5, max_iter=20)
        >>> result = subgraph.apply(leiden_algo)
    """
    params = {
        "resolution": resolution,
        "max_iter": max_iter,
        "max_phases": max_phases,
        "output_attr": output_attr,
        "theta": theta,
    }
    if seed is not None:
        params["seed"] = seed

    return RustAlgorithmHandle("community.leiden", params)


def connected_components(
    mode: Literal["undirected", "weak", "strong"] = "undirected",
    output_attr: str = "component",
) -> RustAlgorithmHandle:
    """
    Find connected components in a graph.

    Uses efficient Union-Find for undirected/weak connectivity (O(m α(n))),
    or Tarjan's algorithm for strongly connected components (O(m + n)).

    Args:
        mode: Connectivity mode (default: "undirected")
            - "undirected": Ignores edge direction
            - "weak": Directed graph, weakly connected (ignores direction)
            - "strong": Directed graph, strongly connected (respects direction)
        output_attr: Attribute name for component IDs (default: "component")

    Returns:
        Algorithm handle configured for connected components

    Example:
        >>> from groggy.algorithms import community
        >>> cc = community.connected_components(mode="strong")
        >>> result = subgraph.apply(cc)
    """
    params = {
        "mode": mode,
        "output_attr": output_attr,
    }

    return RustAlgorithmHandle("community.connected_components", params)


def girvan_newman(
    num_levels: Optional[int] = None,
    modularity_threshold: Optional[float] = None,
    weight_attr: Optional[str] = None,
    output_attr: str = "community",
) -> RustAlgorithmHandle:
    """Find a partition of an undirected graph by edge betweenness removal.

    Removes individual edge instances locally and returns the best visited
    partition under unweighted modularity of the original selected graph.
    Equal removal scores choose the lowest edge ID; equal partition scores
    retain the earlier partition. Graph topology is not changed.

    Args:
        num_levels: Maximum individual non-loop edge removals; zero returns
            initial components. Default is all non-loop edges, subject to stopping.
        modularity_threshold: Stop when modularity falls below the best by more
            than this finite nonnegative amount (default: 0.0001).
        weight_attr: Optional strictly positive finite numeric edge lengths;
            missing values are one. Lengths do not weight modularity.
        output_attr: Node attribute for community labels.

    Intended for small graphs. Directed graphs are currently unsupported.
    """
    params = {"output_attr": output_attr}
    if num_levels is not None:
        params["num_levels"] = num_levels
    if modularity_threshold is not None:
        params["modularity_threshold"] = modularity_threshold
    if weight_attr is not None:
        params["weight_attr"] = weight_attr
    return RustAlgorithmHandle("community.girvan_newman", params)
