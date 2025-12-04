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
    resolution: float = 1.0, max_iter: int = 100, output_attr: str = "community"
) -> RustAlgorithmHandle:
    """
    Louvain modularity optimization for community detection.

    Finds communities by optimizing modularity, a measure of network
    structure quality. Tends to find more cohesive communities than LPA.

    Args:
        resolution: Resolution parameter (default: 1.0)
        max_iter: Maximum iterations (default: 100)
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
        "output_attr": output_attr,
    }

    return RustAlgorithmHandle("community.louvain", params)


def leiden(
    resolution: float = 1.0,
    max_iter: int = 20,
    max_phases: int = 10,
    seed: Optional[int] = None,
    output_attr: str = "community",
) -> RustAlgorithmHandle:
    """
    Leiden community detection algorithm.

    An improvement over Louvain that guarantees connected communities and
    typically converges faster with better quality. Adds a refinement phase
    that splits poorly connected communities.

    Args:
        resolution: Resolution parameter for modularity (default: 1.0)
        max_iter: Maximum node-move iterations per phase (default: 20)
        max_phases: Maximum number of refinement phases (default: 10)
        seed: Random seed for reproducibility (optional)
        output_attr: Attribute name for community labels (default: "community")

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
