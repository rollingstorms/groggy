"""
Centrality algorithms for measuring node importance.

This module provides access to centrality algorithms like PageRank, Betweenness,
and Closeness centrality.
"""

from typing import Optional

from groggy.algorithms.base import RustAlgorithmHandle


def pagerank(
    damping: float = 0.85,
    max_iter: int = 100,
    tolerance: float = 1e-6,
    personalization_attr: Optional[str] = None,
    output_attr: str = "pagerank",
) -> RustAlgorithmHandle:
    """
    PageRank centrality algorithm.

    Computes the PageRank score for each node, representing its importance
    based on the graph's link structure.

    Args:
        damping: Damping factor (default: 0.85)
        max_iter: Maximum iterations (default: 100)
        tolerance: Convergence tolerance (default: 1e-6)
        personalization_attr: Optional node attribute for personalized PageRank
        output_attr: Attribute name for results (default: "pagerank")

    Returns:
        Algorithm handle configured for PageRank

    Example:
        >>> from groggy.algorithms import centrality
        >>> pr = centrality.pagerank(max_iter=50, damping=0.9)
        >>> result = subgraph.apply(pr)
    """
    params = {
        "damping": damping,
        "max_iter": max_iter,
        "tolerance": tolerance,
        "output_attr": output_attr,
    }
    if personalization_attr is not None:
        params["personalization_attr"] = personalization_attr

    return RustAlgorithmHandle("centrality.pagerank", params)


def betweenness(
    normalized: bool = True,
    weight_attr: Optional[str] = None,
    output_attr: str = "betweenness",
) -> RustAlgorithmHandle:
    """
    Betweenness centrality algorithm.

    Computes betweenness centrality for each node, measuring how often a node
    appears on shortest paths between other nodes.

    Args:
        normalized: Whether to normalize scores (default: True)
        weight_attr: Optional edge weight attribute
        output_attr: Attribute name for results (default: "betweenness")

    Returns:
        Algorithm handle configured for betweenness centrality

    Example:
        >>> from groggy.algorithms import centrality
        >>> bc = centrality.betweenness(normalized=True)
        >>> result = subgraph.apply(bc)
    """
    params = {
        "normalized": normalized,
        "output_attr": output_attr,
    }
    if weight_attr is not None:
        params["weight_attr"] = weight_attr

    return RustAlgorithmHandle("centrality.betweenness", params)


def closeness(
    weight_attr: Optional[str] = None, output_attr: str = "closeness"
) -> RustAlgorithmHandle:
    """
    Closeness centrality algorithm.

    Computes closeness centrality for each node, measuring the average
    distance from the node to all other reachable nodes.

    Args:
        weight_attr: Optional edge weight attribute
        output_attr: Attribute name for results (default: "closeness")

    Returns:
        Algorithm handle configured for closeness centrality

    Example:
        >>> from groggy.algorithms import centrality
        >>> cc = centrality.closeness()
        >>> result = subgraph.apply(cc)
    """
    params = {
        "output_attr": output_attr,
    }
    if weight_attr is not None:
        params["weight_attr"] = weight_attr

    return RustAlgorithmHandle("centrality.closeness", params)
