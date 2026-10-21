"""
Centrality algorithms for measuring node importance.

This module provides access to centrality algorithms like PageRank, Betweenness,
and Closeness centrality.
"""

from typing import Literal, Optional

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
    based on the graph's link structure. Undirected edges are traversed in both
    directions. Dangling mass follows the normalized personalization vector.
    A bounded run can return an approximation: inspect pr.converged in the
    execution profile. Nonpersistent runs return full-precision scores under
    profile["outputs"]["centrality.pagerank.scores"].

    Args:
        damping: Finite link-following probability in [0, 1) (default: 0.85)
        max_iter: Maximum iterations (default: 100)
        tolerance: Finite positive maximum-coordinate update threshold (default: 1e-6)
        personalization_attr: Optional finite nonnegative numeric node weights;
            missing values are one, and their total must be positive
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
    appears on shortest paths between other nodes. Equal-cost parallel edge
    instances count as distinct paths. Directed raw scores count ordered pairs;
    undirected raw scores count unordered pairs.

    Args:
        normalized: Whether to normalize scores (default: True)
        weight_attr: Optional strictly positive finite numeric edge lengths; missing is one
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
    weight_attr: Optional[str] = None,
    output_attr: str = "closeness",
    harmonic: bool = True,
    direction: Literal["outgoing", "incoming"] = "outgoing",
    normalized: bool = False,
) -> RustAlgorithmHandle:
    """
    Closeness centrality algorithm.

    Defaults to raw harmonic scores using outgoing distances. Classic scores
    use reachable_count / distance_sum. Normalization divides harmonic scores
    by n-1, or scales classic scores by the reachable fraction. Disconnected
    isolates score zero. Weighted lengths must be finite and strictly positive.

    Args:
        weight_attr: Optional strictly positive numeric edge lengths; missing is one
        harmonic: Sum reciprocal distances (default: True); False selects classic
        direction: Outgoing distances by default; incoming reverses directed paths
        normalized: Divide harmonic scores by n-1 or apply classic reachability correction
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
        "harmonic": harmonic,
        "direction": direction,
        "normalized": normalized,
    }
    if weight_attr is not None:
        params["weight_attr"] = weight_attr

    return RustAlgorithmHandle("centrality.closeness", params)
