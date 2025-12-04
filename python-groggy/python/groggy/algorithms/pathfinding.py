"""
Pathfinding algorithms for computing shortest paths and distances.

This module provides algorithms for finding paths between nodes,
including Dijkstra, BFS, DFS, and A*.
"""

from typing import Optional

from groggy.algorithms.base import RustAlgorithmHandle


def dijkstra(
    start_attr: str, weight_attr: Optional[str] = None, output_attr: str = "distance"
) -> RustAlgorithmHandle:
    """
    Dijkstra's shortest path algorithm.

    Computes shortest paths from source nodes to all other nodes.
    Supports weighted graphs via the weight_attr parameter.

    Args:
        start_attr: Node attribute marking start nodes (truthy values)
        weight_attr: Optional edge weight attribute
        output_attr: Attribute name for distances (default: "distance")

    Returns:
        Algorithm handle configured for Dijkstra

    Example:
        >>> from groggy.algorithms import pathfinding
        >>> dijkstra_algo = pathfinding.dijkstra(
        ...     start_attr="is_source",
        ...     weight_attr="weight"
        ... )
        >>> result = subgraph.apply(dijkstra_algo)
    """
    params = {
        "start_attr": start_attr,
        "output_attr": output_attr,
    }
    if weight_attr is not None:
        params["weight_attr"] = weight_attr

    return RustAlgorithmHandle("pathfinding.dijkstra", params)


def bfs(start_attr: str, output_attr: str = "distance") -> RustAlgorithmHandle:
    """
    Breadth-First Search for unweighted shortest paths.

    Computes shortest path distances using BFS from source nodes.
    Faster than Dijkstra for unweighted graphs.

    Args:
        start_attr: Node attribute marking start nodes
        output_attr: Attribute name for distances (default: "distance")

    Returns:
        Algorithm handle configured for BFS

    Example:
        >>> from groggy.algorithms import pathfinding
        >>> bfs_algo = pathfinding.bfs(start_attr="is_root")
        >>> result = subgraph.apply(bfs_algo)
    """
    params = {
        "start_attr": start_attr,
        "output_attr": output_attr,
    }

    return RustAlgorithmHandle("pathfinding.bfs", params)


def dfs(start_attr: str, output_attr: str = "discovery_time") -> RustAlgorithmHandle:
    """
    Depth-First Search traversal.

    Performs DFS from source nodes, recording discovery times.

    Args:
        start_attr: Node attribute marking start nodes
        output_attr: Attribute name for discovery times (default: "discovery_time")

    Returns:
        Algorithm handle configured for DFS

    Example:
        >>> from groggy.algorithms import pathfinding
        >>> dfs_algo = pathfinding.dfs(start_attr="is_root")
        >>> result = subgraph.apply(dfs_algo)
    """
    params = {
        "start_attr": start_attr,
        "output_attr": output_attr,
    }

    return RustAlgorithmHandle("pathfinding.dfs", params)


def astar(
    start_attr: str,
    goal_attr: str,
    heuristic_attr: Optional[str] = None,
    weight_attr: Optional[str] = None,
    output_attr: str = "distance",
) -> RustAlgorithmHandle:
    """
    A* pathfinding algorithm.

    Finds shortest paths using A* search with optional heuristic.
    More efficient than Dijkstra when a good heuristic is available.

    Args:
        start_attr: Node attribute marking start nodes
        goal_attr: Node attribute marking goal nodes
        heuristic_attr: Optional node attribute with heuristic estimates
        weight_attr: Optional edge weight attribute
        output_attr: Attribute name for distances (default: "distance")

    Returns:
        Algorithm handle configured for A*

    Example:
        >>> from groggy.algorithms import pathfinding
        >>> astar_algo = pathfinding.astar(
        ...     start_attr="is_start",
        ...     goal_attr="is_goal",
        ...     heuristic_attr="h_score"
        ... )
        >>> result = subgraph.apply(astar_algo)
    """
    params = {
        "start_attr": start_attr,
        "goal_attr": goal_attr,
        "output_attr": output_attr,
    }
    if heuristic_attr is not None:
        params["heuristic_attr"] = heuristic_attr
    if weight_attr is not None:
        params["weight_attr"] = weight_attr

    return RustAlgorithmHandle("pathfinding.astar", params)
