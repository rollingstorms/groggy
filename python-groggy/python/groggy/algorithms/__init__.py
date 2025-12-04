"""
Algorithm discovery and access for Groggy.

This module provides high-level access to all registered algorithms,
grouped by category (centrality, community, pathfinding, etc.).
"""

import json
from typing import Dict, List, Optional

from groggy import _groggy
from groggy.algorithms import centrality, community, pathfinding
from groggy.algorithms.base import (AlgorithmHandle, RustAlgorithmHandle,
                                    algorithm)


def list(category: Optional[str] = None) -> List[str]:
    """
    List all registered algorithms, optionally filtered by category.

    Args:
        category: Optional category filter (e.g., "centrality", "community")

    Returns:
        List of algorithm IDs

    Example:
        >>> from groggy import algorithms
        >>> all_algos = algorithms.list()
        >>> centrality_algos = algorithms.list(category="centrality")
    """
    if category is None:
        algos = _groggy.pipeline.list_algorithms()
        return [entry["id"].value for entry in algos]
    else:
        categories = _groggy.pipeline.list_algorithm_categories()
        return categories.get(category, [])


def categories() -> Dict[str, List[str]]:
    """
    Get all algorithms grouped by category.

    Returns:
        Dictionary mapping category names to algorithm ID lists

    Example:
        >>> from groggy import algorithms
        >>> cats = algorithms.categories()
        >>> print(cats.keys())  # dict_keys(['centrality', 'community', 'pathfinding', ...])
    """
    return _groggy.pipeline.list_algorithm_categories()


def info(algorithm_id: str) -> Dict[str, any]:
    """
    Get detailed information about a specific algorithm.

    Args:
        algorithm_id: The algorithm identifier (e.g., "centrality.pagerank")

    Returns:
        Dictionary with algorithm metadata including parameters, version, etc.

    Example:
        >>> from groggy import algorithms
        >>> pagerank_info = algorithms.info("centrality.pagerank")
        >>> print(pagerank_info["description"])
        >>> print(pagerank_info["parameters"])
    """
    metadata = _groggy.pipeline.get_algorithm_metadata(algorithm_id)

    # Convert AttrValues to Python values
    result = {}
    for key, value in metadata.items():
        if key == "parameters":
            # Parse JSON parameter list
            try:
                params_json = value.value
                result["parameters"] = json.loads(params_json)
            except (json.JSONDecodeError, AttributeError):
                result["parameters"] = value.value
        else:
            result[key] = value.value

    return result


def search(query: str) -> List[str]:
    """
    Search for algorithms by name or description.

    Args:
        query: Search query string

    Returns:
        List of matching algorithm IDs

    Example:
        >>> from groggy import algorithms
        >>> results = algorithms.search("shortest path")
        >>> results = algorithms.search("community")
    """
    query_lower = query.lower()
    matching = []

    for algo_dict in _groggy.pipeline.list_algorithms():
        algo_id = algo_dict["id"].value
        name = algo_dict.get("name", algo_dict["id"]).value
        desc = algo_dict.get("description", "").value

        if (
            query_lower in algo_id.lower()
            or query_lower in name.lower()
            or query_lower in desc.lower()
        ):
            matching.append(algo_id)

    return matching


# Expose submodules
__all__ = [
    "algorithm",
    "AlgorithmHandle",
    "RustAlgorithmHandle",
    "centrality",
    "community",
    "pathfinding",
    "list",
    "categories",
    "info",
    "search",
]
