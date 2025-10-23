"""
Community detection algorithms for finding clusters in graphs.

This module provides algorithms for detecting communities and clusters,
including Label Propagation and Louvain modularity optimization.
"""

from groggy.algorithms.base import RustAlgorithmHandle
from typing import Optional


def lpa(
    max_iter: int = 100,
    seed: Optional[int] = None,
    output_attr: str = "community"
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
    output_attr: str = "community"
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
