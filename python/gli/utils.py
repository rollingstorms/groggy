"""
Utility functions for graph creation and manipulation
"""

import random
from typing import List
from .graph import Graph


def create_random_graph(n_nodes: int = 10, edge_probability: float = 0.3, use_rust: bool = None) -> Graph:
    """Create a random graph efficiently using vectorized operations"""
    
    # Create edges vectorized
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_probability:
                edges.append((f"node_{i}", f"node_{j}"))
    
    # Use fast constructor
    graph = Graph.from_edge_list(edges)
    return graph


def create_clustering_workflow(store, graph: Graph, algorithms: List[str] = None) -> List[str]:
    """Create branches for different clustering algorithms"""
    algorithms = algorithms or ['kmeans', 'spectral', 'hierarchical']
    branches = []
    
    for algo in algorithms:
        branch_name = f"clustering_{algo}"
        try:
            store.create_branch(branch_name, description=f"Clustering with {algo}")
            branches.append(branch_name)
        except ValueError:
            # Branch already exists
            pass
    
    return branches


def create_subgraph_branch(store, subgraph: Graph, branch_name: str, description: str = "") -> str:
    """Create a branch from a subgraph for isolated processing"""
    return store.create_branch(
        branch_name, 
        description=description or f"Subgraph branch: {branch_name}"
    )
