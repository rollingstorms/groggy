# python_new/groggy/algorithms.py

def get_nodes_attribute(graph, node_ids, attr_name):
    """
    Get a specific attribute for multiple nodes efficiently (Rust backend).
    Args:
        graph (Graph): The graph to query.
        node_ids (list): List of node IDs.
        attr_name (str): Name of the attribute to retrieve.
    Returns:
        dict: Mapping node ID to attribute value.
    """
    return graph.nodes.attr().get(node_ids, attr_name)

def get_nodes_attributes(graph, node_ids):
    """
    Get all attributes for multiple nodes efficiently (Rust backend).
    Args:
        graph (Graph): The graph to query.
        node_ids (list): List of node IDs.
    Returns:
        dict: Mapping node ID to attribute dict.
    """
    return graph.nodes.attr().get(node_ids)

def get_edges_attribute(graph, edge_ids, attr_name):
    """
    Get a specific attribute for multiple edges efficiently (Rust backend).
    Args:
        graph (Graph): The graph to query.
        edge_ids (list): List of edge IDs.
        attr_name (str): Name of the attribute to retrieve.
    Returns:
        dict: Mapping edge ID to attribute value.
    """
    return graph.edges.attr().get(edge_ids, attr_name)

def get_edges_attributes(graph, edge_ids):
    """
    Get all attributes for multiple edges efficiently (Rust backend).
    Args:
        graph (Graph): The graph to query.
        edge_ids (list): List of edge IDs.
    Returns:
        dict: Mapping edge ID to attribute dict.
    """
    return graph.edges.attr().get(edge_ids)

def bfs(graph, start_node, **kwargs):
    """
    Performs a breadth-first search (BFS) traversal starting from a given node.
    
    Delegates traversal logic to backend or optimized core. Supports early stopping, filtering, and batch traversal.
    Args:
        graph (Graph): The graph to traverse.
        start_node: Node ID to start from.
        **kwargs: Optional parameters (e.g., max_depth, filter).
    Returns:
        list: List of visited node IDs in BFS order.
    Raises:
        KeyError: If start_node not found.
    """
    # TODO: 1. Validate input; 2. Delegate to backend/core; 3. Support options.
    pass

def dfs(graph, start_node, **kwargs):
    """
    Performs a depth-first search (DFS) traversal starting from a given node.
    
    Delegates traversal logic to backend or optimized core. Supports early stopping, filtering, and batch traversal.
    Args:
        graph (Graph): The graph to traverse.
        start_node: Node ID to start from.
        **kwargs: Optional parameters (e.g., max_depth, filter).
    Returns:
        list: List of visited node IDs in DFS order.
    Raises:
        KeyError: If start_node not found.
    """
    # TODO: 1. Validate input; 2. Delegate to backend/core; 3. Support options.
    pass

def shortest_path(graph, source, target, **kwargs):
    """
    Computes the shortest path between two nodes using the selected algorithm.
    
    Delegates to backend or core for Dijkstra, BFS, or other algorithms. Supports weighted/unweighted graphs and custom filters.
    Args:
        graph (Graph): The graph to search.
        source: Source node ID.
        target: Target node ID.
        **kwargs: Optional parameters (e.g., method, weight_attr).
    Returns:
        list: Sequence of node IDs forming the shortest path, or None if unreachable.
    Raises:
        KeyError: If source or target not found.
    """
    if not graph.nodes.has(source):
        raise KeyError(f"Source node {source} not found.")
    if not graph.nodes.has(target):
        raise KeyError(f"Target node {target} not found.")
    return graph._rust.shortest_path(source, target, **kwargs)

def connected_components(graph):
    """
    Finds all connected components in the graph.
    
    Delegates to backend for efficient batch traversal. Supports both directed and undirected graphs.
    Args:
        graph (Graph): The graph to analyze.
    Returns:
        list: List of components, each as a set of node IDs.
    """
    return graph._rust.connected_components()

def clustering_coefficient(graph, node=None):
    """
    Calculates the clustering coefficient for the graph or a specific node.
    
    Delegates to backend for vectorized computation. Supports batch mode for all nodes.
    Args:
        graph (Graph): The graph to analyze.
        node (optional): Node ID to compute for, or None for all nodes.
    Returns:
        float or dict: Coefficient value(s).
    Raises:
        KeyError: If node not found.
    """
    if node is not None and not graph.nodes.has(node):
        raise KeyError(f"Node {node} not found.")
    return graph._rust.clustering_coefficient(node) if node is not None else graph._rust.clustering_coefficient_all()

def pagerank(graph, **kwargs):
    """
    Computes PageRank scores for all nodes in the graph.
    
    Delegates to backend for efficient iterative computation. Supports weighted edges and convergence criteria.
    Args:
        graph (Graph): The graph to analyze.
        **kwargs: Optional parameters (e.g., damping, max_iter, tol).
    Returns:
        dict: Node ID to PageRank score mapping.
    """
    return graph._rust.pagerank(**kwargs)

def betweenness_centrality(graph, **kwargs):
    """
    Computes betweenness centrality for all nodes or edges in the graph.
    
    Delegates to backend for efficient batch computation. Supports normalization and directed/undirected graphs.
    Args:
        graph (Graph): The graph to analyze.
        **kwargs: Optional parameters (e.g., normalized, endpoints).
    Returns:
        dict: Node or edge ID to centrality score mapping.
    """
    return graph._rust.betweenness_centrality(**kwargs)

def label_propagation(graph, **kwargs):
    """
    Performs community detection using the label propagation algorithm.
    
    Delegates to backend for iterative, batch label updates. Supports custom initialization and stopping criteria.
    Args:
        graph (Graph): The graph to analyze.
        **kwargs: Optional parameters (e.g., max_iter, seed).
    Returns:
        dict: Node ID to label mapping.
    """
    return graph._rust.label_propagation(**kwargs)

def louvain(graph, **kwargs):
    """
    Detects communities using the Louvain modularity maximization algorithm.
    
    Delegates to backend for efficient modularity optimization. Supports weighted graphs and multilevel refinement.
    Args:
        graph (Graph): The graph to analyze.
        **kwargs: Optional parameters (e.g., weight_attr, max_iter).
    Returns:
        dict: Node ID to community label mapping.
    """
    return graph._rust.louvain(**kwargs)

def modularity(graph, **kwargs):
    """
    Calculates the modularity score of a given labeling or partitioning.
    
    Delegates to backend for efficient modularity computation. Supports weighted and directed graphs.
    Args:
        graph (Graph): The graph to analyze.
        **kwargs: Optional parameters (e.g., labels, weight_attr).
    Returns:
        float: Modularity score.
    """
    return graph._rust.modularity(**kwargs)
