"""
Graph Generators - Create various graph families and synthetic datasets

This module provides functions to generate different types of graphs including:
- Classic graph families (complete, cycle, path, star, etc.)
- Random graph models (Erdős-Rényi, Barabási-Albert, Watts-Strogatz)
- Real-world network models (karate club, social networks)
- Synthetic datasets with attributes
"""

import random
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from . import Graph
from .types import NodeId, EdgeId

def complete_graph(n: int, **node_attrs) -> Graph:
    """
    Generate a complete graph with n nodes (every pair of nodes connected).
    
    Args:
        n: Number of nodes
        **node_attrs: Additional attributes to set on all nodes
        
    Returns:
        Graph: Complete graph with n nodes and n*(n-1)/2 edges
        
    Example:
        >>> g = complete_graph(5, group="test")
        >>> print(f"Nodes: {g.node_count()}, Edges: {g.edge_count()}")
        Nodes: 5, Edges: 10
    """
    g = Graph()
    
    # Add all nodes
    nodes = []
    for i in range(n):
        node_id = g.add_node(index=i, **node_attrs)
        nodes.append(node_id)
    
    # Add all edges (complete connectivity)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(nodes[i], nodes[j])
    
    return g

def erdos_renyi(n: int, p: float, directed: bool = False, seed: Optional[int] = None, **node_attrs) -> Graph:
    """
    Generate an Erdős-Rényi random graph G(n,p).
    
    Args:
        n: Number of nodes
        p: Probability of edge creation between any pair of nodes (0 <= p <= 1)
        directed: If True, create directed edges
        seed: Random seed for reproducibility
        **node_attrs: Additional attributes to set on all nodes
        
    Returns:
        Graph: Random graph with n nodes and approximately p*n*(n-1)/2 edges
        
    Example:
        >>> g = erdos_renyi(100, 0.05, seed=42)
        >>> print(f"Nodes: {g.node_count()}, Edges: {g.edge_count()}")
    """
    if seed is not None:
        random.seed(seed)
    
    g = Graph()
    
    # Add all nodes
    nodes = []
    for i in range(n):
        node_id = g.add_node(index=i, **node_attrs)
        nodes.append(node_id)
    
    # Add edges randomly with probability p
    for i in range(n):
        start_j = 0 if directed else i + 1
        for j in range(start_j, n):
            if i != j and random.random() < p:
                g.add_edge(nodes[i], nodes[j])
    
    return g

def barabasi_albert(n: int, m: int, seed: Optional[int] = None, **node_attrs) -> Graph:
    """
    Generate a Barabási-Albert scale-free network using preferential attachment.
    
    Args:
        n: Number of nodes
        m: Number of edges to attach from a new node to existing nodes
        seed: Random seed for reproducibility
        **node_attrs: Additional attributes to set on all nodes
        
    Returns:
        Graph: Scale-free graph with n nodes and approximately n*m edges
        
    Example:
        >>> g = barabasi_albert(1000, 3, seed=42)
        >>> print(f"Nodes: {g.node_count()}, Edges: {g.edge_count()}")
    """
    if seed is not None:
        random.seed(seed)
    
    if m >= n:
        raise ValueError("m must be less than n")
    
    g = Graph()
    
    # Start with a complete graph of m+1 nodes
    nodes = []
    for i in range(m + 1):
        node_id = g.add_node(index=i, **node_attrs)
        nodes.append(node_id)
    
    # Connect initial nodes in a complete graph
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            g.add_edge(nodes[i], nodes[j])
    
    # Keep track of degree for preferential attachment
    degrees = [m] * (m + 1)  # Each initial node has degree m
    total_degree = sum(degrees)
    
    # Add remaining nodes with preferential attachment
    for i in range(m + 1, n):
        node_id = g.add_node(index=i, **node_attrs)
        nodes.append(node_id)
        
        # Choose m nodes to connect to based on degree (preferential attachment)
        targets = []
        for _ in range(m):
            # Weighted random selection based on degrees
            rand_val = random.randint(0, total_degree - 1)
            cumsum = 0
            for j, degree in enumerate(degrees):
                cumsum += degree
                if rand_val < cumsum and j not in targets:
                    targets.append(j)
                    break
            
            # Fallback: if we couldn't find a unique target, pick randomly
            if len(targets) == 0 or targets[-1] in targets[:-1]:
                available = [j for j in range(len(nodes)-1) if j not in targets]
                if available:
                    targets[-1] = random.choice(available)
        
        # Add edges to selected targets
        for target_idx in targets:
            g.add_edge(node_id, nodes[target_idx])
            degrees[target_idx] += 1
        
        # Add degree for new node
        degrees.append(m)
        total_degree += 2 * m  # Each edge adds 2 to total degree
    
    return g

def watts_strogatz(n: int, k: int, p: float, seed: Optional[int] = None, **node_attrs) -> Graph:
    """
    Generate a Watts-Strogatz small-world network.
    
    Args:
        n: Number of nodes
        k: Each node is connected to k nearest neighbors in ring topology
        p: Probability of rewiring each edge
        seed: Random seed for reproducibility
        **node_attrs: Additional attributes to set on all nodes
        
    Returns:
        Graph: Small-world network with n nodes
        
    Example:
        >>> g = watts_strogatz(1000, 6, 0.1, seed=42)
        >>> print(f"Nodes: {g.node_count()}, Edges: {g.edge_count()}")
    """
    if seed is not None:
        random.seed(seed)
    
    if k >= n:
        raise ValueError("k must be less than n")
    if k % 2 != 0:
        raise ValueError("k must be even")
    
    g = Graph()
    
    # Add all nodes
    nodes = []
    for i in range(n):
        node_id = g.add_node(index=i, **node_attrs)
        nodes.append(node_id)
    
    # Create ring lattice (each node connected to k/2 neighbors on each side)
    edges = []
    for i in range(n):
        for j in range(1, k // 2 + 1):
            neighbor = (i + j) % n
            edges.append((i, neighbor))
    
    # Rewire edges with probability p
    rewired_edges = []
    for i, j in edges:
        if random.random() < p:
            # Rewire: choose new target randomly
            possible_targets = [x for x in range(n) if x != i and x not in [edge[1] for edge in rewired_edges if edge[0] == i]]
            if possible_targets:
                j = random.choice(possible_targets)
        rewired_edges.append((i, j))
    
    # Add all edges to graph
    for i, j in rewired_edges:
        g.add_edge(nodes[i], nodes[j])
    
    return g

def cycle_graph(n: int, **node_attrs) -> Graph:
    """
    Generate a cycle graph with n nodes.
    
    Args:
        n: Number of nodes
        **node_attrs: Additional attributes to set on all nodes
        
    Returns:
        Graph: Cycle graph with n nodes and n edges
    """
    g = Graph()
    
    # Add all nodes
    nodes = []
    for i in range(n):
        node_id = g.add_node(index=i, **node_attrs)
        nodes.append(node_id)
    
    # Add cycle edges
    for i in range(n):
        next_node = (i + 1) % n
        g.add_edge(nodes[i], nodes[next_node])
    
    return g

def path_graph(n: int, **node_attrs) -> Graph:
    """
    Generate a path graph with n nodes.
    
    Args:
        n: Number of nodes
        **node_attrs: Additional attributes to set on all nodes
        
    Returns:
        Graph: Path graph with n nodes and n-1 edges
    """
    g = Graph()
    
    # Add all nodes
    nodes = []
    for i in range(n):
        node_id = g.add_node(index=i, **node_attrs)
        nodes.append(node_id)
    
    # Add path edges
    for i in range(n - 1):
        g.add_edge(nodes[i], nodes[i + 1])
    
    return g

def star_graph(n: int, **node_attrs) -> Graph:
    """
    Generate a star graph with n nodes (one central node connected to all others).
    
    Args:
        n: Number of nodes
        **node_attrs: Additional attributes to set on all nodes
        
    Returns:
        Graph: Star graph with n nodes and n-1 edges
    """
    g = Graph()
    
    # Add all nodes
    nodes = []
    for i in range(n):
        node_id = g.add_node(index=i, **node_attrs)
        nodes.append(node_id)
    
    # Connect center (node 0) to all other nodes
    center = nodes[0]
    for i in range(1, n):
        g.add_edge(center, nodes[i])
    
    return g

def grid_graph(dims: List[int], **node_attrs) -> Graph:
    """
    Generate a grid graph with given dimensions.
    
    Args:
        dims: List of dimensions [width, height] or [width, height, depth]
        **node_attrs: Additional attributes to set on all nodes
        
    Returns:
        Graph: Grid graph
        
    Example:
        >>> g = grid_graph([10, 10])  # 10x10 2D grid
        >>> g = grid_graph([5, 5, 5])  # 5x5x5 3D grid
    """
    g = Graph()
    
    if len(dims) == 2:
        width, height = dims
        
        # Create nodes
        nodes = {}
        for x in range(width):
            for y in range(height):
                node_id = g.add_node(x=x, y=y, **node_attrs)
                nodes[(x, y)] = node_id
        
        # Create edges (4-connected grid)
        for x in range(width):
            for y in range(height):
                current = nodes[(x, y)]
                # Right neighbor
                if x + 1 < width:
                    g.add_edge(current, nodes[(x + 1, y)])
                # Down neighbor  
                if y + 1 < height:
                    g.add_edge(current, nodes[(x, y + 1)])
    
    elif len(dims) == 3:
        width, height, depth = dims
        
        # Create nodes
        nodes = {}
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    node_id = g.add_node(x=x, y=y, z=z, **node_attrs)
                    nodes[(x, y, z)] = node_id
        
        # Create edges (6-connected grid)
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    current = nodes[(x, y, z)]
                    # Right neighbor
                    if x + 1 < width:
                        g.add_edge(current, nodes[(x + 1, y, z)])
                    # Down neighbor
                    if y + 1 < height:
                        g.add_edge(current, nodes[(x, y + 1, z)])
                    # Forward neighbor
                    if z + 1 < depth:
                        g.add_edge(current, nodes[(x, y, z + 1)])
    
    else:
        raise ValueError("Only 2D and 3D grids are supported")
    
    return g

def tree(n: int, branching_factor: int = 2, **node_attrs) -> Graph:
    """
    Generate a regular tree with n nodes.
    
    Args:
        n: Number of nodes
        branching_factor: Number of children per internal node
        **node_attrs: Additional attributes to set on all nodes
        
    Returns:
        Graph: Tree with n nodes and n-1 edges
    """
    g = Graph()
    
    if n <= 0:
        return g
    
    # Add root node
    nodes = []
    root_id = g.add_node(index=0, level=0, **node_attrs)
    nodes.append(root_id)
    
    # Add nodes level by level
    level = 0
    while len(nodes) < n:
        level += 1
        level_start = len(nodes)
        
        # For each node in previous level, add children
        prev_level_start = 0 if level == 1 else sum(branching_factor ** i for i in range(level - 1))
        prev_level_end = sum(branching_factor ** i for i in range(level))
        
        for parent_idx in range(max(0, len(nodes) - (prev_level_end - prev_level_start)), len(nodes)):
            if len(nodes) >= n:
                break
            parent_id = nodes[parent_idx]
            
            for child_num in range(branching_factor):
                if len(nodes) >= n:
                    break
                child_id = g.add_node(index=len(nodes), level=level, **node_attrs)
                nodes.append(child_id)
                g.add_edge(parent_id, child_id)
    
    return g

# Real-world network models
def karate_club() -> Graph:
    """
    Generate Zachary's karate club graph.
    
    Returns:
        Graph: The famous karate club social network (34 nodes, 78 edges)
    """
    g = Graph()
    
    # Create 34 nodes
    nodes = []
    for i in range(34):
        node_id = g.add_node(index=i, name=f"Member_{i}")
        nodes.append(node_id)
    
    # Define the edges from the original dataset
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), 
        (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
        (3, 7), (3, 12), (3, 13), (4, 6), (4, 10), (5, 6), (5, 10), (5, 16),
        (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32), (14, 33),
        (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32), (20, 33),
        (22, 32), (22, 33), (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31), (25, 31), (26, 29), (26, 33), (27, 33),
        (28, 31), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33), (31, 32), (31, 33), (32, 33)
    ]
    
    # Add all edges
    for i, j in edges:
        g.add_edge(nodes[i], nodes[j], relationship="friendship")
    
    return g

def social_network(n: int, communities: int = 3, 
                  node_attrs: Optional[List[str]] = None,
                  edge_attrs: Optional[List[str]] = None,
                  seed: Optional[int] = None) -> Graph:
    """
    Generate a synthetic social network with realistic attributes.
    
    Args:
        n: Number of nodes (people)
        communities: Number of communities/clusters
        node_attrs: List of node attribute names to generate
        edge_attrs: List of edge attribute names to generate  
        seed: Random seed for reproducibility
        
    Returns:
        Graph: Synthetic social network with attributes
    """
    if seed is not None:
        random.seed(seed)
    
    if node_attrs is None:
        node_attrs = ['age', 'income', 'location']
    if edge_attrs is None:
        edge_attrs = ['strength', 'frequency']
    
    g = Graph()
    
    # Generate realistic attribute values
    locations = ['NYC', 'SF', 'LA', 'Chicago', 'Boston', 'Austin', 'Seattle', 'Denver']
    
    # Create nodes with attributes
    nodes = []
    for i in range(n):
        attrs = {'index': i, 'community': i % communities}
        
        if 'age' in node_attrs:
            attrs['age'] = random.randint(18, 65)
        if 'income' in node_attrs:
            attrs['income'] = random.randint(30000, 200000)
        if 'location' in node_attrs:
            attrs['location'] = random.choice(locations)
        
        node_id = g.add_node(**attrs)
        nodes.append(node_id)
    
    # Create edges with community structure (higher probability within communities)
    for i in range(n):
        for j in range(i + 1, n):
            # Higher probability of connection within same community
            same_community = (i % communities) == (j % communities)
            p = 0.15 if same_community else 0.02
            
            if random.random() < p:
                edge_attrs_dict = {}
                if 'strength' in edge_attrs:
                    edge_attrs_dict['strength'] = random.uniform(0.1, 1.0)
                if 'frequency' in edge_attrs:
                    edge_attrs_dict['frequency'] = random.choice(['daily', 'weekly', 'monthly', 'rarely'])
                
                g.add_edge(nodes[i], nodes[j], **edge_attrs_dict)
    
    return g