# python_new/groggy/graph.py

import groggy._core
from groggy.collections.nodes import NodeCollection
from groggy.collections.edges import EdgeCollection
import time

class Graph:
    """
    Main Graph class with new collection-based API.

    Provides a unified interface for node/edge collections, subgraph views, and backend selection.
    Delegates storage and algorithm logic to modular components. Supports efficient batch operations and filtering.
    """

    def __init__(self, directed=False, backend=None):

    
        """
        Initializes a new Graph instance.
        
        Args:
            directed (bool): Whether the graph is directed.
            backend (str, optional): Backend implementation to use.
        
        Selects and initializes the appropriate backend. Sets up node and edge collections.
        Raises:
            ValueError: If backend is invalid or unavailable.
        """
        start = time.perf_counter()
        # Only Rust backend is supported for now
        if backend not in (None, 'rust'):
            raise ValueError(f"Unsupported backend: {backend}")
        self._rust = groggy._core.FastGraph()
        # Note: directed parameter is currently ignored as FastGraph defaults to directed=True
        # TODO: Add support for setting directed in Rust FastGraph constructor
        self._nodes = NodeCollection(self)
        self._edges = EdgeCollection(self)
        
        elapsed = time.perf_counter() - start
        print(f"[Groggy] Graph.__init__: constructed in {elapsed:.6f} seconds")

    def info(self):
        """
        Returns comprehensive information about the graph.
        """
        info = self._rust.info()
        # Manually convert PyO3 GraphInfo to a Python dict
        return {
            "name": info.name() if hasattr(info, "name") else None,
            "directed": info.directed() if hasattr(info, "directed") else None,
            "node_count": info.node_count() if hasattr(info, "node_count") else None,
            "edge_count": info.edge_count() if hasattr(info, "edge_count") else None,
            "attributes": dict(info.attributes()) if hasattr(info, "attributes") else {},
        }

    def size(self):
        """
        Returns the total number of entities in the graph (nodes + edges).
        
        Fast lookup from collection metadata; does not require iteration.
        Returns:
            int: Total count of nodes and edges.
        """
        return self._rust.size()

    def is_directed(self):
        """
        Checks whether the graph is directed.
        
        Returns:
            bool: True if directed, False otherwise.
        """
        return self._rust.is_directed()

    @property
    def attributes(self):
        """
        Expose the Rust attribute_manager for direct access to memory_usage_breakdown and other diagnostics.
        """
        return self._rust.attribute_manager

    @property
    def nodes(self):
        """
        Returns the NodeCollection for this graph.
        
        Provides access to node-level operations, batch methods, and attribute management.
        Returns:
            NodeCollection: Node collection interface.
        """
        return self._nodes

    @property
    def edges(self):
        """
        Returns the EdgeCollection for this graph.
        
        Provides access to edge-level operations, batch methods, and attribute management.
        Returns:
            EdgeCollection: Edge collection interface.
        """
        return self._edges

    def subgraph(self, node_filter=None, edge_filter=None):
        """
        Creates a subgraph view using node and/or edge filters.
        
        Args:
            node_filter (callable, optional): Function to filter nodes.
            edge_filter (callable, optional): Function to filter edges.
        Returns:
            Graph: A new Graph instance representing the filtered subgraph.
        Raises:
            ValueError: If filters are invalid.
        """
        from groggy.graph.subgraph import Subgraph
        filter_criteria = {}
        if node_filter:
            filter_criteria['node_filter'] = node_filter
        if edge_filter:
            filter_criteria['edge_filter'] = edge_filter
        return Subgraph(self, filter_criteria, metadata={})

    def subgraphs(self):
        """
        Returns all subgraphs partitioned by a given attribute or property.
        
        Useful for community detection, clustering, or attribute-based analysis.
        Returns:
            List[Graph]: List of subgraph instances.
        """
        from groggy.graph.subgraph import Subgraph
        # Default partition attribute
        attr = 'community'
        node_attrs = self.nodes.attr().get(attr_names=None)
        if not node_attrs:
            return []
        # Pick first attribute if 'community' not found
        if attr not in next(iter(node_attrs.values()), {}):
            attr = next(iter(next(iter(node_attrs.values()), {}).keys()), None)
        if not attr:
            return []
        # Partition nodes by attribute value
        partitions = {}
        for nid, attrs in node_attrs.items():
            key = attrs.get(attr)
            if key is not None:
                partitions.setdefault(key, set()).add(nid)
        subgraphs = []
        for value, node_ids in partitions.items():
            filter_criteria = {'node_filter': lambda nid, nattrs=node_ids: nid in nattrs}
            meta = {'partition_attr': attr, 'partition_value': value}
            subgraphs.append(Subgraph(self, filter_criteria, meta))
        return subgraphs
