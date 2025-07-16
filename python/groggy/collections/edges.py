# python_new/groggy/collections/edges.py

from .base import BaseCollection
from .proxy import EdgeProxy

class EdgeCollection(BaseCollection):
    """
    Collection interface for graph edges.
    
    Exposes batch and single-edge operations, attribute management, filtering, and efficient iteration.
    Delegates storage and attribute logic to specialized managers. Supports node linkage and subgraph views.
    """

    def __init__(self, graph):
        """
        Initializes an EdgeCollection for the given graph.
        
        Args:
            graph (Graph): The parent graph instance.
        Sets up attribute manager and internal references.
        """
        super().__init__()
        self.graph = graph
        self._rust = graph._rust.edges()
        self.attr = self._rust.attr()

    def remove(self, edge_ids):
        """
        Removes one or more edges from the collection.
        
        Accepts a single edge ID or list of IDs. Marks edges as deleted in storage for lazy cleanup.
        Args:
            edge_ids: Single ID or list.
        Raises:
            KeyError: If edge does not exist.
        """
        if not isinstance(edge_ids, (list, tuple)):
            edge_ids = [edge_ids]
        try:
            self._rust.remove(edge_ids)
        except Exception as e:
            raise KeyError(f"Failed to remove edges: {e}")

    def filter(self, *args, **kwargs):
        """
        Returns a filtered EdgeCollection view based on attribute values or custom predicates.
        
        Supports chaining and composition of filters. Delegates to storage backend for efficient execution.
        Args:
            *args, **kwargs: Filtering criteria.
        Returns:
            EdgeCollection: Filtered collection view.
        """
        # For now, only support attribute equality via kwargs
        filtered = self._rust.filter().by_kwargs(**kwargs)
        ec = EdgeCollection(self.graph)
        ec._rust = filtered
        return ec

    def size(self):
        """
        Returns the number of edges in the collection.
        
        Fast lookup from storage metadata; does not require iteration.
        Returns:
            int: Edge count.
        """
        return self._rust.size()

    def add(self, edge_data):
        """
        Adds one or more edges to the collection.
        Supports single, batch, or dict input. If dict, adds edges and sets attributes.
        Returns proxy object(s) for added edges.
        """
        from .. import EdgeId
        # If dict, treat keys as (src, tgt) tuples and values as attribute dicts
        if isinstance(edge_data, dict):
            edge_ids = [EdgeId(src, tgt) for (src, tgt) in edge_data.keys()]
            self._rust.add(edge_ids)
            self.attr.set(edge_data)
            return [self.get(eid) for eid in edge_ids]
        is_single = not isinstance(edge_data, (list, tuple))
        if is_single:
            edge_data = [edge_data]
        edge_ids = []
        for edge in edge_data:
            if isinstance(edge, tuple) and len(edge) in (2, 3):
                src, tgt = edge[0], edge[1]
                attrs = edge[2] if len(edge) == 3 else None
                edge_ids.append(EdgeId(src, tgt))
            elif isinstance(edge, EdgeId):
                edge_ids.append(edge)
            else:
                raise ValueError(f"Expected tuple or EdgeId, got {type(edge)}")
        try:
            self._rust.add(edge_ids)
        except Exception as e:
            raise ValueError(f"Failed to add edges: {e}")
        for edge, attrs in zip(edge_data, [e[2] if isinstance(e, tuple) and len(e) == 3 else None for e in edge_data]):
            if attrs:
                self.attr.set({(edge[0], edge[1]): attrs})
        if is_single:
            return self.get(edge_ids[0])
        else:
            return [self.get(edge_id) for edge_id in edge_ids]
        from groggy.collections.nodes import NodeCollection
        node_ids = self.node_ids()
        nc = NodeCollection(self.graph)
        nc._rust = self.graph._rust.nodes().filter().by_ids(node_ids)
        return nc

    def node_ids(self):
        """
        Returns node IDs from the filtered edges in this collection.
        
        Efficiently extracts node IDs from edge storage.
        Returns:
            list: List of node IDs.
        """
        return list(self._rust.node_ids())
        pass

    def __iter__(self):
        """
        Returns an iterator over edges in the collection.
        
        Supports lazy iteration over storage data, yielding EdgeProxy objects or raw IDs as needed.
        Returns:
            Iterator: Iterator over edges.
        """
        from .proxy import EdgeProxy
        for eid in self.ids():
            yield EdgeProxy(self, eid)

    def __getitem__(self, key):
        if not self.has(key):
            raise KeyError(f"Edge {key} not found in collection.")
        return EdgeProxy(self, key)

    def __len__(self):
        """
        Returns the number of edges in the collection (calls size()).
        
        Returns:
            int: Edge count.
        """
        return self.size()

    def get(self, edge_id):
        """
        Returns an EdgeProxy for the given edge ID.
        
        Args:
            edge_id: The edge ID to retrieve.
        Returns:
            EdgeProxy: Proxy object for the edge, or None if not found.
        """
        try:
            return self._rust.get(edge_id)
        except:
            return None

class EdgeAttributeManager:
    """
    Batch attribute management for edges.
    
    Provides efficient get/set/type operations for edge attributes, supporting both single and batch modes.
    Delegates storage logic to the backend for vectorized access.
    """

    def __init__(self, edge_collection):
        """
        Initializes an EdgeAttributeManager for the given edge collection.
        
        Args:
            edge_collection (EdgeCollection): The parent edge collection.
        """
        self.collection = edge_collection
        self._rust = edge_collection._rust.attr()

    def get(self, edge_ids=None, attr_names=None):
        """
        Retrieves one or more attributes for the given edge(s).
        
        Supports single or batch mode. Delegates to storage backend for efficient vectorized access.
        Args:
            edge_ids (optional): Single ID or list of IDs. If None, all edges.
            attr_names (optional): Single name or list. If None, all attributes.
        Returns:
            dict: Attribute values by edge and attribute.
        Raises:
            KeyError: If edge or attribute not found.
        """
        try:
            return self._rust.get(edge_ids, attr_names)
        except Exception as e:
            raise KeyError(f"Failed to get edge attribute(s): {e}")

    def set(self, attr_data):
        """
        Sets one or more attributes for the given edge(s).
        
        Accepts a dict or batch. Delegates to backend for atomic, vectorized update.
        Args:
            attr_data: Dict or batch of (edge_id, attr_name, value).
        Raises:
            KeyError: If edge not found.
            ValueError: On type mismatch or schema error.
        """
        try:
            self._rust.set(attr_data)
        except Exception as e:
            raise ValueError(f"Failed to set edge attribute(s): {e}")

    def set_type(self, attr_name, attr_type):
        """
        Sets the type/schema for a given attribute across all edges.
        
        Updates storage metadata and validates type safety. May trigger migration or validation.
        Args:
            attr_name (str): Attribute name.
            attr_type (type): Python type or schema.
        Raises:
            ValueError: On type mismatch or migration failure.
        """
        try:
            self._rust.set_type(attr_name, attr_type)
        except Exception as e:
            raise ValueError(f"Failed to set attribute type: {e}")
