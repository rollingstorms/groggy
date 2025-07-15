# python_new/groggy/collections/nodes.py

from .base import BaseCollection
from .proxy import NodeProxy

class NodeCollection(BaseCollection):
    """
    Collection interface for graph nodes (composes managers/helpers).
    
    Exposes batch and single-node operations, attribute management, filtering, and efficient iteration.
    Delegates storage and attribute logic to specialized managers.
    """

    def __init__(self, graph):
        """
        Initializes a NodeCollection for the given graph.
        
        Args:
            graph (Graph): The parent graph instance.
        Sets up attribute manager and internal references.
        """
        super().__init__()
        self.graph = graph
        self._rust = graph._rust.nodes()
        self.attr = self._rust.attr()

    def remove(self, node_ids):
        """
        Removes one or more nodes from the collection.
        
        Accepts a single node ID or list of IDs. Marks nodes as deleted in storage for lazy cleanup.
        Args:
            node_ids: Single ID or list.
        Raises:
            KeyError: If node does not exist.
        """
        if not isinstance(node_ids, (list, tuple)):
            node_ids = [node_ids]
        try:
            self._rust.remove(node_ids)
        except Exception as e:
            raise KeyError(f"Failed to remove nodes: {e}")

    def filter(self, *args, **kwargs):
        """
        Returns a filtered NodeCollection view based on attribute values or custom predicates.
        
        Supports chaining and composition of filters. Delegates to storage backend for efficient execution.
        Args:
            *args, **kwargs: Filtering criteria.
        Returns:
            NodeCollection: Filtered collection view.
        """
        # For now, only support attribute equality via kwargs
        filtered = self._rust.filter().by_kwargs(**kwargs)
        # Return a new NodeCollection wrapping the filtered Rust NodeCollection
        nc = NodeCollection(self.graph)
        nc._rust = filtered
        return nc

    @property
    def size(self):
        """
        Returns the number of nodes in the collection.
        """
        return self._rust.size()

    def ids(self):
        """
        Returns all node IDs in the collection.
        """
        return list(self._rust.ids())

    def has(self, node_id):
        """
        Checks if a node exists in the collection.
        """
        return self._rust.has(node_id)

    def __iter__(self):
        """
        Returns an iterator over nodes in the collection.
        """
        return iter(self._rust.ids())

    def __getitem__(self, key):
        if not self.has(key):
            raise KeyError(f"Node {key} not found in collection.")
        return NodeProxy(self, key)

    def __len__(self):
        """
        Returns the number of nodes in the collection (calls size()).
        
        Returns:
            int: Node count.
        """
        return self.size

class NodeAttributeManager:
    """
    Batch attribute management for nodes.
    
    Provides efficient get/set/type operations for node attributes, supporting both single and batch modes.
    Delegates storage logic to the backend for vectorized access.
    """

    def __init__(self, node_collection):
        """
        Initializes a NodeAttributeManager for the given node collection.
        
        Args:
            node_collection (NodeCollection): The parent node collection.
        """
        self.collection = node_collection
        self._rust = node_collection._rust.attr()

    def get(self, node_ids=None, attr_names=None):
        """
        Retrieves one or more attributes for the given node(s).
        
        Supports single or batch mode. Delegates to storage backend for efficient vectorized access.
        Args:
            node_ids (optional): Single ID or list of IDs. If None, all nodes.
            attr_names (optional): Single name or list. If None, all attributes.
        Returns:
            dict: Attribute values by node and attribute.
        Raises:
            KeyError: If node or attribute not found.
        """
        try:
            return self._rust.get(node_ids, attr_names)
        except Exception as e:
            raise KeyError(f"Failed to get node attribute(s): {e}")

    def set(self, attr_data):
        """
        Sets one or more attributes for the given node(s).
        
        Accepts a dict or batch. Delegates to backend for atomic, vectorized update.
        Args:
            attr_data: Dict or batch of (node_id, attr_name, value).
        Raises:
            KeyError: If node not found.
            ValueError: On type mismatch or schema error.
        """
        try:
            self._rust.set(attr_data)
        except Exception as e:
            raise ValueError(f"Failed to set node attribute(s): {e}")

    def set_type(self, attr_name, attr_type):
        """
        Sets the type/schema for a given attribute across all nodes.
        
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

class NodeProxy:
    # ... (other methods)

    @property
    def attrs(self):
        """
        Returns all attributes for the node as a dictionary.
        
        Args:
            None
        Returns:
            dict: Attribute values by name.
        """
        return self.collection.attr.get(node_ids=self.id)
