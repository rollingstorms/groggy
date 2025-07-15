class EntityProxy:
    """
    Generic proxy for a graph entity (node or edge).
    Provides attribute access and mutation via the collection's attribute manager.
    """
    def __init__(self, collection, entity_id):
        self._collection = collection
        self._id = entity_id
        self._attr = collection.attr()

    @property
    def id(self):
        return self._id

    def get(self, attr_name):
        return self._attr.get(self._id, attr_name)

    def set(self, attr_name, value):
        return self._attr.set({self._id: {attr_name: value}})

    def __getitem__(self, attr_name):
        return self.get(attr_name)

    def __setitem__(self, attr_name, value):
        self.set(attr_name, value)

    @property
    def attrs(self):
        """
        Returns all attributes for this entity as a dict.
        Example:
            node = graph.nodes[123]
            print(node.attrs)  # {'color': 'red', 'weight': 5}
        """
        try:
            return self._attr.get(self._id)
        except Exception:
            return {}

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self._id} attrs={self.attrs}>"

class NodeProxy(EntityProxy):
    """Proxy for a node entity."""
    pass

class EdgeProxy(EntityProxy):
    """Proxy for an edge entity."""
    @property
    def attrs(self):
        """
        Returns all attributes for this edge as a dict.
        Example:
            edge = graph.edges[42]
            print(edge.attrs)
        """
        try:
            return self._attr.get(self._id)
        except Exception:
            return {}
