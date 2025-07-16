# python_new/groggy/collections/base.py

class BaseCollection:
    """
    Abstract base for NodeCollection/EdgeCollection (composition, not inheritance).
    
    Defines the shared API for collection types, including batch operations, filtering, attribute access, and iteration.
    Concrete collections must implement these methods, delegating to efficient storage backends as appropriate.
    """

    def add(self, entity_data):
        """
        Adds one or more entities (nodes/edges) to the collection.
        Supports single or batch addition. Delegates to storage backend for efficient insertion.
        """
        if not isinstance(entity_data, (list, tuple)):
            entity_data = [entity_data]
        try:
            self._rust.add(entity_data)
        except Exception as e:
            raise ValueError(f"Failed to add entities: {e}")

    def remove(self, entity_ids):
        """
        Removes one or more entities (nodes/edges) from the collection.
        Supports single or batch removal. Marks entities as deleted in storage for lazy cleanup.
        """
        if not isinstance(entity_ids, (list, tuple)):
            entity_ids = [entity_ids]
        try:
            self._rust.remove(entity_ids)
        except Exception as e:
            raise KeyError(f"Failed to remove entities: {e}")

    def filter(self, *args, **kwargs):
        """
        Returns a filtered view of the collection based on attribute values or custom predicates.
        Supports chaining and composition of filters. Delegates to storage backend for efficient execution.
        """
        filtered = self._rust.filter().by_kwargs(**kwargs)
        # Return a new instance of the same class wrapping the filtered Rust collection
        filtered_collection = self.__class__.__new__(self.__class__)
        filtered_collection.__dict__.update(self.__dict__)
        filtered_collection._rust = filtered
        return filtered_collection

    def size(self):
        """
        Returns the number of entities in the collection.
        """
        return self._rust.size()

    def ids(self):
        """
        Returns all IDs in the collection.
        """
        return list(self._rust.ids())

    def has(self, entity_id):
        """
        Checks if an entity exists in the collection.
        """
        return self._rust.has(entity_id)

    def attr(self):
        """
        Returns the attribute manager for this collection.
        """
        # Subclasses should override if needed
        return self._rust.attr()

    def __iter__(self):
        """
        Returns an iterator over entities in the collection.
        """
        return iter(self._rust.ids())

    def __getitem__(self, key):
        """
        Returns a proxy or data object for the given entity ID.
        
        Provides indexed access to entity data and attributes, referencing storage directly.
        Args:
            key: Entity ID.
        Returns:
            Proxy: Proxy object for entity.
        Raises:
            KeyError: If entity does not exist.
        """
        if not self.has(key):
            raise KeyError(f"Entity {key} not found in collection.")
        # Return a proxy object if available, else just the ID
        # Subclasses can override for richer proxy types
        return key

    def describe(self):
        """
        Returns a summary of the collection, including size, attribute schema, and sample data.
        
        Useful for diagnostics, debugging, and API introspection.
        Returns:
            dict: Collection summary.
        """
        summary = {
            "size": self.size,
            "ids_sample": self.ids()[:5],
            "attributes": getattr(self._rust, "schema", lambda: None)(),
        }
        return summary
