# python_new/groggy/collections/base.py

class BaseCollection:
    """
    Abstract base for NodeCollection/EdgeCollection (composition, not inheritance).
    
    Defines the shared API for collection types, including batch operations, filtering, attribute access, and iteration.
    Concrete collections must implement these methods, delegating to efficient storage backends as appropriate.
    """

    def add(self, *args, **kwargs):
        """
        Adds one or more entities (nodes/edges) to the collection.
        
        Supports single or batch addition. Delegates to storage backend for efficient insertion.
        Args:
            *args, **kwargs: Entity data or batch.
        Raises:
            ValueError: On invalid input or duplicate IDs.
        """
        # TODO: 1. Detect single/batch; 2. Delegate to backend; 3. Handle errors.
        pass

    def remove(self, *args, **kwargs):
        """
        Removes one or more entities (nodes/edges) from the collection.
        
        Supports single or batch removal. Marks entities as deleted in storage for lazy cleanup.
        Args:
            *args, **kwargs: IDs or batch.
        Raises:
            KeyError: If entity does not exist.
        """
        # TODO: 1. Detect single/batch; 2. Mark as deleted; 3. Schedule cleanup.
        pass

    def filter(self, *args, **kwargs):
        """
        Returns a filtered view of the collection based on attribute values or custom predicates.
        
        Supports chaining and composition of filters. Delegates to storage backend for efficient execution.
        Args:
            *args, **kwargs: Filtering criteria.
        Returns:
            BaseCollection: Filtered collection view.
        """
        # TODO: 1. Build filter plan; 2. Delegate to backend; 3. Return filtered view.
        pass

    def size(self):
        """
        Returns the number of entities in the collection.
        
        Fast lookup from storage metadata; does not require iteration.
        Returns:
            int: Entity count.
        """
        # TODO: 1. Query storage metadata.
        pass

    def ids(self):
        """
        Returns all IDs in the collection.
        
        Reads directly from storage index for efficiency. May return a view or copy.
        Returns:
            list: List of entity IDs.
        """
        # TODO: 1. Access storage index; 2. Return IDs.
        pass

    def has(self, *args, **kwargs):
        """
        Checks if an entity exists in the collection.
        
        Fast O(1) lookup in storage index. Returns True if present and not deleted.
        Args:
            *args, **kwargs: Entity ID or batch.
        Returns:
            bool: True if entity exists, False otherwise.
        """
        # TODO: 1. Lookup in index; 2. Check deleted flag.
        pass

    def attr(self):
        """
        Returns the attribute manager for this collection.
        
        Provides access to fast, vectorized attribute operations for all entities.
        Returns:
            AttributeManager: Attribute manager interface.
        """
        # TODO: 1. Instantiate AttributeManager; 2. Bind collection context.
        pass

    def __iter__(self):
        """
        Returns an iterator over entities in the collection.
        
        Supports lazy iteration over storage data, yielding proxies or raw IDs as needed.
        Returns:
            Iterator: Iterator over entities.
        """
        # TODO: 1. Create iterator; 2. Yield proxies or IDs.
        pass

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
        # TODO: 1. Lookup entity; 2. Return proxy or error.
        pass

    def describe(self):
        """
        Returns a summary of the collection, including size, attribute schema, and sample data.
        
        Useful for diagnostics, debugging, and API introspection.
        Returns:
            dict: Collection summary.
        """
        # TODO: 1. Gather stats; 2. Query attribute schema; 3. Return summary dict.
        pass
