//! Core traits for the BaseArray chaining system

use crate::storage::array::ArrayIterator;

/// Core array operations trait - the foundation for all array types
pub trait ArrayOps<T> {
    /// Get the number of elements in the array
    fn len(&self) -> usize;

    /// Check if the array is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get element at index (None if out of bounds)
    fn get(&self, index: usize) -> Option<&T>;

    /// Create an eager iterator for chaining operations (immediate execution)
    fn iter(&self) -> ArrayIterator<T>
    where
        T: Clone + 'static;

    /// Create a lazy iterator for chaining operations (deferred execution with optimization)
    fn lazy_iter(&self) -> crate::storage::array::LazyArrayIterator<T>
    where
        T: Clone + 'static,
    {
        crate::storage::array::LazyArrayIterator::new(self.to_vec())
    }

    /// Convert to Vec (helper for lazy iterator)
    fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        (0..self.len())
            .filter_map(|i| self.get(i).cloned())
            .collect()
    }
}

// =============================================================================
// Marker traits for trait-based method injection
// =============================================================================
// These traits enable automatic method availability based on element type

/// Marker trait for subgraph-like objects
/// Types implementing this get access to subgraph operations in ArrayIterator
pub trait SubgraphLike {
    // Marker trait - no required methods
    // Methods are provided via ArrayIterator<T: SubgraphLike> impl blocks
}

/// Marker trait for node ID-like objects  
/// Types implementing this get access to node operations in ArrayIterator
pub trait NodeIdLike {
    // Marker trait - no required methods
    // Methods are provided via ArrayIterator<T: NodeIdLike> impl blocks
}

/// Marker trait for meta-node-like objects
/// Types implementing this get access to meta-node operations in ArrayIterator  
pub trait MetaNodeLike {
    // Marker trait - no required methods
    // Methods are provided via ArrayIterator<T: MetaNodeLike> impl blocks
}

/// Marker trait for edge-like objects
/// Types implementing this get access to edge operations in ArrayIterator
pub trait EdgeLike {
    // Marker trait - no required methods
    // Methods are provided via ArrayIterator<T: EdgeLike> impl blocks
}
