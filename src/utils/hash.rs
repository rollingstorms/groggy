// src_new/utils/hash.rs
// Fast hashing utilities for nodes and edges using xxhash and serialization.
// Adapted from src_old/utils/hash.rs for high-performance content addressing.
/// Fast hashing utilities for nodes and edges

use xxhash_rust::xxh3::xxh3_64;

/// Node hashing functions
pub fn hash_node<T: serde::Serialize>(node: &T) -> u64 {
    // Efficient node hashing using xxhash (as in src_old)
    let bytes = bincode::serialize(node).unwrap();
    xxhash_rust::xxh3::xxh3_64(&bytes)
}

/// Edge hashing functions
pub fn hash_edge<T: serde::Serialize>(edge: &T) -> u64 {
    // Efficient edge hashing using xxhash (as in src_old)
    let bytes = bincode::serialize(edge).unwrap();
    xxhash_rust::xxh3::xxh3_64(&bytes)
}
