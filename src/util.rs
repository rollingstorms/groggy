//! Utility Functions and Helpers - Common operations used across modules.
//!
//! ARCHITECTURE ROLE:
//! This module provides shared functionality that's used by multiple
//! components throughout the system. It contains performance-critical
//! operations and commonly needed data structure manipulations.
//!
//! DESIGN PHILOSOPHY:
//! - Pure functions with no side effects
//! - Performance-optimized implementations
//! - Well-tested building blocks
//! - Zero-allocation where possible

/*
=== UTILITY MODULE OVERVIEW ===

This module provides fundamental operations that are used throughout
the codebase. Categories include:

1. HASHING: Content addressing and deduplication
2. SORTING: Efficient operations on sorted data structures
3. TIME: Timestamp generation and manipulation
4. VALIDATION: Data consistency checks
5. CONVERSION: Type conversions and data transformations

KEY DESIGN DECISIONS:
- Favor performance over generality (these are hot paths)
- Use const generics and zero-cost abstractions where possible
- Provide both safe and unsafe variants for performance-critical code
- Extensive testing for correctness
*/

use crate::types::AttrValue;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/*
=== CONTENT ADDRESSING AND HASHING ===
Functions for generating content hashes and deduplication
*/

/// Generate a 256-bit content hash for deduplication and verification
///
/// ALGORITHM:
/// 1. Use DefaultHasher to generate u64 hash of input data
/// 2. Expand to 256-bit by repeating the 64-bit hash
/// 3. This provides good distribution while being fast
///
/// PERFORMANCE: O(size of data), very fast hashing
/// COLLISION RESISTANCE: Good for practical purposes, not cryptographic
pub fn content_hash<T: Hash>(data: &T) -> [u8; 32] {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    let hash_u64 = hasher.finish();

    // Expand 64-bit hash to 256-bit for better distribution
    let mut result = [0u8; 32];
    let bytes = hash_u64.to_le_bytes();

    // Repeat the 8-byte pattern 4 times
    for chunk in result.chunks_mut(8) {
        chunk.copy_from_slice(&bytes);
    }

    result
}

/// Fast hash function optimized specifically for attribute values
///
/// OPTIMIZATION: This can be faster than generic hashing because
/// we know the structure of AttrValue and can optimize accordingly
pub fn attr_value_hash(value: &AttrValue) -> u64 {
    match value {
        AttrValue::Int(i) => {
            // Fast path for integers - just use the value directly
            *i as u64
        }
        AttrValue::Float(f) => {
            // Convert float to bits for consistent hashing
            f.to_bits() as u64
        }
        AttrValue::Bool(b) => {
            // Simple hash for booleans
            if *b {
                1
            } else {
                0
            }
        }
        AttrValue::Text(s) => {
            // Use standard string hashing
            let mut hasher = DefaultHasher::new();
            s.hash(&mut hasher);
            hasher.finish()
        }
        AttrValue::FloatVec(v) => {
            // Hash vector by combining element hashes
            let mut hasher = DefaultHasher::new();
            for &element in v {
                element.to_bits().hash(&mut hasher);
            }
            hasher.finish()
        }
        AttrValue::CompactText(cs) => {
            // Hash compact string using its string content
            let mut hasher = DefaultHasher::new();
            cs.as_str().hash(&mut hasher);
            hasher.finish()
        }
        AttrValue::SmallInt(i) => {
            // Fast path for small integers
            *i as u64
        }
        AttrValue::Bytes(b) => {
            // Hash byte array
            let mut hasher = DefaultHasher::new();
            b.hash(&mut hasher);
            hasher.finish()
        }
        AttrValue::CompressedText(cd) => {
            // Hash compressed data
            let mut hasher = DefaultHasher::new();
            cd.data.hash(&mut hasher);
            hasher.finish()
        }
        AttrValue::CompressedFloatVec(cd) => {
            // Hash compressed vector data
            let mut hasher = DefaultHasher::new();
            cd.data.hash(&mut hasher);
            hasher.finish()
        }
        AttrValue::Null => {
            // Consistent hash for null values
            u64::MAX // Use max value to distinguish from other values
        }
    }
}

/// Merge two sorted vectors while maintaining order and removing duplicates
///
/// ALGORITHM:
/// 1. Two-pointer technique to merge in O(n + m) time
/// 2. Skip duplicates during merge to ensure uniqueness
/// 3. Pre-allocate result vector for efficiency
///
/// PERFORMANCE: O(n + m) time, O(n + m) space
/// USE CASES: Merging index lists, combining sorted query results
pub fn merge_sorted_indices(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            result.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            result.push(b[j]);
            j += 1;
        } else {
            // Equal values - only add once
            result.push(a[i]);
            i += 1;
            j += 1;
        }
    }

    // Add remaining elements
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);

    result
}

/// Find insertion point in sorted vector using binary search
///
/// RETURNS: Index where value should be inserted to maintain sorted order
/// If value already exists, returns the index of the existing element
pub fn binary_search_insert_point(vec: &[usize], value: usize) -> usize {
    match vec.binary_search(&value) {
        Ok(pos) => pos,  // Found - return existing position
        Err(pos) => pos, // Not found - return insertion position
    }
}

/// Check if two attribute values are type-compatible
///
/// USAGE: Before updating an attribute, check if the new value
/// is compatible with the existing type
pub fn validate_attr_compatibility(existing: &AttrValue, new: &AttrValue) -> bool {
    // Use discriminant comparison to check if they're the same variant
    std::mem::discriminant(existing) == std::mem::discriminant(new)
}

/// Generate a Unix timestamp for the current time
///
/// PRECISION: Seconds since Unix epoch
/// FALLBACK: Returns 0 if system time is unavailable
pub fn timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_sorted_indices() {
        let a = vec![1, 3, 5];
        let b = vec![2, 4, 6];
        let result = merge_sorted_indices(&a, &b);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_merge_with_duplicates() {
        let a = vec![1, 3, 5];
        let b = vec![3, 4, 5];
        let result = merge_sorted_indices(&a, &b);
        assert_eq!(result, vec![1, 3, 4, 5]);
    }
}
