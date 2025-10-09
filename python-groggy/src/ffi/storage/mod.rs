//! Python FFI bindings for storage types
//!
//! This module contains Python bindings for storage and view types

pub mod array;
pub mod array_array; // Array of arrays with aggregation
pub mod matrix;
pub mod num_array;
pub mod table; // Re-enabled NEW BaseTable FFI for Phase 5 // Statistical array with numerical operations
               // Removed: bool_array and simple_stats_array - functionality integrated into unified NumArray
pub mod accessors; // Essential FFI - keep enabled
pub mod components;
pub mod edges_array; // Phase 2: Specialized arrays
pub mod matrix_array;
pub mod nodes_array; // Phase 2: Specialized arrays
pub mod subgraph_array; // Phase 2: Specialized arrays
pub mod table_array; // Phase 2: Specialized arrays (OLD implementation)
pub mod table_array_core; // NEW: Core-delegating TableArray
pub mod views; // Essential FFI - keep enabled // Phase 2: Specialized arrays

// Array of arrays with aggregation
// Re-enabled NEW BaseTable FFI for Phase 5 // Statistical array with numerical operations
// Removed: bool_array and simple_stats_array exports - functionality integrated into unified NumArray
// Essential FFI
// Phase 2: Specialized arrays
// Phase 2: Specialized arrays
// Phase 2: Specialized arrays
// Phase 2: Specialized arrays (OLD implementation)
// NEW: Core-delegating TableArray
// Essential FFI // Phase 2: Specialized arrays
