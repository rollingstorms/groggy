//! Python FFI bindings for storage types
//!
//! This module contains Python bindings for storage and view types

pub mod matrix;
pub mod table; // Re-enabled NEW BaseTable FFI for Phase 5
pub mod array;
pub mod num_array; // Statistical array with numerical operations
pub mod simple_stats_array; // Minimal test version
pub mod accessors; // Essential FFI - keep enabled
pub mod views; // Essential FFI - keep enabled  
pub mod components;
pub mod subgraph_array; // Phase 2: Specialized arrays
pub mod table_array; // Phase 2: Specialized arrays
pub mod nodes_array; // Phase 2: Specialized arrays
pub mod edges_array; // Phase 2: Specialized arrays
pub mod matrix_array; // Phase 2: Specialized arrays

pub use matrix::*;
pub use table::*; // Re-enabled NEW BaseTable FFI for Phase 5
pub use array::*;
pub use num_array::*; // Statistical array with numerical operations
pub use simple_stats_array::*; // Minimal test version
pub use accessors::*; // Essential FFI
pub use views::*; // Essential FFI
pub use components::*;
pub use subgraph_array::*; // Phase 2: Specialized arrays
pub use table_array::*; // Phase 2: Specialized arrays
pub use nodes_array::*; // Phase 2: Specialized arrays
pub use edges_array::*; // Phase 2: Specialized arrays
pub use matrix_array::*; // Phase 2: Specialized arrays