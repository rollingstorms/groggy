//! Table module - unified table system built on BaseArray columns

pub mod traits;
pub mod base;
pub mod nodes;
pub mod edges;
pub mod graph_table;
pub mod integration_tests;

// Re-export core types
pub use traits::{Table, TableIterator, TableOperation};
pub use base::BaseTable;
pub use nodes::NodesTable;
pub use edges::{EdgesTable, EdgeConfig};
pub use graph_table::{GraphTable, ValidationPolicy, ValidationStrictness, ValidationReport, BundleMetadata, ConflictResolution};