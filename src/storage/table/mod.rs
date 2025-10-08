//! Table module - unified table system built on BaseArray columns

pub mod base;
pub mod edges;
pub mod graph_table;
pub mod integration_tests;
pub mod nodes;
pub mod table_array;
pub mod traits;

// Re-export core types
pub use base::BaseTable;
pub use edges::{EdgeConfig, EdgesTable};
pub use graph_table::{
    BundleMetadata, ConflictResolution, GraphTable, ValidationPolicy, ValidationReport,
    ValidationStrictness,
};
pub use nodes::NodesTable;
pub use table_array::TableArray;
pub use traits::{Table, TableRow, TableRowIterator};
