//! Table module - unified table system built on BaseArray columns

pub mod base;
pub mod edges;
pub mod graph_table;
mod group_views;
pub mod integration_tests;
mod materialize;
pub mod nodes;
mod numeric;
pub mod projection;
mod provenance;
mod structural;
pub mod table_array;
mod text;
pub mod traits;
mod visualization;
mod writeback;
pub use writeback::TableWriteback;

// Re-export core types
pub use base::BaseTable;
pub use edges::{EdgeConfig, EdgesTable};
pub use graph_table::{
    BundleMetadata, ConflictResolution, GraphTable, GraphTableShape, ValidationPolicy,
    ValidationReport, ValidationStrictness,
};
pub use nodes::NodesTable;
pub use table_array::TableArray;
pub use traits::{Table, TableRow, TableRowIterator};
