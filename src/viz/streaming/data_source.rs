//! DataSource trait for unified streaming data access
//!
//! Provides a common interface for all data structures to support streaming,
//! virtual scrolling, and real-time updates.

use crate::types::AttrValue;
use crate::viz::display::{ColumnSchema, DataType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unified interface that ALL data structures implement via delegation
pub trait DataSource: Send + Sync + std::fmt::Debug {
    /// Total number of rows in the data source
    fn total_rows(&self) -> usize;

    /// Total number of columns in the data source
    fn total_cols(&self) -> usize;

    /// Get a window of data for streaming/virtual scrolling
    fn get_window(&self, start: usize, count: usize) -> DataWindow;

    /// Get schema information for columns
    fn get_schema(&self) -> DataSchema;

    /// Check if this data source supports real-time streaming
    fn supports_streaming(&self) -> bool;

    /// Get data types for all columns
    fn get_column_types(&self) -> Vec<DataType>;

    /// Get column names
    fn get_column_names(&self) -> Vec<String>;

    /// Get a unique identifier for caching
    fn get_cache_key(&self, start: usize, count: usize) -> WindowKey {
        WindowKey {
            source_id: self.get_source_id(),
            start,
            count,
            version: self.get_version(),
        }
    }

    /// Get source identifier for caching
    fn get_source_id(&self) -> String;

    /// Get version for cache invalidation
    fn get_version(&self) -> u64;

    // NEW: Graph visualization support
    /// Check if this data source supports graph visualization
    fn supports_graph_view(&self) -> bool {
        false
    }

    /// Get graph nodes for visualization
    fn get_graph_nodes(&self) -> Vec<GraphNode> {
        Vec::new()
    }

    /// Get graph edges for visualization  
    fn get_graph_edges(&self) -> Vec<GraphEdge> {
        Vec::new()
    }

    /// Get graph metadata for visualization
    fn get_graph_metadata(&self) -> GraphMetadata {
        GraphMetadata::default()
    }

    /// Compute layout positions for nodes
    fn compute_layout(&self, _algorithm: LayoutAlgorithm) -> Vec<NodePosition> {
        Vec::new()
    }
}

/// Data window returned by DataSource for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataWindow {
    /// Column headers
    pub headers: Vec<String>,

    /// Rows of data (each row is a vec of values)
    pub rows: Vec<Vec<AttrValue>>,

    /// Schema information
    pub schema: DataSchema,

    /// Total number of rows in complete dataset
    pub total_rows: usize,

    /// Starting offset of this window
    pub start_offset: usize,

    /// Metadata for this window
    pub metadata: DataWindowMetadata,
}

/// Schema information for a data source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchema {
    /// Column schemas
    pub columns: Vec<ColumnSchema>,

    /// Primary key column (if any)
    pub primary_key: Option<String>,

    /// Source type (table, array, matrix)
    pub source_type: String,
}

/// Cache key for data windows
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct WindowKey {
    pub source_id: String,
    pub start: usize,
    pub count: usize,
    pub version: u64,
}

/// Metadata for data windows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataWindowMetadata {
    /// When this window was created
    pub created_at: std::time::SystemTime,

    /// Whether this is a cached result
    pub is_cached: bool,

    /// Load time in milliseconds
    pub load_time_ms: u64,

    /// Additional metadata
    pub extra: HashMap<String, String>,
}

impl DataWindow {
    /// Create a new data window
    pub fn new(
        headers: Vec<String>,
        rows: Vec<Vec<AttrValue>>,
        schema: DataSchema,
        total_rows: usize,
        start_offset: usize,
    ) -> Self {
        let created_at = std::time::SystemTime::now();

        Self {
            headers,
            rows,
            schema,
            total_rows,
            start_offset,
            metadata: DataWindowMetadata {
                created_at,
                is_cached: false,
                load_time_ms: 0,
                extra: HashMap::new(),
            },
        }
    }

    /// Check if window is empty
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Get window size
    pub fn size(&self) -> usize {
        self.rows.len()
    }

    /// Mark as cached
    pub fn mark_cached(&mut self) {
        self.metadata.is_cached = true;
    }

    /// Set load time
    pub fn set_load_time(&mut self, load_time_ms: u64) {
        self.metadata.load_time_ms = load_time_ms;
    }
}

// =============================================================================
// GRAPH VISUALIZATION DATA STRUCTURES
// =============================================================================

/// Node data for graph visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: Option<String>,
    pub attributes: HashMap<String, AttrValue>,
    pub position: Option<Position>,
}

/// Edge data for graph visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub label: Option<String>,
    pub weight: Option<f64>,
    pub attributes: HashMap<String, AttrValue>,
}

/// 2D position for nodes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

/// Graph metadata for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub node_count: usize,
    pub edge_count: usize,
    pub is_directed: bool,
    pub has_weights: bool,
    pub attribute_types: HashMap<String, String>,
}

impl Default for GraphMetadata {
    fn default() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            is_directed: false,
            has_weights: false,
            attribute_types: HashMap::new(),
        }
    }
}

/// Layout algorithms for graph visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutAlgorithm {
    /// Force-directed layout with physics simulation
    ForceDirected {
        charge: f64,
        distance: f64,
        iterations: usize,
    },
    /// Circular layout arranging nodes in a circle
    Circular {
        radius: Option<f64>,
        start_angle: f64,
    },
    /// Hierarchical layout for tree-like structures
    Hierarchical {
        direction: HierarchicalDirection,
        layer_spacing: f64,
        node_spacing: f64,
    },
    /// Grid layout for regular arrangements
    Grid { columns: usize, cell_size: f64 },
    /// Honeycomb layout arranging nodes in hexagonal grid
    Honeycomb {
        cell_size: f64,
        energy_optimization: bool,
        iterations: usize,
    },
}

/// Direction for hierarchical layouts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HierarchicalDirection {
    TopDown,
    BottomUp,
    LeftRight,
    RightLeft,
}

/// Node position for layout algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePosition {
    pub node_id: String,
    pub position: Position,
}
