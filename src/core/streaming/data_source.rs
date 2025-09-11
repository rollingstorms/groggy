//! DataSource trait for unified streaming data access
//!
//! Provides a common interface for all data structures to support streaming,
//! virtual scrolling, and real-time updates.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::types::AttrValue;
use crate::core::display::{DataType, ColumnSchema};

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