//! BaseTable - unified table implementation built on BaseArray columns

use super::traits::{Table, TableIterator};
use crate::storage::array::{BaseArray, ArrayOps};
use crate::errors::GraphResult;
use crate::types::AttrValue;
use crate::core::display::{DisplayEngine, DisplayConfig, ColumnSchema, DataType, OutputFormat};
use crate::core::{DisplayDataWindow, DisplayDataSchema, StreamingDataWindow, StreamingDataSchema};
use crate::core::streaming::{DataSource, StreamingServer, StreamingConfig};
use std::collections::HashMap;

/// Unified table implementation using BaseArray columns
/// This provides the foundation for all table types in the system
#[derive(Debug)]
pub struct BaseTable {
    /// Columns stored as BaseArrays with AttrValue type
    columns: HashMap<String, BaseArray<AttrValue>>,
    /// Column order for consistent iteration
    column_order: Vec<String>,
    /// Number of rows (derived from first column)
    nrows: usize,
    /// Display engine for unified formatting (FOUNDATION ONLY - specialized types delegate)
    display_engine: DisplayEngine,
    /// Streaming server for real-time updates (FOUNDATION ONLY - Phase 2)
    streaming_server: Option<StreamingServer>,
    /// Active server handles to keep them alive
    active_server_handles: Vec<crate::core::streaming::websocket_server::ServerHandle>,
    /// Streaming configuration
    streaming_config: StreamingConfig,
    /// Source ID for caching
    source_id: String,
    /// Version for cache invalidation
    version: u64,
}

impl Clone for BaseTable {
    fn clone(&self) -> Self {
        Self {
            columns: self.columns.clone(),
            column_order: self.column_order.clone(),
            nrows: self.nrows,
            display_engine: self.display_engine.clone(),
            streaming_server: None, // Don't clone server, create new instance when needed
            active_server_handles: Vec::new(), // Don't clone server handles
            streaming_config: self.streaming_config.clone(),
            source_id: self.source_id.clone(),
            version: self.version,
        }
    }
}

impl BaseTable {
    /// Create a new empty BaseTable
    pub fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        
        Self {
            columns: HashMap::new(),
            column_order: Vec::new(),
            nrows: 0,
            display_engine: DisplayEngine::new(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: StreamingConfig::default(),
            source_id: format!("basetable_{}", timestamp),
            version: 1,
        }
    }
    
    /// Create a BaseTable from columns
    pub fn from_columns(columns: HashMap<String, BaseArray<AttrValue>>) -> GraphResult<Self> {
        if columns.is_empty() {
            return Ok(Self::new());
        }
        
        // Find the maximum column length to handle sparse attributes
        let max_len = columns.values().map(|col| col.len()).max().unwrap();
        
        // Pad shorter columns with nulls to handle sparse attribute matrices
        let mut normalized_columns = HashMap::new();
        for (name, column) in columns {
            if column.len() < max_len {
                // Pad with nulls to match max length
                let mut data = column.data().clone();
                data.resize(max_len, crate::types::AttrValue::Null);
                normalized_columns.insert(name, BaseArray::from_attr_values(data));
            } else {
                normalized_columns.insert(name, column);
            }
        }
        
        let column_order: Vec<String> = normalized_columns.keys().cloned().collect();
        
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        
        Ok(Self {
            columns: normalized_columns,
            column_order,
            nrows: max_len,
            display_engine: DisplayEngine::new(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: StreamingConfig::default(),
            source_id: format!("basetable_{}", timestamp),
            version: 1,
        })
    }
    
    /// Create a BaseTable with specific column order
    pub fn with_column_order(columns: HashMap<String, BaseArray<AttrValue>>, column_order: Vec<String>) -> GraphResult<Self> {
        // Validate column order matches available columns
        for col in &column_order {
            if !columns.contains_key(col) {
                return Err(crate::errors::GraphError::InvalidInput(
                    format!("Column '{}' specified in order but not found in columns", col)
                ));
            }
        }
        
        if columns.is_empty() {
            return Ok(Self::new());
        }
        
        // Validate all columns have same length
        let first_len = columns.values().next().unwrap().len();
        for (name, column) in &columns {
            if column.len() != first_len {
                return Err(crate::errors::GraphError::InvalidInput(
                    format!("Column '{}' has length {} but expected {}", name, column.len(), first_len)
                ));
            }
        }
        
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        
        Ok(Self {
            columns,
            column_order,
            nrows: first_len,
            display_engine: DisplayEngine::new(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: StreamingConfig::default(),
            source_id: format!("basetable_{}", timestamp),
            version: 1,
        })
    }
    
    /// Get internal columns reference (for advanced usage)
    pub fn columns(&self) -> &HashMap<String, BaseArray<AttrValue>> {
        &self.columns
    }
    
    /// Get column order
    pub fn column_order(&self) -> &[String] {
        &self.column_order
    }
    
    /// Convert to NodesTable if appropriate (contains node_id column)
    pub fn as_nodes_table(&self) -> Option<super::nodes::NodesTable> {
        if self.has_column("node_id") {
            Some(super::nodes::NodesTable::from_base_table(self.clone()).ok()?)
        } else {
            None
        }
    }
    
    /// Convert to NodesTable with UID key validation (Phase 2 plan method)
    pub fn to_nodes(self, uid_key: &str) -> GraphResult<super::nodes::NodesTable> {
        // Validate that the UID key column exists
        if !self.has_column(uid_key) {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("UID column '{}' not found", uid_key)
            ));
        }
        
        // If uid_key is not "node_id", we need to rename it
        let table_with_node_id = if uid_key == "node_id" {
            self
        } else {
            // For now, require that the UID column is already named "node_id"
            // Future enhancement could rename the column
            return Err(crate::errors::GraphError::InvalidInput(
                format!("UID column must be named 'node_id', found '{}'", uid_key)
            ));
        };
        
        super::nodes::NodesTable::from_base_table(table_with_node_id)
    }
    
    /// Convert to EdgesTable if appropriate (contains edge_id column)  
    pub fn as_edges_table(&self) -> Option<super::edges::EdgesTable> {
        if self.has_column("edge_id") {
            Some(super::edges::EdgesTable::from_base_table(self.clone()).ok()?)
        } else {
            None
        }
    }
    
    /// Convert to EdgesTable with validation (Phase 3 plan method)
    pub fn to_edges(self) -> GraphResult<super::edges::EdgesTable> {
        // Validate that required edge columns exist
        let required_cols = ["edge_id", "source", "target"];
        for col in &required_cols {
            if !self.has_column(col) {
                return Err(crate::errors::GraphError::InvalidInput(
                    format!("EdgesTable requires '{}' column", col)
                ));
            }
        }
        
        super::edges::EdgesTable::from_base_table(self)
    }
    
    // =============================================================================
    // Setting Methods - Comprehensive assignment and modification operations
    // =============================================================================
    
    /// Assign updates to multiple columns at once
    /// 
    /// # Arguments
    /// * `updates` - HashMap mapping column names to new values
    /// 
    /// # Examples
    /// ```
    /// let mut updates = HashMap::new();
    /// updates.insert("bonus".to_string(), vec![AttrValue::Float(1000.0), AttrValue::Float(1500.0)]);
    /// table.assign(updates)?;
    /// ```
    pub fn assign(&mut self, updates: HashMap<String, Vec<crate::types::AttrValue>>) -> GraphResult<()> {
        use crate::types::{AttrValue, AttrValueType};
        
        // Validate that all update vectors have the correct length
        for (col_name, values) in &updates {
            if values.len() != self.nrows {
                return Err(crate::errors::GraphError::InvalidInput(
                    format!("Column '{}' update has {} values but table has {} rows", col_name, values.len(), self.nrows)
                ));
            }
        }
        
        // Apply updates
        for (col_name, values) in updates {
            // Infer dtype from the first non-null value
            let dtype = values.iter()
                .find(|v| !matches!(v, AttrValue::Null))
                .map(|v| match v {
                    AttrValue::Int(_) => AttrValueType::Int,
                    AttrValue::SmallInt(_) => AttrValueType::SmallInt,
                    AttrValue::Float(_) => AttrValueType::Float,
                    AttrValue::Text(_) => AttrValueType::Text,
                    AttrValue::CompactText(_) => AttrValueType::CompactText,
                    AttrValue::Bool(_) => AttrValueType::Bool,
                    AttrValue::FloatVec(_) => AttrValueType::FloatVec,
                    AttrValue::Bytes(_) => AttrValueType::Bytes,
                    AttrValue::Null => AttrValueType::Null,
                    _ => AttrValueType::Text, // fallback for other types
                })
                .unwrap_or(AttrValueType::Text);
            
            let new_column = BaseArray::new(values);
            
            // Add to column order if it's a new column
            if !self.columns.contains_key(&col_name) {
                self.column_order.push(col_name.clone());
            }
            
            self.columns.insert(col_name, new_column);
        }
        
        Ok(())
    }
    
    /// Set an entire column with new values
    /// 
    /// # Arguments
    /// * `column_name` - Name of the column to set
    /// * `values` - Vector of new values for the column
    pub fn set_column(&mut self, column_name: &str, values: Vec<crate::types::AttrValue>) -> GraphResult<()> {
        use crate::types::{AttrValue, AttrValueType};
        
        if values.len() != self.nrows {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("Column '{}' has {} values but table has {} rows", column_name, values.len(), self.nrows)
            ));
        }
        
        // Infer dtype from the first non-null value
        let dtype = values.iter()
            .find(|v| !matches!(v, AttrValue::Null))
            .map(|v| match v {
                AttrValue::Int(_) => AttrValueType::Int,
                AttrValue::SmallInt(_) => AttrValueType::SmallInt,
                AttrValue::Float(_) => AttrValueType::Float,
                AttrValue::Text(_) => AttrValueType::Text,
                AttrValue::CompactText(_) => AttrValueType::CompactText,
                AttrValue::Bool(_) => AttrValueType::Bool,
                AttrValue::FloatVec(_) => AttrValueType::FloatVec,
                AttrValue::Bytes(_) => AttrValueType::Bytes,
                AttrValue::Null => AttrValueType::Null,
                _ => AttrValueType::Text,
            })
            .unwrap_or(AttrValueType::Text);
        
        let new_column = BaseArray::new(values);
        
        // Add to column order if it's a new column
        if !self.columns.contains_key(column_name) {
            self.column_order.push(column_name.to_string());
        }
        
        self.columns.insert(column_name.to_string(), new_column);
        Ok(())
    }
    
    /// Set a single value at a specific row and column
    /// 
    /// # Arguments
    /// * `row` - Row index (0-based)
    /// * `column_name` - Name of the column
    /// * `value` - New value to set
    pub fn set_value(&mut self, row: usize, column_name: &str, value: crate::types::AttrValue) -> GraphResult<()> {
        if row >= self.nrows {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("Row index {} out of bounds (table has {} rows)", row, self.nrows)
            ));
        }
        
        let column = self.columns.get_mut(column_name)
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                format!("Column '{}' not found in table", column_name)
            ))?;
        
        column.set(row, value)?;
        Ok(())
    }
    
    /// Set values for multiple rows in a column using a boolean mask
    /// 
    /// # Arguments
    /// * `mask` - Boolean vector indicating which rows to update
    /// * `column_name` - Name of the column to update
    /// * `value` - Value to set for all masked rows
    pub fn set_values_by_mask(&mut self, mask: &[bool], column_name: &str, value: crate::types::AttrValue) -> GraphResult<()> {
        if mask.len() != self.nrows {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("Mask length {} does not match table rows {}", mask.len(), self.nrows)
            ));
        }
        
        let column = self.columns.get_mut(column_name)
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                format!("Column '{}' not found in table", column_name)
            ))?;
        
        for (i, &should_update) in mask.iter().enumerate() {
            if should_update {
                column.set(i, value.clone())?;
            }
        }
        
        Ok(())
    }
    
    /// Set values for a range of rows in a column
    /// 
    /// # Arguments
    /// * `start` - Starting row index (inclusive)
    /// * `end` - Ending row index (exclusive)
    /// * `step` - Step size (default 1 for consecutive rows)
    /// * `column_name` - Name of the column to update
    /// * `value` - Value to set for all rows in the range
    pub fn set_values_by_range(&mut self, start: usize, end: usize, step: usize, column_name: &str, value: crate::types::AttrValue) -> GraphResult<()> {
        if start >= self.nrows || end > self.nrows {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("Range [{}, {}) is out of bounds for table with {} rows", start, end, self.nrows)
            ));
        }
        
        if step == 0 {
            return Err(crate::errors::GraphError::InvalidInput(
                "Step size must be greater than 0".to_string()
            ));
        }
        
        let column = self.columns.get_mut(column_name)
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                format!("Column '{}' not found in table", column_name)
            ))?;
        
        let mut i = start;
        while i < end {
            column.set(i, value.clone())?;
            i += step;
        }
        
        Ok(())
    }
    
    /// Set values for multiple columns and multiple rows simultaneously
    /// 
    /// # Arguments
    /// * `row_indices` - Vector of row indices to update
    /// * `column_updates` - HashMap mapping column names to values
    pub fn set_multiple_values(&mut self, row_indices: &[usize], column_updates: &HashMap<String, crate::types::AttrValue>) -> GraphResult<()> {
        // Validate row indices
        for &row_idx in row_indices {
            if row_idx >= self.nrows {
                return Err(crate::errors::GraphError::InvalidInput(
                    format!("Row index {} out of bounds (table has {} rows)", row_idx, self.nrows)
                ));
            }
        }
        
        // Validate columns exist
        for col_name in column_updates.keys() {
            if !self.columns.contains_key(col_name) {
                return Err(crate::errors::GraphError::InvalidInput(
                    format!("Column '{}' not found in table", col_name)
                ));
            }
        }
        
        // Apply updates
        for &row_idx in row_indices {
            for (col_name, value) in column_updates {
                let column = self.columns.get_mut(col_name).unwrap(); // Safe due to validation above
                column.set(row_idx, value.clone())?;
            }
        }
        
        Ok(())
    }
}

impl Default for BaseTable {
    fn default() -> Self {
        Self::new()
    }
}

impl Table for BaseTable {
    fn nrows(&self) -> usize {
        self.nrows
    }
    
    fn ncols(&self) -> usize {
        self.columns.len()
    }
    
    fn column_names(&self) -> &[String] {
        &self.column_order
    }
    
    fn column(&self, name: &str) -> Option<&BaseArray<AttrValue>> {
        self.columns.get(name)
    }
    
    fn column_by_index(&self, index: usize) -> Option<&BaseArray<AttrValue>> {
        self.column_order.get(index)
            .and_then(|name| self.columns.get(name))
    }
    
    fn has_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }
    
    fn head(&self, n: usize) -> Self {
        let end = std::cmp::min(n, self.nrows);
        self.slice(0, end)
    }
    
    fn tail(&self, n: usize) -> Self {
        let start = self.nrows.saturating_sub(n);
        self.slice(start, self.nrows)
    }
    
    fn slice(&self, start: usize, end: usize) -> Self {
        if start >= self.nrows || end <= start {
            return Self::new();
        }
        
        let actual_end = std::cmp::min(end, self.nrows);
        let mut new_columns = HashMap::new();
        
        for (name, column) in &self.columns {
            new_columns.insert(name.clone(), column.slice(start, actual_end));
        }
        
        Self {
            columns: new_columns,
            column_order: self.column_order.clone(),
            nrows: actual_end - start,
            display_engine: self.display_engine.clone(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: self.streaming_config.clone(),
            source_id: self.source_id.clone(),
            version: self.version + 1, // Increment version for new slice
        }
    }
    
    fn sort_by(&self, column: &str, ascending: bool) -> GraphResult<Self> {
        let sort_column = self.column(column)
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                format!("Column '{}' not found", column)
            ))?;
        
        // Get sort indices
        let indices = sort_column.sort_indices(ascending);
        
        // Apply indices to all columns
        let mut new_columns = HashMap::new();
        for (name, col) in &self.columns {
            new_columns.insert(name.clone(), col.take_indices(&indices)?);
        }
        
        Ok(Self {
            columns: new_columns,
            column_order: self.column_order.clone(),
            nrows: self.nrows,
            display_engine: self.display_engine.clone(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: self.streaming_config.clone(),
            source_id: self.source_id.clone(),
            version: self.version + 1,
        })
    }
    
    fn filter(&self, predicate: &str) -> GraphResult<Self> {
        // Enhanced predicate parsing supporting multiple operators
        let mask = self.evaluate_predicate(predicate)?;
        
        // Apply mask to all columns
        let mut new_columns = HashMap::new();
        for (name, col) in &self.columns {
            new_columns.insert(name.clone(), col.filter_by_mask(&mask)?);
        }
        
        let new_nrows = mask.iter().filter(|&&x| x).count();
        
        Ok(Self {
            columns: new_columns,
            column_order: self.column_order.clone(),
            nrows: new_nrows,
            display_engine: self.display_engine.clone(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: self.streaming_config.clone(),
            source_id: self.source_id.clone(),
            version: self.version + 1,
        })
    }
    
    fn group_by(&self, columns: &[String]) -> GraphResult<Vec<Self>> {
        if columns.is_empty() {
            return Ok(vec![self.clone()]);
        }

        // Validate that all group columns exist
        for col_name in columns {
            if !self.columns.contains_key(col_name) {
                return Err(crate::errors::GraphError::InvalidInput(
                    format!("Column '{}' not found in table", col_name)
                ));
            }
        }

        // Create groups by building composite keys
        let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        
        for row_idx in 0..self.nrows {
            // Build composite key for this row
            let mut key = Vec::new();
            for col_name in columns {
                if let Some(column) = self.columns.get(col_name) {
                    if let Some(value) = column.get(row_idx) {
                        key.push(format!("{:?}", value));
                    } else {
                        key.push("NULL".to_string());
                    }
                } else {
                    key.push("NULL".to_string());
                }
            }
            
            groups.entry(key).or_insert_with(Vec::new).push(row_idx);
        }

        // Create a table for each group
        let mut result = Vec::new();
        for (_, row_indices) in groups {
            let group_table = self.select_rows(&row_indices)?;
            result.push(group_table);
        }

        Ok(result)
    }

    
    fn select(&self, column_names: &[String]) -> GraphResult<Self> {
        let mut new_columns = HashMap::new();
        
        for name in column_names {
            let column = self.column(name)
                .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                    format!("Column '{}' not found", name)
                ))?;
            new_columns.insert(name.clone(), column.clone());
        }
        
        Ok(Self {
            columns: new_columns,
            column_order: column_names.to_vec(),
            nrows: self.nrows,
            display_engine: self.display_engine.clone(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: self.streaming_config.clone(),
            source_id: self.source_id.clone(),
            version: self.version + 1,
        })
    }
    
    fn with_column(&self, name: String, column: BaseArray<AttrValue>) -> GraphResult<Self> {
        if column.len() != self.nrows && self.nrows > 0 {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("New column has {} rows but table has {} rows", column.len(), self.nrows)
            ));
        }
        
        let mut new_columns = self.columns.clone();
        let mut new_order = self.column_order.clone();
        
        if !new_columns.contains_key(&name) {
            new_order.push(name.clone());
        }
        
        new_columns.insert(name, column.clone());
        
        Ok(Self {
            columns: new_columns,
            column_order: new_order,
            nrows: if self.nrows == 0 { column.len() } else { self.nrows },
            display_engine: self.display_engine.clone(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: self.streaming_config.clone(),
            source_id: self.source_id.clone(),
            version: self.version + 1,
        })
    }
    
    fn drop_columns(&self, column_names: &[String]) -> GraphResult<Self> {
        let mut new_columns = self.columns.clone();
        let mut new_order = self.column_order.clone();
        
        for name in column_names {
            new_columns.remove(name);
            new_order.retain(|x| x != name);
        }
        
        Ok(Self {
            columns: new_columns,
            column_order: new_order,
            nrows: self.nrows,
            display_engine: self.display_engine.clone(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: self.streaming_config.clone(),
            source_id: self.source_id.clone(),
            version: self.version + 1,
        })
    }
    
    fn iter(&self) -> TableIterator<Self> {
        TableIterator::new(self.clone())
    }
}

impl BaseTable {
    /// Filter by boolean mask
    pub fn filter_by_mask(&self, mask: &[bool]) -> GraphResult<Self> {
        if mask.len() != self.nrows {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("Mask length {} doesn't match table rows {}", mask.len(), self.nrows)
            ));
        }
        
        // Apply mask to all columns
        let mut new_columns = HashMap::new();
        for (name, col) in &self.columns {
            new_columns.insert(name.clone(), col.filter_by_mask(mask)?);
        }
        
        let new_nrows = mask.iter().filter(|&&x| x).count();
        
        Ok(Self {
            columns: new_columns,
            column_order: self.column_order.clone(),
            nrows: new_nrows,
            display_engine: self.display_engine.clone(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: self.streaming_config.clone(),
            source_id: self.source_id.clone(),
            version: self.version + 1,
        })
    }
}

/// Foundation display methods - ONLY implemented here, all other types delegate
impl BaseTable {
    /// Convert BaseTable data to unified DataWindow format
    fn to_data_window(&self, config: &DisplayConfig) -> DisplayDataWindow {
        // Check if we should use ellipses truncation pattern
        if (self.nrows > config.max_rows && config.max_rows > 6) || 
           (self.column_order.len() > config.max_cols && config.max_cols > 6) {
            // Use ellipses pattern for both dense matrices and large tables
            return self.create_ellipses_data_window(config);
        }
        
        // Standard truncation for small tables  
        let max_rows = config.max_rows.min(self.nrows);
        let max_cols = config.max_cols.min(self.column_order.len());
        
        // Extract headers (limited by max_cols)
        let headers: Vec<String> = self.column_order.iter()
            .take(max_cols)
            .cloned()
            .collect();
        
        // Extract rows data
        let mut rows = Vec::with_capacity(max_rows);
        for row_idx in 0..max_rows {
            let mut row = Vec::with_capacity(headers.len());
            for col_name in &headers {
                if let Some(column) = self.columns.get(col_name) {
                    let value_str = if let Some(value) = column.get(row_idx) {
                        value.to_string()
                    } else {
                        "".to_string()
                    };
                    row.push(value_str);
                } else {
                    row.push("ERROR".to_string());
                }
            }
            rows.push(row);
        }
        
        // Create schema with inferred data types
        let mut columns_schema = Vec::with_capacity(headers.len());
        for col_name in &headers {
            let data_type = if let Some(column) = self.columns.get(col_name) {
                // Infer data type from first non-null value
                self.infer_column_data_type(column)
            } else {
                DataType::Unknown
            };
            
            columns_schema.push(ColumnSchema {
                name: col_name.clone(),
                data_type,
            });
        }
        
        let schema = DisplayDataSchema::new(columns_schema);
        
        // Create DataWindow with full dataset info
        DisplayDataWindow::with_window_info(
            headers,
            rows,
            schema,
            self.nrows,          // total_rows
            self.column_order.len(), // total_cols
            0                    // start_offset
        )
    }
    
    /// Create data window with ellipses pattern for large tables and matrices
    fn create_ellipses_data_window(&self, config: &DisplayConfig) -> DisplayDataWindow {
        // Calculate how many rows/cols to show at beginning and end
        let display_rows = config.max_rows;
        let display_cols = config.max_cols;
        
        // Show roughly equal parts at start and end, with ellipses in middle
        let rows_per_side = (display_rows - 1) / 2; // -1 for ellipses row
        let cols_per_side = (display_cols - 1) / 2; // -1 for ellipses col
        
        // Extract column names for display (first N, ellipsis marker, last N)
        let mut display_headers = Vec::new();
        let total_cols = self.column_order.len();
        
        // First columns
        for i in 0..cols_per_side.min(total_cols) {
            display_headers.push(self.column_order[i].clone());
        }
        
        // Ellipsis column marker (if needed)
        if total_cols > display_cols {
            display_headers.push(if config.show_headers { "⋯ (cols)" } else { "⋯" }.to_string());
        }
        
        // Last columns (only if we have more than can fit in first part)
        if total_cols > cols_per_side {
            let start_last = (total_cols - cols_per_side).max(cols_per_side + 1);
            for i in start_last..total_cols {
                display_headers.push(self.column_order[i].clone());
            }
        }
        
        // Extract rows data with ellipses pattern
        let mut display_rows_data = Vec::new();
        let total_rows = self.nrows;
        
        // Helper closure to create a row of data
        let create_row = |row_idx: usize| -> Vec<String> {
            let mut row = Vec::new();
            
            // First columns
            for i in 0..cols_per_side.min(total_cols) {
                let col_name = &self.column_order[i];
                let value_str = if let Some(column) = self.columns.get(col_name) {
                    if let Some(value) = column.get(row_idx) {
                        value.to_string()
                    } else {
                        "0.00".to_string()
                    }
                } else {
                    "ERROR".to_string()
                };
                row.push(value_str);
            }
            
            // Ellipsis column (if needed)
            if total_cols > display_cols {
                row.push("⋯".to_string());
            }
            
            // Last columns
            if total_cols > cols_per_side {
                let start_last = (total_cols - cols_per_side).max(cols_per_side + 1);
                for i in start_last..total_cols {
                    let col_name = &self.column_order[i];
                    let value_str = if let Some(column) = self.columns.get(col_name) {
                        if let Some(value) = column.get(row_idx) {
                            value.to_string()
                        } else {
                            "0.00".to_string()
                        }
                    } else {
                        "ERROR".to_string()
                    };
                    row.push(value_str);
                }
            }
            
            row
        };
        
        // First rows
        for row_idx in 0..rows_per_side.min(total_rows) {
            display_rows_data.push(create_row(row_idx));
        }
        
        // Ellipsis row (if we have truncation)
        if total_rows > display_rows {
            let ellipsis_row: Vec<String> = display_headers.iter()
                .map(|_| "⋮".to_string())
                .collect();
            display_rows_data.push(ellipsis_row);
        }
        
        // Last rows
        if total_rows > rows_per_side {
            let start_last = (total_rows - rows_per_side).max(rows_per_side + 1);
            for row_idx in start_last..total_rows {
                display_rows_data.push(create_row(row_idx));
            }
        }
        
        // Create schema with proper data type inference
        let columns_schema: Vec<ColumnSchema> = display_headers.iter()
            .map(|name| ColumnSchema {
                name: name.clone(),
                data_type: if name == "⋯" || name.contains("⋮") || name.contains("(cols)") {
                    DataType::String
                } else if let Some(column) = self.columns.get(name) {
                    // Infer actual data type from column
                    self.infer_column_data_type(column)
                } else {
                    DataType::Unknown
                },
            })
            .collect();
        
        let schema = DisplayDataSchema::new(columns_schema);
        
        // Create DataWindow with full dataset info
        DisplayDataWindow::with_window_info(
            display_headers,
            display_rows_data,
            schema,
            self.nrows,              // total_rows
            self.column_order.len(), // total_cols
            0                        // start_offset
        )
    }
    
    /// Infer data type from AttrValue column
    fn infer_column_data_type(&self, column: &BaseArray<AttrValue>) -> DataType {
        // Look at first few non-null values to infer type
        for i in 0..std::cmp::min(10, column.len()) {
            if let Some(value) = column.get(i) {
                match value {
                    crate::types::AttrValue::Int(_) => return DataType::Integer,
                    crate::types::AttrValue::SmallInt(_) => return DataType::Integer,
                    crate::types::AttrValue::Float(_) => return DataType::Float,
                    crate::types::AttrValue::Bool(_) => return DataType::Boolean,
                    crate::types::AttrValue::Text(_) => return DataType::String,
                    crate::types::AttrValue::CompactText(_) => return DataType::String,
                    crate::types::AttrValue::Bytes(_) => return DataType::String,
                    crate::types::AttrValue::Null => continue,
                    _ => return DataType::Unknown,
                }
            }
        }
        DataType::Unknown
    }
    
    /// Main display method using new unified system (__repr__ equivalent)
    pub fn __repr__(&self) -> String {
        let data_window = self.to_data_window(&self.display_engine.config);
        self.display_engine.format_unicode(&data_window)
    }
    
    /// HTML display method (_repr_html_ equivalent) 
    pub fn _repr_html_(&self) -> String {
        let data_window = self.to_data_window(&self.display_engine.config);
        self.display_engine.format_html(&data_window)
    }
    
    /// Rich display with configurable output format
    pub fn rich_display(&self, config: Option<DisplayConfig>) -> String {
        let config = config.unwrap_or(self.display_engine.config.clone());
        let data_window = self.to_data_window(&config);
        
        match config.output_format {
            OutputFormat::Unicode => {
                let mut engine = self.display_engine.clone();
                engine.set_config(config);
                engine.format_unicode(&data_window)
            },
            OutputFormat::Html => {
                let mut engine = self.display_engine.clone();
                engine.set_config(config);
                engine.format_html(&data_window)
            },
            OutputFormat::Interactive => {
                let mut engine = self.display_engine.clone();
                engine.set_config(config);
                engine.rich_display(&data_window, OutputFormat::Interactive)
            },
        }
    }
    
    /// Interactive display method (placeholder for Phase 3)
    pub fn interactive_display(&self, config: Option<DisplayConfig>) -> String {
        let config = config.unwrap_or_else(|| DisplayConfig::interactive());
        self.rich_display(Some(config))
    }
    
    /// Get display configuration
    pub fn get_display_config(&self) -> &DisplayConfig {
        &self.display_engine.config
    }
    
    /// Set display configuration
    pub fn set_display_config(&mut self, config: DisplayConfig) {
        self.display_engine.set_config(config);
    }
}

/// Standard Display trait implementation using new unified system
impl std::fmt::Display for BaseTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Use our new unified display system
        write!(f, "{}", self.__repr__())
    }
}

// Additional BaseTable methods (not part of trait)
impl BaseTable {
    /// Evaluate a predicate string and return a boolean mask
    pub fn evaluate_predicate(&self, predicate: &str) -> GraphResult<Vec<bool>> {
        // Support various operators: ==, !=, <, <=, >, >=
        let operators = [">=", "<=", "!=", "==", ">", "<"];
        
        let mut column_name = "";
        let mut operator = "";
        let mut value_str = "";
        
        // Find the operator
        for op in &operators {
            if let Some(pos) = predicate.find(op) {
                column_name = predicate[..pos].trim();
                operator = op;
                value_str = predicate[pos + op.len()..].trim().trim_matches('"').trim_matches('\'');
                break;
            }
        }
        
        if operator.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("Unsupported predicate format: '{}'. Use operators: ==, !=, <, <=, >, >=", predicate)
            ));
        }
        
        let filter_column = self.column(column_name)
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                format!("Column '{}' not found", column_name)
            ))?;
        
        // Generate mask based on operator and value type
        let mut mask = Vec::with_capacity(self.nrows);
        
        for i in 0..self.nrows {
            let row_value = filter_column.get(i);
            let matches = match row_value {
                Some(attr_val) => Self::compare_attr_value(attr_val, operator, value_str)?,
                None => false, // Null values don't match any comparison
            };
            mask.push(matches);
        }
        
        Ok(mask)
    }
    
    /// Compare an AttrValue with a string value using the given operator
    pub fn compare_attr_value(attr_val: &crate::types::AttrValue, operator: &str, value_str: &str) -> GraphResult<bool> {
        use crate::types::AttrValue;
        
        let result = match attr_val {
            AttrValue::Int(i) => {
                if let Ok(val) = value_str.parse::<i64>() {
                    match operator {
                        "==" => *i == val,
                        "!=" => *i != val,
                        "<" => *i < val,
                        "<=" => *i <= val,
                        ">" => *i > val,
                        ">=" => *i >= val,
                        _ => false,
                    }
                } else {
                    false
                }
            },
            AttrValue::SmallInt(i) => {
                if let Ok(val) = value_str.parse::<i32>() {
                    let i_val = *i as i64;
                    let val_i64 = val as i64;
                    match operator {
                        "==" => i_val == val_i64,
                        "!=" => i_val != val_i64,
                        "<" => i_val < val_i64,
                        "<=" => i_val <= val_i64,
                        ">" => i_val > val_i64,
                        ">=" => i_val >= val_i64,
                        _ => false,
                    }
                } else {
                    false
                }
            },
            AttrValue::Float(f) => {
                if let Ok(val) = value_str.parse::<f32>() {
                    match operator {
                        "==" => (*f - val).abs() < f32::EPSILON,
                        "!=" => (*f - val).abs() >= f32::EPSILON,
                        "<" => *f < val,
                        "<=" => *f <= val,
                        ">" => *f > val,
                        ">=" => *f >= val,
                        _ => false,
                    }
                } else {
                    false
                }
            },
            AttrValue::Bool(b) => {
                if let Ok(val) = value_str.parse::<bool>() {
                    match operator {
                        "==" => *b == val,
                        "!=" => *b != val,
                        _ => false, // Other operators don't make sense for bools
                    }
                } else {
                    // Support string representations
                    let val_lower = value_str.to_lowercase();
                    let val_bool = val_lower == "true" || val_lower == "1";
                    match operator {
                        "==" => *b == val_bool,
                        "!=" => *b != val_bool,
                        _ => false,
                    }
                }
            },
            AttrValue::Text(s) => {
                match operator {
                    "==" => s == value_str,
                    "!=" => s != value_str,
                    "<" => s.as_str() < value_str,
                    "<=" => s.as_str() <= value_str,
                    ">" => s.as_str() > value_str,
                    ">=" => s.as_str() >= value_str,
                    _ => false,
                }
            },
            AttrValue::CompactText(s) => {
                let s_str = s.as_str();
                match operator {
                    "==" => s_str == value_str,
                    "!=" => s_str != value_str,
                    "<" => s_str < value_str,
                    "<=" => s_str <= value_str,
                    ">" => s_str > value_str,
                    ">=" => s_str >= value_str,
                    _ => false,
                }
            },
            AttrValue::Null => false, // Null values don't match any comparison
            _ => {
                // For other types, convert to string and compare
                let attr_str = format!("{}", attr_val);
                match operator {
                    "==" => attr_str == value_str,
                    "!=" => attr_str != value_str,
                    _ => false, // Other operators not supported for complex types
                }
            },
        };
        
        Ok(result)
    }
}

// File I/O Implementation
impl BaseTable {
    /// Export table to CSV file
    pub fn to_csv<P: AsRef<std::path::Path>>(&self, path: P) -> GraphResult<()> {
        use std::io::Write;
        
        let path = path.as_ref();
        let mut writer = csv::Writer::from_path(path)
            .map_err(|e| crate::errors::GraphError::InvalidInput(
                format!("Failed to create CSV writer: {}", e)
            ))?;
            
        // Write headers
        writer.write_record(&self.column_order)
            .map_err(|e| crate::errors::GraphError::InvalidInput(
                format!("Failed to write CSV headers: {}", e)
            ))?;
            
        // Write data rows
        for i in 0..self.nrows {
            let mut record = Vec::new();
            for col_name in &self.column_order {
                if let Some(column) = self.columns.get(col_name) {
                    let value_str = if let Some(value) = column.get(i) {
                        self.attr_value_to_csv_string(value)
                    } else {
                        String::new() // Empty string for null values
                    };
                    record.push(value_str);
                } else {
                    record.push(String::new());
                }
            }
            writer.write_record(&record)
                .map_err(|e| crate::errors::GraphError::InvalidInput(
                    format!("Failed to write CSV record at row {}: {}", i, e)
                ))?;
        }
        
        writer.flush()
            .map_err(|e| crate::errors::GraphError::InvalidInput(
                format!("Failed to flush CSV writer: {}", e)
            ))?;
            
        Ok(())
    }
    
    /// Import table from CSV file  
    pub fn from_csv<P: AsRef<std::path::Path>>(path: P) -> GraphResult<Self> {
        let path = path.as_ref();
        let mut reader = csv::Reader::from_path(path)
            .map_err(|e| crate::errors::GraphError::InvalidInput(
                format!("Failed to read CSV file '{}': {}", path.display(), e)
            ))?;
            
        // Get headers
        let headers = reader.headers()
            .map_err(|e| crate::errors::GraphError::InvalidInput(
                format!("Failed to read CSV headers: {}", e)
            ))?;
        let column_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
        
        // Initialize column data vectors
        let mut column_data: std::collections::HashMap<String, Vec<crate::types::AttrValue>> = 
            column_names.iter().map(|name| (name.clone(), Vec::new())).collect();
            
        // Read all records
        for (row_idx, result) in reader.records().enumerate() {
            let record = result
                .map_err(|e| crate::errors::GraphError::InvalidInput(
                    format!("Failed to read CSV record at row {}: {}", row_idx + 1, e)
                ))?;
                
            // Process each field in the record
            for (col_idx, field) in record.iter().enumerate() {
                if let Some(col_name) = column_names.get(col_idx) {
                    let attr_value = Self::parse_csv_field(field);
                    if let Some(col_vec) = column_data.get_mut(col_name) {
                        col_vec.push(attr_value);
                    }
                }
            }
        }
        
        // Convert to BaseArrays
        let mut columns = std::collections::HashMap::new();
        for (name, data) in column_data {
            columns.insert(name, crate::storage::array::BaseArray::from_attr_values(data));
        }
        
        Self::with_column_order(columns, column_names)
    }
    
    /// Export table to Parquet file 
    pub fn to_parquet<P: AsRef<std::path::Path>>(&self, path: P) -> GraphResult<()> {
        // For now, save as JSON instead of Parquet for simplicity 
        // TODO: Implement proper Parquet support
        self.to_json(path)
    }
    
    /// Import table from Parquet file
    pub fn from_parquet<P: AsRef<std::path::Path>>(path: P) -> GraphResult<Self> {
        // For now, load from JSON instead of Parquet for simplicity
        // TODO: Implement proper Parquet support
        Self::from_json(path)
    }
    
    /// Export table to JSON file
    pub fn to_json<P: AsRef<std::path::Path>>(&self, path: P) -> GraphResult<()> {
        use std::fs::File;
        use std::io::Write;
        
        let path = path.as_ref();
        let mut file = File::create(path)
            .map_err(|e| crate::errors::GraphError::InvalidInput(
                format!("Failed to create JSON file '{}': {}", path.display(), e)
            ))?;
            
        // Create JSON structure with metadata
        let mut json_data = serde_json::Map::new();
        
        // Add metadata
        json_data.insert("columns".to_string(), serde_json::Value::Array(
            self.column_order.iter().map(|s| serde_json::Value::String(s.clone())).collect()
        ));
        json_data.insert("nrows".to_string(), serde_json::Value::Number(serde_json::Number::from(self.nrows)));
        
        // Add data rows
        let mut rows = Vec::new();
        for i in 0..self.nrows {
            let mut row = serde_json::Map::new();
            for col_name in &self.column_order {
                if let Some(column) = self.columns.get(col_name) {
                    let value = if let Some(attr_value) = column.get(i) {
                        self.attr_value_to_json(attr_value)
                    } else {
                        serde_json::Value::Null
                    };
                    row.insert(col_name.clone(), value);
                }
            }
            rows.push(serde_json::Value::Object(row));
        }
        json_data.insert("data".to_string(), serde_json::Value::Array(rows));
        
        // Write to file
        let json_string = serde_json::to_string_pretty(&json_data)
            .map_err(|e| crate::errors::GraphError::InvalidInput(
                format!("Failed to serialize to JSON: {}", e)
            ))?;
            
        file.write_all(json_string.as_bytes())
            .map_err(|e| crate::errors::GraphError::InvalidInput(
                format!("Failed to write JSON file: {}", e)
            ))?;
            
        Ok(())
    }
    
    /// Import table from JSON file
    pub fn from_json<P: AsRef<std::path::Path>>(path: P) -> GraphResult<Self> {
        use std::fs::File;
        use std::io::Read;
        
        let path = path.as_ref();
        let mut file = File::open(path)
            .map_err(|e| crate::errors::GraphError::InvalidInput(
                format!("Failed to open JSON file '{}': {}", path.display(), e)
            ))?;
            
        let mut json_string = String::new();
        file.read_to_string(&mut json_string)
            .map_err(|e| crate::errors::GraphError::InvalidInput(
                format!("Failed to read JSON file: {}", e)
            ))?;
            
        let json_data: serde_json::Value = serde_json::from_str(&json_string)
            .map_err(|e| crate::errors::GraphError::InvalidInput(
                format!("Failed to parse JSON: {}", e)
            ))?;
            
        // Extract metadata
        let json_obj = json_data.as_object()
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                "JSON data must be an object".to_string()
            ))?;
            
        let column_names: Vec<String> = json_obj.get("columns")
            .and_then(|v| v.as_array())
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                "Missing or invalid 'columns' field in JSON".to_string()
            ))?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();
            
        let rows = json_obj.get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                "Missing or invalid 'data' field in JSON".to_string()
            ))?;
            
        // Initialize column data
        let mut column_data: std::collections::HashMap<String, Vec<crate::types::AttrValue>> = 
            column_names.iter().map(|name| (name.clone(), Vec::new())).collect();
            
        // Process each row
        for row_value in rows {
            let row_obj = row_value.as_object()
                .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                    "Each data row must be an object".to_string()
                ))?;
                
            for col_name in &column_names {
                let attr_value = if let Some(value) = row_obj.get(col_name) {
                    Self::json_value_to_attr_value(value)
                } else {
                    crate::types::AttrValue::Null
                };
                
                if let Some(col_vec) = column_data.get_mut(col_name) {
                    col_vec.push(attr_value);
                }
            }
        }
        
        // Convert to BaseArrays
        let mut columns = std::collections::HashMap::new();
        for (name, data) in column_data {
            columns.insert(name, crate::storage::array::BaseArray::from_attr_values(data));
        }
        
        Self::with_column_order(columns, column_names)
    }
    
    /// Helper method to convert AttrValue to CSV string representation
    fn attr_value_to_csv_string(&self, value: &crate::types::AttrValue) -> String {
        match value {
            crate::types::AttrValue::Int(i) => i.to_string(),
            crate::types::AttrValue::SmallInt(i) => i.to_string(),
            crate::types::AttrValue::Float(f) => f.to_string(),
            crate::types::AttrValue::Text(s) => {
                // Escape quotes by doubling them, and wrap in quotes if contains comma/quote/newline
                if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
                    format!("\"{}\"", s.replace("\"", "\"\""))
                } else {
                    s.clone()
                }
            },
            crate::types::AttrValue::CompactText(s) => {
                let text = s.as_str();
                if text.contains(',') || text.contains('"') || text.contains('\n') || text.contains('\r') {
                    format!("\"{}\"", text.replace("\"", "\"\""))
                } else {
                    text.to_string()
                }
            },
            crate::types::AttrValue::Bool(b) => b.to_string(),
            crate::types::AttrValue::Null => String::new(),
            crate::types::AttrValue::FloatVec(v) => format!("[{}]", v.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(",")),
            crate::types::AttrValue::Bytes(b) => format!("bytes:{}", b.len()),
            _ => format!("{:?}", value), // Fallback for other types
        }
    }
    
    /// Helper method to parse CSV field to AttrValue
    fn parse_csv_field(field: &str) -> crate::types::AttrValue {
        if field.is_empty() {
            return crate::types::AttrValue::Null;
        }
        
        // Try parsing as different types
        // 1. Boolean
        match field.to_lowercase().as_str() {
            "true" => return crate::types::AttrValue::Bool(true),
            "false" => return crate::types::AttrValue::Bool(false),
            _ => {}
        }
        
        // 2. Integer
        if let Ok(i) = field.parse::<i64>() {
            return crate::types::AttrValue::Int(i);
        }
        
        // 3. Float
        if let Ok(f) = field.parse::<f32>() {
            return crate::types::AttrValue::Float(f);
        }
        
        // 4. Default to text
        crate::types::AttrValue::Text(field.to_string())
    }
    
    /// Helper method to convert AttrValue to JSON value
    fn attr_value_to_json(&self, value: &crate::types::AttrValue) -> serde_json::Value {
        match value {
            crate::types::AttrValue::Int(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
            crate::types::AttrValue::SmallInt(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
            crate::types::AttrValue::Float(f) => {
                serde_json::Value::Number(serde_json::Number::from_f64(*f as f64).unwrap_or(serde_json::Number::from(0)))
            },
            crate::types::AttrValue::Text(s) => serde_json::Value::String(s.clone()),
            crate::types::AttrValue::CompactText(s) => serde_json::Value::String(s.as_str().to_string()),
            crate::types::AttrValue::Bool(b) => serde_json::Value::Bool(*b),
            crate::types::AttrValue::Null => serde_json::Value::Null,
            crate::types::AttrValue::FloatVec(v) => {
                serde_json::Value::Array(v.iter().map(|f| serde_json::Value::Number(
                    serde_json::Number::from_f64(*f as f64).unwrap_or(serde_json::Number::from(0))
                )).collect())
            },
            _ => serde_json::Value::String(format!("{:?}", value)), // Fallback
        }
    }
    
    /// Helper method to convert JSON value to AttrValue  
    fn json_value_to_attr_value(value: &serde_json::Value) -> crate::types::AttrValue {
        match value {
            serde_json::Value::Null => crate::types::AttrValue::Null,
            serde_json::Value::Bool(b) => crate::types::AttrValue::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    crate::types::AttrValue::Int(i)
                } else if let Some(f) = n.as_f64() {
                    crate::types::AttrValue::Float(f as f32)
                } else {
                    crate::types::AttrValue::Null
                }
            },
            serde_json::Value::String(s) => crate::types::AttrValue::Text(s.clone()),
            serde_json::Value::Array(arr) => {
                // Try to convert to float vector
                let floats: Result<Vec<f32>, _> = arr.iter().map(|v| {
                    v.as_f64().map(|f| f as f32).ok_or("Not a number")
                }).collect();
                
                if let Ok(float_vec) = floats {
                    crate::types::AttrValue::FloatVec(float_vec)
                } else {
                    // Fallback to string representation
                    crate::types::AttrValue::Text(format!("{:?}", arr))
                }
            },
            serde_json::Value::Object(_) => {
                // Convert object to string representation
                crate::types::AttrValue::Text(value.to_string())
            }
        }
    }

    /// Group by columns and apply aggregations
    pub fn group_by_agg(&self, group_cols: &[String], agg_specs: HashMap<String, String>) -> GraphResult<Self> {
        if group_cols.is_empty() {
            return self.aggregate(agg_specs);
        }

        // Validate group columns exist
        for col_name in group_cols {
            if !self.columns.contains_key(col_name) {
                return Err(crate::errors::GraphError::InvalidInput(
                    format!("Group column '{}' not found in table", col_name)
                ));
            }
        }

        // Create groups
        let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        
        for row_idx in 0..self.nrows {
            let mut key = Vec::new();
            for col_name in group_cols {
                if let Some(column) = self.columns.get(col_name) {
                    if let Some(value) = column.get(row_idx) {
                        key.push(format!("{:?}", value));
                    } else {
                        key.push("NULL".to_string());
                    }
                } else {
                    key.push("NULL".to_string());
                }
            }
            groups.entry(key).or_insert_with(Vec::new).push(row_idx);
        }

        // Build result table
        let mut result_columns: HashMap<String, Vec<crate::types::AttrValue>> = HashMap::new();
        let mut column_order = group_cols.to_vec();

        // Initialize group columns
        for col_name in group_cols {
            result_columns.insert(col_name.clone(), Vec::new());
        }

        // Initialize aggregation columns
        for (col_name, _agg_func) in &agg_specs {
            column_order.push(col_name.clone());
            result_columns.insert(col_name.clone(), Vec::new());
        }

        // Process each group
        for (group_key, row_indices) in groups {
            // Add group key values
            for (i, col_name) in group_cols.iter().enumerate() {
                if let Some(column) = result_columns.get_mut(col_name) {
                    let key_value = &group_key[i];
                    if key_value == "NULL" {
                        column.push(crate::types::AttrValue::Null);
                    } else {
                        // Parse key value back - simplified for now
                        if let Ok(int_val) = key_value.parse::<i64>() {
                            column.push(crate::types::AttrValue::Int(int_val));
                        } else if let Ok(float_val) = key_value.parse::<f64>() {
                            column.push(crate::types::AttrValue::Float(float_val as f32));
                        } else {
                            // Remove quotes from debug format
                            let clean_str = key_value.trim_matches('"');
                            column.push(crate::types::AttrValue::Text(clean_str.to_string()));
                        }
                    }
                }
            }

            // Apply aggregations
            for (agg_col, agg_func) in &agg_specs {
                let agg_result = self.apply_aggregation(&row_indices, agg_col, agg_func)?;
                if let Some(column) = result_columns.get_mut(agg_col) {
                    column.push(agg_result);
                }
            }
        }

        // Convert Vec<AttrValue> to BaseArray for each column
        let mut base_columns = HashMap::new();
        for (name, values) in result_columns {
            base_columns.insert(name, BaseArray::from_attr_values(values));
        }

        // Create result table
        let mut result = BaseTable::from_columns(base_columns)?;
        result.column_order = column_order;
        Ok(result)
    }

    /// Apply aggregation to a subset of rows
    fn apply_aggregation(&self, row_indices: &[usize], col_name: &str, agg_func: &str) -> GraphResult<crate::types::AttrValue> {
        let column = self.columns.get(col_name).ok_or_else(|| crate::errors::GraphError::InvalidInput(
            format!("Column '{}' not found for aggregation", col_name)
        ))?;

        match agg_func.to_lowercase().as_str() {
            "count" => Ok(crate::types::AttrValue::Int(row_indices.len() as i64)),
            "sum" => {
                let mut sum = 0.0;
                let mut count = 0;
                for &idx in row_indices {
                    if let Some(value) = column.get(idx) {
                        match value {
                            crate::types::AttrValue::Int(i) => {
                                sum += *i as f64;
                                count += 1;
                            }
                            crate::types::AttrValue::Float(f) => {
                                sum += *f as f64;
                                count += 1;
                            }
                            _ => {} // Skip non-numeric values
                        }
                    }
                }
                if count == 0 {
                    Ok(crate::types::AttrValue::Null)
                } else if sum.fract() == 0.0 && sum.abs() <= i64::MAX as f64 {
                    Ok(crate::types::AttrValue::Int(sum as i64))
                } else {
                    Ok(crate::types::AttrValue::Float(sum as f32))
                }
            }
            "avg" | "mean" => {
                let mut sum = 0.0;
                let mut count = 0;
                for &idx in row_indices {
                    if let Some(value) = column.get(idx) {
                        match value {
                            crate::types::AttrValue::Int(i) => {
                                sum += *i as f64;
                                count += 1;
                            }
                            crate::types::AttrValue::Float(f) => {
                                // Skip NaN values (common with meta nodes)
                                if !f.is_nan() {
                                    sum += *f as f64;
                                    count += 1;
                                }
                            }
                            _ => {} // Skip non-numeric values and nulls
                        }
                    }
                }
                if count == 0 {
                    Ok(crate::types::AttrValue::Null)
                } else {
                    Ok(crate::types::AttrValue::Float((sum / count as f64) as f32))
                }
            }
            "min" => {
                let mut min_val: Option<crate::types::AttrValue> = None;
                for &idx in row_indices {
                    if let Some(value) = column.get(idx) {
                        match value {
                            crate::types::AttrValue::Int(_) => {
                                if min_val.is_none() || self.compare_values(value, min_val.as_ref().unwrap()) < 0 {
                                    min_val = Some(value.clone());
                                }
                            }
                            crate::types::AttrValue::Float(f) => {
                                // Skip NaN values (common with meta nodes)
                                if !f.is_nan() {
                                    if min_val.is_none() || self.compare_values(value, min_val.as_ref().unwrap()) < 0 {
                                        min_val = Some(value.clone());
                                    }
                                }
                            }
                            _ => {} // Skip non-numeric values and nulls for min/max
                        }
                    }
                }
                Ok(min_val.unwrap_or(crate::types::AttrValue::Null))
            }
            "max" => {
                let mut max_val: Option<crate::types::AttrValue> = None;
                for &idx in row_indices {
                    if let Some(value) = column.get(idx) {
                        match value {
                            crate::types::AttrValue::Int(_) => {
                                if max_val.is_none() || self.compare_values(value, max_val.as_ref().unwrap()) > 0 {
                                    max_val = Some(value.clone());
                                }
                            }
                            crate::types::AttrValue::Float(f) => {
                                // Skip NaN values (common with meta nodes)
                                if !f.is_nan() {
                                    if max_val.is_none() || self.compare_values(value, max_val.as_ref().unwrap()) > 0 {
                                        max_val = Some(value.clone());
                                    }
                                }
                            }
                            _ => {} // Skip non-numeric values and nulls for min/max
                        }
                    }
                }
                Ok(max_val.unwrap_or(crate::types::AttrValue::Null))
            }
            _ => Err(crate::errors::GraphError::InvalidInput(
                format!("Unknown aggregation function: {}", agg_func)
            ))
        }
    }

    /// Aggregate entire table (no grouping)
    pub fn aggregate(&self, agg_specs: HashMap<String, String>) -> GraphResult<Self> {
        if agg_specs.is_empty() {
            return Ok(BaseTable::new());
        }

        let mut result_columns = HashMap::new();
        let mut column_order = Vec::new();

        for (col_name, agg_func) in &agg_specs {
            column_order.push(col_name.clone());
            let row_indices: Vec<usize> = (0..self.nrows).collect();
            let agg_result = self.apply_aggregation(&row_indices, col_name, agg_func)?;
            result_columns.insert(col_name.clone(), BaseArray::from_attr_values(vec![agg_result]));
        }

        let mut result = BaseTable::from_columns(result_columns)?;
        result.column_order = column_order;
        Ok(result)
    }

    /// Helper method to compare AttrValues for min/max operations
    fn compare_values(&self, a: &crate::types::AttrValue, b: &crate::types::AttrValue) -> i32 {
        use crate::types::AttrValue;
        match (a, b) {
            (AttrValue::Int(a), AttrValue::Int(b)) => a.cmp(b) as i32,
            (AttrValue::Float(a), AttrValue::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) as i32,
            (AttrValue::Int(a), AttrValue::Float(b)) => (*a as f32).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) as i32,
            (AttrValue::Float(a), AttrValue::Int(b)) => a.partial_cmp(&(*b as f32)).unwrap_or(std::cmp::Ordering::Equal) as i32,
            _ => 0, // Equal for non-comparable types
        }
    }

    /// Select specific rows by indices
    pub fn select_rows(&self, row_indices: &[usize]) -> GraphResult<Self> {
        let mut new_columns = HashMap::new();
        
        for (col_name, column) in &self.columns {
            let mut new_values = Vec::new();
            for &row_idx in row_indices {
                if row_idx < self.nrows {
                    if let Some(value) = column.get(row_idx) {
                        new_values.push(value.clone());
                    } else {
                        new_values.push(crate::types::AttrValue::Null);
                    }
                } else {
                    new_values.push(crate::types::AttrValue::Null);
                }
            }
            new_columns.insert(col_name.clone(), BaseArray::from_attr_values(new_values));
        }

        Ok(Self {
            columns: new_columns,
            column_order: self.column_order.clone(),
            nrows: row_indices.len(),
            display_engine: self.display_engine.clone(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: self.streaming_config.clone(),
            source_id: self.source_id.clone(),
            version: self.version + 1,
        })
    }

    /// Inner join with another table on specified columns
    pub fn inner_join(&self, other: &Self, left_on: &str, right_on: &str) -> GraphResult<Self> {
        // Validate join columns exist
        if !self.has_column(left_on) {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("Left join column '{}' not found in table", left_on)
            ));
        }
        if !other.has_column(right_on) {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("Right join column '{}' not found in other table", right_on)
            ));
        }

        let left_col = self.column(left_on).unwrap();
        let right_col = other.column(right_on).unwrap();

        // Build index for right table
        let mut right_index: HashMap<String, Vec<usize>> = HashMap::new();
        for i in 0..other.nrows {
            if let Some(value) = right_col.get(i) {
                let key = format!("{:?}", value);
                right_index.entry(key).or_insert_with(Vec::new).push(i);
            }
        }

        // Find matching rows
        let mut result_rows = Vec::new();
        for left_idx in 0..self.nrows {
            if let Some(left_value) = left_col.get(left_idx) {
                let key = format!("{:?}", left_value);
                if let Some(right_indices) = right_index.get(&key) {
                    for &right_idx in right_indices {
                        result_rows.push((left_idx, right_idx));
                    }
                }
            }
        }

        // Build result table
        let mut result_columns = HashMap::new();
        let mut column_order = Vec::new();

        // Add left table columns (with left_ prefix for duplicates)
        for col_name in &self.column_order {
            let final_name = if other.has_column(col_name) && col_name != left_on {
                format!("left_{}", col_name)
            } else {
                col_name.clone()
            };
            column_order.push(final_name.clone());
            
            let mut new_values = Vec::new();
            if let Some(column) = self.columns.get(col_name) {
                for &(left_idx, _) in &result_rows {
                    if let Some(value) = column.get(left_idx) {
                        new_values.push(value.clone());
                    } else {
                        new_values.push(crate::types::AttrValue::Null);
                    }
                }
            }
            result_columns.insert(final_name, BaseArray::from_attr_values(new_values));
        }

        // Add right table columns (with right_ prefix for duplicates, skip join column)
        for col_name in &other.column_order {
            if col_name == right_on {
                continue; // Skip join column from right table
            }
            
            let final_name = if self.has_column(col_name) {
                format!("right_{}", col_name)
            } else {
                col_name.clone()
            };
            column_order.push(final_name.clone());
            
            let mut new_values = Vec::new();
            if let Some(column) = other.columns.get(col_name) {
                for &(_, right_idx) in &result_rows {
                    if let Some(value) = column.get(right_idx) {
                        new_values.push(value.clone());
                    } else {
                        new_values.push(crate::types::AttrValue::Null);
                    }
                }
            }
            result_columns.insert(final_name, BaseArray::from_attr_values(new_values));
        }

        let mut result = BaseTable::from_columns(result_columns)?;
        result.column_order = column_order;
        Ok(result)
    }

    /// Left join with another table on specified columns
    pub fn left_join(&self, other: &Self, left_on: &str, right_on: &str) -> GraphResult<Self> {
        // Validate join columns exist
        if !self.has_column(left_on) {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("Left join column '{}' not found in table", left_on)
            ));
        }
        if !other.has_column(right_on) {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("Right join column '{}' not found in other table", right_on)
            ));
        }

        let left_col = self.column(left_on).unwrap();
        let right_col = other.column(right_on).unwrap();

        // Build index for right table
        let mut right_index: HashMap<String, Vec<usize>> = HashMap::new();
        for i in 0..other.nrows {
            if let Some(value) = right_col.get(i) {
                let key = format!("{:?}", value);
                right_index.entry(key).or_insert_with(Vec::new).push(i);
            }
        }

        // Find matching rows (include all left rows)
        let mut result_rows = Vec::new();
        for left_idx in 0..self.nrows {
            if let Some(left_value) = left_col.get(left_idx) {
                let key = format!("{:?}", left_value);
                if let Some(right_indices) = right_index.get(&key) {
                    for &right_idx in right_indices {
                        result_rows.push((left_idx, Some(right_idx)));
                    }
                } else {
                    result_rows.push((left_idx, None)); // No match in right table
                }
            } else {
                result_rows.push((left_idx, None)); // NULL left value
            }
        }

        // Build result table
        let mut result_columns = HashMap::new();
        let mut column_order = Vec::new();

        // Add left table columns
        for col_name in &self.column_order {
            let final_name = if other.has_column(col_name) && col_name != left_on {
                format!("left_{}", col_name)
            } else {
                col_name.clone()
            };
            column_order.push(final_name.clone());
            
            let mut new_values = Vec::new();
            if let Some(column) = self.columns.get(col_name) {
                for &(left_idx, _) in &result_rows {
                    if let Some(value) = column.get(left_idx) {
                        new_values.push(value.clone());
                    } else {
                        new_values.push(crate::types::AttrValue::Null);
                    }
                }
            }
            result_columns.insert(final_name, BaseArray::from_attr_values(new_values));
        }

        // Add right table columns (NULL when no match)
        for col_name in &other.column_order {
            if col_name == right_on {
                continue; // Skip join column from right table
            }
            
            let final_name = if self.has_column(col_name) {
                format!("right_{}", col_name)
            } else {
                col_name.clone()
            };
            column_order.push(final_name.clone());
            
            let mut new_values = Vec::new();
            if let Some(column) = other.columns.get(col_name) {
                for &(_, right_idx_opt) in &result_rows {
                    if let Some(right_idx) = right_idx_opt {
                        if let Some(value) = column.get(right_idx) {
                            new_values.push(value.clone());
                        } else {
                            new_values.push(crate::types::AttrValue::Null);
                        }
                    } else {
                        new_values.push(crate::types::AttrValue::Null); // No match
                    }
                }
            }
            result_columns.insert(final_name, BaseArray::from_attr_values(new_values));
        }

        let mut result = BaseTable::from_columns(result_columns)?;
        result.column_order = column_order;
        Ok(result)
    }

    /// Union with another table (combines all rows, removes duplicates)
    pub fn union(&self, other: &Self) -> GraphResult<Self> {
        // Tables must have the same schema
        if self.column_order != other.column_order {
            return Err(crate::errors::GraphError::InvalidInput(
                "Union requires tables with identical column schemas".to_string()
            ));
        }

        let mut result_columns: HashMap<String, Vec<crate::types::AttrValue>> = HashMap::new();
        for col_name in &self.column_order {
            result_columns.insert(col_name.clone(), Vec::new());
        }

        // Track unique rows
        let mut unique_rows = std::collections::HashSet::new();

        // Add rows from first table
        for row_idx in 0..self.nrows {
            let mut row_key = Vec::new();
            for col_name in &self.column_order {
                if let Some(column) = self.columns.get(col_name) {
                    if let Some(value) = column.get(row_idx) {
                        row_key.push(format!("{:?}", value));
                    } else {
                        row_key.push("NULL".to_string());
                    }
                }
            }
            
            let row_hash = row_key.join("|");
            if unique_rows.insert(row_hash) {
                // Add this unique row
                for col_name in &self.column_order {
                    if let Some(column) = self.columns.get(col_name) {
                        if let Some(values) = result_columns.get_mut(col_name) {
                            if let Some(value) = column.get(row_idx) {
                                values.push(value.clone());
                            } else {
                                values.push(crate::types::AttrValue::Null);
                            }
                        }
                    }
                }
            }
        }

        // Add rows from second table
        for row_idx in 0..other.nrows {
            let mut row_key = Vec::new();
            for col_name in &other.column_order {
                if let Some(column) = other.columns.get(col_name) {
                    if let Some(value) = column.get(row_idx) {
                        row_key.push(format!("{:?}", value));
                    } else {
                        row_key.push("NULL".to_string());
                    }
                }
            }
            
            let row_hash = row_key.join("|");
            if unique_rows.insert(row_hash) {
                // Add this unique row
                for col_name in &other.column_order {
                    if let Some(column) = other.columns.get(col_name) {
                        if let Some(values) = result_columns.get_mut(col_name) {
                            if let Some(value) = column.get(row_idx) {
                                values.push(value.clone());
                            } else {
                                values.push(crate::types::AttrValue::Null);
                            }
                        }
                    }
                }
            }
        }

        // Convert to BaseArray columns
        let mut base_columns = HashMap::new();
        for (name, values) in result_columns {
            base_columns.insert(name, BaseArray::from_attr_values(values));
        }

        let mut result = BaseTable::from_columns(base_columns)?;
        result.column_order = self.column_order.clone();
        Ok(result)
    }

    /// Stack tables vertically (concatenate rows) - similar to pandas.concat(axis=0)
    /// TODO: Implement vertical stacking/concatenation for tables
    /// This should append rows from other table to this table, handling schema differences
    pub fn stack(&self, other: &Self) -> GraphResult<Self> {
        // TODO: Allow stacking tables with different schemas by:
        // 1. Create union of all column names from both tables
        // 2. Fill missing columns with AttrValue::Null
        // 3. Concatenate all rows maintaining original order
        // 4. Return new table with combined schema and all rows
        
        // For now, require same schema like union but without deduplication
        if self.column_order != other.column_order {
            return Err(crate::errors::GraphError::InvalidInput(
                "Stack requires tables with identical column schemas (TODO: support different schemas)".to_string()
            ));
        }

        let mut result_columns: HashMap<String, Vec<crate::types::AttrValue>> = HashMap::new();
        for col_name in &self.column_order {
            result_columns.insert(col_name.clone(), Vec::new());
        }

        // Add all rows from first table (no deduplication)
        for row_idx in 0..self.nrows {
            for col_name in &self.column_order {
                if let Some(column) = self.columns.get(col_name) {
                    if let Some(values) = result_columns.get_mut(col_name) {
                        if let Some(value) = column.get(row_idx) {
                            values.push(value.clone());
                        } else {
                            values.push(crate::types::AttrValue::Null);
                        }
                    }
                }
            }
        }

        // Add all rows from second table (no deduplication)
        for row_idx in 0..other.nrows {
            for col_name in &other.column_order {
                if let Some(column) = other.columns.get(col_name) {
                    if let Some(values) = result_columns.get_mut(col_name) {
                        if let Some(value) = column.get(row_idx) {
                            values.push(value.clone());
                        } else {
                            values.push(crate::types::AttrValue::Null);
                        }
                    }
                }
            }
        }

        // Convert to BaseArray columns
        let mut base_columns = HashMap::new();
        for (name, values) in result_columns {
            base_columns.insert(name, BaseArray::from_attr_values(values));
        }

        let mut result = BaseTable::from_columns(base_columns)?;
        result.column_order = self.column_order.clone();
        Ok(result)
    }

    /// Concatenate tables horizontally (add columns) - similar to pandas.concat(axis=1)
    /// TODO: Implement horizontal concatenation for tables
    pub fn concat(&self, other: &Self) -> GraphResult<Self> {
        // TODO: Implement horizontal concatenation by:
        // 1. Combine column sets from both tables
        // 2. Handle column name conflicts (add suffix like _x, _y)
        // 3. Align rows by index (pad shorter table with nulls)
        // 4. Return new table with combined columns
        
        // Basic implementation for now
        let mut result_columns = self.columns.clone();
        let target_rows = std::cmp::max(self.nrows, other.nrows);
        
        // Add columns from other table, handling conflicts
        for (col_name, column) in &other.columns {
            let final_col_name = if result_columns.contains_key(col_name) {
                format!("{}_y", col_name) // TODO: Better conflict resolution
            } else {
                col_name.clone()
            };
            
            // Pad column to match target row count
            let mut values = column.data().clone();
            while values.len() < target_rows {
                values.push(crate::types::AttrValue::Null);
            }
            
            result_columns.insert(final_col_name.clone(), BaseArray::from_attr_values(values));
        }
        
        // Pad existing columns to match target row count
        for column in result_columns.values_mut() {
            let mut values = column.data().clone();
            while values.len() < target_rows {
                values.push(crate::types::AttrValue::Null);
            }
            *column = BaseArray::from_attr_values(values);
        }

        let mut result = BaseTable::from_columns(result_columns)?;
        // TODO: Proper column order management for concatenated tables
        result.column_order.extend(other.column_order.iter().map(|name| {
            if self.column_order.contains(name) {
                format!("{}_y", name)
            } else {
                name.clone()
            }
        }));
        Ok(result)
    }

    /// Intersect with another table (returns only common rows)
    pub fn intersect(&self, other: &Self) -> GraphResult<Self> {
        // Tables must have the same schema
        if self.column_order != other.column_order {
            return Err(crate::errors::GraphError::InvalidInput(
                "Intersect requires tables with identical column schemas".to_string()
            ));
        }

        // Build set of rows from the other table
        let mut other_rows = std::collections::HashSet::new();
        for row_idx in 0..other.nrows {
            let mut row_key = Vec::new();
            for col_name in &other.column_order {
                if let Some(column) = other.columns.get(col_name) {
                    if let Some(value) = column.get(row_idx) {
                        row_key.push(format!("{:?}", value));
                    } else {
                        row_key.push("NULL".to_string());
                    }
                }
            }
            other_rows.insert(row_key.join("|"));
        }

        let mut result_columns: HashMap<String, Vec<crate::types::AttrValue>> = HashMap::new();
        for col_name in &self.column_order {
            result_columns.insert(col_name.clone(), Vec::new());
        }

        // Find intersecting rows from first table
        let mut seen_rows = std::collections::HashSet::new();
        for row_idx in 0..self.nrows {
            let mut row_key = Vec::new();
            for col_name in &self.column_order {
                if let Some(column) = self.columns.get(col_name) {
                    if let Some(value) = column.get(row_idx) {
                        row_key.push(format!("{:?}", value));
                    } else {
                        row_key.push("NULL".to_string());
                    }
                }
            }
            
            let row_hash = row_key.join("|");
            if other_rows.contains(&row_hash) && seen_rows.insert(row_hash) {
                // Add this intersecting row
                for col_name in &self.column_order {
                    if let Some(column) = self.columns.get(col_name) {
                        if let Some(values) = result_columns.get_mut(col_name) {
                            if let Some(value) = column.get(row_idx) {
                                values.push(value.clone());
                            } else {
                                values.push(crate::types::AttrValue::Null);
                            }
                        }
                    }
                }
            }
        }

        // Convert to BaseArray columns
        let mut base_columns = HashMap::new();
        for (name, values) in result_columns {
            base_columns.insert(name, BaseArray::from_attr_values(values));
        }

        let mut result = BaseTable::from_columns(base_columns)?;
        result.column_order = self.column_order.clone();
        Ok(result)
    }

    // ==================================================================================
    // PHASE 2: STREAMING FUNCTIONALITY (FOUNDATION ONLY - specialized types delegate)
    // ==================================================================================
    
    /// Launch interactive streaming table in browser (FOUNDATION ONLY)
    pub fn interactive(&self, config: Option<InteractiveConfig>) -> GraphResult<BrowserInterface> {
        use std::sync::Arc;
        
        let config = config.unwrap_or_default();
        
        // Create data source from self
        let data_source: Arc<dyn DataSource> = Arc::new(self.clone());
        
        // Launch streaming server asynchronously 
        let server = StreamingServer::new(data_source, config.streaming_config);
        let port = config.port;
        
        // Find an available port if port is 0
        let actual_port = if port == 0 {
            // Find an available port by binding to 0 and checking what we get
            use std::net::{TcpListener, SocketAddr};
            let listener = TcpListener::bind("127.0.0.1:0")
                .map_err(|e| crate::errors::GraphError::InvalidInput(
                    format!("Failed to find available port: {}", e)
                ))?;
            let port = listener.local_addr()
                .map_err(|e| crate::errors::GraphError::InvalidInput(
                    format!("Failed to get local address: {}", e)
                ))?
                .port();
            drop(listener); // Release the port
            port
        } else {
            port
        };
        
        // Start server using the dedicated background runtime to avoid deadlocks
        let port_hint = actual_port; // 0 for ephemeral or your chosen port
        let server = StreamingServer::new(Arc::clone(&server.data_source), server.config.clone());
        
        // Always use the dedicated background runtime to avoid deadlocks
        let server_handle = server
            .start_background("127.0.0.1".parse().unwrap(), port_hint)
            .map_err(|e| crate::errors::GraphError::InvalidInput(
                format!("Failed to start streaming server: {}", e)
            ))?;
        
        // Create browser interface using the actual assigned port
        let actual_port = server_handle.port;
        let browser_interface = BrowserInterface {
            server_handle,
            url: format!("http://127.0.0.1:{}", actual_port),
            config: config.browser_config,
        };
        
        // TODO: Open browser automatically if requested
        println!("🚀 Interactive table launched at: {}", browser_interface.url);
        println!("📊 Streaming {} rows × {} columns", self.total_rows(), self.total_cols());
        
        Ok(browser_interface)
    }
    
    /// Generate embedded iframe HTML for Jupyter notebooks
    pub fn interactive_embed(&mut self, config: Option<InteractiveConfig>) -> GraphResult<String> {
        // Start the streaming server
        let browser_interface = self.interactive(config)?;
        
        // Store the server handle to keep it alive
        self.active_server_handles.push(browser_interface.server_handle);
        
        // Generate iframe HTML that can be embedded in Jupyter
        let iframe_html = format!(r#"
            <iframe 
                src="{}" 
                width="100%" 
                height="600px" 
                frameborder="0" 
                style="border: 1px solid #ddd; border-radius: 4px;">
            </iframe>
            <p style="font-size: 12px; color: #666; margin-top: 5px;">
                📊 Interactive streaming table: {} rows × {} columns
            </p>
        "#, 
            browser_interface.url, 
            self.total_rows(), 
            self.total_cols()
        );
        
        Ok(iframe_html)
    }
    
    /// Increment version for cache invalidation
    pub fn increment_version(&mut self) {
        self.version += 1;
    }
    
    /// Get streaming configuration
    pub fn streaming_config(&self) -> &StreamingConfig {
        &self.streaming_config
    }
    
    /// Update streaming configuration
    pub fn set_streaming_config(&mut self, config: StreamingConfig) {
        self.streaming_config = config;
    }
    
    /// Stop all active streaming servers
    pub fn stop_all_servers(&mut self) {
        self.active_server_handles.clear(); // Drop will handle cleanup
    }
    
    /// Get number of active streaming servers
    pub fn active_servers_count(&self) -> usize {
        self.active_server_handles.len()
    }
}

// ==================================================================================
// DATASOURCE IMPLEMENTATION FOR BASETABLE (FOUNDATION ONLY)
// ==================================================================================

impl DataSource for BaseTable {
    fn total_rows(&self) -> usize {
        self.nrows
    }
    
    fn total_cols(&self) -> usize {
        self.column_order.len()
    }
    
    fn get_window(&self, start: usize, count: usize) -> StreamingDataWindow {
        let end = std::cmp::min(start + count, self.nrows);
        let actual_count = end.saturating_sub(start);
        
        if actual_count == 0 || start >= self.nrows {
            return StreamingDataWindow::new(
                self.column_names().to_vec(),
                vec![],
                self.get_schema(),
                self.nrows,
                start,
            );
        }
        
        // Extract rows for the window
        let mut rows = Vec::with_capacity(actual_count);
        
        for row_idx in start..end {
            let mut row = Vec::with_capacity(self.column_order.len());
            
            for col_name in &self.column_order {
                if let Some(column) = self.columns.get(col_name) {
                    let value = column.get(row_idx).cloned().unwrap_or(AttrValue::Null);
                    row.push(value);
                } else {
                    row.push(AttrValue::Null);
                }
            }
            
            rows.push(row);
        }
        
        StreamingDataWindow::new(
            self.column_names().to_vec(),
            rows,
            self.get_schema(),
            self.nrows,
            start,
        )
    }
    
    fn get_schema(&self) -> StreamingDataSchema {
        let mut columns = Vec::new();
        
        for col_name in &self.column_order {
            let data_type = if let Some(column) = self.columns.get(col_name) {
                self.infer_column_data_type(column)
            } else {
                DataType::String
            };
            
            columns.push(ColumnSchema {
                name: col_name.clone(),
                data_type,
            });
        }
        
        StreamingDataSchema {
            columns,
            primary_key: None, // BaseTable doesn't enforce primary keys
            source_type: "BaseTable".to_string(),
        }
    }
    
    fn supports_streaming(&self) -> bool {
        true // BaseTable supports real-time streaming
    }
    
    fn get_column_types(&self) -> Vec<DataType> {
        self.column_order.iter()
            .map(|col_name| {
                if let Some(column) = self.columns.get(col_name) {
                    self.infer_column_data_type(column)
                } else {
                    DataType::String
                }
            })
            .collect()
    }
    
    fn get_column_names(&self) -> Vec<String> {
        self.column_order.clone()
    }
    
    fn get_source_id(&self) -> String {
        self.source_id.clone()
    }
    
    fn get_version(&self) -> u64 {
        self.version
    }
}


// ==================================================================================
// PHASE 2 CONFIGURATION TYPES
// ==================================================================================

/// Configuration for interactive streaming tables
#[derive(Debug, Clone)]
pub struct InteractiveConfig {
    /// WebSocket port for streaming server
    pub port: u16,
    
    /// Streaming configuration
    pub streaming_config: StreamingConfig,
    
    /// Browser configuration
    pub browser_config: BrowserConfig,
}

impl Default for InteractiveConfig {
    fn default() -> Self {
        Self {
            port: 0,  // Use port 0 for automatic port assignment to avoid conflicts
            streaming_config: StreamingConfig::default(),
            browser_config: BrowserConfig::default(),
        }
    }
}

impl InteractiveConfig {
    /// Create InteractiveConfig with a specific port
    /// 
    /// Use port 0 for automatic port assignment (recommended)
    /// or specify a custom port (e.g., 8080, 3000, etc.)
    pub fn with_port(port: u16) -> Self {
        Self {
            port,
            streaming_config: StreamingConfig::default(),
            browser_config: BrowserConfig::default(),
        }
    }
    
    /// Create InteractiveConfig with automatic port assignment (same as default)
    pub fn auto_port() -> Self {
        Self::default()
    }
}

/// Browser interface configuration
#[derive(Debug, Clone)]
pub struct BrowserConfig {
    /// Auto-open browser
    pub auto_open: bool,
    
    /// Theme for interface
    pub theme: String,
    
    /// Window title
    pub title: String,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            auto_open: true,
            theme: "sleek".to_string(),
            title: "Groggy Interactive Table".to_string(),
        }
    }
}

/// Browser interface handle
#[derive(Debug)]
pub struct BrowserInterface {
    pub server_handle: crate::core::streaming::websocket_server::ServerHandle,
    pub url: String,
    pub config: BrowserConfig,
}