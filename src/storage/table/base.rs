//! BaseTable - unified table implementation built on BaseArray columns

use super::traits::Table;
use crate::core::{DisplayDataSchema, DisplayDataWindow};
use crate::errors::GraphResult;
use crate::storage::array::BaseArray;
use crate::types::AttrValue;
use crate::viz::display::{ColumnSchema, DataType, DisplayConfig, DisplayEngine, OutputFormat};
use crate::viz::streaming::data_source::{DataSchema, DataWindow};
use crate::viz::streaming::data_source::{GraphNode, LayoutAlgorithm, NodePosition};
use crate::viz::streaming::server::StreamingServer;
use crate::viz::streaming::types::StreamingConfig;
use crate::viz::streaming::DataSource;
use crate::viz::VizModule;
use std::collections::HashMap;

/// Statistics calculated for a single column in describe() method
#[derive(Debug, Clone)]
struct ColumnStatistics {
    count: usize,
    mean: f64,
    std: f64,
    min: AttrValue,
    q25: AttrValue,
    q50: AttrValue,
    q75: AttrValue,
    max: AttrValue,
}

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
    active_server_handles: Vec<crate::viz::streaming::types::ServerHandle>,
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
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

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

        // Sort column names for deterministic ordering (like groupby sorting)
        let mut column_order: Vec<String> = normalized_columns.keys().cloned().collect();
        column_order.sort();

        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

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
    pub fn with_column_order(
        columns: HashMap<String, BaseArray<AttrValue>>,
        column_order: Vec<String>,
    ) -> GraphResult<Self> {
        // Validate column order matches available columns
        for col in &column_order {
            if !columns.contains_key(col) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Column '{}' specified in order but not found in columns",
                    col
                )));
            }
        }

        if columns.is_empty() {
            return Ok(Self::new());
        }

        // Validate all columns have same length
        let first_len = columns.values().next().unwrap().len();
        for (name, column) in &columns {
            if column.len() != first_len {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Column '{}' has length {} but expected {}",
                    name,
                    column.len(),
                    first_len
                )));
            }
        }

        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

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
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "UID column '{}' not found",
                uid_key
            )));
        }

        // If uid_key is not "node_id", we need to rename it
        let table_with_node_id = if uid_key == "node_id" {
            self
        } else {
            // For now, require that the UID column is already named "node_id"
            // Future enhancement could rename the column
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "UID column must be named 'node_id', found '{}'",
                uid_key
            )));
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
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "EdgesTable requires '{}' column",
                    col
                )));
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
    pub fn assign(
        &mut self,
        updates: HashMap<String, Vec<crate::types::AttrValue>>,
    ) -> GraphResult<()> {
        use crate::types::{AttrValue, AttrValueType};

        // Validate that all update vectors have the correct length
        for (col_name, values) in &updates {
            if values.len() != self.nrows {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Column '{}' update has {} values but table has {} rows",
                    col_name,
                    values.len(),
                    self.nrows
                )));
            }
        }

        // Apply updates
        for (col_name, values) in updates {
            // Infer dtype from the first non-null value
            let dtype = values
                .iter()
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
    pub fn set_column(
        &mut self,
        column_name: &str,
        values: Vec<crate::types::AttrValue>,
    ) -> GraphResult<()> {
        use crate::types::{AttrValue, AttrValueType};

        if values.len() != self.nrows {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' has {} values but table has {} rows",
                column_name,
                values.len(),
                self.nrows
            )));
        }

        // Infer dtype from the first non-null value
        let dtype = values
            .iter()
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
    pub fn set_value(
        &mut self,
        row: usize,
        column_name: &str,
        value: crate::types::AttrValue,
    ) -> GraphResult<()> {
        if row >= self.nrows {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Row index {} out of bounds (table has {} rows)",
                row, self.nrows
            )));
        }

        let column = self.columns.get_mut(column_name).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column_name
            ))
        })?;

        column.set(row, value)?;
        Ok(())
    }

    /// Set values for multiple rows in a column using a boolean mask
    ///
    /// # Arguments
    /// * `mask` - Boolean vector indicating which rows to update
    /// * `column_name` - Name of the column to update
    /// * `value` - Value to set for all masked rows
    pub fn set_values_by_mask(
        &mut self,
        mask: &[bool],
        column_name: &str,
        value: crate::types::AttrValue,
    ) -> GraphResult<()> {
        if mask.len() != self.nrows {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Mask length {} does not match table rows {}",
                mask.len(),
                self.nrows
            )));
        }

        let column = self.columns.get_mut(column_name).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column_name
            ))
        })?;

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
    pub fn set_values_by_range(
        &mut self,
        start: usize,
        end: usize,
        step: usize,
        column_name: &str,
        value: crate::types::AttrValue,
    ) -> GraphResult<()> {
        if start >= self.nrows || end > self.nrows {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Range [{}, {}) is out of bounds for table with {} rows",
                start, end, self.nrows
            )));
        }

        if step == 0 {
            return Err(crate::errors::GraphError::InvalidInput(
                "Step size must be greater than 0".to_string(),
            ));
        }

        let column = self.columns.get_mut(column_name).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column_name
            ))
        })?;

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
    pub fn set_multiple_values(
        &mut self,
        row_indices: &[usize],
        column_updates: &HashMap<String, crate::types::AttrValue>,
    ) -> GraphResult<()> {
        // Validate row indices
        for &row_idx in row_indices {
            if row_idx >= self.nrows {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Row index {} out of bounds (table has {} rows)",
                    row_idx, self.nrows
                )));
            }
        }

        // Validate columns exist
        for col_name in column_updates.keys() {
            if !self.columns.contains_key(col_name) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Column '{}' not found in table",
                    col_name
                )));
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

    /// Comprehensive random sampling method
    ///
    /// # Parameters
    /// - `n`: Number of rows to sample (mutually exclusive with `fraction`)
    /// - `fraction`: Fraction of rows to sample (0.0 to 1.0, mutually exclusive with `n`)
    /// - `weights`: Optional weights for each row (must match table length)
    /// - `subset`: Optional subset of columns to consider for sampling
    /// - `class_weights`: Optional mapping of column values to weights for stratified sampling
    /// - `replace`: Whether to sample with replacement (default: false)
    ///
    /// # Examples
    /// ```
    /// // Sample 10 rows
    /// let sample = table.sample(Some(10), None, None, None, None, false)?;
    ///
    /// // Sample 20% of rows
    /// let sample = table.sample(None, Some(0.2), None, None, None, false)?;
    ///
    /// // Weighted sampling
    /// let weights = vec![1.0, 2.0, 1.0, 3.0]; // Higher weight for certain rows
    /// let sample = table.sample(Some(5), None, Some(weights), None, None, true)?;
    ///
    /// // Stratified sampling by 'category' column
    /// let mut class_weights = HashMap::new();
    /// class_weights.insert("category", vec![("A", 2.0), ("B", 1.0), ("C", 3.0)]);
    /// let sample = table.sample(Some(10), None, None, None, Some(class_weights), false)?;
    /// ```
    pub fn sample(
        &self,
        n: Option<usize>,
        fraction: Option<f64>,
        weights: Option<Vec<f64>>,
        subset: Option<Vec<String>>,
        class_weights: Option<HashMap<String, Vec<(String, f64)>>>,
        replace: bool,
    ) -> GraphResult<Self> {
        // Validate input parameters
        if n.is_some() && fraction.is_some() {
            return Err(crate::errors::GraphError::InvalidInput(
                "Cannot specify both n and fraction".to_string(),
            ));
        }

        if n.is_none() && fraction.is_none() {
            return Err(crate::errors::GraphError::InvalidInput(
                "Must specify either n or fraction".to_string(),
            ));
        }

        if let Some(frac) = fraction {
            if frac < 0.0 || frac > 1.0 {
                return Err(crate::errors::GraphError::InvalidInput(
                    "Fraction must be between 0.0 and 1.0".to_string(),
                ));
            }
        }

        let total_rows = self.nrows();
        if total_rows == 0 {
            return Ok(self.clone());
        }

        // Calculate actual number of rows to sample
        let sample_size = if let Some(n_val) = n {
            if !replace && n_val > total_rows {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Cannot sample {} rows without replacement from {} total rows",
                    n_val, total_rows
                )));
            }
            n_val
        } else if let Some(frac) = fraction {
            ((frac * total_rows as f64).round() as usize).min(total_rows)
        } else {
            unreachable!()
        };

        if sample_size == 0 {
            return Ok(BaseTable::new());
        }

        // Prepare row weights
        let row_weights = self.calculate_row_weights(weights, subset, class_weights)?;

        // Sample row indices
        let sampled_indices = if replace {
            // Sample with replacement using weighted random selection
            self.weighted_sample_with_replacement(sample_size, &row_weights)?
        } else {
            // Sample without replacement
            self.weighted_sample_without_replacement(sample_size, &row_weights)?
        };

        // Create sampled table
        self.select_rows(&sampled_indices)
    }

    /// Calculate row weights combining all weighting strategies
    fn calculate_row_weights(
        &self,
        weights: Option<Vec<f64>>,
        subset: Option<Vec<String>>,
        class_weights: Option<HashMap<String, Vec<(String, f64)>>>,
    ) -> GraphResult<Vec<f64>> {
        let total_rows = self.nrows();
        let mut final_weights = vec![1.0; total_rows];

        // Apply manual weights if provided
        if let Some(manual_weights) = weights {
            if manual_weights.len() != total_rows {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Weights length {} does not match table rows {}",
                    manual_weights.len(),
                    total_rows
                )));
            }

            for (i, weight) in manual_weights.iter().enumerate() {
                if *weight < 0.0 {
                    return Err(crate::errors::GraphError::InvalidInput(
                        "Weights must be non-negative".to_string(),
                    ));
                }
                final_weights[i] *= weight;
            }
        }

        // Apply subset-based weights (rows with non-null values in subset columns get higher weight)
        if let Some(subset_cols) = subset {
            for row_idx in 0..total_rows {
                let mut non_null_count = 0;
                let mut total_cols = 0;

                for col_name in &subset_cols {
                    if let Some(column) = self.columns().get(col_name) {
                        total_cols += 1;
                        if let Some(value) = column.get(row_idx) {
                            if !matches!(value, AttrValue::Null) {
                                non_null_count += 1;
                            }
                        }
                    }
                }

                if total_cols > 0 {
                    let completeness_ratio = non_null_count as f64 / total_cols as f64;
                    final_weights[row_idx] *= 1.0 + completeness_ratio; // Bonus for completeness
                }
            }
        }

        // Apply class-based weights (stratified sampling)
        if let Some(class_weight_map) = class_weights {
            for (col_name, value_weights) in class_weight_map {
                if let Some(column) = self.columns().get(&col_name) {
                    // Create lookup map for faster access
                    let weight_lookup: HashMap<String, f64> = value_weights.into_iter().collect();

                    for row_idx in 0..total_rows {
                        if let Some(value) = column.get(row_idx) {
                            let value_str = match value {
                                AttrValue::Text(s) => s.clone(),
                                AttrValue::Int(i) => i.to_string(),
                                AttrValue::SmallInt(i) => i.to_string(),
                                AttrValue::Float(f) => f.to_string(),
                                AttrValue::Bool(b) => b.to_string(),
                                _ => continue,
                            };

                            if let Some(&class_weight) = weight_lookup.get(&value_str) {
                                final_weights[row_idx] *= class_weight;
                            }
                        }
                    }
                }
            }
        }

        // Ensure all weights are valid
        for weight in &final_weights {
            if weight.is_nan() || weight.is_infinite() {
                return Err(crate::errors::GraphError::InvalidInput(
                    "Invalid weight calculated".to_string(),
                ));
            }
        }

        Ok(final_weights)
    }

    /// Weighted sampling with replacement using fastrand
    fn weighted_sample_with_replacement(
        &self,
        sample_size: usize,
        weights: &[f64],
    ) -> GraphResult<Vec<usize>> {
        // Calculate cumulative weights for weighted sampling
        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            return Err(crate::errors::GraphError::InvalidInput(
                "Total weight must be positive".to_string(),
            ));
        }

        let mut cumulative_weights = Vec::with_capacity(weights.len());
        let mut cumsum = 0.0;
        for &weight in weights {
            cumsum += weight;
            cumulative_weights.push(cumsum);
        }

        let mut sampled_indices = Vec::with_capacity(sample_size);
        for _ in 0..sample_size {
            let random_value = fastrand::f64() * total_weight;

            // Binary search for the selected index
            let selected_idx = match cumulative_weights.binary_search_by(|&x| {
                if x < random_value {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            }) {
                Ok(idx) => idx,
                Err(idx) => idx,
            };

            sampled_indices.push(selected_idx.min(weights.len() - 1));
        }

        Ok(sampled_indices)
    }

    /// Weighted sampling without replacement using fastrand
    fn weighted_sample_without_replacement(
        &self,
        sample_size: usize,
        weights: &[f64],
    ) -> GraphResult<Vec<usize>> {
        let mut sampled_indices = Vec::new();
        let mut available_indices: Vec<usize> = (0..weights.len()).collect();
        let mut available_weights = weights.to_vec();

        for _ in 0..sample_size {
            if available_indices.is_empty() {
                break;
            }

            // Calculate total weight for remaining items
            let total_weight: f64 = available_weights.iter().sum();
            if total_weight <= 0.0 {
                // All remaining weights are zero, do uniform sampling
                let selected_pos = fastrand::usize(..available_indices.len());
                let selected_idx = available_indices[selected_pos];
                sampled_indices.push(selected_idx);
                available_indices.remove(selected_pos);
                available_weights.remove(selected_pos);
                continue;
            }

            // Weighted selection using cumulative distribution
            let random_value = fastrand::f64() * total_weight;
            let mut cumsum = 0.0;
            let mut selected_pos = 0;

            for (pos, &weight) in available_weights.iter().enumerate() {
                cumsum += weight;
                if cumsum >= random_value {
                    selected_pos = pos;
                    break;
                }
            }

            let selected_idx = available_indices[selected_pos];
            sampled_indices.push(selected_idx);

            // Remove selected item from available pool
            available_indices.remove(selected_pos);
            available_weights.remove(selected_pos);
        }

        Ok(sampled_indices)
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
        self.column_order
            .get(index)
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
        let sort_column = self.column(column).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column))
        })?;

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

    /// Sort table by multiple columns with mixed ascending/descending order
    /// Pandas-style multi-column sorting with priority order
    ///
    /// # Arguments
    /// * `columns` - Vector of column names to sort by (in priority order)
    /// * `ascending` - Vector of booleans for sort direction per column
    ///
    /// # Returns
    /// New sorted table with the same structure
    ///
    /// # Examples
    /// ```rust
    /// // Sort by department ascending, then salary descending
    /// let sorted = table.sort_values(
    ///     vec!["department".to_string(), "salary".to_string()],
    ///     vec![true, false]
    /// )?;
    /// ```
    fn sort_values(&self, columns: Vec<String>, ascending: Vec<bool>) -> GraphResult<Self> {
        if columns.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(
                "At least one column must be specified for sorting".to_string(),
            ));
        }

        if columns.len() != ascending.len() {
            return Err(crate::errors::GraphError::InvalidInput(
                "Length of columns and ascending vectors must match".to_string(),
            ));
        }

        // Validate all columns exist
        for col_name in &columns {
            if !self.has_column(col_name) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Column '{}' not found",
                    col_name
                )));
            }
        }

        // Create vector of (row_index, values_for_sorting) for multi-column sorting
        let mut sort_data: Vec<(usize, Vec<&crate::types::AttrValue>)> =
            Vec::with_capacity(self.nrows);

        for row_idx in 0..self.nrows {
            let mut row_values = Vec::with_capacity(columns.len());
            for col_name in &columns {
                let col = self.column(col_name).unwrap(); // Already validated above
                if let Some(value) = col.get(row_idx) {
                    row_values.push(value);
                } else {
                    // Handle missing values by using Null
                    row_values.push(&crate::types::AttrValue::Null);
                }
            }
            sort_data.push((row_idx, row_values));
        }

        // Multi-column sorting with custom comparator
        sort_data.sort_by(|(_, a_values), (_, b_values)| {
            for (col_idx, (&a_val, &b_val)) in a_values.iter().zip(b_values.iter()).enumerate() {
                let cmp = if ascending[col_idx] {
                    self.compare_attr_values_for_sort(a_val, b_val)
                } else {
                    self.compare_attr_values_for_sort(b_val, a_val) // Reverse for descending
                };

                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
                // If equal, continue to next column
            }
            std::cmp::Ordering::Equal
        });

        // Extract the sorted indices
        let indices: Vec<usize> = sort_data.into_iter().map(|(idx, _)| idx).collect();

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
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Column '{}' not found in table",
                    col_name
                )));
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
            let column = self.column(name).ok_or_else(|| {
                crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", name))
            })?;
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
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "New column has {} rows but table has {} rows",
                column.len(),
                self.nrows
            )));
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
            nrows: if self.nrows == 0 {
                column.len()
            } else {
                self.nrows
            },
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

    fn pivot_table(
        &self,
        index_cols: &[String],
        columns_col: &str,
        values_col: &str,
        agg_func: &str,
    ) -> GraphResult<Self> {
        self.pivot_table(index_cols, columns_col, values_col, agg_func)
    }

    fn melt(
        &self,
        id_vars: Option<&[String]>,
        value_vars: Option<&[String]>,
        var_name: Option<String>,
        value_name: Option<String>,
    ) -> GraphResult<Self> {
        self.melt(id_vars, value_vars, var_name, value_name)
    }
}

impl BaseTable {
    /// Filter by boolean mask
    pub fn filter_by_mask(&self, mask: &[bool]) -> GraphResult<Self> {
        if mask.len() != self.nrows {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Mask length {} doesn't match table rows {}",
                mask.len(),
                self.nrows
            )));
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
        if (self.nrows > config.max_rows && config.max_rows > 6)
            || (self.column_order.len() > config.max_cols && config.max_cols > 6)
        {
            // Use ellipses pattern for both dense matrices and large tables
            return self.create_ellipses_data_window(config);
        }

        // Standard truncation for small tables
        let max_rows = config.max_rows.min(self.nrows);
        let max_cols = config.max_cols.min(self.column_order.len());

        // Extract headers (limited by max_cols)
        let headers: Vec<String> = self.column_order.iter().take(max_cols).cloned().collect();

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
            self.nrows,              // total_rows
            self.column_order.len(), // total_cols
            0,                       // start_offset
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
            display_headers.push(
                if config.show_headers {
                    "⋯ (cols)"
                } else {
                    "⋯"
                }
                .to_string(),
            );
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
            let ellipsis_row: Vec<String> =
                display_headers.iter().map(|_| "⋮".to_string()).collect();
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
        let columns_schema: Vec<ColumnSchema> = display_headers
            .iter()
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
            0,                       // start_offset
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
            }
            OutputFormat::Html => {
                let mut engine = self.display_engine.clone();
                engine.set_config(config);
                engine.format_html(&data_window)
            }
            OutputFormat::Interactive => {
                let mut engine = self.display_engine.clone();
                engine.set_config(config);
                engine.rich_display(&data_window, OutputFormat::Interactive)
            }
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
    /// Enhanced predicate parsing supporting compound expressions and SQL-like operators
    pub fn evaluate_predicate(&self, predicate: &str) -> GraphResult<Vec<bool>> {
        // Handle BETWEEN operator first (before compound AND/OR detection)
        if predicate.contains(" BETWEEN ") || predicate.contains(" between ") {
            return self.evaluate_between_predicate(predicate);
        }

        // Handle compound expressions with AND/OR
        if predicate.contains(" AND ") || predicate.contains(" and ") {
            return self.evaluate_compound_predicate(predicate, true);
        }
        if predicate.contains(" OR ") || predicate.contains(" or ") {
            return self.evaluate_compound_predicate(predicate, false);
        }

        // Handle IN operator
        if predicate.contains(" IN ") || predicate.contains(" in ") {
            return self.evaluate_in_predicate(predicate);
        }

        // Handle LIKE operator
        if predicate.contains(" LIKE ") || predicate.contains(" like ") {
            return self.evaluate_like_predicate(predicate);
        }

        // Fall back to simple predicate evaluation
        self.evaluate_simple_predicate(predicate)
    }

    /// Evaluate simple predicates (backwards compatibility)
    fn evaluate_simple_predicate(&self, predicate: &str) -> GraphResult<Vec<bool>> {
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
                value_str = predicate[pos + op.len()..]
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'');
                break;
            }
        }

        if operator.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Unsupported predicate format: '{}'. Use operators: ==, !=, <, <=, >, >=, IN, LIKE, BETWEEN, AND, OR",
                predicate
            )));
        }

        let filter_column = self.column(column_name).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column_name))
        })?;

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

    /// Evaluate compound predicates with AND/OR
    fn evaluate_compound_predicate(&self, predicate: &str, is_and: bool) -> GraphResult<Vec<bool>> {
        let connector = if is_and { " AND " } else { " OR " };
        let connector_lower = if is_and { " and " } else { " or " };

        // Split on both uppercase and lowercase variants
        let parts: Vec<&str> = if predicate.contains(connector) {
            predicate.split(connector).collect()
        } else {
            predicate.split(connector_lower).collect()
        };

        if parts.len() < 2 {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Invalid compound predicate: '{}'",
                predicate
            )));
        }

        // Evaluate first predicate
        let mut result_mask = self.evaluate_predicate(parts[0].trim())?;

        // Combine with remaining predicates
        for part in parts.iter().skip(1) {
            let part_mask = self.evaluate_predicate(part.trim())?;

            if part_mask.len() != result_mask.len() {
                return Err(crate::errors::GraphError::InvalidInput(
                    "Predicate evaluation returned inconsistent mask sizes".to_string(),
                ));
            }

            for i in 0..result_mask.len() {
                if is_and {
                    result_mask[i] = result_mask[i] && part_mask[i];
                } else {
                    result_mask[i] = result_mask[i] || part_mask[i];
                }
            }
        }

        Ok(result_mask)
    }

    /// Evaluate IN predicates (column IN [val1, val2, val3])
    fn evaluate_in_predicate(&self, predicate: &str) -> GraphResult<Vec<bool>> {
        let in_pos = predicate
            .find(" IN ")
            .or_else(|| predicate.find(" in "))
            .ok_or_else(|| {
                crate::errors::GraphError::InvalidInput("Invalid IN predicate format".to_string())
            })?;

        let column_name = predicate[..in_pos].trim();
        let values_part = predicate[in_pos + 4..].trim(); // Skip " IN "

        // Parse values list [val1, val2, val3] or (val1, val2, val3)
        let values_part = values_part
            .trim_start_matches(['[', '('])
            .trim_end_matches([']', ')']);
        let values: Vec<&str> = values_part
            .split(',')
            .map(|s| s.trim().trim_matches(['"', '\'']))
            .collect();

        let filter_column = self.column(column_name).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column_name))
        })?;

        let mut mask = Vec::with_capacity(self.nrows);

        for i in 0..self.nrows {
            let row_value = filter_column.get(i);
            let matches = match row_value {
                Some(attr_val) => {
                    // Check if attr_val matches any value in the IN list
                    values
                        .iter()
                        .any(|&val| Self::compare_attr_value(attr_val, "==", val).unwrap_or(false))
                }
                None => false,
            };
            mask.push(matches);
        }

        Ok(mask)
    }

    /// Evaluate LIKE predicates with wildcard pattern matching
    fn evaluate_like_predicate(&self, predicate: &str) -> GraphResult<Vec<bool>> {
        let like_pos = predicate
            .find(" LIKE ")
            .or_else(|| predicate.find(" like "))
            .ok_or_else(|| {
                crate::errors::GraphError::InvalidInput("Invalid LIKE predicate format".to_string())
            })?;

        let column_name = predicate[..like_pos].trim();
        let pattern = predicate[like_pos + 6..].trim().trim_matches(['"', '\'']); // Skip " LIKE "

        let filter_column = self.column(column_name).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column_name))
        })?;

        let mut mask = Vec::with_capacity(self.nrows);

        for i in 0..self.nrows {
            let row_value = filter_column.get(i);
            let matches = match row_value {
                Some(attr_val) => {
                    let text = match attr_val {
                        crate::types::AttrValue::Text(s) => s.as_str(),
                        crate::types::AttrValue::CompactText(s) => s.as_str(),
                        _ => {
                            return Err(crate::errors::GraphError::InvalidInput(
                                "LIKE operator only works with text columns".to_string(),
                            ))
                        }
                    };
                    Self::matches_like_pattern(text, pattern)
                }
                None => false,
            };
            mask.push(matches);
        }

        Ok(mask)
    }

    /// Evaluate BETWEEN predicates (column BETWEEN val1 AND val2)
    fn evaluate_between_predicate(&self, predicate: &str) -> GraphResult<Vec<bool>> {
        // Find BETWEEN keyword position (case insensitive)
        let (between_pos, between_len) = if let Some(pos) = predicate.find(" BETWEEN ") {
            (pos, 9) // " BETWEEN " is 9 characters
        } else if let Some(pos) = predicate.find(" between ") {
            (pos, 9) // " between " is 9 characters
        } else {
            return Err(crate::errors::GraphError::InvalidInput(
                "Invalid BETWEEN predicate format".to_string(),
            ));
        };

        let column_name = predicate[..between_pos].trim();
        let values_part = predicate[between_pos + between_len..].trim();

        // Find AND separator in the values part (case insensitive)
        let (and_pos, and_len) = if let Some(pos) = values_part.find(" AND ") {
            (pos, 5) // " AND " is 5 characters
        } else if let Some(pos) = values_part.find(" and ") {
            (pos, 5) // " and " is 5 characters
        } else {
            return Err(crate::errors::GraphError::InvalidInput(
                "BETWEEN requires AND separator".to_string(),
            ));
        };

        let lower_val = values_part[..and_pos].trim().trim_matches(['"', '\'']);
        let upper_val = values_part[and_pos + and_len..]
            .trim()
            .trim_matches(['"', '\'']);

        let filter_column = self.column(column_name).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column_name))
        })?;

        let mut mask = Vec::with_capacity(self.nrows);

        for i in 0..self.nrows {
            let row_value = filter_column.get(i);
            let matches = match row_value {
                Some(attr_val) => {
                    let gte_lower = Self::compare_attr_value(attr_val, ">=", lower_val)?;
                    let lte_upper = Self::compare_attr_value(attr_val, "<=", upper_val)?;
                    gte_lower && lte_upper
                }
                None => false,
            };
            mask.push(matches);
        }

        Ok(mask)
    }

    /// Helper function for LIKE pattern matching with SQL wildcards
    /// Supports % (match any sequence) and _ (match single character)
    fn matches_like_pattern(text: &str, pattern: &str) -> bool {
        // For now, implement basic % wildcard support
        if pattern == "%" {
            true // Match everything
        } else if pattern.starts_with('%') && pattern.ends_with('%') {
            let inner = &pattern[1..pattern.len() - 1];
            text.contains(inner)
        } else if pattern.starts_with('%') {
            let suffix = &pattern[1..];
            text.ends_with(suffix)
        } else if pattern.ends_with('%') {
            let prefix = &pattern[..pattern.len() - 1];
            text.starts_with(prefix)
        } else if pattern.contains('_') {
            // Basic single character wildcard support
            if pattern.len() != text.len() {
                return false;
            }
            pattern
                .chars()
                .zip(text.chars())
                .all(|(p, t)| p == '_' || p == t)
        } else {
            // Exact match
            text == pattern
        }
    }

    /// Compare an AttrValue with a string value using the given operator
    pub fn compare_attr_value(
        attr_val: &crate::types::AttrValue,
        operator: &str,
        value_str: &str,
    ) -> GraphResult<bool> {
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
            }
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
            }
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
            }
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
            }
            AttrValue::Text(s) => match operator {
                "==" => s == value_str,
                "!=" => s != value_str,
                "<" => s.as_str() < value_str,
                "<=" => s.as_str() <= value_str,
                ">" => s.as_str() > value_str,
                ">=" => s.as_str() >= value_str,
                _ => false,
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
            }
            AttrValue::Null => false, // Null values don't match any comparison
            _ => {
                // For other types, convert to string and compare
                let attr_str = format!("{}", attr_val);
                match operator {
                    "==" => attr_str == value_str,
                    "!=" => attr_str != value_str,
                    _ => false, // Other operators not supported for complex types
                }
            }
        };

        Ok(result)
    }

    // ==================================================================================
    // ADVANCED FILTERING METHODS - PHASE 3.2
    // ==================================================================================

    /// Check if values in a column are in a provided set (pandas-style isin)
    ///
    /// # Arguments
    /// * `column_name` - Name of the column to check
    /// * `values` - Vector of values to check membership against
    ///
    /// # Examples
    /// ```rust
    /// use crate::types::AttrValue;
    /// let mask = table.isin("department", vec![
    ///     AttrValue::Text("Engineering".to_string()),
    ///     AttrValue::Text("Marketing".to_string())
    /// ])?;
    /// let filtered = table.filter_by_mask(&mask)?;
    /// ```
    pub fn isin(
        &self,
        column_name: &str,
        values: Vec<crate::types::AttrValue>,
    ) -> GraphResult<Vec<bool>> {
        let filter_column = self.column(column_name).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column_name))
        })?;

        // Delegate to BaseArray.isin() for consistent implementation
        let mask_array = filter_column.isin(values)?;

        // Convert BaseArray<Bool> to Vec<bool>
        let mut mask = Vec::with_capacity(mask_array.len());
        for i in 0..mask_array.len() {
            if let Some(crate::types::AttrValue::Bool(b)) = mask_array.get(i) {
                mask.push(*b); // Dereference is needed since b is &bool
            } else {
                mask.push(false); // Default to false for non-boolean values
            }
        }

        Ok(mask)
    }

    /// Get the N largest values in a column (pandas-style nlargest)
    ///
    /// # Arguments
    /// * `n` - Number of largest values to return
    /// * `column_name` - Name of the column to sort by
    ///
    /// # Examples
    /// ```rust
    /// let top_5_salaries = table.nlargest(5, "salary")?;
    /// ```
    pub fn nlargest(&self, n: usize, column_name: &str) -> GraphResult<Self> {
        let filter_column = self.column(column_name).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column_name))
        })?;

        // Create vector of (value, index) pairs for sorting
        let mut value_indices: Vec<(crate::types::AttrValue, usize)> = Vec::new();

        for i in 0..self.nrows {
            if let Some(value) = filter_column.get(i) {
                // Only include numeric values for sorting
                match value {
                    crate::types::AttrValue::Int(_)
                    | crate::types::AttrValue::SmallInt(_)
                    | crate::types::AttrValue::Float(_) => {
                        value_indices.push((value.clone(), i));
                    }
                    _ => {} // Skip non-numeric values
                }
            }
        }

        // Sort by value in descending order (largest first)
        value_indices.sort_by(|(a, _), (b, _)| {
            self.compare_attr_values_for_sort(b, a) // Reverse order for largest first
        });

        // Take the first n indices
        let top_indices: Vec<usize> = value_indices
            .into_iter()
            .take(n)
            .map(|(_, idx)| idx)
            .collect();

        // Return subset of table with top N rows
        self.select_rows(&top_indices)
    }

    /// Get the N smallest values in a column (pandas-style nsmallest)
    ///
    /// # Arguments
    /// * `n` - Number of smallest values to return
    /// * `column_name` - Name of the column to sort by
    ///
    /// # Examples
    /// ```rust
    /// let bottom_5_salaries = table.nsmallest(5, "salary")?;
    /// ```
    pub fn nsmallest(&self, n: usize, column_name: &str) -> GraphResult<Self> {
        let filter_column = self.column(column_name).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column_name))
        })?;

        // Create vector of (value, index) pairs for sorting
        let mut value_indices: Vec<(crate::types::AttrValue, usize)> = Vec::new();

        for i in 0..self.nrows {
            if let Some(value) = filter_column.get(i) {
                // Only include numeric values for sorting
                match value {
                    crate::types::AttrValue::Int(_)
                    | crate::types::AttrValue::SmallInt(_)
                    | crate::types::AttrValue::Float(_) => {
                        value_indices.push((value.clone(), i));
                    }
                    _ => {} // Skip non-numeric values
                }
            }
        }

        // Sort by value in ascending order (smallest first)
        value_indices.sort_by(|(a, _), (b, _)| {
            self.compare_attr_values_for_sort(a, b) // Normal order for smallest first
        });

        // Take the first n indices
        let top_indices: Vec<usize> = value_indices
            .into_iter()
            .take(n)
            .map(|(_, idx)| idx)
            .collect();

        // Return subset of table with top N rows
        self.select_rows(&top_indices)
    }

    /// Helper method to compare AttrValues for sorting (used by nlargest/nsmallest)
    fn compare_attr_values_for_sort(
        &self,
        a: &crate::types::AttrValue,
        b: &crate::types::AttrValue,
    ) -> std::cmp::Ordering {
        use crate::types::AttrValue;
        match (a, b) {
            (AttrValue::Int(a), AttrValue::Int(b)) => a.cmp(b),
            (AttrValue::SmallInt(a), AttrValue::SmallInt(b)) => a.cmp(b),
            (AttrValue::Float(a), AttrValue::Float(b)) => {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            }
            (AttrValue::Int(a), AttrValue::Float(b)) => (*a as f32)
                .partial_cmp(b)
                .unwrap_or(std::cmp::Ordering::Equal),
            (AttrValue::Float(a), AttrValue::Int(b)) => a
                .partial_cmp(&(*b as f32))
                .unwrap_or(std::cmp::Ordering::Equal),
            (AttrValue::SmallInt(a), AttrValue::Float(b)) => (*a as f32)
                .partial_cmp(b)
                .unwrap_or(std::cmp::Ordering::Equal),
            (AttrValue::Float(a), AttrValue::SmallInt(b)) => a
                .partial_cmp(&(*b as f32))
                .unwrap_or(std::cmp::Ordering::Equal),
            (AttrValue::Int(a), AttrValue::SmallInt(b)) => (*a).cmp(&(*b as i64)),
            (AttrValue::SmallInt(a), AttrValue::Int(b)) => (*a as i64).cmp(b),
            _ => std::cmp::Ordering::Equal, // For non-comparable or mixed types
        }
    }

    /// Pandas-style query method for string-based filtering
    ///
    /// # Arguments
    /// * `expr` - String expression to evaluate (same as enhanced filter syntax)
    ///
    /// # Examples
    /// ```rust
    /// let result = table.query("age > 25 AND department == Engineering")?;
    /// let result = table.query("salary BETWEEN 50000 AND 100000")?;
    /// let result = table.query("name LIKE A%")?;
    /// ```
    pub fn query(&self, expr: &str) -> GraphResult<Self> {
        // Delegate to the enhanced predicate evaluation system
        let mask = self.evaluate_predicate(expr)?;
        self.filter_by_mask(&mask)
    }

    // ==================================================================================
    // STRING OPERATIONS - PHASE 2.3b
    // ==================================================================================

    /// Get column by name (public interface)
    /// Enables table["column_name"] syntax via Index trait
    ///
    /// # Examples
    /// ```rust
    /// let name_column = table.get_column("name")?;
    /// let upper_names = name_column.str().upper();
    /// ```
    pub fn get_column(&self, name: &str) -> Option<&BaseArray<AttrValue>> {
        self.columns.get(name)
    }

    /// Pandas-style pivot table operation for data reshaping
    ///
    /// Creates a pivot table that spreads unique values from one column (columns_col)
    /// into new columns, grouping by index columns (index_cols), and aggregating
    /// values (values_col) using the specified aggregation function.
    ///
    /// # Arguments
    /// * `index_cols` - Columns to group by (become row index)
    /// * `columns_col` - Column whose unique values become new column names
    /// * `values_col` - Column whose values get aggregated
    /// * `agg_func` - Aggregation function: "sum", "mean", "count", "min", "max"
    ///
    /// # Example
    /// ```
    /// // Original data:
    /// // | dept | level | salary |
    /// // | Eng  | Sr    | 100k   |
    /// // | Eng  | Jr    | 80k    |
    /// // | Sales| Sr    | 90k    |
    ///
    /// let pivot = table.pivot_table(&["dept"], "level", "salary", "mean")?;
    /// // Result:
    /// // | dept  | Jr  | Sr  |
    /// // | Eng   | 80k | 100k|
    /// // | Sales |  -  | 90k |
    /// ```
    pub fn pivot_table(
        &self,
        index_cols: &[String],
        columns_col: &str,
        values_col: &str,
        agg_func: &str,
    ) -> GraphResult<Self> {
        // Validation
        if index_cols.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(
                "At least one index column must be specified".to_string(),
            ));
        }

        // Validate columns exist
        for col in index_cols {
            if !self.has_column(col) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Index column '{}' not found",
                    col
                )));
            }
        }

        if !self.has_column(columns_col) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Columns column '{}' not found",
                columns_col
            )));
        }

        if !self.has_column(values_col) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Values column '{}' not found",
                values_col
            )));
        }

        // Get unique values from columns_col to determine new column names
        let columns_array = self.column(columns_col).unwrap();
        let mut unique_values = std::collections::HashSet::new();
        for i in 0..self.nrows {
            if let Some(value) = columns_array.get(i) {
                unique_values.insert(value.clone());
            }
        }

        let mut unique_sorted: Vec<AttrValue> = unique_values.into_iter().collect();
        unique_sorted.sort_by(|a, b| self.compare_attr_values_for_sort(a, b));

        // Create groups based on index columns
        let mut groups: HashMap<Vec<AttrValue>, Vec<usize>> = HashMap::new();

        for row_idx in 0..self.nrows {
            let mut group_key = Vec::new();
            for col_name in index_cols {
                let col = self.column(col_name).unwrap();
                if let Some(value) = col.get(row_idx) {
                    group_key.push(value.clone());
                } else {
                    group_key.push(AttrValue::Null);
                }
            }
            groups
                .entry(group_key)
                .or_insert_with(Vec::new)
                .push(row_idx);
        }

        // Build the pivot table
        let mut new_columns = HashMap::new();
        let mut new_column_order = Vec::new();

        // Add index columns to result
        for col_name in index_cols {
            new_column_order.push(col_name.clone());
        }

        // Add columns for each unique value in columns_col
        for unique_val in &unique_sorted {
            let col_name = format!("{}_{}", columns_col, self.attr_value_to_string(unique_val));
            new_column_order.push(col_name);
        }

        let num_groups = groups.len();

        // Initialize column data vectors
        let mut column_data: std::collections::HashMap<String, Vec<AttrValue>> =
            std::collections::HashMap::new();
        for col_name in &new_column_order {
            column_data.insert(col_name.clone(), Vec::new());
        }

        // Process each group
        for (group_key, row_indices) in groups {
            // Add index values to their columns
            for (i, col_name) in index_cols.iter().enumerate() {
                let col_data = column_data.get_mut(col_name).unwrap();
                col_data.push(group_key[i].clone());
            }

            // For each unique value in columns_col, calculate aggregated value
            for unique_val in &unique_sorted {
                let col_name = format!("{}_{}", columns_col, self.attr_value_to_string(unique_val));
                let col_data = column_data.get_mut(&col_name).unwrap();

                // Find rows that match this unique value and collect values to aggregate
                let mut values_to_agg = Vec::new();
                let columns_array = self.column(columns_col).unwrap();
                let values_array = self.column(values_col).unwrap();

                for &row_idx in &row_indices {
                    if let (Some(col_val), Some(val)) =
                        (columns_array.get(row_idx), values_array.get(row_idx))
                    {
                        if self.compare_attr_values_for_sort(col_val, unique_val)
                            == std::cmp::Ordering::Equal
                        {
                            values_to_agg.push(val);
                        }
                    }
                }

                // Apply aggregation function
                let aggregated_value = if values_to_agg.is_empty() {
                    AttrValue::Null
                } else {
                    self.aggregate_values(&values_to_agg, agg_func)?
                };

                col_data.push(aggregated_value);
            }
        }

        // Convert vectors to BaseArrays
        for (col_name, data) in column_data {
            new_columns.insert(col_name, BaseArray::new(data));
        }

        Ok(Self {
            columns: new_columns,
            column_order: new_column_order,
            nrows: num_groups,
            display_engine: self.display_engine.clone(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: self.streaming_config.clone(),
            source_id: format!("{}_pivot", self.source_id),
            version: self.version + 1,
        })
    }

    /// Pandas-style melt operation for data unpivoting
    ///
    /// Transforms wide-format data to long-format by unpivoting columns into rows.
    /// - id_vars: Column(s) to use as identifier variables (remain unchanged)
    /// - value_vars: Column(s) to unpivot (if None, uses all columns except id_vars)
    /// - var_name: Name for the new variable column (default: "variable")
    /// - value_name: Name for the new value column (default: "value")
    pub fn melt(
        &self,
        id_vars: Option<&[String]>,
        value_vars: Option<&[String]>,
        var_name: Option<String>,
        value_name: Option<String>,
    ) -> GraphResult<Self> {
        // Set default names for variable and value columns
        let var_name = var_name.unwrap_or_else(|| "variable".to_string());
        let value_name = value_name.unwrap_or_else(|| "value".to_string());

        // Determine id_vars (columns to keep as identifiers)
        let id_vars = id_vars.unwrap_or(&[]);

        // Validate that id_vars exist in the table
        for col in id_vars {
            if !self.has_column(col) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Column '{}' not found",
                    col
                )));
            }
        }

        // Determine value_vars (columns to unpivot)
        let value_vars = if let Some(value_vars) = value_vars {
            // Validate that value_vars exist in the table
            for col in value_vars {
                if !self.has_column(col) {
                    return Err(crate::errors::GraphError::InvalidInput(format!(
                        "Column '{}' not found",
                        col
                    )));
                }
            }
            value_vars.to_vec()
        } else {
            // Use all columns except id_vars
            self.column_order
                .iter()
                .filter(|col| !id_vars.contains(col))
                .cloned()
                .collect()
        };

        if value_vars.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(
                "No columns to melt".to_string(),
            ));
        }

        // Calculate the number of rows in the melted table
        let melted_rows = self.nrows * value_vars.len();

        // Create new column order: id_vars + var_name + value_name
        let mut new_column_order = id_vars.to_vec();
        new_column_order.push(var_name.clone());
        new_column_order.push(value_name.clone());

        // Initialize column data vectors
        let mut column_data: std::collections::HashMap<String, Vec<AttrValue>> =
            std::collections::HashMap::new();
        for col_name in &new_column_order {
            column_data.insert(col_name.clone(), Vec::with_capacity(melted_rows));
        }

        // Fill the melted data
        for row_idx in 0..self.nrows {
            for value_var in &value_vars {
                // Add id_vars values for this row
                for id_var in id_vars {
                    let value = self
                        .column(id_var)
                        .and_then(|col| col.get(row_idx))
                        .cloned()
                        .unwrap_or(AttrValue::Null);

                    let col_data = column_data.get_mut(id_var).unwrap();
                    col_data.push(value);
                }

                // Add variable name
                let var_col_data = column_data.get_mut(&var_name).unwrap();
                var_col_data.push(AttrValue::Text(value_var.clone()));

                // Add the value from the current value_var column
                let value = self
                    .column(value_var)
                    .and_then(|col| col.get(row_idx))
                    .cloned()
                    .unwrap_or(AttrValue::Null);

                let value_col_data = column_data.get_mut(&value_name).unwrap();
                value_col_data.push(value);
            }
        }

        // Convert vectors to BaseArrays
        let mut new_columns = std::collections::HashMap::new();
        for (col_name, data) in column_data {
            new_columns.insert(col_name, BaseArray::new(data));
        }

        Ok(Self {
            columns: new_columns,
            column_order: new_column_order,
            nrows: melted_rows,
            display_engine: self.display_engine.clone(),
            streaming_server: None,
            active_server_handles: Vec::new(),
            streaming_config: self.streaming_config.clone(),
            source_id: format!("{}_melt", self.source_id),
            version: self.version + 1,
        })
    }

    /// Count the frequency of unique values in a column (pandas-style value_counts)
    ///
    /// # Arguments
    /// * `column` - Name of the column to analyze
    /// * `sort` - Whether to sort results by frequency (default: true)
    /// * `ascending` - Sort order when sort=true (default: false, most frequent first)
    /// * `dropna` - Whether to exclude null values (default: true)
    ///
    /// # Returns
    /// Table with 'value' and 'count' columns showing frequency of each unique value
    ///
    /// # Examples
    /// ```rust
    /// let counts = table.value_counts("category", true, false, true)?;
    /// // Returns frequency table for the 'category' column
    /// ```
    pub fn value_counts(
        &self,
        column: &str,
        sort: bool,
        ascending: bool,
        dropna: bool,
    ) -> GraphResult<Self> {
        // Validate column exists
        if !self.has_column(column) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column
            )));
        }

        // Get the column and delegate to BaseArray::value_counts
        let array = self.column(column).unwrap();
        array.value_counts(sort, ascending, dropna)
    }

    /// Compute quantile for a specific column (pandas-style quantile)
    ///
    /// # Arguments
    /// * `column` - Column name to analyze
    /// * `q` - Quantile to compute (0.0 to 1.0)
    /// * `interpolation` - Method for interpolation ("linear", "lower", "higher", "midpoint", "nearest")
    ///
    /// # Returns
    /// AttrValue containing the computed quantile
    ///
    /// # Examples
    /// ```rust
    /// // Get median (50th percentile) of sales column
    /// let median = table.quantile("sales", 0.5, "linear")?;
    ///
    /// // Get 95th percentile with nearest interpolation
    /// let p95 = table.quantile("price", 0.95, "nearest")?;
    /// ```
    pub fn quantile(&self, column: &str, q: f64, interpolation: &str) -> GraphResult<AttrValue> {
        // Validate column exists
        if !self.has_column(column) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column
            )));
        }

        // Get the column and delegate to BaseArray::quantile
        let array = self.column(column).unwrap();
        array.quantile(q, interpolation)
    }

    /// Compute multiple quantiles for a specific column
    ///
    /// # Arguments
    /// * `column` - Column name to analyze
    /// * `quantiles` - Slice of quantiles to compute (each 0.0 to 1.0)
    /// * `interpolation` - Method for interpolation
    ///
    /// # Returns
    /// BaseArray containing the computed quantiles
    pub fn quantiles(
        &self,
        column: &str,
        quantiles: &[f64],
        interpolation: &str,
    ) -> GraphResult<BaseArray<AttrValue>> {
        // Validate column exists
        if !self.has_column(column) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column
            )));
        }

        // Get the column and delegate to BaseArray::quantiles
        let array = self.column(column).unwrap();
        array.quantiles(quantiles, interpolation)
    }

    /// Compute percentile for a specific column
    ///
    /// # Arguments
    /// * `column` - Column name to analyze
    /// * `percentile` - Percentile to compute (0.0 to 100.0)
    /// * `interpolation` - Method for interpolation
    ///
    /// # Examples
    /// ```rust
    /// // Get median (50th percentile)
    /// let median = table.percentile("sales", 50.0, "linear")?;
    /// ```
    pub fn get_percentile(
        &self,
        column: &str,
        percentile: f64,
        interpolation: &str,
    ) -> GraphResult<AttrValue> {
        // Validate column exists
        if !self.has_column(column) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column
            )));
        }

        // Get the column and delegate to BaseArray::get_percentile
        let array = self.column(column).unwrap();
        array.get_percentile(percentile, interpolation)
    }

    /// Compute multiple percentiles for a specific column
    pub fn percentiles(
        &self,
        column: &str,
        percentiles: &[f64],
        interpolation: &str,
    ) -> GraphResult<BaseArray<AttrValue>> {
        // Validate column exists
        if !self.has_column(column) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column
            )));
        }

        // Get the column and delegate to BaseArray::percentiles
        let array = self.column(column).unwrap();
        array.percentiles(percentiles, interpolation)
    }

    /// Calculate median for a specific column
    ///
    /// # Arguments
    /// * `column` - Column name to analyze
    ///
    /// # Returns
    /// Median value as f64
    ///
    /// # Examples
    /// ```rust
    /// // Get median of age column
    /// let median_age = table.median("age")?;
    /// ```
    pub fn median(&self, column: &str) -> GraphResult<f64> {
        // Validate column exists
        if !self.has_column(column) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column
            )));
        }

        // Get the column and delegate to BaseArray::median
        let array = self.column(column).unwrap();
        array.median()
    }

    /// Calculate standard deviation for a specific column
    ///
    /// # Arguments
    /// * `column` - Column name to analyze
    ///
    /// # Returns
    /// Standard deviation as f64
    ///
    /// # Examples
    /// ```rust
    /// // Get standard deviation of scores column
    /// let std_scores = table.std("scores")?;
    /// ```
    pub fn std(&self, column: &str) -> GraphResult<f64> {
        // Validate column exists
        if !self.has_column(column) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column
            )));
        }

        // Get the column and delegate to BaseArray::std
        let array = self.column(column).unwrap();
        array.std()
    }

    /// Calculate variance for a specific column
    ///
    /// # Arguments
    /// * `column` - Column name to analyze
    ///
    /// # Returns
    /// Variance as f64
    ///
    /// # Examples
    /// ```rust
    /// // Get variance of prices column
    /// let var_prices = table.var("prices")?;
    /// ```
    pub fn var(&self, column: &str) -> GraphResult<f64> {
        // Validate column exists
        if !self.has_column(column) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column
            )));
        }

        // Get the column and delegate to BaseArray::var
        let array = self.column(column).unwrap();
        array.var()
    }

    /// Compute correlation between two columns (pandas-style corr for column pairs)
    ///
    /// # Arguments
    /// * `column1` - First column name
    /// * `column2` - Second column name
    /// * `method` - Correlation method ("pearson", "spearman", "kendall")
    ///
    /// # Returns
    /// Correlation coefficient as AttrValue
    ///
    /// # Examples
    /// ```rust
    /// // Compute Pearson correlation between sales and advertising
    /// let corr = table.corr_columns("sales", "advertising", "pearson")?;
    ///
    /// // Compute Spearman rank correlation
    /// let spearman = table.corr_columns("price", "demand", "spearman")?;
    /// ```
    pub fn corr_columns(
        &self,
        column1: &str,
        column2: &str,
        method: &str,
    ) -> GraphResult<AttrValue> {
        // Validate both columns exist
        if !self.has_column(column1) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column1
            )));
        }
        if !self.has_column(column2) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column2
            )));
        }

        // Get both columns and delegate to BaseArray::corr
        let array1 = self.column(column1).unwrap();
        let array2 = self.column(column2).unwrap();
        array1.corr(array2, method)
    }

    /// Compute correlation matrix for all numeric columns (pandas-style corr)
    ///
    /// # Arguments
    /// * `method` - Correlation method ("pearson", "spearman", "kendall")
    ///
    /// # Returns
    /// Correlation matrix as BaseTable with columns and rows representing variables
    ///
    /// # Examples
    /// ```rust
    /// // Get full correlation matrix
    /// let corr_matrix = table.corr("pearson")?;
    ///
    /// // Spearman rank correlation matrix
    /// let spearman_matrix = table.corr("spearman")?;
    /// ```
    pub fn corr(&self, method: &str) -> GraphResult<Self> {
        // Find all numeric columns
        let mut numeric_columns = Vec::new();
        for col_name in &self.column_order {
            if let Some(column) = self.columns.get(col_name) {
                // Check if column has any numeric values
                let has_numeric = column.iter().any(|val| {
                    matches!(
                        val,
                        AttrValue::Int(_) | AttrValue::SmallInt(_) | AttrValue::Float(_)
                    )
                });
                if has_numeric {
                    numeric_columns.push(col_name.clone());
                }
            }
        }

        if numeric_columns.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(
                "No numeric columns found for correlation matrix".to_string(),
            ));
        }

        // Create correlation matrix
        let mut matrix_data = HashMap::new();

        // Add index column (row names)
        matrix_data.insert(
            "index".to_string(),
            BaseArray::from_attr_values(
                numeric_columns
                    .iter()
                    .map(|name| AttrValue::Text(name.clone()))
                    .collect(),
            ),
        );

        // Calculate correlations for each column pair
        for col1 in &numeric_columns {
            let mut correlations = Vec::new();
            let array1 = self.column(col1).unwrap();

            for col2 in &numeric_columns {
                let array2 = self.column(col2).unwrap();
                let corr_value = if col1 == col2 {
                    AttrValue::Float(1.0) // Self-correlation is always 1
                } else {
                    array1.corr(array2, method).unwrap_or(AttrValue::Null)
                };
                correlations.push(corr_value);
            }

            matrix_data.insert(col1.clone(), BaseArray::from_attr_values(correlations));
        }

        // Create column order: index first, then all numeric columns
        let mut column_order = vec!["index".to_string()];
        column_order.extend(numeric_columns);

        Self::with_column_order(matrix_data, column_order)
    }

    /// Compute covariance between two columns (pandas-style cov for column pairs)
    ///
    /// # Arguments
    /// * `column1` - First column name
    /// * `column2` - Second column name
    /// * `ddof` - Delta degrees of freedom (default: 1 for sample covariance)
    ///
    /// # Returns
    /// Covariance as AttrValue
    pub fn cov_columns(&self, column1: &str, column2: &str, ddof: i32) -> GraphResult<AttrValue> {
        // Validate both columns exist
        if !self.has_column(column1) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column1
            )));
        }
        if !self.has_column(column2) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found in table",
                column2
            )));
        }

        // Get both columns and delegate to BaseArray::cov
        let array1 = self.column(column1).unwrap();
        let array2 = self.column(column2).unwrap();
        array1.cov(array2, ddof)
    }

    /// Compute covariance matrix for all numeric columns (pandas-style cov)
    ///
    /// # Arguments
    /// * `ddof` - Delta degrees of freedom (default: 1 for sample covariance)
    ///
    /// # Returns
    /// Covariance matrix as BaseTable
    ///
    /// # Examples
    /// ```rust
    /// // Sample covariance matrix
    /// let cov_matrix = table.cov(1)?;
    ///
    /// // Population covariance matrix
    /// let pop_cov = table.cov(0)?;
    /// ```
    pub fn cov(&self, ddof: i32) -> GraphResult<Self> {
        // Find all numeric columns
        let mut numeric_columns = Vec::new();
        for col_name in &self.column_order {
            if let Some(column) = self.columns.get(col_name) {
                // Check if column has any numeric values
                let has_numeric = column.iter().any(|val| {
                    matches!(
                        val,
                        AttrValue::Int(_) | AttrValue::SmallInt(_) | AttrValue::Float(_)
                    )
                });
                if has_numeric {
                    numeric_columns.push(col_name.clone());
                }
            }
        }

        if numeric_columns.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(
                "No numeric columns found for covariance matrix".to_string(),
            ));
        }

        // Create covariance matrix
        let mut matrix_data = HashMap::new();

        // Add index column (row names)
        matrix_data.insert(
            "index".to_string(),
            BaseArray::from_attr_values(
                numeric_columns
                    .iter()
                    .map(|name| AttrValue::Text(name.clone()))
                    .collect(),
            ),
        );

        // Calculate covariances for each column pair
        for col1 in &numeric_columns {
            let mut covariances = Vec::new();
            let array1 = self.column(col1).unwrap();

            for col2 in &numeric_columns {
                let array2 = self.column(col2).unwrap();
                let cov_value = array1.cov(array2, ddof).unwrap_or(AttrValue::Null);
                covariances.push(cov_value);
            }

            matrix_data.insert(col1.clone(), BaseArray::from_attr_values(covariances));
        }

        // Create column order: index first, then all numeric columns
        let mut column_order = vec!["index".to_string()];
        column_order.extend(numeric_columns);

        Self::with_column_order(matrix_data, column_order)
    }

    /// Apply a function to each column in the table (pandas-style apply with axis=0)
    ///
    /// # Arguments
    /// * `func` - Function to apply to each column, takes a BaseArray and returns AttrValue
    ///
    /// # Returns
    /// New single-row table with one column per original column containing the function results
    ///
    /// # Examples
    /// ```rust
    /// use crate::storage::table::BaseTable;
    /// use crate::types::AttrValue;
    ///
    /// // Get column sums
    /// let result = table.apply_to_columns(|col| {
    ///     let sum: f64 = col.iter()
    ///         .filter_map(|v| match v {
    ///             AttrValue::Int(n) => Some(*n as f64),
    ///             AttrValue::Float(f) => Some(*f),
    ///             _ => None
    ///         })
    ///         .sum();
    ///     AttrValue::Float(sum)
    /// })?;
    /// ```
    pub fn apply_to_columns<F>(&self, func: F) -> GraphResult<Self>
    where
        F: Fn(&BaseArray<AttrValue>) -> AttrValue,
    {
        let mut result_data = HashMap::new();

        for col_name in &self.column_order {
            if let Some(column) = self.columns.get(col_name) {
                let result_value = func(column);
                result_data.insert(col_name.clone(), vec![result_value]);
            }
        }

        // Create result table
        let mut result_columns = HashMap::new();
        for (col_name, values) in result_data {
            result_columns.insert(col_name, BaseArray::from_attr_values(values));
        }

        Self::from_columns(result_columns)
    }

    /// Apply a function to each row in the table (pandas-style apply with axis=1)
    ///
    /// # Arguments
    /// * `func` - Function to apply to each row, takes a HashMap<String, AttrValue> and returns AttrValue
    /// * `result_name` - Name for the result column
    ///
    /// # Returns
    /// New single-column table with one row per original row containing the function results
    ///
    /// # Examples
    /// ```rust
    /// use crate::storage::table::BaseTable;
    /// use crate::types::AttrValue;
    /// use std::collections::HashMap;
    ///
    /// // Compute row sums
    /// let result = table.apply_to_rows(
    ///     |row| {
    ///         let sum: f64 = row.values()
    ///             .filter_map(|v| match v {
    ///                 AttrValue::Int(n) => Some(*n as f64),
    ///                 AttrValue::Float(f) => Some(*f),
    ///                 _ => None
    ///             })
    ///             .sum();
    ///         AttrValue::Float(sum)
    ///     },
    ///     "row_sum"
    /// )?;
    /// ```
    pub fn apply_to_rows<F>(&self, func: F, result_name: &str) -> GraphResult<Self>
    where
        F: Fn(&HashMap<String, AttrValue>) -> AttrValue,
    {
        let mut result_values = Vec::with_capacity(self.nrows);

        for row_idx in 0..self.nrows {
            let mut row_data = HashMap::new();

            for col_name in &self.column_order {
                if let Some(column) = self.columns.get(col_name) {
                    let value = column.get(row_idx).cloned().unwrap_or(AttrValue::Null);
                    row_data.insert(col_name.clone(), value);
                }
            }

            let result_value = func(&row_data);
            result_values.push(result_value);
        }

        // Create result table
        let mut result_columns = HashMap::new();
        result_columns.insert(
            result_name.to_string(),
            BaseArray::from_attr_values(result_values),
        );

        Self::from_columns(result_columns)
    }

    /// Append a new row to the table (pandas-style append)
    ///
    /// # Arguments
    /// * `row_data` - HashMap mapping column names to values
    ///
    /// # Returns
    /// New table with the row appended
    ///
    /// # Examples
    /// ```rust
    /// use crate::storage::table::BaseTable;
    /// use crate::types::AttrValue;
    /// use std::collections::HashMap;
    ///
    /// let mut row_data = HashMap::new();
    /// row_data.insert("name".to_string(), AttrValue::Text("Alice".to_string()));
    /// row_data.insert("age".to_string(), AttrValue::Int(30));
    ///
    /// let new_table = table.append(row_data)?;
    /// ```
    pub fn append(&self, row_data: HashMap<String, AttrValue>) -> GraphResult<Self> {
        let mut new_columns = HashMap::new();

        // Extend each existing column with the new row data
        for col_name in &self.column_order {
            if let Some(existing_column) = self.columns.get(col_name) {
                let mut new_data = existing_column.data().clone();
                let new_value = row_data.get(col_name).cloned().unwrap_or(AttrValue::Null);
                new_data.push(new_value);
                new_columns.insert(col_name.clone(), BaseArray::from_attr_values(new_data));
            }
        }

        // Add any new columns from row_data that don't exist in the table
        for (col_name, value) in &row_data {
            if !self.columns.contains_key(col_name) {
                // Create column with nulls for existing rows, then the new value
                let mut new_data = vec![AttrValue::Null; self.nrows];
                new_data.push(value.clone());
                new_columns.insert(col_name.clone(), BaseArray::from_attr_values(new_data));
            }
        }

        // Update column order to include any new columns
        let mut new_column_order = self.column_order.clone();
        for col_name in row_data.keys() {
            if !new_column_order.contains(col_name) {
                new_column_order.push(col_name.clone());
            }
        }

        Self::with_column_order(new_columns, new_column_order)
    }

    /// Extend the table with multiple rows (pandas-style extend/concat)
    ///
    /// # Arguments
    /// * `rows_data` - Vector of HashMaps, each representing a row
    ///
    /// # Returns
    /// New table with all rows appended
    ///
    /// # Examples
    /// ```rust
    /// use crate::storage::table::BaseTable;
    /// use crate::types::AttrValue;
    /// use std::collections::HashMap;
    ///
    /// let rows = vec![
    ///     HashMap::from([
    ///         ("name".to_string(), AttrValue::Text("Alice".to_string())),
    ///         ("age".to_string(), AttrValue::Int(30))
    ///     ]),
    ///     HashMap::from([
    ///         ("name".to_string(), AttrValue::Text("Bob".to_string())),
    ///         ("age".to_string(), AttrValue::Int(25))
    ///     ])
    /// ];
    ///
    /// let new_table = table.extend(rows)?;
    /// ```
    pub fn extend(&self, rows_data: Vec<HashMap<String, AttrValue>>) -> GraphResult<Self> {
        if rows_data.is_empty() {
            return Ok(self.clone());
        }

        // Collect all unique column names from existing table and new rows
        let mut all_columns: std::collections::HashSet<String> =
            self.column_order.iter().cloned().collect();
        for row in &rows_data {
            all_columns.extend(row.keys().cloned());
        }

        let mut new_columns = HashMap::new();
        let new_row_count = self.nrows + rows_data.len();

        // Process each column
        for col_name in &all_columns {
            let mut new_data = Vec::with_capacity(new_row_count);

            // Add existing data
            if let Some(existing_column) = self.columns.get(col_name) {
                new_data.extend(existing_column.data().clone());
            } else {
                // New column - fill with nulls for existing rows
                new_data.resize(self.nrows, AttrValue::Null);
            }

            // Add new row data
            for row in &rows_data {
                let value = row.get(col_name).cloned().unwrap_or(AttrValue::Null);
                new_data.push(value);
            }

            new_columns.insert(col_name.clone(), BaseArray::from_attr_values(new_data));
        }

        // Update column order to include new columns
        let mut new_column_order = self.column_order.clone();
        for col_name in &all_columns {
            if !new_column_order.contains(col_name) {
                new_column_order.push(col_name.clone());
            }
        }

        Self::with_column_order(new_columns, new_column_order)
    }

    /// Helper method to aggregate a list of values using the specified function
    fn aggregate_values(&self, values: &[&AttrValue], agg_func: &str) -> GraphResult<AttrValue> {
        if values.is_empty() {
            return Ok(AttrValue::Null);
        }

        match agg_func {
            "count" => Ok(AttrValue::Int(values.len() as i64)),
            "sum" => {
                let mut sum = 0.0;
                for value in values {
                    match value {
                        AttrValue::Int(i) => sum += *i as f64,
                        AttrValue::SmallInt(i) => sum += *i as f64,
                        AttrValue::Float(f) => sum += *f as f64,
                        AttrValue::Null => continue, // Skip nulls
                        _ => {
                            return Err(crate::errors::GraphError::InvalidInput(format!(
                                "Cannot sum non-numeric value: {:?}",
                                value
                            )))
                        }
                    }
                }
                Ok(AttrValue::Float(sum as f32))
            }
            "mean" => {
                let mut sum = 0.0;
                let mut count = 0;
                for value in values {
                    match value {
                        AttrValue::Int(i) => {
                            sum += *i as f64;
                            count += 1;
                        }
                        AttrValue::SmallInt(i) => {
                            sum += *i as f64;
                            count += 1;
                        }
                        AttrValue::Float(f) => {
                            sum += *f as f64;
                            count += 1;
                        }
                        AttrValue::Null => continue, // Skip nulls
                        _ => {
                            return Err(crate::errors::GraphError::InvalidInput(format!(
                                "Cannot calculate mean of non-numeric value: {:?}",
                                value
                            )))
                        }
                    }
                }
                if count == 0 {
                    Ok(AttrValue::Null)
                } else {
                    Ok(AttrValue::Float((sum / count as f64) as f32))
                }
            }
            "min" => {
                let mut min_val = values[0];
                for &value in values.iter().skip(1) {
                    if self.compare_attr_values_for_sort(value, min_val) == std::cmp::Ordering::Less
                    {
                        min_val = value;
                    }
                }
                Ok(min_val.clone())
            }
            "max" => {
                let mut max_val = values[0];
                for &value in values.iter().skip(1) {
                    if self.compare_attr_values_for_sort(value, max_val)
                        == std::cmp::Ordering::Greater
                    {
                        max_val = value;
                    }
                }
                Ok(max_val.clone())
            }
            _ => Err(crate::errors::GraphError::InvalidInput(format!(
                "Unsupported aggregation function: {}",
                agg_func
            ))),
        }
    }

    /// Helper method to convert AttrValue to string for column naming
    fn attr_value_to_string(&self, value: &AttrValue) -> String {
        match value {
            AttrValue::Int(i) => i.to_string(),
            AttrValue::SmallInt(i) => i.to_string(),
            AttrValue::Float(f) => f.to_string(),
            AttrValue::Bool(b) => b.to_string(),
            AttrValue::Text(s) => s.clone(),
            AttrValue::CompactText(s) => s.as_str().to_string(),
            AttrValue::Null => "null".to_string(),
            _ => format!("{:?}", value),
        }
    }

    /// Rolling window operation on a specific column
    ///
    /// # Arguments
    /// * `column` - Column name to apply rolling operation to
    /// * `window` - Window size for rolling operations
    /// * `operation` - Function to apply to each window (e.g., "mean", "sum", "min", "max", "std")
    ///
    /// # Returns
    /// BaseArray with rolling operation results
    pub fn rolling(
        &self,
        column: &str,
        window: usize,
        operation: &str,
    ) -> GraphResult<BaseArray<AttrValue>> {
        let column_data = self.column(column).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column))
        })?;
        column_data.rolling(window, operation)
    }

    /// Expanding window operation on a specific column
    ///
    /// # Arguments
    /// * `column` - Column name to apply expanding operation to
    /// * `operation` - Function to apply to expanding window (e.g., "mean", "sum", "min", "max", "std")
    ///
    /// # Returns
    /// BaseArray with expanding operation results
    pub fn expanding(&self, column: &str, operation: &str) -> GraphResult<BaseArray<AttrValue>> {
        let column_data = self.column(column).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column))
        })?;
        column_data.expanding(operation)
    }

    /// Cumulative sum operation on a specific column
    ///
    /// # Arguments
    /// * `column` - Column name to apply cumulative sum to
    ///
    /// # Returns
    /// BaseArray with cumulative sum values
    pub fn cumsum(&self, column: &str) -> GraphResult<BaseArray<AttrValue>> {
        let column_data = self.column(column).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column))
        })?;
        column_data.cumsum()
    }

    /// Cumulative minimum operation on a specific column
    ///
    /// # Arguments
    /// * `column` - Column name to apply cumulative minimum to
    ///
    /// # Returns
    /// BaseArray with cumulative minimum values
    pub fn cummin(&self, column: &str) -> GraphResult<BaseArray<AttrValue>> {
        let column_data = self.column(column).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column))
        })?;
        column_data.cummin()
    }

    /// Cumulative maximum operation on a specific column
    ///
    /// # Arguments
    /// * `column` - Column name to apply cumulative maximum to
    ///
    /// # Returns
    /// BaseArray with cumulative maximum values
    pub fn cummax(&self, column: &str) -> GraphResult<BaseArray<AttrValue>> {
        let column_data = self.column(column).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column))
        })?;
        column_data.cummax()
    }

    /// Shift operation on a specific column
    ///
    /// # Arguments
    /// * `column` - Column name to shift
    /// * `periods` - Number of periods to shift (positive = shift right, negative = shift left)
    /// * `fill_value` - Value to use for filling gaps (default: Null)
    ///
    /// # Returns
    /// BaseArray with shifted values
    pub fn shift(
        &self,
        column: &str,
        periods: i32,
        fill_value: Option<AttrValue>,
    ) -> GraphResult<BaseArray<AttrValue>> {
        let column_data = self.column(column).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column))
        })?;
        column_data.shift(periods, fill_value)
    }

    /// Percentage change operation on a specific column
    ///
    /// # Arguments
    /// * `column` - Column name to compute percentage change for
    /// * `periods` - Number of periods to use for comparison (default: 1)
    ///
    /// # Returns
    /// BaseArray with percentage change values
    pub fn pct_change(
        &self,
        column: &str,
        periods: Option<usize>,
    ) -> GraphResult<BaseArray<AttrValue>> {
        let column_data = self.column(column).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column))
        })?;
        column_data.pct_change(periods)
    }

    /// Apply rolling window operations to all numeric columns
    ///
    /// # Arguments
    /// * `window` - Window size for rolling operations
    /// * `operation` - Function to apply to each window
    ///
    /// # Returns
    /// New table with rolling operations applied to all numeric columns
    pub fn rolling_all(&self, window: usize, operation: &str) -> GraphResult<Self> {
        let mut result_columns = HashMap::new();

        for column_name in &self.column_order {
            if let Some(column) = self.columns.get(column_name) {
                // Check if column is numeric
                let is_numeric = column
                    .data()
                    .iter()
                    .any(|val| matches!(val, AttrValue::Int(_) | AttrValue::Float(_)));

                if is_numeric {
                    let rolling_result = column.rolling(window, operation)?;
                    result_columns.insert(
                        format!("{}_rolling_{}_{}", column_name, window, operation),
                        rolling_result,
                    );
                } else {
                    // Keep non-numeric columns as-is
                    result_columns.insert(column_name.clone(), column.clone());
                }
            }
        }

        Self::from_columns(result_columns)
    }

    /// Apply expanding window operations to all numeric columns
    ///
    /// # Arguments
    /// * `operation` - Function to apply to expanding window
    ///
    /// # Returns
    /// New table with expanding operations applied to all numeric columns
    pub fn expanding_all(&self, operation: &str) -> GraphResult<Self> {
        let mut result_columns = HashMap::new();

        for column_name in &self.column_order {
            if let Some(column) = self.columns.get(column_name) {
                // Check if column is numeric
                let is_numeric = column
                    .data()
                    .iter()
                    .any(|val| matches!(val, AttrValue::Int(_) | AttrValue::Float(_)));

                if is_numeric {
                    let expanding_result = column.expanding(operation)?;
                    result_columns.insert(
                        format!("{}_expanding_{}", column_name, operation),
                        expanding_result,
                    );
                } else {
                    // Keep non-numeric columns as-is
                    result_columns.insert(column_name.clone(), column.clone());
                }
            }
        }

        Self::from_columns(result_columns)
    }

    /// Comprehensive data profiling and quality assessment
    ///
    /// # Returns
    /// BaseTable containing detailed statistics and quality metrics for each column
    pub fn profile(&self) -> GraphResult<Self> {
        let mut profile_data = HashMap::new();
        let mut column_names = Vec::new();
        let mut data_types = Vec::new();
        let mut non_null_counts = Vec::new();
        let mut null_counts = Vec::new();
        let mut null_percentages = Vec::new();
        let mut unique_counts = Vec::new();
        let mut cardinalities = Vec::new();
        let mut means = Vec::new();
        let mut medians = Vec::new();
        let mut stds = Vec::new();
        let mut mins = Vec::new();
        let mut maxs = Vec::new();
        let mut q25s = Vec::new();
        let mut q75s = Vec::new();

        for column_name in &self.column_order {
            if let Some(column) = self.columns.get(column_name) {
                column_names.push(AttrValue::Text(column_name.clone()));

                // Analyze data types and nulls
                let mut type_counts: HashMap<String, usize> = HashMap::new();
                let mut null_count = 0;
                let mut unique_values: std::collections::HashSet<String> =
                    std::collections::HashSet::new();

                for value in column.data() {
                    match value {
                        AttrValue::Null => null_count += 1,
                        AttrValue::Int(_) => {
                            *type_counts.entry("integer".to_string()).or_insert(0) += 1
                        }
                        AttrValue::SmallInt(_) => {
                            *type_counts.entry("small_integer".to_string()).or_insert(0) += 1
                        }
                        AttrValue::Float(_) => {
                            *type_counts.entry("float".to_string()).or_insert(0) += 1
                        }
                        AttrValue::Bool(_) => {
                            *type_counts.entry("boolean".to_string()).or_insert(0) += 1
                        }
                        AttrValue::Text(s) => {
                            *type_counts.entry("text".to_string()).or_insert(0) += 1;
                            unique_values.insert(s.clone());
                        }
                        AttrValue::CompactText(s) => {
                            *type_counts.entry("compact_text".to_string()).or_insert(0) += 1;
                            unique_values.insert(s.as_str().to_string());
                        }
                        _ => *type_counts.entry("other".to_string()).or_insert(0) += 1,
                    }

                    // Add to unique values for all types
                    unique_values.insert(format!("{:?}", value));
                }

                let total_count = column.len();
                let non_null_count = total_count - null_count;

                // Determine primary data type
                let primary_type = type_counts
                    .iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(type_name, _)| type_name.clone())
                    .unwrap_or_else(|| "unknown".to_string());

                data_types.push(AttrValue::Text(primary_type));
                non_null_counts.push(AttrValue::Int(non_null_count as i64));
                null_counts.push(AttrValue::Int(null_count as i64));
                null_percentages.push(AttrValue::Float(
                    (null_count as f32 / total_count as f32) * 100.0,
                ));
                unique_counts.push(AttrValue::Int(unique_values.len() as i64));
                cardinalities.push(AttrValue::Float(
                    unique_values.len() as f32 / total_count as f32,
                ));

                // Statistical analysis for numeric columns
                let numeric_values: Vec<f64> = column
                    .data()
                    .iter()
                    .filter_map(|val| match val {
                        AttrValue::Int(i) => Some(*i as f64),
                        AttrValue::SmallInt(i) => Some(*i as f64),
                        AttrValue::Float(f) => Some(*f as f64),
                        _ => None,
                    })
                    .collect();

                if !numeric_values.is_empty() {
                    let mean_val = numeric_values.iter().sum::<f64>() / numeric_values.len() as f64;
                    means.push(AttrValue::Float(mean_val as f32));

                    // Calculate median
                    let mut sorted_values = numeric_values.clone();
                    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median_val = if sorted_values.len() % 2 == 0 {
                        (sorted_values[sorted_values.len() / 2 - 1]
                            + sorted_values[sorted_values.len() / 2])
                            / 2.0
                    } else {
                        sorted_values[sorted_values.len() / 2]
                    };
                    medians.push(AttrValue::Float(median_val as f32));

                    // Calculate standard deviation
                    let variance = numeric_values
                        .iter()
                        .map(|x| (x - mean_val).powi(2))
                        .sum::<f64>()
                        / numeric_values.len() as f64;
                    stds.push(AttrValue::Float(variance.sqrt() as f32));

                    // Min and Max
                    let min_val = numeric_values.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_val = numeric_values
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    mins.push(AttrValue::Float(min_val as f32));
                    maxs.push(AttrValue::Float(max_val as f32));

                    // Quartiles
                    let q25_idx = (sorted_values.len() as f64 * 0.25) as usize;
                    let q75_idx = (sorted_values.len() as f64 * 0.75) as usize;
                    q25s.push(AttrValue::Float(
                        sorted_values.get(q25_idx).unwrap_or(&0.0).clone() as f32,
                    ));
                    q75s.push(AttrValue::Float(
                        sorted_values.get(q75_idx).unwrap_or(&0.0).clone() as f32,
                    ));
                } else {
                    // Non-numeric columns
                    means.push(AttrValue::Null);
                    medians.push(AttrValue::Null);
                    stds.push(AttrValue::Null);
                    mins.push(AttrValue::Null);
                    maxs.push(AttrValue::Null);
                    q25s.push(AttrValue::Null);
                    q75s.push(AttrValue::Null);
                }
            }
        }

        // Create profile table
        profile_data.insert(
            "column".to_string(),
            BaseArray::from_attr_values(column_names),
        );
        profile_data.insert("dtype".to_string(), BaseArray::from_attr_values(data_types));
        profile_data.insert(
            "non_null_count".to_string(),
            BaseArray::from_attr_values(non_null_counts),
        );
        profile_data.insert(
            "null_count".to_string(),
            BaseArray::from_attr_values(null_counts),
        );
        profile_data.insert(
            "null_percentage".to_string(),
            BaseArray::from_attr_values(null_percentages),
        );
        profile_data.insert(
            "unique_count".to_string(),
            BaseArray::from_attr_values(unique_counts),
        );
        profile_data.insert(
            "cardinality".to_string(),
            BaseArray::from_attr_values(cardinalities),
        );
        profile_data.insert("mean".to_string(), BaseArray::from_attr_values(means));
        profile_data.insert("median".to_string(), BaseArray::from_attr_values(medians));
        profile_data.insert("std".to_string(), BaseArray::from_attr_values(stds));
        profile_data.insert("min".to_string(), BaseArray::from_attr_values(mins));
        profile_data.insert("q25".to_string(), BaseArray::from_attr_values(q25s));
        profile_data.insert("q75".to_string(), BaseArray::from_attr_values(q75s));
        profile_data.insert("max".to_string(), BaseArray::from_attr_values(maxs));

        Self::from_columns(profile_data)
    }

    /// Detect outliers in numeric columns using IQR method
    ///
    /// # Arguments
    /// * `column` - Column name to analyze for outliers
    /// * `factor` - IQR multiplication factor (default: 1.5)
    ///
    /// # Returns
    /// BaseArray with boolean values indicating outliers
    pub fn check_outliers(
        &self,
        column: &str,
        factor: Option<f64>,
    ) -> GraphResult<BaseArray<AttrValue>> {
        let factor = factor.unwrap_or(1.5);

        let column_data = self.column(column).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", column))
        })?;

        // Extract numeric values
        let numeric_values: Vec<f64> = column_data
            .data()
            .iter()
            .filter_map(|val| match val {
                AttrValue::Int(i) => Some(*i as f64),
                AttrValue::SmallInt(i) => Some(*i as f64),
                AttrValue::Float(f) => Some(*f as f64),
                _ => None,
            })
            .collect();

        if numeric_values.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' contains no numeric values",
                column
            )));
        }

        // Calculate quartiles
        let mut sorted_values = numeric_values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1_idx = (sorted_values.len() as f64 * 0.25) as usize;
        let q3_idx = (sorted_values.len() as f64 * 0.75) as usize;

        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - factor * iqr;
        let upper_bound = q3 + factor * iqr;

        // Mark outliers
        let mut outlier_flags = Vec::new();
        for val in column_data.data() {
            let is_outlier = match val {
                AttrValue::Int(i) => {
                    let v = *i as f64;
                    v < lower_bound || v > upper_bound
                }
                AttrValue::SmallInt(i) => {
                    let v = *i as f64;
                    v < lower_bound || v > upper_bound
                }
                AttrValue::Float(f) => {
                    let v = *f as f64;
                    v < lower_bound || v > upper_bound
                }
                _ => false, // Non-numeric values are not outliers
            };
            outlier_flags.push(AttrValue::Bool(is_outlier));
        }

        Ok(BaseArray::from_attr_values(outlier_flags))
    }

    /// Validate table schema against expected schema
    ///
    /// # Arguments
    /// * `expected_columns` - Map of column names to expected data types
    ///
    /// # Returns
    /// BaseTable with validation results
    pub fn validate_schema(&self, expected_columns: &HashMap<String, String>) -> GraphResult<Self> {
        let mut validation_data = HashMap::new();
        let mut column_names = Vec::new();
        let mut expected_types = Vec::new();
        let mut actual_types = Vec::new();
        let mut type_matches = Vec::new();
        let mut issues = Vec::new();

        for (expected_col, expected_type) in expected_columns {
            column_names.push(AttrValue::Text(expected_col.clone()));
            expected_types.push(AttrValue::Text(expected_type.clone()));

            if let Some(column) = self.columns.get(expected_col) {
                // Determine actual type
                let mut type_counts: HashMap<String, usize> = HashMap::new();
                for value in column.data() {
                    let type_name = match value {
                        AttrValue::Null => "null",
                        AttrValue::Int(_) => "integer",
                        AttrValue::SmallInt(_) => "small_integer",
                        AttrValue::Float(_) => "float",
                        AttrValue::Bool(_) => "boolean",
                        AttrValue::Text(_) => "text",
                        AttrValue::CompactText(_) => "compact_text",
                        _ => "other",
                    };
                    *type_counts.entry(type_name.to_string()).or_insert(0) += 1;
                }

                let actual_type = type_counts
                    .iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(type_name, _)| type_name.clone())
                    .unwrap_or_else(|| "unknown".to_string());

                actual_types.push(AttrValue::Text(actual_type.clone()));

                let matches = actual_type == *expected_type;
                type_matches.push(AttrValue::Bool(matches));

                if matches {
                    issues.push(AttrValue::Text("OK".to_string()));
                } else {
                    issues.push(AttrValue::Text(format!(
                        "Type mismatch: expected {}, found {}",
                        expected_type, actual_type
                    )));
                }
            } else {
                actual_types.push(AttrValue::Text("missing".to_string()));
                type_matches.push(AttrValue::Bool(false));
                issues.push(AttrValue::Text("Column missing".to_string()));
            }
        }

        // Check for unexpected columns
        for actual_col in &self.column_order {
            if !expected_columns.contains_key(actual_col) {
                column_names.push(AttrValue::Text(actual_col.clone()));
                expected_types.push(AttrValue::Text("not_expected".to_string()));

                if let Some(column) = self.columns.get(actual_col) {
                    let mut type_counts: HashMap<String, usize> = HashMap::new();
                    for value in column.data() {
                        let type_name = match value {
                            AttrValue::Null => "null",
                            AttrValue::Int(_) => "integer",
                            AttrValue::SmallInt(_) => "small_integer",
                            AttrValue::Float(_) => "float",
                            AttrValue::Bool(_) => "boolean",
                            AttrValue::Text(_) => "text",
                            AttrValue::CompactText(_) => "compact_text",
                            _ => "other",
                        };
                        *type_counts.entry(type_name.to_string()).or_insert(0) += 1;
                    }

                    let actual_type = type_counts
                        .iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(type_name, _)| type_name.clone())
                        .unwrap_or_else(|| "unknown".to_string());

                    actual_types.push(AttrValue::Text(actual_type));
                } else {
                    actual_types.push(AttrValue::Text("unknown".to_string()));
                }

                type_matches.push(AttrValue::Bool(false));
                issues.push(AttrValue::Text("Unexpected column".to_string()));
            }
        }

        validation_data.insert(
            "column".to_string(),
            BaseArray::from_attr_values(column_names),
        );
        validation_data.insert(
            "expected_type".to_string(),
            BaseArray::from_attr_values(expected_types),
        );
        validation_data.insert(
            "actual_type".to_string(),
            BaseArray::from_attr_values(actual_types),
        );
        validation_data.insert(
            "type_match".to_string(),
            BaseArray::from_attr_values(type_matches),
        );
        validation_data.insert("issues".to_string(), BaseArray::from_attr_values(issues));

        Self::from_columns(validation_data)
    }
}

// File I/O Implementation
impl BaseTable {
    /// Export table to CSV file
    pub fn to_csv<P: AsRef<std::path::Path>>(&self, path: P) -> GraphResult<()> {
        let path = path.as_ref();
        let mut writer = csv::Writer::from_path(path).map_err(|e| {
            crate::errors::GraphError::InvalidInput(format!("Failed to create CSV writer: {}", e))
        })?;

        // Write headers
        writer.write_record(&self.column_order).map_err(|e| {
            crate::errors::GraphError::InvalidInput(format!("Failed to write CSV headers: {}", e))
        })?;

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
            writer.write_record(&record).map_err(|e| {
                crate::errors::GraphError::InvalidInput(format!(
                    "Failed to write CSV record at row {}: {}",
                    i, e
                ))
            })?;
        }

        writer.flush().map_err(|e| {
            crate::errors::GraphError::InvalidInput(format!("Failed to flush CSV writer: {}", e))
        })?;

        Ok(())
    }

    /// Import table from CSV file  
    pub fn from_csv<P: AsRef<std::path::Path>>(path: P) -> GraphResult<Self> {
        let path = path.as_ref();
        let mut reader = csv::Reader::from_path(path).map_err(|e| {
            crate::errors::GraphError::InvalidInput(format!(
                "Failed to read CSV file '{}': {}",
                path.display(),
                e
            ))
        })?;

        // Get headers
        let headers = reader.headers().map_err(|e| {
            crate::errors::GraphError::InvalidInput(format!("Failed to read CSV headers: {}", e))
        })?;
        let column_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();

        // Initialize column data vectors
        let mut column_data: std::collections::HashMap<String, Vec<crate::types::AttrValue>> =
            column_names
                .iter()
                .map(|name| (name.clone(), Vec::new()))
                .collect();

        // Read all records
        for (row_idx, result) in reader.records().enumerate() {
            let record = result.map_err(|e| {
                crate::errors::GraphError::InvalidInput(format!(
                    "Failed to read CSV record at row {}: {}",
                    row_idx + 1,
                    e
                ))
            })?;

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
            columns.insert(
                name,
                crate::storage::array::BaseArray::from_attr_values(data),
            );
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
        let mut file = File::create(path).map_err(|e| {
            crate::errors::GraphError::InvalidInput(format!(
                "Failed to create JSON file '{}': {}",
                path.display(),
                e
            ))
        })?;

        // Create JSON structure with metadata
        let mut json_data = serde_json::Map::new();

        // Add metadata
        json_data.insert(
            "columns".to_string(),
            serde_json::Value::Array(
                self.column_order
                    .iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            ),
        );
        json_data.insert(
            "nrows".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.nrows)),
        );

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
        let json_string = serde_json::to_string_pretty(&json_data).map_err(|e| {
            crate::errors::GraphError::InvalidInput(format!("Failed to serialize to JSON: {}", e))
        })?;

        file.write_all(json_string.as_bytes()).map_err(|e| {
            crate::errors::GraphError::InvalidInput(format!("Failed to write JSON file: {}", e))
        })?;

        Ok(())
    }

    /// Import table from JSON file
    pub fn from_json<P: AsRef<std::path::Path>>(path: P) -> GraphResult<Self> {
        use std::fs::File;
        use std::io::Read;

        let path = path.as_ref();
        let mut file = File::open(path).map_err(|e| {
            crate::errors::GraphError::InvalidInput(format!(
                "Failed to open JSON file '{}': {}",
                path.display(),
                e
            ))
        })?;

        let mut json_string = String::new();
        file.read_to_string(&mut json_string).map_err(|e| {
            crate::errors::GraphError::InvalidInput(format!("Failed to read JSON file: {}", e))
        })?;

        let json_data: serde_json::Value = serde_json::from_str(&json_string).map_err(|e| {
            crate::errors::GraphError::InvalidInput(format!("Failed to parse JSON: {}", e))
        })?;

        // Extract metadata
        let json_obj = json_data.as_object().ok_or_else(|| {
            crate::errors::GraphError::InvalidInput("JSON data must be an object".to_string())
        })?;

        let column_names: Vec<String> = json_obj
            .get("columns")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                crate::errors::GraphError::InvalidInput(
                    "Missing or invalid 'columns' field in JSON".to_string(),
                )
            })?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();

        let rows = json_obj
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                crate::errors::GraphError::InvalidInput(
                    "Missing or invalid 'data' field in JSON".to_string(),
                )
            })?;

        // Initialize column data
        let mut column_data: std::collections::HashMap<String, Vec<crate::types::AttrValue>> =
            column_names
                .iter()
                .map(|name| (name.clone(), Vec::new()))
                .collect();

        // Process each row
        for row_value in rows {
            let row_obj = row_value.as_object().ok_or_else(|| {
                crate::errors::GraphError::InvalidInput(
                    "Each data row must be an object".to_string(),
                )
            })?;

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
            columns.insert(
                name,
                crate::storage::array::BaseArray::from_attr_values(data),
            );
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
            }
            crate::types::AttrValue::CompactText(s) => {
                let text = s.as_str();
                if text.contains(',')
                    || text.contains('"')
                    || text.contains('\n')
                    || text.contains('\r')
                {
                    format!("\"{}\"", text.replace("\"", "\"\""))
                } else {
                    text.to_string()
                }
            }
            crate::types::AttrValue::Bool(b) => b.to_string(),
            crate::types::AttrValue::Null => String::new(),
            crate::types::AttrValue::FloatVec(v) => format!(
                "[{}]",
                v.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
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
            crate::types::AttrValue::Int(i) => {
                serde_json::Value::Number(serde_json::Number::from(*i))
            }
            crate::types::AttrValue::SmallInt(i) => {
                serde_json::Value::Number(serde_json::Number::from(*i))
            }
            crate::types::AttrValue::Float(f) => serde_json::Value::Number(
                serde_json::Number::from_f64(*f as f64).unwrap_or(serde_json::Number::from(0)),
            ),
            crate::types::AttrValue::Text(s) => serde_json::Value::String(s.clone()),
            crate::types::AttrValue::CompactText(s) => {
                serde_json::Value::String(s.as_str().to_string())
            }
            crate::types::AttrValue::Bool(b) => serde_json::Value::Bool(*b),
            crate::types::AttrValue::Null => serde_json::Value::Null,
            crate::types::AttrValue::FloatVec(v) => serde_json::Value::Array(
                v.iter()
                    .map(|f| {
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(*f as f64)
                                .unwrap_or(serde_json::Number::from(0)),
                        )
                    })
                    .collect(),
            ),
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
            }
            serde_json::Value::String(s) => crate::types::AttrValue::Text(s.clone()),
            serde_json::Value::Array(arr) => {
                // Try to convert to float vector
                let floats: Result<Vec<f32>, _> = arr
                    .iter()
                    .map(|v| v.as_f64().map(|f| f as f32).ok_or("Not a number"))
                    .collect();

                if let Ok(float_vec) = floats {
                    crate::types::AttrValue::FloatVec(float_vec)
                } else {
                    // Fallback to string representation
                    crate::types::AttrValue::Text(format!("{:?}", arr))
                }
            }
            serde_json::Value::Object(_) => {
                // Convert object to string representation
                crate::types::AttrValue::Text(value.to_string())
            }
        }
    }

    /// Group by columns and apply aggregations
    pub fn group_by_agg(
        &self,
        group_cols: &[String],
        agg_specs: HashMap<String, String>,
    ) -> GraphResult<Self> {
        if group_cols.is_empty() {
            return self.aggregate(agg_specs);
        }

        // Validate group columns exist
        for col_name in group_cols {
            if !self.columns.contains_key(col_name) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Group column '{}' not found in table",
                    col_name
                )));
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
    fn apply_aggregation(
        &self,
        row_indices: &[usize],
        col_name: &str,
        agg_func: &str,
    ) -> GraphResult<crate::types::AttrValue> {
        let column = self.columns.get(col_name).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!(
                "Column '{}' not found for aggregation",
                col_name
            ))
        })?;

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
                                if min_val.is_none()
                                    || self.compare_values(value, min_val.as_ref().unwrap()) < 0
                                {
                                    min_val = Some(value.clone());
                                }
                            }
                            crate::types::AttrValue::Float(f) => {
                                // Skip NaN values (common with meta nodes)
                                if !f.is_nan() {
                                    if min_val.is_none()
                                        || self.compare_values(value, min_val.as_ref().unwrap()) < 0
                                    {
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
                                if max_val.is_none()
                                    || self.compare_values(value, max_val.as_ref().unwrap()) > 0
                                {
                                    max_val = Some(value.clone());
                                }
                            }
                            crate::types::AttrValue::Float(f) => {
                                // Skip NaN values (common with meta nodes)
                                if !f.is_nan() {
                                    if max_val.is_none()
                                        || self.compare_values(value, max_val.as_ref().unwrap()) > 0
                                    {
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
            _ => Err(crate::errors::GraphError::InvalidInput(format!(
                "Unknown aggregation function: {}",
                agg_func
            ))),
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
            result_columns.insert(
                col_name.clone(),
                BaseArray::from_attr_values(vec![agg_result]),
            );
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
            (AttrValue::Float(a), AttrValue::Float(b)) => {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) as i32
            }
            (AttrValue::Int(a), AttrValue::Float(b)) => (*a as f32)
                .partial_cmp(b)
                .unwrap_or(std::cmp::Ordering::Equal)
                as i32,
            (AttrValue::Float(a), AttrValue::Int(b)) => {
                a.partial_cmp(&(*b as f32))
                    .unwrap_or(std::cmp::Ordering::Equal) as i32
            }
            _ => 0, // Equal for non-comparable types
        }
    }

    /// Select specific rows by indices
    pub fn select_rows(&self, row_indices: &[usize]) -> GraphResult<Self> {
        let mut new_columns = HashMap::new();

        // Delegate to BaseArray take_indices operation for each column
        for (col_name, column) in &self.columns {
            let selected_column = column.take_indices(row_indices)?;
            new_columns.insert(col_name.clone(), selected_column);
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
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Left join column '{}' not found in table",
                left_on
            )));
        }
        if !other.has_column(right_on) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Right join column '{}' not found in other table",
                right_on
            )));
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
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Left join column '{}' not found in table",
                left_on
            )));
        }
        if !other.has_column(right_on) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Right join column '{}' not found in other table",
                right_on
            )));
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

    /// Right join with another table on specified columns
    /// Returns all rows from the right table, with matching rows from the left table
    pub fn right_join(&self, other: &Self, left_on: &str, right_on: &str) -> GraphResult<Self> {
        // Right join is equivalent to left join with the tables swapped
        other.left_join(self, right_on, left_on)
    }

    /// Full outer join with another table on specified columns
    /// Returns all rows from both tables, with null values where no match exists
    pub fn outer_join(&self, other: &Self, left_on: &str, right_on: &str) -> GraphResult<Self> {
        // Validate join columns exist
        if !self.has_column(left_on) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Left join column '{}' not found in table",
                left_on
            )));
        }
        if !other.has_column(right_on) {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Right join column '{}' not found in other table",
                right_on
            )));
        }

        let left_col = self.column(left_on).unwrap();
        let right_col = other.column(right_on).unwrap();

        // Build indexes for both tables
        let mut left_index: HashMap<String, Vec<usize>> = HashMap::new();
        for i in 0..self.nrows {
            if let Some(value) = left_col.get(i) {
                let key = format!("{:?}", value);
                left_index.entry(key).or_insert_with(Vec::new).push(i);
            }
        }

        let mut right_index: HashMap<String, Vec<usize>> = HashMap::new();
        for i in 0..other.nrows {
            if let Some(value) = right_col.get(i) {
                let key = format!("{:?}", value);
                right_index.entry(key).or_insert_with(Vec::new).push(i);
            }
        }

        // Collect all unique keys from both tables
        let mut all_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
        for key in left_index.keys() {
            all_keys.insert(key.clone());
        }
        for key in right_index.keys() {
            all_keys.insert(key.clone());
        }

        // Create result rows
        let mut result_rows = Vec::new();

        let empty_vec = Vec::new();
        for key in all_keys {
            let left_indices = left_index.get(&key).unwrap_or(&empty_vec);
            let right_indices = right_index.get(&key).unwrap_or(&empty_vec);

            if left_indices.is_empty() {
                // Only right matches - add rows with null left values
                for &right_idx in right_indices {
                    result_rows.push((None, Some(right_idx)));
                }
            } else if right_indices.is_empty() {
                // Only left matches - add rows with null right values
                for &left_idx in left_indices {
                    result_rows.push((Some(left_idx), None));
                }
            } else {
                // Both sides have matches - cartesian product
                for &left_idx in left_indices {
                    for &right_idx in right_indices {
                        result_rows.push((Some(left_idx), Some(right_idx)));
                    }
                }
            }
        }

        // Build column order: left columns + right columns (avoiding duplicates)
        let mut column_order = self.column_order.clone();
        for col in &other.column_order {
            if col != right_on || left_on != right_on {
                let mut final_name = col.clone();
                if self.has_column(col) && col != right_on {
                    final_name = format!("{}_right", col);
                }
                column_order.push(final_name);
            }
        }

        // Create result columns
        let mut result_columns = HashMap::new();

        // Add left table columns
        for col_name in &self.column_order {
            let column = self.column(col_name).unwrap();
            let mut new_values = Vec::with_capacity(result_rows.len());

            for (left_idx_opt, _) in &result_rows {
                if let Some(left_idx) = left_idx_opt {
                    if let Some(value) = column.get(*left_idx) {
                        new_values.push(value.clone());
                    } else {
                        new_values.push(crate::types::AttrValue::Null);
                    }
                } else {
                    new_values.push(crate::types::AttrValue::Null); // No left match
                }
            }
            result_columns.insert(col_name.clone(), BaseArray::from_attr_values(new_values));
        }

        // Add right table columns
        for col_name in &other.column_order {
            if col_name == right_on && left_on == right_on {
                continue; // Skip duplicate join column
            }

            let column = other.column(col_name).unwrap();
            let mut final_name = col_name.clone();
            if self.has_column(col_name) && col_name != right_on {
                final_name = format!("{}_right", col_name);
            }

            let mut new_values = Vec::with_capacity(result_rows.len());

            for (_, right_idx_opt) in &result_rows {
                if let Some(right_idx) = right_idx_opt {
                    if let Some(value) = column.get(*right_idx) {
                        new_values.push(value.clone());
                    } else {
                        new_values.push(crate::types::AttrValue::Null);
                    }
                } else {
                    new_values.push(crate::types::AttrValue::Null); // No right match
                }
            }
            result_columns.insert(final_name, BaseArray::from_attr_values(new_values));
        }

        let mut result = BaseTable::from_columns(result_columns)?;
        result.column_order = column_order;
        Ok(result)
    }

    /// Cross join (Cartesian product) with another table
    /// Returns all possible combinations of rows from both tables
    pub fn cross_join(&self, other: &Self) -> GraphResult<Self> {
        let result_size = self.nrows * other.nrows;

        // Build column order: left columns + right columns (handling name conflicts)
        let mut column_order = self.column_order.clone();
        for col in &other.column_order {
            let mut final_name = col.clone();
            if self.has_column(col) {
                final_name = format!("{}_right", col);
            }
            column_order.push(final_name);
        }

        // Create result columns
        let mut result_columns = HashMap::new();

        // Add left table columns (each value repeated for all right rows)
        for col_name in &self.column_order {
            let column = self.column(col_name).unwrap();
            let mut new_values = Vec::with_capacity(result_size);

            for left_idx in 0..self.nrows {
                let value = column
                    .get(left_idx)
                    .cloned()
                    .unwrap_or(crate::types::AttrValue::Null);
                // Repeat this value for all right table rows
                for _ in 0..other.nrows {
                    new_values.push(value.clone());
                }
            }
            result_columns.insert(col_name.clone(), BaseArray::from_attr_values(new_values));
        }

        // Add right table columns (cycle through all values for each left row)
        for col_name in &other.column_order {
            let column = other.column(col_name).unwrap();
            let mut final_name = col_name.clone();
            if self.has_column(col_name) {
                final_name = format!("{}_right", col_name);
            }

            let mut new_values = Vec::with_capacity(result_size);

            for _ in 0..self.nrows {
                // For each left row, add all right values
                for right_idx in 0..other.nrows {
                    let value = column
                        .get(right_idx)
                        .cloned()
                        .unwrap_or(crate::types::AttrValue::Null);
                    new_values.push(value);
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
                "Union requires tables with identical column schemas".to_string(),
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
        result
            .column_order
            .extend(other.column_order.iter().map(|name| {
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
                "Intersect requires tables with identical column schemas".to_string(),
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

    /// Generate descriptive statistics for all numeric columns
    ///
    /// Returns a summary table with statistics like count, mean, std, min, max, etc.
    /// Similar to pandas.DataFrame.describe()
    ///
    /// Creates a table where:
    /// - Columns are the original numeric column names
    /// - Rows are statistics (count, mean, std, min, 25%, 50%, 75%, max)
    pub fn describe(&self) -> GraphResult<BaseTable> {
        use std::collections::HashMap;

        // Find all numeric columns first
        let numeric_columns: Vec<String> = self
            .column_order
            .iter()
            .filter(|col_name| {
                if let Some(column) = self.columns.get(*col_name) {
                    column.data().iter().any(|v| {
                        matches!(
                            v,
                            AttrValue::Int(_) | AttrValue::SmallInt(_) | AttrValue::Float(_)
                        )
                    })
                } else {
                    false
                }
            })
            .cloned()
            .collect();

        // If no numeric columns, return empty table
        if numeric_columns.is_empty() {
            return Ok(BaseTable::new());
        }

        // Statistics to calculate
        let stats = vec!["count", "mean", "std", "min", "25%", "50%", "75%", "max"];

        // Initialize result columns - one column per original numeric column
        let mut result_columns: HashMap<String, Vec<AttrValue>> = HashMap::new();
        for col_name in &numeric_columns {
            result_columns.insert(col_name.clone(), Vec::new());
        }

        // Calculate statistics for each numeric column
        for stat in &stats {
            for col_name in &numeric_columns {
                if let Some(column) = self.columns.get(col_name) {
                    let stats_result = self.calculate_column_statistics(column)?;

                    let stat_value = match stat.as_ref() {
                        "count" => AttrValue::Float(stats_result.count as f32),
                        "mean" => AttrValue::Float(stats_result.mean as f32),
                        "std" => AttrValue::Float(stats_result.std as f32),
                        "min" => stats_result.min,
                        "25%" => stats_result.q25,
                        "50%" => stats_result.q50,
                        "75%" => stats_result.q75,
                        "max" => stats_result.max,
                        _ => AttrValue::Null,
                    };

                    result_columns.get_mut(col_name).unwrap().push(stat_value);
                }
            }
        }

        // Convert to BaseArray columns
        let mut base_columns = HashMap::new();
        for (col_name, values) in result_columns {
            if !values.is_empty() {
                base_columns.insert(col_name, BaseArray::from_attr_values(values));
            }
        }

        // Create result table
        let mut result = BaseTable::from_columns(base_columns)?;
        result.column_order = numeric_columns;

        Ok(result)
    }

    /// Helper function to calculate statistics for a single column
    fn calculate_column_statistics(
        &self,
        column: &BaseArray<AttrValue>,
    ) -> GraphResult<ColumnStatistics> {
        // Extract numeric values
        let mut numeric_values: Vec<f64> = Vec::new();

        for value in column.data() {
            match value {
                AttrValue::Int(i) => numeric_values.push(*i as f64),
                AttrValue::SmallInt(i) => numeric_values.push(*i as f64),
                AttrValue::Float(f) => numeric_values.push(*f as f64),
                AttrValue::Null => {} // Skip nulls
                _ => {}               // Skip non-numeric
            }
        }

        if numeric_values.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(
                "No numeric values found in column".to_string(),
            ));
        }

        // Sort for percentiles
        numeric_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate basic statistics
        let count = numeric_values.len();
        let sum: f64 = numeric_values.iter().sum();
        let mean = sum / count as f64;

        // Calculate standard deviation
        let variance = numeric_values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / count as f64;
        let std = variance.sqrt();

        // Calculate percentiles
        let min = numeric_values[0];
        let max = numeric_values[count - 1];
        let q25 = Self::percentile(&numeric_values, 0.25);
        let q50 = Self::percentile(&numeric_values, 0.50); // median
        let q75 = Self::percentile(&numeric_values, 0.75);

        Ok(ColumnStatistics {
            count,
            mean,
            std,
            min: AttrValue::Float(min as f32),
            q25: AttrValue::Float(q25 as f32),
            q50: AttrValue::Float(q50 as f32),
            q75: AttrValue::Float(q75 as f32),
            max: AttrValue::Float(max as f32),
        })
    }

    /// Calculate percentile from sorted numeric values
    fn percentile(sorted_values: &[f64], p: f64) -> f64 {
        let n = sorted_values.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return sorted_values[0];
        }

        let index = p * (n - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted_values[lower]
        } else {
            let weight = index - lower as f64;
            sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
        }
    }

    // ==================================================================================
    // MISSING VALUE HANDLING METHODS
    // ==================================================================================

    /// Remove rows with any null values
    /// Returns a new table with rows containing null values removed
    /// Similar to pandas DataFrame.dropna()
    pub fn dropna(&self) -> GraphResult<BaseTable> {
        // Create a mask for rows that have no null values
        let mut row_mask = vec![true; self.nrows];

        // Check each column for null values
        for (_, column) in &self.columns {
            for (row_idx, value) in column.data().iter().enumerate() {
                if matches!(value, AttrValue::Null) && row_idx < row_mask.len() {
                    row_mask[row_idx] = false;
                }
            }
        }

        // Filter all columns using the row mask
        let mut filtered_columns = HashMap::new();
        for (col_name, column) in &self.columns {
            let filtered_data: Vec<AttrValue> = column
                .data()
                .iter()
                .zip(row_mask.iter())
                .filter_map(|(value, &keep)| if keep { Some(value.clone()) } else { None })
                .collect();

            filtered_columns.insert(col_name.clone(), BaseArray::from_attr_values(filtered_data));
        }

        let mut result = BaseTable::from_columns(filtered_columns)?;
        result.column_order = self.column_order.clone();
        Ok(result)
    }

    /// Remove rows with null values in specified columns
    /// Similar to pandas DataFrame.dropna(subset=['col1', 'col2'])
    pub fn dropna_subset(&self, subset: &[&str]) -> GraphResult<BaseTable> {
        // Create a mask for rows that have no null values in specified columns
        let mut row_mask = vec![true; self.nrows];

        // Check only the specified columns for null values
        for col_name in subset {
            if let Some(column) = self.columns.get(*col_name) {
                for (row_idx, value) in column.data().iter().enumerate() {
                    if matches!(value, AttrValue::Null) && row_idx < row_mask.len() {
                        row_mask[row_idx] = false;
                    }
                }
            }
        }

        // Filter all columns using the row mask
        let mut filtered_columns = HashMap::new();
        for (col_name, column) in &self.columns {
            let filtered_data: Vec<AttrValue> = column
                .data()
                .iter()
                .zip(row_mask.iter())
                .filter_map(|(value, &keep)| if keep { Some(value.clone()) } else { None })
                .collect();

            filtered_columns.insert(col_name.clone(), BaseArray::from_attr_values(filtered_data));
        }

        let mut result = BaseTable::from_columns(filtered_columns)?;
        result.column_order = self.column_order.clone();
        Ok(result)
    }

    /// Detect missing values in the entire table
    /// Returns a new table of the same shape with boolean values indicating null positions
    /// Similar to pandas DataFrame.isna()
    pub fn isna(&self) -> GraphResult<BaseTable> {
        let mut null_mask_columns = HashMap::new();

        for (col_name, column) in &self.columns {
            let null_mask: Vec<AttrValue> = column
                .data()
                .iter()
                .map(|val| AttrValue::Bool(matches!(val, AttrValue::Null)))
                .collect();

            null_mask_columns.insert(col_name.clone(), BaseArray::from_attr_values(null_mask));
        }

        let mut result = BaseTable::from_columns(null_mask_columns)?;
        result.column_order = self.column_order.clone();
        Ok(result)
    }

    /// Detect non-missing values in the entire table
    /// Returns a new table of the same shape with boolean values indicating non-null positions
    /// Similar to pandas DataFrame.notna()
    pub fn notna(&self) -> GraphResult<BaseTable> {
        let mut not_null_mask_columns = HashMap::new();

        for (col_name, column) in &self.columns {
            let not_null_mask: Vec<AttrValue> = column
                .data()
                .iter()
                .map(|val| AttrValue::Bool(!matches!(val, AttrValue::Null)))
                .collect();

            not_null_mask_columns
                .insert(col_name.clone(), BaseArray::from_attr_values(not_null_mask));
        }

        let mut result = BaseTable::from_columns(not_null_mask_columns)?;
        result.column_order = self.column_order.clone();
        Ok(result)
    }

    /// Check if the table contains any null values
    pub fn has_nulls(&self) -> bool {
        for (_, column) in &self.columns {
            if column.has_nulls() {
                return true;
            }
        }
        false
    }

    /// Count null values in each column
    /// Returns a HashMap with column names and their null counts
    pub fn null_counts(&self) -> HashMap<String, usize> {
        let mut null_counts = HashMap::new();

        for (col_name, column) in &self.columns {
            null_counts.insert(col_name.clone(), column.null_count());
        }

        null_counts
    }

    /// Fill null values with specified values per column
    /// Returns a new table with nulls replaced by the fill values
    /// Similar to pandas DataFrame.fillna()
    pub fn fillna(&self, fill_values: HashMap<String, AttrValue>) -> GraphResult<BaseTable> {
        let mut filled_columns = HashMap::new();

        for (col_name, column) in &self.columns {
            let filled_column = if let Some(fill_value) = fill_values.get(col_name) {
                column.fillna(fill_value.clone())
            } else {
                column.clone()
            };

            filled_columns.insert(col_name.clone(), filled_column);
        }

        let mut result = BaseTable::from_columns(filled_columns)?;
        result.column_order = self.column_order.clone();
        Ok(result)
    }

    /// Fill null values with a single value for all columns
    pub fn fillna_all(&self, fill_value: AttrValue) -> GraphResult<BaseTable> {
        let mut filled_columns = HashMap::new();

        for (col_name, column) in &self.columns {
            filled_columns.insert(col_name.clone(), column.fillna(fill_value.clone()));
        }

        let mut result = BaseTable::from_columns(filled_columns)?;
        result.column_order = self.column_order.clone();
        Ok(result)
    }

    // ==================================================================================
    // GROUPBY OPERATIONS
    // ==================================================================================

    /// Group table by one or more columns, returning a TableArray
    /// This enables powerful fluent operations like groupby().sum(), groupby().agg()
    ///
    /// # Arguments
    /// * `by` - Column names to group by
    ///
    /// # Returns
    /// A TableArray where each table represents one group, with group keys attached
    ///
    /// # Examples
    /// ```rust
    /// // Group by single column
    /// let grouped = table.groupby(&["category"])?;
    /// let sums = grouped.sum()?;  // Sum each group
    ///
    /// // Group by multiple columns
    /// let grouped = table.groupby(&["category", "region"])?;
    /// let aggregated = grouped.agg(hashmap!{
    ///     "sales" => "sum",
    ///     "price" => "mean"
    /// })?;
    /// ```
    pub fn groupby(&self, by: &[&str]) -> GraphResult<super::TableArray> {
        use super::TableArray;

        // Validate that all groupby columns exist
        for col_name in by {
            if !self.columns.contains_key(*col_name) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Column '{}' not found in table",
                    col_name
                )));
            }
        }

        // Create group keys by iterating through rows
        let mut groups: HashMap<Vec<AttrValue>, Vec<usize>> = HashMap::new();

        for row_idx in 0..self.nrows {
            let mut key = Vec::new();

            // Build group key from specified columns
            for col_name in by {
                if let Some(column) = self.columns.get(*col_name) {
                    if let Some(value) = column.get(row_idx) {
                        key.push(value.clone());
                    } else {
                        key.push(AttrValue::Null);
                    }
                }
            }

            groups.entry(key).or_insert_with(Vec::new).push(row_idx);
        }

        // Create tables for each group
        let mut group_tables = Vec::new();
        let mut group_keys = Vec::new();

        for (group_key, row_indices) in groups {
            // Create a table for this group by selecting the relevant rows
            let group_table = self.select_rows(&row_indices)?;

            // Create group key mapping
            let mut key_map = HashMap::new();
            for (i, col_name) in by.iter().enumerate() {
                key_map.insert(col_name.to_string(), group_key[i].clone());
            }

            group_tables.push(group_table);
            group_keys.push(key_map);
        }

        Ok(TableArray::from_tables_with_keys(group_tables, group_keys))
    }

    /// Group by single column (convenience method)
    pub fn groupby_single(&self, column: &str) -> GraphResult<super::TableArray> {
        self.groupby(&[column])
    }

    // ==================================================================================
    // PHASE 2.1: COLUMN MANAGEMENT OPERATIONS
    // ==================================================================================

    /// Rename columns using a mapping
    pub fn rename(&self, columns: HashMap<String, String>) -> GraphResult<Self> {
        let mut new_columns = HashMap::new();
        let mut new_column_order = Vec::new();

        // Process each column according to the mapping
        for col_name in &self.column_order {
            let new_name = columns
                .get(col_name)
                .cloned()
                .unwrap_or_else(|| col_name.clone());

            // Check for duplicate new names
            if new_columns.contains_key(&new_name) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Duplicate column name after rename: '{}'",
                    new_name
                )));
            }

            if let Some(column) = self.columns.get(col_name) {
                new_columns.insert(new_name.clone(), column.clone());
                new_column_order.push(new_name);
            }
        }

        let mut result = self.clone();
        result.columns = new_columns;
        result.column_order = new_column_order;
        result.version += 1;

        Ok(result)
    }

    /// Add prefix to all column names
    pub fn add_prefix(&self, prefix: &str) -> GraphResult<Self> {
        let mut new_columns = HashMap::new();
        let mut new_column_order = Vec::new();

        for col_name in &self.column_order {
            let new_name = format!("{}{}", prefix, col_name);

            if let Some(column) = self.columns.get(col_name) {
                new_columns.insert(new_name.clone(), column.clone());
                new_column_order.push(new_name);
            }
        }

        let mut result = self.clone();
        result.columns = new_columns;
        result.column_order = new_column_order;
        result.version += 1;

        Ok(result)
    }

    /// Add suffix to all column names
    pub fn add_suffix(&self, suffix: &str) -> GraphResult<Self> {
        let mut new_columns = HashMap::new();
        let mut new_column_order = Vec::new();

        for col_name in &self.column_order {
            let new_name = format!("{}{}", col_name, suffix);

            if let Some(column) = self.columns.get(col_name) {
                new_columns.insert(new_name.clone(), column.clone());
                new_column_order.push(new_name);
            }
        }

        let mut result = self.clone();
        result.columns = new_columns;
        result.column_order = new_column_order;
        result.version += 1;

        Ok(result)
    }

    /// Reorder columns according to specified order
    pub fn reorder_columns(&self, new_order: Vec<String>) -> GraphResult<Self> {
        // Validate that all columns in new_order exist
        for col_name in &new_order {
            if !self.columns.contains_key(col_name) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Column '{}' does not exist",
                    col_name
                )));
            }
        }

        // Check for duplicates in new_order
        let mut seen = std::collections::HashSet::new();
        for col_name in &new_order {
            if !seen.insert(col_name) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Duplicate column name in reorder: '{}'",
                    col_name
                )));
            }
        }

        // Check that all existing columns are included
        if new_order.len() != self.column_order.len() {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "New order must include all {} columns, got {}",
                self.column_order.len(),
                new_order.len()
            )));
        }

        let mut result = self.clone();
        result.column_order = new_order;
        result.version += 1;

        Ok(result)
    }

    // ==================================================================================
    // PHASE 2.2: ROW OPERATIONS
    // ==================================================================================

    /// Append a single row to the table
    pub fn append_row(&self, row: HashMap<String, AttrValue>) -> GraphResult<Self> {
        // Validate that all columns in the table are present in the row
        for col_name in &self.column_order {
            if !row.contains_key(col_name) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Missing value for column '{}'",
                    col_name
                )));
            }
        }

        // Check for unexpected columns
        for key in row.keys() {
            if !self.columns.contains_key(key) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Unknown column '{}'",
                    key
                )));
            }
        }

        let mut new_columns = HashMap::new();
        for col_name in &self.column_order {
            // Delegate to BaseArray append_element operation
            let new_array = self.columns[col_name].append_element(row[col_name].clone());
            new_columns.insert(col_name.clone(), new_array);
        }

        let mut result = self.clone();
        result.columns = new_columns;
        result.nrows += 1;
        result.version += 1;

        Ok(result)
    }

    /// Extend table with multiple rows
    pub fn extend_rows(&self, rows: Vec<HashMap<String, AttrValue>>) -> GraphResult<Self> {
        if rows.is_empty() {
            return Ok(self.clone());
        }

        // Validate all rows first
        for (i, row) in rows.iter().enumerate() {
            for col_name in &self.column_order {
                if !row.contains_key(col_name) {
                    return Err(crate::errors::GraphError::InvalidInput(format!(
                        "Row {} missing value for column '{}'",
                        i, col_name
                    )));
                }
            }

            for key in row.keys() {
                if !self.columns.contains_key(key) {
                    return Err(crate::errors::GraphError::InvalidInput(format!(
                        "Row {} has unknown column '{}'",
                        i, key
                    )));
                }
            }
        }

        let mut new_columns = HashMap::new();
        for col_name in &self.column_order {
            // Collect all values for this column from the rows
            let new_values: Vec<AttrValue> = rows.iter().map(|row| row[col_name].clone()).collect();

            // Delegate to BaseArray extend_elements operation
            let new_array = self.columns[col_name].extend_elements(new_values);
            new_columns.insert(col_name.clone(), new_array);
        }

        let mut result = self.clone();
        result.columns = new_columns;
        result.nrows += rows.len();
        result.version += 1;

        Ok(result)
    }

    /// Drop rows by indices
    pub fn drop_rows(&self, indices: &[usize]) -> GraphResult<Self> {
        if indices.is_empty() {
            return Ok(self.clone());
        }

        // Validate indices
        for &idx in indices {
            if idx >= self.nrows {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Index {} out of range for table with {} rows",
                    idx, self.nrows
                )));
            }
        }

        let mut new_columns = HashMap::new();
        for col_name in &self.column_order {
            // Delegate to BaseArray drop_elements operation
            let new_array = self.columns[col_name].drop_elements(indices)?;
            new_columns.insert(col_name.clone(), new_array);
        }

        let new_nrows = self.nrows - indices.len();
        let mut result = self.clone();
        result.columns = new_columns;
        result.nrows = new_nrows;
        result.version += 1;

        Ok(result)
    }

    /// Drop duplicate rows based on specified columns (or all columns if none specified)
    pub fn drop_duplicates(&self, subset: Option<&[String]>) -> GraphResult<Self> {
        let columns_to_check = match subset {
            Some(cols) => {
                // Validate that all specified columns exist
                for col in cols {
                    if !self.columns.contains_key(col) {
                        return Err(crate::errors::GraphError::InvalidInput(format!(
                            "Column '{}' does not exist",
                            col
                        )));
                    }
                }
                cols.to_vec()
            }
            None => self.column_order.clone(),
        };

        let mut seen_rows = std::collections::HashSet::new();
        let mut keep_indices = Vec::new();

        for i in 0..self.nrows {
            // Create a tuple of values for the columns we're checking
            let mut row_key = Vec::new();
            for col_name in &columns_to_check {
                if let Some(val) = self.columns[col_name].get(i) {
                    // Convert AttrValue to a hashable representation
                    let key_part = match val {
                        AttrValue::Int(x) => format!("i:{}", x),
                        AttrValue::SmallInt(x) => format!("si:{}", x),
                        AttrValue::Float(x) => format!("f:{}", x),
                        AttrValue::Text(x) => format!("t:{}", x),
                        AttrValue::CompactText(x) => format!("ct:{:?}", x),
                        AttrValue::Bool(x) => format!("b:{}", x),
                        AttrValue::Null => "null".to_string(),
                        AttrValue::FloatVec(v) => format!("fv:{:?}", v),
                        AttrValue::Bytes(b) => format!("by:{:?}", b),
                        _ => format!("other:{:?}", val),
                    };
                    row_key.push(key_part);
                }
            }

            let row_signature = row_key.join("|");
            if seen_rows.insert(row_signature) {
                keep_indices.push(i);
            }
        }

        // If all rows are unique, return clone
        if keep_indices.len() == self.nrows {
            return Ok(self.clone());
        }

        // Delegate to select_rows which uses BaseArray take_indices
        let mut result = self.select_rows(&keep_indices)?;
        result.version += 1;

        Ok(result)
    }

    // ==================================================================================
    // PHASE 2: STREAMING FUNCTIONALITY (FOUNDATION ONLY - specialized types delegate)
    // ==================================================================================

    /// Launch interactive streaming table in browser (FOUNDATION ONLY)
    pub fn interactive(&self, _config: Option<InteractiveConfig>) -> GraphResult<VizModule> {
        use std::sync::Arc;

        // Create VizModule from this BaseTable following the delegation pattern
        // This provides unified visualization capabilities for all table types
        let data_source: Arc<dyn DataSource> = Arc::new(self.clone());
        let viz_module = VizModule::new(data_source);

        Ok(viz_module)
    }

    /// Generate embedded iframe HTML for Jupyter notebooks
    ///
    /// This method bridges table streaming to the unified viz/streaming infrastructure.
    /// It starts a WebSocket server and returns iframe HTML for embedding.
    ///
    /// # Arguments
    /// * `config` - Optional interactive configuration (port, theme, etc.)
    ///
    /// # Returns
    /// * `Ok(String)` - HTML iframe code for embedding
    /// * `Err(GraphError)` - If server failed to start
    ///
    /// # Examples
    /// ```rust
    /// let iframe_html = table.interactive_embed(None)?;
    /// // Use iframe_html in Jupyter notebook or web page
    /// ```
    pub fn interactive_embed(&mut self, config: Option<InteractiveConfig>) -> GraphResult<String> {
        use crate::viz::streaming::server::StreamingServer;
        use crate::viz::streaming::types::StreamingConfig;
        use std::sync::Arc;

        // Use provided config or default
        let config = config.unwrap_or_default();

        // Create data source from this table
        let data_source: Arc<dyn crate::viz::streaming::DataSource> = Arc::new(self.clone());

        // Configure streaming server
        let streaming_config = StreamingConfig {
            port: config.port,
            max_connections: 100,
            auto_broadcast: true,
            update_throttle_ms: 100,
            scroll_config: crate::viz::streaming::VirtualScrollConfig::default(),
        };

        // Create and start server
        let server = StreamingServer::new(data_source, streaming_config);

        // Start server with automatic port assignment if port=0
        let addr = std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1));
        let handle = server.start_background(addr, config.port).map_err(|e| {
            crate::errors::GraphError::InvalidInput(format!(
                "Failed to start streaming server: {}",
                e
            ))
        })?;

        let actual_port = handle.port;

        // Store handle to keep server alive
        self.active_server_handles.push(handle);

        // Generate iframe HTML
        let iframe_html = format!(
            r#"<iframe src="http://127.0.0.1:{port}/" width="100%" height="420" style="border:0;border-radius:12px;"></iframe>"#,
            port = actual_port
        );

        Ok(iframe_html)
    }

    /// Close all active streaming servers for this table
    pub fn close_streaming(&mut self) {
        self.active_server_handles.clear(); // Dropping handles stops servers
    }

    /// Convert AttrValue to a primitive AttrValue that serializes to simple JSON
    /// This function only returns Int, Float, Text, Bool, or Null - no complex enum variants
    fn attr_value_to_primitive_value(&self, value: &AttrValue) -> AttrValue {
        match value {
            // Simple primitive types - pass through directly
            AttrValue::Int(i) => AttrValue::Int(*i),
            AttrValue::Float(f) => AttrValue::Float(*f),
            AttrValue::Text(s) => AttrValue::Text(s.clone()),
            AttrValue::Bool(b) => AttrValue::Bool(*b),
            AttrValue::Null => AttrValue::Null,

            // Convert all other types to their string representation
            AttrValue::SmallInt(i) => AttrValue::Text(i.to_string()), // Convert to text to avoid enum variant
            AttrValue::CompactText(s) => AttrValue::Text(s.as_str().to_string()),
            AttrValue::FloatVec(v) => AttrValue::Text(format!(
                "[{}]",
                v.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            AttrValue::Bytes(b) => AttrValue::Text(format!("bytes[{}]", b.len())),
            AttrValue::CompressedText(cd) => match cd.decompress_text() {
                Ok(text) => AttrValue::Text(text),
                Err(_) => AttrValue::Text("[compressed text]".to_string()),
            },
            AttrValue::CompressedFloatVec(_) => {
                AttrValue::Text("[compressed float vec]".to_string())
            }
            AttrValue::SubgraphRef(id) => AttrValue::Text(format!("subgraph:{}", id)),
            AttrValue::NodeArray(nodes) => AttrValue::Text(format!("nodes[{}]", nodes.len())),
            AttrValue::EdgeArray(edges) => AttrValue::Text(format!("edges[{}]", edges.len())),
            AttrValue::IntVec(v) => AttrValue::Text(format!(
                "[{}]",
                v.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            AttrValue::TextVec(v) => AttrValue::Text(format!(
                "[{}]",
                v.iter()
                    .map(|s| format!("\"{}\"", s))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            AttrValue::BoolVec(v) => AttrValue::Text(format!(
                "[{}]",
                v.iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            AttrValue::Json(s) => AttrValue::Text(s.clone()),
        }
    }

    /// Convert AttrValue to a simple JSON-compatible value for frontend rendering (unused now)
    fn attr_value_to_display_value(&self, value: &AttrValue) -> serde_json::Value {
        match value {
            // Simple types - convert to JSON primitives
            AttrValue::Int(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
            AttrValue::SmallInt(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
            AttrValue::Float(f) => serde_json::Value::Number(
                serde_json::Number::from_f64(*f as f64).unwrap_or(serde_json::Number::from(0)),
            ),
            AttrValue::Text(s) => serde_json::Value::String(s.clone()),
            AttrValue::CompactText(s) => serde_json::Value::String(s.as_str().to_string()),
            AttrValue::Bool(b) => serde_json::Value::Bool(*b),
            AttrValue::Null => serde_json::Value::Null,

            // Complex types - convert to readable strings
            AttrValue::FloatVec(v) => serde_json::Value::String(format!(
                "[{}]",
                v.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            AttrValue::Bytes(b) => serde_json::Value::String(format!("bytes[{}]", b.len())),
            AttrValue::CompressedText(cd) => match cd.decompress_text() {
                Ok(text) => serde_json::Value::String(text),
                Err(_) => serde_json::Value::String("[compressed text]".to_string()),
            },
            AttrValue::CompressedFloatVec(_) => {
                serde_json::Value::String("[compressed float vec]".to_string())
            }
            AttrValue::SubgraphRef(id) => serde_json::Value::String(format!("subgraph:{}", id)),
            AttrValue::NodeArray(nodes) => {
                serde_json::Value::String(format!("nodes[{}]", nodes.len()))
            }
            AttrValue::EdgeArray(edges) => {
                serde_json::Value::String(format!("edges[{}]", edges.len()))
            }
            AttrValue::IntVec(v) => serde_json::Value::Array(
                v.iter()
                    .map(|&i| serde_json::Value::Number(i.into()))
                    .collect(),
            ),
            AttrValue::TextVec(v) => serde_json::Value::Array(
                v.iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            ),
            AttrValue::BoolVec(v) => {
                serde_json::Value::Array(v.iter().map(|&b| serde_json::Value::Bool(b)).collect())
            }
            AttrValue::Json(s) => {
                // Try to parse as JSON, fallback to string if invalid
                serde_json::from_str(s).unwrap_or_else(|_| serde_json::Value::String(s.clone()))
            }
        }
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

    fn get_window(&self, start: usize, count: usize) -> DataWindow {
        use crate::viz::streaming::data_source::DataWindowMetadata;

        let end = std::cmp::min(start + count, self.nrows);
        let actual_count = end.saturating_sub(start);

        if actual_count == 0 || start >= self.nrows {
            return DataWindow {
                headers: self.column_names().to_vec(),
                rows: vec![],
                schema: DataSchema {
                    columns: self
                        .column_order
                        .iter()
                        .map(|name| ColumnSchema {
                            name: name.clone(),
                            data_type: DataType::String,
                        })
                        .collect(),
                    primary_key: None,
                    source_type: "BaseTable".to_string(),
                },
                total_rows: self.nrows,
                start_offset: start,
                metadata: DataWindowMetadata {
                    created_at: std::time::SystemTime::now(),
                    is_cached: false,
                    load_time_ms: 0,
                    extra: HashMap::new(),
                },
            };
        }

        // Extract rows for the window, converting to string values to avoid AttrValue serialization issues
        let mut rows = Vec::with_capacity(actual_count);

        for row_idx in start..end {
            let mut row = Vec::with_capacity(self.column_order.len());

            for col_name in &self.column_order {
                if let Some(column) = self.columns.get(col_name) {
                    let attr_value = column.get(row_idx).cloned().unwrap_or(AttrValue::Null);
                    // Convert to string representation to bypass AttrValue enum serialization
                    let string_value = match &attr_value {
                        AttrValue::Int(i) => AttrValue::Text(i.to_string()),
                        AttrValue::SmallInt(i) => AttrValue::Text(i.to_string()),
                        AttrValue::Float(f) => AttrValue::Text(f.to_string()),
                        AttrValue::Bool(b) => AttrValue::Text(b.to_string()),
                        AttrValue::Text(s) => AttrValue::Text(s.clone()),
                        AttrValue::CompactText(s) => AttrValue::Text(s.as_str().to_string()),
                        AttrValue::Null => AttrValue::Text("".to_string()),
                        _ => AttrValue::Text(format!("{:?}", attr_value)), // Fallback for complex types
                    };
                    row.push(string_value);
                } else {
                    row.push(AttrValue::Text("".to_string())); // Empty string for missing columns
                }
            }

            rows.push(row);
        }

        DataWindow {
            headers: self.column_names().to_vec(),
            rows,
            schema: DataSchema {
                columns: self
                    .column_order
                    .iter()
                    .map(|name| {
                        let data_type = if let Some(column) = self.columns.get(name) {
                            self.infer_column_data_type(column)
                        } else {
                            DataType::String
                        };
                        ColumnSchema {
                            name: name.clone(),
                            data_type,
                        }
                    })
                    .collect(),
                primary_key: None,
                source_type: "BaseTable".to_string(),
            },
            total_rows: self.nrows,
            start_offset: start,
            metadata: DataWindowMetadata {
                created_at: std::time::SystemTime::now(),
                is_cached: false,
                load_time_ms: 0,
                extra: HashMap::new(),
            },
        }
    }

    fn get_schema(&self) -> DataSchema {
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

        DataSchema {
            columns,
            primary_key: None, // BaseTable doesn't enforce primary keys
            source_type: "BaseTable".to_string(),
        }
    }

    fn supports_streaming(&self) -> bool {
        true // BaseTable supports real-time streaming
    }

    fn get_column_types(&self) -> Vec<DataType> {
        self.column_order
            .iter()
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

    /// Get graph nodes representing table rows
    fn get_graph_nodes(&self) -> Vec<GraphNode> {
        let rows = self.nrows();
        (0..rows)
            .map(|i| GraphNode {
                id: i.to_string(),
                label: Some(format!("Row {}", i)),
                attributes: std::collections::HashMap::new(),
                position: None,
            })
            .collect()
    }

    /// Compute layout for table data visualization
    /// Creates a simple grid or circular layout based on table structure
    fn compute_layout(&self, algorithm: LayoutAlgorithm) -> Vec<NodePosition> {
        use crate::viz::streaming::data_source::{NodePosition, Position};

        let rows = self.nrows();
        if rows == 0 {
            return Vec::new();
        }

        // Create node positions for each row
        let nodes: Vec<_> = (0..rows)
            .map(|i| GraphNode {
                id: i.to_string(),
                label: Some(format!("Row {}", i)),
                attributes: std::collections::HashMap::new(),
                position: None,
            })
            .collect();

        match algorithm {
            LayoutAlgorithm::Circular {
                radius,
                start_angle,
            } => {
                let angle_step = 2.0 * std::f64::consts::PI / nodes.len() as f64;

                nodes
                    .into_iter()
                    .enumerate()
                    .map(|(i, node)| {
                        let angle = start_angle + (i as f64 * angle_step);
                        let actual_radius = radius.unwrap_or(100.0);
                        NodePosition {
                            node_id: node.id,
                            position: Position {
                                x: actual_radius * angle.cos(),
                                y: actual_radius * angle.sin(),
                            },
                        }
                    })
                    .collect()
            }
            LayoutAlgorithm::Grid { columns, cell_size } => nodes
                .into_iter()
                .enumerate()
                .map(|(i, node)| {
                    let row = i / columns;
                    let col = i % columns;
                    NodePosition {
                        node_id: node.id,
                        position: Position {
                            x: col as f64 * cell_size,
                            y: row as f64 * cell_size,
                        },
                    }
                })
                .collect(),
            _ => {
                // Default circular layout for other algorithms
                let radius = 200.0;
                let angle_step = 2.0 * std::f64::consts::PI / nodes.len() as f64;

                nodes
                    .into_iter()
                    .enumerate()
                    .map(|(i, node)| {
                        let angle = i as f64 * angle_step;
                        NodePosition {
                            node_id: node.id,
                            position: Position {
                                x: radius * angle.cos(),
                                y: radius * angle.sin(),
                            },
                        }
                    })
                    .collect()
            }
        }
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
            port: 0, // Use port 0 for automatic port assignment to avoid conflicts
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

/// Implement Index trait for BaseTable to enable table["column_name"] syntax
impl std::ops::Index<&str> for BaseTable {
    type Output = BaseArray<AttrValue>;

    fn index(&self, column_name: &str) -> &Self::Output {
        self.get_column(column_name)
            .unwrap_or_else(|| panic!("Column '{}' not found in table", column_name))
    }
}

/// Browser interface handle
#[derive(Debug)]
pub struct BrowserInterface {
    pub server_handle: crate::viz::streaming::types::ServerHandle,
    pub url: String,
    pub config: BrowserConfig,
}
