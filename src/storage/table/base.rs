//! BaseTable - unified table implementation built on BaseArray columns

use super::traits::{Table, TableIterator};
use crate::storage::array::{BaseArray, ArrayOps};
use crate::errors::GraphResult;
use crate::types::AttrValue;
use std::collections::HashMap;

/// Unified table implementation using BaseArray columns
/// This provides the foundation for all table types in the system
#[derive(Clone, Debug)]
pub struct BaseTable {
    /// Columns stored as BaseArrays with AttrValue type
    columns: HashMap<String, BaseArray<AttrValue>>,
    /// Column order for consistent iteration
    column_order: Vec<String>,
    /// Number of rows (derived from first column)
    nrows: usize,
}

impl BaseTable {
    /// Create a new empty BaseTable
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
            column_order: Vec::new(),
            nrows: 0,
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
        
        Ok(Self {
            columns: normalized_columns,
            column_order,
            nrows: max_len,
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
        
        Ok(Self {
            columns,
            column_order,
            nrows: first_len,
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
        })
    }
}

impl std::fmt::Display for BaseTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BaseTable[{} x {}]", self.nrows, self.ncols())?;
        
        if self.nrows == 0 {
            return Ok(());
        }
        
        // Show column headers
        write!(f, "| ")?;
        for col_name in &self.column_order {
            write!(f, "{:>10} | ", col_name)?;
        }
        writeln!(f)?;
        
        // Show separator
        write!(f, "|")?;
        for _ in &self.column_order {
            write!(f, "------------|")?;
        }
        writeln!(f)?;
        
        // Show first few rows
        let display_rows = std::cmp::min(5, self.nrows);
        for i in 0..display_rows {
            write!(f, "| ")?;
            for col_name in &self.column_order {
                if let Some(column) = self.columns.get(col_name) {
                    let value_str = if let Some(value) = column.get(i) {
                        format!("{}", value)
                    } else {
                        "None".to_string()
                    };
                    write!(f, "{:>10} | ", value_str)?;
                } else {
                    write!(f, "{:>10} | ", "ERROR")?;
                }
            }
            writeln!(f)?;
        }
        
        if self.nrows > display_rows {
            writeln!(f, "... ({} more rows)", self.nrows - display_rows)?;
        }
        
        Ok(())
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
}