//! BaseTable - unified table implementation built on BaseArray columns

use super::traits::{Table, TableIterator};
use crate::storage::array::{BaseArray, ArrayOps};
use crate::errors::GraphResult;
use std::collections::HashMap;

/// Unified table implementation using BaseArray columns
/// This provides the foundation for all table types in the system
#[derive(Clone, Debug)]
pub struct BaseTable {
    /// Columns stored as BaseArrays
    columns: HashMap<String, BaseArray>,
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
    pub fn from_columns(columns: HashMap<String, BaseArray>) -> GraphResult<Self> {
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
        
        let column_order: Vec<String> = columns.keys().cloned().collect();
        
        Ok(Self {
            columns,
            column_order,
            nrows: first_len,
        })
    }
    
    /// Create a BaseTable with specific column order
    pub fn with_column_order(columns: HashMap<String, BaseArray>, column_order: Vec<String>) -> GraphResult<Self> {
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
    pub fn columns(&self) -> &HashMap<String, BaseArray> {
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
    
    fn column(&self, name: &str) -> Option<&BaseArray> {
        self.columns.get(name)
    }
    
    fn column_by_index(&self, index: usize) -> Option<&BaseArray> {
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
        // Simple predicate parsing - extend as needed
        // For now, support basic column comparisons like "column_name == value"
        let parts: Vec<&str> = predicate.split("==").map(|s| s.trim()).collect();
        if parts.len() != 2 {
            return Err(crate::errors::GraphError::InvalidInput(
                "Only '==' predicates supported currently".to_string()
            ));
        }
        
        let column_name = parts[0];
        let value = parts[1].trim_matches('"').trim_matches('\'');
        
        let filter_column = self.column(column_name)
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                format!("Column '{}' not found", column_name)
            ))?;
        
        // Get filter mask
        let mask = filter_column.eq_string(value);
        
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
    
    fn group_by(&self, _columns: &[String]) -> GraphResult<Vec<Self>> {
        // TODO: Implement grouping logic
        Err(crate::errors::GraphError::NotImplemented {
            feature: "group_by for BaseTable".to_string(),
            tracking_issue: None,
        })
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
    
    fn with_column(&self, name: String, column: BaseArray) -> GraphResult<Self> {
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