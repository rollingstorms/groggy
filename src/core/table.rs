//! GraphTable - Heterogeneous collection of GraphArrays (like pandas DataFrame)
//!
//! This module provides GraphTable as a collection of GraphArrays with mixed types,
//! implementing pandas-like functionality for graph data manipulation.
//!
//! # Design Principles
//! - GraphTable is a collection of GraphArrays (columns) with different types
//! - Supports row-based and column-based access patterns
//! - Lazy evaluation with intelligent caching
//! - Rich statistical operations and data manipulation
//! - Native integration with graph attributes and entities

use crate::types::{NodeId, EdgeId, AttrValue, AttrValueType};
use crate::errors::{GraphResult, GraphError};
use crate::core::array::GraphArray;
use std::collections::HashMap;
use std::fmt;

/// Metadata about the table's origin and properties
#[derive(Debug, Clone)]
pub struct TableMetadata {
    /// Source of the data ("nodes", "edges", "custom")
    pub source_type: String,
    /// Description or name of the table
    pub name: Option<String>,
    /// Creation timestamp
    pub created_at: Option<std::time::SystemTime>,
    /// Additional properties
    pub properties: HashMap<String, String>,
}

impl TableMetadata {
    pub fn new(source_type: String) -> Self {
        Self {
            source_type,
            name: None,
            created_at: Some(std::time::SystemTime::now()),
            properties: HashMap::new(),
        }
    }
    
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
}

/// Join types for table operations
#[derive(Debug, Clone, Copy)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Outer,
}

/// Aggregate operations for statistical functions
#[derive(Debug, Clone)]
pub enum AggregateOp {
    Sum,
    Mean,
    Count,
    Min,
    Max,
    Std,
    Var,
    First,
    Last,
    Unique,
}

/// Group-by result container
#[derive(Debug)]
pub struct GroupBy {
    groups: HashMap<AttrValue, Vec<usize>>, // group_key -> row_indices
    table: GraphTable,
    group_column: String,
}

impl GroupBy {
    /// Apply aggregation to each group
    pub fn agg(&self, ops: HashMap<String, AggregateOp>) -> GraphResult<GraphTable> {
        let mut result_columns = Vec::new();
        let mut result_column_names = Vec::new();
        
        // Group key column
        let group_keys: Vec<AttrValue> = self.groups.keys().cloned().collect();
        result_columns.push(GraphArray::from_vec(group_keys).with_name(self.group_column.clone()));
        result_column_names.push(self.group_column.clone());
        
        // Aggregate each specified column
        for (column_name, op) in ops {
            if let Some(column) = self.table.get_column_by_name(&column_name) {
                let mut agg_values = Vec::new();
                
                for group_indices in self.groups.values() {
                    let group_values: Vec<AttrValue> = group_indices.iter()
                        .filter_map(|&idx| column.get(idx).cloned())
                        .collect();
                    
                    let agg_result = match op {
                        AggregateOp::Sum => {
                            let sum: f32 = group_values.iter()
                                .filter_map(|v| v.as_float())
                                .sum();
                            AttrValue::Float(sum)
                        }
                        AggregateOp::Mean => {
                            let values: Vec<f32> = group_values.iter()
                                .filter_map(|v| v.as_float())
                                .collect();
                            if values.is_empty() {
                                AttrValue::Int(0)
                            } else {
                                AttrValue::Float(values.iter().sum::<f32>() / values.len() as f32)
                            }
                        }
                        AggregateOp::Count => AttrValue::Int(group_values.len() as i64),
                        AggregateOp::Min => {
                            group_values.iter().min().cloned().unwrap_or(AttrValue::Int(0))
                        }
                        AggregateOp::Max => {
                            group_values.iter().max().cloned().unwrap_or(AttrValue::Int(0))
                        }
                        _ => AttrValue::Int(0), // TODO: Implement other operations
                    };
                    
                    agg_values.push(agg_result);
                }
                
                let agg_column_name = format!("{}_{:?}", column_name, op).to_lowercase();
                result_columns.push(GraphArray::from_vec(agg_values).with_name(agg_column_name.clone()));
                result_column_names.push(agg_column_name);
            }
        }
        
        GraphTable::from_arrays_standalone(result_columns, Some(result_column_names))
    }
}

/// Heterogeneous collection of GraphArrays (pandas-like DataFrame)
#[derive(Debug, Clone)]
pub struct GraphTable {
    /// Columns stored as GraphArrays (mixed types allowed)
    columns: Vec<GraphArray>,
    /// Column names/labels
    column_names: Vec<String>,
    /// Optional row index/labels
    index: Option<GraphArray>,
    /// Table metadata
    metadata: TableMetadata,
    /// Reference to the source graph (optional)
    graph: Option<std::rc::Rc<crate::api::graph::Graph>>,
}

impl GraphTable {
    /// Create a new GraphTable from a collection of GraphArrays
    pub fn from_arrays(
        arrays: Vec<GraphArray>, 
        column_names: Option<Vec<String>>, 
        graph: Option<std::rc::Rc<crate::api::graph::Graph>>
    ) -> GraphResult<Self> {
        if arrays.is_empty() {
            return Err(GraphError::InvalidInput("Cannot create table from empty array list".to_string()));
        }
        
        // Check all arrays have the same length
        let expected_len = arrays[0].len();
        for (i, array) in arrays.iter().enumerate() {
            if array.len() != expected_len {
                return Err(GraphError::InvalidInput(format!("Array {} has length {} but expected {}", i, array.len(), expected_len)));
            }
        }
        
        // Generate column names if not provided
        let names = column_names.unwrap_or_else(|| {
            (0..arrays.len()).map(|i| format!("col_{}", i)).collect()
        });
        
        if names.len() != arrays.len() {
            return Err(GraphError::InvalidInput(format!("Expected {} column names, got {}", arrays.len(), names.len())));
        }
        
        Ok(Self {
            columns: arrays,
            column_names: names,
            index: None,
            metadata: TableMetadata::new("custom".to_string()),
            graph,
        })
    }
    
    /// Create a new GraphTable from arrays without a graph reference
    pub fn from_arrays_standalone(arrays: Vec<GraphArray>, column_names: Option<Vec<String>>) -> GraphResult<Self> {
        Self::from_arrays(arrays, column_names, None)
    }
    
    /// Create GraphTable from graph node attributes
    pub fn from_graph_nodes(
        graph: std::rc::Rc<crate::api::graph::Graph>,
        nodes: &[NodeId],
        attrs: Option<&[&str]>
    ) -> GraphResult<Self> {
        let mut columns = Vec::new();
        let mut column_names = Vec::new();
        
        // Always include node ID column
        let node_ids: Vec<AttrValue> = nodes.iter()
            .map(|&id| AttrValue::Int(id as i64))
            .collect();
        columns.push(GraphArray::from_vec(node_ids).with_name("id".to_string()));
        column_names.push("id".to_string());
        
        // Add attribute columns if specified
        if let Some(attr_names) = attrs {
            for &attr_name in attr_names {
                let attr_array = GraphArray::from_graph_attribute(&graph, attr_name, nodes)?;
                columns.push(attr_array.with_name(attr_name.to_string()));
                column_names.push(attr_name.to_string());
            }
        }
        
        let mut table = Self::from_arrays(columns, Some(column_names), Some(graph))?;
        table.metadata = TableMetadata::new("nodes".to_string());
        Ok(table)
    }
    
    /// Create GraphTable from graph edge attributes
    pub fn from_graph_edges(
        graph: std::rc::Rc<crate::api::graph::Graph>,
        edges: &[EdgeId],
        attrs: Option<&[&str]>
    ) -> GraphResult<Self> {
        let mut columns = Vec::new();
        let mut column_names = Vec::new();
        
        // Always include edge ID column
        let edge_ids: Vec<AttrValue> = edges.iter()
            .map(|&id| AttrValue::Int(id as i64))
            .collect();
        columns.push(GraphArray::from_vec(edge_ids).with_name("id".to_string()));
        column_names.push("id".to_string());
        
        // Add source and target columns
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        for &edge_id in edges {
            if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                sources.push(AttrValue::Int(source as i64));
                targets.push(AttrValue::Int(target as i64));
            } else {
                sources.push(AttrValue::Int(0));
                targets.push(AttrValue::Int(0));
            }
        }
        
        columns.push(GraphArray::from_vec(sources).with_name("source".to_string()));
        columns.push(GraphArray::from_vec(targets).with_name("target".to_string()));
        column_names.push("source".to_string());
        column_names.push("target".to_string());
        
        // Add attribute columns if specified
        if let Some(attr_names) = attrs {
            for &attr_name in attr_names {
                let attr_values: Vec<AttrValue> = edges.iter()
                    .map(|&edge_id| {
                        graph.get_edge_attr(edge_id, &attr_name.to_string())
                            .unwrap_or(Some(AttrValue::Int(0)))
                            .unwrap_or(AttrValue::Int(0))
                    })
                    .collect();
                
                columns.push(GraphArray::from_vec(attr_values).with_name(attr_name.to_string()));
                column_names.push(attr_name.to_string());
            }
        }
        
        let mut table = Self::from_arrays(columns, Some(column_names), Some(graph))?;
        table.metadata = TableMetadata::new("edges".to_string());
        Ok(table)
    }
    
    /// Get table dimensions
    pub fn shape(&self) -> (usize, usize) {
        if self.columns.is_empty() {
            (0, 0)
        } else {
            (self.columns[0].len(), self.columns.len())
        }
    }
    
    /// Get column names
    pub fn columns(&self) -> &[String] {
        &self.column_names
    }
    
    /// Get data types for each column
    pub fn dtypes(&self) -> HashMap<String, AttrValueType> {
        self.column_names.iter()
            .zip(self.columns.iter())
            .map(|(name, array)| (name.clone(), array.dtype()))
            .collect()
    }
    
    /// Get optional row index
    pub fn index(&self) -> Option<&GraphArray> {
        self.index.as_ref()
    }
    
    /// Set row index
    pub fn set_index(&mut self, index: GraphArray) -> GraphResult<()> {
        if index.len() != self.shape().0 {
            return Err(GraphError::InvalidInput(format!("Index length {} doesn't match table row count {}. Index must have same length as table rows", index.len(), self.shape().0)));
        }
        self.index = Some(index);
        Ok(())
    }
    
    /// Get metadata
    pub fn metadata(&self) -> &TableMetadata {
        &self.metadata
    }
    
    /// Set metadata
    pub fn set_metadata(&mut self, metadata: TableMetadata) {
        self.metadata = metadata;
    }
    
    /// Get optional graph reference
    pub fn graph(&self) -> Option<&std::rc::Rc<crate::api::graph::Graph>> {
        self.graph.as_ref()
    }
    
    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.columns.iter()
            .map(|col| col.len() * std::mem::size_of::<AttrValue>())
            .sum::<usize>()
            + self.index.as_ref().map_or(0, |idx| idx.len() * std::mem::size_of::<AttrValue>())
    }
    
    /// Get a column by name
    pub fn get_column_by_name(&self, name: &str) -> Option<&GraphArray> {
        self.column_names.iter()
            .position(|n| n == name)
            .and_then(|idx| self.columns.get(idx))
    }
    
    /// Get a column by index
    pub fn get_column(&self, idx: usize) -> Option<&GraphArray> {
        self.columns.get(idx)
    }
    
    /// Get a row as a HashMap (position-based)
    pub fn iloc(&self, row: usize) -> Option<HashMap<String, AttrValue>> {
        let (rows, _) = self.shape();
        if row >= rows {
            return None;
        }
        
        let mut row_data = HashMap::new();
        for (col_name, array) in self.column_names.iter().zip(self.columns.iter()) {
            if let Some(value) = array.get(row) {
                row_data.insert(col_name.clone(), value.clone());
            }
        }
        Some(row_data)
    }
    
    /// Get a row by index label (label-based)
    pub fn loc(&self, label: &AttrValue) -> Option<HashMap<String, AttrValue>> {
        if let Some(index_array) = &self.index {
            // Find position of label in index
            for (pos, index_value) in index_array.iter().enumerate() {
                if index_value == label {
                    return self.iloc(pos);
                }
            }
        }
        None
    }
    
    /// Select specific columns by name
    pub fn select(&self, column_names: &[&str]) -> GraphResult<GraphTable> {
        let mut selected_arrays = Vec::new();
        let mut selected_names = Vec::new();
        
        for &col_name in column_names {
            if let Some(array) = self.get_column_by_name(col_name) {
                selected_arrays.push(array.clone());
                selected_names.push(col_name.to_string());
            } else {
                return Err(GraphError::InvalidInput(format!("Column '{}' not found. Check available column names", col_name)));
            }
        }
        
        let mut table = Self::from_arrays(selected_arrays, Some(selected_names), self.graph.clone())?;
        table.index = self.index.clone();
        table.metadata = self.metadata.clone();
        Ok(table)
    }
    
    /// Filter rows based on a predicate
    pub fn filter_rows<F>(&self, predicate: F) -> GraphTable 
    where
        F: Fn(&HashMap<String, AttrValue>) -> bool,
    {
        let (rows, _) = self.shape();
        let mut filtered_indices = Vec::new();
        
        // Find rows that match the predicate
        for row_idx in 0..rows {
            if let Some(row_data) = self.iloc(row_idx) {
                if predicate(&row_data) {
                    filtered_indices.push(row_idx);
                }
            }
        }
        
        // Create new arrays with filtered values
        let filtered_arrays: Vec<GraphArray> = self.columns.iter()
            .map(|array| {
                let filtered_values: Vec<AttrValue> = filtered_indices.iter()
                    .filter_map(|&idx| array.get(idx).cloned())
                    .collect();
                GraphArray::from_vec(filtered_values)
            })
            .collect();
        
        let mut table = GraphTable {
            columns: filtered_arrays,
            column_names: self.column_names.clone(),
            index: None, // Reset index for filtered table
            metadata: self.metadata.clone(),
            graph: self.graph.clone(),
        };
        
        // Filter index if present
        if let Some(index_array) = &self.index {
            let filtered_index: Vec<AttrValue> = filtered_indices.iter()
                .filter_map(|&idx| index_array.get(idx).cloned())
                .collect();
            table.index = Some(GraphArray::from_vec(filtered_index));
        }
        
        table
    }
    
    /// Get first n rows
    pub fn head(&self, n: usize) -> GraphTable {
        let (rows, _) = self.shape();
        let limit = n.min(rows);
        
        let head_arrays: Vec<GraphArray> = self.columns.iter()
            .map(|array| {
                let head_values: Vec<AttrValue> = (0..limit)
                    .filter_map(|idx| array.get(idx).cloned())
                    .collect();
                GraphArray::from_vec(head_values)
            })
            .collect();
        
        let mut table = GraphTable {
            columns: head_arrays,
            column_names: self.column_names.clone(),
            index: None,
            metadata: self.metadata.clone(),
            graph: self.graph.clone(),
        };
        
        // Take head of index if present
        if let Some(index_array) = &self.index {
            let head_index: Vec<AttrValue> = (0..limit)
                .filter_map(|idx| index_array.get(idx).cloned())
                .collect();
            table.index = Some(GraphArray::from_vec(head_index));
        }
        
        table
    }
    
    /// Get last n rows
    pub fn tail(&self, n: usize) -> GraphTable {
        let (rows, _) = self.shape();
        let start = if n >= rows { 0 } else { rows - n };
        
        let tail_arrays: Vec<GraphArray> = self.columns.iter()
            .map(|array| {
                let tail_values: Vec<AttrValue> = (start..rows)
                    .filter_map(|idx| array.get(idx).cloned())
                    .collect();
                GraphArray::from_vec(tail_values)
            })
            .collect();
        
        let mut table = GraphTable {
            columns: tail_arrays,
            column_names: self.column_names.clone(),
            index: None,
            metadata: self.metadata.clone(),
            graph: self.graph.clone(),
        };
        
        // Take tail of index if present
        if let Some(index_array) = &self.index {
            let tail_index: Vec<AttrValue> = (start..rows)
                .filter_map(|idx| index_array.get(idx).cloned())
                .collect();
            table.index = Some(GraphArray::from_vec(tail_index));
        }
        
        table
    }
    
    /// Sort table by column
    /// todo: multi-column sort
    pub fn sort_by(&self, column: &str, ascending: bool) -> GraphResult<GraphTable> {
        let sort_array = self.get_column_by_name(column)
            .ok_or_else(|| GraphError::InvalidInput(format!("Column '{}' not found. Check available column names", column)))?;
        
        // Create indices and sort them based on the column values
        let (rows, _) = self.shape();
        let mut indices: Vec<usize> = (0..rows).collect();
        
        indices.sort_by(|&a, &b| {
            let val_a = sort_array.get(a);
            let val_b = sort_array.get(b);
            
            match (val_a, val_b) {
                (Some(a), Some(b)) => {
                    if ascending { a.cmp(b) } else { b.cmp(a) }
                }
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
        });
        
        // Create new arrays with sorted order
        let sorted_arrays: Vec<GraphArray> = self.columns.iter()
            .map(|array| {
                let sorted_values: Vec<AttrValue> = indices.iter()
                    .filter_map(|&idx| array.get(idx).cloned())
                    .collect();
                GraphArray::from_vec(sorted_values)
            })
            .collect();
        
        let mut table = GraphTable {
            columns: sorted_arrays,
            column_names: self.column_names.clone(),
            index: None,
            metadata: self.metadata.clone(),
            graph: self.graph.clone(),
        };
        
        // Sort index if present
        if let Some(index_array) = &self.index {
            let sorted_index: Vec<AttrValue> = indices.iter()
                .filter_map(|&idx| index_array.get(idx).cloned())
                .collect();
            table.index = Some(GraphArray::from_vec(sorted_index));
        }
        
        Ok(table)
    }
    
    /// Group by column
    pub fn group_by(&self, column: &str) -> GraphResult<GroupBy> {
        let group_array = self.get_column_by_name(column)
            .ok_or_else(|| GraphError::InvalidInput(format!("Column '{}' not found. Check available column names", column)))?;
        
        let mut groups = HashMap::new();
        
        // Group rows by unique values in the column
        for (row_idx, value) in group_array.iter().enumerate() {
            groups.entry(value.clone())
                .or_insert_with(Vec::new)
                .push(row_idx);
        }
        
        Ok(GroupBy {
            groups,
            table: self.clone(),
            group_column: column.to_string(),
        })
    }
    
    /// Aggregate all columns with specified operations
    pub fn aggregate(&self, ops: HashMap<String, AggregateOp>) -> GraphResult<HashMap<String, AttrValue>> {
        let mut results = HashMap::new();
        
        for (column_name, op) in ops {
            if let Some(array) = self.get_column_by_name(&column_name) {
                let result = match op {
                    AggregateOp::Sum => array.sum().map(|f| AttrValue::Float(f as f32)).unwrap_or(AttrValue::Int(0)),
                    AggregateOp::Mean => array.mean().map(|f| AttrValue::Float(f as f32)).unwrap_or(AttrValue::Int(0)),
                    AggregateOp::Count => AttrValue::Int(array.count() as i64),
                    AggregateOp::Min => array.min().unwrap_or(AttrValue::Int(0)),
                    AggregateOp::Max => array.max().unwrap_or(AttrValue::Int(0)),
                    AggregateOp::Std => array.std().map(|f| AttrValue::Float(f as f32)).unwrap_or(AttrValue::Int(0)),
                    _ => AttrValue::Int(0), // TODO: Implement other operations
                };
                results.insert(column_name, result);
            }
        }
        
        Ok(results)
    }
    
    /// Get summary statistics for all numeric columns
    pub fn describe(&self) -> GraphTable {
        let mut desc_arrays = Vec::new();
        let mut desc_columns = Vec::new();
        
        // Statistics to compute
        let stats = vec!["count", "mean", "std", "min", "max"];
        
        for stat_name in &stats {
            let mut stat_values = Vec::new();
            
            for array in &self.columns {
                let value = match *stat_name {
                    "count" => AttrValue::Int(array.count() as i64),
                    "mean" => array.mean().map(|f| AttrValue::Float(f as f32)).unwrap_or(AttrValue::Int(0)),
                    "std" => array.std().map(|f| AttrValue::Float(f as f32)).unwrap_or(AttrValue::Int(0)),
                    "min" => array.min().unwrap_or(AttrValue::Int(0)),
                    "max" => array.max().unwrap_or(AttrValue::Int(0)),
                    _ => AttrValue::Int(0),
                };
                stat_values.push(value);
            }
            
            desc_arrays.push(GraphArray::from_vec(stat_values).with_name(stat_name.to_string()));
            desc_columns.push(stat_name.to_string());
        }
        
        // Set column names as the index
        let column_names_array = GraphArray::from_vec(
            self.column_names.iter().map(|name| AttrValue::Text(name.clone())).collect()
        );
        
        let desc_table = GraphTable {
            columns: desc_arrays,
            column_names: self.column_names.clone(),
            index: Some(column_names_array),
            metadata: TableMetadata::new("describe".to_string()),
            graph: None, // Describe table doesn't need graph reference
        };
        
        desc_table
    }
    
    /// Convert to dictionary format
    pub fn to_dict(&self) -> HashMap<String, Vec<AttrValue>> {
        let mut dict = HashMap::new();
        
        for (name, array) in self.column_names.iter().zip(self.columns.iter()) {
            let values: Vec<AttrValue> = array.iter().cloned().collect();
            dict.insert(name.clone(), values);
        }
        
        dict
    }
    
    /// Convert back to arrays
    pub fn to_arrays(&self) -> Vec<GraphArray> {
        self.columns.clone()
    }
    
    /// Convert to JSON string
    pub fn to_json(&self) -> GraphResult<String> {
        let dict = self.to_dict();
        serde_json::to_string(&dict).map_err(|e| GraphError::SerializationError {
            data_type: "GraphTable".to_string(),
            operation: "serialize".to_string(),
            underlying_error: e.to_string(),
        })
    }
}

impl fmt::Display for GraphTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.shape();
        writeln!(f, "GraphTable ({} rows, {} columns)", rows, cols)?;
        writeln!(f, "Source: {}", self.metadata.source_type)?;
        
        // Show column names and types
        writeln!(f, "Columns:")?;
        for (name, array) in self.column_names.iter().zip(self.columns.iter()) {
            writeln!(f, "  {} ({:?})", name, array.dtype())?;
        }
        
        // Show first few rows
        let display_rows = std::cmp::min(rows, 5);
        if display_rows > 0 {
            writeln!(f, "\nData (first {} rows):", display_rows)?;
            
            // Header
            write!(f, "  ")?;
            for (i, name) in self.column_names.iter().enumerate() {
                if i > 0 { write!(f, " ")?; }
                write!(f, "{:>12}", name)?;
            }
            writeln!(f)?;
            
            // Rows
            for row in 0..display_rows {
                write!(f, "  ")?;
                for (col_idx, array) in self.columns.iter().enumerate() {
                    if col_idx > 0 { write!(f, " ")?; }
                    if let Some(value) = array.get(row) {
                        write!(f, "{:>12}", format!("{}", value))?;
                    } else {
                        write!(f, "{:>12}", "null")?;
                    }
                }
                writeln!(f)?;
            }
            
            if rows > display_rows {
                writeln!(f, "  ... ({} more rows)", rows - display_rows)?;
            }
        }
        
        Ok(())
    }
}