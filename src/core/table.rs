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

use crate::core::array::GraphArray;
use crate::core::matrix::JoinType;
use crate::errors::{GraphError, GraphResult};
use crate::types::{AttrValue, AttrValueType, EdgeId, NodeId};
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
                    let group_values: Vec<AttrValue> = group_indices
                        .iter()
                        .filter_map(|&idx| column.get(idx).cloned())
                        .collect();

                    let agg_result = match op {
                        AggregateOp::Sum => {
                            let sum: f64 = group_values
                                .iter()
                                .filter_map(|v| match v {
                                    AttrValue::Int(i) => Some(*i as f64),
                                    AttrValue::SmallInt(i) => Some(*i as f64),
                                    AttrValue::Float(f) => Some(*f as f64),
                                    _ => None,
                                })
                                .sum();
                            AttrValue::Float(sum as f32)
                        }
                        AggregateOp::Mean => {
                            let values: Vec<f64> = group_values
                                .iter()
                                .filter_map(|v| match v {
                                    AttrValue::Int(i) => Some(*i as f64),
                                    AttrValue::SmallInt(i) => Some(*i as f64),
                                    AttrValue::Float(f) => Some(*f as f64),
                                    _ => None,
                                })
                                .collect();
                            if values.is_empty() {
                                AttrValue::Int(0)
                            } else {
                                AttrValue::Float(
                                    (values.iter().sum::<f64>() / values.len() as f64) as f32,
                                )
                            }
                        }
                        AggregateOp::Count => AttrValue::Int(group_values.len() as i64),
                        AggregateOp::Min => group_values
                            .iter()
                            .min()
                            .cloned()
                            .unwrap_or(AttrValue::Int(0)),
                        AggregateOp::Max => group_values
                            .iter()
                            .max()
                            .cloned()
                            .unwrap_or(AttrValue::Int(0)),
                        _ => AttrValue::Int(0), // TODO: Implement other operations
                    };

                    agg_values.push(agg_result);
                }

                let agg_column_name = format!("{}_{:?}", column_name, op).to_lowercase();
                result_columns
                    .push(GraphArray::from_vec(agg_values).with_name(agg_column_name.clone()));
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
        graph: Option<std::rc::Rc<crate::api::graph::Graph>>,
    ) -> GraphResult<Self> {
        if arrays.is_empty() {
            return Err(GraphError::InvalidInput(
                "Cannot create table from empty array list".to_string(),
            ));
        }

        // Check all arrays have the same length
        let expected_len = arrays[0].len();
        for (i, array) in arrays.iter().enumerate() {
            if array.len() != expected_len {
                return Err(GraphError::InvalidInput(format!(
                    "Array {} has length {} but expected {}",
                    i,
                    array.len(),
                    expected_len
                )));
            }
        }

        // Generate column names if not provided
        let names = column_names
            .unwrap_or_else(|| (0..arrays.len()).map(|i| format!("col_{}", i)).collect());

        if names.len() != arrays.len() {
            return Err(GraphError::InvalidInput(format!(
                "Expected {} column names, got {}",
                arrays.len(),
                names.len()
            )));
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
    pub fn from_arrays_standalone(
        arrays: Vec<GraphArray>,
        column_names: Option<Vec<String>>,
    ) -> GraphResult<Self> {
        Self::from_arrays(arrays, column_names, None)
    }

    /// Create GraphTable from graph node attributes
    pub fn from_graph_nodes(
        graph: std::rc::Rc<crate::api::graph::Graph>,
        nodes: &[NodeId],
        attrs: Option<&[&str]>,
    ) -> GraphResult<Self> {
        let mut columns = Vec::new();
        let mut column_names = Vec::new();

        // Always include node ID column
        let node_ids: Vec<AttrValue> = nodes.iter().map(|&id| AttrValue::Int(id as i64)).collect();
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
        attrs: Option<&[&str]>,
    ) -> GraphResult<Self> {
        let mut columns = Vec::new();
        let mut column_names = Vec::new();

        // Always include edge ID column
        let edge_ids: Vec<AttrValue> = edges.iter().map(|&id| AttrValue::Int(id as i64)).collect();
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
                let attr_values: Vec<AttrValue> = edges
                    .iter()
                    .map(|&edge_id| {
                        graph
                            .get_edge_attr(edge_id, &attr_name.to_string())
                            .unwrap_or(Some(AttrValue::Null))
                            .unwrap_or(AttrValue::Null)
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
        self.column_names
            .iter()
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
        self.columns
            .iter()
            .map(|col| col.len() * std::mem::size_of::<AttrValue>())
            .sum::<usize>()
            + self
                .index
                .as_ref()
                .map_or(0, |idx| idx.len() * std::mem::size_of::<AttrValue>())
    }

    /// Get a column by name
    pub fn get_column_by_name(&self, name: &str) -> Option<&GraphArray> {
        self.column_names
            .iter()
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
                return Err(GraphError::InvalidInput(format!(
                    "Column '{}' not found. Check available column names",
                    col_name
                )));
            }
        }

        let mut table =
            Self::from_arrays(selected_arrays, Some(selected_names), self.graph.clone())?;
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
        let filtered_arrays: Vec<GraphArray> = self
            .columns
            .iter()
            .map(|array| {
                let filtered_values: Vec<AttrValue> = filtered_indices
                    .iter()
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
            let filtered_index: Vec<AttrValue> = filtered_indices
                .iter()
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

        let head_arrays: Vec<GraphArray> = self
            .columns
            .iter()
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

        let tail_arrays: Vec<GraphArray> = self
            .columns
            .iter()
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
        let sort_array = self.get_column_by_name(column).ok_or_else(|| {
            GraphError::InvalidInput(format!(
                "Column '{}' not found. Check available column names",
                column
            ))
        })?;

        // Create indices and sort them based on the column values
        let (rows, _) = self.shape();
        let mut indices: Vec<usize> = (0..rows).collect();

        indices.sort_by(|&a, &b| {
            let val_a = sort_array.get(a);
            let val_b = sort_array.get(b);

            match (val_a, val_b) {
                (Some(a), Some(b)) => {
                    if ascending {
                        a.cmp(b)
                    } else {
                        b.cmp(a)
                    }
                }
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
        });

        // Create new arrays with sorted order
        let sorted_arrays: Vec<GraphArray> = self
            .columns
            .iter()
            .map(|array| {
                let sorted_values: Vec<AttrValue> = indices
                    .iter()
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
            let sorted_index: Vec<AttrValue> = indices
                .iter()
                .filter_map(|&idx| index_array.get(idx).cloned())
                .collect();
            table.index = Some(GraphArray::from_vec(sorted_index));
        }

        Ok(table)
    }

    /// Group by column
    pub fn group_by(&self, column: &str) -> GraphResult<GroupBy> {
        let group_array = self.get_column_by_name(column).ok_or_else(|| {
            GraphError::InvalidInput(format!(
                "Column '{}' not found. Check available column names",
                column
            ))
        })?;

        let mut groups = HashMap::new();

        // Group rows by unique values in the column
        for (row_idx, value) in group_array.iter().enumerate() {
            groups
                .entry(value.clone())
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
    pub fn aggregate(
        &self,
        ops: HashMap<String, AggregateOp>,
    ) -> GraphResult<HashMap<String, AttrValue>> {
        let mut results = HashMap::new();

        for (column_name, op) in ops {
            if let Some(array) = self.get_column_by_name(&column_name) {
                let result = match op {
                    AggregateOp::Sum => array
                        .sum()
                        .map(|f| AttrValue::Float(f as f32))
                        .unwrap_or(AttrValue::Int(0)),
                    AggregateOp::Mean => array
                        .mean()
                        .map(|f| AttrValue::Float(f as f32))
                        .unwrap_or(AttrValue::Int(0)),
                    AggregateOp::Count => AttrValue::Int(array.count() as i64),
                    AggregateOp::Min => array.min().unwrap_or(AttrValue::Int(0)),
                    AggregateOp::Max => array.max().unwrap_or(AttrValue::Int(0)),
                    AggregateOp::Std => array
                        .std()
                        .map(|f| AttrValue::Float(f as f32))
                        .unwrap_or(AttrValue::Int(0)),
                    _ => AttrValue::Int(0), // TODO: Implement other operations
                };
                results.insert(column_name, result);
            }
        }

        Ok(results)
    }

    /// Calculate mean for a specific column
    pub fn mean(&self, column_name: &str) -> GraphResult<f64> {
        let column = self.get_column_by_name(column_name).ok_or_else(|| {
            GraphError::InvalidInput(format!(
                "Column '{}' not found. Available columns: {:?}",
                column_name, self.column_names
            ))
        })?;

        column.mean().ok_or_else(|| {
            GraphError::InvalidInput(format!(
                "Cannot calculate mean for column '{}': contains non-numeric data",
                column_name
            ))
        })
    }

    /// Calculate sum for a specific column
    pub fn sum(&self, column_name: &str) -> GraphResult<f64> {
        let column = self.get_column_by_name(column_name).ok_or_else(|| {
            GraphError::InvalidInput(format!(
                "Column '{}' not found. Available columns: {:?}",
                column_name, self.column_names
            ))
        })?;

        column.sum().ok_or_else(|| {
            GraphError::InvalidInput(format!(
                "Cannot calculate sum for column '{}': contains non-numeric data",
                column_name
            ))
        })
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
                    "mean" => array
                        .mean()
                        .map(|f| AttrValue::Float(f as f32))
                        .unwrap_or(AttrValue::Int(0)),
                    "std" => array
                        .std()
                        .map(|f| AttrValue::Float(f as f32))
                        .unwrap_or(AttrValue::Int(0)),
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
            self.column_names
                .iter()
                .map(|name| AttrValue::Text(name.clone()))
                .collect(),
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

    /// Inner join with another table on specified columns
    pub fn inner_join(
        &self,
        other: &GraphTable,
        left_on: &str,
        right_on: &str,
    ) -> GraphResult<GraphTable> {
        self.join_impl(other, left_on, right_on, JoinType::Inner)
    }

    /// Left join with another table on specified columns
    pub fn left_join(
        &self,
        other: &GraphTable,
        left_on: &str,
        right_on: &str,
    ) -> GraphResult<GraphTable> {
        self.join_impl(other, left_on, right_on, JoinType::Left)
    }

    /// Right join with another table on specified columns
    pub fn right_join(
        &self,
        other: &GraphTable,
        left_on: &str,
        right_on: &str,
    ) -> GraphResult<GraphTable> {
        self.join_impl(other, left_on, right_on, JoinType::Right)
    }

    /// Outer join with another table on specified columns
    pub fn outer_join(
        &self,
        other: &GraphTable,
        left_on: &str,
        right_on: &str,
    ) -> GraphResult<GraphTable> {
        self.join_impl(other, left_on, right_on, JoinType::Outer)
    }

    /// Implementation of JOIN operations
    fn join_impl(
        &self,
        other: &GraphTable,
        left_on: &str,
        right_on: &str,
        join_type: JoinType,
    ) -> GraphResult<GraphTable> {
        // Get join columns
        let left_col = self.get_column_by_name(left_on).ok_or_else(|| {
            GraphError::InvalidInput(format!("Column '{}' not found in left table", left_on))
        })?;
        let right_col = other.get_column_by_name(right_on).ok_or_else(|| {
            GraphError::InvalidInput(format!("Column '{}' not found in right table", right_on))
        })?;

        // Build join index from right table for efficient lookup
        let mut right_index: HashMap<AttrValue, Vec<usize>> = HashMap::new();
        for (idx, value) in right_col.iter().enumerate() {
            right_index
                .entry(value.clone())
                .or_insert_with(Vec::new)
                .push(idx);
        }

        // Result vectors
        let mut result_rows = Vec::new();
        let mut matched_right_indices = std::collections::HashSet::new();

        // Process left table
        for left_idx in 0..self.shape().0 {
            if let Some(left_value) = left_col.get(left_idx) {
                if let Some(right_indices) = right_index.get(left_value) {
                    // Found matches - add all combinations
                    for &right_idx in right_indices {
                        result_rows.push((Some(left_idx), Some(right_idx)));
                        matched_right_indices.insert(right_idx);
                    }
                } else if matches!(join_type, JoinType::Left | JoinType::Outer) {
                    // No match in right table, include left row with nulls for right
                    result_rows.push((Some(left_idx), None));
                }
            }
        }

        // For right and outer joins, add unmatched rows from right table
        if matches!(join_type, JoinType::Right | JoinType::Outer) {
            for right_idx in 0..other.shape().0 {
                if !matched_right_indices.contains(&right_idx) {
                    result_rows.push((None, Some(right_idx)));
                }
            }
        }

        // Build result columns
        let mut result_columns = Vec::new();
        let mut result_column_names = Vec::new();

        // Add columns from left table
        for (col_idx, col_name) in self.column_names.iter().enumerate() {
            let left_array = &self.columns[col_idx];
            let result_values: Vec<AttrValue> = result_rows
                .iter()
                .map(|(left_idx, _)| {
                    if let Some(idx) = left_idx {
                        left_array.get(*idx).cloned().unwrap_or(AttrValue::Int(0))
                    } else {
                        AttrValue::Int(0) // NULL value for unmatched rows
                    }
                })
                .collect();
            result_columns.push(GraphArray::from_vec(result_values));
            result_column_names.push(format!("left_{}", col_name));
        }

        // Add columns from right table (excluding the join column to avoid duplicates)
        for (col_idx, col_name) in other.column_names.iter().enumerate() {
            if col_name != right_on {
                let right_array = &other.columns[col_idx];
                let result_values: Vec<AttrValue> = result_rows
                    .iter()
                    .map(|(_, right_idx)| {
                        if let Some(idx) = right_idx {
                            right_array.get(*idx).cloned().unwrap_or(AttrValue::Int(0))
                        } else {
                            AttrValue::Int(0) // NULL value for unmatched rows
                        }
                    })
                    .collect();
                result_columns.push(GraphArray::from_vec(result_values));
                result_column_names.push(format!("right_{}", col_name));
            }
        }

        // Create result table
        GraphTable::from_arrays_standalone(result_columns, Some(result_column_names))
    }

    /// Union with another table (combine rows)
    pub fn union(&self, other: &GraphTable) -> GraphResult<GraphTable> {
        // Check that column names match
        if self.column_names != other.column_names {
            return Err(GraphError::InvalidInput(
                "Cannot union tables with different column names".to_string(),
            ));
        }

        let mut result_columns = Vec::new();

        // Combine columns
        for (idx, _col_name) in self.column_names.iter().enumerate() {
            let left_array = &self.columns[idx];
            let right_array = &other.columns[idx];

            // Combine values from both arrays
            let mut combined_values = left_array.iter().cloned().collect::<Vec<_>>();
            combined_values.extend(right_array.iter().cloned());

            result_columns.push(GraphArray::from_vec(combined_values));
        }

        GraphTable::from_arrays_standalone(result_columns, Some(self.column_names.clone()))
    }

    /// Intersect with another table (rows present in both)
    pub fn intersect(&self, other: &GraphTable) -> GraphResult<GraphTable> {
        // Check that column names match
        if self.column_names != other.column_names {
            return Err(GraphError::InvalidInput(
                "Cannot intersect tables with different column names".to_string(),
            ));
        }

        // Build set of rows from other table for efficient lookup
        let mut other_rows: std::collections::HashSet<Vec<AttrValue>> =
            std::collections::HashSet::new();
        for row_idx in 0..other.shape().0 {
            if let Some(row_data) = other.iloc(row_idx) {
                let row_values: Vec<AttrValue> = self
                    .column_names
                    .iter()
                    .map(|col_name| row_data.get(col_name).cloned().unwrap_or(AttrValue::Int(0)))
                    .collect();
                other_rows.insert(row_values);
            }
        }

        // Find matching rows in this table
        let mut result_indices = Vec::new();
        for row_idx in 0..self.shape().0 {
            if let Some(row_data) = self.iloc(row_idx) {
                let row_values: Vec<AttrValue> = self
                    .column_names
                    .iter()
                    .map(|col_name| row_data.get(col_name).cloned().unwrap_or(AttrValue::Int(0)))
                    .collect();
                if other_rows.contains(&row_values) {
                    result_indices.push(row_idx);
                }
            }
        }

        // Build result columns with matching rows
        let mut result_columns = Vec::new();
        for array in &self.columns {
            let result_values: Vec<AttrValue> = result_indices
                .iter()
                .filter_map(|&idx| array.get(idx).cloned())
                .collect();
            result_columns.push(GraphArray::from_vec(result_values));
        }

        GraphTable::from_arrays_standalone(result_columns, Some(self.column_names.clone()))
    }

    /// Filter table rows based on graph-aware predicates
    /// This allows filtering based on node properties, connectivity, and graph topology
    pub fn filter_by_graph_predicate<F>(
        &self,
        graph: &crate::api::graph::Graph,
        node_id_column: &str,
        predicate: F,
    ) -> GraphResult<Self>
    where
        F: Fn(&crate::api::graph::Graph, NodeId, &HashMap<String, AttrValue>) -> bool,
    {
        let node_col = self.get_column_by_name(node_id_column).ok_or_else(|| {
            GraphError::InvalidInput(format!("Node ID column '{}' not found", node_id_column))
        })?;

        let (rows, _) = self.shape();
        let mut filtered_indices = Vec::new();

        // Check each row against the graph-aware predicate
        for row_idx in 0..rows {
            // Get node ID from the specified column
            if let Some(AttrValue::Int(node_id)) = node_col.get(row_idx) {
                let node_id = *node_id as NodeId;

                // Collect row attributes for the predicate
                let mut row_attrs = HashMap::new();
                for (col_idx, col_name) in self.column_names.iter().enumerate() {
                    if let Some(array) = self.columns.get(col_idx) {
                        if let Some(value) = array.get(row_idx) {
                            row_attrs.insert(col_name.clone(), value.clone());
                        }
                    }
                }

                // Apply the graph-aware predicate
                if predicate(graph, node_id, &row_attrs) {
                    filtered_indices.push(row_idx);
                }
            }
        }

        // Create filtered table
        let filtered_arrays: Vec<GraphArray> = self
            .columns
            .iter()
            .map(|array| {
                let filtered_values: Vec<AttrValue> = filtered_indices
                    .iter()
                    .filter_map(|&idx| array.get(idx).cloned())
                    .collect();
                GraphArray::from_vec(filtered_values)
            })
            .collect();

        let mut filtered_table =
            Self::from_arrays_standalone(filtered_arrays, Some(self.column_names.clone()))?;
        filtered_table.metadata = TableMetadata::new("filtered".to_string()).with_name(format!(
            "filtered_{}",
            self.metadata.name.as_deref().unwrap_or("table")
        ));

        Ok(filtered_table)
    }

    /// Filter nodes by degree (number of connections)
    pub fn filter_by_degree(
        &self,
        graph: &crate::api::graph::Graph,
        node_id_column: &str,
        min_degree: Option<usize>,
        max_degree: Option<usize>,
    ) -> GraphResult<Self> {
        self.filter_by_graph_predicate(graph, node_id_column, |graph, node_id, _attrs| {
            let degree = graph
                .neighbors(node_id)
                .map(|neighbors| neighbors.len())
                .unwrap_or(0);

            let meets_min = min_degree.map_or(true, |min| degree >= min);
            let meets_max = max_degree.map_or(true, |max| degree <= max);

            meets_min && meets_max
        })
    }

    /// Filter nodes by connectivity to specific target nodes
    pub fn filter_by_connectivity(
        &self,
        graph: &crate::api::graph::Graph,
        node_id_column: &str,
        target_nodes: &[NodeId],
        connection_type: ConnectivityType,
    ) -> GraphResult<Self> {
        self.filter_by_graph_predicate(graph, node_id_column, |graph, node_id, _attrs| {
            let neighbors = graph.neighbors(node_id).unwrap_or_default();

            match connection_type {
                ConnectivityType::ConnectedToAny => target_nodes
                    .iter()
                    .any(|&target| neighbors.contains(&target)),
                ConnectivityType::ConnectedToAll => target_nodes
                    .iter()
                    .all(|&target| neighbors.contains(&target)),
                ConnectivityType::NotConnectedToAny => !target_nodes
                    .iter()
                    .any(|&target| neighbors.contains(&target)),
            }
        })
    }

    /// Filter nodes within a certain distance of target nodes
    pub fn filter_by_distance(
        &self,
        graph: &crate::api::graph::Graph,
        node_id_column: &str,
        target_nodes: &[NodeId],
        max_distance: usize,
    ) -> GraphResult<Self> {
        use std::collections::{HashSet, VecDeque};

        // Pre-compute reachable nodes within max_distance from any target
        let mut reachable_nodes = HashSet::new();

        for &target in target_nodes {
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();

            queue.push_back((target, 0));
            visited.insert(target);

            while let Some((current_node, distance)) = queue.pop_front() {
                if distance <= max_distance {
                    reachable_nodes.insert(current_node);
                }

                if distance < max_distance {
                    if let Ok(neighbors) = graph.neighbors(current_node) {
                        for neighbor in neighbors {
                            if !visited.contains(&neighbor) {
                                visited.insert(neighbor);
                                queue.push_back((neighbor, distance + 1));
                            }
                        }
                    }
                }
            }
        }

        self.filter_by_graph_predicate(graph, node_id_column, |_graph, node_id, _attrs| {
            reachable_nodes.contains(&node_id)
        })
    }

    /// Filter nodes by attribute values combined with graph properties
    pub fn filter_by_attribute_and_graph<F>(
        &self,
        graph: &crate::api::graph::Graph,
        node_id_column: &str,
        attribute_predicate: F,
    ) -> GraphResult<Self>
    where
        F: Fn(&HashMap<String, AttrValue>) -> bool,
    {
        self.filter_by_graph_predicate(graph, node_id_column, |graph, node_id, row_attrs| {
            // First check if the row attributes satisfy the predicate
            if !attribute_predicate(row_attrs) {
                return false;
            }

            // Additional graph-based checks can be added here
            // For now, just ensure the node exists in the graph
            graph.neighbors(node_id).is_ok()
        })
    }

    /// Convert this table to a GraphMatrix if all columns are compatible numeric types
    ///
    /// # Examples
    /// ```rust,no_run
    /// // This works - both columns are numeric (example with setup)
    /// // let table = /* ... create table ... */;
    /// // let matrix = table.select_columns(&["age", "height"])?.matrix()?;
    ///
    /// // This fails - mixed numeric and text types
    /// // let result = table.select_columns(&["age", "name"])?.matrix(); // Returns error
    /// ```
    ///
    /// # Errors
    /// Returns an error if:
    /// - The table has columns with incompatible types (e.g., mixing numeric and text)
    /// - Any column contains non-numeric data when numeric is expected
    /// - The table is empty
    pub fn matrix(&self) -> GraphResult<crate::core::matrix::GraphMatrix> {
        use crate::core::matrix::GraphMatrix;

        if self.columns.is_empty() {
            return Err(GraphError::InvalidInput(
                "Cannot convert empty table to matrix".to_string(),
            ));
        }

        // Check if all columns have compatible types for matrix conversion
        let first_dtype = self.columns[0].dtype();

        // Only allow numeric types for matrix conversion
        if !first_dtype.is_numeric() {
            return Err(GraphError::InvalidInput(format!(
                "Cannot convert table to matrix: column '{}' has non-numeric type {:?}. Only numeric columns (Int, SmallInt, Float, Bool) can be converted to matrices.", 
                self.column_names.get(0).unwrap_or(&"<unnamed>".to_string()), 
                first_dtype
            )));
        }

        // Verify all columns have compatible numeric types
        for (i, column) in self.columns.iter().enumerate() {
            let column_dtype = column.dtype();

            // Fix the temporary borrow issue
            let default_name = format!("column_{}", i);
            let column_name = self.column_names.get(i).unwrap_or(&default_name);

            if !column_dtype.is_numeric() {
                return Err(GraphError::InvalidInput(format!(
                    "Cannot convert table to matrix: column '{}' has non-numeric type {:?}. All columns must be numeric (Int, SmallInt, Float, Bool) for matrix conversion.",
                    column_name,
                    column_dtype
                )));
            }
        }

        // Create GraphMatrix from the table's columns
        let matrix = GraphMatrix::from_arrays(self.columns.clone())
            .map_err(|e| GraphError::InvalidInput(format!(
                "Failed to convert table to matrix: {}. This usually means the column types are not compatible for matrix operations.", 
                e
            )))?;

        Ok(matrix)
    }

    // ==================================================================================
    // LAZY EVALUATION & MATERIALIZATION METHODS
    // ==================================================================================

    /// Get a preview of the table for display purposes (first N rows)
    /// This is used by repr() and does not materialize the full table
    pub fn preview(
        &self,
        row_limit: usize,
        col_limit: Option<usize>,
    ) -> (Vec<Vec<String>>, Vec<String>) {
        let num_cols = col_limit
            .unwrap_or(self.columns.len())
            .min(self.columns.len());
        let num_rows = row_limit.min(self.shape().0);

        // Get column names preview
        let col_names = self.column_names.iter().take(num_cols).cloned().collect();

        // Get data preview
        let mut preview_data = Vec::new();
        for row_idx in 0..num_rows {
            let mut row = Vec::new();
            for col_idx in 0..num_cols {
                let value = self.columns[col_idx]
                    .get(row_idx)
                    .map(|v| format!("{:?}", v))
                    .unwrap_or_else(|| "None".to_string());
                row.push(value);
            }
            preview_data.push(row);
        }

        (preview_data, col_names)
    }

    /// Materialize the table to nested vectors for Python consumption
    /// This is the primary materialization method used by .data property
    pub fn materialize(&self) -> Vec<Vec<AttrValue>> {
        let (rows, _cols) = self.shape();
        let mut materialized = Vec::with_capacity(rows);

        for row_idx in 0..rows {
            let mut row = Vec::with_capacity(self.columns.len());
            for column in &self.columns {
                let value = column.get(row_idx).cloned().unwrap_or(AttrValue::Null);
                row.push(value);
            }
            materialized.push(row);
        }

        materialized
    }

    /// Fill null/missing values with a specified value
    pub fn fill_na(&self, fill_value: AttrValue) -> GraphResult<Self> {
        let mut new_columns = Vec::with_capacity(self.columns.len());

        for column in &self.columns {
            let filled_values: Vec<AttrValue> = column
                .materialize()
                .iter()
                .map(|value| {
                    if matches!(value, AttrValue::Null) {
                        fill_value.clone()
                    } else {
                        value.clone()
                    }
                })
                .collect();

            let new_column = GraphArray::from_vec(filled_values).with_name(
                column
                    .name()
                    .cloned()
                    .unwrap_or_else(|| "unnamed".to_string()),
            );
            new_columns.push(new_column);
        }

        let mut new_table = Self::from_arrays(
            new_columns,
            Some(self.columns().to_vec()),
            self.graph.clone(),
        )?;
        new_table.metadata = self.metadata.clone();
        Ok(new_table)
    }

    /// Fill null/missing values in place with a specified value
    pub fn fill_na_inplace(&mut self, fill_value: AttrValue) -> GraphResult<()> {
        // We need to rebuild the columns since GraphArray doesn't expose mutable access to values
        for i in 0..self.columns.len() {
            let filled_values: Vec<AttrValue> = self.columns[i]
                .materialize()
                .iter()
                .map(|value| {
                    if matches!(value, AttrValue::Null) {
                        fill_value.clone()
                    } else {
                        value.clone()
                    }
                })
                .collect();

            let column_name = self.columns[i]
                .name()
                .cloned()
                .unwrap_or_else(|| "unnamed".to_string());
            self.columns[i] = GraphArray::from_vec(filled_values).with_name(column_name);
        }
        Ok(())
    }

    /// Drop rows containing any null/missing values
    pub fn drop_na(&self) -> GraphResult<Self> {
        let (rows, _) = self.shape();
        let mut keep_rows = Vec::new();

        // Find rows without any null values
        for row_idx in 0..rows {
            let mut has_null = false;
            for column in &self.columns {
                if let Some(value) = column.get(row_idx) {
                    if matches!(value, AttrValue::Null) {
                        has_null = true;
                        break;
                    }
                }
            }
            if !has_null {
                keep_rows.push(row_idx);
            }
        }

        // Create new columns with only the non-null rows
        let mut new_columns = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            let filtered_values: Vec<AttrValue> = keep_rows
                .iter()
                .filter_map(|&row_idx| column.get(row_idx).cloned())
                .collect();

            let new_column = GraphArray::from_vec(filtered_values).with_name(
                column
                    .name()
                    .cloned()
                    .unwrap_or_else(|| "unnamed".to_string()),
            );
            new_columns.push(new_column);
        }

        let mut new_table = Self::from_arrays(
            new_columns,
            Some(self.columns().to_vec()),
            self.graph.clone(),
        )?;
        new_table.metadata = self.metadata.clone();
        Ok(new_table)
    }

    /// Check if the table is effectively sparse (has many default/zero values)
    pub fn is_sparse(&self) -> bool {
        let total_elements = self.shape().0 * self.shape().1;
        if total_elements == 0 {
            return false;
        }

        let default_count: usize = self
            .columns
            .iter()
            .map(|col| {
                col.to_list()
                    .iter()
                    .filter(|v| self.is_default_value(v))
                    .count()
            })
            .sum();

        // Consider sparse if >50% are default values
        (default_count as f64) / (total_elements as f64) > 0.5
    }

    /// Check if a value is considered a "default" value for sparsity
    fn is_default_value(&self, value: &AttrValue) -> bool {
        match value {
            AttrValue::Int(0) | AttrValue::SmallInt(0) => true,
            AttrValue::Float(f) if f.abs() < 1e-10 => true,
            AttrValue::Bool(false) => true,
            AttrValue::Text(s) if s.is_empty() => true,
            _ => false,
        }
    }

    /// Get summary information for lazy display without full materialization
    pub fn summary_info(&self) -> String {
        let (rows, cols) = self.shape();
        let is_sparse = self.is_sparse();
        let source = &self.metadata.source_type;
        let name = self.metadata.name.as_deref().unwrap_or("unnamed");

        format!(
            "GraphTable('{}', shape=({}, {}), source='{}', sparse={})",
            name, rows, cols, source, is_sparse
        )
    }

    /// Create a lazy view of selected columns without materializing data
    pub fn select_lazy(&self, column_names: &[String]) -> GraphResult<Self> {
        let mut selected_arrays = Vec::new();
        let mut selected_names = Vec::new();

        for name in column_names {
            if let Some(col) = self.get_column_by_name(name) {
                selected_arrays.push(col.clone());
                selected_names.push(name.clone());
            } else {
                return Err(GraphError::InvalidInput(format!(
                    "Column '{}' not found",
                    name
                )));
            }
        }

        let mut result = Self::from_arrays_standalone(selected_arrays, Some(selected_names))?;
        result.metadata = TableMetadata::new("view".to_string()).with_name(format!(
            "view_of_{}",
            self.metadata.name.as_deref().unwrap_or("table")
        ));

        Ok(result)
    }
}

/// Types of connectivity for filtering
#[derive(Debug, Clone, Copy)]
pub enum ConnectivityType {
    /// Node is connected to at least one of the target nodes
    ConnectedToAny,
    /// Node is connected to all of the target nodes
    ConnectedToAll,
    /// Node is not connected to any of the target nodes
    NotConnectedToAny,
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
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{:>12}", name)?;
            }
            writeln!(f)?;

            // Rows
            for row in 0..display_rows {
                write!(f, "  ")?;
                for (col_idx, array) in self.columns.iter().enumerate() {
                    if col_idx > 0 {
                        write!(f, " ")?;
                    }
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
