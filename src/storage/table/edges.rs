//! EdgesTable - specialized table for edge data with edge-specific operations

use super::base::BaseTable;
use super::traits::{Table, TableIterator};
use crate::storage::array::BaseArray;
use crate::types::{EdgeId, NodeId, AttrValue};
use crate::errors::GraphResult;
use std::collections::HashMap;

/// Specialized table for edge data - requires edge_id, source, target columns
#[derive(Clone, Debug)]
pub struct EdgesTable {
    /// Underlying BaseTable
    base: BaseTable,
}

impl EdgesTable {
    /// Create a new EdgesTable with edge_id, source, target columns
    pub fn new(edges: Vec<(EdgeId, NodeId, NodeId)>) -> Self {
        let mut columns = HashMap::new();
        
        let (edge_ids, sources, targets): (Vec<_>, Vec<_>, Vec<_>) = {
            let mut edge_ids = Vec::new();
            let mut sources = Vec::new();
            let mut targets = Vec::new();
            
            for (edge_id, source, target) in edges {
                edge_ids.push(edge_id);
                sources.push(source);
                targets.push(target);
            }
            
            (edge_ids, sources, targets)
        };
        
        columns.insert("edge_id".to_string(), BaseArray::from_edge_ids(edge_ids));
        columns.insert("source".to_string(), BaseArray::from_node_ids(sources));
        columns.insert("target".to_string(), BaseArray::from_node_ids(targets));
        
        let column_order = vec!["edge_id".to_string(), "source".to_string(), "target".to_string()];
        let base = BaseTable::with_column_order(columns, column_order).expect("Valid edge table");
        
        Self { base }
    }
    
    /// Create EdgesTable from BaseTable (validates required columns exist)
    pub fn from_base_table(base: BaseTable) -> GraphResult<Self> {
        let required_cols = ["edge_id", "source", "target"];
        for col in &required_cols {
            if !base.has_column(col) {
                return Err(crate::errors::GraphError::InvalidInput(
                    format!("EdgesTable requires '{}' column", col)
                ));
            }
        }
        
        Ok(Self { base })
    }
    
    /// Get the edge IDs as a typed array
    pub fn edge_ids(&self) -> GraphResult<Vec<EdgeId>> {
        let edge_id_column = self.base.column("edge_id")
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                "edge_id column not found".to_string()
            ))?;
        
        edge_id_column.as_edge_ids()
    }
    
    /// Get the source node IDs
    pub fn sources(&self) -> GraphResult<Vec<NodeId>> {
        let source_column = self.base.column("source")
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                "source column not found".to_string()
            ))?;
        
        source_column.as_node_ids()
    }
    
    /// Get the target node IDs
    pub fn targets(&self) -> GraphResult<Vec<NodeId>> {
        let target_column = self.base.column("target")
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                "target column not found".to_string()
            ))?;
        
        target_column.as_node_ids()
    }
    
    /// Get edges as tuples (edge_id, source, target)
    pub fn as_tuples(&self) -> GraphResult<Vec<(EdgeId, NodeId, NodeId)>> {
        let edge_ids = self.edge_ids()?;
        let sources = self.sources()?;
        let targets = self.targets()?;
        
        Ok(edge_ids.into_iter()
            .zip(sources.into_iter())
            .zip(targets.into_iter())
            .map(|((edge_id, source), target)| (edge_id, source, target))
            .collect())
    }
    
    /// Filter edges by source nodes
    pub fn filter_by_sources(&self, source_nodes: &[NodeId]) -> GraphResult<Self> {
        // Create a simple predicate for now - could be optimized
        let source_strings: Vec<String> = source_nodes.iter().map(|id| id.to_string()).collect();
        let predicate = format!("source IN [{}]", source_strings.join(", "));
        
        // For now, use a simpler approach - filter manually
        let sources = self.sources()?;
        let source_set: std::collections::HashSet<_> = source_nodes.iter().collect();
        
        let mask: Vec<bool> = sources.iter().map(|s| source_set.contains(s)).collect();
        
        let filtered_base = self.base.filter_by_mask(&mask)?;
        Self::from_base_table(filtered_base)
    }
    
    /// Filter edges by target nodes
    pub fn filter_by_targets(&self, target_nodes: &[NodeId]) -> GraphResult<Self> {
        let targets = self.targets()?;
        let target_set: std::collections::HashSet<_> = target_nodes.iter().collect();
        
        let mask: Vec<bool> = targets.iter().map(|t| target_set.contains(t)).collect();
        
        let filtered_base = self.base.filter_by_mask(&mask)?;
        Self::from_base_table(filtered_base)
    }
    
    /// Add edge attributes from a HashMap
    pub fn with_attributes(mut self, attr_name: String, attributes: HashMap<EdgeId, AttrValue>) -> GraphResult<Self> {
        let edge_ids = self.edge_ids()?;
        let mut attr_values = Vec::new();
        
        for edge_id in &edge_ids {
            attr_values.push(attributes.get(edge_id).cloned().unwrap_or(AttrValue::Null));
        }
        
        let attr_column = BaseArray::from_attr_values(attr_values);
        self.base = self.base.with_column(attr_name, attr_column)?;
        
        Ok(self)
    }
    
    /// Convert back to BaseTable (loses edge-specific typing)
    pub fn into_base_table(self) -> BaseTable {
        self.base
    }
    
    /// Get reference to underlying BaseTable
    pub fn base_table(&self) -> &BaseTable {
        &self.base
    }
}


// Delegate all Table trait methods to the base table
impl Table for EdgesTable {
    fn nrows(&self) -> usize {
        self.base.nrows()
    }
    
    fn ncols(&self) -> usize {
        self.base.ncols()
    }
    
    fn column_names(&self) -> &[String] {
        self.base.column_names()
    }
    
    fn column(&self, name: &str) -> Option<&BaseArray> {
        self.base.column(name)
    }
    
    fn column_by_index(&self, index: usize) -> Option<&BaseArray> {
        self.base.column_by_index(index)
    }
    
    fn has_column(&self, name: &str) -> bool {
        self.base.has_column(name)
    }
    
    fn head(&self, n: usize) -> Self {
        Self { base: self.base.head(n) }
    }
    
    fn tail(&self, n: usize) -> Self {
        Self { base: self.base.tail(n) }
    }
    
    fn slice(&self, start: usize, end: usize) -> Self {
        Self { base: self.base.slice(start, end) }
    }
    
    fn sort_by(&self, column: &str, ascending: bool) -> GraphResult<Self> {
        Ok(Self { base: self.base.sort_by(column, ascending)? })
    }
    
    fn filter(&self, predicate: &str) -> GraphResult<Self> {
        Ok(Self { base: self.base.filter(predicate)? })
    }
    
    fn group_by(&self, columns: &[String]) -> GraphResult<Vec<Self>> {
        let base_groups = self.base.group_by(columns)?;
        base_groups.into_iter()
            .map(|base| Self::from_base_table(base))
            .collect()
    }
    
    fn select(&self, column_names: &[String]) -> GraphResult<Self> {
        // Ensure required columns are always included
        let mut cols = vec!["edge_id".to_string(), "source".to_string(), "target".to_string()];
        for col in column_names {
            if !["edge_id", "source", "target"].contains(&col.as_str()) {
                cols.push(col.clone());
            }
        }
        
        Ok(Self { base: self.base.select(&cols)? })
    }
    
    fn with_column(&self, name: String, column: BaseArray) -> GraphResult<Self> {
        Ok(Self { base: self.base.with_column(name, column)? })
    }
    
    fn drop_columns(&self, column_names: &[String]) -> GraphResult<Self> {
        // Prevent dropping required columns
        let required = ["edge_id", "source", "target"];
        let filtered_names: Vec<String> = column_names.iter()
            .filter(|&name| !required.contains(&name.as_str()))
            .cloned()
            .collect();
        
        Ok(Self { base: self.base.drop_columns(&filtered_names)? })
    }
    
    fn iter(&self) -> TableIterator<Self> {
        TableIterator::new(self.clone())
    }
}

impl std::fmt::Display for EdgesTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "EdgesTable[{} x {}]", self.nrows(), self.ncols())?;
        write!(f, "{}", self.base)
    }
}

impl From<Vec<(EdgeId, NodeId, NodeId)>> for EdgesTable {
    fn from(edges: Vec<(EdgeId, NodeId, NodeId)>) -> Self {
        Self::new(edges)
    }
}