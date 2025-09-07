//! EdgesTable - specialized table for edge data with edge-specific operations

use super::base::BaseTable;
use super::traits::{Table, TableIterator};
use crate::storage::array::{BaseArray, ArrayOps};
use crate::types::{EdgeId, NodeId, AttrValue};
use crate::errors::{GraphResult, GraphError};
use std::collections::{HashMap, HashSet};

/// Configuration for edge validation policies
#[derive(Clone, Debug)]
pub struct EdgeConfig {
    /// Whether to allow self-loops (edges from a node to itself)
    pub allow_self_loops: bool,
    /// Whether to allow multiple edges between the same source-target pair
    pub allow_multi_edges: bool,
    /// Whether to validate that source and target nodes exist in a provided node set
    pub validate_node_references: bool,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            allow_self_loops: true,
            allow_multi_edges: true,
            validate_node_references: false,
        }
    }
}

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
    
    /// Get the edge IDs as a typed array (only returns valid integer edge IDs)
    pub fn edge_ids(&self) -> GraphResult<Vec<EdgeId>> {
        let edge_id_column = self.base.column("edge_id")
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                "edge_id column not found".to_string()
            ))?;
        
        let (edge_ids, _) = edge_id_column.as_edge_ids_filtered();
        Ok(edge_ids)
    }
    
    /// Get the source node IDs (only returns valid integer node IDs)
    pub fn sources(&self) -> GraphResult<Vec<NodeId>> {
        let source_column = self.base.column("source")
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                "source column not found".to_string()
            ))?;
        
        let (sources, _) = source_column.as_node_ids_filtered();
        Ok(sources)
    }
    
    /// Get the target node IDs (only returns valid integer node IDs)
    pub fn targets(&self) -> GraphResult<Vec<NodeId>> {
        let target_column = self.base.column("target")
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                "target column not found".to_string()
            ))?;
        
        let (targets, _) = target_column.as_node_ids_filtered();
        Ok(targets)
    }
    
    /// Get edges as tuples (edge_id, source, target) - only returns complete rows
    pub fn as_tuples(&self) -> GraphResult<Vec<(EdgeId, NodeId, NodeId)>> {
        let edge_id_column = self.base.column("edge_id").unwrap();
        let source_column = self.base.column("source").unwrap();
        let target_column = self.base.column("target").unwrap();
        
        let (edge_ids, edge_indices) = edge_id_column.as_edge_ids_filtered();
        let (sources, source_indices) = source_column.as_node_ids_filtered();
        let (targets, target_indices) = target_column.as_node_ids_filtered();
        
        // Find rows where all three columns have valid values
        let valid_indices: Vec<usize> = edge_indices.iter()
            .filter(|&&i| source_indices.contains(&i) && target_indices.contains(&i))
            .cloned()
            .collect();
        
        let mut result = Vec::new();
        for &index in &valid_indices {
            let edge_id_pos = edge_indices.iter().position(|&i| i == index).unwrap();
            let source_pos = source_indices.iter().position(|&i| i == index).unwrap(); 
            let target_pos = target_indices.iter().position(|&i| i == index).unwrap();
            
            result.push((edge_ids[edge_id_pos], sources[source_pos], targets[target_pos]));
        }
        
        Ok(result)
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
    
    // =============================================================================
    // Phase 3: Edge-specific validation and iteration methods  
    // =============================================================================
    
    /// Validate edge structure according to EdgeConfig policies
    pub fn validate_edges(&self, config: &EdgeConfig) -> GraphResult<()> {
        let edge_ids = self.edge_ids()?;
        let sources = self.sources()?;
        let targets = self.targets()?;
        
        // Validate edge IDs are unique
        let mut seen_edge_ids = HashSet::new();
        for edge_id in &edge_ids {
            if !seen_edge_ids.insert(edge_id) {
                return Err(GraphError::InvalidInput(
                    format!("Duplicate edge_id found: {}", edge_id)
                ));
            }
        }
        
        // Check for null values in required columns
        let edge_id_column = self.base.column("edge_id").unwrap();
        let source_column = self.base.column("source").unwrap();
        let target_column = self.base.column("target").unwrap();
        
        for i in 0..edge_id_column.len() {
            if let Some(AttrValue::Null) = edge_id_column.get(i) {
                return Err(GraphError::InvalidInput("Null edge_id found".to_string()));
            }
            if let Some(AttrValue::Null) = source_column.get(i) {
                return Err(GraphError::InvalidInput("Null source found".to_string()));
            }
            if let Some(AttrValue::Null) = target_column.get(i) {
                return Err(GraphError::InvalidInput("Null target found".to_string()));
            }
        }
        
        // Policy-based validation
        if !config.allow_self_loops {
            for (source, target) in sources.iter().zip(targets.iter()) {
                if source == target {
                    return Err(GraphError::InvalidInput(
                        format!("Self-loop detected: {} -> {} (disallowed by config)", source, target)
                    ));
                }
            }
        }
        
        if !config.allow_multi_edges {
            let mut seen_pairs = HashSet::new();
            for (source, target) in sources.iter().zip(targets.iter()) {
                let pair = (*source, *target);
                if !seen_pairs.insert(pair) {
                    return Err(GraphError::InvalidInput(
                        format!("Multi-edge detected: {} -> {} (disallowed by config)", source, target)
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Edge-specific structural validation warnings
    pub fn validate_edge_structure(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        
        // Check required columns exist
        for required_col in &["edge_id", "source", "target"] {
            if !self.base.has_column(required_col) {
                warnings.push(format!("Missing required '{}' column", required_col));
            }
        }
        
        // Check for reasonable edge count
        if self.nrows() == 0 {
            warnings.push("Empty edges table".to_string());
        } else if self.nrows() > 10_000_000 {
            warnings.push(format!("Large edges table ({} rows) - consider partitioning", self.nrows()));
        }
        
        // Check for suspicious column patterns
        for col_name in self.column_names() {
            if col_name.contains(' ') {
                warnings.push(format!("Column '{}' contains spaces - may cause query issues", col_name));
            }
        }
        
        // Validate edges with default config
        if let Err(e) = self.validate_edges(&EdgeConfig::default()) {
            warnings.push(format!("Edge validation failed: {}", e));
        }
        
        warnings
    }
    
    /// Get an edge by its ID
    pub fn get_by_edge_id(&self, edge_id: EdgeId) -> Option<HashMap<String, AttrValue>> {
        let edge_id_column = self.base.column("edge_id")?;
        
        // Find the row with matching edge_id
        for i in 0..edge_id_column.len() {
            match edge_id_column.get(i) {
                Some(AttrValue::Int(id)) if (*id) as EdgeId == edge_id => {
                    // Found the row, collect all column values
                    let mut row = HashMap::new();
                    for col_name in self.column_names() {
                        if let Some(column) = self.base.column(col_name) {
                            if let Some(value) = column.get(i) {
                                row.insert(col_name.clone(), value.clone());
                            }
                        }
                    }
                    return Some(row);
                },
                _ => continue,
            }
        }
        
        None
    }
    
    /// Iterator over (EdgeId, source, target, row_data) tuples
    pub fn iter_with_edge_info(&self) -> impl Iterator<Item = (EdgeId, NodeId, NodeId, HashMap<String, AttrValue>)> + '_ {
        let edge_id_column = self.base.column("edge_id");
        let source_column = self.base.column("source");
        let target_column = self.base.column("target");
        
        (0..self.nrows()).filter_map(move |i| {
            // Get edge info for this row
            let edge_id = match edge_id_column?.get(i) {
                Some(AttrValue::Int(id)) => (*id) as EdgeId,
                _ => return None,
            };
            
            let source = match source_column?.get(i) {
                Some(AttrValue::Int(id)) => (*id) as NodeId,
                _ => return None,
            };
            
            let target = match target_column?.get(i) {
                Some(AttrValue::Int(id)) => (*id) as NodeId,
                _ => return None,
            };
            
            // Collect row data
            let mut row = HashMap::new();
            for col_name in self.column_names() {
                if let Some(column) = self.base.column(col_name) {
                    if let Some(value) = column.get(i) {
                        row.insert(col_name.clone(), value.clone());
                    }
                }
            }
            
            Some((edge_id, source, target, row))
        })
    }
    
    /// Get all incoming edges for a set of target nodes
    pub fn incoming_edges(&self, target_nodes: &[NodeId]) -> GraphResult<Self> {
        self.filter_by_targets(target_nodes)
    }
    
    /// Get all outgoing edges for a set of source nodes
    pub fn outgoing_edges(&self, source_nodes: &[NodeId]) -> GraphResult<Self> {
        self.filter_by_sources(source_nodes)
    }
    
    /// Get edge statistics
    pub fn edge_stats(&self) -> GraphResult<HashMap<String, usize>> {
        let mut stats = HashMap::new();
        
        stats.insert("total_edges".to_string(), self.nrows());
        stats.insert("total_columns".to_string(), self.ncols());
        
        // Count self-loops
        let sources = self.sources()?;
        let targets = self.targets()?;
        let self_loops = sources.iter().zip(targets.iter())
            .filter(|(s, t)| s == t)
            .count();
        stats.insert("self_loops".to_string(), self_loops);
        
        // Count unique sources and targets
        let unique_sources: HashSet<_> = sources.into_iter().collect();
        let unique_targets: HashSet<_> = targets.into_iter().collect();
        stats.insert("unique_sources".to_string(), unique_sources.len());
        stats.insert("unique_targets".to_string(), unique_targets.len());
        
        Ok(stats)
    }
    
    /// Access underlying BaseTable (Phase 3 plan method)
    pub fn base(&self) -> &BaseTable {
        &self.base
    }
    
    /// Convert into BaseTable (Phase 3 plan method)
    pub fn into_base(self) -> BaseTable {
        self.base
    }
    
    /// Filter edges by attribute value
    pub fn filter_by_attr(&self, attr_name: &str, value: &AttrValue) -> GraphResult<Self> {
        let predicate = match value {
            AttrValue::Text(s) => format!("{} == \"{}\"", attr_name, s),
            AttrValue::Int(i) => format!("{} == {}", attr_name, i),
            AttrValue::Float(f) => format!("{} == {}", attr_name, f),
            AttrValue::Bool(b) => format!("{} == {}", attr_name, b),
            _ => format!("{} == null", attr_name),
        };
        
        let filtered_base = self.base.filter(&predicate)?;
        Self::from_base_table(filtered_base)
    }
    
    /// Get unique values for an attribute
    pub fn unique_attr_values(&self, attr_name: &str) -> GraphResult<Vec<AttrValue>> {
        let column = self.base.column(attr_name)
            .ok_or_else(|| crate::errors::GraphError::InvalidInput(
                format!("Column '{}' not found", attr_name)
            ))?;
        
        column.unique_values()
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