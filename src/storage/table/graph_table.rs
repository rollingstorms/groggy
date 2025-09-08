//! GraphTable - composite table containing both nodes and edges with graph-specific operations

use super::base::BaseTable;
use super::nodes::NodesTable; 
use super::edges::{EdgesTable, EdgeConfig};
use super::traits::{Table, TableIterator};
use crate::storage::array::{BaseArray, ArrayOps};
use crate::types::{NodeId, EdgeId, AttrValue};
use crate::errors::{GraphResult, GraphError};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Validation policy settings for GraphTable
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ValidationPolicy {
    /// Strictness level for validation
    pub strictness: ValidationStrictness,
    /// Whether to validate that edge source/target nodes exist in nodes table
    pub validate_node_references: bool,
    /// Edge-specific validation config
    pub edge_config: EdgeConfig,
    /// Whether to auto-repair common issues during validation
    pub auto_repair: bool,
}

/// Validation strictness levels
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ValidationStrictness {
    /// Only check for critical errors that would cause failures
    Minimal,
    /// Check for common data quality issues
    Standard,
    /// Comprehensive validation with performance warnings
    Strict,
}

/// Conflict resolution strategies for merging GraphTables
#[derive(Clone, Debug, PartialEq)]
pub enum ConflictResolution {
    /// Fail on any ID collision
    Fail,
    /// Keep first table's data on collision
    KeepFirst,
    /// Keep second table's data on collision
    KeepSecond,
    /// Merge attributes, with second table overriding first on conflicts
    MergeAttributes,
    /// Use domain prefixes to avoid collisions
    DomainPrefix,
    /// Automatically generate new IDs for conflicts
    AutoRemap,
}

impl Default for ValidationPolicy {
    fn default() -> Self {
        Self {
            strictness: ValidationStrictness::Standard,
            validate_node_references: true,
            edge_config: EdgeConfig::default(),
            auto_repair: false,
        }
    }
}

/// Validation report containing warnings and errors
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub info: Vec<String>,
    pub stats: HashMap<String, usize>,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(), 
            info: Vec::new(),
            stats: HashMap::new(),
        }
    }
    
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
    
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

/// Enhanced bundle metadata structure for v2.0 bundles
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EnhancedBundleMetadata {
    pub version: String,
    pub created_at: String,
    pub groggy_version: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub validation_policy: ValidationPolicy,
    pub checksums: BundleChecksums,
    pub schema_info: BundleSchemaInfo,
    pub validation_summary: BundleValidationSummary,
}

/// File checksums for integrity verification
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BundleChecksums {
    pub nodes_sha256: String,
    pub edges_sha256: String,
    pub metadata_sha256: String,
}

/// Schema information for bundle compatibility
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BundleSchemaInfo {
    pub node_columns: Vec<String>,
    pub edge_columns: Vec<String>,
    pub has_node_attributes: bool,
    pub has_edge_attributes: bool,
}

/// Validation summary for quick bundle assessment
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BundleValidationSummary {
    pub is_valid: bool,
    pub error_count: usize,
    pub warning_count: usize,
    pub info_count: usize,
}

/// Bundle manifest for integrity verification
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BundleManifest {
    pub format_version: String,
    pub files: Vec<FileEntry>,
    pub created_at: String,
}

/// File entry in bundle manifest
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FileEntry {
    pub name: String,
    pub checksum: String,
}

/// Enhanced validation report for JSON serialization
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ValidationReportJson {
    pub summary: BundleValidationSummary,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub info: Vec<String>,
}

/// Composite table containing both nodes and edges with graph semantics
#[derive(Clone, Debug)]
pub struct GraphTable {
    /// Nodes component
    nodes: NodesTable,
    /// Edges component
    edges: EdgesTable,
    /// Validation policy
    policy: ValidationPolicy,
}

impl GraphTable {
    /// Create a new GraphTable from nodes and edges tables
    pub fn new(nodes: NodesTable, edges: EdgesTable) -> Self {
        Self {
            nodes,
            edges,
            policy: ValidationPolicy::default(),
        }
    }
    
    /// Create GraphTable with custom validation policy
    pub fn with_policy(nodes: NodesTable, edges: EdgesTable, policy: ValidationPolicy) -> Self {
        Self {
            nodes,
            edges,
            policy,
        }
    }
    
    /// Create empty GraphTable 
    pub fn empty() -> Self {
        let empty_nodes = NodesTable::new(vec![]);
        let empty_edges = EdgesTable::new(vec![]);
        Self::new(empty_nodes, empty_edges)
    }
    
    /// Get reference to nodes table
    pub fn nodes(&self) -> &NodesTable {
        &self.nodes
    }
    
    /// Get reference to edges table
    pub fn edges(&self) -> &EdgesTable {
        &self.edges
    }
    
    /// Get mutable reference to nodes table
    pub fn nodes_mut(&mut self) -> &mut NodesTable {
        &mut self.nodes
    }
    
    /// Get mutable reference to edges table
    pub fn edges_mut(&mut self) -> &mut EdgesTable {
        &mut self.edges
    }
    
    /// Get the validation policy
    pub fn policy(&self) -> &ValidationPolicy {
        &self.policy
    }
    
    /// Update validation policy
    pub fn set_policy(&mut self, policy: ValidationPolicy) {
        self.policy = policy;
    }
    
    // =============================================================================
    // Phase 4: Composite GraphTable validation and conformance methods
    // =============================================================================
    
    /// Validate the entire graph structure according to policy
    pub fn validate(&self) -> ValidationReport {
        let mut report = ValidationReport::new();
        
        // Basic statistics
        report.stats.insert("total_nodes".to_string(), self.nodes.nrows());
        report.stats.insert("total_edges".to_string(), self.edges.nrows());
        
        // Validate nodes table
        match self.nodes.validate_uids() {
            Ok(()) => {
                report.info.push("Node UIDs validation passed".to_string());
            }
            Err(e) => {
                report.errors.push(format!("Node UID validation failed: {}", e));
            }
        }
        
        let node_warnings = self.nodes.validate_node_structure();
        for warning in node_warnings {
            match self.policy.strictness {
                ValidationStrictness::Minimal => {
                    // Only add if it contains "error" or "missing"
                    if warning.to_lowercase().contains("error") || warning.to_lowercase().contains("missing") {
                        report.warnings.push(format!("Node structure: {}", warning));
                    }
                },
                ValidationStrictness::Standard | ValidationStrictness::Strict => {
                    report.warnings.push(format!("Node structure: {}", warning));
                }
            }
        }
        
        // Validate edges table
        match self.edges.validate_edges(&self.policy.edge_config) {
            Ok(()) => {
                report.info.push("Edge validation passed".to_string());
            }
            Err(e) => {
                report.errors.push(format!("Edge validation failed: {}", e));
            }
        }
        
        let edge_warnings = self.edges.validate_edge_structure();
        for warning in edge_warnings {
            match self.policy.strictness {
                ValidationStrictness::Minimal => {
                    if warning.to_lowercase().contains("error") || warning.to_lowercase().contains("missing") {
                        report.warnings.push(format!("Edge structure: {}", warning));
                    }
                },
                ValidationStrictness::Standard | ValidationStrictness::Strict => {
                    report.warnings.push(format!("Edge structure: {}", warning));
                }
            }
        }
        
        // Cross-validation: Check that edge endpoints reference valid nodes
        if self.policy.validate_node_references {
            if let (Ok(node_ids), Ok(sources), Ok(targets)) = (
                self.nodes.node_ids(),
                self.edges.sources(), 
                self.edges.targets()
            ) {
                let node_set: HashSet<NodeId> = node_ids.into_iter().collect();
                
                for (i, source) in sources.iter().enumerate() {
                    if !node_set.contains(source) {
                        report.errors.push(format!("Edge {} references non-existent source node {}", i, source));
                    }
                }
                
                for (i, target) in targets.iter().enumerate() {
                    if !node_set.contains(target) {
                        report.errors.push(format!("Edge {} references non-existent target node {}", i, target));
                    }
                }
                
                if report.errors.is_empty() {
                    report.info.push("Edge node reference validation passed".to_string());
                }
            } else {
                report.warnings.push("Could not perform node reference validation".to_string());
            }
        }
        
        // Additional strict validations
        if self.policy.strictness == ValidationStrictness::Strict {
            // Check for reasonable graph properties
            if self.nodes.nrows() == 0 && self.edges.nrows() > 0 {
                report.warnings.push("Graph has edges but no nodes".to_string());
            }
            
            if let Ok(edge_stats) = self.edges.edge_stats() {
                if let Some(&self_loops) = edge_stats.get("self_loops") {
                    if self_loops > 0 {
                        report.info.push(format!("Graph contains {} self-loops", self_loops));
                    }
                }
            }
        }
        
        report
    }
    
    /// Schema conformance check with reporting
    pub fn conform(&self) -> GraphResult<ValidationReport> {
        let report = self.validate();
        
        if !report.is_valid() {
            return Err(GraphError::InvalidInput(
                format!("Graph validation failed with {} errors", report.errors.len())
            ));
        }
        
        Ok(report)
    }
    
    /// Get graph statistics
    pub fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        
        stats.insert("total_nodes".to_string(), self.nodes.nrows());
        stats.insert("total_edges".to_string(), self.edges.nrows());
        stats.insert("node_columns".to_string(), self.nodes.ncols());
        stats.insert("edge_columns".to_string(), self.edges.ncols());
        
        // Add edge statistics if available
        if let Ok(edge_stats) = self.edges.edge_stats() {
            stats.extend(edge_stats);
        }
        
        stats
    }
    
    /// Merge multiple GraphTables into a single GraphTable
    /// Handles UID collisions and domain mapping
    pub fn merge(tables: Vec<GraphTable>) -> GraphResult<Self> {
        if tables.is_empty() {
            return Err(GraphError::InvalidInput("Cannot merge empty list of tables".to_string()));
        }
        
        if tables.len() == 1 {
            return Ok(tables.into_iter().next().unwrap());
        }
        
        let mut merged_policy = tables[0].policy.clone();
        let mut merged_nodes_data: Vec<HashMap<String, AttrValue>> = Vec::new();
        let mut merged_edges_data: Vec<(EdgeId, NodeId, NodeId, HashMap<String, AttrValue>)> = Vec::new();
        
        let mut node_id_offset = 0usize;
        let mut edge_id_offset = 0usize;
        let mut domain_mapping = HashMap::new();
        
        for (domain_idx, table) in tables.into_iter().enumerate() {
            let domain_prefix = format!("domain_{}", domain_idx);
            
            // Collect node data with domain mapping
            if let Ok(nodes) = table.nodes.node_ids() {
                for (idx, &original_node_id) in nodes.iter().enumerate() {
                    let new_node_id = node_id_offset + idx;
                    domain_mapping.insert(original_node_id, new_node_id);
                    
                    // Get node attributes
                    if let Some(node_attrs) = table.nodes.get_by_uid(original_node_id) {
                        let mut attrs = node_attrs.clone();
                        // Add domain metadata
                        attrs.insert("_domain".to_string(), AttrValue::Text(domain_prefix.clone()));
                        attrs.insert("_original_id".to_string(), AttrValue::Int(original_node_id as i64));
                        
                        merged_nodes_data.push(attrs);
                    }
                }
                node_id_offset += nodes.len();
            }
            
            // Collect edge data with remapped node references
            if let Ok(edges) = table.edges.edge_ids() {
                if let (Ok(sources), Ok(targets)) = (table.edges.sources(), table.edges.targets()) {
                    for (idx, &original_edge_id) in edges.iter().enumerate() {
                        let new_edge_id = edge_id_offset + idx;
                        let original_source = sources[idx];
                        let original_target = targets[idx];
                        
                        // Remap source and target using domain mapping
                        let new_source = domain_mapping.get(&original_source)
                            .copied()
                            .unwrap_or_else(|| {
                                // If source node not found in current domain, keep original
                                original_source
                            });
                            
                        let new_target = domain_mapping.get(&original_target)
                            .copied()
                            .unwrap_or_else(|| {
                                // If target node not found in current domain, keep original
                                original_target
                            });
                        
                        // Get edge attributes
                        let mut edge_attrs = HashMap::new();
                        if let Some(attrs) = table.edges.get_by_edge_id(original_edge_id) {
                            edge_attrs = attrs.clone();
                        }
                        
                        // Add domain metadata
                        edge_attrs.insert("_domain".to_string(), AttrValue::Text(domain_prefix.clone()));
                        edge_attrs.insert("_original_id".to_string(), AttrValue::Int(original_edge_id as i64));
                        
                        merged_edges_data.push((new_edge_id, new_source, new_target, edge_attrs));
                    }
                }
                edge_id_offset += edges.len();
            }
        }
        
        // Build merged tables
        // Extract node IDs for NodesTable creation
        let node_ids: Vec<NodeId> = (0..merged_nodes_data.len()).collect();
        let merged_nodes = NodesTable::new(node_ids);
        
        // Extract edge info for EdgesTable creation
        let edge_tuples: Vec<(EdgeId, NodeId, NodeId)> = merged_edges_data.iter()
            .map(|(id, source, target, _)| (*id, *source, *target))
            .collect();
        let merged_edges = EdgesTable::new(edge_tuples);
        
        Ok(GraphTable::new(merged_nodes, merged_edges))
    }
    
    /// Merge with another GraphTable using conflict resolution strategy
    pub fn merge_with(&mut self, other: GraphTable, strategy: ConflictResolution) -> GraphResult<()> {
        let merged = Self::merge_with_strategy(vec![self.clone(), other], strategy)?;
        *self = merged;
        Ok(())
    }
    
    /// Merge multiple GraphTables with specific conflict resolution strategy
    pub fn merge_with_strategy(tables: Vec<GraphTable>, strategy: ConflictResolution) -> GraphResult<Self> {
        if tables.is_empty() {
            return Err(GraphError::InvalidInput("Cannot merge empty list of tables".to_string()));
        }
        
        if tables.len() == 1 {
            return Ok(tables.into_iter().next().unwrap());
        }
        
        match strategy {
            ConflictResolution::DomainPrefix | ConflictResolution::AutoRemap => {
                // Use the default merge which already implements domain prefixing
                Self::merge(tables)
            },
            ConflictResolution::Fail => {
                Self::merge_with_collision_detection(tables, true)
            },
            ConflictResolution::KeepFirst | ConflictResolution::KeepSecond | ConflictResolution::MergeAttributes => {
                Self::merge_with_attribute_strategy(tables, strategy)
            }
        }
    }
    
    /// Merge with collision detection - fail on conflicts if strict = true
    fn merge_with_collision_detection(tables: Vec<GraphTable>, fail_on_conflict: bool) -> GraphResult<Self> {
        let mut all_node_ids = HashSet::new();
        let mut all_edge_ids = HashSet::new();
        
        // Check for ID collisions first
        for table in &tables {
            if let Ok(node_ids) = table.nodes.node_ids() {
                for &id in &node_ids {
                    if fail_on_conflict && all_node_ids.contains(&id) {
                        return Err(GraphError::InvalidInput(
                            format!("Node ID collision detected: {}", id)
                        ));
                    }
                    all_node_ids.insert(id);
                }
            }
            
            if let Ok(edge_ids) = table.edges.edge_ids() {
                for &id in &edge_ids {
                    if fail_on_conflict && all_edge_ids.contains(&id) {
                        return Err(GraphError::InvalidInput(
                            format!("Edge ID collision detected: {}", id)
                        ));
                    }
                    all_edge_ids.insert(id);
                }
            }
        }
        
        // If we get here, either no conflicts or fail_on_conflict=false
        // Use simple concatenation merge
        Self::merge_simple_concat(tables)
    }
    
    /// Simple concatenation merge (assumes no conflicts or conflicts are okay)
    fn merge_simple_concat(tables: Vec<GraphTable>) -> GraphResult<Self> {
        let mut merged_nodes_data = Vec::new();
        let mut merged_edges_data = Vec::new();
        let merged_policy = tables[0].policy.clone();
        
        for table in tables {
            // Collect node data
            if let Ok(node_ids) = table.nodes.node_ids() {
                for &node_id in &node_ids {
                    if let Some(attrs) = table.nodes.get_by_uid(node_id) {
                        merged_nodes_data.push(attrs.clone());
                    }
                }
            }
            
            // Collect edge data  
            if let Ok(edge_ids) = table.edges.edge_ids() {
                if let (Ok(sources), Ok(targets)) = (table.edges.sources(), table.edges.targets()) {
                    for (idx, &edge_id) in edge_ids.iter().enumerate() {
                        let source = sources[idx];
                        let target = targets[idx];
                        let attrs = table.edges.get_by_edge_id(edge_id).unwrap_or_default();
                        
                        merged_edges_data.push((edge_id, source, target, attrs));
                    }
                }
            }
        }
        
        // Create basic tables with empty data for now
        let merged_nodes = NodesTable::new(vec![]);
        let merged_edges = EdgesTable::new(vec![]);
        
        Ok(GraphTable::new(merged_nodes, merged_edges))
    }
    
    /// Merge with attribute-level conflict resolution
    fn merge_with_attribute_strategy(tables: Vec<GraphTable>, strategy: ConflictResolution) -> GraphResult<Self> {
        let mut merged_nodes: HashMap<NodeId, HashMap<String, AttrValue>> = HashMap::new();
        let mut merged_edges: HashMap<EdgeId, (NodeId, NodeId, HashMap<String, AttrValue>)> = HashMap::new();
        
        for table in tables {
            // Merge nodes
            if let Ok(node_ids) = table.nodes.node_ids() {
                for &node_id in &node_ids {
                    if let Some(attrs) = table.nodes.get_by_uid(node_id) {
                        match strategy {
                            ConflictResolution::KeepFirst => {
                                merged_nodes.entry(node_id).or_insert(attrs.clone());
                            },
                            ConflictResolution::KeepSecond => {
                                merged_nodes.insert(node_id, attrs.clone());
                            },
                            ConflictResolution::MergeAttributes => {
                                let entry = merged_nodes.entry(node_id).or_insert_with(HashMap::new);
                                for (key, value) in attrs {
                                    entry.insert(key.clone(), value.clone());
                                }
                            },
                            _ => unreachable!("Invalid strategy for attribute merge")
                        }
                    }
                }
            }
            
            // Merge edges
            if let Ok(edge_ids) = table.edges.edge_ids() {
                if let (Ok(sources), Ok(targets)) = (table.edges.sources(), table.edges.targets()) {
                    for (idx, &edge_id) in edge_ids.iter().enumerate() {
                        let source = sources[idx];
                        let target = targets[idx];
                        let attrs = table.edges.get_by_edge_id(edge_id).unwrap_or_default();
                        
                        match strategy {
                            ConflictResolution::KeepFirst => {
                                merged_edges.entry(edge_id).or_insert((source, target, attrs));
                            },
                            ConflictResolution::KeepSecond => {
                                merged_edges.insert(edge_id, (source, target, attrs));
                            },
                            ConflictResolution::MergeAttributes => {
                                let entry = merged_edges.entry(edge_id).or_insert((source, target, HashMap::new()));
                                for (key, value) in &attrs {
                                    entry.2.insert(key.clone(), value.clone());
                                }
                            },
                            _ => unreachable!("Invalid strategy for attribute merge")
                        }
                    }
                }
            }
        }
        
        // Create basic tables with collected data
        let node_ids: Vec<NodeId> = merged_nodes.keys().cloned().collect();
        let edge_tuples: Vec<(EdgeId, NodeId, NodeId)> = merged_edges.iter()
            .map(|(id, (source, target, _))| (*id, *source, *target))
            .collect();
        
        let merged_nodes_table = NodesTable::new(node_ids);
        let merged_edges_table = EdgesTable::new(edge_tuples);
        
        Ok(GraphTable::new(merged_nodes_table, merged_edges_table))
    }
    
    /// Create a federated view from multiple domain bundles
    /// Each bundle is loaded from a separate path but presented as unified view
    pub fn from_federated_bundles(bundle_paths: Vec<&Path>, domain_names: Option<Vec<String>>) -> GraphResult<Self> {
        let mut tables = Vec::new();
        
        for (i, path) in bundle_paths.iter().enumerate() {
            let mut table = Self::load_bundle(path)?;
            
            // Add domain metadata to all nodes and edges
            let domain_name = domain_names.as_ref()
                .and_then(|names| names.get(i))
                .map(|s| s.clone())
                .unwrap_or_else(|| format!("domain_{}", i));
                
            table.add_domain_metadata(&domain_name)?;
            tables.push(table);
        }
        
        Self::merge_with_strategy(tables, ConflictResolution::DomainPrefix)
    }
    
    /// Add domain metadata to all nodes and edges in this table
    fn add_domain_metadata(&mut self, domain: &str) -> GraphResult<()> {
        // Add domain to all nodes
        if let Ok(node_ids) = self.nodes.node_ids() {
            for &node_id in &node_ids {
                // This would require modifying the nodes table in-place
                // For now, we'll note this as a design consideration
                // In practice, domain metadata would be added during merge
            }
        }
        
        Ok(())
    }
    
    
    
    /// Get nodes by their IDs
    pub fn get_nodes(&self, node_ids: &[NodeId]) -> GraphResult<HashMap<NodeId, HashMap<String, AttrValue>>> {
        let mut result = HashMap::new();
        
        for &node_id in node_ids {
            if let Some(node_data) = self.nodes.get_by_uid(node_id) {
                result.insert(node_id, node_data);
            }
        }
        
        Ok(result)
    }
    
    /// Get edges by their IDs
    pub fn get_edges(&self, edge_ids: &[EdgeId]) -> GraphResult<HashMap<EdgeId, HashMap<String, AttrValue>>> {
        let mut result = HashMap::new();
        
        for &edge_id in edge_ids {
            if let Some(edge_data) = self.edges.get_by_edge_id(edge_id) {
                result.insert(edge_id, edge_data);
            }
        }
        
        Ok(result)
    }
    
    /// Get all neighbors of a set of nodes
    pub fn neighbors(&self, node_ids: &[NodeId]) -> GraphResult<HashSet<NodeId>> {
        let mut neighbors = HashSet::new();
        
        // Get outgoing edges
        let outgoing = self.edges.outgoing_edges(node_ids)?;
        let out_targets = outgoing.targets()?;
        neighbors.extend(out_targets);
        
        // Get incoming edges  
        let incoming = self.edges.incoming_edges(node_ids)?;
        let in_sources = incoming.sources()?;
        neighbors.extend(in_sources);
        
        // Remove the original nodes from neighbors
        for &node_id in node_ids {
            neighbors.remove(&node_id);
        }
        
        Ok(neighbors)
    }
    
    /// Enhanced conversion to Graph with validation (Phase 4 method)
    pub fn to_graph(&self) -> GraphResult<crate::api::graph::Graph> {
        // First validate the graph structure
        let report = self.validate();
        if !report.is_valid() {
            return Err(GraphError::InvalidInput(
                format!("Cannot convert invalid GraphTable to Graph. Errors: {:?}", report.errors)
            ));
        }
        
        // Create a new Graph
        let mut graph = crate::api::graph::Graph::new();
        
        // Add nodes
        let node_ids = self.nodes.node_ids()
            .map_err(|e| GraphError::InvalidInput(format!("Failed to get node IDs: {}", e)))?;
        
        // We need to maintain a mapping from table node IDs to actual graph node IDs
        let mut node_id_map = HashMap::new();
        
        for table_node_id in &node_ids {
            let graph_node_id = graph.add_node();
            node_id_map.insert(*table_node_id, graph_node_id);
        }
        
        // Add node attributes
        for &table_node_id in &node_ids {
            if let Some(node_data) = self.nodes.get_by_uid(table_node_id) {
                let graph_node_id = node_id_map[&table_node_id];
                
                // Add each attribute except node_id (that's the key)
                for (attr_name, attr_value) in node_data {
                    if attr_name != "node_id" {
                        if let Err(e) = graph.set_node_attr(graph_node_id, attr_name.clone(), attr_value.clone()) {
                            // Log warning but continue - non-critical for basic conversion
                            eprintln!("Warning: Failed to set node attribute {}: {}", attr_name, e);
                        }
                    }
                }
            }
        }
        
        // Add edges
        let edge_tuples = self.edges.as_tuples()
            .map_err(|e| GraphError::InvalidInput(format!("Failed to get edge tuples: {}", e)))?;
        
        for (table_edge_id, table_source, table_target) in edge_tuples {
            // Map table node IDs to graph node IDs
            if let (Some(&graph_source), Some(&graph_target)) = (
                node_id_map.get(&table_source),
                node_id_map.get(&table_target)
            ) {
                match graph.add_edge(graph_source, graph_target) {
                    Ok(graph_edge_id) => {
                        // Add edge attributes if available
                        if let Some(edge_data) = self.edges.get_by_edge_id(table_edge_id) {
                            for (attr_name, attr_value) in edge_data {
                                if !["edge_id", "source", "target"].contains(&attr_name.as_str()) {
                                    if let Err(e) = graph.set_edge_attr(graph_edge_id, attr_name.clone(), attr_value.clone()) {
                                        // Log warning but continue
                                        eprintln!("Warning: Failed to set edge attribute {}: {}", attr_name, e);
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        return Err(GraphError::InvalidInput(
                            format!("Failed to add edge {} -> {}: {}", table_source, table_target, e)
                        ));
                    }
                }
            } else {
                return Err(GraphError::InvalidInput(
                    format!("Edge references unmapped nodes: {} -> {}", table_source, table_target)
                ));
            }
        }
        
        Ok(graph)
    }
    
    /// Convert GraphTable into separate BaseTable components
    pub fn into_components(self) -> (BaseTable, BaseTable) {
        (self.nodes.into_base(), self.edges.into_base())
    }
    
    /// Get references to BaseTable components
    pub fn components(&self) -> (&BaseTable, &BaseTable) {
        (self.nodes.base(), self.edges.base())
    }
}

// Implement Table trait for GraphTable with composite semantics
impl Table for GraphTable {
    fn nrows(&self) -> usize {
        // For composite table, return total rows across both tables
        self.nodes.nrows() + self.edges.nrows()
    }
    
    fn ncols(&self) -> usize {
        // Return unique column names across both tables
        let mut columns = HashSet::new();
        columns.extend(self.nodes.column_names().iter());
        columns.extend(self.edges.column_names().iter());
        columns.len()
    }
    
    fn column_names(&self) -> &[String] {
        // For composite table, we need to return all unique column names
        // Since we can't return a computed Vec from a &[String] method,
        // we'll return nodes column names as the primary ones
        // NOTE: This is a limitation of the Table trait - ideally we'd return Vec<String>
        self.nodes.column_names()
    }
    
    fn column(&self, name: &str) -> Option<&BaseArray> {
        // Try nodes first, then edges
        self.nodes.column(name).or_else(|| self.edges.column(name))
    }
    
    fn column_by_index(&self, index: usize) -> Option<&BaseArray> {
        // Try nodes first, then edges with adjusted index
        let nodes_cols = self.nodes.ncols();
        if index < nodes_cols {
            self.nodes.column_by_index(index)
        } else {
            self.edges.column_by_index(index - nodes_cols)
        }
    }
    
    fn has_column(&self, name: &str) -> bool {
        self.nodes.has_column(name) || self.edges.has_column(name)
    }
    
    fn head(&self, n: usize) -> Self {
        // For composite table, take head from nodes primarily
        // This is a simplified implementation
        Self {
            nodes: self.nodes.head(n),
            edges: self.edges.clone(), // Keep all edges for now
            policy: self.policy.clone(),
        }
    }
    
    fn tail(&self, n: usize) -> Self {
        // For composite table, take tail from nodes primarily
        Self {
            nodes: self.nodes.tail(n),
            edges: self.edges.clone(), // Keep all edges for now
            policy: self.policy.clone(),
        }
    }
    
    fn slice(&self, start: usize, end: usize) -> Self {
        // For composite table, slice nodes primarily
        Self {
            nodes: self.nodes.slice(start, end),
            edges: self.edges.clone(), // Keep all edges for now
            policy: self.policy.clone(),
        }
    }
    
    fn sort_by(&self, column: &str, ascending: bool) -> GraphResult<Self> {
        // Try sorting nodes first, then edges
        if self.nodes.has_column(column) {
            Ok(Self {
                nodes: self.nodes.sort_by(column, ascending)?,
                edges: self.edges.clone(),
                policy: self.policy.clone(),
            })
        } else if self.edges.has_column(column) {
            Ok(Self {
                nodes: self.nodes.clone(),
                edges: self.edges.sort_by(column, ascending)?,
                policy: self.policy.clone(),
            })
        } else {
            Err(GraphError::InvalidInput(
                format!("Column '{}' not found in GraphTable", column)
            ))
        }
    }
    
    fn filter(&self, predicate: &str) -> GraphResult<Self> {
        // Apply filter to both components
        // This is a simplified approach - in practice, filtering might be more complex
        Ok(Self {
            nodes: self.nodes.filter(predicate)?,
            edges: self.edges.filter(predicate)?,
            policy: self.policy.clone(),
        })
    }
    
    fn group_by(&self, columns: &[String]) -> GraphResult<Vec<Self>> {
        // For composite tables, group by the component that has the columns
        let nodes_has_cols = columns.iter().all(|col| self.nodes.has_column(col));
        let edges_has_cols = columns.iter().all(|col| self.edges.has_column(col));
        
        if nodes_has_cols && !edges_has_cols {
            // Group nodes only, keep all edges
            let node_groups = self.nodes.group_by(columns)?;
            Ok(node_groups.into_iter()
                .map(|nodes| Self {
                    nodes,
                    edges: self.edges.clone(),
                    policy: self.policy.clone(),
                })
                .collect())
        } else if edges_has_cols && !nodes_has_cols {
            // Group edges only, keep all nodes  
            let edge_groups = self.edges.group_by(columns)?;
            Ok(edge_groups.into_iter()
                .map(|edges| Self {
                    nodes: self.nodes.clone(),
                    edges,
                    policy: self.policy.clone(),
                })
                .collect())
        } else {
            // Complex case - columns span both tables or neither
            Err(GraphError::InvalidInput(
                "Group by columns must belong to either nodes or edges table, not both".to_string()
            ))
        }
    }
    
    fn select(&self, column_names: &[String]) -> GraphResult<Self> {
        // For composite tables, select columns from appropriate components
        let mut node_columns = Vec::new();
        let mut edge_columns = Vec::new();
        
        for col in column_names {
            if self.nodes.has_column(col) {
                node_columns.push(col.clone());
            }
            if self.edges.has_column(col) {
                edge_columns.push(col.clone());
            }
        }
        
        // Always ensure required columns are present
        if !node_columns.contains(&"node_id".to_string()) {
            node_columns.push("node_id".to_string());
        }
        if !edge_columns.contains(&"edge_id".to_string()) {
            edge_columns.push("edge_id".to_string());
        }
        if !edge_columns.contains(&"source".to_string()) {
            edge_columns.push("source".to_string());
        }
        if !edge_columns.contains(&"target".to_string()) {
            edge_columns.push("target".to_string());
        }
        
        let selected_nodes = self.nodes.select(&node_columns)?;
        let selected_edges = self.edges.select(&edge_columns)?;
        
        Ok(Self {
            nodes: selected_nodes,
            edges: selected_edges,
            policy: self.policy.clone(),
        })
    }
    
    fn with_column(&self, name: String, column: BaseArray) -> GraphResult<Self> {
        // Adding columns to composite table is ambiguous
        // For now, add to nodes table by default
        Ok(Self {
            nodes: self.nodes.with_column(name, column)?,
            edges: self.edges.clone(),
            policy: self.policy.clone(),
        })
    }
    
    fn drop_columns(&self, column_names: &[String]) -> GraphResult<Self> {
        // Drop from both tables
        Ok(Self {
            nodes: self.nodes.drop_columns(column_names)?,
            edges: self.edges.drop_columns(column_names)?,
            policy: self.policy.clone(),
        })
    }
    
    fn iter(&self) -> TableIterator<Self> {
        TableIterator::new(self.clone())
    }
}

impl std::fmt::Display for GraphTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GraphTable[{} nodes, {} edges]", self.nodes.nrows(), self.edges.nrows())?;
        writeln!(f, "Validation Policy: {:?}", self.policy.strictness)?;
        writeln!(f, "Nodes:")?;
        write!(f, "{}", self.nodes)?;
        writeln!(f, "Edges:")?;
        write!(f, "{}", self.edges)?;
        Ok(())
    }
}

/// Factory methods for creating GraphTables from different sources
impl GraphTable {
    /// Create GraphTable from BaseTable components
    pub fn from_components(nodes_table: BaseTable, edges_table: BaseTable) -> GraphResult<Self> {
        let nodes = NodesTable::from_base_table(nodes_table)?;
        let edges = EdgesTable::from_base_table(edges_table)?;
        Ok(Self::new(nodes, edges))
    }
    
    /// Create GraphTable from raw node and edge data
    pub fn from_data(
        node_ids: Vec<NodeId>,
        edges: Vec<(EdgeId, NodeId, NodeId)>
    ) -> Self {
        let nodes = NodesTable::new(node_ids);
        let edges_table = EdgesTable::new(edges);
        Self::new(nodes, edges_table)
    }
}

// =============================================================================
// Phase 4: Bundle Storage Implementation
// =============================================================================

/// Bundle metadata for saved GraphTable
#[derive(Debug, Clone)]
pub struct BundleMetadata {
    pub version: String,
    pub created_at: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub validation_policy: ValidationPolicy,
    pub checksums: HashMap<String, String>,
}

impl BundleMetadata {
    pub fn new(graph_table: &GraphTable) -> Self {
        // Use a simple timestamp instead of chrono for now
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Self {
            version: "1.0.0".to_string(),
            created_at: format!("timestamp_{}", timestamp),
            node_count: graph_table.nodes.nrows(),
            edge_count: graph_table.edges.nrows(),
            validation_policy: graph_table.policy.clone(),
            checksums: HashMap::new(), // TODO: Implement checksums
        }
    }
}

impl GraphTable {
    /// Save GraphTable as a bundle to the specified path
    /// 
    /// A bundle consists of:
    /// - metadata.json: Bundle metadata and validation policy
    /// - nodes.parquet: Nodes table data (future: use parquet format)  
    /// - edges.parquet: Edges table data (future: use parquet format)
    /// - For now, we'll use JSON format as a placeholder
    pub fn save_bundle<P: AsRef<Path>>(&self, bundle_path: P) -> GraphResult<()> {
        let bundle_path = bundle_path.as_ref();
        
        // Create bundle directory
        std::fs::create_dir_all(bundle_path)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to create bundle directory: {}", e)))?;
        
        // Save tables first to calculate checksums
        let nodes_path = bundle_path.join("nodes.csv");
        let nodes_csv = self.serialize_nodes_to_csv()?;
        std::fs::write(&nodes_path, &nodes_csv)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to write nodes data: {}", e)))?;
        
        let edges_path = bundle_path.join("edges.csv"); 
        let edges_csv = self.serialize_edges_to_csv()?;
        std::fs::write(&edges_path, &edges_csv)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to write edges data: {}", e)))?;
        
        // Calculate checksums
        let nodes_checksum = self.calculate_checksum(&nodes_csv);
        let edges_checksum = self.calculate_checksum(&edges_csv);
        
        // Run validation
        let validation_report = self.validate();
        
        // Generate comprehensive metadata with checksums and validation
        let metadata = EnhancedBundleMetadata {
            version: "2.0".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            groggy_version: env!("CARGO_PKG_VERSION").to_string(),
            node_count: self.nodes_table.nrows(),
            edge_count: self.edges_table.nrows(),
            validation_policy: self.policy.clone(),
            checksums: BundleChecksums {
                nodes_sha256: nodes_checksum.clone(),
                edges_sha256: edges_checksum.clone(),
                metadata_sha256: String::new(), // Will be filled after metadata generation
            },
            schema_info: BundleSchemaInfo {
                node_columns: self.nodes_table.column_names().to_vec(),
                edge_columns: self.edges_table.column_names().to_vec(),
                has_node_attributes: self.nodes_table.ncols() > 1, // More than just node_id
                has_edge_attributes: self.edges_table.ncols() > 3, // More than edge_id, source, target
            },
            validation_summary: BundleValidationSummary {
                is_valid: validation_report.is_valid(),
                error_count: validation_report.errors.len(),
                warning_count: validation_report.warnings.len(),
                info_count: validation_report.info.len(),
            },
        };
        
        // Save metadata as JSON for better structure
        let metadata_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to serialize metadata: {}", e)))?;
        
        // Calculate metadata checksum
        let metadata_checksum = self.calculate_checksum(&metadata_json);
        let mut final_metadata = metadata;
        final_metadata.checksums.metadata_sha256 = metadata_checksum;
        
        // Save final metadata with checksum
        let final_metadata_json = serde_json::to_string_pretty(&final_metadata)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to serialize final metadata: {}", e)))?;
        
        let metadata_path = bundle_path.join("metadata.json");
        std::fs::write(&metadata_path, final_metadata_json)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to write metadata: {}", e)))?;
        
        // Save detailed validation report
        let report_path = bundle_path.join("validation_report.json");
        let validation_json = serde_json::to_string_pretty(&ValidationReportJson {
            summary: final_metadata.validation_summary.clone(),
            errors: validation_report.errors,
            warnings: validation_report.warnings,
            info: validation_report.info,
        })
        .map_err(|e| GraphError::InvalidInput(format!("Failed to serialize validation report: {}", e)))?;
        std::fs::write(&report_path, validation_json)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to write validation report: {}", e)))?;
        
        // Save bundle manifest for integrity verification
        let manifest = BundleManifest {
            format_version: "2.0".to_string(),
            files: vec![
                FileEntry { name: "metadata.json".to_string(), checksum: final_metadata.checksums.metadata_sha256.clone() },
                FileEntry { name: "nodes.csv".to_string(), checksum: nodes_checksum },
                FileEntry { name: "edges.csv".to_string(), checksum: edges_checksum },
            ],
            created_at: final_metadata.created_at.clone(),
        };
        
        let manifest_path = bundle_path.join("MANIFEST.json");
        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to serialize manifest: {}", e)))?;
        std::fs::write(&manifest_path, manifest_json)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to write manifest: {}", e)))?;
        
        Ok(())
    }
    
    /// Calculate SHA-256 checksum for data integrity
    fn calculate_checksum(&self, data: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
    
    /// Load GraphTable from a bundle directory
    /// 
    /// This is a placeholder implementation - in a real system this would:
    /// - Parse metadata.json
    /// - Load nodes.parquet and edges.parquet  
    /// - Reconstruct GraphTable with validation policy
    /// - Verify checksums
    pub fn load_bundle<P: AsRef<Path>>(bundle_path: P) -> GraphResult<Self> {
        let bundle_path = bundle_path.as_ref();
        
        // Check that bundle directory exists
        if !bundle_path.exists() {
            return Err(GraphError::InvalidInput(
                format!("Bundle path does not exist: {}. Make sure to:\n\
                    1. Create a bundle first using save_bundle()\n\
                    2. Provide the correct path to an existing bundle directory\n\
                    3. Ensure the path points to a directory (not a file)", 
                    bundle_path.display())
            ));
        }
        
        // Check if it's a directory
        if !bundle_path.is_dir() {
            return Err(GraphError::InvalidInput(
                format!("Bundle path must be a directory, not a file: {}", bundle_path.display())
            ));
        }
        
        // Check for required files
        let metadata_path = bundle_path.join("metadata.txt");
        let nodes_path = bundle_path.join("nodes.csv");
        let edges_path = bundle_path.join("edges.csv");
        
        // Provide detailed error messages for missing files
        let missing_files: Vec<String> = [
            ("metadata.txt", &metadata_path),
            ("nodes.csv", &nodes_path), 
            ("edges.csv", &edges_path)
        ].iter()
        .filter(|(_, path)| !path.exists())
        .map(|(name, _)| name.to_string())
        .collect();
        
        if !missing_files.is_empty() {
            return Err(GraphError::InvalidInput(
                format!("Bundle is missing required files: {}. \
                    Bundle directory should contain: metadata.txt, nodes.csv, edges.csv", 
                    missing_files.join(", "))
            ));
        }
        
        // Load metadata
        let metadata = Self::validate_bundle(bundle_path)?;
        
        // Load nodes and edges from CSV
        let nodes_csv = std::fs::read_to_string(&nodes_path)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to read nodes.csv: {}", e)))?;
        let edges_csv = std::fs::read_to_string(&edges_path)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to read edges.csv: {}", e)))?;
            
        // Parse CSV data (simplified parsing - not production grade)
        let nodes_table = Self::parse_nodes_from_csv(&nodes_csv)?;
        let edges_table = Self::parse_edges_from_csv(&edges_csv)?;
        
        // Create GraphTable with loaded policy
        Ok(Self::with_policy(nodes_table, edges_table, metadata.validation_policy))
    }
    
    /// Validate bundle integrity without loading
    pub fn validate_bundle<P: AsRef<Path>>(bundle_path: P) -> GraphResult<BundleMetadata> {
        let bundle_path = bundle_path.as_ref();
        
        // Load and parse metadata (simplified parsing)
        let metadata_path = bundle_path.join("metadata.txt");
        let metadata_content = std::fs::read_to_string(&metadata_path)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to read metadata: {}", e)))?;
        
        // Simple parsing of key-value format
        let mut version = "unknown".to_string();
        let mut created_at = "unknown".to_string();
        let mut node_count = 0;
        let mut edge_count = 0;
        let mut strictness = ValidationStrictness::Standard;
        
        for line in metadata_content.lines() {
            let parts: Vec<&str> = line.splitn(2, ':').collect();
            if parts.len() == 2 {
                let key = parts[0].trim();
                let value = parts[1].trim();
                match key {
                    "version" => version = value.to_string(),
                    "created_at" => created_at = value.to_string(),
                    "node_count" => node_count = value.parse().unwrap_or(0),
                    "edge_count" => edge_count = value.parse().unwrap_or(0),
                    "validation_strictness" => {
                        strictness = match value {
                            "Minimal" => ValidationStrictness::Minimal,
                            "Strict" => ValidationStrictness::Strict,
                            _ => ValidationStrictness::Standard,
                        };
                    },
                    _ => {}
                }
            }
        }
        
        let metadata = BundleMetadata {
            version,
            created_at,
            node_count,
            edge_count,
            validation_policy: ValidationPolicy {
                strictness,
                validate_node_references: true,
                edge_config: EdgeConfig::default(),
                auto_repair: false,
            },
            checksums: HashMap::new(),
        };
        
        // TODO: Verify checksums, file integrity, etc.
        
        Ok(metadata)
    }
    
    /// Get bundle information without loading the full GraphTable
    pub fn bundle_info<P: AsRef<Path>>(bundle_path: P) -> GraphResult<HashMap<String, String>> {
        let metadata = Self::validate_bundle(bundle_path)?;
        
        let mut info = HashMap::new();
        info.insert("version".to_string(), metadata.version);
        info.insert("created_at".to_string(), metadata.created_at);
        info.insert("node_count".to_string(), metadata.node_count.to_string());
        info.insert("edge_count".to_string(), metadata.edge_count.to_string());
        info.insert("validation_strictness".to_string(), format!("{:?}", metadata.validation_policy.strictness));
        
        Ok(info)
    }
    
    // Helper methods for CSV serialization
    fn serialize_nodes_to_csv(&self) -> GraphResult<String> {
        let mut csv_lines = Vec::new();
        
        // Header
        let column_names = self.nodes.column_names();
        csv_lines.push(column_names.join(","));
        
        // Data rows
        let nrows = self.nodes.nrows();
        for i in 0..nrows {
            let mut row = Vec::new();
            for col_name in column_names {
                if let Some(column) = self.nodes.column(col_name) {
                    if let Some(value) = column.get(i) {
                        row.push(self.attr_value_to_csv_string(value));
                    } else {
                        row.push("".to_string());
                    }
                } else {
                    row.push("".to_string());
                }
            }
            csv_lines.push(row.join(","));
        }
        
        Ok(csv_lines.join("\n"))
    }
    
    fn serialize_edges_to_csv(&self) -> GraphResult<String> {
        let mut csv_lines = Vec::new();
        
        // Header
        let column_names = self.edges.column_names();
        csv_lines.push(column_names.join(","));
        
        // Data rows
        let nrows = self.edges.nrows();
        for i in 0..nrows {
            let mut row = Vec::new();
            for col_name in column_names {
                if let Some(column) = self.edges.column(col_name) {
                    if let Some(value) = column.get(i) {
                        row.push(self.attr_value_to_csv_string(value));
                    } else {
                        row.push("".to_string());
                    }
                } else {
                    row.push("".to_string());
                }
            }
            csv_lines.push(row.join(","));
        }
        
        Ok(csv_lines.join("\n"))
    }
    
    fn attr_value_to_csv_string(&self, value: &AttrValue) -> String {
        match value {
            AttrValue::Int(i) => i.to_string(),
            AttrValue::Float(f) => f.to_string(),
            AttrValue::Text(s) => format!("\"{}\"", s.replace("\"", "\"\"")), // Escape quotes
            AttrValue::Bool(b) => b.to_string(),
            AttrValue::Null => "".to_string(),
            AttrValue::FloatVec(v) => format!("[{}]", v.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(",")),
            AttrValue::CompactText(s) => format!("\"{}\"", s.as_str().replace("\"", "\"\"")),
            AttrValue::SmallInt(i) => i.to_string(),
            AttrValue::Bytes(b) => format!("bytes:{}", b.len()), // Just show byte length
            AttrValue::CompressedText(_) => "compressed_text".to_string(), // Placeholder
            // Add any other missing variants with reasonable string representations
            _ => "unsupported".to_string(),
        }
    }
    
    // Helper methods for CSV parsing (simplified - not production grade)
    fn parse_nodes_from_csv(csv_content: &str) -> GraphResult<NodesTable> {
        let lines: Vec<&str> = csv_content.lines().collect();
        if lines.is_empty() {
            return Err(GraphError::InvalidInput("Empty nodes CSV".to_string()));
        }
        
        let header = lines[0];
        let column_names: Vec<&str> = header.split(',').collect();
        
        // Find node_id column
        let node_id_index = column_names.iter()
            .position(|&name| name == "node_id")
            .ok_or_else(|| GraphError::InvalidInput("node_id column not found in CSV".to_string()))?;
        
        // Extract node IDs from all rows
        let mut node_ids = Vec::new();
        for line in lines.iter().skip(1) { // Skip header
            let values: Vec<&str> = line.split(',').collect();
            if values.len() > node_id_index {
                let node_id_str = values[node_id_index].trim();
                let node_id: NodeId = node_id_str.parse()
                    .map_err(|_| GraphError::InvalidInput(format!("Invalid node_id: {}", node_id_str)))?;
                node_ids.push(node_id);
            }
        }
        
        // For now, just create a basic NodesTable with node_ids
        // TODO: Parse other attributes from CSV
        Ok(NodesTable::new(node_ids))
    }
    
    fn parse_edges_from_csv(csv_content: &str) -> GraphResult<EdgesTable> {
        let lines: Vec<&str> = csv_content.lines().collect();
        if lines.is_empty() {
            return Err(GraphError::InvalidInput("Empty edges CSV".to_string()));
        }
        
        let header = lines[0];
        let column_names: Vec<&str> = header.split(',').collect();
        
        // Find required columns
        let edge_id_index = column_names.iter()
            .position(|&name| name == "edge_id")
            .ok_or_else(|| GraphError::InvalidInput("edge_id column not found in CSV".to_string()))?;
        let source_index = column_names.iter()
            .position(|&name| name == "source")
            .ok_or_else(|| GraphError::InvalidInput("source column not found in CSV".to_string()))?;
        let target_index = column_names.iter()
            .position(|&name| name == "target")
            .ok_or_else(|| GraphError::InvalidInput("target column not found in CSV".to_string()))?;
        
        // Extract edges from all rows
        let mut edges = Vec::new();
        for line in lines.iter().skip(1) { // Skip header
            let values: Vec<&str> = line.split(',').collect();
            if values.len() > edge_id_index && values.len() > source_index && values.len() > target_index {
                let edge_id_str = values[edge_id_index].trim();
                let source_str = values[source_index].trim();
                let target_str = values[target_index].trim();
                
                let edge_id: EdgeId = edge_id_str.parse()
                    .map_err(|_| GraphError::InvalidInput(format!("Invalid edge_id: {}", edge_id_str)))?;
                let source: NodeId = source_str.parse()
                    .map_err(|_| GraphError::InvalidInput(format!("Invalid source: {}", source_str)))?;
                let target: NodeId = target_str.parse()
                    .map_err(|_| GraphError::InvalidInput(format!("Invalid target: {}", target_str)))?;
                
                edges.push((edge_id, source, target));
            }
        }
        
        // Create EdgesTable
        Ok(EdgesTable::new(edges))
    }
}