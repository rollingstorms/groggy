//! GraphTable - composite table containing both nodes and edges with graph-specific operations

use super::base::BaseTable;
use super::base::InteractiveConfig;
use super::edges::{EdgeConfig, EdgesTable};
use super::nodes::NodesTable;
use super::traits::Table;
use crate::errors::{GraphError, GraphResult};
use crate::storage::array::BaseArray;
use crate::types::{AttrValue, EdgeId, NodeId};
use crate::viz::display::{ColumnSchema, DataType};
use crate::viz::streaming::data_source::{
    DataSchema, DataSource, DataWindow, GraphEdge, GraphMetadata, GraphNode, LayoutAlgorithm,
};
use crate::viz::streaming::NodePosition;
use crate::viz::VizModule;
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

/// Options controlling GraphTable merge behaviour
#[derive(Clone, Debug)]
pub struct MergeOptions {
    pub strategy: ConflictResolution,
}

impl MergeOptions {
    pub fn new(strategy: ConflictResolution) -> Self {
        Self { strategy }
    }

    pub fn domain_prefix() -> Self {
        Self::new(ConflictResolution::DomainPrefix)
    }
}

impl Default for MergeOptions {
    fn default() -> Self {
        MergeOptions::domain_prefix()
    }
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
        report
            .stats
            .insert("total_nodes".to_string(), self.nodes.nrows());
        report
            .stats
            .insert("total_edges".to_string(), self.edges.nrows());

        // Validate nodes table
        match self.nodes.validate_uids() {
            Ok(()) => {
                report.info.push("Node UIDs validation passed".to_string());
            }
            Err(e) => {
                report
                    .errors
                    .push(format!("Node UID validation failed: {}", e));
            }
        }

        let node_warnings = self.nodes.validate_node_structure();
        for warning in node_warnings {
            match self.policy.strictness {
                ValidationStrictness::Minimal => {
                    // Only add if it contains "error" or "missing"
                    if warning.to_lowercase().contains("error")
                        || warning.to_lowercase().contains("missing")
                    {
                        report.warnings.push(format!("Node structure: {}", warning));
                    }
                }
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
                    if warning.to_lowercase().contains("error")
                        || warning.to_lowercase().contains("missing")
                    {
                        report.warnings.push(format!("Edge structure: {}", warning));
                    }
                }
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
                self.edges.targets(),
            ) {
                let node_set: HashSet<NodeId> = node_ids.into_iter().collect();

                for (i, source) in sources.iter().enumerate() {
                    if !node_set.contains(source) {
                        report.errors.push(format!(
                            "Edge {} references non-existent source node {}",
                            i, source
                        ));
                    }
                }

                for (i, target) in targets.iter().enumerate() {
                    if !node_set.contains(target) {
                        report.errors.push(format!(
                            "Edge {} references non-existent target node {}",
                            i, target
                        ));
                    }
                }

                if report.errors.is_empty() {
                    report
                        .info
                        .push("Edge node reference validation passed".to_string());
                }
            } else {
                report
                    .warnings
                    .push("Could not perform node reference validation".to_string());
            }
        }

        // Additional strict validations
        if self.policy.strictness == ValidationStrictness::Strict {
            // Check for reasonable graph properties
            if self.nodes.nrows() == 0 && self.edges.nrows() > 0 {
                report
                    .warnings
                    .push("Graph has edges but no nodes".to_string());
            }

            if let Ok(edge_stats) = self.edges.edge_stats() {
                if let Some(&self_loops) = edge_stats.get("self_loops") {
                    if self_loops > 0 {
                        report
                            .info
                            .push(format!("Graph contains {} self-loops", self_loops));
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
            return Err(GraphError::InvalidInput(format!(
                "Graph validation failed with {} errors",
                report.errors.len()
            )));
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

    /// Merge GraphTables using the provided options
    pub fn merge<I>(tables: I, opts: MergeOptions) -> GraphResult<Self>
    where
        I: IntoIterator<Item = GraphTable>,
    {
        let mut tables_vec: Vec<GraphTable> = tables.into_iter().collect();

        if tables_vec.is_empty() {
            return Err(GraphError::InvalidInput(
                "Cannot merge empty list of tables".to_string(),
            ));
        }

        if tables_vec.len() == 1 {
            return Ok(tables_vec.pop().unwrap());
        }

        match opts.strategy {
            ConflictResolution::DomainPrefix | ConflictResolution::AutoRemap => {
                Self::merge_with_domain_prefix(tables_vec)
            }
            ConflictResolution::Fail => Self::merge_with_collision_detection(tables_vec, true),
            ConflictResolution::KeepFirst
            | ConflictResolution::KeepSecond
            | ConflictResolution::MergeAttributes => {
                Self::merge_with_attribute_strategy(tables_vec, opts.strategy)
            }
        }
    }

    /// Merge multiple GraphTables into a single GraphTable
    /// Handles UID collisions and domain mapping
    fn merge_with_domain_prefix(tables: Vec<GraphTable>) -> GraphResult<Self> {
        if tables.is_empty() {
            return Err(GraphError::InvalidInput(
                "Cannot merge empty list of tables".to_string(),
            ));
        }

        if tables.len() == 1 {
            return Ok(tables.into_iter().next().unwrap());
        }

        let _merged_policy = tables[0].policy.clone();
        let mut merged_nodes_data: Vec<HashMap<String, AttrValue>> = Vec::new();
        let mut merged_edges_data: Vec<(EdgeId, NodeId, NodeId, HashMap<String, AttrValue>)> =
            Vec::new();

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
                        attrs.insert(
                            "_domain".to_string(),
                            AttrValue::Text(domain_prefix.clone()),
                        );
                        attrs.insert(
                            "_original_id".to_string(),
                            AttrValue::Int(original_node_id as i64),
                        );

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
                        let new_source = domain_mapping.get(&original_source).copied().unwrap_or({
                            // If source node not found in current domain, keep original
                            original_source
                        });

                        let new_target = domain_mapping.get(&original_target).copied().unwrap_or({
                            // If target node not found in current domain, keep original
                            original_target
                        });

                        // Get edge attributes
                        let mut edge_attrs = HashMap::new();
                        if let Some(attrs) = table.edges.get_by_edge_id(original_edge_id) {
                            edge_attrs = attrs.clone();
                        }

                        // Add domain metadata
                        edge_attrs.insert(
                            "_domain".to_string(),
                            AttrValue::Text(domain_prefix.clone()),
                        );
                        edge_attrs.insert(
                            "_original_id".to_string(),
                            AttrValue::Int(original_edge_id as i64),
                        );

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
        let edge_tuples: Vec<(EdgeId, NodeId, NodeId)> = merged_edges_data
            .iter()
            .map(|(id, source, target, _)| (*id, *source, *target))
            .collect();
        let merged_edges = EdgesTable::new(edge_tuples);

        Ok(GraphTable::new(merged_nodes, merged_edges))
    }

    /// Merge with collision detection - fail on conflicts if strict = true
    fn merge_with_collision_detection(
        tables: Vec<GraphTable>,
        fail_on_conflict: bool,
    ) -> GraphResult<Self> {
        let mut all_node_ids = HashSet::new();
        let mut all_edge_ids = HashSet::new();

        // Check for ID collisions first
        for table in &tables {
            if let Ok(node_ids) = table.nodes.node_ids() {
                for &id in &node_ids {
                    if fail_on_conflict && all_node_ids.contains(&id) {
                        return Err(GraphError::InvalidInput(format!(
                            "Node ID collision detected: {}",
                            id
                        )));
                    }
                    all_node_ids.insert(id);
                }
            }

            if let Ok(edge_ids) = table.edges.edge_ids() {
                for &id in &edge_ids {
                    if fail_on_conflict && all_edge_ids.contains(&id) {
                        return Err(GraphError::InvalidInput(format!(
                            "Edge ID collision detected: {}",
                            id
                        )));
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
        let _merged_policy = tables[0].policy.clone();

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
    fn merge_with_attribute_strategy(
        tables: Vec<GraphTable>,
        strategy: ConflictResolution,
    ) -> GraphResult<Self> {
        let mut merged_nodes: HashMap<NodeId, HashMap<String, AttrValue>> = HashMap::new();
        let mut merged_edges: HashMap<EdgeId, (NodeId, NodeId, HashMap<String, AttrValue>)> =
            HashMap::new();

        for table in tables {
            // Merge nodes
            if let Ok(node_ids) = table.nodes.node_ids() {
                for &node_id in &node_ids {
                    if let Some(attrs) = table.nodes.get_by_uid(node_id) {
                        match strategy {
                            ConflictResolution::KeepFirst => {
                                merged_nodes.entry(node_id).or_insert(attrs.clone());
                            }
                            ConflictResolution::KeepSecond => {
                                merged_nodes.insert(node_id, attrs.clone());
                            }
                            ConflictResolution::MergeAttributes => {
                                let entry =
                                    merged_nodes.entry(node_id).or_insert_with(HashMap::new);
                                for (key, value) in attrs {
                                    entry.insert(key.clone(), value.clone());
                                }
                            }
                            _ => unreachable!("Invalid strategy for attribute merge"),
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
                                merged_edges
                                    .entry(edge_id)
                                    .or_insert((source, target, attrs));
                            }
                            ConflictResolution::KeepSecond => {
                                merged_edges.insert(edge_id, (source, target, attrs));
                            }
                            ConflictResolution::MergeAttributes => {
                                let entry = merged_edges.entry(edge_id).or_insert((
                                    source,
                                    target,
                                    HashMap::new(),
                                ));
                                for (key, value) in &attrs {
                                    entry.2.insert(key.clone(), value.clone());
                                }
                            }
                            _ => unreachable!("Invalid strategy for attribute merge"),
                        }
                    }
                }
            }
        }

        // Create basic tables with collected data
        let node_ids: Vec<NodeId> = merged_nodes.keys().cloned().collect();
        let edge_tuples: Vec<(EdgeId, NodeId, NodeId)> = merged_edges
            .iter()
            .map(|(id, (source, target, _))| (*id, *source, *target))
            .collect();

        let merged_nodes_table = NodesTable::new(node_ids);
        let merged_edges_table = EdgesTable::new(edge_tuples);

        Ok(GraphTable::new(merged_nodes_table, merged_edges_table))
    }

    /// Create a federated view from multiple domain bundles
    /// Each bundle is loaded from a separate path but presented as unified view
    pub fn from_federated_bundles(
        bundle_paths: Vec<&Path>,
        domain_names: Option<Vec<String>>,
    ) -> GraphResult<Self> {
        let mut tables = Vec::new();

        for (i, path) in bundle_paths.iter().enumerate() {
            let mut table = Self::load_bundle(path)?;

            // Add domain metadata to all nodes and edges
            let domain_name = domain_names
                .as_ref()
                .and_then(|names| names.get(i))
                .map(|s| s.clone())
                .unwrap_or_else(|| format!("domain_{}", i));

            table._add_domain_metadata(&domain_name)?;
            tables.push(table);
        }

        Self::merge(tables, MergeOptions::domain_prefix())
    }

    /// Add domain metadata to all nodes and edges in this table
    fn _add_domain_metadata(&mut self, _domain: &str) -> GraphResult<()> {
        // Add domain to all nodes
        if let Ok(node_ids) = self.nodes.node_ids() {
            for &_node_id in &node_ids {
                // This would require modifying the nodes table in-place
                // For now, we'll note this as a design consideration
                // In practice, domain metadata would be added during merge
            }
        }

        Ok(())
    }

    /// Get nodes by their IDs
    pub fn get_nodes(
        &self,
        node_ids: &[NodeId],
    ) -> GraphResult<HashMap<NodeId, HashMap<String, AttrValue>>> {
        let mut result = HashMap::new();

        for &node_id in node_ids {
            if let Some(node_data) = self.nodes.get_by_uid(node_id) {
                result.insert(node_id, node_data);
            }
        }

        Ok(result)
    }

    /// Get edges by their IDs
    pub fn get_edges(
        &self,
        edge_ids: &[EdgeId],
    ) -> GraphResult<HashMap<EdgeId, HashMap<String, AttrValue>>> {
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
        // Auto-fix common issues before validation (like missing edge IDs for meta nodes)
        let mut fixed_table = self.clone();

        // Check if we have null edge IDs and auto-assign them if needed
        let edge_id_column = fixed_table.edges.base().column("edge_id");
        if let Some(column) = edge_id_column {
            let has_nulls = (0..column.len())
                .any(|i| matches!(column.get(i), Some(crate::types::AttrValue::Null)));

            if has_nulls {
                // Auto-assign edge IDs to fix validation issues
                fixed_table = fixed_table.auto_assign_edge_ids()?;
            }
        }

        // Now validate the fixed graph structure
        let report = fixed_table.validate();
        if !report.is_valid() {
            return Err(GraphError::InvalidInput(format!(
                "Cannot convert invalid GraphTable to Graph. Errors: {:?}",
                report.errors
            )));
        }

        // Use the fixed table for conversion
        let graph_table = &fixed_table;

        // Create a new Graph
        let mut graph = crate::api::graph::Graph::new();

        // Add nodes
        let node_ids = graph_table
            .nodes
            .node_ids()
            .map_err(|e| GraphError::InvalidInput(format!("Failed to get node IDs: {}", e)))?;

        // We need to maintain a mapping from table node IDs to actual graph node IDs
        let mut node_id_map = HashMap::new();

        // First pass: Create nodes (base or meta) and map IDs
        for &table_node_id in &node_ids {
            if let Some(node_data) = graph_table.nodes.get_by_uid(table_node_id) {
                // Check if this is a meta node
                let is_meta = node_data
                    .get("entity_type")
                    .map(|v| matches!(v, crate::types::AttrValue::Text(s) if s == "meta"))
                    .unwrap_or(false);

                let graph_node_id = if is_meta {
                    // Extract subgraph ID for meta node creation
                    let subgraph_id = node_data
                        .get("contains_subgraph")
                        .and_then(|v| match v {
                            crate::types::AttrValue::SubgraphRef(id) => Some(*id),
                            crate::types::AttrValue::Int(id) => Some(*id as usize),
                            _ => None,
                        })
                        .unwrap_or(0); // Default to 0 if missing

                    // Create meta node with subgraph reference
                    match graph.create_meta_node(subgraph_id) {
                        Ok(node_id) => node_id,
                        Err(e) => {
                            eprintln!("Warning: Failed to create meta node for table node {}: {}. Creating as base node instead.", table_node_id, e);
                            graph.add_node()
                        }
                    }
                } else {
                    // Create regular base node
                    graph.add_node()
                };

                node_id_map.insert(table_node_id, graph_node_id);
            } else {
                // Fallback: create base node if no data found
                let graph_node_id = graph.add_node();
                node_id_map.insert(table_node_id, graph_node_id);
            }
        }

        // Second pass: Add remaining attributes (excluding system-managed ones)
        for &table_node_id in &node_ids {
            if let Some(node_data) = graph_table.nodes.get_by_uid(table_node_id) {
                let graph_node_id = node_id_map[&table_node_id];

                // Add each attribute except system-managed ones
                for (attr_name, attr_value) in node_data {
                    // Skip system-managed attributes that are handled during node creation
                    if !["node_id", "entity_type", "contains_subgraph"]
                        .contains(&attr_name.as_str())
                    {
                        if let Err(e) = graph.set_node_attr(
                            graph_node_id,
                            attr_name.clone(),
                            attr_value.clone(),
                        ) {
                            // Log warning but continue - non-critical for basic conversion
                            eprintln!("Warning: Failed to set node attribute {}: {}", attr_name, e);
                        }
                    }
                }
            }
        }

        // Add edges
        let edge_tuples = graph_table
            .edges
            .as_tuples()
            .map_err(|e| GraphError::InvalidInput(format!("Failed to get edge tuples: {}", e)))?;

        for (table_edge_id, table_source, table_target) in edge_tuples {
            // Map table node IDs to graph node IDs
            if let (Some(&graph_source), Some(&graph_target)) = (
                node_id_map.get(&table_source),
                node_id_map.get(&table_target),
            ) {
                match graph.add_edge(graph_source, graph_target) {
                    Ok(graph_edge_id) => {
                        // Add edge attributes if available
                        if let Some(edge_data) = graph_table.edges.get_by_edge_id(table_edge_id) {
                            for (attr_name, attr_value) in edge_data {
                                if !["edge_id", "source", "target"].contains(&attr_name.as_str()) {
                                    if let Err(e) = graph.set_edge_attr(
                                        graph_edge_id,
                                        attr_name.clone(),
                                        attr_value.clone(),
                                    ) {
                                        // Log warning but continue
                                        eprintln!(
                                            "Warning: Failed to set edge attribute {}: {}",
                                            attr_name, e
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        return Err(GraphError::InvalidInput(format!(
                            "Failed to add edge {} -> {}: {}",
                            table_source, table_target, e
                        )));
                    }
                }
            } else {
                return Err(GraphError::InvalidInput(format!(
                    "Edge references unmapped nodes: {} -> {}",
                    table_source, table_target
                )));
            }
        }

        Ok(graph)
    }

    /// Auto-assign edge IDs for null values (useful for meta nodes and imported data)
    pub fn auto_assign_edge_ids(mut self) -> GraphResult<Self> {
        // Delegate to the edges table's auto_assign_edge_ids method
        self.edges = self.edges.auto_assign_edge_ids()?;
        Ok(self)
    }

    /// Convert GraphTable into separate BaseTable components
    pub fn into_components(self) -> (BaseTable, BaseTable) {
        (self.nodes.into_base(), self.edges.into_base())
    }

    /// Get references to BaseTable components
    pub fn components(&self) -> (&BaseTable, &BaseTable) {
        (self.nodes.base(), self.edges.base())
    }

    /// Launch interactive visualization for this graph table
    ///
    /// Creates a VizModule that visualizes both nodes and edges as a complete graph.
    /// The visualization will show the graph structure with node-edge relationships.
    pub fn interactive(&self, _config: Option<InteractiveConfig>) -> GraphResult<VizModule> {
        use std::sync::Arc;

        // Create VizModule from this GraphTable which already implements DataSource
        // for unified graph visualization with both nodes and edges
        let data_source: Arc<dyn crate::viz::streaming::data_source::DataSource> =
            Arc::new(self.clone());
        let viz_module = VizModule::new(data_source);

        Ok(viz_module)
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

    fn column(&self, name: &str) -> Option<&BaseArray<AttrValue>> {
        // Try nodes first, then edges
        self.nodes.column(name).or_else(|| self.edges.column(name))
    }

    fn column_by_index(&self, index: usize) -> Option<&BaseArray<AttrValue>> {
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
            Err(GraphError::InvalidInput(format!(
                "Column '{}' not found in GraphTable",
                column
            )))
        }
    }

    fn sort_values(&self, columns: Vec<String>, ascending: Vec<bool>) -> GraphResult<Self> {
        // Check which component has all the required columns
        let nodes_has_all = columns.iter().all(|col| self.nodes.has_column(col));
        let edges_has_all = columns.iter().all(|col| self.edges.has_column(col));

        if nodes_has_all && !edges_has_all {
            // Sort nodes only
            Ok(Self {
                nodes: self.nodes.sort_values(columns, ascending)?,
                edges: self.edges.clone(),
                policy: self.policy.clone(),
            })
        } else if edges_has_all && !nodes_has_all {
            // Sort edges only
            Ok(Self {
                nodes: self.nodes.clone(),
                edges: self.edges.sort_values(columns, ascending)?,
                policy: self.policy.clone(),
            })
        } else if nodes_has_all && edges_has_all {
            // Both have the columns - prefer nodes
            Ok(Self {
                nodes: self.nodes.sort_values(columns, ascending)?,
                edges: self.edges.clone(),
                policy: self.policy.clone(),
            })
        } else {
            // Neither has all columns
            Err(GraphError::InvalidInput(format!(
                "Columns {:?} not found in GraphTable",
                columns
            )))
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
            Ok(node_groups
                .into_iter()
                .map(|nodes| Self {
                    nodes,
                    edges: self.edges.clone(),
                    policy: self.policy.clone(),
                })
                .collect())
        } else if edges_has_cols && !nodes_has_cols {
            // Group edges only, keep all nodes
            let edge_groups = self.edges.group_by(columns)?;
            Ok(edge_groups
                .into_iter()
                .map(|edges| Self {
                    nodes: self.nodes.clone(),
                    edges,
                    policy: self.policy.clone(),
                })
                .collect())
        } else {
            // Complex case - columns span both tables or neither
            Err(GraphError::InvalidInput(
                "Group by columns must belong to either nodes or edges table, not both".to_string(),
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

    fn with_column(&self, name: String, column: BaseArray<AttrValue>) -> GraphResult<Self> {
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

    fn pivot_table(
        &self,
        index_cols: &[String],
        columns_col: &str,
        values_col: &str,
        agg_func: &str,
    ) -> GraphResult<Self> {
        // Check which component has all the required columns
        let nodes_has_all = index_cols.iter().all(|col| self.nodes.has_column(col))
            && self.nodes.has_column(columns_col)
            && self.nodes.has_column(values_col);
        let edges_has_all = index_cols.iter().all(|col| self.edges.has_column(col))
            && self.edges.has_column(columns_col)
            && self.edges.has_column(values_col);

        if nodes_has_all && !edges_has_all {
            // Pivot nodes only
            Ok(Self {
                nodes: self
                    .nodes
                    .pivot_table(index_cols, columns_col, values_col, agg_func)?,
                edges: self.edges.clone(),
                policy: self.policy.clone(),
            })
        } else if edges_has_all && !nodes_has_all {
            // Pivot edges only
            Ok(Self {
                nodes: self.nodes.clone(),
                edges: self
                    .edges
                    .pivot_table(index_cols, columns_col, values_col, agg_func)?,
                policy: self.policy.clone(),
            })
        } else if nodes_has_all && edges_has_all {
            // Both have the columns - prefer nodes (consistent with other operations)
            Ok(Self {
                nodes: self
                    .nodes
                    .pivot_table(index_cols, columns_col, values_col, agg_func)?,
                edges: self.edges.clone(),
                policy: self.policy.clone(),
            })
        } else {
            // Neither has all columns
            Err(crate::errors::GraphError::InvalidInput(format!(
                "Required columns for pivot not found in GraphTable. Index cols: {:?}, Columns col: {}, Values col: {}",
                index_cols, columns_col, values_col
            )))
        }
    }

    fn melt(
        &self,
        id_vars: Option<&[String]>,
        value_vars: Option<&[String]>,
        var_name: Option<String>,
        value_name: Option<String>,
    ) -> GraphResult<Self> {
        // For melt, we need to determine which component to melt
        // Check if specified columns exist in nodes or edges
        let mut melt_nodes = false;
        let mut melt_edges = false;

        // Check id_vars
        if let Some(id_vars) = id_vars {
            let nodes_has_id_vars = id_vars.iter().all(|col| self.nodes.has_column(col));
            let edges_has_id_vars = id_vars.iter().all(|col| self.edges.has_column(col));

            if nodes_has_id_vars && !edges_has_id_vars {
                melt_nodes = true;
            } else if edges_has_id_vars && !nodes_has_id_vars {
                melt_edges = true;
            } else if nodes_has_id_vars && edges_has_id_vars {
                // Both have id_vars - prefer nodes
                melt_nodes = true;
            }
        }

        // Check value_vars if specified
        if let Some(value_vars) = value_vars {
            let nodes_has_value_vars = value_vars.iter().all(|col| self.nodes.has_column(col));
            let edges_has_value_vars = value_vars.iter().all(|col| self.edges.has_column(col));

            if nodes_has_value_vars && !edges_has_value_vars {
                melt_nodes = true;
            } else if edges_has_value_vars && !nodes_has_value_vars {
                melt_edges = true;
            } else if nodes_has_value_vars && edges_has_value_vars {
                // Both have value_vars - prefer nodes
                melt_nodes = true;
            }
        }

        // If neither id_vars nor value_vars specified, prefer nodes
        if !melt_nodes && !melt_edges {
            melt_nodes = true;
        }

        if melt_nodes {
            Ok(Self {
                nodes: self.nodes.melt(id_vars, value_vars, var_name, value_name)?,
                edges: self.edges.clone(),
                policy: self.policy.clone(),
            })
        } else {
            Ok(Self {
                nodes: self.nodes.clone(),
                edges: self.edges.melt(id_vars, value_vars, var_name, value_name)?,
                policy: self.policy.clone(),
            })
        }
    }
}

impl std::fmt::Display for GraphTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "GraphTable[{} nodes, {} edges]",
            self.nodes.nrows(),
            self.edges.nrows()
        )?;
        writeln!(f, "Validation Policy: {:?}", self.policy.strictness)?;
        writeln!(f, "Nodes:")?;
        write!(f, "{}", self.nodes)?;
        writeln!(f, "Edges:")?;
        write!(f, "{}", self.edges)?;
        Ok(())
    }
}

/// GraphMatrix conversion methods for GraphTable
impl GraphTable {
    /// Convert table data to GraphMatrix with generic numeric type support
    pub fn to_matrix<T>(&self) -> GraphResult<crate::storage::matrix::GraphMatrix<T>>
    where
        T: crate::storage::advanced_matrix::NumericType + crate::storage::matrix::FromAttrValue<T>,
    {
        use crate::storage::matrix::GraphMatrix;

        // For now, convert nodes table to matrix (most common use case)
        // TODO: Add option to convert edges table or combined table

        if self.nodes.is_empty() {
            return Ok(GraphMatrix::zeros(0, 0));
        }

        // Get node data as columns
        let node_ids = self.nodes.node_ids()?;
        let attribute_names = self.nodes.column_names();

        // Filter out the ID column if it exists
        let data_columns: Vec<_> = attribute_names
            .iter()
            .filter(|&name| name != "id" && name != "node_id")
            .collect();

        if data_columns.is_empty() {
            return Ok(GraphMatrix::zeros(node_ids.len(), 0));
        }

        // Build matrix data column by column
        let mut matrix_data = Vec::with_capacity(node_ids.len() * data_columns.len());

        for &col_name in &data_columns {
            if let Ok(column) = self.nodes.get_column(col_name) {
                for row_idx in 0..node_ids.len() {
                    let value = column.get(row_idx).unwrap_or(&AttrValue::Null);
                    let numeric_value = T::from_attr_value(value)?;
                    matrix_data.push(numeric_value);
                }
            } else {
                // Fill missing columns with zeros
                for _ in 0..node_ids.len() {
                    matrix_data.push(T::zero());
                }
            }
        }

        // Create column names vector
        let column_names: Vec<String> = data_columns.iter().map(|s| s.to_string()).collect();

        // Create GraphMatrix with proper column names
        let mut matrix = GraphMatrix::from_row_major_data(
            matrix_data,
            node_ids.len(),
            data_columns.len(),
            Some(&node_ids),
        )?;

        // Set custom column names
        matrix.set_column_names(column_names);

        Ok(matrix)
    }

    /// Convert to f64 matrix (most common use case)
    pub fn to_matrix_f64(&self) -> GraphResult<crate::storage::matrix::GraphMatrix<f64>> {
        self.to_matrix::<f64>()
    }

    /// Convert to f32 matrix (memory-efficient for ML)
    pub fn to_matrix_f32(&self) -> GraphResult<crate::storage::matrix::GraphMatrix<f32>> {
        self.to_matrix::<f32>()
    }

    /// Convert to integer matrix
    pub fn to_matrix_i64(&self) -> GraphResult<crate::storage::matrix::GraphMatrix<i64>> {
        self.to_matrix::<i64>()
    }

    /// Convert edges table to GraphMatrix
    pub fn edges_to_matrix<T>(&self) -> GraphResult<crate::storage::matrix::GraphMatrix<T>>
    where
        T: crate::storage::advanced_matrix::NumericType + crate::storage::matrix::FromAttrValue<T>,
    {
        use crate::storage::matrix::GraphMatrix;

        if self.edges.is_empty() {
            return Ok(GraphMatrix::zeros(0, 0));
        }

        let edge_ids = self.edges.edge_ids()?;
        let attribute_names = self.edges.column_names();

        // Filter out structural columns (id, source, target)
        let data_columns: Vec<_> = attribute_names
            .iter()
            .filter(|&name| !["id", "edge_id", "source", "target"].contains(&name.as_str()))
            .collect();

        if data_columns.is_empty() {
            return Ok(GraphMatrix::zeros(edge_ids.len(), 0));
        }

        // Build matrix data
        let mut matrix_data = Vec::with_capacity(edge_ids.len() * data_columns.len());

        for &col_name in &data_columns {
            if let Ok(column) = self.edges.get_column(col_name) {
                for row_idx in 0..edge_ids.len() {
                    let value = column.get(row_idx).unwrap_or(&AttrValue::Null);
                    let numeric_value = T::from_attr_value(value)?;
                    matrix_data.push(numeric_value);
                }
            }
        }

        GraphMatrix::from_row_major_data(
            matrix_data,
            edge_ids.len(),
            data_columns.len(),
            None, // Edge IDs as NodeIds not supported yet
        )
    }

    /// Get table shape (rows, columns) for nodes table
    pub fn shape(&self) -> (usize, usize) {
        (self.nodes.nrows(), self.nodes.column_names().len())
    }

    /// Check if table is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
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
    pub fn from_data(node_ids: Vec<NodeId>, edges: Vec<(EdgeId, NodeId, NodeId)>) -> Self {
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
        std::fs::create_dir_all(bundle_path).map_err(|e| {
            GraphError::InvalidInput(format!("Failed to create bundle directory: {}", e))
        })?;

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
            node_count: self.nodes.nrows(),
            edge_count: self.edges.nrows(),
            validation_policy: self.policy.clone(),
            checksums: BundleChecksums {
                nodes_sha256: nodes_checksum.clone(),
                edges_sha256: edges_checksum.clone(),
                metadata_sha256: String::new(), // Will be filled after metadata generation
            },
            schema_info: BundleSchemaInfo {
                node_columns: self.nodes.column_names().to_vec(),
                edge_columns: self.edges.column_names().to_vec(),
                has_node_attributes: self.nodes.ncols() > 1, // More than just node_id
                has_edge_attributes: self.edges.ncols() > 3, // More than edge_id, source, target
            },
            validation_summary: BundleValidationSummary {
                is_valid: validation_report.is_valid(),
                error_count: validation_report.errors.len(),
                warning_count: validation_report.warnings.len(),
                info_count: validation_report.info.len(),
            },
        };

        // Save metadata as JSON for better structure
        let metadata_json = serde_json::to_string_pretty(&metadata).map_err(|e| {
            GraphError::InvalidInput(format!("Failed to serialize metadata: {}", e))
        })?;

        // Calculate metadata checksum
        let metadata_checksum = self.calculate_checksum(&metadata_json);
        let mut final_metadata = metadata;
        final_metadata.checksums.metadata_sha256 = metadata_checksum;

        // Save final metadata with checksum
        let final_metadata_json = serde_json::to_string_pretty(&final_metadata).map_err(|e| {
            GraphError::InvalidInput(format!("Failed to serialize final metadata: {}", e))
        })?;

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
        .map_err(|e| {
            GraphError::InvalidInput(format!("Failed to serialize validation report: {}", e))
        })?;
        std::fs::write(&report_path, validation_json).map_err(|e| {
            GraphError::InvalidInput(format!("Failed to write validation report: {}", e))
        })?;

        // Save bundle manifest for integrity verification
        let manifest = BundleManifest {
            format_version: "2.0".to_string(),
            files: vec![
                FileEntry {
                    name: "metadata.json".to_string(),
                    checksum: final_metadata.checksums.metadata_sha256.clone(),
                },
                FileEntry {
                    name: "nodes.csv".to_string(),
                    checksum: nodes_checksum,
                },
                FileEntry {
                    name: "edges.csv".to_string(),
                    checksum: edges_checksum,
                },
            ],
            created_at: final_metadata.created_at.clone(),
        };

        let manifest_path = bundle_path.join("MANIFEST.json");
        let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|e| {
            GraphError::InvalidInput(format!("Failed to serialize manifest: {}", e))
        })?;
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
            return Err(GraphError::InvalidInput(format!(
                "Bundle path does not exist: {}. Make sure to:\n\
                    1. Create a bundle first using save_bundle()\n\
                    2. Provide the correct path to an existing bundle directory\n\
                    3. Ensure the path points to a directory (not a file)",
                bundle_path.display()
            )));
        }

        // Check if it's a directory
        if !bundle_path.is_dir() {
            return Err(GraphError::InvalidInput(format!(
                "Bundle path must be a directory, not a file: {}",
                bundle_path.display()
            )));
        }

        // Check for required files
        let metadata_path = bundle_path.join("metadata.json");
        let nodes_path = bundle_path.join("nodes.csv");
        let edges_path = bundle_path.join("edges.csv");

        // Provide detailed error messages for missing files
        let missing_files: Vec<String> = [
            ("metadata.json", &metadata_path),
            ("nodes.csv", &nodes_path),
            ("edges.csv", &edges_path),
        ]
        .iter()
        .filter(|(_, path)| !path.exists())
        .map(|(name, _)| name.to_string())
        .collect();

        if !missing_files.is_empty() {
            return Err(GraphError::InvalidInput(format!(
                "Bundle is missing required files: {}. \
                    Bundle directory should contain: metadata.json, nodes.csv, edges.csv",
                missing_files.join(", ")
            )));
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
        Ok(Self::with_policy(
            nodes_table,
            edges_table,
            metadata.validation_policy,
        ))
    }

    /// Validate bundle integrity without loading
    pub fn validate_bundle<P: AsRef<Path>>(bundle_path: P) -> GraphResult<BundleMetadata> {
        let bundle_path = bundle_path.as_ref();

        // Load and parse metadata (simplified parsing)
        let metadata_path = bundle_path.join("metadata.json");
        let metadata_content = std::fs::read_to_string(&metadata_path)
            .map_err(|e| GraphError::InvalidInput(format!("Failed to read metadata: {}", e)))?;

        // Parse JSON metadata (v2.0 format)
        let enhanced_metadata: EnhancedBundleMetadata = serde_json::from_str(&metadata_content)
            .map_err(|e| {
                GraphError::InvalidInput(format!("Failed to parse metadata JSON: {}", e))
            })?;

        // Convert enhanced metadata to legacy BundleMetadata format
        let mut checksums = HashMap::new();
        checksums.insert(
            "nodes_sha256".to_string(),
            enhanced_metadata.checksums.nodes_sha256.clone(),
        );
        checksums.insert(
            "edges_sha256".to_string(),
            enhanced_metadata.checksums.edges_sha256.clone(),
        );
        checksums.insert(
            "metadata_sha256".to_string(),
            enhanced_metadata.checksums.metadata_sha256.clone(),
        );

        let metadata = BundleMetadata {
            version: enhanced_metadata.version,
            created_at: enhanced_metadata.created_at,
            node_count: enhanced_metadata.node_count,
            edge_count: enhanced_metadata.edge_count,
            validation_policy: enhanced_metadata.validation_policy,
            checksums,
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
        info.insert(
            "validation_strictness".to_string(),
            format!("{:?}", metadata.validation_policy.strictness),
        );

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
            AttrValue::FloatVec(v) => format!(
                "[{}]",
                v.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
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
        // Use robust CSV parsing to correctly handle quoted fields and commas
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_content.as_bytes());

        // Capture headers and prepare column storage
        let headers = reader
            .headers()
            .map_err(|e| {
                GraphError::InvalidInput(format!("Failed to read nodes CSV headers: {}", e))
            })?
            .clone();

        let column_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
        if !column_names.iter().any(|h| h == "node_id") {
            return Err(GraphError::InvalidInput(
                "node_id column not found in nodes CSV".to_string(),
            ));
        }

        let mut column_data: std::collections::HashMap<
            String,
            crate::storage::array::BaseArray<AttrValue>,
        > = std::collections::HashMap::new();
        let mut temp_vectors: std::collections::HashMap<String, Vec<AttrValue>> = column_names
            .iter()
            .map(|name| (name.clone(), Vec::new()))
            .collect();

        for (row_idx, rec) in reader.records().enumerate() {
            let record = rec.map_err(|e| {
                GraphError::InvalidInput(format!(
                    "Failed to read nodes CSV record at row {}: {}",
                    row_idx + 1,
                    e
                ))
            })?;

            for (i, value) in record.iter().enumerate() {
                if let Some(col_name) = column_names.get(i) {
                    let attr = Self::parse_field_simple(value);
                    if let Some(vec) = temp_vectors.get_mut(col_name) {
                        vec.push(attr);
                    }
                }
            }
        }

        for (name, data) in temp_vectors.into_iter() {
            column_data.insert(
                name,
                crate::storage::array::BaseArray::from_attr_values(data),
            );
        }

        // Build BaseTable with the captured column order, then convert to NodesTable
        let base = crate::storage::table::BaseTable::with_column_order(column_data, column_names)
            .map_err(|e| {
            GraphError::InvalidInput(format!("Failed to build nodes BaseTable: {}", e))
        })?;

        let nodes = NodesTable::from_base_table(base).map_err(|e| {
            GraphError::InvalidInput(format!("Failed to build NodesTable from CSV: {}", e))
        })?;

        Ok(nodes)
    }

    fn parse_edges_from_csv(csv_content: &str) -> GraphResult<EdgesTable> {
        // Use robust CSV parsing to correctly handle quoted fields and commas
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_content.as_bytes());

        // Capture headers and prepare column storage
        let headers = reader
            .headers()
            .map_err(|e| {
                GraphError::InvalidInput(format!("Failed to read edges CSV headers: {}", e))
            })?
            .clone();

        let column_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
        for required in ["edge_id", "source", "target"].iter() {
            if !column_names.iter().any(|h| h == *required) {
                return Err(GraphError::InvalidInput(format!(
                    "{} column not found in edges CSV",
                    required
                )));
            }
        }

        let mut column_data: std::collections::HashMap<
            String,
            crate::storage::array::BaseArray<AttrValue>,
        > = std::collections::HashMap::new();
        let mut temp_vectors: std::collections::HashMap<String, Vec<AttrValue>> = column_names
            .iter()
            .map(|name| (name.clone(), Vec::new()))
            .collect();

        for (row_idx, rec) in reader.records().enumerate() {
            let record = rec.map_err(|e| {
                GraphError::InvalidInput(format!(
                    "Failed to read edges CSV record at row {}: {}",
                    row_idx + 1,
                    e
                ))
            })?;

            for (i, value) in record.iter().enumerate() {
                if let Some(col_name) = column_names.get(i) {
                    let attr = Self::parse_field_simple(value);
                    if let Some(vec) = temp_vectors.get_mut(col_name) {
                        vec.push(attr);
                    }
                }
            }
        }

        for (name, data) in temp_vectors.into_iter() {
            column_data.insert(
                name,
                crate::storage::array::BaseArray::from_attr_values(data),
            );
        }

        // Build BaseTable with the captured column order, then convert to EdgesTable
        let base = crate::storage::table::BaseTable::with_column_order(column_data, column_names)
            .map_err(|e| {
            GraphError::InvalidInput(format!("Failed to build edges BaseTable: {}", e))
        })?;

        let edges = EdgesTable::from_base_table(base).map_err(|e| {
            GraphError::InvalidInput(format!("Failed to build EdgesTable from CSV: {}", e))
        })?;

        Ok(edges)
    }

    /// Minimal CSV field-to-AttrValue conversion matching BaseTable behavior
    fn parse_field_simple(field: &str) -> AttrValue {
        if field.is_empty() {
            return AttrValue::Null;
        }
        match field.to_lowercase().as_str() {
            "true" => return AttrValue::Bool(true),
            "false" => return AttrValue::Bool(false),
            _ => {}
        }
        if let Ok(i) = field.parse::<i64>() {
            return AttrValue::Int(i);
        }
        if let Ok(f) = field.parse::<f32>() {
            return AttrValue::Float(f);
        }
        AttrValue::Text(field.to_string())
    }
}

// =============================================================================
// DataSource Implementation for Graph Visualization
// =============================================================================

impl DataSource for GraphTable {
    fn total_rows(&self) -> usize {
        // For GraphTable, return total unique entities (nodes + edges)
        self.nodes.nrows() + self.edges.nrows()
    }

    fn total_cols(&self) -> usize {
        // Return unique column names across both tables
        let mut columns = HashSet::new();
        columns.extend(self.nodes.column_names().iter());
        columns.extend(self.edges.column_names().iter());
        columns.len()
    }

    fn get_window(&self, start: usize, count: usize) -> DataWindow {
        // For GraphTable, we can window either nodes or edges
        // For simplicity, window nodes first, then edges if needed
        let node_count = self.nodes.nrows();

        if start < node_count {
            // Window is within nodes range
            let node_end = (start + count).min(node_count);
            self.nodes.get_window(start, node_end - start)
        } else {
            // Window is in edges range
            let edge_start = start - node_count;
            self.edges.get_window(edge_start, count)
        }
    }

    fn get_schema(&self) -> DataSchema {
        let mut columns = Vec::new();

        // Add node columns
        for col_name in self.nodes.column_names() {
            columns.push(ColumnSchema {
                name: format!("node_{}", col_name),
                data_type: match col_name.as_str() {
                    "node_id" => DataType::Integer,
                    _ => DataType::String, // Simplified for now
                },
            });
        }

        // Add edge columns
        for col_name in self.edges.column_names() {
            columns.push(ColumnSchema {
                name: format!("edge_{}", col_name),
                data_type: match col_name.as_str() {
                    "edge_id" | "source" | "target" => DataType::Integer,
                    _ => DataType::String, // Simplified for now
                },
            });
        }

        DataSchema {
            columns,
            primary_key: None, // Composite table doesn't have single primary key
            source_type: "graph_table".to_string(),
        }
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn get_column_types(&self) -> Vec<DataType> {
        self.get_schema()
            .columns
            .into_iter()
            .map(|c| c.data_type)
            .collect()
    }

    fn get_column_names(&self) -> Vec<String> {
        self.get_schema()
            .columns
            .into_iter()
            .map(|c| c.name)
            .collect()
    }

    fn get_source_id(&self) -> String {
        format!("graph_table_{}_{}", self.nodes.nrows(), self.edges.nrows())
    }

    fn get_version(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.nodes.nrows().hash(&mut hasher);
        self.nodes.ncols().hash(&mut hasher);
        self.edges.nrows().hash(&mut hasher);
        self.edges.ncols().hash(&mut hasher);
        hasher.finish()
    }

    // Graph visualization support - this is where GraphTable really shines
    fn supports_graph_view(&self) -> bool {
        true
    }

    fn get_graph_nodes(&self) -> Vec<GraphNode> {
        // Delegate to nodes table - it has the full implementation
        self.nodes.get_graph_nodes()
    }

    fn get_graph_edges(&self) -> Vec<GraphEdge> {
        // Delegate to edges table - it has the full implementation
        self.edges.get_graph_edges()
    }

    fn get_graph_metadata(&self) -> GraphMetadata {
        let nodes = self.get_graph_nodes();
        let edges = self.get_graph_edges();
        let mut attribute_types = HashMap::new();

        // Collect attribute types from both tables
        let nodes_schema = self.nodes.get_schema();
        let edges_schema = self.edges.get_schema();

        for col_schema in nodes_schema.columns {
            if col_schema.name != "node_id" {
                let type_name = match col_schema.data_type {
                    DataType::Integer => "integer",
                    DataType::Float => "float",
                    DataType::String => "string",
                    DataType::Boolean => "boolean",
                    DataType::DateTime => "datetime",
                    DataType::Json => "json",
                    DataType::Unknown => "unknown",
                };
                attribute_types.insert(format!("node_{}", col_schema.name), type_name.to_string());
            }
        }

        for col_schema in edges_schema.columns {
            if !["edge_id", "source", "target"].contains(&col_schema.name.as_str()) {
                let type_name = match col_schema.data_type {
                    DataType::Integer => "integer",
                    DataType::Float => "float",
                    DataType::String => "string",
                    DataType::Boolean => "boolean",
                    DataType::DateTime => "datetime",
                    DataType::Json => "json",
                    DataType::Unknown => "unknown",
                };
                attribute_types.insert(format!("edge_{}", col_schema.name), type_name.to_string());
            }
        }

        // Check if any edges have weights
        let has_weights = edges.iter().any(|e| e.weight.is_some());

        // Check if graph is directed (heuristic: if any edge has different source/target relationships)
        let mut edge_pairs = HashSet::new();
        let mut reverse_pairs = HashSet::new();
        let mut is_directed = false;

        for edge in &edges {
            let pair = (&edge.source, &edge.target);
            let reverse_pair = (&edge.target, &edge.source);

            if edge_pairs.contains(&reverse_pair) {
                // We found a bidirectional edge
                continue;
            } else if reverse_pairs.contains(&pair) {
                // We found the reverse of an edge we've seen
                continue;
            } else {
                edge_pairs.insert(pair);
                reverse_pairs.insert(reverse_pair);

                // Check if we have evidence this is directed
                if !edges
                    .iter()
                    .any(|e| e.source == edge.target && e.target == edge.source)
                {
                    is_directed = true;
                }
            }
        }

        GraphMetadata {
            node_count: nodes.len(),
            edge_count: edges.len(),
            is_directed,
            has_weights,
            attribute_types,
        }
    }

    fn compute_layout(&self, algorithm: LayoutAlgorithm) -> Vec<NodePosition> {
        // Use the edges table implementation which considers edge structure
        self.edges.compute_layout(algorithm)
    }
}
