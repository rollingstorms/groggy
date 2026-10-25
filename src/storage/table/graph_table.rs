//! GraphTable - composite table containing both nodes and edges with graph-specific operations

#[path = "bundle.rs"]
mod bundle;
#[path = "domain_merge.rs"]
mod domain_merge;
#[path = "graph_streaming.rs"]
mod streaming;

use super::base::BaseTable;
use super::base::InteractiveConfig;
use super::edges::{EdgeConfig, EdgesTable};
use super::nodes::NodesTable;
use super::traits::Table;
use crate::errors::{GraphError, GraphResult};
use crate::storage::array::BaseArray;
use crate::types::{AttrValue, EdgeId, NodeId};
use crate::viz::VizModule;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

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

/// Composite node/edge table bundle. Clones share immutable components;
/// mutable component access detaches that component before returning a reference.
#[derive(Clone, Debug)]
pub struct GraphTable {
    /// Process-local presentation identity; never an entity/history ID.
    presentation_id: u64,
    /// Nodes component
    nodes: Arc<NodesTable>,
    /// Edges component
    edges: Arc<EdgesTable>,
    /// Validation policy
    policy: ValidationPolicy,
}

/// Independent component dimensions; GraphTable is not one rectangular table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphTableShape {
    /// Node rows and columns.
    pub nodes: (usize, usize),
    /// Edge rows and columns.
    pub edges: (usize, usize),
}

/// Internal component routing; GraphTable retains distinct node/edge ownership.
#[derive(Clone, Copy)]
enum Component {
    Nodes,
    Edges,
}

impl GraphTable {
    // Build a node-to-group map, then dispatch every edge once. Node grouping
    // partitions rows; crossing edges belong to no group. Row projection retains
    // captured edge identity and typed columns through the shared BaseTable path.
    fn induced_groups(&self, groups: Vec<NodesTable>) -> GraphResult<Vec<Self>> {
        let original = self.nodes.node_ids()?;
        let original: HashSet<_> = original.iter().copied().collect::<HashSet<_>>();
        if original.len() != self.nodes.nrows() {
            return Err(GraphError::InvalidInput(
                "Duplicate node_id in graph table".into(),
            ));
        }
        let mut membership = HashMap::new();
        for (group, nodes) in groups.iter().enumerate() {
            for id in nodes.node_ids()? {
                if !original.contains(&id) || membership.insert(id, group).is_some() {
                    return Err(GraphError::InvalidInput(
                        "Invalid or repeated node group identity".into(),
                    ));
                }
            }
        }
        let ids = self.edges.edge_ids()?;
        let sources = super::structural::column(self.edges.base(), "source")?;
        let targets = super::structural::column(self.edges.base(), "target")?;
        let mut seen = HashSet::new();
        let mut rows = vec![Vec::new(); groups.len()];
        for (row, id) in ids.into_iter().enumerate() {
            if !seen.insert(id) {
                return Err(GraphError::InvalidInput(
                    "Duplicate edge_id in graph table".into(),
                ));
            }
            let from = super::structural::id(&sources.data()[row], "source", row)?;
            let to = super::structural::id(&targets.data()[row], "target", row)?;
            if !original.contains(&from) || !original.contains(&to) {
                return Err(GraphError::InvalidInput(format!(
                    "Edge {id} has an endpoint outside the node table"
                )));
            }
            if let Some(&group) = membership.get(&from) {
                if membership.get(&to) == Some(&group) {
                    rows[group].push(row);
                }
            }
        }
        groups
            .into_iter()
            .zip(rows)
            .map(|(nodes, rows)| {
                let edges = if rows.len() == self.edges.nrows() {
                    self.edges.clone()
                } else {
                    Arc::new(EdgesTable::from_base_table(
                        self.edges.base().select_rows(&rows)?,
                    )?)
                };
                Ok(Self {
                    presentation_id: streaming::next_identity(),
                    nodes: nodes.into(),
                    edges,
                    policy: self.policy.clone(),
                })
            })
            .collect()
    }

    // Resolve all operands together. Sort/reshape retain their historical node
    // preference for shared columns; grouping requires an unambiguous component.
    fn component_for(&self, names: &[&str], prefer_nodes: bool) -> GraphResult<Component> {
        let nodes = names.iter().all(|name| self.nodes.has_column(name));
        let edges = names.iter().all(|name| self.edges.has_column(name));
        match (nodes, edges) {
            (true, false) => Ok(Component::Nodes),
            (false, true) => Ok(Component::Edges),
            (true, true) if prefer_nodes => Ok(Component::Nodes),
            (true, true) => Err(GraphError::InvalidInput(format!(
                "Ambiguous GraphTable columns {names:?}; operate on nodes or edges explicitly"
            ))),
            (false, false) => Err(GraphError::InvalidInput(format!(
                "Required columns {names:?} do not belong together to nodes or edges"
            ))),
        }
    }

    /// Create a new GraphTable from nodes and edges tables
    pub fn new(nodes: NodesTable, edges: EdgesTable) -> Self {
        Self {
            presentation_id: streaming::next_identity(),
            nodes: nodes.into(),
            edges: edges.into(),
            policy: ValidationPolicy::default(),
        }
    }

    /// Create GraphTable with custom validation policy
    pub fn with_policy(nodes: NodesTable, edges: EdgesTable, policy: ValidationPolicy) -> Self {
        Self {
            presentation_id: streaming::next_identity(),
            nodes: nodes.into(),
            edges: edges.into(),
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

    /// Return an independent node-table wrapper over the existing snapshot columns.
    pub fn to_nodes(&self) -> NodesTable {
        self.nodes.as_ref().clone()
    }

    /// Return an independent edge-table wrapper over the existing snapshot columns.
    pub fn to_edges(&self) -> EdgesTable {
        self.edges.as_ref().clone()
    }

    /// Get mutable nodes, copying that component first if it is shared.
    pub fn nodes_mut(&mut self) -> &mut NodesTable {
        self.presentation_id = streaming::next_identity();
        Arc::make_mut(&mut self.nodes)
    }

    /// Get mutable edges, copying that component first if it is shared.
    pub fn edges_mut(&mut self) -> &mut EdgesTable {
        self.presentation_id = streaming::next_identity();
        Arc::make_mut(&mut self.edges)
    }

    /// Get the validation policy
    pub fn policy(&self) -> &ValidationPolicy {
        &self.policy
    }

    /// Update validation policy
    pub fn set_policy(&mut self, policy: ValidationPolicy) {
        self.presentation_id = streaming::next_identity();
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

    /// Get nodes by their IDs
    pub fn get_nodes(
        &self,
        node_ids: &[NodeId],
    ) -> GraphResult<HashMap<NodeId, HashMap<String, AttrValue>>> {
        self.nodes.base_table().identity_rows("node_id", node_ids)
    }

    /// Get edges by their IDs
    pub fn get_edges(
        &self,
        edge_ids: &[EdgeId],
    ) -> GraphResult<HashMap<EdgeId, HashMap<String, AttrValue>>> {
        self.edges.base_table().identity_rows("edge_id", edge_ids)
    }

    /// Get all neighbors of a set of nodes
    pub fn neighbors(&self, node_ids: &[NodeId]) -> GraphResult<HashSet<NodeId>> {
        let selected: HashSet<_> = node_ids.iter().copied().collect();
        let sources = super::structural::column(self.edges.base_table(), "source")?;
        let targets = super::structural::column(self.edges.base_table(), "target")?;
        let mut neighbors = HashSet::new();
        for (row, (source, target)) in sources.iter().zip(targets.iter()).enumerate() {
            let source = super::structural::id(source, "source", row)?;
            let target = super::structural::id(target, "target", row)?;
            if selected.contains(&source) && !selected.contains(&target) {
                neighbors.insert(target);
            }
            if selected.contains(&target) && !selected.contains(&source) {
                neighbors.insert(source);
            }
        }
        Ok(neighbors)
    }

    /// Auto-assign edge IDs for null values (useful for meta nodes and imported data)
    pub fn auto_assign_edge_ids(mut self) -> GraphResult<Self> {
        self.presentation_id = streaming::next_identity();
        // Delegate to the edges table's auto_assign_edge_ids method
        self.edges = Arc::unwrap_or_clone(self.edges)
            .auto_assign_edge_ids()?
            .into();
        Ok(self)
    }

    /// Convert GraphTable into separate BaseTable components
    pub fn into_components(self) -> (BaseTable, BaseTable) {
        (
            Arc::unwrap_or_clone(self.nodes).into_base(),
            Arc::unwrap_or_clone(self.edges).into_base(),
        )
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
        // Create VizModule from this GraphTable which already implements DataSource
        // for unified graph visualization with both nodes and edges
        let data_source: Arc<dyn crate::viz::streaming::data_source::DataSource> =
            Arc::new(self.clone());
        let viz_module = VizModule::new(data_source);

        Ok(viz_module)
    }
}

// Graph bundles own two rectangles; graph operations are not a rectangular Table.
impl GraphTable {
    pub fn nrows(&self) -> usize {
        // For composite table, return total rows across both tables
        self.nodes.nrows() + self.edges.nrows()
    }

    pub fn ncols(&self) -> usize {
        // Return unique column names across both tables
        let mut columns = HashSet::new();
        columns.extend(self.nodes.column_names().iter());
        columns.extend(self.edges.column_names().iter());
        columns.len()
    }

    /// Separate node/edge column names, without a synthetic shared rectangle.
    pub fn column_names(&self) -> (&[String], &[String]) {
        (self.nodes.column_names(), self.edges.column_names())
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.nodes.has_column(name) || self.edges.has_column(name)
    }

    /// Select the first node rows and only their internal edges.
    pub fn head(&self, n: usize) -> GraphResult<Self> {
        self.induced_groups(vec![self.nodes.head(n)])?
            .pop()
            .ok_or_else(|| GraphError::InvalidInput("Missing node selection".into()))
    }

    /// Select the last node rows and only their internal edges.
    pub fn tail(&self, n: usize) -> GraphResult<Self> {
        self.induced_groups(vec![self.nodes.tail(n)])?
            .pop()
            .ok_or_else(|| GraphError::InvalidInput("Missing node selection".into()))
    }

    /// Select node rows in the clamped half-open range and their internal edges.
    pub fn slice(&self, start: usize, end: usize) -> GraphResult<Self> {
        self.induced_groups(vec![self.nodes.slice(start, end)])?
            .pop()
            .ok_or_else(|| GraphError::InvalidInput("Missing node selection".into()))
    }

    pub fn sort_by(&self, column: &str, ascending: bool) -> GraphResult<Self> {
        self.sort_values(vec![column.to_string()], vec![ascending])
    }

    pub fn sort_values(&self, columns: Vec<String>, ascending: Vec<bool>) -> GraphResult<Self> {
        match self.component_for(
            &columns.iter().map(String::as_str).collect::<Vec<_>>(),
            true,
        )? {
            Component::Nodes => Ok(Self {
                presentation_id: streaming::next_identity(),
                nodes: self.nodes.sort_values(columns, ascending)?.into(),
                edges: self.edges.clone(),
                policy: self.policy.clone(),
            }),
            Component::Edges => Ok(Self {
                presentation_id: streaming::next_identity(),
                nodes: self.nodes.clone(),
                edges: self.edges.sort_values(columns, ascending)?.into(),
                policy: self.policy.clone(),
            }),
        }
    }

    /// Apply the existing predicate to both components, then enforce node closure.
    pub fn filter(&self, predicate: &str) -> GraphResult<Self> {
        let filtered = Self {
            presentation_id: streaming::next_identity(),
            nodes: self.nodes.clone(),
            edges: self.edges.filter(predicate)?.into(),
            policy: self.policy.clone(),
        };
        filtered
            .induced_groups(vec![self.nodes.filter(predicate)?])?
            .pop()
            .ok_or_else(|| GraphError::InvalidInput("Missing filtered selection".into()))
    }

    pub fn group_by(&self, columns: &[String]) -> GraphResult<Vec<Self>> {
        match self.component_for(
            &columns.iter().map(String::as_str).collect::<Vec<_>>(),
            false,
        )? {
            Component::Nodes => self.induced_groups(self.nodes.group_by(columns)?),
            Component::Edges => self
                .edges
                .group_by(columns)?
                .into_iter()
                .map(|edges| {
                    Ok(Self {
                        presentation_id: streaming::next_identity(),
                        nodes: self.nodes.clone(),
                        edges: edges.into(),
                        policy: self.policy.clone(),
                    })
                })
                .collect(),
        }
    }

    pub fn select(&self, column_names: &[String]) -> GraphResult<Self> {
        let mut seen = HashSet::new();
        for name in column_names {
            if !seen.insert(name) {
                return Err(GraphError::InvalidInput(format!(
                    "Column '{name}' repeated"
                )));
            }
            if !self.has_column(name) {
                return Err(GraphError::InvalidInput(format!(
                    "Column '{name}' not found in GraphTable"
                )));
            }
        }
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
            presentation_id: streaming::next_identity(),
            nodes: selected_nodes.into(),
            edges: selected_edges.into(),
            policy: self.policy.clone(),
        })
    }

    pub fn with_column(&self, name: String, column: BaseArray<AttrValue>) -> GraphResult<Self> {
        // Compatibility default: with_column targets the node component.
        Ok(Self {
            presentation_id: streaming::next_identity(),
            nodes: self.nodes.with_column(name, column)?.into(),
            edges: self.edges.clone(),
            policy: self.policy.clone(),
        })
    }

    pub fn drop_columns(&self, column_names: &[String]) -> GraphResult<Self> {
        // Drop from both tables
        Ok(Self {
            presentation_id: streaming::next_identity(),
            nodes: self.nodes.drop_columns(column_names)?.into(),
            edges: self.edges.drop_columns(column_names)?.into(),
            policy: self.policy.clone(),
        })
    }

    pub fn pivot_table(
        &self,
        index_cols: &[String],
        columns_col: &str,
        values_col: &str,
        agg_func: &str,
    ) -> GraphResult<Self> {
        let mut required = index_cols.iter().map(String::as_str).collect::<Vec<_>>();
        required.extend([columns_col, values_col]);
        match self.component_for(&required, true)? {
            Component::Nodes => Ok(Self {
                presentation_id: streaming::next_identity(),
                nodes: self
                    .nodes
                    .pivot_table(index_cols, columns_col, values_col, agg_func)?
                    .into(),
                edges: self.edges.clone(),
                policy: self.policy.clone(),
            }),
            Component::Edges => Ok(Self {
                presentation_id: streaming::next_identity(),
                nodes: self.nodes.clone(),
                edges: self
                    .edges
                    .pivot_table(index_cols, columns_col, values_col, agg_func)?
                    .into(),
                policy: self.policy.clone(),
            }),
        }
    }

    pub fn melt(
        &self,
        id_vars: Option<&[String]>,
        value_vars: Option<&[String]>,
        var_name: Option<String>,
        value_name: Option<String>,
    ) -> GraphResult<Self> {
        let required = id_vars
            .into_iter()
            .flatten()
            .chain(value_vars.into_iter().flatten())
            .map(String::as_str)
            .collect::<Vec<_>>();
        match self.component_for(&required, true)? {
            Component::Nodes => Ok(Self {
                presentation_id: streaming::next_identity(),
                nodes: self
                    .nodes
                    .melt(id_vars, value_vars, var_name, value_name)?
                    .into(),
                edges: self.edges.clone(),
                policy: self.policy.clone(),
            }),
            Component::Edges => Ok(Self {
                presentation_id: streaming::next_identity(),
                nodes: self.nodes.clone(),
                edges: self
                    .edges
                    .melt(id_vars, value_vars, var_name, value_name)?
                    .into(),
                policy: self.policy.clone(),
            }),
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
    /// Copy node properties to a row-major feature matrix in table column order.
    ///
    /// Structural IDs are excluded from features and must be unique nonnegative
    /// integers. Node IDs label the rows. Empty dimensions and feature names are
    /// preserved. Every remaining column uses `FromAttrValue` conversion (null
    /// becomes zero); unsupported values return a column/row error. This owned
    /// numerical copy has no connected writeback identity.
    pub fn to_matrix<T>(&self) -> GraphResult<crate::storage::matrix::GraphMatrix<T>>
    where
        T: crate::storage::matrix::NumericType + crate::storage::matrix::FromAttrValue<T>,
    {
        Self::feature_matrix(self.nodes.base_table(), "node_id", &["id", "node_id"], true)
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

    /// Copy edge properties to a row-major feature matrix in table column order.
    ///
    /// Excludes IDs/endpoints and preserves feature names and empty dimensions.
    /// Edge IDs are validated but are not presented as node row labels. Property
    /// conversion and owned-copy semantics match [`Self::to_matrix`].
    pub fn edges_to_matrix<T>(&self) -> GraphResult<crate::storage::matrix::GraphMatrix<T>>
    where
        T: crate::storage::matrix::NumericType + crate::storage::matrix::FromAttrValue<T>,
    {
        Self::feature_matrix(
            self.edges.base_table(),
            "edge_id",
            &["id", "edge_id", "source", "target"],
            false,
        )
    }

    // A feature matrix is an owned numerical copy, not a graph adjacency or a
    // connected table. Resolve columns once, then fill the row-major buffer.
    fn feature_matrix<T>(
        table: &BaseTable,
        id_column: &str,
        structural: &[&str],
        node_labels: bool,
    ) -> GraphResult<crate::storage::matrix::GraphMatrix<T>>
    where
        T: crate::storage::matrix::NumericType + crate::storage::matrix::FromAttrValue<T>,
    {
        use crate::storage::matrix::GraphMatrix;
        let ids = super::structural::ids(table, id_column)?;
        let mut seen = HashSet::with_capacity(ids.len());
        if ids.iter().any(|id| !seen.insert(*id)) {
            return Err(GraphError::InvalidInput(format!("Duplicate {id_column}")));
        }
        let names: Vec<_> = table
            .column_names()
            .iter()
            .filter(|name| !structural.contains(&name.as_str()))
            .cloned()
            .collect();
        let columns = names
            .iter()
            .map(|name| super::structural::column(table, name))
            .collect::<GraphResult<Vec<_>>>()?;
        let count = table
            .nrows()
            .checked_mul(columns.len())
            .ok_or_else(|| GraphError::InvalidInput("Feature matrix shape overflow".into()))?;
        let mut data = Vec::new();
        data.try_reserve_exact(count)
            .map_err(|error| GraphError::InvalidInput(error.to_string()))?;
        for row in 0..table.nrows() {
            for (name, column) in names.iter().zip(&columns) {
                data.push(T::from_attr_value(&column.data()[row]).map_err(|error| {
                    GraphError::InvalidInput(format!("Feature {name} row {row}: {error}"))
                })?);
            }
        }
        let mut matrix = GraphMatrix::from_row_major_data(
            data,
            table.nrows(),
            columns.len(),
            node_labels.then_some(ids.as_slice()),
        )?;
        matrix.set_column_names(names);
        Ok(matrix)
    }

    /// Independent node and edge rectangle shapes.
    pub fn shape(&self) -> GraphTableShape {
        GraphTableShape {
            nodes: self.nodes.shape(),
            edges: self.edges.shape(),
        }
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
            version: "3.0".to_string(),
            created_at: format!("timestamp_{}", timestamp),
            node_count: graph_table.nodes.nrows(),
            edge_count: graph_table.edges.nrows(),
            validation_policy: graph_table.policy.clone(),
            checksums: HashMap::new(), // Populated only for an encoded bundle.
        }
    }
}

impl GraphTable {
    /// Save an owned, typed v3 table bundle using atomic file replacement.
    /// Source view pins/history and pooled SubgraphRef values are not serialized.
    pub fn save_bundle<P: AsRef<Path>>(&self, bundle_path: P) -> GraphResult<()> {
        bundle::save(self, bundle_path.as_ref())
    }

    /// Load a checksum-verified v3 binary or legacy v2 CSV table bundle.
    pub fn load_bundle<P: AsRef<Path>>(bundle_path: P) -> GraphResult<Self> {
        bundle::load(bundle_path.as_ref()).map(|(table, _)| table)
    }

    /// Validate checksum, format, schema and row counts before exposing metadata.
    pub fn validate_bundle<P: AsRef<Path>>(bundle_path: P) -> GraphResult<BundleMetadata> {
        bundle::load(bundle_path.as_ref()).map(|(_, metadata)| metadata)
    }

    /// Read verified metadata with its native value types for language adapters.
    pub fn bundle_metadata_value<P: AsRef<Path>>(bundle_path: P) -> GraphResult<serde_json::Value> {
        bundle::metadata_value(bundle_path.as_ref())
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
        if let Ok(f) = field.parse::<f64>() {
            return AttrValue::Float64(f);
        }
        AttrValue::Text(field.to_string())
    }
}
