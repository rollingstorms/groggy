//! NodesTable - specialized table for node data with node-specific operations

use super::base::BaseTable;
use super::base::InteractiveConfig;
use super::traits::Table;
use crate::errors::{GraphError, GraphResult};
use crate::storage::array::BaseArray;
use crate::types::{AttrValue, NodeId};
use crate::viz::display::{ColumnSchema, DataType};
use crate::viz::streaming::data_source::{
    DataSchema, DataSource, DataWindow, GraphEdge, GraphMetadata, GraphNode, LayoutAlgorithm,
    NodePosition, Position,
};
use crate::viz::VizModule;
use std::collections::{HashMap, HashSet};

/// Specialized table for node data - requires a node_id column
#[derive(Clone, Debug)]
pub struct NodesTable {
    /// Underlying BaseTable
    base: BaseTable,
}

impl NodesTable {
    /// Create a new NodesTable with node_id column
    pub fn new(node_ids: Vec<NodeId>) -> Self {
        let mut columns = HashMap::new();
        columns.insert("node_id".to_string(), BaseArray::from_node_ids(node_ids));

        let base = BaseTable::from_columns(columns).expect("Valid node table");
        Self { base }
    }

    /// Create NodesTable from BaseTable (validates node_id column exists)
    pub fn from_base_table(mut base: BaseTable) -> GraphResult<Self> {
        // Map alternative column names to standard names
        let column_mapping = [
            ("node_ids", "node_id"), // node_ids -> node_id
            ("id", "node_id"),       // id -> node_id (for general CSV files)
        ];

        // Apply column renaming for any alternative names found
        for (alt_name, standard_name) in column_mapping.iter() {
            if base.has_column(alt_name) && !base.has_column(standard_name) {
                if let Some(column) = base.column(alt_name).cloned() {
                    base = base.drop_columns(&[alt_name.to_string()])?;
                    base = base.with_column(standard_name.to_string(), column)?;
                }
            }
        }

        // Check if we now have the required column
        if base.has_column("node_id") {
            Ok(Self { base })
        } else {
            Err(crate::errors::GraphError::InvalidInput(
                "NodesTable requires 'node_id' column (also accepts 'node_ids', 'id')".to_string(),
            ))
        }
    }

    /// Get the node IDs as a typed array
    pub fn node_ids(&self) -> GraphResult<Vec<NodeId>> {
        let node_id_column = self.base.column("node_id").ok_or_else(|| {
            crate::errors::GraphError::InvalidInput("node_id column not found".to_string())
        })?;

        node_id_column.as_node_ids()
    }

    /// Add node attributes from a HashMap
    pub fn with_attributes(
        mut self,
        attr_name: String,
        attributes: HashMap<NodeId, AttrValue>,
    ) -> GraphResult<Self> {
        let node_ids = self.node_ids()?;
        let mut attr_values = Vec::new();

        for node_id in &node_ids {
            attr_values.push(attributes.get(node_id).cloned().unwrap_or(AttrValue::Null));
        }

        let attr_column = BaseArray::from_attr_values(attr_values);
        self.base = self.base.with_column(attr_name, attr_column)?;

        Ok(self)
    }

    /// Filter nodes by attribute value
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
        let column = self.base.column(attr_name).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!("Column '{}' not found", attr_name))
        })?;

        column.unique_values()
    }

    /// Convert back to BaseTable (loses node-specific typing)
    pub fn into_base_table(self) -> BaseTable {
        self.base
    }

    /// Get reference to underlying BaseTable
    pub fn base_table(&self) -> &BaseTable {
        &self.base
    }

    /// Get mutable reference to underlying BaseTable
    pub fn base_table_mut(&mut self) -> &mut BaseTable {
        &mut self.base
    }

    // =============================================================================
    // Phase 2: Node-specific validation and access methods
    // =============================================================================

    /// Validate that all UIDs (node_ids) are unique and not null
    pub fn validate_uids(&self) -> GraphResult<()> {
        let node_id_column = self
            .base
            .column("node_id")
            .ok_or_else(|| GraphError::InvalidInput("node_id column required".to_string()))?;

        let mut seen_ids = HashSet::new();
        for i in 0..node_id_column.len() {
            match node_id_column.get(i) {
                Some(AttrValue::Int(id)) => {
                    if !seen_ids.insert((*id) as NodeId) {
                        return Err(GraphError::InvalidInput(format!(
                            "Duplicate node_id found: {}",
                            id
                        )));
                    }
                }
                Some(AttrValue::Null) => {
                    return Err(GraphError::InvalidInput(
                        "Null node_id found - all node IDs must be non-null".to_string(),
                    ));
                }
                _ => {
                    return Err(GraphError::InvalidInput(
                        "Invalid node_id type - must be integer".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Node-specific validation warnings
    pub fn validate_node_structure(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check if node_id column exists
        if !self.base.has_column("node_id") {
            warnings.push("Missing required 'node_id' column".to_string());
        }

        // Check for reasonable node count
        if self.nrows() == 0 {
            warnings.push("Empty nodes table".to_string());
        } else if self.nrows() > 1_000_000 {
            warnings.push(format!(
                "Large nodes table ({} rows) - consider partitioning",
                self.nrows()
            ));
        }

        // Check for suspicious column patterns
        for col_name in self.column_names() {
            if col_name.contains(' ') {
                warnings.push(format!(
                    "Column '{}' contains spaces - may cause query issues",
                    col_name
                ));
            }
        }

        // Validate UIDs if possible
        if let Err(e) = self.validate_uids() {
            warnings.push(format!("UID validation failed: {}", e));
        }

        warnings
    }

    /// Get a node row by its UID (node_id)
    pub fn get_by_uid(&self, uid: NodeId) -> Option<HashMap<String, AttrValue>> {
        let node_id_column = self.base.column("node_id")?;

        // Find the row with matching node_id
        for i in 0..node_id_column.len() {
            match node_id_column.get(i) {
                Some(AttrValue::Int(id)) if (*id) as NodeId == uid => {
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
                }
                _ => continue,
            }
        }

        None
    }

    /// Iterator over (NodeId, row_data) pairs
    pub fn iter_with_ids(&self) -> impl Iterator<Item = (NodeId, HashMap<String, AttrValue>)> + '_ {
        let node_id_column = self.base.column("node_id");

        (0..self.nrows()).filter_map(move |i| {
            // Get node_id for this row
            let node_id = match node_id_column?.get(i) {
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

            Some((node_id, row))
        })
    }

    /// Access underlying BaseTable (Phase 2 plan method)
    pub fn base(&self) -> &BaseTable {
        &self.base
    }

    /// Convert into BaseTable (Phase 2 plan method)
    pub fn into_base(self) -> BaseTable {
        self.base
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.nrows() == 0
    }

    /// Get a column by name (alias for column method)
    pub fn get_column(&self, name: &str) -> GraphResult<&BaseArray<AttrValue>> {
        self.column(name)
            .ok_or_else(|| GraphError::InvalidInput(format!("Column '{}' not found", name)))
    }

    /// Launch interactive visualization for this nodes table
    ///
    /// Delegates to BaseTable.interactive() following the delegation pattern.
    /// The visualization will show the node data with any associated attributes.
    pub fn interactive(&self, config: Option<InteractiveConfig>) -> GraphResult<VizModule> {
        // Delegate to the base table's interactive method
        self.base.interactive(config)
    }
}

// Delegate all Table trait methods to the base table
impl Table for NodesTable {
    fn nrows(&self) -> usize {
        self.base.nrows()
    }

    fn ncols(&self) -> usize {
        self.base.ncols()
    }

    fn column_names(&self) -> &[String] {
        self.base.column_names()
    }

    fn column(&self, name: &str) -> Option<&BaseArray<AttrValue>> {
        self.base.column(name)
    }

    fn column_by_index(&self, index: usize) -> Option<&BaseArray<AttrValue>> {
        self.base.column_by_index(index)
    }

    fn has_column(&self, name: &str) -> bool {
        self.base.has_column(name)
    }

    fn head(&self, n: usize) -> Self {
        Self {
            base: self.base.head(n),
        }
    }

    fn tail(&self, n: usize) -> Self {
        Self {
            base: self.base.tail(n),
        }
    }

    fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            base: self.base.slice(start, end),
        }
    }

    fn sort_by(&self, column: &str, ascending: bool) -> GraphResult<Self> {
        Ok(Self {
            base: self.base.sort_by(column, ascending)?,
        })
    }

    fn sort_values(&self, columns: Vec<String>, ascending: Vec<bool>) -> GraphResult<Self> {
        Ok(Self {
            base: self.base.sort_values(columns, ascending)?,
        })
    }

    fn filter(&self, predicate: &str) -> GraphResult<Self> {
        Ok(Self {
            base: self.base.filter(predicate)?,
        })
    }

    fn group_by(&self, columns: &[String]) -> GraphResult<Vec<Self>> {
        let base_groups = self.base.group_by(columns)?;
        base_groups.into_iter().map(Self::from_base_table).collect()
    }

    fn select(&self, column_names: &[String]) -> GraphResult<Self> {
        // Ensure node_id is always included
        let mut cols = vec!["node_id".to_string()];
        for col in column_names {
            if col != "node_id" {
                cols.push(col.clone());
            }
        }

        Ok(Self {
            base: self.base.select(&cols)?,
        })
    }

    fn with_column(&self, name: String, column: BaseArray<AttrValue>) -> GraphResult<Self> {
        Ok(Self {
            base: self.base.with_column(name, column)?,
        })
    }

    fn drop_columns(&self, column_names: &[String]) -> GraphResult<Self> {
        // Prevent dropping node_id column
        let filtered_names: Vec<String> = column_names
            .iter()
            .filter(|&name| name != "node_id")
            .cloned()
            .collect();

        Ok(Self {
            base: self.base.drop_columns(&filtered_names)?,
        })
    }

    fn pivot_table(
        &self,
        index_cols: &[String],
        columns_col: &str,
        values_col: &str,
        agg_func: &str,
    ) -> GraphResult<Self> {
        Ok(Self {
            base: self
                .base
                .pivot_table(index_cols, columns_col, values_col, agg_func)?,
        })
    }

    fn melt(
        &self,
        id_vars: Option<&[String]>,
        value_vars: Option<&[String]>,
        var_name: Option<String>,
        value_name: Option<String>,
    ) -> GraphResult<Self> {
        Ok(Self {
            base: self.base.melt(id_vars, value_vars, var_name, value_name)?,
        })
    }
}

impl std::fmt::Display for NodesTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NodesTable[{} x {}]", self.nrows(), self.ncols())?;
        write!(f, "{}", self.base)
    }
}

impl From<Vec<NodeId>> for NodesTable {
    fn from(node_ids: Vec<NodeId>) -> Self {
        Self::new(node_ids)
    }
}

// =============================================================================
// DataSource Implementation for Graph Visualization
// =============================================================================

impl DataSource for NodesTable {
    fn total_rows(&self) -> usize {
        self.nrows()
    }

    fn total_cols(&self) -> usize {
        self.ncols()
    }

    fn get_window(&self, start: usize, count: usize) -> DataWindow {
        let end = (start + count).min(self.nrows());
        let windowed_table = self.slice(start, end);

        // Convert table data to DataWindow format
        let headers = windowed_table.column_names().to_vec();
        let mut rows = Vec::new();

        for i in 0..windowed_table.nrows() {
            let mut row = Vec::new();
            for col_name in &headers {
                if let Some(column) = windowed_table.column(col_name) {
                    if let Some(value) = column.get(i) {
                        row.push(value.clone());
                    } else {
                        row.push(AttrValue::Null);
                    }
                } else {
                    row.push(AttrValue::Null);
                }
            }
            rows.push(row);
        }

        let schema = self.get_schema();
        DataWindow::new(headers, rows, schema, self.total_rows(), start)
    }

    fn get_schema(&self) -> DataSchema {
        let mut columns = Vec::new();

        for col_name in self.column_names() {
            let data_type = if col_name == "node_id" {
                DataType::Integer
            } else if let Some(column) = self.column(col_name) {
                // Infer type from first non-null value
                let mut data_type = DataType::String; // Default fallback
                for i in 0..column.len() {
                    match column.get(i) {
                        Some(AttrValue::Int(_)) => {
                            data_type = DataType::Integer;
                            break;
                        }
                        Some(AttrValue::Float(_)) => {
                            data_type = DataType::Float;
                            break;
                        }
                        Some(AttrValue::Text(_)) => {
                            data_type = DataType::String;
                            break;
                        }
                        Some(AttrValue::Bool(_)) => {
                            data_type = DataType::Boolean;
                            break;
                        }
                        _ => continue,
                    };
                }
                data_type
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
            primary_key: Some("node_id".to_string()),
            source_type: "nodes_table".to_string(),
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
        self.column_names().to_vec()
    }

    fn get_source_id(&self) -> String {
        format!("nodes_table_{}", self.nrows())
    }

    fn get_version(&self) -> u64 {
        // Simple hash-based version - could be improved
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.nrows().hash(&mut hasher);
        self.ncols().hash(&mut hasher);
        hasher.finish()
    }

    // Graph visualization support
    fn supports_graph_view(&self) -> bool {
        true
    }

    fn get_graph_nodes(&self) -> Vec<GraphNode> {
        let mut nodes = Vec::new();

        if let Ok(node_ids) = self.node_ids() {
            for (i, node_id) in node_ids.iter().enumerate() {
                let mut attributes = HashMap::new();

                // Collect all attributes for this node
                for col_name in self.column_names() {
                    if col_name != "node_id" {
                        if let Some(column) = self.column(col_name) {
                            if let Some(value) = column.get(i) {
                                attributes.insert(col_name.clone(), value.clone());
                            }
                        }
                    }
                }

                // Try to get a label from common label columns
                let label = attributes
                    .get("label")
                    .or_else(|| attributes.get("name"))
                    .or_else(|| attributes.get("title"))
                    .map(|v| match v {
                        AttrValue::Text(s) => s.clone(),
                        AttrValue::Int(i) => i.to_string(),
                        AttrValue::Float(f) => f.to_string(),
                        _ => node_id.to_string(),
                    })
                    .or_else(|| Some(node_id.to_string()));

                nodes.push(GraphNode {
                    id: node_id.to_string(),
                    label,
                    attributes,
                    position: None, // Will be computed by layout algorithm
                });
            }
        }

        nodes
    }

    fn get_graph_edges(&self) -> Vec<GraphEdge> {
        // NodesTable doesn't contain edge information
        Vec::new()
    }

    fn get_graph_metadata(&self) -> GraphMetadata {
        let nodes = self.get_graph_nodes();
        let mut attribute_types = HashMap::new();

        // Infer attribute types from the schema
        for col_schema in self.get_schema().columns {
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
                attribute_types.insert(col_schema.name, type_name.to_string());
            }
        }

        GraphMetadata {
            node_count: nodes.len(),
            edge_count: 0,
            is_directed: false, // No edges in nodes table
            has_weights: false,
            attribute_types,
        }
    }

    fn compute_layout(&self, algorithm: LayoutAlgorithm) -> Vec<NodePosition> {
        let nodes = self.get_graph_nodes();

        match algorithm {
            LayoutAlgorithm::Circular {
                radius,
                start_angle,
            } => {
                let radius = radius.unwrap_or(200.0);
                let angle_step = 2.0 * std::f64::consts::PI / nodes.len() as f64;

                nodes
                    .into_iter()
                    .enumerate()
                    .map(|(i, node)| {
                        let angle = start_angle + i as f64 * angle_step;
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
