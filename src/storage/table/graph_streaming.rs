//! Native presentation adapter for a graph bundle's separate component tables.
use super::*;
use crate::viz::{
    display::{ColumnSchema, DataType},
    streaming::{
        data_source::{
            DataSchema, DataSource, DataWindow, GraphEdge, GraphMetadata, GraphNode,
            LayoutAlgorithm,
        },
        NodePosition,
    },
};
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
pub(super) fn next_identity() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

impl DataSource for GraphTable {
    fn total_rows(&self) -> usize {
        // For GraphTable, return total unique entities (nodes + edges)
        self.nodes.nrows() + self.edges.nrows()
    }

    fn total_cols(&self) -> usize {
        1 + self.nodes.ncols() + self.edges.ncols()
    }

    fn get_window(&self, start: usize, count: usize) -> DataWindow {
        let total = self.total_rows();
        let node_count = self.nodes.nrows();
        let begin = start.min(total);
        let end = start.saturating_add(count).min(total);
        let schema = self.get_schema();
        let headers = schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect();
        let mut rows = Vec::new();
        let node_start = begin.min(node_count);
        let node_end = end.min(node_count);
        for values in self
            .nodes
            .get_window(node_start, node_end - node_start)
            .rows
        {
            let mut row = Vec::with_capacity(self.total_cols());
            row.push(AttrValue::Text("node".into()));
            row.extend(values);
            row.resize(self.total_cols(), AttrValue::Null);
            rows.push(row);
        }
        let edge_start = begin.saturating_sub(node_count);
        let edge_end = end.saturating_sub(node_count);
        for values in self
            .edges
            .get_window(edge_start, edge_end - edge_start)
            .rows
        {
            let mut row = Vec::with_capacity(self.total_cols());
            row.push(AttrValue::Text("edge".into()));
            row.resize(1 + self.nodes.ncols(), AttrValue::Null);
            row.extend(values);
            rows.push(row);
        }
        DataWindow::new(headers, rows, schema, total, start)
    }

    fn get_schema(&self) -> DataSchema {
        let mut columns = vec![ColumnSchema {
            name: "entity_kind".into(),
            data_type: DataType::String,
        }];
        for (prefix, schema) in [
            ("node", self.nodes.base().get_schema()),
            ("edge", self.edges.base().get_schema()),
        ] {
            columns.extend(schema.columns.into_iter().map(|mut column| {
                column.name = format!("{prefix}_{}", column.name);
                column
            }));
        }
        DataSchema {
            columns,
            primary_key: None,
            source_type: "graph_table_projection".into(),
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
        format!("graph-table-{}", self.presentation_id)
    }

    fn get_version(&self) -> u64 {
        1
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
