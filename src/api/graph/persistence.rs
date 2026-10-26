//! Active graph export/import. Durable pool/history recovery is a separate format.
use super::*;
use crate::storage::value_codec;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

const FORMAT: &str = "groggy.graph";
const VERSION: u32 = 2;

#[derive(Serialize, Deserialize)]
struct Document {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    version: Option<u32>,
    #[serde(default)]
    config: Option<Configuration>,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}
#[derive(Serialize, Deserialize)]
struct Configuration {
    graph_type: String,
}
#[derive(Serialize, Deserialize)]
struct Node {
    id: NodeId,
    #[serde(default)]
    attributes: BTreeMap<String, Value>,
}
#[derive(Serialize, Deserialize)]
struct Edge {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    id: Option<EdgeId>,
    source: NodeId,
    target: NodeId,
    #[serde(default)]
    attributes: BTreeMap<String, Value>,
}

fn io_error(path: &Path, operation: &str, error: impl std::fmt::Display) -> GraphError {
    GraphError::IoError {
        operation: operation.into(),
        path: path.to_string_lossy().into(),
        underlying_error: error.to_string(),
    }
}

pub(super) fn load(path: &Path) -> GraphResult<Graph> {
    let file = std::fs::File::open(path).map_err(|e| io_error(path, "open_graph_file", e))?;
    let document: Document = serde_json::from_reader(std::io::BufReader::new(file))
        .map_err(|e| io_error(path, "parse_graph_json", e))?;
    let decode = match (&document.format, document.version) {
        (None, None) => value_codec::decode_legacy,
        (Some(format), Some(VERSION)) if format == FORMAT => value_codec::decode,
        _ => {
            return Err(GraphError::InvalidInput(
                "Unsupported graph export format/version".into(),
            ))
        }
    };
    let typed = document.version.is_some();
    let mut config = GraphConfig::new();
    if let Some(config_data) = document.config {
        config.graph_type = match config_data.graph_type.as_str() {
            "directed" => crate::types::GraphType::Directed,
            "undirected" => crate::types::GraphType::Undirected,
            _ => {
                return Err(GraphError::InvalidInput(
                    "Invalid stored graph direction".into(),
                ))
            }
        };
    }
    let mut graph = Graph::with_config_manual(config)?;
    for node in document.nodes {
        if node.id == usize::MAX || graph.has_node(node.id) {
            return Err(GraphError::InvalidInput(format!(
                "Invalid or duplicate stored node ID {}",
                node.id
            )));
        }
        graph.pool.borrow_mut().ensure_node_id_exists(node.id)?;
        graph.space.activate_node(node.id);
        graph.change_tracker.record_node_addition(node.id);
        graph.set_node_attr_internal(
            node.id,
            "entity_type".into(),
            AttrValue::Text("base".into()),
        )?;
        for (key, value) in node.attributes {
            let value = decode(value)?;
            if key == EXTERNAL_ID {
                let id = value
                    .get_text()
                    .ok_or_else(|| GraphError::InvalidInput("external_id must be text".into()))?;
                graph.set_node_external_id(node.id, id)?;
            } else if key == "entity_type" {
                if value.get_text().as_deref() != Some("base") {
                    return Err(GraphError::InvalidInput(
                        "Meta-node export requires stored subgraph pool contents".into(),
                    ));
                }
                graph.set_node_attr_internal(node.id, key, value)?;
            } else {
                graph.set_node_attr(node.id, key, value)?;
            }
        }
    }
    for edge in document.edges {
        if !graph.has_node(edge.source) || !graph.has_node(edge.target) {
            return Err(GraphError::InvalidInput(format!(
                "Stored edge has inactive endpoints ({}, {})",
                edge.source, edge.target
            )));
        }
        let id = match edge.id {
            Some(id) => {
                if id == usize::MAX || graph.has_edge(id) {
                    return Err(GraphError::InvalidInput(format!(
                        "Invalid or duplicate stored edge ID {id}"
                    )));
                }
                graph
                    .pool
                    .borrow_mut()
                    .add_edge_with_id(id, edge.source, edge.target)?;
                graph.space.activate_edge(id, edge.source, edge.target);
                graph
                    .change_tracker
                    .record_edge_addition(id, edge.source, edge.target);
                id
            }
            None if !typed => graph.add_edge(edge.source, edge.target)?,
            None => {
                return Err(GraphError::InvalidInput(
                    "Typed export edge is missing its ID".into(),
                ))
            }
        };
        for (key, value) in edge.attributes {
            let value = decode(value)?;
            if key == EXTERNAL_ID {
                let name = value
                    .get_text()
                    .ok_or_else(|| GraphError::InvalidInput("external_id must be text".into()))?;
                graph.set_edge_external_id(id, name)?;
            } else {
                graph.set_edge_attr(id, key, value)?;
            }
        }
    }
    Ok(graph)
}

pub(super) fn save(graph: &Graph, path: &Path) -> GraphResult<()> {
    let mut nodes = Vec::new();
    let mut node_ids = graph.node_ids();
    node_ids.sort_unstable();
    for id in node_ids {
        let mut attributes = BTreeMap::new();
        for (key, value) in graph.get_node_attrs(id)? {
            if key == "entity_type" && value.get_text().as_deref() != Some("base") {
                return Err(GraphError::InvalidInput(
                    "Meta-node export requires stored subgraph pool contents".into(),
                ));
            }
            attributes.insert(key, value_codec::encode(&value)?);
        }
        nodes.push(Node { id, attributes });
    }
    let mut edges = Vec::new();
    let mut edge_ids = graph.edge_ids();
    edge_ids.sort_unstable();
    for id in edge_ids {
        let (source, target) = graph.edge_endpoints(id)?;
        let mut attributes = BTreeMap::new();
        for (key, value) in graph.get_edge_attrs(id)? {
            attributes.insert(key, value_codec::encode(&value)?);
        }
        edges.push(Edge {
            id: Some(id),
            source,
            target,
            attributes,
        });
    }
    let document = Document {
        format: Some(FORMAT.into()),
        version: Some(VERSION),
        config: Some(Configuration {
            graph_type: if graph.is_directed() {
                "directed"
            } else {
                "undirected"
            }
            .into(),
        }),
        nodes,
        edges,
    };
    // Complete validation/encoding before opening or truncating the destination.
    let bytes =
        serde_json::to_vec_pretty(&document).map_err(|e| io_error(path, "encode_graph_json", e))?;
    std::fs::write(path, bytes).map_err(|e| io_error(path, "write_graph_json", e))
}
