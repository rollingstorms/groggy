//! Graph conversion utilities
//!
//! This module provides conversion functionality for graphs and subgraphs,
//! including exports to NetworkX format and other external representations.

use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::subgraphs::subgraph::Subgraph;
use crate::types::{AttrValue, NodeId};
use std::collections::HashMap;

/// NetworkX graph representation
///
/// This struct holds the data needed to create a NetworkX graph in Python.
/// It uses standard Rust types that can be easily serialized to Python.
#[derive(Debug, Clone)]
pub struct NetworkXGraph {
    /// Graph type (directed or undirected)
    pub directed: bool,
    /// Nodes with their attributes
    pub nodes: Vec<NetworkXNode>,
    /// Edges with their attributes
    pub edges: Vec<NetworkXEdge>,
    /// Graph-level attributes
    pub graph_attrs: HashMap<String, NetworkXValue>,
}

/// NetworkX node representation
#[derive(Debug, Clone)]
pub struct NetworkXNode {
    pub id: NodeId,
    pub attributes: HashMap<String, NetworkXValue>,
}

/// NetworkX edge representation
#[derive(Debug, Clone)]
pub struct NetworkXEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub attributes: HashMap<String, NetworkXValue>,
}

/// NetworkX-compatible attribute value
#[derive(Debug, Clone)]
pub enum NetworkXValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
}

impl From<&AttrValue> for NetworkXValue {
    fn from(attr_value: &AttrValue) -> Self {
        match attr_value {
            AttrValue::Text(s) => NetworkXValue::String(s.clone()),
            AttrValue::Int(i) => NetworkXValue::Integer(*i),
            AttrValue::Float(f) => NetworkXValue::Float((*f) as f64),
            AttrValue::SmallInt(i) => NetworkXValue::Integer(*i as i64),
            AttrValue::Bool(b) => NetworkXValue::Boolean(*b),
            AttrValue::Null => NetworkXValue::Null,
            AttrValue::CompactText(s) => NetworkXValue::String(s.as_str().to_string()),
            AttrValue::FloatVec(vec) => {
                // Convert float vector to a string representation for NetworkX
                let formatted = format!("{:?}", vec);
                NetworkXValue::String(formatted)
            }
            AttrValue::Bytes(bytes) => {
                // Convert bytes to a hex string representation
                let hex_string = bytes
                    .iter()
                    .map(|b| format!("{:02x}", b))
                    .collect::<Vec<_>>()
                    .join("");
                NetworkXValue::String(format!("bytes:{}", hex_string))
            }
            AttrValue::CompressedText(compressed) => {
                // Try to decompress, fallback to a placeholder
                match compressed.decompress_text() {
                    Ok(text) => NetworkXValue::String(text),
                    Err(_) => NetworkXValue::String("<compressed_text>".to_string()),
                }
            }
            AttrValue::CompressedFloatVec(compressed) => {
                // Try to decompress, fallback to a placeholder
                match compressed.decompress_float_vec() {
                    Ok(vec) => {
                        let formatted = format!("{:?}", vec);
                        NetworkXValue::String(formatted)
                    }
                    Err(_) => NetworkXValue::String("<compressed_float_vec>".to_string()),
                }
            }
            AttrValue::SubgraphRef(id) => NetworkXValue::String(format!("subgraph_ref:{}", id)),
            AttrValue::NodeArray(nodes) => {
                let formatted = format!("{:?}", nodes);
                NetworkXValue::String(formatted)
            }
            AttrValue::EdgeArray(edges) => {
                let formatted = format!("{:?}", edges);
                NetworkXValue::String(formatted)
            }
            AttrValue::IntVec(v) => {
                let formatted = format!("{:?}", v);
                NetworkXValue::String(formatted)
            }
            AttrValue::TextVec(v) => {
                let formatted = format!("{:?}", v);
                NetworkXValue::String(formatted)
            }
            AttrValue::BoolVec(v) => {
                let formatted = format!("{:?}", v);
                NetworkXValue::String(formatted)
            }
            AttrValue::Json(s) => NetworkXValue::String(s.clone()),
        }
    }
}

/// Convert a Graph to NetworkX format
pub fn graph_to_networkx(graph: &Graph) -> GraphResult<NetworkXGraph> {
    let directed = graph.is_directed();

    // Convert nodes
    let mut nodes = Vec::new();
    for node_id in graph.node_ids() {
        let attributes = graph
            .get_node_attrs(node_id)?
            .into_iter()
            .map(|(k, v)| (k, NetworkXValue::from(&v)))
            .collect();

        nodes.push(NetworkXNode {
            id: node_id,
            attributes,
        });
    }

    // Convert edges
    let mut edges = Vec::new();
    for edge_id in graph.edge_ids() {
        let (source, target) = graph.edge_endpoints(edge_id)?;
        let attributes = graph
            .get_edge_attrs(edge_id)?
            .into_iter()
            .map(|(k, v)| (k, NetworkXValue::from(&v)))
            .collect();

        edges.push(NetworkXEdge {
            source,
            target,
            attributes,
        });
    }

    // Graph-level attributes (can be extended as needed)
    let mut graph_attrs = HashMap::new();
    graph_attrs.insert(
        "groggy_version".to_string(),
        NetworkXValue::String(crate::VERSION.to_string()),
    );
    graph_attrs.insert("directed".to_string(), NetworkXValue::Boolean(directed));

    Ok(NetworkXGraph {
        directed,
        nodes,
        edges,
        graph_attrs,
    })
}

/// Convert a Subgraph to NetworkX format
pub fn subgraph_to_networkx(subgraph: &Subgraph) -> GraphResult<NetworkXGraph> {
    let graph = subgraph.graph();
    let graph_borrow = graph.borrow();
    let directed = graph_borrow.is_directed();

    // Convert nodes (only those in the subgraph)
    let mut nodes = Vec::new();
    for &node_id in subgraph.nodes() {
        let attributes = graph_borrow
            .get_node_attrs(node_id)?
            .into_iter()
            .map(|(k, v)| (k, NetworkXValue::from(&v)))
            .collect();

        nodes.push(NetworkXNode {
            id: node_id,
            attributes,
        });
    }

    // Convert edges (only those in the subgraph)
    let mut edges = Vec::new();
    for &edge_id in subgraph.edges() {
        // Check if edge exists before trying to get attributes
        if let Ok((source, target)) = graph_borrow.edge_endpoints(edge_id) {
            // Only include edges where both endpoints are in the subgraph
            if subgraph.nodes().contains(&source) && subgraph.nodes().contains(&target) {
                match graph_borrow.get_edge_attrs(edge_id) {
                    Ok(attrs) => {
                        let attributes = attrs
                            .into_iter()
                            .map(|(k, v)| (k, NetworkXValue::from(&v)))
                            .collect();

                        edges.push(NetworkXEdge {
                            source,
                            target,
                            attributes,
                        });
                    }
                    Err(_) => {
                        // Skip edges that can't be accessed
                        continue;
                    }
                }
            }
        }
    }

    // Graph-level attributes
    let mut graph_attrs = HashMap::new();
    graph_attrs.insert(
        "groggy_version".to_string(),
        NetworkXValue::String(crate::VERSION.to_string()),
    );
    graph_attrs.insert("directed".to_string(), NetworkXValue::Boolean(directed));
    graph_attrs.insert(
        "subgraph_type".to_string(),
        NetworkXValue::String(subgraph.subgraph_type().to_string()),
    );
    graph_attrs.insert(
        "subgraph_node_count".to_string(),
        NetworkXValue::Integer(subgraph.node_count() as i64),
    );
    graph_attrs.insert(
        "subgraph_edge_count".to_string(),
        NetworkXValue::Integer(subgraph.edge_count() as i64),
    );

    Ok(NetworkXGraph {
        directed,
        nodes,
        edges,
        graph_attrs,
    })
}

/// Convert NetworkX graph back to Graph (for future bidirectional conversion)
pub fn networkx_to_graph(nx_graph: NetworkXGraph) -> GraphResult<Graph> {
    let mut graph = if nx_graph.directed {
        Graph::new_directed()
    } else {
        Graph::new_undirected()
    };

    // Create a mapping from original node IDs to new node IDs
    let mut node_id_map = HashMap::new();

    // Add nodes
    for nx_node in &nx_graph.nodes {
        let new_node_id = graph.add_node();
        node_id_map.insert(nx_node.id, new_node_id);

        // Set node attributes (skip entity_type as it's automatically managed)
        for (attr_name, nx_value) in &nx_node.attributes {
            // Skip entity_type - it's immutable and set automatically by add_node
            if attr_name == "entity_type" {
                continue;
            }
            let attr_value = networkx_value_to_attr_value(nx_value)?;
            graph.set_node_attr(new_node_id, attr_name.clone(), attr_value)?;
        }
    }

    // Add edges
    for nx_edge in &nx_graph.edges {
        let source = node_id_map.get(&nx_edge.source).ok_or_else(|| {
            GraphError::InvalidInput(format!(
                "Source node {} not found in node mapping",
                nx_edge.source
            ))
        })?;
        let target = node_id_map.get(&nx_edge.target).ok_or_else(|| {
            GraphError::InvalidInput(format!(
                "Target node {} not found in node mapping",
                nx_edge.target
            ))
        })?;

        let edge_id = graph.add_edge(*source, *target)?;

        // Set edge attributes
        for (attr_name, nx_value) in &nx_edge.attributes {
            let attr_value = networkx_value_to_attr_value(nx_value)?;
            graph.set_edge_attr(edge_id, attr_name.clone(), attr_value)?;
        }
    }

    Ok(graph)
}

/// Convert NetworkXValue to AttrValue
fn networkx_value_to_attr_value(nx_value: &NetworkXValue) -> GraphResult<AttrValue> {
    match nx_value {
        NetworkXValue::String(s) => Ok(AttrValue::Text(s.clone())),
        NetworkXValue::Integer(i) => Ok(AttrValue::Int(*i)),
        NetworkXValue::Float(f) => Ok(AttrValue::Float(*f as f32)),
        NetworkXValue::Boolean(b) => Ok(AttrValue::Bool(*b)),
        NetworkXValue::Null => Ok(AttrValue::Null),
    }
}

/// Trait to add to_networkx method to Graph
pub trait ToNetworkX {
    /// Convert to NetworkX representation
    fn to_networkx(&self) -> GraphResult<NetworkXGraph>;
}

impl ToNetworkX for Graph {
    fn to_networkx(&self) -> GraphResult<NetworkXGraph> {
        graph_to_networkx(self)
    }
}

impl ToNetworkX for Subgraph {
    fn to_networkx(&self) -> GraphResult<NetworkXGraph> {
        subgraph_to_networkx(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use crate::types::AttrValue;

    #[test]
    fn test_graph_to_networkx() {
        let mut graph = Graph::new_directed();

        // Add nodes with attributes
        let node1 = graph.add_node();
        let node2 = graph.add_node();
        graph
            .set_node_attr(
                node1,
                "name".to_string(),
                AttrValue::Text("Alice".to_string()),
            )
            .unwrap();
        graph
            .set_node_attr(
                node2,
                "name".to_string(),
                AttrValue::Text("Bob".to_string()),
            )
            .unwrap();

        // Add edge with attributes
        let edge = graph.add_edge(node1, node2).unwrap();
        graph
            .set_edge_attr(edge, "weight".to_string(), AttrValue::Float(0.8))
            .unwrap();

        // Convert to NetworkX
        let nx_graph = graph.to_networkx().unwrap();

        assert!(nx_graph.directed);
        assert_eq!(nx_graph.nodes.len(), 2);
        assert_eq!(nx_graph.edges.len(), 1);

        // Check node attributes
        let alice_node = nx_graph.nodes.iter().find(|n| n.id == node1).unwrap();
        match alice_node.attributes.get("name").unwrap() {
            NetworkXValue::String(name) => assert_eq!(name, "Alice"),
            _ => panic!("Expected string attribute"),
        }

        // Check edge attributes
        let edge_data = &nx_graph.edges[0];
        match edge_data.attributes.get("weight").unwrap() {
            NetworkXValue::Float(weight) => assert!((*weight - 0.8).abs() < 1e-6),
            _ => panic!("Expected float attribute"),
        }
    }

    #[test]
    fn test_subgraph_to_networkx() {
        let mut graph = Graph::new_undirected();

        // Create a simple graph
        let node1 = graph.add_node();
        let node2 = graph.add_node();
        let node3 = graph.add_node();

        graph
            .set_node_attr(node1, "value".to_string(), AttrValue::Int(1))
            .unwrap();
        graph
            .set_node_attr(node2, "value".to_string(), AttrValue::Int(2))
            .unwrap();
        graph
            .set_node_attr(node3, "value".to_string(), AttrValue::Int(3))
            .unwrap();

        let _edge1 = graph.add_edge(node1, node2).unwrap();
        let _edge2 = graph.add_edge(node2, node3).unwrap();

        // Create subgraph with just node2 and node3
        use std::cell::RefCell;
        use std::collections::HashSet;
        use std::rc::Rc;

        let graph_rc = Rc::new(RefCell::new(graph));
        let subgraph_nodes = HashSet::from([node2, node3]);
        let subgraph = Subgraph::from_nodes(graph_rc, subgraph_nodes, "test".to_string()).unwrap();

        // Convert subgraph to NetworkX
        let nx_graph = subgraph.to_networkx().unwrap();

        assert!(!nx_graph.directed);
        assert_eq!(nx_graph.nodes.len(), 2);
        assert_eq!(nx_graph.edges.len(), 1);

        // Check that only subgraph nodes are included
        let node_ids: Vec<NodeId> = nx_graph.nodes.iter().map(|n| n.id).collect();
        assert!(node_ids.contains(&node2));
        assert!(node_ids.contains(&node3));
        assert!(!node_ids.contains(&node1));
    }

    #[test]
    fn test_bidirectional_conversion() {
        let mut original_graph = Graph::new_undirected();

        // Create a simple graph
        let node1 = original_graph.add_node();
        let node2 = original_graph.add_node();
        original_graph
            .set_node_attr(node1, "name".to_string(), AttrValue::Text("A".to_string()))
            .unwrap();
        original_graph
            .set_node_attr(node2, "name".to_string(), AttrValue::Text("B".to_string()))
            .unwrap();

        let edge = original_graph.add_edge(node1, node2).unwrap();
        original_graph
            .set_edge_attr(edge, "weight".to_string(), AttrValue::Float(1.5))
            .unwrap();

        // Convert to NetworkX and back
        let nx_graph = original_graph.to_networkx().unwrap();
        let reconstructed_graph = networkx_to_graph(nx_graph).unwrap();

        // Check that the reconstructed graph has the same structure
        assert_eq!(reconstructed_graph.node_ids().len(), 2);
        assert_eq!(reconstructed_graph.edge_ids().len(), 1);

        // Note: Node IDs may be different after reconstruction, so we check by count
        // In a more sophisticated test, we could check attributes match
    }
}
