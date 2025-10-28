use std::collections::HashMap;
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::algorithms::pathfinding::utils::dijkstra;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

#[derive(Clone, Debug)]
pub struct DijkstraShortestPath {
    start_attr: AttrName,
    weight_attr: Option<AttrName>,
    output_attr: AttrName,
}

impl DijkstraShortestPath {
    pub fn new(start_attr: AttrName, weight_attr: Option<AttrName>, output_attr: AttrName) -> Self {
        Self {
            start_attr,
            weight_attr,
            output_attr,
        }
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "pathfinding.dijkstra".to_string(),
            name: "Dijkstra Shortest Path".to_string(),
            description: "Single-source Dijkstra shortest-path distances.".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Linearithmic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "start_attr".to_string(),
                    description: "Node attribute containing source node id.".to_string(),
                    value_type: ParameterType::Text,
                    required: true,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "weight_attr".to_string(),
                    description: "Optional edge weight attribute (defaults to 1.0).".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store distances.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("distance".to_string())),
                },
            ],
        }
    }

    fn resolve_source(&self, subgraph: &Subgraph) -> Result<NodeId> {
        subgraph
            .nodes()
            .iter()
            .find(|&&node| {
                subgraph
                    .get_node_attribute(node, &self.start_attr)
                    .ok()
                    .flatten()
                    .is_some()
            })
            .copied()
            .ok_or_else(|| anyhow!("no node in subgraph carries start attribute"))
    }

    fn run(&self, ctx: &mut Context, subgraph: &Subgraph) -> Result<HashMap<NodeId, AttrValue>> {
        let source = self.resolve_source(subgraph)?;
        let weights = self.weight_attr.clone();
        let graph_ref = subgraph.graph();
        let graph = graph_ref.borrow();

        let mut weight_map: HashMap<(NodeId, NodeId), f64> = HashMap::new();
        if let Some(attr_name) = &weights {
            for &edge_id in subgraph.edge_set() {
                if let Ok((u, v)) = graph.edge_endpoints(edge_id) {
                    if let Ok(Some(value)) = graph.get_edge_attr(edge_id, attr_name) {
                        if let Some(weight) = match value {
                            AttrValue::Float(f) => Some(f as f64),
                            AttrValue::Int(i) => Some(i as f64),
                            _ => None,
                        } {
                            weight_map.insert((u, v), weight);
                        }
                    }
                }
            }
        }

        let dist = dijkstra(subgraph, source, |u, v| {
            weight_map.get(&(u, v)).copied().unwrap_or(1.0)
        });

        ctx.emit_iteration(0, dist.len());
        Ok(dist
            .into_iter()
            .map(|(node, cost)| (node, AttrValue::Float(cost as f32)))
            .collect())
    }
}

impl Algorithm for DijkstraShortestPath {
    fn id(&self) -> &'static str {
        "pathfinding.dijkstra"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let start = Instant::now();
        let distances = self.run(ctx, &subgraph)?;
        ctx.record_duration("pathfinding.dijkstra", start.elapsed());

        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = distances
                .iter()
                .map(|(&node, value)| (node, value.clone()))
                .collect();

            ctx.with_scoped_timer("pathfinding.dijkstra.write_attrs", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })
            .map_err(|err| anyhow!("failed to persist dijkstra distances: {err}"))?;
        }
        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = DijkstraShortestPath::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let start_attr = spec.params.expect_text("start_attr")?.to_string();
        let weight_attr = spec.params.get_text("weight_attr").map(|s| s.to_string());
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("distance")
            .to_string();
        Ok(Box::new(DijkstraShortestPath::new(
            start_attr.into(),
            weight_attr.map(AttrName::from),
            output_attr.into(),
        )) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::Context;
    use crate::api::graph::Graph;
    use crate::traits::SubgraphOperations;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn dijkstra_finds_shortest_distance() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(a, c).unwrap();

        graph
            .set_node_attr(a, "source".into(), AttrValue::Bool(true))
            .unwrap();

        let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let algo = DijkstraShortestPath::new("source".into(), None, "distance".into());
        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr_name: AttrName = "distance".to_string();
        let dist_c = result.get_node_attribute(c, &attr_name).unwrap().unwrap();
        if let AttrValue::Float(distance) = dist_c {
            assert!((distance - 1.0).abs() <= f32::EPSILON);
        } else {
            panic!("expected float distance");
        }
    }
}
