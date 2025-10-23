use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::algorithms::pathfinding::utils::{collect_edge_weights, dijkstra};
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

#[derive(Clone, Debug)]
pub struct ClosenessCentrality {
    harmonic: bool,
    weight_attr: Option<AttrName>,
    output_attr: AttrName,
}

impl ClosenessCentrality {
    pub fn new(harmonic: bool, weight_attr: Option<AttrName>, output_attr: AttrName) -> Self {
        Self {
            harmonic,
            weight_attr,
            output_attr,
        }
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "centrality.closeness".to_string(),
            name: "Closeness Centrality".to_string(),
            description: "Closeness (classic or harmonic) centrality via BFS/Dijkstra.".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Quadratic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "harmonic".to_string(),
                    description: "Use harmonic closeness (sum of reciprocal distances)."
                        .to_string(),
                    value_type: ParameterType::Bool,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Bool(true)),
                },
                ParameterMetadata {
                    name: "weight_attr".to_string(),
                    description: "Optional edge weight attribute.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store closeness scores.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("closeness".to_string())),
                },
            ],
        }
    }

    fn bfs_distances(subgraph: &Subgraph, source: NodeId) -> Result<HashMap<NodeId, usize>> {
        let mut dist: HashMap<NodeId, usize> = HashMap::new();
        let mut queue = VecDeque::new();
        dist.insert(source, 0);
        queue.push_back(source);

        while let Some(node) = queue.pop_front() {
            let neighbors = subgraph.neighbors(node)?;
            for neighbor in neighbors {
                if !dist.contains_key(&neighbor) {
                    dist.insert(neighbor, dist[&node] + 1);
                    queue.push_back(neighbor);
                }
            }
        }

        Ok(dist)
    }

    fn compute(&self, ctx: &mut Context, subgraph: &Subgraph) -> Result<HashMap<NodeId, f64>> {
        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        let n = nodes.len();
        if n == 0 {
            return Ok(HashMap::new());
        }

        let weight_map = self
            .weight_attr
            .as_ref()
            .map(|attr| collect_edge_weights(subgraph, attr));

        let mut scores: HashMap<NodeId, f64> = HashMap::new();
        for (idx, &source) in nodes.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("closeness cancelled"));
            }

            let mut sum = 0.0;
            let mut reachable = 0;

            if let Some(ref weights) = weight_map {
                let dist = dijkstra(subgraph, source, |u, v| {
                    weights.get(&(u, v)).map(|w| *w).unwrap_or(1.0)
                });
                for (&target, &d) in &dist {
                    if target == source {
                        continue;
                    }
                    if self.harmonic {
                        if d > 0.0 {
                            sum += 1.0 / d;
                        }
                    } else {
                        sum += d;
                        reachable += 1;
                    }
                }
            } else {
                let dist = Self::bfs_distances(subgraph, source)?;
                for (&target, &d) in &dist {
                    if target == source {
                        continue;
                    }
                    if self.harmonic {
                        if d > 0 {
                            sum += 1.0 / d as f64;
                        }
                    } else {
                        sum += d as f64;
                        reachable += 1;
                    }
                }
            }

            let score = if self.harmonic {
                sum
            } else if reachable > 0 {
                (reachable as f64) / sum
            } else {
                0.0
            };
            scores.insert(source, score);
            ctx.emit_iteration(idx, 0);
        }

        Ok(scores)
    }
}

impl Algorithm for ClosenessCentrality {
    fn id(&self) -> &'static str {
        "centrality.closeness"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let start = Instant::now();
        let scores = self.compute(ctx, &subgraph)?;
        ctx.record_duration("centrality.closeness", start.elapsed());

        let mut attrs: HashMap<AttrName, Vec<(NodeId, AttrValue)>> = HashMap::new();
        attrs.insert(
            self.output_attr.clone(),
            scores
                .into_iter()
                .map(|(node, score)| (node, AttrValue::Float(score as f32)))
                .collect(),
        );

        subgraph
            .set_node_attrs(attrs)
            .map_err(|err| anyhow!("failed to persist closeness scores: {err}"))?;
        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = ClosenessCentrality::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let harmonic = spec.params.get_bool("harmonic").unwrap_or(true);
        let weight_attr = spec
            .params
            .get_text("weight_attr")
            .map(|s| AttrName::from(s.to_string()));
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("closeness")
            .to_string();
        Ok(Box::new(ClosenessCentrality::new(
            harmonic,
            weight_attr,
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
    fn closeness_prefers_central_node() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, a).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, b).unwrap();

        let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let algo = ClosenessCentrality::new(true, None, "closeness".into());
        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr_name: AttrName = "closeness".to_string();
        let score_b = result.get_node_attribute(b, &attr_name).unwrap().unwrap();
        let score_a = result.get_node_attribute(a, &attr_name).unwrap().unwrap();
        let score_c = result.get_node_attribute(c, &attr_name).unwrap().unwrap();

        match (score_b, score_a, score_c) {
            (AttrValue::Float(sb), AttrValue::Float(sa), AttrValue::Float(sc)) => {
                assert!(sb > sa);
                assert!(sb > sc);
            }
            _ => panic!("expected float results"),
        }
    }
}
