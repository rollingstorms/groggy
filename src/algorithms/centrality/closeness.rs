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

    fn bfs_distances(
        source: NodeId,
        node_to_index: &HashMap<NodeId, usize>,
        neighbors_map: &std::collections::HashMap<NodeId, Vec<(NodeId, crate::types::EdgeId)>>,
        distance: &mut Vec<f64>,
    ) -> Result<()> {
        let source_idx = node_to_index[&source];
        let mut queue = VecDeque::new();
        distance[source_idx] = 0.0;
        queue.push_back(source);

        while let Some(node) = queue.pop_front() {
            let node_idx = node_to_index[&node];
            let node_dist = distance[node_idx];

            // Direct adjacency access (no subgraph.neighbors() call!)
            if let Some(neighbors) = neighbors_map.get(&node) {
                for &(neighbor, _edge_id) in neighbors {
                    if let Some(&neighbor_idx) = node_to_index.get(&neighbor) {
                        if distance[neighbor_idx] < 0.0 {
                            // Not visited
                            distance[neighbor_idx] = node_dist + 1.0;
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn compute(&self, ctx: &mut Context, subgraph: &Subgraph) -> Result<HashMap<NodeId, f64>> {
        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        let n = nodes.len();
        if n == 0 {
            return Ok(HashMap::new());
        }

        // ⚡ OPTIMIZATION: Get adjacency snapshot ONCE (not per source!)
        let graph = subgraph.graph();
        let graph_ref = graph.borrow();
        let pool = graph_ref.pool();
        let space = graph_ref.space();
        let (_, _, _, neighbors_map) = space.snapshot(&pool);

        // ⚡ OPTIMIZATION: Create node ID → index mapping for O(1) array access
        let node_to_index: HashMap<NodeId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();

        // ⚡ OPTIMIZATION: Pre-allocate distance array (reused for ALL sources!)
        let mut distance = vec![-1.0; n];

        let weight_map = self
            .weight_attr
            .as_ref()
            .map(|attr| collect_edge_weights(subgraph, attr));

        let mut scores: Vec<f64> = vec![0.0; n];

        for (idx, &source) in nodes.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("closeness cancelled"));
            }

            let source_idx = node_to_index[&source];
            let mut sum = 0.0;
            let mut reachable = 0;

            if let Some(ref weights) = weight_map {
                // Weighted case: use Dijkstra
                let dist = dijkstra(subgraph, source, |u, v| {
                    weights.get(&(u, v)).copied().unwrap_or(1.0)
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
                // ⚡ Unweighted case: use optimized BFS
                // Reset distance array (much faster than allocating HashMap!)
                for i in 0..n {
                    distance[i] = -1.0;
                }

                Self::bfs_distances(source, &node_to_index, &neighbors_map, &mut distance)?;

                // Sum distances from distance array
                for i in 0..n {
                    if i == source_idx {
                        continue;
                    }
                    let d = distance[i];
                    if d > 0.0 {
                        if self.harmonic {
                            sum += 1.0 / d;
                        } else {
                            sum += d;
                            reachable += 1;
                        }
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
            scores[source_idx] = score;
            ctx.emit_iteration(idx, 0);
        }

        // Convert Vec → HashMap for result
        let result: HashMap<NodeId, f64> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, scores[i]))
            .collect();

        Ok(result)
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

        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = scores
                .iter()
                .map(|(&node, score)| (node, AttrValue::Float(*score as f32)))
                .collect();

            ctx.with_scoped_timer("centrality.closeness.write_attrs", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })
            .map_err(|err| anyhow!("failed to persist closeness scores: {err}"))?;
        }
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
