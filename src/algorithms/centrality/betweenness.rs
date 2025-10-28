use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::algorithms::pathfinding::utils::collect_edge_weights;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

const EPS: f64 = 1e-9;

#[derive(Clone, Debug)]
pub struct BetweennessCentrality {
    normalized: bool,
    weight_attr: Option<AttrName>,
    output_attr: AttrName,
}

impl BetweennessCentrality {
    pub fn new(normalized: bool, weight_attr: Option<AttrName>, output_attr: AttrName) -> Self {
        Self {
            normalized,
            weight_attr,
            output_attr,
        }
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "centrality.betweenness".to_string(),
            name: "Betweenness Centrality".to_string(),
            description: "Brandes betweenness centrality for un/weighted graphs.".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Quadratic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "normalized".to_string(),
                    description: "Normalize scores by 2/((n-1)(n-2)).".to_string(),
                    value_type: ParameterType::Bool,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Bool(true)),
                },
                ParameterMetadata {
                    name: "weight_attr".to_string(),
                    description: "Optional edge weight attribute for weighted betweenness."
                        .to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store betweenness scores.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("betweenness".to_string())),
                },
            ],
        }
    }

    fn shortest_paths(
        &self,
        source: NodeId,
        nodes: &[NodeId],
        node_to_index: &HashMap<NodeId, usize>,
        neighbors_map: &std::collections::HashMap<NodeId, Vec<(NodeId, crate::types::EdgeId)>>,
        weight_map: Option<&HashMap<(NodeId, NodeId), f64>>,
        // Pre-allocated arrays (reused between sources)
        sigma: &mut Vec<f64>,
        distance: &mut Vec<f64>,
        predecessors: &mut Vec<Vec<NodeId>>,
    ) -> Result<Vec<NodeId>> {
        let n = nodes.len();
        let source_idx = node_to_index[&source];

        // Reset arrays (much faster than allocating new HashMaps!)
        for i in 0..n {
            sigma[i] = 0.0;
            distance[i] = if weight_map.is_some() {
                f64::INFINITY
            } else {
                -1.0
            };
            predecessors[i].clear();
        }
        sigma[source_idx] = 1.0;

        if weight_map.is_none() {
            // Unweighted BFS
            distance[source_idx] = 0.0;
            let mut stack = Vec::with_capacity(n);
            let mut queue = VecDeque::with_capacity(n);
            queue.push_back(source);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                let v_idx = node_to_index[&v];
                let v_dist = distance[v_idx];

                // Direct adjacency access (no subgraph.neighbors() call!)
                if let Some(neighbors) = neighbors_map.get(&v) {
                    for &(w, _edge_id) in neighbors {
                        if let Some(&w_idx) = node_to_index.get(&w) {
                            let w_dist = distance[w_idx];

                            if w_dist < 0.0 {
                                distance[w_idx] = v_dist + 1.0;
                                queue.push_back(w);
                            }

                            if (distance[w_idx] - (v_dist + 1.0)).abs() < EPS {
                                sigma[w_idx] += sigma[v_idx];
                                predecessors[w_idx].push(v);
                            }
                        }
                    }
                }
            }

            Ok(stack)
        } else {
            // Weighted Dijkstra variant
            let weights = weight_map.unwrap();
            distance[source_idx] = 0.0;

            #[derive(Copy, Clone, Debug)]
            struct State {
                cost: f64,
                node: NodeId,
            }
            impl Eq for State {}
            impl PartialEq for State {
                fn eq(&self, other: &Self) -> bool {
                    self.node == other.node && (self.cost - other.cost).abs() <= EPS
                }
            }
            impl Ord for State {
                fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                    other
                        .cost
                        .partial_cmp(&self.cost)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| self.node.cmp(&other.node))
                }
            }
            impl PartialOrd for State {
                fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                    Some(self.cmp(other))
                }
            }

            let mut heap = std::collections::BinaryHeap::with_capacity(n);
            heap.push(State {
                cost: 0.0,
                node: source,
            });

            let mut order = Vec::with_capacity(n);

            while let Some(State { cost, node }) = heap.pop() {
                let node_idx = node_to_index[&node];
                if cost > distance[node_idx] + EPS {
                    continue;
                }
                order.push(node);

                // Direct adjacency access
                if let Some(neighbors) = neighbors_map.get(&node) {
                    for &(neighbor, _edge_id) in neighbors {
                        if let Some(&neighbor_idx) = node_to_index.get(&neighbor) {
                            let weight = weights.get(&(node, neighbor)).copied().unwrap_or(1.0);
                            let next = cost + weight;
                            let current = distance[neighbor_idx];

                            if next + EPS < current {
                                distance[neighbor_idx] = next;
                                sigma[neighbor_idx] = sigma[node_idx];
                                predecessors[neighbor_idx].clear();
                                predecessors[neighbor_idx].push(node);
                                heap.push(State {
                                    cost: next,
                                    node: neighbor,
                                });
                            } else if (next - current).abs() <= EPS {
                                sigma[neighbor_idx] += sigma[node_idx];
                                predecessors[neighbor_idx].push(node);
                            }
                        }
                    }
                }
            }

            Ok(order)
        }
    }

    fn compute(&self, ctx: &mut Context, subgraph: &Subgraph) -> Result<HashMap<NodeId, f64>> {
        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        let n = nodes.len();

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

        // ⚡ OPTIMIZATION: Pre-allocate arrays (reused for ALL sources!)
        let mut sigma = vec![0.0; n];
        let mut distance = vec![0.0; n];
        let mut predecessors: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        let mut delta = vec![0.0; n];

        // Centrality scores (final result)
        let mut centrality = vec![0.0; n];

        let weight_map = self
            .weight_attr
            .as_ref()
            .map(|attr| collect_edge_weights(subgraph, attr));

        // Main loop: compute betweenness from each source
        for (idx, &source) in nodes.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("betweenness cancelled"));
            }

            // ⚡ Compute shortest paths (arrays are reset inside, not reallocated!)
            let order = self.shortest_paths(
                source,
                &nodes,
                &node_to_index,
                &neighbors_map,
                weight_map.as_ref(),
                &mut sigma,
                &mut distance,
                &mut predecessors,
            )?;

            // ⚡ Reset delta array (faster than allocating new HashMap!)
            for i in 0..n {
                delta[i] = 0.0;
            }

            // Dependency accumulation
            for &w in order.iter().rev() {
                let w_idx = node_to_index[&w];

                // ⚡ Direct array access (no HashMap lookups!)
                for &v in &predecessors[w_idx] {
                    let v_idx = node_to_index[&v];
                    let sigma_w = sigma[w_idx];
                    if sigma_w > 0.0 {
                        let sigma_v = sigma[v_idx];
                        delta[v_idx] += (sigma_v / sigma_w) * (1.0 + delta[w_idx]);
                    }
                }

                if w != source {
                    centrality[w_idx] += delta[w_idx];
                }
            }

            ctx.emit_iteration(idx, 0);
        }

        // Post-processing: divide by 2 (undirected graphs count edges twice)
        for value in centrality.iter_mut() {
            *value /= 2.0;
        }

        // Normalization
        if self.normalized && n > 2 {
            let scale = 2.0 / ((n as f64 - 1.0) * (n as f64 - 2.0));
            for value in centrality.iter_mut() {
                *value *= scale;
            }
        }

        // Convert Vec → HashMap for result
        let result: HashMap<NodeId, f64> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, centrality[i]))
            .collect();

        Ok(result)
    }
}

impl Algorithm for BetweennessCentrality {
    fn id(&self) -> &'static str {
        "centrality.betweenness"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let start = Instant::now();
        let scores = self.compute(ctx, &subgraph)?;
        ctx.record_duration("centrality.betweenness", start.elapsed());

        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = scores
                .iter()
                .map(|(&node, score)| (node, AttrValue::Float(*score as f32)))
                .collect();

            ctx.with_scoped_timer("centrality.betweenness.write_attrs", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })
            .map_err(|err| anyhow!("failed to persist betweenness scores: {err}"))?;
        }
        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = BetweennessCentrality::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let normalized = spec.params.get_bool("normalized").unwrap_or(true);
        let weight_attr = spec
            .params
            .get_text("weight_attr")
            .map(|s| AttrName::from(s.to_string()));
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("betweenness")
            .to_string();
        Ok(Box::new(BetweennessCentrality::new(
            normalized,
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
    fn betweenness_high_on_bridge() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        let d = graph.add_node();

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, a).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, b).unwrap();
        graph.add_edge(c, d).unwrap();
        graph.add_edge(d, c).unwrap();

        let nodes: HashSet<NodeId> = [a, b, c, d].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let algo = BetweennessCentrality::new(true, None, "betweenness".into());
        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr_name: AttrName = "betweenness".to_string();
        let score_b = result.get_node_attribute(b, &attr_name).unwrap().unwrap();
        let score_c = result.get_node_attribute(c, &attr_name).unwrap().unwrap();
        let score_a = result.get_node_attribute(a, &attr_name).unwrap().unwrap();

        match (score_b, score_c, score_a) {
            (AttrValue::Float(sb), AttrValue::Float(sc), AttrValue::Float(sa)) => {
                assert!(sb > sa);
                assert!(sc > sa);
            }
            _ => panic!("expected float results"),
        }
    }
}
