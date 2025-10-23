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
        subgraph: &Subgraph,
        source: NodeId,
        weight_map: Option<&HashMap<(NodeId, NodeId), f64>>,
    ) -> Result<(
        Vec<NodeId>,
        HashMap<NodeId, Vec<NodeId>>,
        HashMap<NodeId, f64>,
    )> {
        if weight_map.is_none() {
            // Unweighted BFS
            let mut stack = Vec::new();
            let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
            let mut sigma: HashMap<NodeId, f64> =
                subgraph.nodes().iter().map(|&v| (v, 0.0)).collect();
            sigma.insert(source, 1.0);

            let mut distance: HashMap<NodeId, i64> =
                subgraph.nodes().iter().map(|&v| (v, -1)).collect();
            distance.insert(source, 0);

            let mut queue = VecDeque::new();
            queue.push_back(source);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                let neighbors = subgraph.neighbors(v)?;
                for w in neighbors {
                    if distance[&w] < 0 {
                        distance.insert(w, distance[&v] + 1);
                        queue.push_back(w);
                    }
                    if distance[&w] == distance[&v] + 1 {
                        let sigma_v = sigma[&v];
                        if let Some(val) = sigma.get_mut(&w) {
                            *val += sigma_v;
                        }
                        predecessors.entry(w).or_default().push(v);
                    }
                }
            }

            Ok((stack, predecessors, sigma))
        } else {
            // Weighted Dijkstra variant
            let weights = weight_map.unwrap();
            let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
            let mut sigma: HashMap<NodeId, f64> =
                subgraph.nodes().iter().map(|&v| (v, 0.0)).collect();
            sigma.insert(source, 1.0);

            let mut dist: HashMap<NodeId, f64> = subgraph
                .nodes()
                .iter()
                .map(|&v| (v, f64::INFINITY))
                .collect();
            dist.insert(source, 0.0);

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

            let mut heap = std::collections::BinaryHeap::new();
            heap.push(State {
                cost: 0.0,
                node: source,
            });

            let mut order = Vec::new();

            while let Some(State { cost, node }) = heap.pop() {
                if cost > dist[&node] + EPS {
                    continue;
                }
                order.push(node);
                let neighbors = subgraph.neighbors(node)?;
                for neighbor in neighbors {
                    let weight = weights.get(&(node, neighbor)).copied().unwrap_or(1.0);
                    let next = cost + weight;
                    let current = dist.get(&neighbor).copied().unwrap_or(f64::INFINITY);
                    if next + EPS < current {
                        dist.insert(neighbor, next);
                        sigma.insert(neighbor, sigma[&node]);
                        predecessors.insert(neighbor, vec![node]);
                        heap.push(State {
                            cost: next,
                            node: neighbor,
                        });
                    } else if (next - current).abs() <= EPS {
                        let sigma_node = sigma[&node];
                        if let Some(val) = sigma.get_mut(&neighbor) {
                            *val += sigma_node;
                        }
                        predecessors.entry(neighbor).or_default().push(node);
                    }
                }
            }

            Ok((order, predecessors, sigma))
        }
    }

    fn compute(&self, ctx: &mut Context, subgraph: &Subgraph) -> Result<HashMap<NodeId, f64>> {
        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        let n = nodes.len();
        let mut centrality: HashMap<NodeId, f64> = nodes.iter().map(|&v| (v, 0.0)).collect();

        let weight_map = self
            .weight_attr
            .as_ref()
            .map(|attr| collect_edge_weights(subgraph, attr));

        for (idx, &source) in nodes.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("betweenness cancelled"));
            }

            let (order, predecessors, sigma) =
                self.shortest_paths(subgraph, source, weight_map.as_ref())?;

            let mut delta: HashMap<NodeId, f64> = nodes.iter().map(|&v| (v, 0.0)).collect();
            for &w in order.iter().rev() {
                if let Some(preds) = predecessors.get(&w) {
                    for &v in preds {
                        let sigma_w = sigma[&w];
                        if sigma_w > 0.0 {
                            let sigma_v = sigma[&v];
                            let addition = (sigma_v / sigma_w) * (1.0 + delta[&w]);
                            delta.entry(v).and_modify(|d| *d += addition);
                        }
                    }
                }
                if w != source {
                    centrality.entry(w).and_modify(|c| *c += delta[&w]);
                }
            }

            ctx.emit_iteration(idx, 0);
        }

        for value in centrality.values_mut() {
            *value /= 2.0;
        }

        if self.normalized && n > 2 {
            let scale = 2.0 / ((n as f64 - 1.0) * (n as f64 - 2.0));
            for value in centrality.values_mut() {
                *value *= scale;
            }
        }

        Ok(centrality)
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
            .map_err(|err| anyhow!("failed to persist betweenness scores: {err}"))?;
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
