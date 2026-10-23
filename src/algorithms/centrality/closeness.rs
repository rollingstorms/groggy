use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;

use crate::algorithms::pathfinding::utils::{checked_dijkstra_weights, DijkstraState};
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmOutput, AlgorithmParamValue, Context, CostHint,
    ParameterMetadata, ParameterType,
};
use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

/// Efficient NodeId → dense index mapper
enum NodeIndexer {
    Dense { min_id: NodeId, indices: Vec<u32> },
    Sparse(FxHashMap<NodeId, usize>),
}

impl NodeIndexer {
    fn new(nodes: &[NodeId]) -> Self {
        if nodes.is_empty() {
            return Self::Sparse(FxHashMap::default());
        }

        let min = nodes.iter().copied().min().unwrap_or(0);
        let max = nodes.iter().copied().max().unwrap_or(0);
        let span = max.checked_sub(min).and_then(|delta| delta.checked_add(1));

        if let Some(span) = span.filter(|&span| {
            span <= nodes.len().saturating_mul(3) / 2 && nodes.len() < u32::MAX as usize
        }) {
            let mut indices = vec![u32::MAX; span];
            for (i, &node) in nodes.iter().enumerate() {
                indices[(node - min) as usize] = i as u32;
            }
            Self::Dense {
                min_id: min,
                indices,
            }
        } else {
            let mut map = FxHashMap::default();
            map.reserve(nodes.len());
            for (i, &node) in nodes.iter().enumerate() {
                map.insert(node, i);
            }
            Self::Sparse(map)
        }
    }

    fn get(&self, node: NodeId) -> Option<usize> {
        match self {
            Self::Dense { min_id, indices } => {
                let offset = node.checked_sub(*min_id)? as usize;
                indices.get(offset).and_then(|&idx| {
                    if idx == u32::MAX {
                        None
                    } else {
                        Some(idx as usize)
                    }
                })
            }
            Self::Sparse(map) => map.get(&node).copied(),
        }
    }
}

/// Direction of distances on a directed graph; undirected scores are identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClosenessDirection {
    Outgoing,
    Incoming,
}

impl ClosenessDirection {
    /// Parse the directed distance convention, case insensitively.
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_lowercase().as_str() {
            "outgoing" => Ok(Self::Outgoing),
            "incoming" => Ok(Self::Incoming),
            _ => Err(anyhow!("direction must be outgoing or incoming")),
        }
    }
}

fn transpose(csr: &Csr) -> Csr {
    let n = csr.node_count();
    let mut result = Csr {
        offsets: vec![0; n + 1],
        neighbors: vec![0; csr.neighbors.len()],
    };
    for &target in &csr.neighbors {
        result.offsets[target + 1] += 1;
    }
    for row in 0..n {
        result.offsets[row + 1] += result.offsets[row];
    }
    let mut positions = result.offsets[..n].to_vec();
    for source in 0..n {
        for &target in csr.neighbors(source) {
            result.neighbors[positions[target]] = source;
            positions[target] += 1;
        }
    }
    result
}

#[derive(Clone, Debug)]
pub struct ClosenessCentrality {
    harmonic: bool,
    direction: ClosenessDirection,
    normalized: bool,
    weight_attr: Option<AttrName>,
    output_attr: AttrName,
}

impl ClosenessCentrality {
    pub fn new(harmonic: bool, weight_attr: Option<AttrName>, output_attr: AttrName) -> Self {
        Self {
            harmonic,
            direction: ClosenessDirection::Outgoing,
            normalized: false,
            weight_attr,
            output_attr,
        }
    }

    /// Select outgoing (default) or incoming distances on directed graphs.
    pub fn with_direction(mut self, direction: ClosenessDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Normalize harmonic scores by n-1, or apply the reachable-fraction
    /// correction to classic scores. Defaults to false for compatibility.
    pub fn with_normalization(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    pub(crate) fn compute_read(
        &self,
        input: &crate::algorithms::AlgorithmRead,
        cancel: Option<&AtomicBool>,
    ) -> crate::GraphResult<Vec<(NodeId, f64)>> {
        let run = || -> Result<_> {
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                return Err(anyhow!("closeness cancelled"));
            }
            let weights = input.checked_weights(self.weight_attr.as_deref())?;
            if weights.values().any(|&weight| weight <= 0.0) {
                return Err(anyhow!("closeness edge lengths must be strictly positive"));
            }
            let incoming =
                input.view().is_directed() && self.direction == ClosenessDirection::Incoming;
            let transposed = incoming.then(|| transpose(input.csr()));
            self.compute(
                &mut Context::default(),
                transposed.as_ref().unwrap_or(input.csr()),
                input.node_ids(),
                self.weight_attr.as_ref().map(|_| &weights),
                incoming,
                cancel,
            )
        };
        run().map_err(|error| crate::GraphError::InvalidInput(error.to_string()))
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "centrality.closeness".to_string(),
            name: "Closeness Centrality".to_string(),
            description: "Closeness (classic or harmonic) centrality via BFS/Dijkstra.".to_string(),
            version: "0.2.0".to_string(),
            cost_hint: CostHint::Quadratic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "direction".into(),
                    description: "Distances from nodes ('outgoing') or to nodes ('incoming')."
                        .into(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("outgoing".into())),
                },
                ParameterMetadata {
                    name: "normalized".into(),
                    description:
                        "Normalize harmonic scores by n-1, or classic scores by reachable fraction."
                            .into(),
                    value_type: ParameterType::Bool,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Bool(false)),
                },
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

    fn compute(
        &self,
        ctx: &mut Context,
        csr: &Csr,
        nodes: &[NodeId],
        weights: Option<&HashMap<(NodeId, NodeId), f64>>,
        reverse_weights: bool,
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<(NodeId, f64)>> {
        let n = nodes.len();
        let mut scores = Vec::with_capacity(n);
        let mut distances = vec![f64::INFINITY; n];
        let mut queue = VecDeque::with_capacity(n);
        let mut heap = BinaryHeap::new();
        for (source, &node) in nodes.iter().enumerate() {
            if ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                return Err(anyhow!("closeness cancelled"));
            }
            distances.fill(f64::INFINITY);
            distances[source] = 0.0;
            if let Some(weights) = weights {
                heap.clear();
                heap.push(DijkstraState {
                    cost: 0.0,
                    node_idx: source,
                });
                while let Some(DijkstraState { cost, node_idx: u }) = heap.pop() {
                    if ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed))
                    {
                        return Err(anyhow!("closeness cancelled"));
                    }
                    if cost > distances[u] {
                        continue;
                    }
                    for &v in csr.neighbors(u) {
                        let pair = if reverse_weights {
                            (nodes[v], nodes[u])
                        } else {
                            (nodes[u], nodes[v])
                        };
                        let weight = weights.get(&pair).copied().ok_or_else(|| {
                            anyhow!("missing checked closeness weight for {pair:?}")
                        })?;
                        let candidate = cost + weight;
                        if !candidate.is_finite() {
                            return Err(anyhow!("closeness distance overflow"));
                        }
                        if candidate < distances[v] {
                            distances[v] = candidate;
                            heap.push(DijkstraState {
                                cost: candidate,
                                node_idx: v,
                            });
                        }
                    }
                }
            } else {
                queue.clear();
                queue.push_back(source);
                while let Some(u) = queue.pop_front() {
                    if ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed))
                    {
                        return Err(anyhow!("closeness cancelled"));
                    }
                    for &v in csr.neighbors(u) {
                        if distances[v].is_infinite() {
                            distances[v] = distances[u] + 1.0;
                            queue.push_back(v);
                        }
                    }
                }
            }
            let mut reachable = 0;
            let mut sum = 0.0;
            for (target, &distance) in distances.iter().enumerate() {
                if target != source && distance.is_finite() {
                    if distance <= 0.0 {
                        return Err(anyhow!(
                            "closeness requires strictly positive distances between distinct nodes"
                        ));
                    }
                    sum += if self.harmonic {
                        1.0 / distance
                    } else {
                        distance
                    };
                    reachable += 1;
                }
            }
            let mut score = if self.harmonic {
                sum
            } else if reachable > 0 {
                reachable as f64 / sum
            } else {
                0.0
            };
            if self.normalized && n > 1 {
                score *= if self.harmonic {
                    1.0 / (n - 1) as f64
                } else {
                    reachable as f64 / (n - 1) as f64
                };
            }
            if !sum.is_finite() || !score.is_finite() || score < 0.0 {
                return Err(anyhow!("nonfinite closeness score for node {node}"));
            }
            scores.push((node, score));
            ctx.emit_iteration(source, 0);
        }
        if ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Err(anyhow!("closeness cancelled"));
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
        let t0 = Instant::now();
        if ctx.is_cancelled() {
            return Err(anyhow!("closeness cancelled"));
        }
        let weights = checked_dijkstra_weights(&subgraph, self.weight_attr.as_ref())?;
        if weights.values().any(|&weight| weight <= 0.0) {
            return Err(anyhow!("closeness edge lengths must be strictly positive"));
        }

        // Phase 1: Collect nodes
        let nodes_start = Instant::now();
        let nodes = subgraph.ordered_nodes();
        ctx.record_call("closeness.collect_nodes", nodes_start.elapsed());
        ctx.record_stat("closeness.count.input_nodes", nodes.len() as f64);

        let n = nodes.len();
        if n == 0 {
            if !ctx.persist_results() {
                ctx.add_output(
                    format!("{}.scores", self.id()),
                    AlgorithmOutput::NodeScores(Vec::new()),
                );
            }
            return Ok(subgraph);
        }

        // Phase 2: Build indexer
        let idx_start = Instant::now();
        let indexer = NodeIndexer::new(&nodes);
        ctx.record_call("closeness.build_indexer", idx_start.elapsed());

        // Phase 3: Build or retrieve CSR
        let add_reverse = subgraph.graph().borrow().is_undirected();
        let csr = if let Some(cached) = subgraph.csr_cache_get(add_reverse) {
            ctx.record_call(
                "closeness.csr_cache_hit",
                std::time::Duration::from_nanos(0),
            );
            cached
        } else {
            let csr_start = Instant::now();

            let edges = subgraph.ordered_edges();
            let graph_ref = subgraph.graph();
            let graph_borrow = graph_ref.borrow();

            let mut csr_new = Csr::default();
            let csr_time = build_csr_from_edges_with_scratch(
                &mut csr_new,
                nodes.len(),
                edges.iter().copied(),
                |nid| indexer.get(nid),
                |eid| graph_borrow.edge_endpoints(eid).ok(),
                CsrOptions {
                    add_reverse_edges: add_reverse,
                    sort_neighbors: false,
                },
            );
            ctx.record_call("closeness.csr_cache_miss", csr_start.elapsed());
            ctx.record_call("closeness.build_csr", csr_time);

            let csr_arc = Arc::new(csr_new);
            subgraph.csr_cache_store(add_reverse, csr_arc.clone());
            csr_arc
        };

        ctx.record_stat("closeness.count.csr_edges", csr.neighbors.len() as f64);

        // Phase 4: Compute closeness (uses CSR-optimized BFS/Dijkstra)
        let compute_start = Instant::now();
        let incoming = !add_reverse && self.direction == ClosenessDirection::Incoming;
        // Incoming adjacency is private: the shared false cache always means
        // outgoing directed adjacency for other algorithms.
        let transposed = incoming.then(|| transpose(&csr));
        let scores = self.compute(
            ctx,
            transposed.as_ref().unwrap_or(&csr),
            &nodes,
            self.weight_attr.as_ref().map(|_| &weights),
            incoming,
            None,
        )?;
        ctx.record_call("closeness.compute", compute_start.elapsed());

        // Phase 5: Write results
        if ctx.is_cancelled() {
            return Err(anyhow!("closeness cancelled"));
        }
        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = scores
                .iter()
                .map(|&(node, score)| {
                    if !score.is_finite() {
                        return Err(anyhow!(
                            "closeness score for node {node} exceeds finite f64 property range"
                        ));
                    }
                    Ok((node, AttrValue::Float64(score)))
                })
                .collect::<Result<_>>()?;

            ctx.with_scoped_timer("closeness.write_attributes", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })
            .map_err(|err| anyhow!("failed to persist closeness scores: {err}"))?;
        } else {
            ctx.add_output(
                format!("{}.scores", self.id()),
                AlgorithmOutput::NodeScores(scores),
            );
        }

        ctx.record_duration("closeness.total_execution", t0.elapsed());
        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = ClosenessCentrality::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let harmonic = spec.params.get_bool("harmonic").unwrap_or(true);
        let normalized = spec.params.get_bool("normalized").unwrap_or(false);
        let direction = match spec.params.get_text("direction").unwrap_or("outgoing") {
            "outgoing" => ClosenessDirection::Outgoing,
            "incoming" => ClosenessDirection::Incoming,
            other => {
                return Err(anyhow!(
                    "invalid closeness direction {other:?}; use outgoing or incoming"
                ))
            }
        };
        let weight_attr = spec
            .params
            .get_text("weight_attr")
            .map(|s| AttrName::from(s.to_string()));
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("closeness")
            .to_string();
        Ok(Box::new(
            ClosenessCentrality::new(harmonic, weight_attr, output_attr.into())
                .with_direction(direction)
                .with_normalization(normalized),
        ) as Box<dyn Algorithm>)
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
            (AttrValue::Float64(sb), AttrValue::Float64(sa), AttrValue::Float64(sc)) => {
                assert!(sb > sa);
                assert!(sb > sc);
            }
            _ => panic!("expected float results"),
        }
    }
}
