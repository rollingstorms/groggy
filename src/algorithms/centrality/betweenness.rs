use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;

use crate::algorithms::pathfinding::utils::checked_dijkstra_weights;
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
                    description: "Normalize ordered-pair scores by 1/((n-1)(n-2)).".to_string(),
                    value_type: ParameterType::Bool,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Bool(true)),
                },
                ParameterMetadata {
                    name: "weight_attr".to_string(),
                    description:
                        "Optional strictly positive finite numeric edge lengths; missing is one."
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
        indexer: &NodeIndexer,
        csr: &Csr,
        weight_map: Option<&HashMap<(NodeId, NodeId), f64>>,
        // Pre-allocated arrays (reused between sources)
        sigma: &mut [f64],
        distance: &mut [f64],
        predecessors: &mut [Vec<NodeId>],
        cancelled: impl Fn() -> bool,
    ) -> Result<Vec<NodeId>> {
        let n = nodes.len();
        let source_idx = indexer
            .get(source)
            .ok_or_else(|| anyhow!("source not in indexer"))?;

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

        if let Some(weights) = weight_map {
            // Weighted Dijkstra variant
            distance[source_idx] = 0.0;

            #[derive(Copy, Clone, Debug)]
            struct State {
                cost: f64,
                node: NodeId,
            }
            impl Eq for State {}
            impl PartialEq for State {
                fn eq(&self, other: &Self) -> bool {
                    self.node == other.node && self.cost.total_cmp(&other.cost).is_eq()
                }
            }
            impl Ord for State {
                fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                    other
                        .cost
                        .total_cmp(&self.cost)
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
                if cancelled() {
                    return Err(anyhow!("betweenness cancelled"));
                }
                let node_idx = indexer
                    .get(node)
                    .ok_or_else(|| anyhow!("betweenness node outside prepared rows"))?;
                if cost > distance[node_idx] {
                    continue;
                }
                order.push(node);

                // Use CSR neighbors
                for &neighbor_idx in csr.neighbors(node_idx) {
                    if neighbor_idx >= n {
                        continue;
                    }
                    let neighbor = nodes[neighbor_idx];
                    if neighbor == node {
                        continue; // Positive self-loops cannot participate in a shortest path.
                    }
                    let weight = weights.get(&(node, neighbor)).copied().ok_or_else(|| {
                        anyhow!("missing checked betweenness weight for {node} -> {neighbor}")
                    })?;
                    let next = cost + weight;
                    if !next.is_finite() || next <= cost {
                        return Err(anyhow!("weighted betweenness distance cannot advance in f64 on {node} -> {neighbor}"));
                    }
                    let current = distance[neighbor_idx];

                    if next < current {
                        distance[neighbor_idx] = next;
                        sigma[neighbor_idx] = sigma[node_idx];
                        predecessors[neighbor_idx].clear();
                        predecessors[neighbor_idx].push(node);
                        heap.push(State {
                            cost: next,
                            node: neighbor,
                        });
                    } else if next == current {
                        sigma[neighbor_idx] += sigma[node_idx];
                        predecessors[neighbor_idx].push(node);
                    }
                    if !sigma[neighbor_idx].is_finite() {
                        return Err(anyhow!("betweenness shortest-path count exceeds f64 range"));
                    }
                }
            }

            Ok(order)
        } else {
            // Unweighted BFS using CSR
            distance[source_idx] = 0.0;
            let mut stack = Vec::with_capacity(n);
            let mut queue = VecDeque::with_capacity(n);
            queue.push_back(source);

            while let Some(v) = queue.pop_front() {
                if cancelled() {
                    return Err(anyhow!("betweenness cancelled"));
                }
                stack.push(v);
                let v_idx = indexer
                    .get(v)
                    .ok_or_else(|| anyhow!("betweenness node outside prepared rows"))?;
                let v_dist = distance[v_idx];

                // Use CSR neighbors (slice - no allocation!)
                for &w_idx in csr.neighbors(v_idx) {
                    if w_idx >= n {
                        continue;
                    }
                    let w = nodes[w_idx];
                    let w_dist = distance[w_idx];

                    if w_dist < 0.0 {
                        distance[w_idx] = v_dist + 1.0;
                        queue.push_back(w);
                    }

                    if distance[w_idx] == v_dist + 1.0 {
                        sigma[w_idx] += sigma[v_idx];
                        predecessors[w_idx].push(v);
                        if !sigma[w_idx].is_finite() {
                            return Err(anyhow!(
                                "betweenness shortest-path count exceeds f64 range"
                            ));
                        }
                    }
                }
            }

            Ok(stack)
        }
    }

    fn compute_csr(
        &self,
        ctx: &mut Context,
        nodes: &[NodeId],
        indexer: &NodeIndexer,
        csr: &Csr,
        weight_map: Option<&HashMap<(NodeId, NodeId), f64>>,
        is_directed: bool,
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<(NodeId, f64)>> {
        if ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Err(anyhow!("betweenness cancelled"));
        }
        let n = nodes.len();
        // === PHASE 6: Pre-allocate Arrays ===
        let alloc_start = Instant::now();
        let mut sigma = vec![0.0; n];
        let mut distance = vec![0.0; n];
        let mut predecessors: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        let mut delta = vec![0.0; n];
        let mut centrality = vec![0.0; n];
        ctx.record_call("betweenness.allocate_arrays", alloc_start.elapsed());

        // === PHASE 7: Main Loop - Compute from Each Source ===
        let main_loop_start = Instant::now();
        let mut total_paths_computed = 0;

        for (idx, &source) in nodes.iter().enumerate() {
            if ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                return Err(anyhow!("betweenness cancelled"));
            }

            let source_start = Instant::now();

            // Compute shortest paths (arrays are reset inside)
            let order = self.shortest_paths(
                source,
                nodes,
                indexer,
                csr,
                weight_map,
                &mut sigma,
                &mut distance,
                &mut predecessors,
                || ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)),
            )?;

            total_paths_computed += order.len();

            // Reset delta array
            for i in 0..n {
                delta[i] = 0.0;
            }

            // Dependency accumulation
            for &w in order.iter().rev() {
                if ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                    return Err(anyhow!("betweenness cancelled"));
                }
                let w_idx = indexer
                    .get(w)
                    .ok_or_else(|| anyhow!("betweenness node outside prepared rows"))?;

                for &v in &predecessors[w_idx] {
                    let v_idx = indexer
                        .get(v)
                        .ok_or_else(|| anyhow!("betweenness node outside prepared rows"))?;
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

            ctx.record_call("betweenness.compute.source", source_start.elapsed());
            ctx.emit_iteration(idx, 0);
        }

        ctx.record_call("betweenness.compute.main_loop", main_loop_start.elapsed());
        ctx.record_stat("betweenness.total_sources", nodes.len() as f64);
        ctx.record_stat("betweenness.total_paths", total_paths_computed as f64);

        // === PHASE 8: Post-processing ===
        let post_start = Instant::now();

        // Divide by 2 (undirected graphs count edges twice)
        if !is_directed {
            for value in centrality.iter_mut() {
                *value /= 2.0;
            }
        }

        // Normalization
        if self.normalized && n > 2 {
            let scale =
                (if is_directed { 1.0 } else { 2.0 }) / ((n as f64 - 1.0) * (n as f64 - 2.0));
            for value in centrality.iter_mut() {
                *value *= scale;
            }
        }

        ctx.record_call("betweenness.postprocess", post_start.elapsed());

        if centrality
            .iter()
            .any(|&score| !score.is_finite() || score < 0.0)
        {
            return Err(anyhow!("nonfinite or negative betweenness score"));
        }
        if ctx.is_cancelled() || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Err(anyhow!("betweenness cancelled"));
        }
        Ok(nodes.iter().copied().zip(centrality).collect())
    }

    pub(crate) fn compute_read(
        &self,
        input: &crate::algorithms::AlgorithmRead,
        cancel: Option<&AtomicBool>,
    ) -> crate::GraphResult<Vec<(NodeId, f64)>> {
        let run = || -> Result<_> {
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                return Err(anyhow!("betweenness cancelled"));
            }
            let weights = input.checked_weights(self.weight_attr.as_deref())?;
            if self.weight_attr.is_some() && weights.values().any(|&weight| weight <= 0.0) {
                return Err(anyhow!(
                    "weighted betweenness requires strictly positive edge weights"
                ));
            }
            let nodes = input.node_ids();
            let indexer = NodeIndexer::new(nodes);
            let mut weighted = Csr::default();
            let csr = if let Some(attr) = &self.weight_attr {
                let mut edges = Vec::new();
                for edge in input.view().edge_ids() {
                    if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                        return Err(anyhow!("betweenness cancelled"));
                    }
                    let pair = input.view().edge_endpoints(edge)?;
                    let value = input
                        .view()
                        .read_state()
                        .snapshot()
                        .edge_attributes
                        .get(&edge)
                        .and_then(|attrs| attrs.get(attr));
                    let length = crate::query::weights::dijkstra_weight(value, edge)?;
                    if weights.get(&pair) == Some(&length) {
                        edges.push(edge);
                    }
                }
                // Keep every minimum-length instance; retain all selected nodes,
                // including isolates, for normalization and zero-score rows.
                build_csr_from_edges_with_scratch(
                    &mut weighted,
                    nodes.len(),
                    edges.into_iter(),
                    |node| indexer.get(node),
                    |edge| {
                        input
                            .view()
                            .read_state()
                            .snapshot()
                            .edges
                            .get(&edge)
                            .copied()
                    },
                    CsrOptions {
                        add_reverse_edges: !input.view().is_directed(),
                        sort_neighbors: true,
                    },
                );
                &weighted
            } else {
                input.csr()
            };
            self.compute_csr(
                &mut Context::default(),
                nodes,
                &indexer,
                csr,
                self.weight_attr.as_ref().map(|_| &weights),
                input.view().is_directed(),
                cancel,
            )
        };
        run().map_err(|error| crate::GraphError::InvalidInput(error.to_string()))
    }

    fn compute(&self, ctx: &mut Context, subgraph: &Subgraph) -> Result<Vec<(NodeId, f64)>> {
        let compute_start = Instant::now();
        if ctx.is_cancelled() {
            return Err(anyhow!("betweenness cancelled"));
        }
        // Validate even empty/trivial selections before early returns or CSR cache use.
        let weight_start = Instant::now();
        let checked_weights = checked_dijkstra_weights(subgraph, self.weight_attr.as_ref())?;
        if self.weight_attr.is_some() && checked_weights.values().any(|&w| w <= 0.0) {
            return Err(anyhow!(
                "weighted betweenness requires strictly positive edge weights"
            ));
        }
        let weight_map = self.weight_attr.as_ref().map(|_| checked_weights);
        ctx.record_call("betweenness.collect_weights", weight_start.elapsed());

        // === PHASE 1: Collect Nodes ===
        let collect_start = Instant::now();
        let nodes = subgraph.ordered_nodes();
        let n = nodes.len();
        ctx.record_call("betweenness.collect_nodes", collect_start.elapsed());
        ctx.record_stat("betweenness.count.input_nodes", n as f64);

        if n == 0 {
            return Ok(Vec::new());
        }

        // === PHASE 2: Build Indexer ===
        let idx_start = Instant::now();
        let indexer = NodeIndexer::new(&nodes);
        ctx.record_call("betweenness.build_indexer", idx_start.elapsed());

        // === PHASE 3: Collect Edges ===
        let edges_start = Instant::now();
        let edges = subgraph.ordered_edges();
        // Keep each minimum-length edge instance. Equal parallel edges contribute
        // distinct paths; more expensive parallel edges cannot be shortest.
        let mut shortest_edges = Vec::new();
        if let (Some(attr), Some(weights)) = (&self.weight_attr, &weight_map) {
            shortest_edges.reserve(edges.len());
            let graph = subgraph.graph();
            let graph = graph.borrow();
            for &edge in edges.iter() {
                let pair = graph.edge_endpoints(edge)?;
                let value = graph.get_edge_attr(edge, attr)?;
                let weight = crate::query::weights::dijkstra_weight(value.as_ref(), edge)?;
                if weights.get(&pair) == Some(&weight) {
                    shortest_edges.push(edge);
                }
            }
        }
        ctx.record_call("betweenness.collect_edges", edges_start.elapsed());
        ctx.record_stat("betweenness.count.input_edges", edges.len() as f64);

        // === PHASE 4: Build or Get CSR ===
        let is_directed = subgraph.graph().borrow().is_directed();
        let add_reverse = !is_directed; // Betweenness needs bidirectional

        // A weighted CSR is local: the shared topology cache includes all edges.
        let cached = if weight_map.is_none() {
            subgraph.csr_cache_get(add_reverse)
        } else {
            None
        };
        let csr = match cached {
            Some(cached) => {
                ctx.record_call(
                    "betweenness.csr_cache_hit",
                    std::time::Duration::from_nanos(0),
                );
                cached
            }
            None => {
                let cache_miss_start = Instant::now();

                let graph = subgraph.graph();
                let graph_ref = graph.borrow();
                let pool_ref = graph_ref.pool();

                let mut csr = Csr::default();
                let build_start = Instant::now();
                let endpoint_duration = build_csr_from_edges_with_scratch(
                    &mut csr,
                    nodes.len(),
                    if weight_map.is_some() {
                        shortest_edges.as_slice()
                    } else {
                        edges.as_ref()
                    }
                    .iter()
                    .copied(),
                    |nid| indexer.get(nid),
                    |eid| pool_ref.get_edge_endpoints(eid),
                    CsrOptions {
                        add_reverse_edges: add_reverse,
                        sort_neighbors: false,
                    },
                );
                let total_build = build_start.elapsed();
                let core_build = total_build.saturating_sub(endpoint_duration);

                ctx.record_call("betweenness.csr_cache_miss", cache_miss_start.elapsed());
                ctx.record_call("betweenness.collect_edge_endpoints", endpoint_duration);
                ctx.record_call("betweenness.build_csr", core_build);

                let csr_arc = Arc::new(csr);
                if weight_map.is_none() {
                    subgraph.csr_cache_store(add_reverse, csr_arc.clone());
                }
                csr_arc
            }
        };

        ctx.record_stat("betweenness.count.csr_nodes", csr.node_count() as f64);
        ctx.record_stat("betweenness.count.csr_edges", csr.neighbors.len() as f64);

        let result = self.compute_csr(
            ctx,
            &nodes,
            &indexer,
            &csr,
            weight_map.as_ref(),
            is_directed,
            None,
        )?;

        ctx.record_call("betweenness.compute", compute_start.elapsed());

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
        let start_time = Instant::now();
        let scores = self.compute(ctx, &subgraph)?;

        if ctx.persist_results()
            && scores
                .iter()
                .any(|&(_, score)| !score.is_finite() || score.abs() > f32::MAX as f64)
        {
            return Err(anyhow!("betweenness score exceeds finite f32 output range"));
        }

        // === Persist Results ===
        if ctx.persist_results() {
            let write_start = Instant::now();
            let attr_values: Vec<(NodeId, AttrValue)> = scores
                .iter()
                .map(|&(node, score)| (node, AttrValue::Float(score as f32)))
                .collect();

            subgraph
                .set_node_attr_column(self.output_attr.clone(), attr_values)
                .map_err(|err| anyhow!("failed to persist betweenness scores: {err}"))?;

            ctx.record_call("betweenness.write_attributes", write_start.elapsed());
        } else {
            let store_start = Instant::now();
            ctx.add_output(
                format!("{}.scores", self.id()),
                AlgorithmOutput::NodeScores(scores),
            );
            ctx.record_call("betweenness.store_output", store_start.elapsed());
        }

        ctx.record_call("betweenness.total_execution", start_time.elapsed());

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
    fn indexer_handles_full_width_sparse_ids_without_overflow() {
        let index = NodeIndexer::new(&[0, usize::MAX]);
        assert_eq!(index.get(0), Some(0));
        assert_eq!(index.get(usize::MAX), Some(1));
        assert_eq!(index.get(1), None);
        let dense = NodeIndexer::new(&[usize::MAX - 1, usize::MAX]);
        assert_eq!(dense.get(usize::MAX - 1), Some(0));
        assert_eq!(dense.get(usize::MAX), Some(1));
    }

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
