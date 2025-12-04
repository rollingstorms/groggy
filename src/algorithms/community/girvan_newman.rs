use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;

use crate::algorithms::community::modularity::ModularityData;
use crate::algorithms::pathfinding::utils::collect_edge_weights;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, Context, CostHint, ParameterMetadata, ParameterType,
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

        let min = *nodes.iter().min().unwrap();
        let max = *nodes.iter().max().unwrap();
        let span = (max - min) as usize + 1;

        if span <= nodes.len() * 3 / 2 {
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

/// Girvan-Newman hierarchical community detection via edge betweenness.
///
/// This algorithm iteratively removes edges with highest betweenness centrality,
/// producing a dendrogram of community structure. It's computationally expensive
/// (O(m²n)) but produces high-quality hierarchical clusterings for small graphs.
#[derive(Clone, Debug)]
pub struct GirvanNewman {
    num_levels: Option<usize>,
    modularity_threshold: Option<f64>,
    weight_attr: Option<AttrName>,
    output_attr: AttrName,
}

impl GirvanNewman {
    pub fn new(
        num_levels: Option<usize>,
        modularity_threshold: Option<f64>,
        weight_attr: Option<AttrName>,
        output_attr: AttrName,
    ) -> Self {
        Self {
            num_levels,
            modularity_threshold,
            weight_attr,
            output_attr,
        }
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "community.girvan_newman".to_string(),
            name: "Girvan-Newman".to_string(),
            description: "Hierarchical community detection via iterative edge removal based on betweenness centrality. O(m²n) complexity - best for small graphs (<10K edges).".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Cubic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "num_levels".to_string(),
                    description: "Maximum number of edge removal iterations. If None, stops when modularity stops improving.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "modularity_threshold".to_string(),
                    description: "Stop when modularity improvement falls below this threshold.".to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "weight_attr".to_string(),
                    description: "Optional edge weight attribute for weighted betweenness.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Node attribute to store community assignments.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
            ],
        }
    }

    /// Compute edge betweenness using CSR-optimized Brandes algorithm
    fn compute_edge_betweenness_csr(
        &self,
        csr: &Csr,
        nodes: &[NodeId],
        _indexer: &NodeIndexer,
        active_edges: &HashSet<(usize, usize)>,
        weight_map: Option<&HashMap<(NodeId, NodeId), f64>>,
        // Pre-allocated buffers (reused across iterations)
        distance: &mut [f64],
        sigma: &mut [f64],
        predecessors: &mut [Vec<usize>],
        delta: &mut [f64],
        queue: &mut VecDeque<usize>,
        stack: &mut Vec<usize>,
    ) -> HashMap<(usize, usize), f64> {
        let mut edge_betweenness: HashMap<(usize, usize), f64> = HashMap::new();

        // Branch based on whether we have weights
        if let Some(wmap) = weight_map {
            // Weighted version: use Dijkstra-like shortest paths
            self.compute_weighted_betweenness(
                csr,
                nodes,
                active_edges,
                wmap,
                distance,
                sigma,
                predecessors,
                delta,
                stack,
                &mut edge_betweenness,
            );
        } else {
            // Unweighted version: use BFS
            self.compute_unweighted_betweenness(
                csr,
                nodes,
                active_edges,
                distance,
                sigma,
                predecessors,
                delta,
                queue,
                stack,
                &mut edge_betweenness,
            );
        }

        edge_betweenness
    }

    /// Unweighted betweenness computation (BFS-based)
    fn compute_unweighted_betweenness(
        &self,
        csr: &Csr,
        nodes: &[NodeId],
        active_edges: &HashSet<(usize, usize)>,
        distance: &mut [f64],
        sigma: &mut [f64],
        predecessors: &mut [Vec<usize>],
        delta: &mut [f64],
        queue: &mut VecDeque<usize>,
        stack: &mut Vec<usize>,
        edge_betweenness: &mut HashMap<(usize, usize), f64>,
    ) {
        let n = nodes.len();

        for source_idx in 0..n {
            // Reset buffers
            distance.fill(f64::INFINITY);
            sigma.fill(0.0);
            for pred in predecessors.iter_mut() {
                pred.clear();
            }
            delta.fill(0.0);
            queue.clear();
            stack.clear();

            // BFS from source
            distance[source_idx] = 0.0;
            sigma[source_idx] = 1.0;
            queue.push_back(source_idx);

            while let Some(v_idx) = queue.pop_front() {
                stack.push(v_idx);

                for &w_idx in csr.neighbors(v_idx) {
                    if !active_edges.contains(&(v_idx, w_idx)) {
                        continue;
                    }

                    let alt_dist = distance[v_idx] + 1.0;

                    // First visit to w
                    if distance[w_idx as usize].is_infinite() {
                        distance[w_idx as usize] = alt_dist;
                        queue.push_back(w_idx as usize);
                    }

                    // Shortest path to w via v
                    if (distance[w_idx as usize] - alt_dist).abs() < 1e-9 {
                        sigma[w_idx as usize] += sigma[v_idx];
                        predecessors[w_idx as usize].push(v_idx);
                    }
                }
            }

            // Accumulate edge betweenness from leaves back to root
            while let Some(w_idx) = stack.pop() {
                for &v_idx in &predecessors[w_idx] {
                    if sigma[w_idx] > 0.0 {
                        let coeff = sigma[v_idx] * (1.0 + delta[w_idx]) / sigma[w_idx];
                        delta[v_idx] += coeff;

                        // Accumulate on edge (v, w) using edge-indexed storage
                        *edge_betweenness.entry((v_idx, w_idx)).or_insert(0.0) += coeff;
                    }
                }
            }
        }
    }

    /// Weighted betweenness computation (Dijkstra-based)
    fn compute_weighted_betweenness(
        &self,
        csr: &Csr,
        nodes: &[NodeId],
        active_edges: &HashSet<(usize, usize)>,
        weight_map: &HashMap<(NodeId, NodeId), f64>,
        distance: &mut [f64],
        sigma: &mut [f64],
        predecessors: &mut [Vec<usize>],
        delta: &mut [f64],
        stack: &mut Vec<usize>,
        edge_betweenness: &mut HashMap<(usize, usize), f64>,
    ) {
        let n = nodes.len();
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Copy, Clone)]
        struct State {
            idx: usize,
            dist: f64,
        }

        impl PartialEq for State {
            fn eq(&self, other: &Self) -> bool {
                self.dist == other.dist
            }
        }

        impl Eq for State {}

        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .dist
                    .partial_cmp(&self.dist)
                    .unwrap_or(Ordering::Equal)
            }
        }

        for source_idx in 0..n {
            // Reset buffers
            distance.fill(f64::INFINITY);
            sigma.fill(0.0);
            for pred in predecessors.iter_mut() {
                pred.clear();
            }
            delta.fill(0.0);
            stack.clear();

            let mut heap = BinaryHeap::new();
            distance[source_idx] = 0.0;
            sigma[source_idx] = 1.0;
            heap.push(State {
                idx: source_idx,
                dist: 0.0,
            });

            while let Some(State { idx: v_idx, dist }) = heap.pop() {
                if dist > distance[v_idx] {
                    continue;
                }
                stack.push(v_idx);

                for &w_idx in csr.neighbors(v_idx) {
                    if !active_edges.contains(&(v_idx, w_idx)) {
                        continue;
                    }

                    // Get edge weight
                    let v_node = nodes[v_idx];
                    let w_node = nodes[w_idx as usize];
                    let weight = weight_map
                        .get(&(v_node, w_node))
                        .or_else(|| weight_map.get(&(w_node, v_node)))
                        .copied()
                        .unwrap_or(1.0);

                    let alt_dist = distance[v_idx] + weight;

                    // Found shorter path
                    if alt_dist < distance[w_idx as usize] {
                        distance[w_idx as usize] = alt_dist;
                        sigma[w_idx as usize] = sigma[v_idx];
                        predecessors[w_idx as usize].clear();
                        predecessors[w_idx as usize].push(v_idx);
                        heap.push(State {
                            idx: w_idx as usize,
                            dist: alt_dist,
                        });
                    }
                    // Found equal path
                    else if (alt_dist - distance[w_idx as usize]).abs() < 1e-9 {
                        sigma[w_idx as usize] += sigma[v_idx];
                        predecessors[w_idx as usize].push(v_idx);
                    }
                }
            }

            // Accumulate edge betweenness from leaves back to root
            while let Some(w_idx) = stack.pop() {
                for &v_idx in &predecessors[w_idx] {
                    if sigma[w_idx] > 0.0 {
                        let coeff = sigma[v_idx] * (1.0 + delta[w_idx]) / sigma[w_idx];
                        delta[v_idx] += coeff;

                        // Accumulate on edge (v, w) using edge-indexed storage
                        *edge_betweenness.entry((v_idx, w_idx)).or_insert(0.0) += coeff;
                    }
                }
            }
        }
    }

    /// Compute communities from active edges using Union-Find
    fn compute_communities_unionfind(
        n: usize,
        active_edges: &HashSet<(usize, usize)>,
    ) -> Vec<usize> {
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        fn find(parent: &mut [usize], mut node: usize) -> usize {
            let mut root = node;
            while parent[root] != root {
                root = parent[root];
            }
            // Path compression
            while parent[node] != root {
                let next = parent[node];
                parent[node] = root;
                node = next;
            }
            root
        }

        fn union(parent: &mut [usize], rank: &mut [usize], u: usize, v: usize) {
            let root_u = find(parent, u);
            let root_v = find(parent, v);

            if root_u != root_v {
                if rank[root_u] < rank[root_v] {
                    parent[root_u] = root_v;
                } else if rank[root_u] > rank[root_v] {
                    parent[root_v] = root_u;
                } else {
                    parent[root_v] = root_u;
                    rank[root_u] += 1;
                }
            }
        }

        // Union all active edges
        for &(u, v) in active_edges {
            union(&mut parent, &mut rank, u, v);
        }

        // Find all roots and assign community IDs
        let mut communities = vec![0; n];
        let mut root_to_comm: FxHashMap<usize, usize> = FxHashMap::default();
        let mut next_comm = 0;

        for i in 0..n {
            let root = find(&mut parent, i);
            let comm = *root_to_comm.entry(root).or_insert_with(|| {
                let c = next_comm;
                next_comm += 1;
                c
            });
            communities[i] = comm;
        }

        communities
    }

    fn compute(&self, ctx: &mut Context, subgraph: &Subgraph) -> Result<HashMap<NodeId, usize>> {
        let t0 = Instant::now();

        // Phase 1: Collect nodes
        let nodes_start = Instant::now();
        let nodes = subgraph.ordered_nodes();
        ctx.record_call("girvan_newman.collect_nodes", nodes_start.elapsed());
        ctx.record_stat("girvan_newman.count.input_nodes", nodes.len() as f64);

        let num_nodes = nodes.len();
        if num_nodes < 2 {
            return Ok(nodes.iter().map(|&n| (n, 0)).collect());
        }

        // Phase 2: Build indexer
        let idx_start = Instant::now();
        let indexer = NodeIndexer::new(&nodes);
        ctx.record_call("girvan_newman.build_indexer", idx_start.elapsed());

        // Phase 3: Build CSR (used for all iterations)
        let edges = subgraph.ordered_edges();
        let graph_ref = subgraph.graph();
        let graph_borrow = graph_ref.borrow();

        let mut csr = Csr::default();
        let csr_time = build_csr_from_edges_with_scratch(
            &mut csr,
            nodes.len(),
            edges.iter().copied(),
            |nid| indexer.get(nid),
            |eid| graph_borrow.edge_endpoints(eid).ok(),
            CsrOptions {
                add_reverse_edges: false,
                sort_neighbors: false,
            },
        );
        ctx.record_call("girvan_newman.build_csr", csr_time);
        ctx.record_stat("girvan_newman.count.csr_edges", csr.neighbors.len() as f64);
        drop(graph_borrow);

        // Build initial edge list (as index pairs)
        let mut all_edge_pairs: Vec<(usize, usize)> = Vec::new();
        for (u_idx, &u) in nodes.iter().enumerate() {
            for &w_idx in csr.neighbors(u_idx) {
                let v = nodes[w_idx];
                if u < v {
                    all_edge_pairs.push((u_idx, w_idx));
                }
            }
        }

        // Initialize active edges (track as both directions)
        let mut active_edges: HashSet<(usize, usize)> = HashSet::new();
        for &(u_idx, v_idx) in &all_edge_pairs {
            active_edges.insert((u_idx, v_idx));
            active_edges.insert((v_idx, u_idx));
        }

        let initial_edge_count = all_edge_pairs.len();
        ctx.record_stat(
            "girvan_newman.count.initial_edges",
            initial_edge_count as f64,
        );

        // Phase 4: Pre-allocate buffers (reused across all iterations)
        let n = nodes.len();
        let mut distance = vec![f64::INFINITY; n];
        let mut sigma = vec![0.0; n];
        let mut predecessors = vec![Vec::new(); n];
        let mut delta = vec![0.0; n];
        let mut queue = VecDeque::with_capacity(n);
        let mut stack = Vec::with_capacity(n);

        let weight_map = self
            .weight_attr
            .as_ref()
            .map(|attr| collect_edge_weights(subgraph, attr));

        // Create modularity data from original edges
        let original_edges: Vec<(NodeId, NodeId)> = all_edge_pairs
            .iter()
            .map(|&(u_idx, v_idx)| (nodes[u_idx], nodes[v_idx]))
            .collect();
        let mod_data = ModularityData::new(&original_edges);

        let mut best_communities = (0..n).collect::<Vec<usize>>();
        let mut best_modularity = -1.0;

        let max_iterations = self.num_levels.unwrap_or(initial_edge_count);
        let mod_threshold = self.modularity_threshold.unwrap_or(0.0001);

        // Phase 5: Iterative edge removal
        for iteration in 0..max_iterations {
            if ctx.is_cancelled() {
                return Err(anyhow!("Girvan-Newman cancelled"));
            }

            let iter_start = Instant::now();

            // Compute current communities
            let communities_idx = Self::compute_communities_unionfind(n, &active_edges);

            // Convert to NodeId-based map for modularity
            let mut communities_map: HashMap<NodeId, usize> = HashMap::new();
            for (i, &node) in nodes.iter().enumerate() {
                communities_map.insert(node, communities_idx[i]);
            }

            // Build edge list for current active edges
            let current_edges: Vec<(NodeId, NodeId)> = active_edges
                .iter()
                .filter_map(|&(u_idx, v_idx)| {
                    if u_idx < v_idx {
                        Some((nodes[u_idx], nodes[v_idx]))
                    } else {
                        None
                    }
                })
                .collect();

            // Compute modularity
            let modularity = crate::algorithms::community::modularity::modularity(
                &communities_map,
                &current_edges,
                &mod_data,
            );

            // Track best partition
            if modularity > best_modularity {
                best_modularity = modularity;
                best_communities = communities_idx.clone();
            }

            ctx.record_stat(
                format!("girvan_newman.iteration_{}.modularity", iteration),
                modularity,
            );

            // Check stopping criterion
            if iteration > 0 && modularity < best_modularity - mod_threshold {
                ctx.record_stat("girvan_newman.count.iterations", iteration as f64);
                break;
            }

            // If no edges left, stop
            if active_edges.is_empty() {
                ctx.record_stat("girvan_newman.count.iterations", iteration as f64);
                break;
            }

            // Compute edge betweenness on remaining graph
            let betweenness_start = Instant::now();
            let edge_betweenness = self.compute_edge_betweenness_csr(
                &csr,
                &nodes,
                &indexer,
                &active_edges,
                weight_map.as_ref(),
                &mut distance,
                &mut sigma,
                &mut predecessors,
                &mut delta,
                &mut queue,
                &mut stack,
            );
            ctx.record_call(
                format!("girvan_newman.iteration_{}.compute_betweenness", iteration),
                betweenness_start.elapsed(),
            );

            // Find edge with max betweenness
            let mut max_betweenness = -1.0;
            let mut edge_to_remove: Option<(usize, usize)> = None;

            for &(u_idx, v_idx) in &active_edges {
                if u_idx < v_idx {
                    let betweenness = edge_betweenness
                        .get(&(u_idx, v_idx))
                        .copied()
                        .unwrap_or(0.0)
                        + edge_betweenness
                            .get(&(v_idx, u_idx))
                            .copied()
                            .unwrap_or(0.0);
                    if betweenness > max_betweenness {
                        max_betweenness = betweenness;
                        edge_to_remove = Some((u_idx, v_idx));
                    }
                }
            }

            // Remove the edge with highest betweenness
            if let Some((u_idx, v_idx)) = edge_to_remove {
                active_edges.remove(&(u_idx, v_idx));
                active_edges.remove(&(v_idx, u_idx));
            } else {
                ctx.record_stat("girvan_newman.count.iterations", iteration as f64);
                break;
            }

            ctx.record_call(
                format!("girvan_newman.iteration_{}.total", iteration),
                iter_start.elapsed(),
            );
        }

        ctx.record_call("girvan_newman.compute", t0.elapsed());

        // Convert best communities to NodeId map
        let result: HashMap<NodeId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, best_communities[i]))
            .collect();

        Ok(result)
    }
}

impl Algorithm for GirvanNewman {
    fn id(&self) -> &'static str {
        "community.girvan_newman"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let t0 = Instant::now();

        let communities = self.compute(ctx, &subgraph)?;

        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = communities
                .iter()
                .map(|(&node, &comm)| (node, AttrValue::Int(comm as i64)))
                .collect();

            ctx.with_scoped_timer("girvan_newman.write_attributes", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })
            .map_err(|err| anyhow!("failed to persist community assignments: {err}"))?;
        }

        ctx.record_duration("girvan_newman.total_execution", t0.elapsed());
        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = GirvanNewman::metadata_template();
    let id = metadata.id.clone();

    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let num_levels = spec.params.get_int("num_levels").map(|i| i as usize);

        let modularity_threshold = spec.params.get_float("modularity_threshold");

        let weight_attr = spec
            .params
            .get_text("weight_attr")
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string().into());

        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("community")
            .to_string()
            .into();

        Ok(Box::new(GirvanNewman::new(
            num_levels,
            modularity_threshold,
            weight_attr,
            output_attr,
        )) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::Context;
    use crate::api::graph::Graph;
    use crate::subgraphs::Subgraph;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_girvan_newman_small_graph() {
        let mut graph = Graph::new();

        // Two triangles connected by a bridge
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node();
        let n4 = graph.add_node();
        let n5 = graph.add_node();

        // Triangle 1
        graph.add_edge(n0, n1).unwrap();
        graph.add_edge(n1, n2).unwrap();
        graph.add_edge(n2, n0).unwrap();

        // Bridge
        graph.add_edge(n2, n3).unwrap();

        // Triangle 2
        graph.add_edge(n3, n4).unwrap();
        graph.add_edge(n4, n5).unwrap();
        graph.add_edge(n5, n3).unwrap();

        let nodes: HashSet<NodeId> = vec![n0, n1, n2, n3, n4, n5].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let algo = GirvanNewman::new(Some(5), None, None, "community".into());

        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr_name: AttrName = "community".to_string();

        // Get community assignments
        let comm_0 = result.get_node_attribute(n0, &attr_name).unwrap().unwrap();
        let comm_1 = result.get_node_attribute(n1, &attr_name).unwrap().unwrap();
        let comm_2 = result.get_node_attribute(n2, &attr_name).unwrap().unwrap();
        let comm_3 = result.get_node_attribute(n3, &attr_name).unwrap().unwrap();
        let comm_4 = result.get_node_attribute(n4, &attr_name).unwrap().unwrap();
        let comm_5 = result.get_node_attribute(n5, &attr_name).unwrap().unwrap();

        // Nodes 0,1,2 should be in one community
        assert_eq!(comm_0, comm_1);
        assert_eq!(comm_1, comm_2);

        // Nodes 3,4,5 should be in another community
        assert_eq!(comm_3, comm_4);
        assert_eq!(comm_4, comm_5);

        // The two communities should be different
        assert_ne!(comm_0, comm_3);
    }

    #[test]
    fn test_girvan_newman_disconnected() {
        let mut graph = Graph::new();

        // Two disconnected triangles
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node();
        let n4 = graph.add_node();
        let n5 = graph.add_node();

        // Triangle 1
        graph.add_edge(n0, n1).unwrap();
        graph.add_edge(n1, n2).unwrap();
        graph.add_edge(n2, n0).unwrap();

        // Triangle 2 (disconnected)
        graph.add_edge(n3, n4).unwrap();
        graph.add_edge(n4, n5).unwrap();
        graph.add_edge(n5, n3).unwrap();

        let nodes: HashSet<NodeId> = vec![n0, n1, n2, n3, n4, n5].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let algo = GirvanNewman::new(Some(5), None, None, "community".into());

        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr_name: AttrName = "community".to_string();

        // Get community assignments
        let comm_0 = result.get_node_attribute(n0, &attr_name).unwrap().unwrap();
        let comm_1 = result.get_node_attribute(n1, &attr_name).unwrap().unwrap();
        let comm_2 = result.get_node_attribute(n2, &attr_name).unwrap().unwrap();
        let comm_3 = result.get_node_attribute(n3, &attr_name).unwrap().unwrap();

        // Nodes within triangles should be in same community
        assert_eq!(comm_0, comm_1);
        assert_eq!(comm_1, comm_2);

        // Disconnected triangles should be in different communities
        assert_ne!(comm_0, comm_3);
    }

    #[test]
    fn test_girvan_newman_weighted() {
        let mut graph = Graph::new();

        // Two triangles connected by TWO bridges with different weights
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node();
        let n4 = graph.add_node();
        let n5 = graph.add_node();

        // Triangle 1
        graph.add_edge(n0, n1).unwrap();
        graph.add_edge(n1, n2).unwrap();
        graph.add_edge(n2, n0).unwrap();

        // Two bridges with different weights
        let bridge1 = graph.add_edge(n2, n3).unwrap(); // Low weight bridge
        let bridge2 = graph.add_edge(n1, n4).unwrap(); // High weight bridge

        // Triangle 2
        graph.add_edge(n3, n4).unwrap();
        graph.add_edge(n4, n5).unwrap();
        graph.add_edge(n5, n3).unwrap();

        // Set edge weights
        graph
            .set_edge_attr(bridge1, "weight".into(), AttrValue::Float(1.0))
            .unwrap();
        graph
            .set_edge_attr(bridge2, "weight".into(), AttrValue::Float(100.0))
            .unwrap();

        let nodes: HashSet<NodeId> = vec![n0, n1, n2, n3, n4, n5].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        // Run weighted version
        let algo_weighted =
            GirvanNewman::new(Some(1), None, Some("weight".into()), "community".into());
        let mut ctx_weighted = Context::new();
        let result_weighted = algo_weighted
            .execute(&mut ctx_weighted, subgraph.clone())
            .unwrap();

        // Run unweighted version
        let algo_unweighted = GirvanNewman::new(Some(1), None, None, "community_unweighted".into());
        let mut ctx_unweighted = Context::new();
        let result_unweighted = algo_unweighted
            .execute(&mut ctx_unweighted, subgraph)
            .unwrap();

        let attr_weighted: AttrName = "community".to_string();
        let attr_unweighted: AttrName = "community_unweighted".to_string();

        // Both versions should successfully complete
        result_weighted
            .get_node_attribute(n0, &attr_weighted)
            .unwrap()
            .unwrap();
        result_unweighted
            .get_node_attribute(n0, &attr_unweighted)
            .unwrap()
            .unwrap();
    }
}
