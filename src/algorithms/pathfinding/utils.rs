use std::collections::{BinaryHeap, HashMap, VecDeque};

use rustc_hash::FxHashMap;

use crate::state::topology::Csr;
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

/// Efficient NodeId → dense index mapper (used by CSR-optimized functions)
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

pub fn bfs_layers(subgraph: &Subgraph, source: NodeId) -> HashMap<NodeId, usize> {
    // Try CSR path first for optimal performance
    if let Some(csr) = subgraph.csr_cache_get(false) {
        let nodes = subgraph.ordered_nodes();
        let indexer = NodeIndexer::new(&nodes);

        if let Some(source_idx) = indexer.get(source) {
            let n = nodes.len();
            let mut distances = vec![usize::MAX; n];
            let mut queue = VecDeque::with_capacity(n);

            // BFS using CSR
            distances[source_idx] = 0;
            queue.push_back(source_idx);

            while let Some(u) = queue.pop_front() {
                let current_dist = distances[u];
                for &v in csr.neighbors(u) {
                    if distances[v] == usize::MAX {
                        distances[v] = current_dist + 1;
                        queue.push_back(v);
                    }
                }
            }

            // Convert to HashMap
            let mut result = HashMap::new();
            for (i, &node) in nodes.iter().enumerate() {
                if distances[i] != usize::MAX {
                    result.insert(node, distances[i]);
                }
            }
            return result;
        }
    }

    // Fallback to trait-based implementation
    let mut dist = HashMap::new();
    let mut queue = VecDeque::new();
    dist.insert(source, 0);
    queue.push_back(source);

    while let Some(node) = queue.pop_front() {
        if let Ok(neighbors) = subgraph.neighbors(node) {
            for neighbor in neighbors {
                if !dist.contains_key(&neighbor) {
                    dist.insert(neighbor, dist[&node] + 1);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    dist
}

#[derive(Copy, Clone, Debug)]
struct State {
    pub cost: f64,
    pub node: NodeId,
}

impl Eq for State {}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && (self.cost - other.cost).abs() <= f64::EPSILON
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

pub fn dijkstra<SF>(subgraph: &Subgraph, source: NodeId, mut weight_fn: SF) -> HashMap<NodeId, f64>
where
    SF: FnMut(NodeId, NodeId) -> f64,
{
    // Try CSR path first for optimal performance
    if let Some(csr) = subgraph.csr_cache_get(false) {
        let nodes = subgraph.ordered_nodes();
        let indexer = NodeIndexer::new(&nodes);

        if let Some(source_idx) = indexer.get(source) {
            let n = nodes.len();
            let mut distances = vec![f64::INFINITY; n];
            let mut heap = BinaryHeap::with_capacity(n);

            // Dijkstra using CSR
            distances[source_idx] = 0.0;
            heap.push(DijkstraState {
                cost: 0.0,
                node_idx: source_idx,
            });

            while let Some(DijkstraState { cost, node_idx: u }) = heap.pop() {
                if cost > distances[u] + f64::EPSILON {
                    continue;
                }

                let u_id = nodes[u];
                for &v in csr.neighbors(u) {
                    let v_id = nodes[v];
                    let weight = weight_fn(u_id, v_id);
                    let next_cost = cost + weight;

                    if next_cost + f64::EPSILON < distances[v] {
                        distances[v] = next_cost;
                        heap.push(DijkstraState {
                            cost: next_cost,
                            node_idx: v,
                        });
                    }
                }
            }

            // Convert to HashMap
            let mut result = HashMap::new();
            for (i, &node) in nodes.iter().enumerate() {
                if distances[i] < f64::INFINITY {
                    result.insert(node, distances[i]);
                }
            }
            return result;
        }
    }

    // Fallback to trait-based implementation
    let mut dist: HashMap<NodeId, f64> = HashMap::new();
    let mut heap: BinaryHeap<State> = BinaryHeap::new();

    dist.insert(source, 0.0);
    heap.push(State {
        cost: 0.0,
        node: source,
    });

    while let Some(State { cost, node }) = heap.pop() {
        if cost > dist[&node] + f64::EPSILON {
            continue;
        }
        if let Ok(neighbors) = subgraph.neighbors(node) {
            for neighbor in neighbors {
                let weight = weight_fn(node, neighbor);
                let next = cost + weight;
                let best = dist.get(&neighbor).copied().unwrap_or(f64::INFINITY);
                if next + f64::EPSILON < best {
                    dist.insert(neighbor, next);
                    heap.push(State {
                        cost: next,
                        node: neighbor,
                    });
                }
            }
        }
    }

    dist
}

pub fn collect_edge_weights(
    subgraph: &Subgraph,
    attr: &AttrName,
) -> HashMap<(NodeId, NodeId), f64> {
    let graph_ref = subgraph.graph();
    let graph = graph_ref.borrow();
    let mut weights = HashMap::new();
    for &edge_id in subgraph.edge_set() {
        if let Ok((u, v)) = graph.edge_endpoints(edge_id) {
            if let Ok(Some(value)) = graph.get_edge_attr(edge_id, attr) {
                if let Some(weight) = match value {
                    AttrValue::Float(f) => Some(f as f64),
                    AttrValue::Int(i) => Some(i as f64),
                    _ => None,
                } {
                    weights.insert((u, v), weight);
                }
            }
        }
    }
    weights
}

/// CSR-optimized BFS traversal with pre-allocated buffers (zero inner-loop allocations).
///
/// # Arguments
/// * `csr` - Pre-built CSR adjacency structure
/// * `nodes` - Original NodeId array (for reverse mapping)
/// * `source_idx` - Dense index of source node
/// * `distances` - Pre-allocated distance buffer (will be cleared and filled)
/// * `queue` - Pre-allocated queue buffer (will be cleared and reused)
///
/// # Returns
/// Number of reachable nodes (including source)
pub fn bfs_layers_csr(
    csr: &Csr,
    _nodes: &[NodeId],
    source_idx: usize,
    distances: &mut Vec<usize>,
    queue: &mut VecDeque<usize>,
) -> usize {
    let n = csr.node_count();

    // Reset state
    distances.clear();
    distances.resize(n, usize::MAX);
    queue.clear();

    // Initialize BFS
    distances[source_idx] = 0;
    queue.push_back(source_idx);
    let mut reachable = 1;

    // BFS traversal
    while let Some(u) = queue.pop_front() {
        let current_dist = distances[u];
        for &v in csr.neighbors(u) {
            if distances[v] == usize::MAX {
                distances[v] = current_dist + 1;
                queue.push_back(v);
                reachable += 1;
            }
        }
    }

    reachable
}

#[derive(Copy, Clone, Debug)]
struct DijkstraState {
    pub cost: f64,
    pub node_idx: usize,
}

impl Eq for DijkstraState {}

impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool {
        self.node_idx == other.node_idx && (self.cost - other.cost).abs() <= f64::EPSILON
    }
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.node_idx.cmp(&other.node_idx))
    }
}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// CSR-optimized Dijkstra SSSP with pre-allocated buffers (zero inner-loop allocations).
///
/// # Arguments
/// * `csr` - Pre-built CSR adjacency structure
/// * `nodes` - Original NodeId array (for weight map lookups)
/// * `source_idx` - Dense index of source node
/// * `weight_map` - Optional edge weights as (src_id, tgt_id) -> weight; defaults to 1.0 if None
/// * `distances` - Pre-allocated distance buffer (will be cleared and filled)
/// * `heap` - Pre-allocated priority queue (will be cleared and reused)
///
/// # Returns
/// Number of reachable nodes (including source)
#[allow(private_interfaces)]
pub fn dijkstra_csr(
    csr: &Csr,
    nodes: &[NodeId],
    source_idx: usize,
    weight_map: Option<&HashMap<(NodeId, NodeId), f64>>,
    distances: &mut Vec<f64>,
    heap: &mut BinaryHeap<DijkstraState>,
) -> usize {
    let n = csr.node_count();

    // Reset state
    distances.clear();
    distances.resize(n, f64::INFINITY);
    heap.clear();

    // Initialize Dijkstra
    distances[source_idx] = 0.0;
    heap.push(DijkstraState {
        cost: 0.0,
        node_idx: source_idx,
    });
    let mut reachable = 0;

    // Dijkstra traversal
    while let Some(DijkstraState { cost, node_idx: u }) = heap.pop() {
        if cost > distances[u] + f64::EPSILON {
            continue;
        }

        reachable += 1;

        let u_id = nodes[u];
        for &v in csr.neighbors(u) {
            let v_id = nodes[v];
            let weight = if let Some(map) = weight_map {
                map.get(&(u_id, v_id)).copied().unwrap_or(1.0)
            } else {
                1.0
            };

            let next_cost = cost + weight;
            if next_cost + f64::EPSILON < distances[v] {
                distances[v] = next_cost;
                heap.push(DijkstraState {
                    cost: next_cost,
                    node_idx: v,
                });
            }
        }
    }

    reachable
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use crate::state::topology::{build_csr_from_edges_with_scratch, CsrOptions};
    use crate::subgraphs::Subgraph;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    fn create_test_graph() -> (Graph, Vec<NodeId>) {
        let mut g = Graph::new();
        // Create a simple graph: 0 -> 1 -> 2
        //                          |    |
        //                          v    v
        //                          3    4
        let mut nodes = Vec::new();
        for _ in 0..5 {
            nodes.push(g.add_node());
        }
        g.add_edge(nodes[0], nodes[1]).unwrap();
        g.add_edge(nodes[1], nodes[2]).unwrap();
        g.add_edge(nodes[0], nodes[3]).unwrap();
        g.add_edge(nodes[1], nodes[4]).unwrap();
        (g, nodes)
    }

    #[test]
    fn test_bfs_csr_matches_legacy() {
        let (graph, test_nodes) = create_test_graph();
        let node_set: HashSet<NodeId> = test_nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), node_set, "test".into()).unwrap();

        let source = test_nodes[0];

        // Legacy BFS
        let legacy_result = bfs_layers(&subgraph, source);

        // CSR BFS
        let nodes = subgraph.ordered_nodes();
        let indexer = NodeIndexer::new(&nodes);
        let edges = subgraph.ordered_edges();

        let mut csr = Csr::default();
        let graph_ref = subgraph.graph();
        let graph_borrow = graph_ref.borrow();

        build_csr_from_edges_with_scratch(
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

        let source_idx = indexer.get(source).unwrap();
        let mut distances = Vec::new();
        let mut queue = VecDeque::new();
        let reachable = bfs_layers_csr(&csr, &nodes, source_idx, &mut distances, &mut queue);

        // Compare results
        assert_eq!(reachable, legacy_result.len());
        for (i, &node) in nodes.iter().enumerate() {
            let csr_dist = distances[i];
            let legacy_dist = legacy_result.get(&node);

            if csr_dist == usize::MAX {
                assert!(legacy_dist.is_none(), "Node {} should be unreachable", node);
            } else {
                assert_eq!(
                    Some(&csr_dist),
                    legacy_dist,
                    "Distance mismatch for node {}",
                    node
                );
            }
        }
    }

    #[test]
    fn test_dijkstra_csr_matches_legacy() {
        let (graph, test_nodes) = create_test_graph();
        let node_set: HashSet<NodeId> = test_nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), node_set, "test".into()).unwrap();

        let source = test_nodes[0];

        // Legacy Dijkstra (unweighted)
        let legacy_result = dijkstra(&subgraph, source, |_, _| 1.0);

        // CSR Dijkstra
        let nodes = subgraph.ordered_nodes();
        let indexer = NodeIndexer::new(&nodes);
        let edges = subgraph.ordered_edges();

        let mut csr = Csr::default();
        let graph_ref = subgraph.graph();
        let graph_borrow = graph_ref.borrow();

        build_csr_from_edges_with_scratch(
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

        let source_idx = indexer.get(source).unwrap();
        let mut distances = Vec::new();
        let mut heap = BinaryHeap::new();
        let reachable = dijkstra_csr(&csr, &nodes, source_idx, None, &mut distances, &mut heap);

        // Compare results
        assert_eq!(reachable, legacy_result.len());
        for (i, &node) in nodes.iter().enumerate() {
            let csr_dist = distances[i];
            let legacy_dist = legacy_result.get(&node).copied();

            if csr_dist.is_infinite() {
                assert!(
                    legacy_dist.is_none() || legacy_dist == Some(f64::INFINITY),
                    "Node {} should be unreachable",
                    node
                );
            } else {
                assert!(
                    (csr_dist - legacy_dist.unwrap()).abs() < 1e-6,
                    "Distance mismatch for node {}: CSR={}, legacy={:?}",
                    node,
                    csr_dist,
                    legacy_dist
                );
            }
        }
    }

    #[test]
    fn test_bfs_csr_reachability() {
        let mut g = Graph::new();
        // Disconnected graph: 0-1-2  3-4
        let mut test_nodes = Vec::new();
        for _ in 0..5 {
            test_nodes.push(g.add_node());
        }
        g.add_edge(test_nodes[0], test_nodes[1]).unwrap();
        g.add_edge(test_nodes[1], test_nodes[2]).unwrap();
        g.add_edge(test_nodes[3], test_nodes[4]).unwrap();

        let node_set: HashSet<NodeId> = test_nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(g)), node_set, "test".into()).unwrap();

        let nodes = subgraph.ordered_nodes();
        let indexer = NodeIndexer::new(&nodes);
        let edges = subgraph.ordered_edges();

        let mut csr = Csr::default();
        let graph_ref = subgraph.graph();
        let graph_borrow = graph_ref.borrow();

        build_csr_from_edges_with_scratch(
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

        // BFS from node 0 should reach 0, 1, 2 (3 nodes)
        let source_idx = indexer.get(test_nodes[0]).unwrap();
        let mut distances = Vec::new();
        let mut queue = VecDeque::new();
        let reachable = bfs_layers_csr(&csr, &nodes, source_idx, &mut distances, &mut queue);

        assert_eq!(reachable, 3);

        // Nodes 0, 1, 2 should be reachable
        assert_ne!(distances[indexer.get(test_nodes[0]).unwrap()], usize::MAX);
        assert_ne!(distances[indexer.get(test_nodes[1]).unwrap()], usize::MAX);
        assert_ne!(distances[indexer.get(test_nodes[2]).unwrap()], usize::MAX);

        // Nodes 3, 4 should be unreachable
        assert_eq!(distances[indexer.get(test_nodes[3]).unwrap()], usize::MAX);
        assert_eq!(distances[indexer.get(test_nodes[4]).unwrap()], usize::MAX);
    }

    #[test]
    fn test_bfs_auto_uses_csr() {
        let (graph, test_nodes) = create_test_graph();
        let node_set: HashSet<NodeId> = test_nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), node_set, "test".into()).unwrap();

        let source = test_nodes[0];

        // Build CSR in cache
        let nodes = subgraph.ordered_nodes();
        let indexer = NodeIndexer::new(&nodes);
        let edges = subgraph.ordered_edges();
        let mut csr = Csr::default();
        let graph_ref = subgraph.graph();
        let graph_borrow = graph_ref.borrow();

        build_csr_from_edges_with_scratch(
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
        subgraph.csr_cache_store(false, std::sync::Arc::new(csr));

        // Now call bfs_layers - it should use CSR path
        let result = bfs_layers(&subgraph, source);

        // Verify we got results
        assert!(!result.is_empty());
        assert_eq!(result.get(&source), Some(&0));
    }
}
