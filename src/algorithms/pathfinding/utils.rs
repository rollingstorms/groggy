use std::collections::{BinaryHeap, HashMap, VecDeque};

use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

pub fn bfs_layers(subgraph: &Subgraph, source: NodeId) -> HashMap<NodeId, usize> {
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
