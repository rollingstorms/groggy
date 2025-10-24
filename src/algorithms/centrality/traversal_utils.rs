use std::collections::{BinaryHeap, HashMap, VecDeque};

use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::NodeId;

/// Breadth-first traversal that returns predecessor stacks and distances.
#[allow(dead_code)]
pub fn bfs_shortest_paths(
    subgraph: &Subgraph,
    source: NodeId,
) -> (HashMap<NodeId, usize>, HashMap<NodeId, Vec<NodeId>>) {
    let mut distances: HashMap<NodeId, usize> = HashMap::new();
    let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut queue = VecDeque::new();

    distances.insert(source, 0);
    queue.push_back(source);

    while let Some(node) = queue.pop_front() {
        let dist = distances[&node];
        if let Ok(neighbors) = subgraph.neighbors(node) {
            for neighbor in neighbors {
                if let std::collections::hash_map::Entry::Vacant(entry) = distances.entry(neighbor)
                {
                    entry.insert(dist + 1);
                    queue.push_back(neighbor);
                }
                if distances[&neighbor] == dist + 1 {
                    predecessors.entry(neighbor).or_default().push(node);
                }
            }
        }
    }

    (distances, predecessors)
}

/// Dijkstra traversal skeleton for weighted graphs.
#[allow(dead_code)]
pub fn dijkstra_shortest_paths<F>(
    subgraph: &Subgraph,
    source: NodeId,
    mut weight_fn: F,
) -> (HashMap<NodeId, f64>, HashMap<NodeId, Vec<NodeId>>)
where
    F: FnMut(NodeId, NodeId) -> f64,
{
    #[derive(Copy, Clone, Debug)]
    struct State {
        cost: f64,
        node: NodeId,
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

    let mut distances: HashMap<NodeId, f64> = HashMap::new();
    let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut heap: BinaryHeap<State> = BinaryHeap::new();

    distances.insert(source, 0.0);
    heap.push(State {
        cost: 0.0,
        node: source,
    });

    while let Some(State { cost, node }) = heap.pop() {
        if cost > distances[&node] + f64::EPSILON {
            continue;
        }
        if let Ok(neighbors) = subgraph.neighbors(node) {
            for neighbor in neighbors {
                let weight = weight_fn(node, neighbor);
                let next = cost + weight;
                let current = distances.get(&neighbor).copied().unwrap_or(f64::INFINITY);
                if next + f64::EPSILON < current {
                    distances.insert(neighbor, next);
                    predecessors.insert(neighbor, vec![node]);
                    heap.push(State {
                        cost: next,
                        node: neighbor,
                    });
                } else if (current - next).abs() <= f64::EPSILON {
                    predecessors.entry(neighbor).or_default().push(node);
                }
            }
        }
    }

    (distances, predecessors)
}
