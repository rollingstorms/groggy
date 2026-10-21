//! Edge-instance CSR and Brandes accumulation for undirected Girvan–Newman.
use std::collections::{BinaryHeap, VecDeque};

use anyhow::{anyhow, Result};

use crate::algorithms::Context;
use crate::types::EdgeId;

#[derive(Clone, Copy)]
pub(super) struct Edge {
    pub id: EdgeId,
    pub u: usize,
    pub v: usize,
    pub length: f64,
}

pub(super) struct Topology {
    pub edges: Vec<Edge>,
    offsets: Vec<usize>,
    arcs: Vec<(usize, usize)>, // (neighbor index, edge index)
}

#[derive(Clone, Copy, Debug)]
struct State {
    cost: f64,
    node: usize,
}
impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.cost.total_cmp(&other.cost).is_eq()
    }
}
impl Eq for State {}
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

impl Topology {
    pub fn new(n: usize, edges: Vec<Edge>) -> Self {
        let mut offsets = vec![0; n + 1];
        for edge in &edges {
            if edge.u != edge.v {
                offsets[edge.u + 1] += 1;
                offsets[edge.v + 1] += 1;
            }
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let mut arcs = vec![(0, 0); offsets[n]];
        let mut cursor = offsets[..n].to_vec();
        for (i, edge) in edges.iter().enumerate() {
            if edge.u != edge.v {
                arcs[cursor[edge.u]] = (edge.v, i);
                cursor[edge.u] += 1;
                arcs[cursor[edge.v]] = (edge.u, i);
                cursor[edge.v] += 1;
            }
        }
        Self {
            edges,
            offsets,
            arcs,
        }
    }

    fn neighbors(&self, node: usize) -> &[(usize, usize)] {
        &self.arcs[self.offsets[node]..self.offsets[node + 1]]
    }

    pub fn components(&self, active: &[bool]) -> Vec<usize> {
        let n = self.offsets.len() - 1;
        let mut labels = vec![usize::MAX; n];
        let mut stack = Vec::new();
        let mut label = 0;
        for node in 0..n {
            if labels[node] != usize::MAX {
                continue;
            }
            labels[node] = label;
            stack.push(node);
            while let Some(u) = stack.pop() {
                for &(v, edge) in self.neighbors(u) {
                    if active[edge] && labels[v] == usize::MAX {
                        labels[v] = label;
                        stack.push(v);
                    }
                }
            }
            label += 1;
        }
        labels
    }

    pub fn betweenness(&self, active: &[bool], weighted: bool, ctx: &Context) -> Result<Vec<f64>> {
        let n = self.offsets.len() - 1;
        let mut scores = vec![0.; self.edges.len()];
        let mut distances = vec![f64::INFINITY; n];
        let mut sigma = vec![0_f64; n];
        let mut predecessors: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        let mut delta = vec![0.; n];
        let mut order = Vec::with_capacity(n);
        let mut heap = BinaryHeap::new();
        let mut queue = VecDeque::new();
        for source in 0..n {
            distances.fill(f64::INFINITY);
            sigma.fill(0.);
            delta.fill(0.);
            for pred in &mut predecessors {
                pred.clear();
            }
            order.clear();
            heap.clear();
            queue.clear();
            distances[source] = 0.;
            sigma[source] = 1.;
            if weighted {
                heap.push(State {
                    cost: 0.,
                    node: source,
                });
            } else {
                queue.push_back(source);
            }
            loop {
                if ctx.is_cancelled() {
                    return Err(anyhow!("Girvan-Newman cancelled"));
                }
                let next = if weighted {
                    heap.pop()
                } else {
                    queue.pop_front().map(|node| State {
                        cost: distances[node],
                        node,
                    })
                };
                let Some(State { node: u, cost }) = next else {
                    break;
                };
                if cost > distances[u] {
                    continue;
                }
                order.push(u);
                for &(v, edge_idx) in self.neighbors(u) {
                    if !active[edge_idx] {
                        continue;
                    }
                    let next_cost = cost + self.edges[edge_idx].length;
                    if !next_cost.is_finite() || next_cost <= cost {
                        return Err(anyhow!(
                            "Girvan-Newman distance cannot advance in f64 on edge {}",
                            self.edges[edge_idx].id
                        ));
                    }
                    if next_cost < distances[v] {
                        distances[v] = next_cost;
                        sigma[v] = sigma[u];
                        predecessors[v].clear();
                        predecessors[v].push((u, edge_idx));
                        if weighted {
                            heap.push(State {
                                cost: next_cost,
                                node: v,
                            });
                        } else {
                            queue.push_back(v);
                        }
                    } else if next_cost == distances[v] {
                        sigma[v] += sigma[u];
                        predecessors[v].push((u, edge_idx));
                    }
                    if !sigma[v].is_finite() {
                        return Err(anyhow!(
                            "Girvan-Newman shortest-path count exceeds f64 range"
                        ));
                    }
                }
            }
            for &w in order.iter().rev() {
                for &(v, edge) in &predecessors[w] {
                    let contribution = (sigma[v] / sigma[w]) * (1. + delta[w]);
                    delta[v] += contribution;
                    scores[edge] += contribution;
                    if !delta[v].is_finite() || !scores[edge].is_finite() {
                        return Err(anyhow!("Girvan-Newman dependency exceeds f64 range"));
                    }
                }
            }
        }
        for score in &mut scores {
            *score *= 0.5;
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_instances_survive_individual_removal_and_expensive_edges_become_paths() {
        for lengths in [[1., 3.], [1., 1.]] {
            let topology = Topology::new(
                2,
                vec![
                    Edge {
                        id: 5,
                        u: 1,
                        v: 0,
                        length: lengths[0],
                    },
                    Edge {
                        id: 9,
                        u: 0,
                        v: 1,
                        length: lengths[1],
                    },
                ],
            );
            let expected = if lengths[0] == lengths[1] {
                vec![0.5, 0.5]
            } else {
                vec![1., 0.]
            };
            assert_eq!(
                topology
                    .betweenness(&[true, true], true, &Context::new())
                    .unwrap(),
                expected
            );
            assert_eq!(topology.components(&[false, true]), vec![0, 0]);
            assert_eq!(
                topology
                    .betweenness(&[false, true], true, &Context::new())
                    .unwrap(),
                vec![0., 1.]
            );
            assert_eq!(topology.components(&[false, false]), vec![0, 1]);
        }
    }

    #[test]
    fn scores_match_exhaustive_edge_instance_paths_after_removals() {
        // Five vertices include an isolate; edges include a loop and parallel lengths.
        let triples = [
            (1, 0, 1.),
            (1, 0, 1.),
            (0, 1, 7.),
            (0, 2, 1.),
            (1, 3, 1.),
            (2, 3, 1.),
            (2, 2, 1.),
        ];
        for weighted in [false, true] {
            let edges: Vec<_> = triples
                .iter()
                .enumerate()
                .map(|(id, &(u, v, w))| Edge {
                    id,
                    u,
                    v,
                    length: if weighted { w } else { 1. },
                })
                .collect();
            let topology = Topology::new(5, edges);
            for removed in [vec![], vec![0], vec![0, 1], vec![0, 1, 4]] {
                let mut active = vec![true; triples.len()];
                for edge in removed {
                    active[edge] = false;
                }
                let actual = topology
                    .betweenness(&active, weighted, &Context::new())
                    .unwrap();
                let mut expected = vec![0.; triples.len()];
                for source in 0..5 {
                    for target in source + 1..5 {
                        let mut candidates = vec![(source, vec![source], Vec::<usize>::new(), 0.)];
                        let mut paths = Vec::new();
                        let mut best = f64::INFINITY;
                        while let Some((here, nodes, path, cost)) = candidates.pop() {
                            if here == target {
                                if cost < best {
                                    best = cost;
                                    paths.clear();
                                }
                                if cost == best {
                                    paths.push(path);
                                }
                                continue;
                            }
                            for (i, e) in topology.edges.iter().enumerate() {
                                if !active[i] {
                                    continue;
                                }
                                let next = if e.u == here {
                                    e.v
                                } else if e.v == here {
                                    e.u
                                } else {
                                    continue;
                                };
                                if nodes.contains(&next) {
                                    continue;
                                }
                                let mut next_nodes = nodes.clone();
                                next_nodes.push(next);
                                let mut next_path = path.clone();
                                next_path.push(i);
                                candidates.push((next, next_nodes, next_path, cost + e.length));
                            }
                        }
                        for path in &paths {
                            for &edge in path {
                                expected[edge] += 1. / paths.len() as f64;
                            }
                        }
                    }
                }
                for (got, want) in actual.iter().zip(expected) {
                    assert!((got - want).abs() < 1e-12, "{actual:?}");
                }
            }
        }
    }

    #[test]
    fn numerical_failures_reject_instead_of_publishing_nonfinite_dependencies() {
        let topology = Topology::new(
            3,
            vec![
                Edge {
                    id: 0,
                    u: 0,
                    v: 1,
                    length: 1e20,
                },
                Edge {
                    id: 1,
                    u: 1,
                    v: 2,
                    length: 1.,
                },
            ],
        );
        assert!(topology
            .betweenness(&[true, true], true, &Context::new())
            .unwrap_err()
            .to_string()
            .contains("cannot advance"));
        let edges: Vec<_> = (0..1029)
            .flat_map(|u| {
                (0..2).map(move |i| Edge {
                    id: u * 2 + i,
                    u,
                    v: u + 1,
                    length: 1.,
                })
            })
            .collect();
        let topology = Topology::new(1030, edges);
        for weighted in [false, true] {
            assert!(topology
                .betweenness(&vec![true; 2058], weighted, &Context::new())
                .unwrap_err()
                .to_string()
                .contains("count exceeds"));
        }
    }
}
