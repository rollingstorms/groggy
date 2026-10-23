//! Immutable graph input for algorithm kernels, independent of working writers.
use crate::{
    state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions},
    subgraphs::GraphView,
    GraphError, GraphResult, NodeId,
};
use rustc_hash::FxHashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// Prepared immutable algorithm input. Clones share state, row identity and CSR.
/// Construction does not publish a mutation or create a checkpoint.
#[derive(Debug, Clone)]
pub struct AlgorithmRead {
    view: GraphView,
    nodes: Arc<[NodeId]>,
    index: Arc<FxHashMap<NodeId, usize>>,
    csr: Arc<Csr>,
}
impl AlgorithmRead {
    /// Prepare exactly the captured view's selected nodes and edge instances.
    pub fn new(view: GraphView) -> GraphResult<Self> {
        let nodes: Arc<[NodeId]> = view.node_ids().into();
        let index: FxHashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(row, &id)| (id, row))
            .collect();
        let edges = view.edge_ids();
        let mut failure = None;
        let mut csr = Csr::default();
        build_csr_from_edges_with_scratch(
            &mut csr,
            nodes.len(),
            edges.iter().copied(),
            |id| index.get(&id).copied(),
            |edge| match view.edge_endpoints(edge) {
                Ok((s, t)) if index.contains_key(&s) && index.contains_key(&t) => Some((s, t)),
                Ok(_) => {
                    failure = Some(GraphError::InvalidInput(
                        "Algorithm edge endpoints must belong to the captured selection".into(),
                    ));
                    None
                }
                Err(error) => {
                    failure = Some(error);
                    None
                }
            },
            CsrOptions {
                add_reverse_edges: !view.is_directed(),
                sort_neighbors: true,
            },
        );
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(Self {
            view,
            nodes,
            index: Arc::new(index),
            csr: Arc::new(csr),
        })
    }
    /// Captured selection and properties used by this prepared input.
    pub fn view(&self) -> &GraphView {
        &self.view
    }
    /// Ascending selected node IDs corresponding to CSR row indices.
    pub fn node_ids(&self) -> &[NodeId] {
        &self.nodes
    }
    /// Shared directed/undirected adjacency, preserving parallel occurrences and loops.
    pub fn csr(&self) -> &Csr {
        &self.csr
    }
    /// Captured classic/harmonic closeness scores, with no property writes.
    pub fn closeness(
        &self,
        algorithm: &super::centrality::ClosenessCentrality,
        cancel: Option<&AtomicBool>,
    ) -> GraphResult<Vec<(NodeId, f64)>> {
        algorithm.compute_read(self, cancel)
    }
    /// Owned PageRank scores and explicit finite-budget convergence status.
    /// Uses captured personalization and selected edge occurrences; writes nothing.
    pub fn pagerank(
        &self,
        algorithm: &super::centrality::PageRank,
        cancel: Option<&AtomicBool>,
    ) -> GraphResult<super::centrality::PageRankResult> {
        use super::centrality::pagerank::{personalization, personalization_value};
        let values = personalization(&self.nodes, |node| {
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                return Err(anyhow::anyhow!("pagerank cancelled"));
            }
            match algorithm.personalization_attr() {
                Some(attr) => personalization_value(
                    node,
                    self.view
                        .read_state()
                        .snapshot()
                        .node_attributes
                        .get(&node)
                        .and_then(|values| values.get(attr)),
                ),
                None => Ok(1.0),
            }
        })
        .map_err(|error| GraphError::InvalidInput(error.to_string()))?;
        algorithm
            .compute(
                &self.csr,
                &self.nodes,
                &values,
                &mut super::Context::default(),
                cancel,
            )
            .map_err(|error| GraphError::InvalidInput(error.to_string()))
    }
    /// Immutable component selections, ordered by smallest node ID. The default
    /// undirected mode and weak mode ignore orientation; strong respects it.
    /// Every result retains only its parent's selected edge instances.
    pub fn connected_components(
        &self,
        mode: super::community::ComponentMode,
        cancel: Option<&AtomicBool>,
    ) -> GraphResult<Vec<GraphView>> {
        use super::community::{components, ComponentMode};
        let cancelled = || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed));
        let check = || {
            if cancelled() {
                Err(GraphError::InvalidInput("components cancelled".into()))
            } else {
                Ok(())
            }
        };
        check()?;
        if self.nodes.len() >= u32::MAX as usize {
            return Err(GraphError::InvalidInput(
                "Components require fewer than u32::MAX nodes".into(),
            ));
        }
        let mut symmetric = Csr::default();
        let csr = if self.view.is_directed() && mode != ComponentMode::Strong {
            let edges = self.view.edge_ids();
            build_csr_from_edges_with_scratch(
                &mut symmetric,
                self.nodes.len(),
                edges.into_iter(),
                |id| self.index.get(&id).copied(),
                |edge| self.view.edge_endpoints(edge).ok(),
                CsrOptions {
                    add_reverse_edges: true,
                    sort_neighbors: true,
                },
            );
            &symmetric
        } else {
            &self.csr
        };
        let mut labels = vec![0; self.nodes.len()];
        // Local telemetry has no writer, cache, or publication side effects.
        let mut context = super::Context::default();
        let count = if mode == ComponentMode::Strong {
            components::tarjan_components(csr, &mut labels, &mut context, cancel)
                .map(|groups| groups.len())
        } else {
            components::bfs_components(csr, &mut labels, &mut context, cancel)
                .map(|count| count as usize)
        }
        .map_err(|error| GraphError::InvalidInput(error.to_string()))?;
        check()?;
        let mut groups = vec![Vec::new(); count];
        for (row, &node) in self.nodes.iter().enumerate() {
            groups[labels[row] as usize].push(node);
        }
        // Group edges in one pass, avoiding one full edge scan per component.
        let mut edges = vec![Vec::new(); count];
        for edge in self.view.edge_ids() {
            check()?;
            let (source, target) = self.view.edge_endpoints(edge)?;
            let group = labels[self.index[&source]] as usize;
            if group == labels[self.index[&target]] as usize {
                edges[group].push(edge);
            }
        }
        let mut members: Vec<_> = groups.into_iter().zip(edges).collect();
        members.sort_by_key(|(nodes, _)| nodes[0]);
        let result = members
            .into_iter()
            .map(|(nodes, edges)| {
                GraphView::from_membership(
                    self.view.read_state().clone(),
                    nodes.into_iter().collect(),
                    edges.into_iter().collect(),
                )
            })
            .collect::<GraphResult<Vec<_>>>()?;
        check()?;
        Ok(result)
    }
    fn source(&self, start: NodeId) -> GraphResult<usize> {
        self.index.get(&start).copied().ok_or_else(|| {
            GraphError::InvalidInput(format!(
                "Traversal start {start} does not belong to the captured selection"
            ))
        })
    }
    /// Reachable hop distances, ordered by original node ID; no attributes are written.
    /// A supplied cancellation flag rejects the operation without partial output.
    pub fn bfs_distances(
        &self,
        start: NodeId,
        cancel: Option<&AtomicBool>,
    ) -> GraphResult<Vec<(NodeId, i64)>> {
        let source = self.source(start)?;
        super::pathfinding::bfs_dfs::bfs_distances(
            || cancel.is_some_and(|token| token.load(Ordering::Relaxed)),
            &self.csr,
            &self.nodes,
            source,
        )
        .map_err(|error| GraphError::InvalidInput(error.to_string()))
    }
    /// Reachable ascending-neighbor DFS preorder, returned in discovery order.
    pub fn dfs_order(
        &self,
        start: NodeId,
        cancel: Option<&AtomicBool>,
    ) -> GraphResult<Vec<(NodeId, i64)>> {
        let source = self.source(start)?;
        super::pathfinding::bfs_dfs::dfs_order(
            || cancel.is_some_and(|token| token.load(Ordering::Relaxed)),
            &self.csr,
            &self.nodes,
            source,
        )
        .map_err(|error| GraphError::InvalidInput(error.to_string()))
    }
    pub(crate) fn checked_weights(
        &self,
        attr: Option<&str>,
    ) -> GraphResult<std::collections::HashMap<(NodeId, NodeId), f64>> {
        crate::query::weights::checked_pairs(
            &self.view.edge_ids(),
            self.view.is_directed(),
            |edge| self.view.edge_endpoints(edge),
            |edge| {
                Ok(attr.and_then(|name| {
                    self.view
                        .read_state()
                        .snapshot()
                        .edge_attributes
                        .get(&edge)
                        .and_then(|attrs| attrs.get(name))
                }))
            },
        )
    }
    /// Nonnegative weighted distances over captured state, ordered by original ID.
    /// Missing weights are one; explicit null/nonnumeric/negative/nonfinite weights fail.
    pub fn dijkstra_distances(
        &self,
        start: NodeId,
        weight_attr: Option<&str>,
        cancel: Option<&AtomicBool>,
    ) -> GraphResult<Vec<(NodeId, f64)>> {
        let source = self.source(start)?;
        if cancel.is_some_and(|token| token.load(Ordering::Relaxed)) {
            return Err(GraphError::InvalidInput("dijkstra cancelled".into()));
        }
        let weights = self.checked_weights(weight_attr)?;
        super::pathfinding::dijkstra::distances(
            || cancel.is_some_and(|token| token.load(Ordering::Relaxed)),
            &self.csr,
            &self.nodes,
            source,
            &weights,
        )
        .map_err(|error| GraphError::InvalidInput(error.to_string()))
    }
    /// Owned A* path over captured values; goal heuristic must be zero and all
    /// selected heuristics finite/nonnegative and consistent with selected weights.
    pub fn astar_path(
        &self,
        start: NodeId,
        goal: NodeId,
        weight_attr: Option<&str>,
        heuristic_attr: Option<&str>,
        cancel: Option<&AtomicBool>,
    ) -> GraphResult<crate::query::Path> {
        let source = self.source(start)?;
        let target = self.index.get(&goal).copied().ok_or_else(|| {
            GraphError::InvalidInput(format!(
                "A* goal {goal} does not belong to the captured selection"
            ))
        })?;
        let cancelled = || cancel.is_some_and(|token| token.load(Ordering::Relaxed));
        if cancelled() {
            return Err(GraphError::InvalidInput("astar cancelled".into()));
        }
        let weights = self.checked_weights(weight_attr)?;
        let heuristics = if let Some(attr) = heuristic_attr {
            crate::query::weights::checked_heuristics(&self.nodes, |node| {
                Ok(self
                    .view
                    .read_state()
                    .snapshot()
                    .node_attributes
                    .get(&node)
                    .and_then(|attrs| attrs.get(attr)))
            })?
        } else {
            std::collections::HashMap::new()
        };
        let result = super::pathfinding::astar::search(
            cancelled,
            super::pathfinding::astar::SearchInput {
                nodes: &self.nodes,
                csr: &self.csr,
                weights: &weights,
                heuristics: &heuristics,
            },
            source,
            target,
        )
        .map_err(|error| GraphError::InvalidInput(error.to_string()))?;
        let nodes: Vec<_> = result.rows.into_iter().map(|(id, _)| id).collect();
        // Resolve path instances from selected IDs only, independent of unrelated
        // captured adjacency. Ascending edge order establishes the equal-cost tie.
        let positions: std::collections::HashMap<_, _> = nodes
            .windows(2)
            .enumerate()
            .map(|(index, pair)| ((pair[0], pair[1]), index))
            .collect();
        let mut selected_edges = vec![None; nodes.len().saturating_sub(1)];
        for edge in self.view.edge_ids() {
            if cancelled() {
                return Err(GraphError::InvalidInput("astar cancelled".into()));
            }
            let (s, t) = self.view.edge_endpoints(edge)?;
            let position = positions.get(&(s, t)).or_else(|| {
                if self.view.is_directed() {
                    None
                } else {
                    positions.get(&(t, s))
                }
            });
            if let Some(&index) = position {
                if selected_edges[index].is_some() {
                    continue;
                }
                let pair = (nodes[index], nodes[index + 1]);
                let weight = weights.get(&pair).ok_or_else(|| {
                    GraphError::InvalidInput("Missing checked path weight".into())
                })?;
                let value = weight_attr.and_then(|name| {
                    self.view
                        .read_state()
                        .snapshot()
                        .edge_attributes
                        .get(&edge)
                        .and_then(|attrs| attrs.get(name))
                });
                if crate::query::weights::dijkstra_weight(value, edge)? == *weight {
                    selected_edges[index] = Some(edge);
                }
            }
        }
        let edges = selected_edges
            .into_iter()
            .map(|edge| {
                edge.ok_or_else(|| {
                    GraphError::InvalidInput("Missing selected path edge instance".into())
                })
            })
            .collect::<GraphResult<Vec<_>>>()?;
        if cancelled() {
            return Err(GraphError::InvalidInput("astar cancelled".into()));
        }
        let length = edges.len();
        Ok(crate::query::Path {
            nodes,
            edges,
            total_weight: result.cost,
            metadata: crate::query::PathMetadata {
                algorithm: "astar".into(),
                is_simple: true,
                length,
            },
        })
    }
}
