//! Sealed native step invocation over read authority. There is deliberately no
//! `subgraph()` or mutable owner accessor. Legacy callbacks retain StepScope.
use super::core::{validate_attachment_targets, SelectedCsr, StepVariables};
use crate::traits::SubgraphOperations;
use crate::{
    algorithms::Context, subgraphs::Subgraph, AttrName, AttrValue, BatchNodeRef, GraphResult,
    GraphView,
};
use anyhow::{anyhow, bail, Result};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Native kernels may be shared by memory and durable adapters. Sealing keeps
/// the portable guarantee independent of custom callbacks' effect declarations.
#[doc(hidden)]
pub trait ReadStep: sealed::Sealed + Send + Sync {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()>;
}

/// Membership and optional retained value pin. New selections bind the current source;
/// foreach pass-through retains the original pin rather than recapturing its values.
pub(crate) struct SelectionPart {
    pub nodes: Arc<HashSet<usize>>,
    pub edges: Arc<HashSet<usize>>,
    pub name: String,
    pub source: Option<GraphView>,
    // Only the memory adapter populates this; detached arrays never carry owners.
    // Preserve identity/caches for a pass-through rather than inventing a revision.
    pub legacy: Option<Subgraph>,
}

enum Source<'a> {
    Memory(&'a Subgraph),
    Detached {
        allocation: (usize, usize),
        base: &'a GraphView,
        current: &'a GraphView,
        frame: Option<&'a GraphView>,
    },
}

/// Read-only invocation input, plus owned intermediates and checked staging.
/// Writable owners and publication remain exclusively in backend coordinators.
#[doc(hidden)]
pub struct ReadStepScope<'a> {
    source: Source<'a>,
    variables: &'a mut StepVariables,
    fixed: bool,
}

impl<'a> ReadStepScope<'a> {
    pub(crate) fn memory(
        subgraph: &'a Subgraph,
        variables: &'a mut StepVariables,
        fixed: bool,
    ) -> Self {
        Self {
            source: Source::Memory(subgraph),
            variables,
            fixed,
        }
    }

    pub(crate) fn detached(
        allocation: (usize, usize),
        base: &'a GraphView,
        current: &'a GraphView,
        frame: Option<&'a GraphView>,
        variables: &'a mut StepVariables,
    ) -> Self {
        Self {
            source: Source::Detached {
                allocation,
                base,
                current,
                frame,
            },
            variables,
            fixed: frame.is_some(),
        }
    }

    /// Run one nested native frame, retaining its pending-read cache across steps.
    pub(crate) fn with_executor<T>(
        &mut self,
        run: impl FnOnce(super::core::StepExecutor<'_>, &mut StepVariables) -> Result<T>,
    ) -> Result<T> {
        let executor = match &self.source {
            Source::Memory(s) => super::core::StepExecutor::for_frame(s, self.fixed),
            Source::Detached {
                allocation,
                base,
                frame,
                ..
            } => match frame {
                Some(frame) => super::core::StepExecutor::detached_fixed(*allocation, base, frame),
                None => super::core::StepExecutor::detached(*allocation, base),
            },
        };
        run(executor, self.variables)
    }

    /// Decode only collections from this owner/history or an issued pending capture.
    pub(crate) fn collection(
        &self,
        ctx: &Context,
        name: &str,
    ) -> Result<Option<Vec<SelectionPart>>> {
        use super::core::StepValue;
        match self.variables.value(name) {
            Some(StepValue::SubgraphArray(array)) => {
                let Source::Memory(parent) = &self.source else {
                    bail!("Owner-connected subgraph arrays cannot run on detached workers");
                };
                let mut out = Vec::with_capacity(array.len());
                for selected in array {
                    if ctx.is_cancelled() {
                        bail!("collection cancelled");
                    }
                    if !Arc::ptr_eq(&parent.graph(), &selected.graph()) {
                        bail!("Subgraph selector belongs to a different graph owner");
                    }
                    let (nodes, edges) = selected.shared_membership();
                    out.push(SelectionPart {
                        nodes,
                        edges,
                        name: selected.subgraph_type().into(),
                        source: selected.frozen_view().cloned(),
                        legacy: Some(selected.clone()),
                    });
                }
                Ok(Some(out))
            }
            Some(StepValue::GraphViewArray(array)) => {
                let base = match &self.source {
                    Source::Memory(s) => {
                        s.graph().borrow().pin_current_state().map(GraphView::all)?
                    }
                    Source::Detached { base, .. } => (*base).clone(),
                };
                let mut out = Vec::with_capacity(array.len());
                for view in array {
                    if ctx.is_cancelled() {
                        bail!("collection cancelled");
                    }
                    let same_history = base
                        .read_state()
                        .origin()
                        .zip(view.read_state().origin())
                        .is_some_and(|(a, b)| Arc::ptr_eq(&a.history, &b.history));
                    let issued = ctx
                        .property_batch
                        .as_ref()
                        .is_some_and(|batch| batch.owns_capture(view));
                    if !same_history && !issued {
                        bail!(
                            "Graph view selector belongs to a different graph owner or execution"
                        );
                    }
                    let (nodes, edges) = view.shared_membership();
                    out.push(SelectionPart {
                        nodes,
                        edges,
                        name: "portable selection".into(),
                        source: Some(view.clone()),
                        legacy: None,
                    });
                }
                Ok(Some(out))
            }
            _ => Ok(None),
        }
    }

    pub(crate) fn collection_from(
        &self,
        ctx: &Context,
        variables: &mut StepVariables,
        name: &str,
    ) -> Result<Option<Vec<SelectionPart>>> {
        let temporary = ReadStepScope {
            source: match &self.source {
                Source::Memory(parent) => Source::Memory(parent),
                Source::Detached {
                    allocation,
                    base,
                    current,
                    frame,
                } => Source::Detached {
                    allocation: *allocation,
                    base,
                    current,
                    frame: *frame,
                },
            },
            variables,
            fixed: self.fixed,
        };
        temporary.collection(ctx, name)
    }

    pub(crate) fn install_collection(
        &mut self,
        ctx: &Context,
        target: &str,
        parts: Vec<SelectionPart>,
    ) -> Result<()> {
        let value = self.collection_value(ctx, parts)?;
        self.variables.set_value(target, value);
        Ok(())
    }
    pub(crate) fn collection_value(
        &self,
        ctx: &Context,
        parts: Vec<SelectionPart>,
    ) -> Result<super::core::StepValue> {
        use super::core::StepValue;
        let value = match &self.source {
            Source::Memory(parent) => {
                let mut out = Vec::with_capacity(parts.len());
                for part in parts {
                    if ctx.is_cancelled() {
                        bail!("collection cancelled");
                    }
                    if let Some(original) = part.legacy {
                        out.push(original);
                        continue;
                    }
                    let source = part.source.as_ref().or(parent.frozen_view());
                    out.push(if let Some(source) = source {
                        parent.bound_selection(
                            GraphView::from_shared_membership(
                                source.read_state().clone(),
                                part.nodes,
                                part.edges,
                            )?,
                            part.name,
                        )
                    } else {
                        Subgraph::new(
                            parent.graph(),
                            part.nodes.as_ref().clone(),
                            part.edges.as_ref().clone(),
                            part.name,
                        )
                    });
                }
                StepValue::SubgraphArray(out)
            }
            Source::Detached { current, .. } => {
                let mut out = Vec::with_capacity(parts.len());
                for part in parts {
                    if ctx.is_cancelled() {
                        bail!("collection cancelled");
                    }
                    let source = part.source.as_ref().unwrap_or(current);
                    out.push(GraphView::from_shared_membership(
                        source.read_state().clone(),
                        part.nodes,
                        part.edges,
                    )?);
                }
                StepValue::GraphViewArray(out)
            }
        };
        if ctx.is_cancelled() {
            bail!("collection cancelled");
        }
        Ok(value)
    }

    /// Build a fixed frame from checked root authority, including provisional IDs.
    /// It reads pending values; a selector's historical values never replace writes.
    pub(crate) fn with_child_executor<T>(
        &self,
        ctx: &mut Context,
        part: &SelectionPart,
        run: impl FnOnce(&mut Context, super::core::StepExecutor<'_>, &mut StepVariables) -> Result<T>,
    ) -> Result<T> {
        let mut variables = StepVariables::default();
        match &self.source {
            Source::Memory(parent) => {
                let frame = Subgraph::new(
                    parent.graph(),
                    part.nodes.as_ref().clone(),
                    part.edges.as_ref().clone(),
                    "for_each frame".into(),
                );
                run(
                    ctx,
                    super::core::StepExecutor::for_frame(&frame, true),
                    &mut variables,
                )
            }
            Source::Detached {
                allocation, base, ..
            } => {
                let frame = ctx
                    .property_batch
                    .as_mut()
                    .ok_or_else(|| anyhow!("missing atomic execution scope"))?
                    .capture_membership(base, &part.nodes, &part.edges, true)?;
                run(
                    ctx,
                    super::core::StepExecutor::detached_fixed(*allocation, base, &frame),
                    &mut variables,
                )
            }
        }
    }

    pub(crate) fn variables(&self) -> &StepVariables {
        self.variables
    }
    pub(crate) fn variables_mut(&mut self) -> &mut StepVariables {
        self.variables
    }
    pub(crate) fn ordered_nodes(&self) -> Arc<[usize]> {
        match &self.source {
            Source::Memory(s) => s.ordered_nodes(),
            Source::Detached { current, .. } => current.node_ids().into(),
        }
    }
    pub(crate) fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        let ids = match &self.source {
            Source::Memory(s) => s.nodes(),
            Source::Detached { current, .. } => current.nodes(),
        };
        ids.iter().copied()
    }
    pub(crate) fn edge_ids(&self) -> impl Iterator<Item = usize> + '_ {
        let ids = match &self.source {
            Source::Memory(s) => s.edges(),
            Source::Detached { current, .. } => current.edges(),
        };
        ids.iter().copied()
    }
    pub(crate) fn node_count(&self) -> usize {
        match &self.source {
            Source::Memory(s) => s.node_count(),
            Source::Detached { current, .. } => current.nodes().len(),
        }
    }
    pub(crate) fn edge_count(&self) -> usize {
        match &self.source {
            Source::Memory(s) => s.edge_count(),
            Source::Detached { current, .. } => current.edges().len(),
        }
    }
    pub(crate) fn validate_selected_read(&self) -> Result<()> {
        if let Source::Memory(s) = &self.source {
            s.validate_selected_read()?;
        }
        Ok(())
    }
    pub(crate) fn is_directed(&self) -> bool {
        match &self.source {
            Source::Memory(s) => s.read_is_directed(),
            Source::Detached { current, .. } => current.is_directed(),
        }
    }
    pub(crate) fn has_node(&self, id: usize) -> bool {
        match &self.source {
            Source::Memory(s) => s.has_node(id),
            Source::Detached { current, .. } => current.nodes().contains(&id),
        }
    }
    pub(crate) fn has_edge(&self, id: usize) -> bool {
        match &self.source {
            Source::Memory(s) => s.edges().contains(&id),
            Source::Detached { current, .. } => current.edges().contains(&id),
        }
    }
    pub(crate) fn ordered_edges(&self) -> Arc<[usize]> {
        match &self.source {
            Source::Memory(s) => s.ordered_edges(),
            Source::Detached { current, .. } => current.edge_ids().into(),
        }
    }
    pub(crate) fn edge_endpoints(&self, id: usize) -> GraphResult<(usize, usize)> {
        match &self.source {
            Source::Memory(s) => s.edge_endpoints(id),
            Source::Detached { current, .. } => current.edge_endpoints(id),
        }
    }
    /// Explicit direction, sorted neighbors, parallel incidences and one self-loop.
    pub(crate) fn selected_csr(
        &self,
        ctx: &Context,
        direction: super::direction::NeighborDirection,
    ) -> Result<SelectedCsr> {
        use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
        if ctx.is_cancelled() {
            return Err(anyhow!("selected topology cancelled"));
        }
        self.validate_selected_read()?;
        let nodes = self.ordered_nodes();
        // Engine-assigned integer IDs use the same index hasher as AlgorithmRead.
        let positions: rustc_hash::FxHashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(row, &node)| (node, row))
            .collect();
        let edges = self.ordered_edges();
        let mapping = direction.edge_mapping();
        let mut failure = None;
        let mut csr = Csr::default();
        build_csr_from_edges_with_scratch(
            &mut csr,
            nodes.len(),
            edges.iter().copied(),
            |node| positions.get(&node).copied(),
            |edge| {
                if ctx.is_cancelled() {
                    failure = Some(anyhow!("selected topology cancelled"));
                    return None;
                }
                match self.edge_endpoints(edge) {
                    Ok(pair) => {
                        if !positions.contains_key(&pair.0) || !positions.contains_key(&pair.1) {
                            failure = Some(anyhow!(
                                "Selected edge {} endpoint is outside selection",
                                edge
                            ));
                            None
                        } else {
                            Some(mapping(pair))
                        }
                    }
                    Err(error) => {
                        failure = Some(error.into());
                        None
                    }
                }
            },
            CsrOptions {
                add_reverse_edges: direction.add_reverse_edges(),
                sort_neighbors: true,
            },
        );
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(SelectedCsr {
            nodes,
            positions,
            csr,
        })
    }

    /// Graph-default adjacency is outgoing for directed selections and symmetric
    /// for undirected selections. This is shared by expression and label updates.
    pub(crate) fn selected_neighbors(
        &self,
        ctx: &Context,
    ) -> Result<(Vec<usize>, HashMap<usize, Vec<usize>>)> {
        use super::direction::NeighborDirection;
        let topology = self.selected_csr(
            ctx,
            if self.is_directed() {
                NeighborDirection::Out
            } else {
                NeighborDirection::Undirected
            },
        )?;
        let rows = topology
            .nodes
            .iter()
            .enumerate()
            .map(|(index, &node)| {
                (
                    node,
                    topology
                        .csr
                        .neighbors(index)
                        .iter()
                        .map(|&i| topology.nodes[i])
                        .collect(),
                )
            })
            .collect();
        Ok((topology.nodes.to_vec(), rows))
    }
    pub(crate) fn node_attribute(
        &self,
        ctx: &Context,
        id: usize,
        name: &AttrName,
    ) -> GraphResult<Option<AttrValue>> {
        match &self.source {
            Source::Memory(s) => ctx.node_attribute(s, id, name),
            Source::Detached { current, .. } => current.node_attr(id, name),
        }
    }
    pub(crate) fn edge_attribute(
        &self,
        ctx: &Context,
        id: usize,
        name: &AttrName,
    ) -> GraphResult<Option<AttrValue>> {
        match &self.source {
            Source::Memory(s) => ctx.edge_attribute(s, id, name),
            Source::Detached { current, .. } => current.edge_attr(id, name),
        }
    }
    pub(crate) fn capture(&self, ctx: &mut Context) -> Result<GraphView> {
        let batch = ctx
            .property_batch
            .as_mut()
            .ok_or_else(|| anyhow!("staged capture requires persistent atomic execution"))?;
        Ok(match &self.source {
            Source::Memory(s) => batch.capture_scoped(s, self.fixed)?,
            Source::Detached { base, frame, .. } => {
                let selected = frame.unwrap_or(base);
                batch.capture_membership(
                    base,
                    selected.nodes(),
                    selected.edges(),
                    frame.is_some(),
                )?
            }
        })
    }
    pub(crate) fn add(
        &self,
        ctx: &mut Context,
        nodes: bool,
        count: usize,
        edges: &[(BatchNodeRef, BatchNodeRef)],
        attributes: &HashMap<AttrName, AttrValue>,
        at: Option<crate::TimestampMicros>,
    ) -> Result<Vec<usize>> {
        let batch = ctx
            .property_batch
            .as_mut()
            .ok_or_else(|| anyhow!("topology addition requires persistent atomic execution"))?;
        match &self.source {
            Source::Memory(s) => {
                let owner = s.graph();
                let graph = owner.borrow();
                if nodes {
                    Ok(batch.stage_nodes(&graph, count, attributes, at)?)
                } else {
                    Ok(batch.stage_edges(&graph, s, edges, attributes, at)?)
                }
            }
            Source::Detached {
                allocation,
                base,
                current,
                ..
            } => {
                if nodes {
                    Ok(batch.stage_nodes_at_allocation(*allocation, count, attributes, at)?)
                } else {
                    Ok(batch.stage_edges_in_view(
                        *allocation,
                        base,
                        current,
                        edges,
                        attributes,
                        at,
                    )?)
                }
            }
        }
    }
    pub(crate) fn attach_table(
        &self,
        ctx: &mut Context,
        table: &crate::storage::table::BaseTable,
        columns: &[(AttrName, AttrName)],
        at: Option<crate::TimestampMicros>,
    ) -> Result<()> {
        let batch = ctx
            .property_batch
            .as_mut()
            .ok_or_else(|| anyhow!("table.attach requires a persistent atomic builder"))?;
        match &self.source {
            Source::Memory(s) => batch.attach_table(s, table, columns, at)?,
            Source::Detached { current, .. } => batch.attach_table_membership(
                current.nodes(),
                current.edges(),
                table,
                columns,
                at,
            )?,
        }
        Ok(())
    }

    pub(crate) fn remove(
        &self,
        ctx: &mut Context,
        nodes: bool,
        ids: Vec<usize>,
        at: Option<crate::TimestampMicros>,
    ) -> Result<()> {
        let label = if nodes {
            "graph.remove_nodes target"
        } else {
            "graph.remove_edges target"
        };
        match &self.source {
            Source::Memory(s) => {
                let owner = s.graph();
                let graph = owner.borrow();
                validate_targets(ctx, &ids, "topology removal cancelled", label, |id| {
                    if nodes {
                        s.nodes().contains(&id) && graph.has_node(id)
                    } else {
                        s.edges().contains(&id) && graph.has_edge(id)
                    }
                })?;
            }
            Source::Detached { base, current, .. } => {
                validate_targets(ctx, &ids, "topology removal cancelled", label, |id| {
                    if nodes {
                        base.contains_node(id) && current.contains_node(id)
                    } else {
                        base.contains_edge(id) && current.contains_edge(id)
                    }
                })?
            }
        }
        ctx.property_batch
            .as_mut()
            .ok_or_else(|| anyhow!("missing atomic effect scope"))?
            .stage_removals(nodes, ids, at);
        Ok(())
    }

    pub(crate) fn revise_endpoints(
        &self,
        ctx: &mut Context,
        rows: &[(usize, BatchNodeRef, BatchNodeRef)],
        at: Option<crate::TimestampMicros>,
    ) -> Result<()> {
        let batch = ctx
            .property_batch
            .as_ref()
            .ok_or_else(|| anyhow!("missing atomic effect scope"))?;
        let prepared = match &self.source {
            Source::Memory(s) => {
                let owner = s.graph();
                let graph = owner.borrow();
                prepare_endpoints(
                    ctx,
                    rows,
                    |edge| s.edges().contains(&edge) && graph.has_edge(edge),
                    |reference| batch.resolve_selected_node(&graph, s, reference),
                )?
            }
            Source::Detached { base, current, .. } => prepare_endpoints(
                ctx,
                rows,
                |edge| base.contains_edge(edge) && current.contains_edge(edge),
                |reference| batch.resolve_node_in_view(base, current, reference),
            )?,
        };
        ctx.property_batch
            .as_mut()
            .ok_or_else(|| anyhow!("missing atomic effect scope"))?
            .stage_endpoint_revisions(prepared, at);
        Ok(())
    }

    pub(crate) fn attach_node_column(
        &self,
        ctx: &mut Context,
        name: AttrName,
        values: Vec<(usize, AttrValue)>,
    ) -> Result<()> {
        if let Source::Memory(s) = &self.source {
            validate_attachment_targets(s, self.fixed, true, values.iter().map(|(id, _)| *id))?;
            return ctx.attach_node_column(s, name, values);
        }
        self.attach(ctx, true, HashMap::from([(name, values)]), None)
    }
    pub(crate) fn attach_node_attrs(
        &self,
        ctx: &mut Context,
        attrs: HashMap<AttrName, Vec<(usize, AttrValue)>>,
    ) -> Result<()> {
        self.attach(ctx, true, attrs, None)
    }
    pub(crate) fn attach_edge_attrs(
        &self,
        ctx: &mut Context,
        attrs: HashMap<AttrName, Vec<(usize, AttrValue)>>,
    ) -> Result<()> {
        self.attach(ctx, false, attrs, None)
    }
    pub(crate) fn attach_attrs_at(
        &self,
        ctx: &mut Context,
        nodes: bool,
        attrs: HashMap<AttrName, Vec<(usize, AttrValue)>>,
        at: crate::TimestampMicros,
    ) -> Result<()> {
        self.attach(ctx, nodes, attrs, Some(at))
    }
    fn attach(
        &self,
        ctx: &mut Context,
        nodes: bool,
        attrs: HashMap<AttrName, Vec<(usize, AttrValue)>>,
        at: Option<crate::TimestampMicros>,
    ) -> Result<()> {
        if ctx.is_cancelled() {
            bail!("attribute attachment cancelled");
        }
        match &self.source {
            Source::Memory(s) => {
                validate_attachment_targets(
                    s,
                    self.fixed,
                    nodes,
                    attrs.values().flatten().map(|(id, _)| *id),
                )?;
                if let Some(at) = at {
                    ctx.attach_attrs_at(s, nodes, attrs, at)
                } else if nodes {
                    ctx.attach_node_attrs(s, attrs)
                } else {
                    ctx.attach_edge_attrs(s, attrs)
                }
            }
            Source::Detached { current, .. } => {
                crate::api::graph::bulk_validation::validate(&attrs, nodes, None, |id| {
                    if nodes {
                        current.contains_node(id)
                    } else {
                        current.contains_edge(id)
                    }
                })?;
                ctx.property_batch
                    .as_mut()
                    .ok_or_else(|| anyhow!("missing atomic attachment scope"))?
                    .stage_at(nodes, attrs, at)?;
                Ok(())
            }
        }
    }
}

fn validate_targets(
    ctx: &Context,
    ids: &[usize],
    cancelled: &str,
    label: &str,
    mut contains: impl FnMut(usize) -> bool,
) -> Result<()> {
    for &id in ids {
        if ctx.is_cancelled() {
            bail!("{cancelled}");
        }
        if !contains(id) {
            bail!("{label} {id} is outside the active execution selection");
        }
    }
    Ok(())
}
fn prepare_endpoints(
    ctx: &Context,
    rows: &[(usize, BatchNodeRef, BatchNodeRef)],
    mut edge_ok: impl FnMut(usize) -> bool,
    mut node: impl FnMut(BatchNodeRef) -> GraphResult<usize>,
) -> Result<Vec<(usize, usize, usize)>> {
    let mut prepared = Vec::with_capacity(rows.len());
    for &(edge, source, target) in rows {
        if ctx.is_cancelled() {
            bail!("endpoint revision cancelled");
        }
        if !edge_ok(edge) {
            bail!("endpoint revision edge {edge} is outside the active execution selection");
        }
        prepared.push((edge, node(source)?, node(target)?));
    }
    if ctx.is_cancelled() {
        bail!("endpoint revision cancelled");
    }
    Ok(prepared)
}

#[cfg(test)]
mod collection_tests {
    use super::*;
    use crate::{algorithms::steps::core::StepValue, Graph};

    fn graph() -> Graph {
        let mut graph = Graph::new();
        graph.add_nodes(2).unwrap();
        graph.add_edge(0, 1).unwrap();
        graph
            .set_node_attr(0, "value".into(), AttrValue::Int(1))
            .unwrap();
        graph
    }
    #[test]
    fn portable_collections_preserve_pins_and_reject_foreign_history_without_replacing_targets() {
        let mut owner = graph();
        let base = owner.view().unwrap();
        let source = base.select_nodes(&[0]).unwrap();
        owner
            .set_node_attr(0, "value".into(), AttrValue::Int(2))
            .unwrap();
        let current = owner.view().unwrap();
        let mut vars = StepVariables::default();
        vars.set_value("old", StepValue::GraphViewArray(vec![source.clone()]));
        let mut scope = ReadStepScope::detached((2, 1), &base, &current, None, &mut vars);
        let parts = scope.collection(&Context::new(), "old").unwrap().unwrap();
        scope
            .install_collection(&Context::new(), "pass", parts)
            .unwrap();
        let Some(StepValue::GraphViewArray(pins)) = scope.variables().value("pass") else {
            panic!("portable array")
        };
        assert_eq!(
            pins[0].node_attr(0, &"value".into()).unwrap(),
            Some(AttrValue::Int(1))
        );
        let foreign = graph().view().unwrap();
        scope
            .variables_mut()
            .set_value("bad", StepValue::GraphViewArray(vec![foreign]));
        assert!(scope.collection(&Context::new(), "bad").is_err());
        let Some(StepValue::GraphViewArray(pins)) = scope.variables().value("pass") else {
            panic!("old target")
        };
        assert_eq!(
            pins[0].node_attr(0, &"value".into()).unwrap(),
            Some(AttrValue::Int(1))
        );
    }
    #[test]
    fn portable_worker_rejects_connected_arrays_and_memory_preserves_legacy_payload() {
        let owner = graph().into_shared();
        let selected =
            Subgraph::from_nodes(owner.clone(), [0, 1].into_iter().collect(), "input".into())
                .unwrap()
                .freeze()
                .unwrap();
        let view = selected.pin_view().unwrap();
        let mut vars = StepVariables::default();
        vars.set_subgraph_array("input", vec![selected.clone()]);
        let scope = ReadStepScope::detached((2, 1), &view, &view, None, &mut vars);
        assert!(scope.collection(&Context::new(), "input").is_err());
        let mut scope = ReadStepScope::memory(&selected, &mut vars, false);
        let parts = scope.collection(&Context::new(), "input").unwrap().unwrap();
        scope
            .install_collection(&Context::new(), "output", parts)
            .unwrap();
        let array = scope.variables().subgraph_array("output").unwrap();
        assert!(Arc::ptr_eq(&array[0].graph(), &owner));
        assert_eq!(array[0].nodes(), selected.nodes());
        assert!(array[0].frozen_view().is_some());
        use crate::traits::GraphEntity;
        assert_eq!(array[0].entity_id(), selected.entity_id());
    }
    #[test]
    fn collection_cancellation_and_invalid_membership_preserve_existing_binding() {
        use std::sync::atomic::AtomicBool;
        let owner = graph();
        let view = owner.view().unwrap();
        let mut vars = StepVariables::default();
        vars.set_value("out", StepValue::GraphViewArray(vec![view.clone()]));
        let mut scope = ReadStepScope::detached((2, 1), &view, &view, None, &mut vars);
        let part = || SelectionPart {
            nodes: Arc::new(HashSet::from([100])),
            edges: Arc::new(HashSet::new()),
            name: "bad".into(),
            source: None,
            legacy: None,
        };
        assert!(scope
            .install_collection(&Context::new(), "out", vec![part()])
            .is_err());
        let ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        assert!(scope.install_collection(&ctx, "out", vec![]).is_err());
        let Some(StepValue::GraphViewArray(old)) = scope.variables().value("out") else {
            panic!("binding")
        };
        assert_eq!(old[0].node_ids(), vec![0, 1]);
    }
}
