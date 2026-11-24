//! Sealed native step invocation over read authority. There is deliberately no
//! `subgraph()` or mutable owner accessor. Legacy callbacks retain StepScope.
use super::core::{validate_attachment_targets, StepVariables};
use crate::{
    algorithms::Context, subgraphs::Subgraph, AttrName, AttrValue, BatchNodeRef, GraphResult,
    GraphView,
};
use anyhow::{anyhow, bail, Result};
use std::{collections::HashMap, sync::Arc};

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Native kernels may be shared by memory and durable adapters. Sealing keeps
/// the portable guarantee independent of custom callbacks' effect declarations.
#[doc(hidden)]
pub trait ReadStep: sealed::Sealed + Send + Sync {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()>;
}

enum Source<'a> {
    Memory(&'a Subgraph),
    Detached {
        allocation: (usize, usize),
        base: &'a GraphView,
        current: &'a GraphView,
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
        variables: &'a mut StepVariables,
    ) -> Self {
        Self {
            source: Source::Detached {
                allocation,
                base,
                current,
            },
            variables,
            fixed: false,
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
                allocation, base, ..
            } => super::core::StepExecutor::detached(*allocation, base),
        };
        run(executor, self.variables)
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
            Source::Detached { base, .. } => {
                batch.capture_membership(base, base.nodes(), base.edges(), false)?
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
