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
