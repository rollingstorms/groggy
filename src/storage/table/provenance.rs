//! Frozen graph provenance and row identity shared by the existing table foundation.
use super::{structural, BaseTable, GraphTable, Table};
use crate::{
    AttrName, BatchEdgeRef as E, BatchNodeRef as N, DurableGraph, Graph, GraphBatch, GraphError,
    GraphResult, GraphSchema, GraphView, StateId, TimestampMicros, ViewBatchResult,
};
use std::{collections::HashSet, sync::Arc};

#[derive(Clone, Debug)]
struct RowBinding {
    nodes: bool,
    ids: Vec<usize>,
}
#[derive(Clone, Debug)]
pub(super) struct TableProvenance {
    sources: Arc<Vec<GraphView>>,
    rows: Option<Arc<RowBinding>>,
}
impl TableProvenance {
    pub(super) fn for_graph(pin: GraphView, nodes: bool, ids: Vec<usize>) -> GraphResult<Self> {
        if ids.iter().any(|&id| {
            if nodes {
                !pin.contains_node(id)
            } else {
                !pin.contains_edge(id)
            }
        }) {
            return Err(invalid(
                "Table source row identity does not match captured selection",
            ));
        }
        Ok(Self {
            sources: Arc::new(vec![pin]),
            rows: Some(Arc::new(RowBinding { nodes, ids })),
        })
    }
    pub(super) fn derived(&self) -> Self {
        Self {
            sources: self.sources.clone(),
            rows: None,
        }
    }
    pub(super) fn for_rows(&self, count: usize) -> Self {
        if self
            .rows
            .as_ref()
            .is_some_and(|rows| rows.ids.len() == count)
        {
            self.clone()
        } else {
            self.derived()
        }
    }
    pub(super) fn take(&self, indices: &[usize]) -> Self {
        let rows = self.rows.as_ref().and_then(|rows| {
            indices
                .iter()
                .map(|&index| rows.ids.get(index).copied())
                .collect::<Option<Vec<_>>>()
                .map(|ids| {
                    Arc::new(RowBinding {
                        nodes: rows.nodes,
                        ids,
                    })
                })
        });
        Self {
            sources: self.sources.clone(),
            rows,
        }
    }
    pub(super) fn slice(&self, start: usize, end: usize) -> Self {
        self.take(&(start..end).collect::<Vec<_>>())
    }
    pub(super) fn filter(&self, mask: &[bool]) -> Self {
        self.take(
            &mask
                .iter()
                .enumerate()
                .filter_map(|(i, &keep)| keep.then_some(i))
                .collect::<Vec<_>>(),
        )
    }
    pub(super) fn joined(left: Option<&Self>, right: Option<&Self>) -> Option<Self> {
        let mut sources: Vec<GraphView> = Vec::new();
        for view in left.into_iter().chain(right).flat_map(|p| p.sources.iter()) {
            if !sources.iter().any(|old| {
                old.read_state().shares_capture(view.read_state())
                    && old.nodes() == view.nodes()
                    && old.edges() == view.edges()
            }) {
                sources.push(view.clone());
            }
        }
        (!sources.is_empty()).then(|| Self {
            sources: Arc::new(sources),
            rows: None,
        })
    }
}
fn invalid(message: impl Into<String>) -> GraphError {
    GraphError::InvalidInput(message.into())
}

impl BaseTable {
    /// A structural conversion keeps writable row identity only if it still
    /// describes the exact captured entity kind, IDs and edge endpoints.
    pub(super) fn retain_entity_conversion_binding(&mut self, nodes: bool) -> GraphResult<()> {
        let Some(provenance) = &self.provenance else {
            return Ok(());
        };
        let Some(rows) = &provenance.rows else {
            return Ok(());
        };
        let ids = structural::ids(self, if nodes { "node_id" } else { "edge_id" })?;
        let mut compatible = rows.nodes == nodes && rows.ids == ids;
        if compatible && !nodes {
            let sources = structural::ids(self, "source")?;
            let targets = structural::ids(self, "target")?;
            compatible = provenance.sources.first().is_some_and(|source| {
                ids.iter()
                    .zip(sources)
                    .zip(targets)
                    .all(|((&id, from), to)| {
                        source
                            .edge_endpoints(id)
                            .is_ok_and(|actual| actual == (from, to))
                    })
            });
        }
        if !compatible {
            self.provenance = Some(provenance.derived());
        }
        Ok(())
    }

    /// Immutable sources retained by this table, including read-only join/aggregate provenance.
    pub fn source_views(&self) -> Vec<GraphView> {
        self.provenance
            .as_ref()
            .map(|p| p.sources.as_ref().clone())
            .unwrap_or_default()
    }
    /// Stable row-to-entity identity, independent of editable structural columns.
    pub fn source_entity_ids(&self) -> Option<&[usize]> {
        self.provenance
            .as_ref()?
            .rows
            .as_ref()
            .map(|rows| rows.ids.as_slice())
    }
    /// "node" or "edge" for a directly bound row projection; None for derived/unbound rows.
    pub fn source_entity_kind(&self) -> Option<&'static str> {
        self.provenance
            .as_ref()?
            .rows
            .as_ref()
            .map(|rows| if rows.nodes { "node" } else { "edge" })
    }
    /// Expected accepted state when rows retain unambiguous write identity.
    pub fn source_state_id(&self) -> Option<StateId> {
        self.provenance.as_ref()?.rows.as_ref()?;
        self.provenance.as_ref()?.sources.first()?.write_state_id()
    }

    fn source_batch(
        &self,
        columns: &[AttrName],
        at: Option<TimestampMicros>,
    ) -> GraphResult<(GraphView, GraphBatch)> {
        let provenance = self.provenance.as_ref().ok_or_else(|| {
            invalid("Table has no pinned graph source; use explicit mapped writeback")
        })?;
        let rows = provenance.rows.as_ref().ok_or_else(|| {
            invalid("Derived/joined rows require explicit identity resolution before writeback")
        })?;
        let source = provenance
            .sources
            .first()
            .ok_or_else(|| invalid("Missing table source pin"))?;
        if rows.ids.len() != self.nrows() {
            return Err(invalid("Table row identity length changed"));
        }
        let mut seen = HashSet::new();
        if rows.ids.iter().any(|id| !seen.insert(id)) {
            return Err(invalid(
                "Duplicate source entity rows; reduce explicitly before writeback",
            ));
        }
        let id_column = if rows.nodes { "node_id" } else { "edge_id" };
        if self.has_column(id_column) && structural::ids(self, id_column)? != rows.ids {
            return Err(invalid(
                "Edited structural IDs differ from retained row identity",
            ));
        }
        if !rows.nodes {
            for (name, part) in [("source", 0), ("target", 1)] {
                if self.has_column(name) {
                    for (row, actual) in structural::ids(self, name)?.into_iter().enumerate() {
                        let (a, b) = source.edge_endpoints(rows.ids[row])?;
                        if actual != if part == 0 { a } else { b } {
                            return Err(invalid(
                                "Edited endpoints differ from captured edge identity",
                            ));
                        }
                    }
                }
            }
        }
        let pin = if rows.nodes {
            source.select_nodes(&rows.ids)?
        } else {
            source.select_edges(&rows.ids)?
        };
        let reserved = if rows.nodes {
            &["node_id", "entity_type", "contains_subgraph", "external_id"][..]
        } else {
            &["edge_id", "source", "target", "external_id"][..]
        };
        let values = super::writeback::columns(self, columns, reserved)?;
        let mut batch = GraphBatch {
            effective_at: at,
            ..Default::default()
        };
        for (name, values) in values {
            if rows.nodes {
                batch.node_attrs.insert(
                    name,
                    rows.ids
                        .iter()
                        .copied()
                        .zip(values)
                        .map(|(id, v)| (N::Existing(id), v))
                        .collect(),
                );
            } else {
                batch.edge_attrs.insert(
                    name,
                    rows.ids
                        .iter()
                        .copied()
                        .zip(values)
                        .map(|(id, v)| (E::Existing(id), v))
                        .collect(),
                );
            }
        }
        Ok((pin, batch))
    }
    /// Publish explicitly chosen property columns using retained row identity and source pin.
    /// Returns a refreshed immutable view; table cells and source pins remain unchanged.
    pub fn update_source_graph(
        &self,
        graph: &mut Graph,
        columns: &[AttrName],
        at: Option<TimestampMicros>,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<ViewBatchResult> {
        let (pin, batch) = self.source_batch(columns, at)?;
        if let Some(schema) = schema {
            graph.apply_view_batch_with_schema(
                &pin,
                batch,
                "source_table_update".into(),
                "groggy".into(),
                schema,
            )
        } else {
            graph.apply_view_batch(&pin, batch, "source_table_update".into(), "groggy".into())
        }
    }
    /// Durable equivalent: source/row validation precedes journal append and sync.
    pub fn update_source_durable(
        &self,
        graph: &mut DurableGraph,
        columns: &[AttrName],
        at: Option<TimestampMicros>,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<ViewBatchResult> {
        let (pin, batch) = self.source_batch(columns, at)?;
        graph.apply_view_batch_with_schema(
            &pin,
            batch,
            "source_table_update".into(),
            "groggy".into(),
            schema,
        )
    }
}
impl GraphTable {
    fn combined_source_batch(
        &self,
        node_columns: &[AttrName],
        edge_columns: &[AttrName],
        at: Option<TimestampMicros>,
    ) -> GraphResult<(GraphView, GraphBatch)> {
        if self.nodes().base().source_entity_kind() != Some("node")
            || self.edges().base().source_entity_kind() != Some("edge")
        {
            return Err(invalid(
                "Graph table components must retain node/edge row kinds",
            ));
        }
        let (nodes, mut batch) = self.nodes().base().source_batch(node_columns, at)?;
        let (edges, edge_batch) = self.edges().base().source_batch(edge_columns, at)?;
        if !nodes.read_state().shares_capture(edges.read_state()) {
            return Err(invalid(
                "Node and edge tables must share the exact source capture",
            ));
        }
        let mut node_ids = nodes.nodes().clone();
        node_ids.extend(edges.nodes());
        let pin = GraphView::from_membership(
            nodes.read_state().clone(),
            node_ids,
            edges.edges().clone(),
        )?;
        batch.edge_attrs = edge_batch.edge_attrs;
        Ok((pin, batch))
    }
    /// Publish mixed node/edge property columns from one shared source capture in one batch.
    pub fn update_source_graph(
        &self,
        graph: &mut Graph,
        node_columns: &[AttrName],
        edge_columns: &[AttrName],
        at: Option<TimestampMicros>,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<ViewBatchResult> {
        let (pin, batch) = self.combined_source_batch(node_columns, edge_columns, at)?;
        if let Some(schema) = schema {
            graph.apply_view_batch_with_schema(
                &pin,
                batch,
                "source_graph_table_update".into(),
                "groggy".into(),
                schema,
            )
        } else {
            graph.apply_view_batch(
                &pin,
                batch,
                "source_graph_table_update".into(),
                "groggy".into(),
            )
        }
    }
    /// Durable mixed-column equivalent with installed and optional per-call schema checks.
    pub fn update_source_durable(
        &self,
        graph: &mut DurableGraph,
        node_columns: &[AttrName],
        edge_columns: &[AttrName],
        at: Option<TimestampMicros>,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<ViewBatchResult> {
        let (pin, batch) = self.combined_source_batch(node_columns, edge_columns, at)?;
        graph.apply_view_batch_with_schema(
            &pin,
            batch,
            "source_graph_table_update".into(),
            "groggy".into(),
            schema,
        )
    }
}
