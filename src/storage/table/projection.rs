//! Columnar graph-to-table projection shared by live and frozen readers.
use super::{BaseTable, EdgesTable, NodesTable};
use crate::{
    storage::array::BaseArray, AttrName, AttrValue, EdgeId, GraphError, GraphResult, NodeId,
};
use std::collections::HashMap;

/// Owned, thread-transferable graph table columns, without display/server state.
/// Missing properties are null cells. Optional immutable source provenance permits
/// explicit checked publication; cell buffers remain materialized copies.
#[derive(Debug)]
pub struct OwnedTableData {
    source: Option<super::provenance::TableProvenance>,
    columns: HashMap<String, Vec<AttrValue>>,
    order: Vec<String>,
}
impl OwnedTableData {
    /// Construct an independent table with stable column order.
    pub fn into_base_table(self) -> GraphResult<BaseTable> {
        let mut table = BaseTable::with_column_order(
            self.columns
                .into_iter()
                .map(|(name, values)| (name, BaseArray::from_attr_values(values)))
                .collect(),
            self.order,
        )?;
        table.provenance = self.source;
        Ok(table)
    }
    pub(crate) fn with_graph_source(
        mut self,
        pin: crate::GraphView,
        nodes: bool,
        ids: Vec<usize>,
    ) -> GraphResult<Self> {
        if self
            .columns
            .values()
            .any(|column| column.len() != ids.len())
        {
            return Err(GraphError::InvalidInput(
                "Source projection length mismatch".into(),
            ));
        }
        self.source = Some(super::provenance::TableProvenance::for_graph(
            pin, nodes, ids,
        )?);
        Ok(self)
    }
    /// Construct a node table, validating its structural column.
    pub fn into_nodes_table(self) -> GraphResult<NodesTable> {
        NodesTable::from_base_table(self.into_base_table()?)
    }
    /// Construct an edge table, validating its structural columns.
    pub fn into_edges_table(self) -> GraphResult<EdgesTable> {
        EdgesTable::from_base_table(self.into_base_table()?)
    }
}
fn id_value(id: usize) -> GraphResult<AttrValue> {
    i64::try_from(id)
        .map(AttrValue::Int)
        .map_err(|_| GraphError::InvalidInput("Graph ID exceeds table integer range".into()))
}
fn properties<'a>(
    ids: &[usize],
    mut names: Vec<AttrName>,
    mut data: OwnedTableData,
    mut value: impl FnMut(usize, &AttrName) -> GraphResult<Option<&'a AttrValue>>,
) -> GraphResult<OwnedTableData> {
    names.sort_unstable();
    names.dedup();
    for name in names {
        let mut present = false;
        let mut values = Vec::with_capacity(ids.len());
        for &id in ids {
            let cell = value(id, &name)?;
            present |= cell.is_some();
            values.push(cell.cloned().unwrap_or(AttrValue::Null));
        }
        if present {
            if data.columns.contains_key(&name) {
                return Err(GraphError::InvalidInput(format!(
                    "Graph property '{name}' conflicts with a structural table column"
                )));
            }
            data.order.push(name.clone());
            data.columns.insert(name, values);
        }
    }
    Ok(data)
}
pub(crate) fn nodes<'a>(
    ids: &[NodeId],
    names: Vec<AttrName>,
    value: impl FnMut(NodeId, &AttrName) -> GraphResult<Option<&'a AttrValue>>,
) -> GraphResult<OwnedTableData> {
    let columns = HashMap::from([(
        "node_id".into(),
        ids.iter()
            .map(|&id| id_value(id))
            .collect::<GraphResult<_>>()?,
    )]);
    properties(
        ids,
        names,
        OwnedTableData {
            source: None,
            columns,
            order: vec!["node_id".into()],
        },
        value,
    )
}
pub(crate) fn edges<'a>(
    ids: &[EdgeId],
    names: Vec<AttrName>,
    mut endpoints: impl FnMut(EdgeId) -> GraphResult<(NodeId, NodeId)>,
    value: impl FnMut(EdgeId, &AttrName) -> GraphResult<Option<&'a AttrValue>>,
) -> GraphResult<OwnedTableData> {
    let mut sources = Vec::with_capacity(ids.len());
    let mut targets = Vec::with_capacity(ids.len());
    for &id in ids {
        let (source, target) = endpoints(id)?;
        sources.push(id_value(source)?);
        targets.push(id_value(target)?);
    }
    let columns = HashMap::from([
        (
            "edge_id".into(),
            ids.iter()
                .map(|&id| id_value(id))
                .collect::<GraphResult<_>>()?,
        ),
        ("source".into(), sources),
        ("target".into(), targets),
    ]);
    properties(
        ids,
        names,
        OwnedTableData {
            source: None,
            columns,
            order: vec!["edge_id".into(), "source".into(), "target".into()],
        },
        value,
    )
}
