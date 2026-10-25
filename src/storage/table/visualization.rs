//! Checked visualization adaptation: preserve table IDs without graph materialization.
use super::{structural, BaseTable, EdgesTable, GraphTable, NodesTable, Table};
use crate::{
    viz::streaming::{
        data_source::{GraphEdge, GraphNode},
        GraphDataSource,
    },
    AttrValue, GraphResult,
};
use std::collections::{BTreeSet, HashMap};

fn attributes(
    table: &BaseTable,
    structural: &[&str],
) -> GraphResult<Vec<HashMap<String, AttrValue>>> {
    let mut rows = vec![HashMap::new(); table.nrows()];
    for name in table.column_names() {
        if !structural.contains(&name.as_str()) {
            for (row, value) in structural::column(table, name)?.iter().enumerate() {
                rows[row].insert(name.clone(), value.clone());
            }
        }
    }
    Ok(rows)
}

fn nodes(table: &NodesTable) -> GraphResult<Vec<GraphNode>> {
    let ids = structural::ids(table.base(), "node_id")?;
    Ok(ids
        .into_iter()
        .zip(attributes(table.base(), &["node_id"])?)
        .map(|(id, attributes)| GraphNode {
            id: id.to_string(),
            label: None,
            attributes,
            position: None,
        })
        .collect())
}

fn edges(table: &EdgesTable) -> GraphResult<Vec<GraphEdge>> {
    let ids = structural::ids(table.base(), "edge_id")?;
    let sources = structural::ids(table.base(), "source")?;
    let targets = structural::ids(table.base(), "target")?;
    Ok(ids
        .into_iter()
        .zip(sources)
        .zip(targets)
        .zip(attributes(table.base(), &["edge_id", "source", "target"])?)
        .map(|(((id, source), target), attributes)| GraphEdge {
            id: id.to_string(),
            source: source.to_string(),
            target: target.to_string(),
            label: None,
            weight: None,
            attributes,
        })
        .collect())
}

impl NodesTable {
    /// Capture node rows and typed properties, preserving IDs and rejecting duplicates.
    pub fn to_viz_data_source(&self) -> GraphResult<GraphDataSource> {
        GraphDataSource::from_entities(nodes(self)?, vec![], false)
    }
}

impl EdgesTable {
    /// Capture edges and their unique endpoint nodes. Direction is explicit;
    /// synthesized endpoint nodes have no properties from an outside graph.
    pub fn to_viz_data_source(&self, is_directed: bool) -> GraphResult<GraphDataSource> {
        let edges = edges(self)?;
        let endpoints: BTreeSet<_> = structural::ids(self.base(), "source")?
            .into_iter()
            .chain(structural::ids(self.base(), "target")?)
            .collect();
        let nodes = endpoints
            .into_iter()
            .map(|id| GraphNode {
                id: id.to_string(),
                label: None,
                attributes: HashMap::new(),
                position: None,
            })
            .collect();
        GraphDataSource::from_entities(nodes, edges, is_directed)
    }
}

impl GraphTable {
    /// Capture both components with exact table IDs and typed properties.
    /// Dangling endpoints reject; direction is explicit because a standalone
    /// node/edge bundle does not define graph direction.
    pub fn to_viz_data_source(&self, is_directed: bool) -> GraphResult<GraphDataSource> {
        GraphDataSource::from_entities(nodes(self.nodes())?, edges(self.edges())?, is_directed)
    }
}
