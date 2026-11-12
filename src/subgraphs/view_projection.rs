//! Bulk projections keep captured sources in Rust and marshal their outputs once.
use super::GraphView;
use crate::{
    storage::{array::BaseArray, table::BaseTable},
    AttrName, AttrValue, GraphResult,
};
use rayon::prelude::*;
use std::collections::HashMap;

fn project_views<T: Send>(
    views: &[GraphView],
    project: impl Fn(&GraphView) -> T + Sync + Send,
) -> Vec<T> {
    // Many modest selections parallelize across groups. One/few large selections
    // retain the existing column reader's row parallelism. Do not retain sorted
    // membership or materialized columns merely to accelerate a subsequent read.
    let rows = views.iter().fold(0usize, |rows, view| {
        rows.saturating_add(view.node_count().max(view.edge_count()))
    });
    if views.len() >= rayon::current_num_threads().max(2) && rows >= 4096 {
        views.par_iter().map(project).collect()
    } else {
        views.iter().map(project).collect()
    }
}

impl GraphView {
    /// Detached aligned node-column tables, in input selection order.
    /// Each selection retains its own capture; no cross-source union is performed.
    /// Missing cells become null and duplicate names coalesce, as in BaseTable.
    /// Empty column lists return empty tables. This read publishes no changes.
    pub fn project_node_columns(views: &[Self], names: &[AttrName]) -> GraphResult<Vec<BaseTable>> {
        project_views(views, |view| {
            let nodes = view.node_accessor();
            BaseTable::from_columns(
                names
                    .iter()
                    .map(|name| (name.clone(), nodes.column(name)))
                    .collect::<HashMap<_, _>>(),
            )
        })
        .into_iter()
        .collect()
    }

    /// Present observations from each selection, in ascending original ID order.
    /// A present node property (including null) takes precedence over edge values.
    /// Otherwise selected edges supply observations. None means absent in both
    /// domains; Some(empty) is not substituted for absence. Input order is kept.
    pub fn project_attribute_observations(
        views: &[Self],
        name: &AttrName,
    ) -> Vec<Option<BaseArray<AttrValue>>> {
        project_views(views, |view| {
            let nodes = view.node_accessor();
            if nodes.has_column(name) {
                return Some(nodes.present_column(name));
            }
            let edges = view.edge_accessor();
            edges.has_column(name).then(|| edges.present_column(name))
        })
    }
}
