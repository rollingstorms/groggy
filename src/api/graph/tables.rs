//! Current-state table reader adapters; graph projections share their column kernels.
use super::Graph;
use crate::{
    storage::table::projection::{self, OwnedTableData},
    EdgeId, GraphError, GraphResult, NodeId,
};
impl Graph {
    pub(crate) fn node_table_data_for_entities(
        &self,
        nodes: &[NodeId],
    ) -> GraphResult<OwnedTableData> {
        for &id in nodes {
            if !self.space().contains_node(id) {
                return Err(GraphError::InvalidInput(format!(
                    "Inactive table node {id}"
                )));
            }
        }
        let pool = self.pool();
        let (names, _) = pool.attribute_names();
        projection::nodes(nodes, names.into_iter().collect(), |id, name| {
            self.space()
                .get_node_attr_index(id, name)
                .map(|index| {
                    pool.get_attr_by_index(name, index, true).ok_or_else(|| {
                        GraphError::InvalidInput(format!(
                            "Invalid table reference for node {id}, attribute {name}"
                        ))
                    })
                })
                .transpose()
        })
    }
    pub(crate) fn edge_table_data_for_entities(
        &self,
        edges: &[EdgeId],
    ) -> GraphResult<OwnedTableData> {
        let pool = self.pool();
        let (_, names) = pool.attribute_names();
        projection::edges(
            edges,
            names.into_iter().collect(),
            |id| self.edge_endpoints(id),
            |id, name| {
                self.space()
                    .get_edge_attr_index(id, name)
                    .map(|index| {
                        pool.get_attr_by_index(name, index, false).ok_or_else(|| {
                            GraphError::InvalidInput(format!(
                                "Invalid table reference for edge {id}, attribute {name}"
                            ))
                        })
                    })
                    .transpose()
            },
        )
    }
}
