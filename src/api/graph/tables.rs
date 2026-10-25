//! Public table exports share the immutable GraphView projection/identity path.
use super::Graph;
use crate::{
    storage::table::{EdgesTable, GraphTable, NodesTable},
    GraphResult,
};
impl Graph {
    /// Capture one current immutable state and export source-bound node/edge tables.
    pub fn table(&self) -> GraphResult<GraphTable> {
        self.read_view()?.table()
    }
    /// Source-bound node table in stable ID order from a fresh immutable capture.
    pub fn nodes_table(&self) -> GraphResult<NodesTable> {
        self.read_view()?.nodes_table()
    }
    /// Source-bound edge table in stable ID order from a fresh immutable capture.
    pub fn edges_table(&self) -> GraphResult<EdgesTable> {
        self.read_view()?.edges_table()
    }
}
