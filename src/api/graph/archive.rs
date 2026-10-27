//! Full retained graph archives. Active JSON export deliberately has a smaller contract.
use super::*;
use crate::state::history::HistoryDocument;
use crate::storage::{pool::PoolDocument, retained_group};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"GROGGYA1";
const VERSION: u32 = 1;
const HEADER: usize = 32;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphArchive {
    version: u32,
    config: GraphConfig,
    pool: PoolDocument,
    history: HistoryDocument,
    identities: ExternalIdentityIndex,
    branch: String,
    state: StateId,
    automatic_publication: bool,
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
    node_indices: Vec<(NodeId, AttrName, usize)>,
    edge_indices: Vec<(EdgeId, AttrName, usize)>,
}
fn invalid(message: impl std::fmt::Display) -> GraphError {
    GraphError::InvalidInput(format!("Invalid graph archive: {message}"))
}
fn io_error(path: &Path, operation: &str, error: impl std::fmt::Display) -> GraphError {
    GraphError::IoError {
        operation: operation.into(),
        path: path.to_string_lossy().into(),
        underlying_error: error.to_string(),
    }
}
#[cfg(test)]
fn same_snapshot(
    a: &crate::state::GraphSnapshot,
    b: &crate::state::GraphSnapshot,
) -> GraphResult<bool> {
    // Compare one borrowed value at a time instead of constructing two complete
    // serialized property maps. Large restored text is already canonical Text;
    // other types retain the exact bit/tag-aware portable codec semantics.
    let same_rows = |left: &HashMap<usize, HashMap<String, AttrValue>>,
                     right: &HashMap<usize, HashMap<String, AttrValue>>|
     -> GraphResult<bool> {
        if left.values().filter(|row| !row.is_empty()).count()
            != right.values().filter(|row| !row.is_empty()).count()
        {
            return Ok(false);
        }
        for (id, attrs) in left.iter().filter(|(_, row)| !row.is_empty()) {
            let Some(other) = right.get(id) else {
                return Ok(false);
            };
            if attrs.len() != other.len() {
                return Ok(false);
            }
            for (name, value) in attrs {
                let Some(other) = other.get(name) else {
                    return Ok(false);
                };
                let equal = same_value(value, other)?;
                if !equal {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    };
    Ok(a.graph_type == b.graph_type
        && a.state_id == b.state_id
        && a.active_nodes.iter().collect::<HashSet<_>>()
            == b.active_nodes.iter().collect::<HashSet<_>>()
        && a.edges == b.edges
        && same_rows(&a.node_attributes, &b.node_attributes)?
        && same_rows(&a.edge_attributes, &b.edge_attributes)?)
}
/// Compare current mappings against accepted history without cloning current values.
fn same_selected_snapshot(
    graph: &Graph,
    recorded: &crate::state::GraphSnapshot,
) -> GraphResult<bool> {
    if graph.graph_type() != recorded.graph_type
        || graph.current_commit != recorded.state_id
        || graph.space.get_active_nodes() != recorded.active_nodes.iter().copied().collect()
        || graph.edge_count() != recorded.edges.len()
    {
        return Ok(false);
    }
    let pool = graph.pool.borrow();
    for edge in graph.edge_ids() {
        let endpoints = pool
            .get_edge_endpoints(edge)
            .ok_or_else(|| invalid("missing selected endpoints"))?;
        if recorded.edges.get(&edge) != Some(&endpoints) {
            return Ok(false);
        }
    }
    for (is_node, rows) in [
        (true, &recorded.node_attributes),
        (false, &recorded.edge_attributes),
    ] {
        let mut count = 0usize;
        for (id, name, index) in graph.space.active_attr_indices(is_node) {
            let value = pool
                .get_attr_by_index(name, index, is_node)
                .ok_or_else(|| invalid("selected property index out of bounds"))?;
            let Some(other) = rows.get(&id).and_then(|row| row.get(name)) else {
                return Ok(false);
            };
            if !same_value(value, other)? {
                return Ok(false);
            }
            count += 1;
        }
        if count != rows.values().map(HashMap::len).sum::<usize>() {
            return Ok(false);
        }
    }
    Ok(true)
}
pub(super) fn same_value(value: &AttrValue, other: &AttrValue) -> GraphResult<bool> {
    match (value, other) {
        (AttrValue::Text(a), AttrValue::Text(b)) => Ok(a == b),
        _ => Ok(retained_group::encode(value)? == retained_group::encode(other)?),
    }
}
/// Detached validated archive bytes, suitable for I/O outside a language runtime lock.
/// Encoding/decoding graph ownership remains separate from file operations.
#[derive(Debug)]
pub struct PreparedArchive {
    bytes: Vec<u8>,
}
impl PreparedArchive {
    /// Write and sync a sibling temporary, then publish without overwriting a file.
    /// Directory-sync errors may leave a complete destination; reconcile before retry.
    pub fn save(&self, path: &Path) -> GraphResult<()> {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut temporary = tempfile::NamedTempFile::new_in(parent)
            .map_err(|e| io_error(path, "create_archive_temporary", e))?;
        temporary
            .write_all(&self.bytes)
            .and_then(|_| temporary.as_file().sync_all())
            .map_err(|e| io_error(path, "write_and_sync_archive", e))?;
        temporary
            .persist_noclobber(path)
            .map_err(|e| io_error(path, "publish_archive_without_overwrite", e.error))?;
        std::fs::File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(|e| io_error(parent, "sync_archive_directory", e))
    }
    /// Read framed bytes without constructing thread-local graph ownership.
    pub fn read(path: &Path) -> GraphResult<Self> {
        let mut file =
            std::fs::File::open(path).map_err(|e| io_error(path, "open_graph_archive", e))?;
        let length = file
            .metadata()
            .map_err(|e| io_error(path, "stat_graph_archive", e))?
            .len();
        let mut header = [0u8; HEADER];
        file.read_exact(&mut header)
            .map_err(|e| io_error(path, "read_archive_header", e))?;
        let payload_length = read_header(&header)?;
        if payload_length.checked_add(HEADER as u64) != Some(length) {
            return Err(invalid("truncated archive or trailing bytes"));
        }
        let size = usize::try_from(length).map_err(invalid)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(size).map_err(invalid)?;
        bytes.extend_from_slice(&header);
        file.read_to_end(&mut bytes)
            .map_err(|e| io_error(path, "read_graph_archive", e))?;
        if bytes.len() != size
            || header[24..32] != xxhash_rust::xxh3::xxh3_64(&bytes[HEADER..]).to_le_bytes()
        {
            return Err(invalid("length/checksum mismatch"));
        }
        Ok(Self { bytes })
    }
    /// Decode and validate into a new owner. Does not alter any existing graph.
    pub fn restore(self) -> GraphResult<Graph> {
        Graph::from_archive_bytes(&self.bytes)
    }
}
impl Graph {
    pub(crate) fn archive_document(&self) -> GraphResult<GraphArchive> {
        self.validate_write_head(self.current_commit)?;
        self.config.validate().map_err(invalid)?;
        let recorded = self.history.reconstruct_state_at(self.current_commit)?;
        if !same_selected_snapshot(self, &recorded)? {
            return Err(invalid("selected state differs from accepted history"));
        }
        let mut nodes = self.node_ids();
        nodes.sort_unstable();
        let mut edges = self.edge_ids();
        edges.sort_unstable();
        let mut node_indices: Vec<_> = self
            .space
            .active_attr_indices(true)
            .map(|(id, name, index)| (id, name.clone(), index))
            .collect();
        let mut edge_indices: Vec<_> = self
            .space
            .active_attr_indices(false)
            .map(|(id, name, index)| (id, name.clone(), index))
            .collect();
        node_indices.sort_unstable();
        edge_indices.sort_unstable();
        Ok(GraphArchive {
            version: if self
                .history_records()
                .iter()
                .any(|c| !c.delta.edge_endpoint_changes.is_empty())
            {
                4
            } else {
                self.pool.borrow().archive_version()
            },
            config: self.config.clone(),
            pool: self.pool.borrow().archive_document()?,
            history: self.history.archive_document()?,
            identities: self.identities.clone(),
            branch: self.current_branch.clone(),
            state: self.current_commit,
            automatic_publication: self.automatic_publication,
            nodes,
            edges,
            node_indices,
            edge_indices,
        })
    }

    /// Prepare an owned, validated binary archive. The resulting bytes can be
    /// written on another thread without retaining or borrowing this graph.
    pub fn prepare_archive(&self) -> GraphResult<PreparedArchive> {
        Ok(PreparedArchive {
            bytes: self.archive_bytes()?,
        })
    }

    /// Save all retained storage/history to a synced binary archive. The destination
    /// must not exist. Pending manual writes must be explicitly accepted first.
    pub fn save_archive(&self, path: &Path) -> GraphResult<()> {
        self.prepare_archive()?.save(path)
    }

    /// Restore a complete independent graph. The stored publication policy is retained.
    pub fn load_archive(path: &Path) -> GraphResult<Self> {
        PreparedArchive::read(path)?.restore()
    }
    pub(crate) fn archive_bytes(&self) -> GraphResult<Vec<u8>> {
        let document = self.archive_document()?;
        // Validate all retained backing data before any destination is created.
        // Explicit archives are a full-copy operation, never part of ordinary writes.
        let mut payload = Vec::new();
        ciborium::ser::into_writer(&document, &mut payload).map_err(invalid)?;
        let mut bytes = Vec::with_capacity(HEADER + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(!(payload.len() as u64)).to_le_bytes());
        bytes.extend_from_slice(&xxhash_rust::xxh3::xxh3_64(&payload).to_le_bytes());
        bytes.extend_from_slice(&payload);
        // Ensures a malformed native raw escape cannot create an unreadable archive.
        Self::from_archive_bytes(&bytes)?;
        Ok(bytes)
    }
    pub(crate) fn from_archive_bytes(bytes: &[u8]) -> GraphResult<Self> {
        let header: &[u8; HEADER] = bytes
            .get(..HEADER)
            .and_then(|h| h.try_into().ok())
            .ok_or_else(|| invalid("truncated archive header"))?;
        Self::decode_archive(header, &bytes[HEADER..])
    }
    fn decode_archive(header: &[u8; HEADER], payload: &[u8]) -> GraphResult<Self> {
        let length = read_header(header)?;
        if length != payload.len() as u64
            || header[24..32] != xxhash_rust::xxh3::xxh3_64(payload).to_le_bytes()
        {
            return Err(invalid("length/checksum mismatch"));
        }
        let mut cursor = std::io::Cursor::new(payload);
        let document: GraphArchive = ciborium::de::from_reader(&mut cursor).map_err(invalid)?;
        if cursor.position() != length {
            return Err(invalid("trailing payload data"));
        }
        document.restore()
    }
}
fn read_header(header: &[u8; HEADER]) -> GraphResult<u64> {
    let number = |range: std::ops::Range<usize>| -> GraphResult<u64> {
        Ok(u64::from_le_bytes(
            header[range].try_into().map_err(invalid)?,
        ))
    };
    let length = number(8..16)?;
    if &header[..8] != MAGIC || number(16..24)? != !length {
        return Err(invalid("unsupported format or invalid header"));
    }
    Ok(length)
}
impl GraphArchive {
    pub(crate) fn encode_document(&self) -> GraphResult<Vec<u8>> {
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(self, &mut bytes).map_err(invalid)?;
        Ok(bytes)
    }
    pub(crate) fn decode_document(bytes: &[u8]) -> GraphResult<Self> {
        let mut reader = std::io::Cursor::new(bytes);
        let document = ciborium::de::from_reader(&mut reader).map_err(invalid)?;
        if reader.position() != bytes.len() as u64 {
            return Err(invalid("trailing purge capture bytes"));
        }
        Ok(document)
    }
    pub(crate) fn replace_purged_storage(
        &mut self,
        outcome: &super::purge::PurgeOutcome,
    ) -> GraphResult<()> {
        self.version = self.version.max(outcome.pool.archive_version());
        self.pool = outcome.pool.archive_document()?;
        self.identities = outcome.identities.clone();
        self.node_indices = outcome
            .nodes
            .iter()
            .flat_map(|(name, rows)| {
                rows.iter()
                    .map(move |(&id, &index)| (id, name.clone(), index))
            })
            .collect();
        self.edge_indices = outcome
            .edges
            .iter()
            .flat_map(|(name, rows)| {
                rows.iter()
                    .map(move |(&id, &index)| (id, name.clone(), index))
            })
            .collect();
        self.node_indices.sort_unstable();
        self.edge_indices.sort_unstable();
        Ok(())
    }
    pub(crate) fn replace_compacted_columns(
        &mut self,
        nodes: &crate::storage::pool::compaction::CompactedColumns,
        edges: &crate::storage::pool::compaction::CompactedColumns,
    ) -> GraphResult<()> {
        let remap = |indices: &[(usize, AttrName, usize)],
                     columns: &crate::storage::pool::compaction::CompactedColumns|
         -> GraphResult<Vec<(usize, AttrName, usize)>> {
            indices
                .iter()
                .map(|(id, name, index)| Ok((*id, name.clone(), columns.offset(name, *index)?)))
                .collect()
        };
        let node_indices = remap(&self.node_indices, nodes)?;
        let edge_indices = remap(&self.edge_indices, edges)?;
        self.pool
            .replace_compacted_columns(&nodes.columns, &edges.columns)?;
        self.node_indices = node_indices;
        self.edge_indices = edge_indices;
        Ok(())
    }

    pub(crate) fn restore(self) -> GraphResult<Graph> {
        if !(VERSION..=4).contains(&self.version) {
            return Err(invalid("unsupported document version"));
        }
        self.config.validate().map_err(invalid)?;
        let pool = self.pool.restore(self.config.graph_type)?;
        if self.version < pool.archive_version() {
            return Err(invalid("storage layout requires a newer document version"));
        }
        let history = self.history.restore(&pool)?;
        if self.version < 4
            && history
                .get_commit_history()
                .iter()
                .any(|c| !c.delta.edge_endpoint_changes.is_empty())
        {
            return Err(invalid("endpoint revisions require document version 4"));
        }
        if self.state != 0 {
            history.get_commit(self.state)?;
        }
        // Detached selections are valid: a branch may retain a later head.
        history.get_branch_head(&self.branch)?;
        for (name, entity) in &self.identities.bindings {
            let bound = match entity {
                ExternalEntityRef::Node(id) => pool.is_node_allocated(*id),
                ExternalEntityRef::Edge(id) => pool.is_edge_allocated(*id),
            };
            if name.is_empty() || !bound {
                return Err(invalid("unallocated external identity reservation"));
            }
        }
        let mut graph = Graph::with_config_manual(self.config)?;
        graph.pool = Rc::new(RefCell::new(pool));
        graph.space = GraphSpace::new(graph.pool.clone(), self.state);
        graph.history = history;
        graph.identities = self.identities;
        graph.current_branch = self.branch;
        graph.current_commit = self.state;
        graph.automatic_publication = self.automatic_publication;
        for node in self.nodes {
            if !graph.pool.borrow().is_node_allocated(node) || graph.space.is_node_active(node) {
                return Err(invalid("duplicate or unallocated active node"));
            }
            graph.space.activate_node(node);
        }
        for edge in self.edges {
            if graph.space.is_edge_active(edge) {
                return Err(invalid("duplicate active edge"));
            }
            let (u, v) = graph
                .pool
                .borrow()
                .get_edge_endpoints(edge)
                .ok_or_else(|| invalid("unallocated selected edge"))?;
            if !graph.has_node(u) || !graph.has_node(v) {
                return Err(invalid("inactive selected edge endpoint"));
            }
            graph.space.activate_edge(edge, u, v);
        }
        for (is_node, indices) in [(true, self.node_indices), (false, self.edge_indices)] {
            let mut seen = HashSet::new();
            for (id, name, index) in indices {
                if !seen.insert((id, name.clone()))
                    || (is_node && !graph.has_node(id))
                    || (!is_node && !graph.has_edge(id))
                {
                    return Err(invalid("duplicate or inactive selected property target"));
                }
                let pool = graph.pool.borrow();
                let column = if is_node {
                    pool.get_node_attribute_column(&name)
                } else {
                    pool.get_edge_attribute_column(&name)
                };
                if column.is_none_or(|column| index >= column.len()) {
                    return Err(invalid("selected property index out of bounds"));
                }
                drop(pool);
                if is_node {
                    graph.space.set_node_attr_index(id, name, index);
                } else {
                    graph.space.set_edge_attr_index(id, name, index);
                }
            }
        }
        for commit in graph.history.get_commit_history() {
            for (is_node, changes) in [
                (true, &commit.delta.node_attr_changes),
                (false, &commit.delta.edge_attr_changes),
            ] {
                for (id, name, old, new) in changes {
                    if name != EXTERNAL_ID {
                        continue;
                    }
                    let entity = if is_node {
                        ExternalEntityRef::Node(*id)
                    } else {
                        ExternalEntityRef::Edge(*id)
                    };
                    for value in old.iter().chain(new.iter()) {
                        let name = value
                            .get_text()
                            .ok_or_else(|| invalid("nontext historical external identity"))?;
                        if graph.identities.bindings.get(&name) != Some(&entity) {
                            return Err(invalid("missing historical identity reservation"));
                        }
                    }
                }
            }
        }
        let recorded = graph.history.reconstruct_state_at(graph.current_commit)?;
        if !same_selected_snapshot(&graph, &recorded)? {
            return Err(invalid(
                "selected columns/topology differ from accepted version",
            ));
        }
        let bindings = graph.snapshot_identity_bindings(&recorded)?;
        for (name, entity) in bindings {
            if graph.identities.bindings.get(&name) != Some(&entity) {
                return Err(invalid("missing active identity reservation"));
            }
        }
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn document() -> GraphArchive {
        let mut graph = Graph::new_publishing();
        graph.try_add_node().unwrap();
        graph
            .writer("test")
            .set_node_external_id(0, "name".into())
            .unwrap();
        graph.archive_document().unwrap()
    }
    #[test]
    fn borrowed_snapshot_validation_retains_bit_tags_absence_and_empty_row_rules() {
        use crate::state::GraphSnapshot;
        let values = vec![
            AttrValue::Text("large\0α".repeat(1024)),
            AttrValue::Null,
            AttrValue::Float(-0.0),
            AttrValue::Float(f32::from_bits(0x7fc0_0001)),
            AttrValue::Float64(-0.0),
            AttrValue::Float64(f64::from_bits(0x7ff8_0000_0000_0001)),
            AttrValue::FloatVec(vec![-0.0, f32::from_bits(0x7fc0_0001)]),
            AttrValue::Float64Vec(vec![-0.0, f64::from_bits(0x7ff8_0000_0000_0001)]),
            AttrValue::Json(r#"{ "x": 1 }"#.into()),
            AttrValue::Bytes(vec![0, 255]),
            AttrValue::Int(1),
            AttrValue::SmallInt(1),
            AttrValue::SubgraphRef(42),
        ];
        for left in &values {
            for right in &values {
                let mut a = GraphSnapshot::empty(1);
                a.active_nodes = vec![0];
                a.node_attributes
                    .insert(0, HashMap::from([("value".into(), left.clone())]));
                let mut b = a.clone();
                b.node_attributes
                    .get_mut(&0)
                    .unwrap()
                    .insert("value".into(), right.clone());
                let expected =
                    retained_group::encode(left).unwrap() == retained_group::encode(right).unwrap();
                assert_eq!(same_snapshot(&a, &b).unwrap(), expected);
            }
        }
        let mut a = GraphSnapshot::empty(1);
        a.active_nodes = vec![0];
        let mut b = a.clone();
        b.node_attributes.insert(0, HashMap::new());
        assert!(same_snapshot(&a, &b).unwrap());
        b.node_attributes
            .get_mut(&0)
            .unwrap()
            .insert("value".into(), AttrValue::Null);
        assert!(!same_snapshot(&a, &b).unwrap());
        a.node_attributes.insert(
            0,
            HashMap::from([("value".into(), AttrValue::Float64(-0.0))]),
        );
        b.node_attributes
            .get_mut(&0)
            .unwrap()
            .insert("value".into(), AttrValue::Float64(0.0));
        assert!(!same_snapshot(&a, &b).unwrap());
    }
    #[test]
    fn direct_selected_validation_matches_snapshot_oracle_and_rejects_corruption() {
        let mut graph = Graph::new();
        graph.add_nodes(2).unwrap();
        graph.add_edge(0, 1).unwrap();
        graph
            .set_node_attr(0, "value".into(), AttrValue::Float64(-0.0))
            .unwrap();
        graph
            .set_edge_attr(
                0,
                "value".into(),
                AttrValue::FloatVec(vec![f32::from_bits(0x7fc0_0001)]),
            )
            .unwrap();
        let selected = graph
            .space
            .capture_read_snapshot(&graph.pool.borrow(), graph.current_commit)
            .unwrap();
        assert!(same_selected_snapshot(&graph, &selected).unwrap());
        let mut cases = Vec::new();
        let mut bad = selected.clone();
        bad.node_attributes
            .get_mut(&0)
            .unwrap()
            .insert("value".into(), AttrValue::Float64(0.0));
        cases.push(bad);
        let mut bad = selected.clone();
        bad.edge_attributes.get_mut(&0).unwrap().remove("value");
        cases.push(bad);
        let mut bad = selected.clone();
        bad.node_attributes
            .entry(1)
            .or_default()
            .insert("extra".into(), AttrValue::Null);
        cases.push(bad);
        let mut bad = selected.clone();
        bad.edges.insert(0, (1, 0));
        cases.push(bad);
        let mut bad = selected.clone();
        bad.active_nodes.pop();
        cases.push(bad);
        let mut bad = selected.clone();
        bad.state_id += 1;
        cases.push(bad);
        let mut same = selected.clone();
        same.node_attributes.entry(1).or_default();
        cases.push(same);
        for snapshot in cases {
            assert_eq!(
                same_selected_snapshot(&graph, &snapshot).unwrap(),
                same_snapshot(&selected, &snapshot).unwrap()
            );
        }
        graph
            .space
            .set_node_attr_index(0, "value".into(), usize::MAX);
        assert!(same_selected_snapshot(&graph, &selected).is_err());
    }
    #[test]
    fn semantic_archive_validation_rejects_bad_version_indices_ids_and_reservations() {
        let mut d = document();
        d.version = 5;
        assert!(d.restore().is_err());
        let mut d = document();
        d.node_indices.push(d.node_indices[0].clone());
        assert!(d.restore().is_err());
        let mut d = document();
        d.node_indices[0].2 = usize::MAX;
        assert!(d.restore().is_err());
        let mut d = document();
        d.nodes.push(99);
        assert!(d.restore().is_err());
        let mut d = document();
        d.identities.bindings.clear();
        assert!(d.restore().is_err());
        let mut d = document();
        d.identities
            .bindings
            .insert("bad".into(), ExternalEntityRef::Edge(99));
        assert!(d.restore().is_err());
        let mut d = document();
        d.branch = "missing".into();
        assert!(d.restore().is_err());
    }
    #[test]
    fn detached_prepared_archive_freezes_bytes_without_retaining_source_owner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frozen");
        let mut graph = Graph::new_publishing();
        graph.try_add_node().unwrap();
        let state = graph.current_state_id();
        let prepared = graph.prepare_archive().unwrap();
        graph.try_add_node().unwrap();
        drop(graph);
        std::thread::spawn(move || prepared.save(&path).unwrap())
            .join()
            .unwrap();
        let graph = Graph::load_archive(&dir.path().join("frozen")).unwrap();
        assert_eq!(graph.current_state_id(), state);
        assert_eq!(graph.node_count(), 1);
    }
    #[test]
    fn archive_preserves_reserved_sparse_allocation_without_allocating_fake_edge_slots() {
        let graph = Graph::new_publishing();
        graph.pool.borrow_mut().preserve_allocation(100, 200);
        let mut copy = graph.prepare_archive().unwrap().restore().unwrap();
        assert_eq!(copy.try_add_nodes(2).unwrap(), vec![100, 101]);
        assert_eq!(
            copy.writer("test")
                .add_edge(100, 101, HashMap::new())
                .unwrap(),
            200
        );
    }
    #[test]
    fn archive_preserves_resolved_merge_parents_and_original_publication_clocks() {
        let mut graph = Graph::new_publishing();
        graph
            .apply_batch(
                GraphBatch {
                    add_nodes: 1,
                    effective_at: Some(10),
                    ..Default::default()
                },
                0,
                "root".into(),
                "test".into(),
            )
            .unwrap();
        graph.create_branch("right".into()).unwrap();
        graph
            .writer("test")
            .with_effective_at(Some(20))
            .set_node_attr(0, "v".into(), AttrValue::Int(1))
            .unwrap();
        let left = graph.current_state_id();
        graph.checkout_branch("right".into()).unwrap();
        graph
            .writer("test")
            .with_effective_at(Some(25))
            .set_node_attr(0, "v".into(), AttrValue::Int(2))
            .unwrap();
        let right = graph.current_state_id();
        let metadata = crate::PublicationMetadata {
            recorded_at: graph
                .publication_metadata(right)
                .unwrap()
                .unwrap()
                .recorded_at
                + 1000,
            effective_at: 30,
            change_times: Default::default(),
        };
        let mut changes = crate::state::change_tracker::ChangeSet::new();
        changes.node_attr_changes.push((
            0,
            "v".into(),
            Some(AttrValue::Int(1)),
            Some(AttrValue::Int(3)),
        ));
        let merge = graph
            .history
            .create_merge_commit_with_publication(
                changes,
                "resolved".into(),
                "test".into(),
                vec![left, right],
                Some(metadata),
            )
            .unwrap();
        graph
            .history
            .update_branch_head(&"right".into(), merge)
            .unwrap();
        graph.current_commit = merge;
        graph
            .reset_to_snapshot(graph.history.reconstruct_state_at(merge).unwrap())
            .unwrap();
        let clocks = graph.publication_metadata(merge).unwrap();
        let copy = graph.prepare_archive().unwrap().restore().unwrap();
        assert_eq!(copy.history.get_parents(merge).unwrap(), vec![left, right]);
        assert_eq!(copy.publication_metadata(merge).unwrap(), clocks);
        assert_eq!(
            copy.read_view_at(30, None)
                .unwrap()
                .node_attr(0, &"v".into())
                .unwrap(),
            Some(AttrValue::Int(3))
        );
    }
    #[test]
    fn raw_escape_corruption_cannot_publish_an_unreadable_archive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid");
        let mut graph = Graph::new_publishing();
        graph.try_add_node().unwrap();
        graph.space_mut().activate_node(100);
        assert!(graph.save_archive(&path).is_err());
        assert!(!path.exists());
    }
    #[test]
    fn archive_version_migration_accepts_legacy_and_rejects_purged_referents_and_downgrades() {
        let original = document();
        let encoded = serde_json::to_value(&original.pool).unwrap();
        assert_eq!(original.version, 1);
        assert!(encoded.get("purged_nodes").is_none());
        original.restore().unwrap();
        for version in [1, 2] {
            let mut d = document();
            d.version = version;
            d.pool.set_test_purged_nodes(vec![(0, 1)]);
            assert!(d.restore().is_err());
        }
        let mut graph = Graph::new();
        graph.pool.borrow_mut().preserve_allocation(0, 100);
        graph.purge_unreferenced().unwrap();
        for version in [1, 2] {
            let mut d = graph.archive_document().unwrap();
            assert_eq!(d.version, 3);
            d.version = version;
            assert!(d.restore().is_err());
        }
        let mut d = document();
        d.version = 2;
        d.pool.set_test_purged_nodes(vec![(0, 1), (0, 1)]);
        assert!(d.restore().is_err());
    }
    #[test]
    fn endpoint_revision_archives_require_the_new_document_version() {
        let mut graph = Graph::new();
        graph.add_nodes(3).unwrap();
        graph.add_edge(0, 1).unwrap();
        graph.set_edge_endpoints(0, 2, 1).unwrap();
        let document = graph.archive_document().unwrap();
        assert_eq!(document.version, 4);
        for version in 1..4 {
            let mut invalid = graph.archive_document().unwrap();
            invalid.version = version;
            assert!(invalid.restore().is_err());
        }
        document.restore().unwrap().check_integrity().unwrap();
    }
}
