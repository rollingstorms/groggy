//! Binary, checksummed, append-only transaction storage for checked publications.
//! One locked writer owns the file. Successful writes sync the journal before
//! publishing memory; recovery replays only complete, validated records.
use super::retained_group as value_codec;
use crate::{
    BatchChangeRef, BatchEdgeRef, BatchNodeRef, BatchResult, Graph, GraphBatch, GraphConfig,
    GraphError, GraphResult, GraphSchema, PublicationMetadata, StateId, TimestampMicros,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"GROGGYJ1";
const FRAME: &[u8; 8] = b"GFRAME01";
const HEADER_SIZE: usize = 32;
const MAX_RECORD: usize = 256 * 1024 * 1024;

fn validate_group_refs<'a>(
    graph: &Graph,
    values: impl Iterator<Item = &'a crate::AttrValue>,
) -> GraphResult<()> {
    let groups = graph.pool().retained_groups();
    for value in values {
        if let crate::AttrValue::SubgraphRef(id) = value {
            if !groups.contains_key(id) {
                return Err(invalid(format!(
                    "SubgraphRef {id} requires a persisted frozen group"
                )));
            }
        }
    }
    Ok(())
}
fn install_meta_kinds(
    prepared: &mut crate::api::batch::PreparedPublication,
    first: usize,
    positions: &[usize],
) -> GraphResult<()> {
    let mut seen = std::collections::HashSet::new();
    for &position in positions {
        if position >= prepared.batch.add_nodes || !seen.insert(position) {
            return Err(invalid("Invalid stored meta-node position"));
        }
        let id = first + position;
        if !prepared
            .batch
            .node_attrs
            .get("contains_subgraph")
            .is_some_and(|rows| {
                rows.iter().any(|(node, value)| {
                    *node == id && matches!(value, crate::AttrValue::SubgraphRef(_))
                })
            })
        {
            return Err(invalid("Stored meta-node lacks a group reference"));
        }
    }
    if !positions.is_empty() {
        prepared.batch.node_attrs.insert(
            "entity_type".into(),
            positions
                .iter()
                .map(|&position| (first + position, crate::AttrValue::Text("meta".into())))
                .collect(),
        );
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredBatch {
    add_nodes: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    meta_nodes: Vec<usize>,
    add_edges: Vec<(BatchNodeRef, BatchNodeRef)>,
    node_attrs: BTreeMap<String, Vec<(BatchNodeRef, Value)>>,
    edge_attrs: BTreeMap<String, Vec<(BatchEdgeRef, Value)>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    unset_node_attrs: BTreeMap<String, Vec<BatchNodeRef>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    unset_edge_attrs: BTreeMap<String, Vec<BatchEdgeRef>>,
    remove_nodes: Vec<BatchNodeRef>,
    remove_edges: Vec<BatchEdgeRef>,
    node_external_ids: Vec<(BatchNodeRef, String)>,
    edge_external_ids: Vec<(BatchEdgeRef, String)>,
    effective_at: Option<TimestampMicros>,
    change_times: Vec<(BatchChangeRef, TimestampMicros)>,
}
impl StoredBatch {
    fn encode(batch: &GraphBatch) -> GraphResult<Self> {
        Ok(Self {
            add_nodes: batch.add_nodes,
            meta_nodes: Vec::new(),
            add_edges: batch.add_edges.clone(),
            node_attrs: batch
                .node_attrs
                .iter()
                .map(|(name, values)| {
                    Ok((
                        name.clone(),
                        values
                            .iter()
                            .map(|(id, value)| Ok((*id, value_codec::encode(value)?)))
                            .collect::<GraphResult<_>>()?,
                    ))
                })
                .collect::<GraphResult<_>>()?,
            edge_attrs: batch
                .edge_attrs
                .iter()
                .map(|(name, values)| {
                    Ok((
                        name.clone(),
                        values
                            .iter()
                            .map(|(id, value)| Ok((*id, value_codec::encode(value)?)))
                            .collect::<GraphResult<_>>()?,
                    ))
                })
                .collect::<GraphResult<_>>()?,
            unset_node_attrs: batch.unset_node_attrs.clone().into_iter().collect(),
            unset_edge_attrs: batch.unset_edge_attrs.clone().into_iter().collect(),
            remove_nodes: batch.remove_nodes.clone(),
            remove_edges: batch.remove_edges.clone(),
            node_external_ids: batch.node_external_ids.clone(),
            edge_external_ids: batch.edge_external_ids.clone(),
            effective_at: batch.effective_at,
            change_times: batch.change_times.clone(),
        })
    }
    fn decode(self) -> GraphResult<GraphBatch> {
        Ok(GraphBatch {
            add_nodes: self.add_nodes,
            add_edges: self.add_edges,
            node_attrs: self
                .node_attrs
                .into_iter()
                .map(|(name, values)| {
                    Ok((
                        name,
                        values
                            .into_iter()
                            .map(|(id, value)| Ok((id, value_codec::decode(value)?)))
                            .collect::<GraphResult<_>>()?,
                    ))
                })
                .collect::<GraphResult<_>>()?,
            edge_attrs: self
                .edge_attrs
                .into_iter()
                .map(|(name, values)| {
                    Ok((
                        name,
                        values
                            .into_iter()
                            .map(|(id, value)| Ok((id, value_codec::decode(value)?)))
                            .collect::<GraphResult<_>>()?,
                    ))
                })
                .collect::<GraphResult<_>>()?,
            unset_node_attrs: self.unset_node_attrs.into_iter().collect(),
            unset_edge_attrs: self.unset_edge_attrs.into_iter().collect(),
            remove_nodes: self.remove_nodes,
            remove_edges: self.remove_edges,
            node_external_ids: self.node_external_ids,
            edge_external_ids: self.edge_external_ids,
            effective_at: self.effective_at,
            change_times: self.change_times,
        })
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum Record {
    Init {
        config: GraphConfig,
        schema: Option<GraphSchema>,
    },
    Batch {
        expected_state: StateId,
        batch: Box<StoredBatch>,
        metadata: PublicationMetadata,
        message: String,
        author: String,
        additional_schema: Option<GraphSchema>,
    },
    RetainGroups {
        groups: Vec<(
            crate::types::SubgraphId,
            super::retained_group::GroupDocument,
        )>,
    },
    CreateBranch {
        name: String,
    },
    CheckoutBranch {
        name: String,
    },
    Schema {
        schema: Option<GraphSchema>,
        revision: u64,
    },
    Checkpoint {
        record: crate::Checkpoint,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    sequence: u64,
    record: Record,
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq)]
enum Failure {
    BeforeHeader,
    AfterHeader,
    AfterPayload,
    AfterSync,
    Rollback,
}

/// A single-writer durable graph. Normal mutation is available only through
/// validated publications; no mutable Graph handle is exposed.
/// The binary journal retains all accepted batches, branches and schema policy.
pub struct DurableGraph {
    graph: Graph,
    file: File,
    path: PathBuf,
    schema: Option<GraphSchema>,
    schema_revision: u64,
    next_sequence: u64,
    poisoned: bool,
    recovered_tail_bytes: u64,
    #[cfg(test)]
    failure: Option<Failure>,
}
fn invalid(message: impl std::fmt::Display) -> GraphError {
    GraphError::InvalidInput(format!("Invalid graph journal: {message}"))
}
fn io_error(path: &Path, operation: &str, error: impl std::fmt::Display) -> GraphError {
    GraphError::IoError {
        operation: operation.into(),
        path: path.to_string_lossy().into(),
        underlying_error: error.to_string(),
    }
}
fn lock(file: &File, path: &Path) -> GraphResult<()> {
    file.try_lock()
        .map_err(|e| io_error(path, "lock_graph_journal", e))
}
fn encode(entry: &Entry) -> GraphResult<Vec<u8>> {
    let mut payload = Vec::new();
    ciborium::ser::into_writer(entry, &mut payload).map_err(invalid)?;
    if payload.len() > MAX_RECORD {
        return Err(invalid("record exceeds 256 MiB limit"));
    }
    let length = payload.len() as u64;
    let mut frame = Vec::with_capacity(HEADER_SIZE + payload.len());
    frame.extend_from_slice(FRAME);
    frame.extend_from_slice(&length.to_le_bytes());
    frame.extend_from_slice(&(!length).to_le_bytes());
    frame.extend_from_slice(&xxhash_rust::xxh3::xxh3_64(&payload).to_le_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}
impl DurableGraph {
    /// Create a new binary journal. Existing files are never overwritten.
    pub fn create(
        path: &Path,
        config: GraphConfig,
        schema: Option<GraphSchema>,
    ) -> GraphResult<Self> {
        config.validate().map_err(invalid)?;
        let graph = Graph::with_config(config.clone());
        if let Some(schema) = &schema {
            graph.check_schema_policy(schema)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| io_error(path, "create_graph_journal", e))?;
        lock(&file, path)?;
        let mut store = Self {
            graph,
            file,
            path: path.into(),
            schema: schema.clone(),
            schema_revision: 0,
            next_sequence: 0,
            poisoned: false,
            recovered_tail_bytes: 0,
            #[cfg(test)]
            failure: None,
        };
        store
            .file
            .write_all(MAGIC)
            .map_err(|e| io_error(path, "write_journal_header", e))?;
        store.append(Record::Init { config, schema })?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        File::open(parent)
            .and_then(|dir| dir.sync_all())
            .map_err(|e| io_error(parent, "sync_journal_directory", e))?;
        Ok(store)
    }
    /// Reopen under an exclusive OS file lock. A partial final frame is discarded;
    /// complete checksum failures, bad versions and semantic corruption are errors.
    pub fn open(path: &Path) -> GraphResult<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| io_error(path, "open_graph_journal", e))?;
        lock(&file, path)?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|e| io_error(path, "read_journal_header", e))?;
        if &magic != MAGIC {
            return Err(invalid("unsupported format/version"));
        }
        let length = file
            .metadata()
            .map_err(|e| io_error(path, "stat_journal", e))?
            .len();
        let mut position = 8u64;
        let mut sequence = 0u64;
        let mut graph = None;
        let mut schema = None;
        let mut schema_revision = 0;
        while position < length {
            let remaining = length - position;
            if remaining < HEADER_SIZE as u64 {
                let mut tail = vec![0; remaining as usize];
                file.read_exact(&mut tail)
                    .map_err(|e| io_error(path, "read_journal_tail", e))?;
                let prefix = tail.len().min(FRAME.len());
                if tail[..prefix] != FRAME[..prefix] {
                    return Err(invalid("corrupt partial frame marker"));
                }
                break;
            }
            let mut header = [0u8; HEADER_SIZE];
            file.read_exact(&mut header)
                .map_err(|e| io_error(path, "read_frame_header", e))?;
            if &header[..8] != FRAME {
                return Err(invalid("bad frame marker"));
            }
            let number = |start| {
                let mut bytes = [0; 8];
                bytes.copy_from_slice(&header[start..start + 8]);
                u64::from_le_bytes(bytes)
            };
            let count = number(8);
            if number(16) != !count {
                return Err(invalid("frame length checksum mismatch"));
            }
            if count > MAX_RECORD as u64 {
                return Err(invalid("record exceeds 256 MiB limit"));
            }
            if count > remaining - HEADER_SIZE as u64 {
                break;
            }
            let mut payload = vec![0; count as usize];
            file.read_exact(&mut payload)
                .map_err(|e| io_error(path, "read_frame_payload", e))?;
            if xxhash_rust::xxh3::xxh3_64(&payload) != number(24) {
                return Err(invalid("frame checksum mismatch"));
            }
            let mut cursor = Cursor::new(&payload);
            let entry: Entry = ciborium::de::from_reader(&mut cursor).map_err(invalid)?;
            if cursor.position() != count {
                return Err(invalid("trailing bytes in frame"));
            }
            if entry.sequence != sequence {
                return Err(invalid("noncontiguous record sequence"));
            }
            match entry.record {
                Record::Init {
                    config,
                    schema: policy,
                } if graph.is_none() && sequence == 0 => {
                    config.validate().map_err(invalid)?;
                    let candidate = Graph::with_config(config);
                    if let Some(schema) = &policy {
                        candidate.check_schema_policy(schema)?;
                    }
                    graph = Some(candidate);
                    schema = policy;
                }
                Record::Init { .. } => return Err(invalid("duplicate initialization")),
                record => {
                    let graph = graph
                        .as_mut()
                        .ok_or_else(|| invalid("missing initialization"))?;
                    Self::replay(graph, &mut schema, &mut schema_revision, record)?;
                }
            }
            sequence = sequence
                .checked_add(1)
                .ok_or_else(|| invalid("sequence exhausted"))?;
            position += HEADER_SIZE as u64 + count;
        }
        let graph = graph.ok_or_else(|| invalid("journal initialization is incomplete"))?;
        let recovered_tail_bytes = length - position;
        if recovered_tail_bytes > 0 {
            file.set_len(position)
                .and_then(|_| file.sync_all())
                .map_err(|e| io_error(path, "truncate_incomplete_journal_tail", e))?;
        }
        file.seek(SeekFrom::End(0))
            .map_err(|e| io_error(path, "seek_journal", e))?;
        Ok(Self {
            graph,
            file,
            path: path.into(),
            schema,
            schema_revision,
            next_sequence: sequence,
            poisoned: false,
            recovered_tail_bytes,
            #[cfg(test)]
            failure: None,
        })
    }
    fn ready(&self) -> GraphResult<()> {
        if self.poisoned {
            Err(invalid(
                "store requires reopen after an indeterminate publication failure",
            ))
        } else {
            Ok(())
        }
    }
    fn write_frame(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        #[cfg(test)]
        if self.failure == Some(Failure::BeforeHeader) {
            return Err(std::io::Error::other("injected before header"));
        }
        self.file.write_all(&bytes[..HEADER_SIZE])?;
        #[cfg(test)]
        if self.failure == Some(Failure::AfterHeader) {
            return Err(std::io::Error::other("injected after header"));
        }
        self.file.write_all(&bytes[HEADER_SIZE..])?;
        #[cfg(test)]
        if self.failure == Some(Failure::AfterPayload) {
            return Err(std::io::Error::other("injected after payload"));
        }
        self.file.sync_all()?;
        #[cfg(test)]
        if matches!(self.failure, Some(Failure::AfterSync | Failure::Rollback)) {
            return Err(std::io::Error::other("injected after sync"));
        }
        Ok(())
    }
    fn rollback_append(&mut self, offset: u64) -> std::io::Result<()> {
        #[cfg(test)]
        if self.failure == Some(Failure::Rollback) {
            return Err(std::io::Error::other("injected rollback failure"));
        }
        self.file.set_len(offset).and_then(|_| self.file.sync_all())
    }
    fn append(&mut self, record: Record) -> GraphResult<()> {
        self.ready()?;
        if self.next_sequence == u64::MAX {
            return Err(invalid("sequence exhausted"));
        }
        let bytes = encode(&Entry {
            sequence: self.next_sequence,
            record,
        })?;
        let start = self
            .file
            .seek(SeekFrom::End(0))
            .map_err(|e| io_error(&self.path, "seek_journal", e))?;
        if let Err(error) = self.write_frame(&bytes) {
            if let Err(rollback) = self.rollback_append(start) {
                self.poisoned = true;
                return Err(io_error(
                    &self.path,
                    "indeterminate_journal_write",
                    format!("{error}; rollback failed: {rollback}; reopen required"),
                ));
            }
            return Err(io_error(&self.path, "append_graph_journal", error));
        }
        self.next_sequence += 1;
        Ok(())
    }
    fn replay(
        graph: &mut Graph,
        schema: &mut Option<GraphSchema>,
        revision: &mut u64,
        record: Record,
    ) -> GraphResult<()> {
        match record {
            Record::Batch {
                expected_state,
                batch,
                metadata,
                message,
                author,
                additional_schema,
            } => {
                let meta_nodes = batch.meta_nodes.clone();
                let batch = batch.decode()?;
                validate_group_refs(
                    graph,
                    batch
                        .node_attrs
                        .values()
                        .flatten()
                        .map(|(_, v)| v)
                        .chain(batch.edge_attrs.values().flatten().map(|(_, v)| v)),
                )?;
                let first_node = graph.pool().statistics().node_count;
                let mut prepared = graph.prepare_publication_with_additional_schema_at(
                    batch,
                    expected_state,
                    schema.as_ref(),
                    additional_schema.as_ref(),
                    Some(metadata.recorded_at),
                )?;
                install_meta_kinds(&mut prepared, first_node, &meta_nodes)?;
                graph.restore_publication_clock(&mut prepared, metadata)?;
                graph.execute_publication(prepared, message, author)?;
            }
            Record::RetainGroups { groups } => {
                let prepared = graph.pool_mut().prepare_group_documents(groups)?;
                for group in prepared {
                    graph.pool_mut().install_stored_subgraph(group);
                }
            }
            Record::CreateBranch { name } => graph.create_branch(name)?,
            Record::CheckoutBranch { name } => {
                graph.checkout_branch(name)?;
                if let Some(schema) = schema {
                    graph.check_schema_policy(schema)?;
                }
            }
            Record::Schema {
                schema: policy,
                revision: next,
            } => {
                if revision.checked_add(1) != Some(next) {
                    return Err(invalid("invalid schema revision"));
                }
                if let Some(schema) = &policy {
                    graph.check_schema_policy(schema)?;
                }
                *schema = policy;
                *revision = next;
            }
            Record::Checkpoint { record } => graph.record_checkpoint(record)?,
            Record::Init { .. } => return Err(invalid("duplicate initialization")),
        }
        Ok(())
    }
    /// Copy all accepted history to a separately recoverable journal.
    /// The destination must not exist. A synced sibling temporary file is
    /// published without replacing existing files, then its directory is synced.
    /// A directory-sync error can leave a complete destination: reconcile it
    /// before retrying. The source graph and its journal remain unchanged.
    pub fn backup(&mut self, destination: &Path) -> GraphResult<()> {
        self.ready()?;
        self.graph
            .validate_write_head(self.graph.current_state_id())?;
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut temporary = tempfile::NamedTempFile::new_in(parent)
            .map_err(|e| io_error(destination, "create_backup_temporary", e))?;
        let mut source = self
            .file
            .try_clone()
            .map_err(|e| io_error(&self.path, "clone_journal_for_backup", e))?;
        source
            .seek(SeekFrom::Start(0))
            .map_err(|e| io_error(&self.path, "seek_backup_source", e))?;
        std::io::copy(&mut source, temporary.as_file_mut())
            .and_then(|_| temporary.as_file().sync_all())
            .map_err(|e| io_error(destination, "copy_and_sync_backup", e))?;
        // Verify the copied record stream before exposing its destination name.
        // This also rejects a corrupt source without poisoning or replacing it.
        let verified = Self::open(temporary.path())?;
        if verified.recovered_tail_bytes() != 0 {
            return Err(invalid("backup source has an incomplete journal tail"));
        }
        drop(verified);
        temporary
            .persist_noclobber(destination)
            .map_err(|e| io_error(destination, "publish_backup_without_overwrite", e.error))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|e| io_error(parent, "sync_backup_directory", e))?;
        Ok(())
    }

    /// Read through a checked immutable handle; failed durable execution requires reopening.
    pub fn graph(&self) -> GraphResult<&Graph> {
        self.ready()?;
        Ok(&self.graph)
    }
    /// Number of discarded bytes from a partial final frame on the last open.
    pub fn recovered_tail_bytes(&self) -> u64 {
        self.recovered_tail_bytes
    }
    /// Installed policy revision, persisted independently of data version IDs.
    pub fn schema_revision(&self) -> u64 {
        self.schema_revision
    }
    /// Current installed schema. Mutation requires set_schema and a durable journal record.
    pub fn schema(&self) -> Option<&GraphSchema> {
        self.schema.as_ref()
    }
    /// Persist and sync an accepted transaction before publishing its in-memory state.
    pub fn apply_batch(
        &mut self,
        batch: GraphBatch,
        expected_state: StateId,
        message: String,
        author: String,
    ) -> GraphResult<BatchResult> {
        self.apply_batch_with_schema(batch, expected_state, message, author, None)
    }

    /// An optional stricter per-call policy is checked in addition to the installed policy.
    pub fn apply_batch_with_schema(
        &mut self,
        batch: GraphBatch,
        expected_state: StateId,
        message: String,
        author: String,
        additional_schema: Option<&GraphSchema>,
    ) -> GraphResult<BatchResult> {
        self.ready()?;
        let stored = StoredBatch::encode(&batch)?;
        let prepared = self.graph.prepare_publication_with_additional_schema(
            batch,
            expected_state,
            self.schema.as_ref(),
            additional_schema,
        )?;
        self.execute_prepared_batch(
            prepared,
            stored,
            expected_state,
            message,
            author,
            additional_schema,
        )
    }

    /// Persist an explicit immutable source group before it is referenced by a batch.
    pub fn retain_group(
        &mut self,
        source: &crate::GraphView,
        kind: String,
    ) -> GraphResult<crate::types::SubgraphId> {
        let root = super::retained_group::RetainedGroup::new(source.clone(), kind)?;
        self.retain_group_revision(root)
    }
    /// Persist a revision whose immutable payload was prepared off the owner thread.
    pub fn retain_group_revision(
        &mut self,
        root: super::retained_group::RetainedGroup,
    ) -> GraphResult<crate::types::SubgraphId> {
        self.ready()?;
        let id = root.id();
        if let Some(old) = self.graph.pool().retained_groups().get(&id) {
            if old.same_content(&root)? {
                return Ok(id);
            }
            return Err(invalid("Frozen group value collision"));
        }
        let groups = super::retained_group::documents(&root)?;
        let prepared = self
            .graph
            .pool_mut()
            .prepare_group_documents(groups.clone())?;
        self.append(Record::RetainGroups { groups })?;
        for group in prepared {
            self.graph.pool_mut().install_stored_subgraph(group);
        }
        Ok(id)
    }

    /// Publish a meta-node referencing an already persisted immutable group.
    pub fn create_meta_node(
        &mut self,
        group: crate::types::SubgraphId,
        expected_state: StateId,
        effective_at: Option<i64>,
    ) -> GraphResult<crate::NodeId> {
        self.ready()?;
        if !self.graph.pool().retained_groups().contains_key(&group) {
            return Err(invalid("Meta-node requires a persisted frozen group"));
        }
        let batch = GraphBatch {
            add_nodes: 1,
            effective_at,
            node_attrs: std::collections::HashMap::from([(
                "contains_subgraph".into(),
                vec![(BatchNodeRef::Added(0), crate::AttrValue::SubgraphRef(group))],
            )]),
            ..Default::default()
        };
        let mut stored = StoredBatch::encode(&batch)?;
        stored.meta_nodes = vec![0];
        let first = self.graph.pool().statistics().node_count;
        let mut prepared =
            self.graph
                .prepare_publication(batch, expected_state, self.schema.as_ref())?;
        install_meta_kinds(&mut prepared, first, &[0])?;
        let result = self.execute_prepared_batch(
            prepared,
            stored,
            expected_state,
            "create_meta_node".into(),
            "groggy".into(),
            None,
        )?;
        Ok(result.added_nodes[0])
    }

    /// Capture the current durable head as an immutable selection with write provenance.
    pub fn read_view(&self) -> GraphResult<crate::GraphView> {
        self.graph()?.read_view()
    }

    /// Capture immutable publication ancestry for detached effective-time replay.
    pub fn prepare_read_at(
        &self,
        effective_at: i64,
        recorded_by: Option<i64>,
    ) -> GraphResult<crate::state::effective_read::EffectiveReadPlan> {
        self.graph()?.prepare_read_at(effective_at, recorded_by)
    }

    /// Prepare using explicit legacy source inputs, without changing stored history.
    pub fn prepare_effective_read(
        &self,
        effective_at: i64,
        options: crate::state::effective_read::EffectiveReadOptions,
    ) -> GraphResult<crate::state::effective_read::EffectiveReadPlan> {
        self.graph()?.prepare_effective_read(effective_at, options)
    }
    /// Prepare an effective read with explicit caller-assigned legacy clocks.
    pub fn prepare_read_at_with_legacy(
        &self,
        effective_at: i64,
        recorded_by: Option<i64>,
        legacy: std::collections::BTreeMap<StateId, crate::PublicationMetadata>,
    ) -> GraphResult<crate::state::effective_read::EffectiveReadPlan> {
        self.graph()?
            .prepare_read_at_with_legacy(effective_at, recorded_by, legacy)
    }
    /// Reconstruct a read-only effective-time state from the accepted durable head.
    pub fn read_view_at(
        &self,
        effective_at: i64,
        recorded_by: Option<i64>,
    ) -> GraphResult<crate::GraphView> {
        self.prepare_read_at(effective_at, recorded_by)?.read_view()
    }

    /// Sync and publish one batch scoped to a current immutable view, returning a new view.
    pub fn apply_view_batch(
        &mut self,
        view: &crate::GraphView,
        batch: GraphBatch,
        message: String,
        author: String,
    ) -> GraphResult<crate::ViewBatchResult> {
        self.apply_view_batch_with_schema(view, batch, message, author, None)
    }

    /// Check both installed durable schema and optional stricter per-call policy.
    pub fn apply_view_batch_with_schema(
        &mut self,
        view: &crate::GraphView,
        batch: GraphBatch,
        message: String,
        author: String,
        additional_schema: Option<&GraphSchema>,
    ) -> GraphResult<crate::ViewBatchResult> {
        self.ready()?;
        let stored = StoredBatch::encode(&batch)?;
        let prepared = self.graph.prepare_view_publication(
            view,
            batch,
            self.schema.as_ref(),
            additional_schema,
        )?;
        let batch = self.execute_prepared_batch(
            prepared.publication,
            stored,
            prepared.expected_state,
            message,
            author,
            additional_schema,
        )?;
        Ok(crate::ViewBatchResult {
            batch,
            view: prepared.view,
        })
    }

    fn execute_prepared_batch(
        &mut self,
        prepared: crate::api::batch::PreparedPublication,
        stored: StoredBatch,
        expected_state: StateId,
        message: String,
        author: String,
        additional_schema: Option<&GraphSchema>,
    ) -> GraphResult<BatchResult> {
        validate_group_refs(
            &self.graph,
            prepared
                .batch
                .node_attrs
                .values()
                .flatten()
                .map(|(_, v)| v)
                .chain(prepared.batch.edge_attrs.values().flatten().map(|(_, v)| v)),
        )?;
        self.append(Record::Batch {
            expected_state,
            batch: Box::new(stored),
            metadata: prepared.metadata.clone(),
            message: message.clone(),
            author: author.clone(),
            additional_schema: additional_schema.cloned(),
        })?;
        match self.graph.execute_publication(prepared, message, author) {
            Ok(result) => Ok(result),
            Err(error) => {
                self.poisoned = true;
                Err(io_error(
                    &self.path,
                    "durable_execution_requires_reopen",
                    error,
                ))
            }
        }
    }
    /// Managed node upserts with the same durable boundary; identical payloads are no-ops.
    pub fn upsert_nodes(
        &mut self,
        rows: Vec<crate::NodeUpsert>,
        expected_state: StateId,
        author: String,
        effective_at: Option<TimestampMicros>,
        additional_schema: Option<&GraphSchema>,
    ) -> GraphResult<(Vec<crate::NodeId>, StateId)> {
        self.ready()?;
        self.graph.validate_write_head(expected_state)?;
        let (mut batch, ids) = self.graph.published_node_upserts(rows)?;
        batch.effective_at = effective_at;
        self.publish_upsert(
            batch,
            expected_state,
            author,
            "upsert_nodes",
            additional_schema,
        )?;
        Ok((ids, self.graph.current_state_id()))
    }
    /// Managed edge upserts retain endpoint identity and do not publish identical payloads.
    pub fn upsert_edges(
        &mut self,
        rows: Vec<crate::EdgeUpsert>,
        expected_state: StateId,
        author: String,
        effective_at: Option<TimestampMicros>,
        additional_schema: Option<&GraphSchema>,
    ) -> GraphResult<(Vec<crate::EdgeId>, StateId)> {
        self.ready()?;
        self.graph.validate_write_head(expected_state)?;
        let (mut batch, ids) = self.graph.published_edge_upserts(rows)?;
        batch.effective_at = effective_at;
        self.publish_upsert(
            batch,
            expected_state,
            author,
            "upsert_edges",
            additional_schema,
        )?;
        Ok((ids, self.graph.current_state_id()))
    }
    fn publish_upsert(
        &mut self,
        batch: GraphBatch,
        expected: StateId,
        author: String,
        message: &str,
        additional_schema: Option<&GraphSchema>,
    ) -> GraphResult<()> {
        if batch.has_operations() {
            self.apply_batch_with_schema(
                batch,
                expected,
                message.into(),
                author,
                additional_schema,
            )?;
        } else {
            if let Some(schema) = &self.schema {
                self.graph.check_schema_policy(schema)?;
            }
            if let Some(schema) = additional_schema {
                self.graph.check_schema_policy(schema)?;
            }
        }
        Ok(())
    }
    /// Install/remove policy only after its current graph passes; revision is durable.
    pub fn set_schema(
        &mut self,
        schema: Option<GraphSchema>,
        expected_state: StateId,
        expected_revision: u64,
    ) -> GraphResult<u64> {
        self.ready()?;
        self.graph.validate_write_head(expected_state)?;
        if expected_revision != self.schema_revision {
            return Err(invalid("stale schema revision"));
        }
        if let Some(policy) = &schema {
            self.graph.check_schema_policy(policy)?;
        }
        let revision = self
            .schema_revision
            .checked_add(1)
            .ok_or_else(|| invalid("schema revision exhausted"))?;
        self.append(Record::Schema {
            schema: schema.clone(),
            revision,
        })?;
        self.schema = schema;
        self.schema_revision = revision;
        Ok(revision)
    }
    /// Sync an explicit checkpoint annotation before making it visible in memory.
    /// The checkpoint preserves the selected state ID and its original clocks.
    pub fn checkpoint(
        &mut self,
        expected_state: StateId,
        message: String,
        author: String,
    ) -> GraphResult<StateId> {
        self.checkpoint_with_schema(expected_state, message, author, None)
    }
    /// Check an additional checkpoint policy before writing the durable annotation.
    pub fn checkpoint_with_schema(
        &mut self,
        expected_state: StateId,
        message: String,
        author: String,
        additional_schema: Option<&GraphSchema>,
    ) -> GraphResult<StateId> {
        self.ready()?;
        self.graph.validate_write_head(expected_state)?;
        if let Some(schema) = additional_schema {
            self.graph.check_schema_policy(schema)?;
        }
        let record = self
            .graph
            .prepare_checkpoint(expected_state, message, author)?;
        self.append(Record::Checkpoint {
            record: record.clone(),
        })?;
        if let Err(error) = self.graph.record_checkpoint(record) {
            self.poisoned = true;
            return Err(error);
        }
        Ok(expected_state)
    }

    /// Create a durable branch from the selected clean head.
    pub fn create_branch(&mut self, name: String, expected_state: StateId) -> GraphResult<()> {
        self.ready()?;
        self.graph.validate_write_head(expected_state)?;
        if self
            .graph
            .list_branches()
            .iter()
            .any(|branch| branch.name == name)
        {
            return Err(invalid("branch already exists"));
        }
        self.append(Record::CreateBranch { name: name.clone() })?;
        if let Err(error) = self.graph.create_branch(name) {
            self.poisoned = true;
            return Err(error);
        }
        Ok(())
    }
    /// Switch branches under the installed policy, preserving lifetime allocations.
    pub fn checkout_branch(&mut self, name: String, expected_state: StateId) -> GraphResult<()> {
        self.ready()?;
        self.graph.validate_write_head(expected_state)?;
        let target = self
            .graph
            .list_branches()
            .into_iter()
            .find(|branch| branch.name == name)
            .ok_or_else(|| invalid("branch does not exist"))?;
        // Check the target before writing the journal. It is an immutable selected state.
        if let Some(schema) = &self.schema {
            self.graph.check_schema_at(schema, target.head)?;
        }
        self.append(Record::CheckoutBranch { name: name.clone() })?;
        if let Err(error) = self.graph.checkout_branch(name) {
            self.poisoned = true;
            return Err(error);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pending(store: &DurableGraph) -> (Record, PublicationMetadata) {
        let batch = GraphBatch {
            add_nodes: 1,
            ..Default::default()
        };
        let metadata = store
            .graph
            .prepare_publication(batch.clone(), 0, None)
            .unwrap()
            .metadata;
        (
            Record::Batch {
                expected_state: 0,
                batch: Box::new(StoredBatch::encode(&batch).unwrap()),
                metadata: metadata.clone(),
                message: "crash".into(),
                author: "test".into(),
                additional_schema: None,
            },
            metadata,
        )
    }
    #[test]
    fn checkpoint_write_failures_do_not_expose_metadata_or_consume_version_ids() {
        for failure in [
            Failure::BeforeHeader,
            Failure::AfterHeader,
            Failure::AfterPayload,
            Failure::AfterSync,
        ] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("checkpoints.ggj");
            let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
            let original = std::fs::read(&path).unwrap();
            store.failure = Some(failure);
            assert!(store
                .checkpoint(0, "rejected".into(), "test".into())
                .is_err());
            assert!(store.graph().unwrap().checkpoints().is_empty());
            assert_eq!(store.graph().unwrap().current_state_id(), 0);
            assert_eq!(std::fs::read(&path).unwrap(), original);
            store.failure = None;
            assert_eq!(
                store
                    .checkpoint(0, "accepted".into(), "test".into())
                    .unwrap(),
                0
            );
            let result = store
                .apply_batch(
                    GraphBatch {
                        add_nodes: 1,
                        ..Default::default()
                    },
                    0,
                    "node".into(),
                    "test".into(),
                )
                .unwrap();
            assert_eq!(result.state_id, 1);
        }
    }
    #[test]
    fn checkpoint_synced_before_memory_publication_recovers_original_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("checkpoints.ggj");
        let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
        let record = store
            .graph
            .prepare_checkpoint(0, "empty".into(), "test".into())
            .unwrap();
        store
            .append(Record::Checkpoint {
                record: record.clone(),
            })
            .unwrap();
        assert!(store.graph.checkpoints().is_empty());
        drop(store);
        let recovered = DurableGraph::open(&path).unwrap();
        assert_eq!(recovered.graph().unwrap().checkpoints(), &[record]);
        assert_eq!(recovered.graph().unwrap().current_state_id(), 0);
    }

    #[test]
    fn recovery_rejects_checkpoint_records_for_a_different_head_or_branch() {
        for wrong_branch in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("bad-checkpoint.ggj");
            let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
            let mut record = store
                .graph
                .prepare_checkpoint(0, "bad".into(), "test".into())
                .unwrap();
            if wrong_branch {
                record.branch = "other".into();
            } else {
                record.state_id = 10;
            }
            store.append(Record::Checkpoint { record }).unwrap();
            drop(store);
            assert!(DurableGraph::open(&path).is_err());
        }
    }

    #[test]
    fn synced_record_replays_after_crash_before_memory_publication() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("graph.ggj");
        let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
        let (record, metadata) = pending(&store);
        store.append(record).unwrap();
        assert_eq!(store.graph.node_count(), 0);
        drop(store);
        let recovered = DurableGraph::open(&path).unwrap();
        assert_eq!(recovered.graph().unwrap().node_count(), 1);
        assert_eq!(
            recovered
                .graph()
                .unwrap()
                .publication_metadata(1)
                .unwrap()
                .unwrap(),
            metadata
        );
    }
    #[test]
    fn partial_header_and_payload_boundaries_never_replay_partial_transaction() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("graph.ggj");
        let store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
        let (record, _) = pending(&store);
        let frame = encode(&Entry {
            sequence: 1,
            record,
        })
        .unwrap();
        drop(store);
        let baseline = std::fs::read(&path).unwrap();
        let cuts = (0..=HEADER_SIZE).chain([HEADER_SIZE + 1, frame.len() / 2, frame.len() - 1]);
        for cut in cuts {
            let mut bytes = baseline.clone();
            bytes.extend_from_slice(&frame[..cut]);
            std::fs::write(&path, bytes).unwrap();
            let recovered = DurableGraph::open(&path).unwrap();
            assert_eq!(recovered.graph().unwrap().node_count(), 0, "cut={cut}");
            assert_eq!(recovered.recovered_tail_bytes(), cut as u64);
            drop(recovered);
            assert_eq!(std::fs::read(&path).unwrap(), baseline);
        }
    }
    #[test]
    fn damaged_length_is_corruption_not_a_recoverable_tail() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("graph.ggj");
        let store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
        let (record, _) = pending(&store);
        let mut frame = encode(&Entry {
            sequence: 1,
            record,
        })
        .unwrap();
        drop(store);
        frame[8] ^= 1;
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(&frame);
        std::fs::write(&path, &bytes).unwrap();
        assert!(DurableGraph::open(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }
    #[test]
    fn failed_write_and_failed_rollback_poison_store_without_exposing_graph() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("graph.ggj");
        let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
        let before = std::fs::read(&path).unwrap();
        // Replace only the test instance's file descriptor with a read-only one.
        store.file = File::open(&path).unwrap();
        assert!(store
            .apply_batch(
                GraphBatch {
                    add_nodes: 1,
                    ..Default::default()
                },
                0,
                "failed".into(),
                "test".into()
            )
            .is_err());
        assert!(store.graph().is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
        drop(store);
        assert_eq!(
            DurableGraph::open(&path)
                .unwrap()
                .graph()
                .unwrap()
                .node_count(),
            0
        );
    }
    #[test]
    fn injected_failures_roll_back_before_exposing_any_graph_mutation() {
        for failure in [
            Failure::BeforeHeader,
            Failure::AfterHeader,
            Failure::AfterPayload,
            Failure::AfterSync,
        ] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("graph.ggj");
            let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
            let baseline = std::fs::read(&path).unwrap();
            store.failure = Some(failure);
            assert!(store
                .apply_batch(
                    GraphBatch {
                        add_nodes: 1,
                        ..Default::default()
                    },
                    0,
                    "failure".into(),
                    "test".into()
                )
                .is_err());
            assert_eq!(store.graph().unwrap().node_count(), 0);
            assert_eq!(std::fs::read(&path).unwrap(), baseline);
            store.failure = None;
            assert_eq!(
                store
                    .apply_batch(
                        GraphBatch {
                            add_nodes: 1,
                            ..Default::default()
                        },
                        0,
                        "retry".into(),
                        "test".into()
                    )
                    .unwrap()
                    .added_nodes,
                vec![0]
            );
        }
    }
    #[test]
    fn indeterminate_synced_write_requires_reopen_and_reconciliation() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("graph.ggj");
        let mut store = DurableGraph::create(&path, GraphConfig::new(), None).unwrap();
        store.failure = Some(Failure::Rollback);
        assert!(store
            .apply_batch(
                GraphBatch {
                    add_nodes: 1,
                    ..Default::default()
                },
                0,
                "ambiguous".into(),
                "test".into()
            )
            .is_err());
        assert!(store.graph().is_err());
        drop(store);
        assert_eq!(
            DurableGraph::open(&path)
                .unwrap()
                .graph()
                .unwrap()
                .node_count(),
            1
        );
    }
}
