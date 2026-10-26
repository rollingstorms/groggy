//! Explicit materialized baselines and locked atomic journal replacement.
use super::*;

/// Physical rewrite results. Retained data is preserved; bytes may grow or shrink.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageCheckpointReport {
    /// File size before rewriting the operation stream.
    pub previous_bytes: u64,
    /// Complete replacement file size.
    pub current_bytes: u64,
    /// Number of prior journal frames replaced by one materialized baseline.
    pub previous_records: u64,
    /// Retained graph versions inside the baseline (including unreferenced versions).
    pub retained_versions: usize,
    /// Selected version; storage maintenance does not publish a graph change.
    pub selected_state: StateId,
}
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CheckpointFailure {
    BeforeWrite,
    AfterSync,
    BeforeReplace,
    AfterReplace,
    AfterDirectorySync,
}

/// Detached file-maintenance plan. Execute then return its outcome to the same
/// DurableGraph owner before further operations; the source inode stays locked.
#[doc(hidden)]
pub struct StorageCheckpointPlan {
    source: File,
    path: PathBuf,
    record: Record,
    report: StorageCheckpointReport,
    schema_revision: u64,
    #[cfg(test)]
    failure: Option<CheckpointFailure>,
}
/// Owns a replacement inode lock even when a post-rename operation failed.
#[doc(hidden)]
pub struct StorageCheckpointOutcome {
    replacement: Option<File>,
    invalidate_owner: bool,
    result: GraphResult<StorageCheckpointReport>,
    source_path: PathBuf,
    source_identity: File,
    expected_state: StateId,
}
impl DurableGraph {
    /// Replace the operation stream with one complete retained baseline. No history,
    /// raw values or identity reservations are pruned and no graph version is created.
    /// Both inodes stay locked across atomic Unix replacement. Errors after replacement
    /// poison this writer until reopen. Large baselines use bounded physical fragments.
    pub fn checkpoint_storage(
        &mut self,
        expected_state: StateId,
    ) -> GraphResult<StorageCheckpointReport> {
        let plan = self.prepare_storage_checkpoint(expected_state)?;
        self.finish_storage_checkpoint(plan.execute())
    }
    /// Prepare detached storage work for runtime integrations. The owner must not
    /// accept any mutations until finish_storage_checkpoint consumes the outcome.
    #[doc(hidden)]
    #[cfg(unix)]
    pub fn prepare_storage_checkpoint(
        &mut self,
        expected_state: StateId,
    ) -> GraphResult<StorageCheckpointPlan> {
        self.ready()?;
        self.graph.validate_write_head(expected_state)?;
        if let Err(error) = check_path(&self.file, &self.path) {
            self.poisoned = true;
            return Err(error);
        }
        let before = self
            .file
            .metadata()
            .map_err(|e| io_error(&self.path, "stat_checkpoint_source", e))?;
        let report = StorageCheckpointReport {
            previous_bytes: before.len(),
            current_bytes: 0,
            previous_records: self.next_sequence,
            retained_versions: self.graph.history_records().len(),
            selected_state: self.graph.current_state_id(),
        };
        let record = Record::InitArchive {
            archive: Box::new(self.graph.archive_document()?),
            schema: self.schema.clone(),
            schema_revision: self.schema_revision,
        };
        let source = self
            .file
            .try_clone()
            .map_err(|e| io_error(&self.path, "clone_checkpoint_lock", e))?;
        // Prevent writes/reads through an owner that has detached maintenance in flight.
        // Completion re-enables it only after matching and consuming the outcome.
        self.poisoned = true;
        Ok(StorageCheckpointPlan {
            source,
            path: self.path.clone(),
            record,
            report,
            schema_revision: self.schema_revision,
            #[cfg(test)]
            failure: self.checkpoint_failure,
        })
    }
    /// Unsupported platforms reject without touching the source journal.
    #[doc(hidden)]
    #[cfg(not(unix))]
    pub fn prepare_storage_checkpoint(
        &mut self,
        _expected_state: StateId,
    ) -> GraphResult<StorageCheckpointPlan> {
        Err(invalid(
            "atomic storage checkpoint replacement currently requires Unix",
        ))
    }
    /// Reattach the locked replacement, or retain the untouched old writer on a
    /// pre-replacement failure. A mismatched/abandoned outcome requires reopening.
    #[doc(hidden)]
    pub fn finish_storage_checkpoint(
        &mut self,
        outcome: StorageCheckpointOutcome,
    ) -> GraphResult<StorageCheckpointReport> {
        let StorageCheckpointOutcome {
            replacement,
            invalidate_owner,
            result,
            source_path,
            source_identity,
            expected_state,
        } = outcome;
        if source_path != self.path
            || expected_state != self.graph.current_state_id()
            || !same_inode(&self.file, &source_identity)?
        {
            self.poisoned = true;
            return Err(invalid(
                "storage checkpoint outcome belongs to a different writer",
            ));
        }
        if let Some(file) = replacement {
            self.file = file;
            self.next_sequence = 1;
            self.recovered_tail_bytes = 0;
        }
        self.poisoned = invalidate_owner;
        result
    }
}
#[cfg(unix)]
fn same_inode(a: &File, b: &File) -> GraphResult<bool> {
    use std::os::unix::fs::MetadataExt;
    let a = a.metadata().map_err(invalid)?;
    let b = b.metadata().map_err(invalid)?;
    Ok(a.dev() == b.dev() && a.ino() == b.ino())
}
#[cfg(not(unix))]
fn same_inode(_a: &File, _b: &File) -> GraphResult<bool> {
    Ok(false)
}
#[cfg(unix)]
fn check_path(source: &File, path: &Path) -> GraphResult<()> {
    use std::os::unix::fs::MetadataExt;
    let owned = source
        .metadata()
        .map_err(|e| io_error(path, "stat_owned_journal", e))?;
    let named =
        std::fs::symlink_metadata(path).map_err(|e| io_error(path, "stat_named_journal", e))?;
    if named.file_type().is_symlink() || named.dev() != owned.dev() || named.ino() != owned.ino() {
        return Err(invalid("journal path changed or is a symbolic link; reopen the actual journal path before checkpointing"));
    }
    Ok(())
}
impl StorageCheckpointPlan {
    /// Perform detached file I/O and candidate verification. Always finish the outcome
    /// on its original owner, including errors, to preserve lock/publication ownership.
    pub fn execute(self) -> StorageCheckpointOutcome {
        let mut replacement = None;
        let mut invalidate_owner = false;
        #[cfg(unix)]
        let result = self.execute_unix(&mut replacement, &mut invalidate_owner);
        #[cfg(not(unix))]
        let result = Err(invalid(
            "atomic storage checkpoint replacement currently requires Unix",
        ));
        StorageCheckpointOutcome {
            replacement,
            invalidate_owner,
            result,
            source_path: self.path,
            source_identity: self.source,
            expected_state: self.report.selected_state,
        }
    }
    #[cfg(unix)]
    fn execute_unix(
        &self,
        replacement: &mut Option<File>,
        invalidate_owner: &mut bool,
    ) -> GraphResult<StorageCheckpointReport> {
        if let Err(error) = check_path(&self.source, &self.path) {
            *invalidate_owner = true;
            return Err(error);
        }
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut temporary = tempfile::NamedTempFile::new_in(parent)
            .map_err(|e| io_error(&self.path, "create_storage_checkpoint", e))?;
        #[cfg(test)]
        self.inject(CheckpointFailure::BeforeWrite)?;
        temporary
            .write_all(MAGIC)
            .map_err(|e| io_error(&self.path, "write_checkpoint_header", e))?;
        baseline::write_initial_record(&mut temporary, &self.record)?;
        temporary
            .as_file()
            .set_permissions(self.source.metadata().map_err(invalid)?.permissions())
            .and_then(|_| temporary.as_file().sync_all())
            .map_err(|e| io_error(&self.path, "write_and_sync_storage_checkpoint", e))?;
        #[cfg(test)]
        self.inject(CheckpointFailure::AfterSync)?;
        let verified = DurableGraph::open(temporary.path())?;
        if verified.graph.current_state_id() != self.report.selected_state
            || verified.schema_revision != self.schema_revision
            || verified.next_sequence != 1
            || verified.recovered_tail_bytes != 0
        {
            return Err(invalid("storage checkpoint verification mismatch"));
        }
        drop(verified);
        lock(temporary.as_file(), &self.path)?;
        if let Err(error) = check_path(&self.source, &self.path) {
            *invalidate_owner = true;
            return Err(error);
        }
        #[cfg(test)]
        self.inject(CheckpointFailure::BeforeReplace)?;
        let file = temporary
            .persist(&self.path)
            .map_err(|e| io_error(&self.path, "replace_storage_checkpoint", e.error))?;
        *replacement = Some(file);
        *invalidate_owner = true;
        #[cfg(test)]
        self.inject(CheckpointFailure::AfterReplace)?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|e| io_error(parent, "sync_storage_checkpoint_directory", e))?;
        #[cfg(test)]
        self.inject(CheckpointFailure::AfterDirectorySync)?;
        let current_bytes = std::fs::metadata(&self.path)
            .map_err(|e| io_error(&self.path, "stat_storage_checkpoint", e))?
            .len();
        *invalidate_owner = false;
        Ok(StorageCheckpointReport {
            current_bytes,
            ..self.report.clone()
        })
    }
    #[cfg(all(test, unix))]
    fn inject(&self, phase: CheckpointFailure) -> GraphResult<()> {
        if self.failure == Some(phase) {
            return Err(invalid(format!(
                "injected storage checkpoint failure {phase:?}"
            )));
        }
        Ok(())
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::{AttrValue as V, BatchNodeRef as N};
    use std::collections::HashMap;
    fn populated(path: &Path) -> DurableGraph {
        let mut store = DurableGraph::create(path, GraphConfig::new(), None).unwrap();
        store
            .apply_batch(
                GraphBatch {
                    add_nodes: 1,
                    node_attrs: HashMap::from([("v".into(), vec![(N::Added(0), V::Int(0))])]),
                    effective_at: Some(0),
                    ..Default::default()
                },
                0,
                "root".into(),
                "test".into(),
            )
            .unwrap();
        for i in 1..=10 {
            let state = store.graph.current_state_id();
            store
                .apply_batch(
                    GraphBatch {
                        node_attrs: HashMap::from([(
                            "v".into(),
                            vec![(N::Existing(0), V::Int(i))],
                        )]),
                        effective_at: Some(i),
                        ..Default::default()
                    },
                    state,
                    "update".into(),
                    "test".into(),
                )
                .unwrap();
        }
        store
    }
    #[test]
    fn every_failure_phase_preserves_complete_recovery_and_continuous_writer_lock() {
        for phase in [
            CheckpointFailure::BeforeWrite,
            CheckpointFailure::AfterSync,
            CheckpointFailure::BeforeReplace,
            CheckpointFailure::AfterReplace,
            CheckpointFailure::AfterDirectorySync,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("journal");
            let mut store = populated(&path);
            let state = store.graph.current_state_id();
            let original = std::fs::read(&path).unwrap();
            store.checkpoint_failure = Some(phase);
            assert!(store.checkpoint_storage(state).is_err(), "{phase:?}");
            let replaced = matches!(
                phase,
                CheckpointFailure::AfterReplace | CheckpointFailure::AfterDirectorySync
            );
            assert_eq!(store.poisoned, replaced, "{phase:?}");
            if !replaced {
                assert_eq!(std::fs::read(&path).unwrap(), original);
            }
            assert!(DurableGraph::open(&path).is_err(), "lock lost at {phase:?}");
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
            drop(store);
            let mut recovered = DurableGraph::open(&path).unwrap();
            assert_eq!(recovered.graph.current_state_id(), state);
            assert_eq!(
                recovered
                    .graph()
                    .unwrap()
                    .get_node_attr(0, &"v".into())
                    .unwrap(),
                Some(V::Int(10))
            );
            assert_eq!(recovered.graph().unwrap().history_records().len(), 11);
            recovered
                .apply_batch(
                    GraphBatch {
                        add_nodes: 1,
                        ..Default::default()
                    },
                    state,
                    "after".into(),
                    "test".into(),
                )
                .unwrap();
        }
    }
    #[test]
    fn detached_maintenance_rejects_mutation_until_finished_and_retains_lock_in_outcome() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal");
        let mut store = populated(&path);
        let state = store.graph.current_state_id();
        let plan = store.prepare_storage_checkpoint(state).unwrap();
        assert!(store.graph().is_err());
        assert!(store
            .apply_batch(GraphBatch::default(), state, "bad".into(), "test".into())
            .is_err());
        let outcome = std::thread::spawn(move || plan.execute()).join().unwrap();
        assert!(DurableGraph::open(&path).is_err());
        let report = store.finish_storage_checkpoint(outcome).unwrap();
        assert_eq!(report.previous_records, 12);
        assert_eq!(report.retained_versions, 11);
        assert_eq!(store.next_sequence, 1);
        assert_eq!(store.graph.current_state_id(), state);
        assert!(DurableGraph::open(&path).is_err());
    }
    #[test]
    fn changed_path_and_symbolic_link_cannot_replace_an_unrelated_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal");
        let moved = dir.path().join("moved");
        let mut store = populated(&path);
        let state = store.graph.current_state_id();
        std::fs::rename(&path, &moved).unwrap();
        std::fs::write(&path, b"unrelated").unwrap();
        assert!(store.checkpoint_storage(state).is_err());
        assert!(store.poisoned);
        assert_eq!(std::fs::read(&path).unwrap(), b"unrelated");
        drop(store);
        let mut moved_store = DurableGraph::open(&moved).unwrap();
        moved_store.checkpoint_storage(state).unwrap();
        drop(moved_store);
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&moved, &link).unwrap();
        let mut linked = DurableGraph::open(&link).unwrap();
        assert!(linked.checkpoint_storage(state).is_err());
        assert!(std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
    }
    #[test]
    fn pre_replacement_failure_can_retry_and_stale_request_leaves_file_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal");
        let mut store = populated(&path);
        let state = store.graph.current_state_id();
        let original = std::fs::read(&path).unwrap();
        assert!(store.checkpoint_storage(state - 1).is_err());
        assert!(!store.poisoned);
        assert_eq!(std::fs::read(&path).unwrap(), original);
        store.checkpoint_failure = Some(CheckpointFailure::AfterSync);
        assert!(store.checkpoint_storage(state).is_err());
        store.checkpoint_failure = None;
        store.checkpoint_storage(state).unwrap();
        assert!(!store.poisoned);
    }
}
