//! Explicit materialized baselines and locked atomic journal replacement.
use super::*;

/// Physical rewrite results. Retained data is preserved; bytes may grow or shrink.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageCheckpointReport {
    /// File size before rewriting the operation stream.
    pub previous_bytes: u64,
    /// Complete replacement file size.
    pub current_bytes: u64,
    /// Number of prior logical journal records replaced by one materialized baseline.
    pub previous_records: u64,
    /// Retained graph versions inside the baseline (including unreferenced versions).
    pub retained_versions: usize,
    /// Selected version; storage maintenance does not publish a graph change.
    pub selected_state: StateId,
    /// Exact slot deduplication counts for explicit compact_storage; None for an
    /// ordinary checkpoint_storage, which preserves raw positions.
    pub pool_compaction: Option<crate::PoolCompactionReport>,
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
    compaction: Option<crate::PoolCompactionPlan>,
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
    compaction: Option<crate::PoolCompactionOutcome>,
}
// Test-only process handshake: no environment-controlled behavior exists in
// production builds. A parent test kills this process after observing the phase.
#[cfg(all(test, unix))]
pub(super) fn pause_for_process_test(phase: &str) {
    if std::env::var("GROGGY_TEST_CHECKPOINT_PHASE")
        .ok()
        .as_deref()
        != Some(phase)
    {
        return;
    }
    let Some(ready) = std::env::var_os("GROGGY_TEST_CHECKPOINT_READY") else {
        return;
    };
    std::fs::write(ready, phase).expect("signal checkpoint phase");
    std::thread::sleep(std::time::Duration::from_secs(30));
    // Prevent an orphan helper if its test parent vanished instead of killing it.
    std::process::exit(86);
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
    /// Deduplicate exact raw property values and publish one verified retained
    /// baseline with locked atomic replacement. History/distinct values/IDs remain.
    /// Pre-replacement errors leave memory and disk unchanged; post-replacement
    /// errors require reopen. Requires Unix like checkpoint_storage.
    pub fn compact_storage(
        &mut self,
        expected_state: StateId,
    ) -> GraphResult<StorageCheckpointReport> {
        let plan = self.prepare_storage_compaction(expected_state)?;
        self.finish_storage_checkpoint(plan.execute())
    }
    /// Capture raw columns and a full baseline for detached compaction and I/O.
    /// This explicit maintenance can temporarily duplicate retained payloads.
    #[doc(hidden)]
    pub fn prepare_storage_compaction(
        &mut self,
        expected_state: StateId,
    ) -> GraphResult<StorageCheckpointPlan> {
        self.ready()?;
        self.graph.validate_write_head(expected_state)?;
        let compaction = self.graph.prepare_pool_compaction()?;
        let mut plan = self.prepare_storage_checkpoint(expected_state)?;
        plan.compaction = Some(compaction);
        Ok(plan)
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
            pool_compaction: None,
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
            compaction: None,
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
            compaction,
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
        if result.is_ok() {
            if let Some(compaction) = compaction {
                if let Err(error) = self.graph.finish_pool_compaction(compaction) {
                    self.poisoned = true;
                    return Err(error);
                }
            }
        }
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
    pub fn execute(mut self) -> StorageCheckpointOutcome {
        let mut replacement = None;
        let mut invalidate_owner = false;
        let mut compaction = None;
        let prepared = (|| -> GraphResult<()> {
            if let Some(plan) = self.compaction.take() {
                let outcome = plan.execute()?;
                let Record::InitArchive { archive, .. } = &mut self.record else {
                    return Err(invalid("compaction requires a full retained baseline"));
                };
                outcome.rewrite_archive(archive)?;
                self.report.pool_compaction = Some(outcome.report());
                compaction = Some(outcome);
            }
            Ok(())
        })();
        #[cfg(unix)]
        let result =
            prepared.and_then(|()| self.execute_unix(&mut replacement, &mut invalidate_owner));
        #[cfg(not(unix))]
        let result = prepared.and_then(|()| {
            Err(invalid(
                "atomic storage checkpoint replacement currently requires Unix",
            ))
        });
        let compaction = if result.is_ok() { compaction } else { None };
        StorageCheckpointOutcome {
            replacement,
            invalidate_owner,
            result,
            source_path: self.path,
            source_identity: self.source,
            expected_state: self.report.selected_state,
            compaction,
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
        pause_for_process_test(&format!("{phase:?}"));
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
    fn checkpoint_process_child() {
        let Some(path) = std::env::var_os("GROGGY_TEST_CHECKPOINT_PATH") else {
            return;
        };
        let mut store = DurableGraph::open(Path::new(&path)).expect("open child journal");
        let state = store.graph.current_state_id();
        if std::env::var("GROGGY_TEST_CHECKPOINT_COMPACT").as_deref() == Ok("true") {
            store
                .compact_storage(state)
                .expect("compaction must reach requested phase");
        } else {
            store
                .checkpoint_storage(state)
                .expect("checkpoint must reach requested phase");
        }
        panic!("checkpoint did not pause at requested boundary");
    }

    #[test]
    #[ignore = "run serially: spawned children briefly inherit process-wide flock descriptors"]
    fn killed_checkpoint_process_recovers_complete_state_and_releases_both_inode_locks() {
        use std::os::unix::process::ExitStatusExt;
        use std::process::{Child, Command, Stdio};
        use std::time::{Duration, Instant};
        struct KillOnDrop(Child);
        impl Drop for KillOnDrop {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        for compact in [false, true] {
            for phase in [
                "BeforeWrite",
                "AfterFirstFragment",
                "AfterSync",
                "BeforeReplace",
                "AfterReplace",
                "AfterDirectorySync",
            ] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("journal");
                let ready = dir.path().join("ready");
                let mut store = populated(&path);
                if compact {
                    let state = store.graph.current_state_id();
                    store
                        .apply_batch(
                            GraphBatch {
                                node_attrs: HashMap::from([(
                                    "v".into(),
                                    vec![(N::Existing(0), V::Int(10))],
                                )]),
                                ..Default::default()
                            },
                            state,
                            "duplicate".into(),
                            "test".into(),
                        )
                        .unwrap();
                }
                if phase == "AfterFirstFragment" {
                    let state = store.graph.current_state_id();
                    store
                        .apply_batch(
                            GraphBatch {
                                node_attrs: HashMap::from([(
                                    "large".into(),
                                    vec![(N::Existing(0), V::Text("x".repeat(9 * 1024 * 1024)))],
                                )]),
                                ..Default::default()
                            },
                            state,
                            "large".into(),
                            "test".into(),
                        )
                        .unwrap();
                }
                let state = store.graph.current_state_id();
                store.set_schema(None, state, 0).unwrap();
                store
                    .checkpoint(state, "label".into(), "test".into())
                    .unwrap();
                let clocks = store.graph.publication_metadata(state).unwrap();
                let history: Vec<_> = store
                    .graph
                    .history_records()
                    .iter()
                    .map(|v| (v.id, v.content_hash))
                    .collect();
                let before = std::fs::read(&path).unwrap();
                drop(store);
                let mut child = KillOnDrop(
                    Command::new(std::env::current_exe().unwrap())
                        .args([
                            "--exact",
                            "storage::journal::checkpoint::tests::checkpoint_process_child",
                            "--nocapture",
                        ])
                        .env("GROGGY_TEST_CHECKPOINT_PATH", &path)
                        .env("GROGGY_TEST_CHECKPOINT_COMPACT", compact.to_string())
                        .env("GROGGY_TEST_CHECKPOINT_PHASE", phase)
                        .env("GROGGY_TEST_CHECKPOINT_READY", &ready)
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn()
                        .unwrap(),
                );
                let deadline = Instant::now() + Duration::from_secs(20);
                while !ready.exists() {
                    assert!(
                        child.0.try_wait().unwrap().is_none(),
                        "helper exited before {phase}"
                    );
                    assert!(Instant::now() < deadline, "helper timed out at {phase}");
                    std::thread::sleep(Duration::from_millis(5));
                }
                child.0.kill().unwrap();
                assert_eq!(child.0.wait().unwrap().signal(), Some(9));
                if !matches!(phase, "AfterReplace" | "AfterDirectorySync") {
                    assert_eq!(std::fs::read(&path).unwrap(), before, "{phase}");
                }
                let mut recovered = DurableGraph::open(&path).unwrap();
                assert_eq!(recovered.graph.current_state_id(), state, "{phase}");
                assert_eq!(recovered.schema_revision, 1, "{phase}");
                assert_eq!(
                    recovered.graph.publication_metadata(state).unwrap(),
                    clocks,
                    "{phase}"
                );
                assert_eq!(
                    recovered
                        .graph
                        .history_records()
                        .iter()
                        .map(|v| (v.id, v.content_hash))
                        .collect::<Vec<_>>(),
                    history,
                    "{phase}"
                );
                assert_eq!(
                    recovered.graph.get_node_attr(0, &"v".into()).unwrap(),
                    Some(V::Int(10)),
                    "{phase}"
                );
                assert_eq!(recovered.graph.checkpoints().len(), 1, "{phase}");
                if compact {
                    let expected = if matches!(phase, "AfterReplace" | "AfterDirectorySync") {
                        11
                    } else {
                        12
                    };
                    assert_eq!(
                        recovered
                            .graph
                            .pool()
                            .get_node_attribute_column(&"v".into())
                            .unwrap()
                            .len(),
                        expected,
                        "{phase}"
                    );
                }
                assert!(
                    DurableGraph::open(&path).is_err(),
                    "reopened writer lock missing at {phase}"
                );
                let appended = recovered
                    .apply_batch(
                        GraphBatch {
                            add_nodes: 1,
                            ..Default::default()
                        },
                        state,
                        "after kill".into(),
                        "test".into(),
                    )
                    .unwrap();
                assert_eq!(appended.added_nodes, vec![1], "{phase}");
                drop(recovered);
                assert_eq!(
                    DurableGraph::open(&path).unwrap().graph.node_count(),
                    2,
                    "{phase}"
                );
            }
        }
    }

    #[test]
    fn compaction_failures_keep_original_memory_and_recover_matching_old_or_new_storage() {
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
            store
                .apply_batch(
                    GraphBatch {
                        node_attrs: HashMap::from([(
                            "v".into(),
                            vec![(N::Existing(0), V::Int(10))],
                        )]),
                        ..Default::default()
                    },
                    state,
                    "duplicate".into(),
                    "test".into(),
                )
                .unwrap();
            let state = store.graph.current_state_id();
            let before = store.graph.check_integrity().unwrap();
            let original = std::fs::read(&path).unwrap();
            store.checkpoint_failure = Some(phase);
            assert!(store.compact_storage(state).is_err(), "{phase:?}");
            let replaced = matches!(
                phase,
                CheckpointFailure::AfterReplace | CheckpointFailure::AfterDirectorySync
            );
            assert_eq!(store.poisoned, replaced);
            // Even a post-rename error never partially installs raw memory.
            assert_eq!(store.graph.check_integrity().unwrap(), before);
            if !replaced {
                assert_eq!(std::fs::read(&path).unwrap(), original);
            }
            assert!(DurableGraph::open(&path).is_err());
            drop(store);
            let mut recovered = DurableGraph::open(&path).unwrap();
            assert_eq!(recovered.graph.current_state_id(), state);
            assert_eq!(
                recovered
                    .graph
                    .pool()
                    .get_node_attribute_column(&"v".into())
                    .unwrap()
                    .len(),
                if replaced { 11 } else { 12 }
            );
            recovered.graph.check_integrity().unwrap();
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
