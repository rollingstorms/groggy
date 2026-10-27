//! Tracked restoration uses retained logical input and shared snapshot publication.
use super::*;
impl DurableGraph {
    pub(super) fn finish_tracked_restore(
        &mut self,
        outcome: crate::StateRestoreOutcome,
        options: crate::GraphPublicationOptions,
        message: String,
        author: String,
        additional_schema: Option<&GraphSchema>,
    ) -> GraphResult<StateId> {
        let target_state = outcome.target_state();
        let expected_state = outcome.expected_state();
        let branch = outcome.branch().to_owned();
        let prepared = self.graph.prepare_tracked_restore(
            outcome,
            &options,
            self.schema.as_ref(),
            additional_schema,
            None,
        )?;
        self.append(Record::TrackedRestore {
            expected_state,
            target_state,
            branch,
            options,
            metadata: prepared.metadata.clone(),
            message: message.clone(),
            author: author.clone(),
            additional_schema: additional_schema.cloned(),
        })?;
        match self
            .graph
            .execute_snapshot_publication(prepared, message, author)
        {
            Ok(state) => Ok(state),
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Graph, StateId, StateId) {
        let mut g = Graph::new();
        g.add_node().unwrap();
        g.set_node_attr(0, "value".into(), crate::AttrValue::Float64(-0.0))
            .unwrap();
        let old = g.current_state_id();
        g.set_node_attr(0, "value".into(), crate::AttrValue::Int(2))
            .unwrap();
        let current = g.current_state_id();
        (g, old, current)
    }
    fn record(store: &DurableGraph, target: StateId, current: StateId) -> Record {
        let outcome = store
            .graph
            .prepare_state_restore(target, current, crate::StateRestoreMode::Tracked)
            .unwrap()
            .execute()
            .unwrap();
        let prepared = store
            .graph
            .prepare_tracked_restore(
                outcome,
                &crate::GraphPublicationOptions::default(),
                None,
                None,
                None,
            )
            .unwrap();
        Record::TrackedRestore {
            expected_state: current,
            target_state: target,
            branch: "main".into(),
            options: Default::default(),
            metadata: prepared.metadata,
            message: "restore".into(),
            author: "test".into(),
            additional_schema: None,
        }
    }
    #[test]
    fn tracked_restore_append_failure_rolls_back_bytes_state_and_version_allocation() {
        for failure in [
            Failure::BeforeHeader,
            Failure::AfterHeader,
            Failure::AfterPayload,
            Failure::AfterSync,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("graph");
            let (g, target, current) = fixture();
            let mut store = DurableGraph::create_from_graph(&path, &g, None).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            let versions = g.history_records().len();
            store.failure = Some(failure);
            assert!(store
                .restore_state(target, current, crate::StateRestoreMode::Tracked)
                .is_err());
            assert_eq!(std::fs::read(&path).unwrap(), bytes);
            assert_eq!(store.graph.current_state_id(), current);
            assert_eq!(store.graph.history_records().len(), versions);
            store.failure = None;
            assert_eq!(
                store
                    .restore_state(target, current, crate::StateRestoreMode::Tracked)
                    .unwrap(),
                current + 1
            );
            drop(store);
            let store = DurableGraph::open(&path).unwrap();
            let Some(crate::AttrValue::Float64(x)) =
                store.graph.get_node_attr(0, &"value".into()).unwrap()
            else {
                panic!("wrong value")
            };
            assert_eq!(x.to_bits(), (-0.0_f64).to_bits());
        }
    }
    #[test]
    fn synced_tracked_restore_replays_before_install_and_invalid_records_do_not_repair_bytes() {
        for variant in 0..6 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("graph");
            let (g, target, current) = fixture();
            let mut store = DurableGraph::create_from_graph(&path, &g, None).unwrap();
            let mut record = record(&store, target, current);
            if let Record::TrackedRestore {
                expected_state,
                target_state,
                branch,
                metadata,
                options,
                ..
            } = &mut record
            {
                match variant {
                    1 => *expected_state = target,
                    2 => *target_state = 999,
                    3 => *branch = "foreign".into(),
                    4 => metadata.change_times.clear(),
                    5 => {
                        options
                            .change_times
                            .insert(crate::ChangeKey::NodeAdded(999), 5);
                    }
                    _ => {}
                }
            }
            store.append(record).unwrap();
            assert_eq!(store.graph.current_state_id(), current);
            drop(store);
            let bytes = std::fs::read(&path).unwrap();
            let opened = DurableGraph::open(&path);
            if variant == 0 {
                assert_eq!(
                    opened.unwrap().graph().unwrap().current_state_id(),
                    current + 1
                );
            } else {
                assert!(opened.is_err(), "variant {variant}");
            }
            assert_eq!(std::fs::read(&path).unwrap(), bytes);
        }
    }
    #[cfg(unix)]
    #[test]
    #[ignore = "Child fixture for serial tracked restoration process-kill matrix"]
    fn tracked_restore_process_child() {
        let Some(path) = std::env::var_os("GROGGY_TEST_RESTORE_PATH") else {
            return;
        };
        let phase: usize = std::env::var("GROGGY_TEST_RESTORE_PHASE")
            .unwrap()
            .parse()
            .unwrap();
        let mut store = DurableGraph::open(Path::new(&path)).unwrap();
        let current = store.graph.current_state_id();
        let target = current - 1;
        if phase == 4 {
            store
                .restore_state(target, current, crate::StateRestoreMode::Tracked)
                .unwrap();
        } else if phase != 0 {
            let record = record(&store, target, current);
            if phase == 3 {
                store.append(record).unwrap();
            } else {
                let frame = encode_record(store.next_sequence, &record).unwrap();
                let cut = if phase == 1 {
                    HEADER_SIZE / 2
                } else {
                    HEADER_SIZE + (frame.len() - HEADER_SIZE) / 2
                };
                store.file.seek(SeekFrom::End(0)).unwrap();
                store.file.write_all(&frame[..cut]).unwrap();
                store.file.sync_all().unwrap();
            }
        }
        std::fs::write(
            std::env::var_os("GROGGY_TEST_RESTORE_READY").unwrap(),
            b"ready",
        )
        .unwrap();
        loop {
            std::thread::park();
        }
    }
    #[cfg(unix)]
    #[test]
    #[ignore = "Run serially: five real SIGKILL/reopen tracked restoration scenarios"]
    fn killed_tracked_restore_recovers_complete_state_and_releases_writer_lock() {
        use std::{
            process::{Command, Stdio},
            time::{Duration, Instant},
        };
        for phase in 0..5 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("graph");
            let ready = dir.path().join("ready");
            let (g, target, current) = fixture();
            drop(DurableGraph::create_from_graph(&path, &g, None).unwrap());
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "storage::journal::restore::tests::tracked_restore_process_child",
                    "--ignored",
                    "--test-threads=1",
                ])
                .env("GROGGY_TEST_RESTORE_PATH", &path)
                .env("GROGGY_TEST_RESTORE_PHASE", phase.to_string())
                .env("GROGGY_TEST_RESTORE_READY", &ready)
                .stdout(Stdio::null())
                .spawn()
                .unwrap();
            let start = Instant::now();
            while !ready.exists() {
                if let Some(status) = child.try_wait().unwrap() {
                    panic!("child exited at phase {phase}: {status}");
                }
                if start.elapsed() > Duration::from_secs(20) {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("barrier timeout phase {phase}");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            child.kill().unwrap();
            assert!(!child.wait().unwrap().success());
            let mut store = DurableGraph::open(&path).unwrap();
            let expected = if phase >= 3 { current + 1 } else { current };
            assert_eq!(store.graph.current_state_id(), expected);
            assert_eq!(store.recovered_tail_bytes() > 0, phase == 1 || phase == 2);
            assert!(DurableGraph::open(&path).is_err());
            store.verify_storage().unwrap();
            assert!(store.graph.snapshot_at_commit(target).is_ok());
            if phase >= 3 {
                let record = store
                    .graph
                    .history_records()
                    .into_iter()
                    .find(|r| r.id == expected)
                    .unwrap();
                assert_eq!(record.parents, vec![current]);
                let Some(crate::AttrValue::Float64(v)) =
                    store.graph.get_node_attr(0, &"value".into()).unwrap()
                else {
                    panic!("wrong value")
                };
                assert_eq!(v.to_bits(), (-0.0_f64).to_bits());
            } else {
                assert_eq!(
                    store.graph.get_node_attr(0, &"value".into()).unwrap(),
                    Some(crate::AttrValue::Int(2))
                );
            }
            let added = store
                .apply_batch(
                    GraphBatch {
                        add_nodes: 1,
                        ..Default::default()
                    },
                    expected,
                    "after kill".into(),
                    "test".into(),
                )
                .unwrap();
            assert_eq!(added.added_nodes, vec![1]);
            drop(store);
            assert_eq!(
                DurableGraph::open(&path).unwrap().graph.current_state_id(),
                added.state_id
            );
        }
    }
}
