//! Exact logical resolved merge records; never persist raw column offsets.
use super::*;
use crate::state::ChangeSet;
use crate::{GraphMergeOptions, GraphMergeOutcome};
type StoredChange = (usize, String, Option<Value>, Option<Value>);
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StoredMerge {
    left: StateId,
    right: StateId,
    base: StateId,
    branch: String,
    nodes_added: Vec<usize>,
    nodes_removed: Vec<usize>,
    edges_added: Vec<(usize, usize, usize)>,
    edges_removed: Vec<usize>,
    edge_endpoint_changes: Vec<crate::EdgeEndpointChange>,
    node_attr_changes: Vec<StoredChange>,
    edge_attr_changes: Vec<StoredChange>,
}
fn encode_changes(
    rows: &[(
        usize,
        String,
        Option<crate::AttrValue>,
        Option<crate::AttrValue>,
    )],
) -> GraphResult<Vec<StoredChange>> {
    rows.iter()
        .map(|(id, name, old, new)| {
            Ok((
                *id,
                name.clone(),
                old.as_ref().map(value_codec::encode).transpose()?,
                new.as_ref().map(value_codec::encode).transpose()?,
            ))
        })
        .collect()
}
fn decode_changes(
    rows: Vec<StoredChange>,
) -> GraphResult<
    Vec<(
        usize,
        String,
        Option<crate::AttrValue>,
        Option<crate::AttrValue>,
    )>,
> {
    rows.into_iter()
        .map(|(id, name, old, new)| {
            Ok((
                id,
                name,
                old.map(value_codec::decode).transpose()?,
                new.map(value_codec::decode).transpose()?,
            ))
        })
        .collect()
}
impl StoredMerge {
    fn encode(outcome: &GraphMergeOutcome) -> GraphResult<Self> {
        let changes = outcome.changes();
        let [left, right] = outcome.parents();
        Ok(Self {
            left,
            right,
            base: outcome.base_state(),
            branch: outcome.branch().into(),
            nodes_added: changes.nodes_added.clone(),
            nodes_removed: changes.nodes_removed.clone(),
            edges_added: changes.edges_added.clone(),
            edges_removed: changes.edges_removed.clone(),
            edge_endpoint_changes: changes.edge_endpoint_changes.clone(),
            node_attr_changes: encode_changes(&changes.node_attr_changes)?,
            edge_attr_changes: encode_changes(&changes.edge_attr_changes)?,
        })
    }
    pub(super) fn outcome(self, graph: &Graph) -> GraphResult<GraphMergeOutcome> {
        let changes = ChangeSet {
            nodes_added: self.nodes_added,
            nodes_removed: self.nodes_removed,
            edges_added: self.edges_added,
            edges_removed: self.edges_removed,
            edge_endpoint_changes: self.edge_endpoint_changes,
            node_attr_changes: decode_changes(self.node_attr_changes)?,
            edge_attr_changes: decode_changes(self.edge_attr_changes)?,
        };
        let outcome = graph.resolved_merge_for_replay(self.left, self.right, self.base, changes)?;
        if outcome.branch() != self.branch {
            return Err(invalid("merge branch differs from selected branch"));
        }
        Ok(outcome)
    }
}
impl DurableGraph {
    /// Sync and publish exactly one resolved two-parent merge under the current
    /// installed schema. Existing entity IDs are selected, never reallocated.
    pub fn finish_merge(
        &mut self,
        outcome: GraphMergeOutcome,
        options: GraphMergeOptions,
        message: String,
        author: String,
    ) -> GraphResult<StateId> {
        self.finish_merge_with_schema(outcome, options, message, author, None)
    }
    /// Add a stricter per-call policy without weakening the installed schema.
    pub fn finish_merge_with_schema(
        &mut self,
        outcome: GraphMergeOutcome,
        options: GraphMergeOptions,
        message: String,
        author: String,
        additional_schema: Option<&GraphSchema>,
    ) -> GraphResult<StateId> {
        self.ready()?;
        let resolved = StoredMerge::encode(&outcome)?;
        let prepared = self.graph.prepare_merge_publication(
            outcome,
            &options,
            self.schema.as_ref(),
            additional_schema,
            None,
        )?;
        self.append(Record::Merge {
            resolved: Box::new(resolved),
            options,
            metadata: prepared.metadata.clone(),
            message: message.clone(),
            author: author.clone(),
            additional_schema: additional_schema.cloned(),
        })?;
        match self
            .graph
            .execute_merge_publication(prepared, message, author)
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
    fn source() -> (Graph, StateId, StateId) {
        let mut graph = Graph::new();
        graph.add_node().unwrap();
        graph.create_branch("right".into()).unwrap();
        graph
            .set_node_attr(0, "left".into(), crate::AttrValue::Int(1))
            .unwrap();
        let left = graph.current_state_id();
        graph.checkout_branch("right".into()).unwrap();
        graph
            .set_node_attr(0, "right".into(), crate::AttrValue::Int(2))
            .unwrap();
        let right = graph.current_state_id();
        graph.checkout_branch("main".into()).unwrap();
        (graph, left, right)
    }
    #[test]
    fn merge_append_failures_preserve_bytes_state_ids_and_both_parent_histories() {
        for failure in [
            Failure::BeforeHeader,
            Failure::AfterHeader,
            Failure::AfterPayload,
            Failure::AfterSync,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("graph");
            let (graph, left, right) = source();
            let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
            let before = std::fs::read(&path).unwrap();
            let versions = store.graph().unwrap().history_records().len();
            store.failure = Some(failure);
            let outcome = store
                .graph()
                .unwrap()
                .prepare_merge_read(right, None, left)
                .unwrap()
                .execute()
                .unwrap()
                .into_outcome()
                .unwrap();
            assert!(store
                .finish_merge(
                    outcome,
                    GraphMergeOptions::default(),
                    "bad".into(),
                    "test".into()
                )
                .is_err());
            assert_eq!(std::fs::read(&path).unwrap(), before);
            assert_eq!(store.graph().unwrap().current_state_id(), left);
            assert_eq!(store.graph().unwrap().history_records().len(), versions);
            store.failure = None;
            let outcome = store
                .graph()
                .unwrap()
                .prepare_merge_read(right, None, left)
                .unwrap()
                .execute()
                .unwrap()
                .into_outcome()
                .unwrap();
            let id = store
                .finish_merge(
                    outcome,
                    GraphMergeOptions::default(),
                    "ok".into(),
                    "test".into(),
                )
                .unwrap();
            assert_eq!(id, right + 1);
            drop(store);
            assert_eq!(
                DurableGraph::open(&path)
                    .unwrap()
                    .graph()
                    .unwrap()
                    .current_state_id(),
                id
            );
        }
    }
    #[test]
    fn synced_merge_recovers_before_memory_install_and_invalid_semantic_frames_never_repair_bytes()
    {
        for variant in 0..7 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("graph");
            let (graph, left, right) = source();
            let mut store = DurableGraph::create_from_graph(&path, &graph, None).unwrap();
            let outcome = store
                .graph()
                .unwrap()
                .prepare_merge_read(right, None, left)
                .unwrap()
                .execute()
                .unwrap()
                .into_outcome()
                .unwrap();
            let mut resolved = StoredMerge::encode(&outcome).unwrap();
            let mut prepared = store
                .graph
                .prepare_merge_publication(outcome, &GraphMergeOptions::default(), None, None, None)
                .unwrap();
            match variant {
                1 => resolved.branch = "other".into(),
                2 => resolved.right = 999,
                3 => prepared.metadata.change_times.clear(),
                4 => {
                    resolved.nodes_added.push(999);
                }
                5 => {
                    resolved.node_attr_changes[0].2 =
                        Some(value_codec::encode(&crate::AttrValue::Int(999)).unwrap())
                }
                6 => resolved.nodes_added.push(0),
                _ => {}
            }
            store
                .append(Record::Merge {
                    resolved: Box::new(resolved),
                    options: GraphMergeOptions::default(),
                    metadata: prepared.metadata,
                    message: "replay".into(),
                    author: "test".into(),
                    additional_schema: None,
                })
                .unwrap();
            assert_eq!(store.graph.current_state_id(), left);
            drop(store);
            let bytes = std::fs::read(&path).unwrap();
            let reopened = DurableGraph::open(&path);
            if variant == 0 {
                assert_eq!(
                    reopened.unwrap().graph().unwrap().current_state_id(),
                    right + 1
                );
            } else {
                assert!(reopened.is_err(), "variant {variant}");
            }
            assert_eq!(std::fs::read(&path).unwrap(), bytes);
        }
    }
}

#[cfg(all(test, unix))]
mod process_tests {
    use super::*;
    fn source() -> (Graph, StateId, StateId) {
        let mut graph = Graph::new();
        graph.add_node().unwrap();
        graph.create_branch("right".into()).unwrap();
        graph
            .set_node_attr(0, "left".into(), crate::AttrValue::Int(1))
            .unwrap();
        let left = graph.current_state_id();
        graph.checkout_branch("right".into()).unwrap();
        graph
            .set_node_attr(0, "right".into(), crate::AttrValue::Float64(-0.0))
            .unwrap();
        let right = graph.current_state_id();
        graph.checkout_branch("main".into()).unwrap();
        (graph, left, right)
    }
    #[test]
    #[ignore = "Child fixture for the dedicated serial real-SIGKILL merge matrix"]
    fn merge_process_child() {
        let Some(path) = std::env::var_os("GROGGY_TEST_MERGE_PATH") else {
            return;
        };
        let phase: usize = std::env::var("GROGGY_TEST_MERGE_PHASE")
            .unwrap()
            .parse()
            .unwrap();
        let mut store = DurableGraph::open(Path::new(&path)).unwrap();
        let left = store.graph.current_state_id();
        let right = store
            .graph
            .history_records()
            .iter()
            .map(|c| c.id)
            .max()
            .unwrap();
        let outcome = store
            .graph
            .prepare_merge_read(right, None, left)
            .unwrap()
            .execute()
            .unwrap()
            .into_outcome()
            .unwrap();
        if phase == 4 {
            store
                .finish_merge(
                    outcome,
                    GraphMergeOptions::default(),
                    "kill".into(),
                    "test".into(),
                )
                .unwrap();
        } else if phase != 0 {
            let resolved = StoredMerge::encode(&outcome).unwrap();
            let prepared = store
                .graph
                .prepare_merge_publication(outcome, &GraphMergeOptions::default(), None, None, None)
                .unwrap();
            let record = Record::Merge {
                resolved: Box::new(resolved),
                options: GraphMergeOptions::default(),
                metadata: prepared.metadata,
                message: "kill".into(),
                author: "test".into(),
                additional_schema: None,
            };
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
            std::env::var_os("GROGGY_TEST_MERGE_READY").unwrap(),
            b"ready",
        )
        .unwrap();
        loop {
            std::thread::park();
        }
    }
    #[test]
    #[ignore = "Run serially: starts and SIGKILLs five independent merge writer processes"]
    fn killed_merge_process_recovers_complete_accepted_state_and_releases_writer_lock() {
        use std::{
            process::{Command, Stdio},
            time::{Duration, Instant},
        };
        for phase in 0..5 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("graph");
            let ready = dir.path().join("ready");
            let (graph, left, right) = source();
            drop(DurableGraph::create_from_graph(&path, &graph, None).unwrap());
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "storage::journal::merge::process_tests::merge_process_child",
                    "--ignored",
                    "--test-threads=1",
                ])
                .env("GROGGY_TEST_MERGE_PATH", &path)
                .env("GROGGY_TEST_MERGE_PHASE", phase.to_string())
                .env("GROGGY_TEST_MERGE_READY", &ready)
                .stdout(Stdio::null())
                .spawn()
                .unwrap();
            let started = Instant::now();
            while !ready.exists() {
                if let Some(status) = child.try_wait().unwrap() {
                    panic!("merge child exited at phase {phase}: {status}");
                }
                if started.elapsed() > Duration::from_secs(20) {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("merge barrier timed out at phase {phase}");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            child.kill().unwrap();
            assert!(!child.wait().unwrap().success());
            let mut reopened = DurableGraph::open(&path).unwrap();
            let expected = if phase >= 3 { right + 1 } else { left };
            assert_eq!(reopened.graph.current_state_id(), expected, "phase {phase}");
            assert_eq!(
                reopened.recovered_tail_bytes() > 0,
                phase == 1 || phase == 2
            );
            assert!(DurableGraph::open(&path).is_err());
            reopened.verify_storage().unwrap();
            if phase >= 3 {
                let record = reopened
                    .graph
                    .history_records()
                    .into_iter()
                    .find(|c| c.id == expected)
                    .unwrap();
                assert_eq!(record.parents, vec![left, right]);
                if let Some(crate::AttrValue::Float64(v)) =
                    reopened.graph.get_node_attr(0, &"right".into()).unwrap()
                {
                    assert_eq!(v.to_bits(), (-0.0_f64).to_bits());
                } else {
                    panic!()
                }
            } else {
                assert!(reopened
                    .graph
                    .get_node_attr(0, &"right".into())
                    .unwrap()
                    .is_none());
            }
            let next = reopened
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
            assert_eq!(next.added_nodes, vec![1]);
            assert!(next.state_id > right);
            drop(reopened);
            assert_eq!(DurableGraph::open(&path).unwrap().graph.node_count(), 2);
        }
    }
}
