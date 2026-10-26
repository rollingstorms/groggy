use groggy::state::{change_tracker::ChangeSet, history::HistoryForest};
use groggy::temporal::TemporalSnapshot;
use groggy::{AttrValue as V, BatchNodeRef as N, Graph, GraphBatch, PublicationMetadata};
use std::collections::HashMap;
fn initial(graph: &mut Graph) -> u64 {
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 1,
                node_attrs: HashMap::from([("v".into(), vec![(N::Added(0), V::Null)])]),
                effective_at: Some(10),
                ..Default::default()
            },
            0,
            "first".into(),
            "test".into(),
        )
        .unwrap()
        .state_id
}
#[test]
fn recorded_and_effective_reads_follow_selected_branch_without_leaking_siblings() {
    let mut graph = Graph::new_publishing();
    graph.create_branch("empty".into()).unwrap();
    let first = initial(&mut graph);
    graph.create_branch("other".into()).unwrap();
    graph
        .writer("test")
        .with_effective_at(Some(20))
        .set_node_attr(0, "v".into(), V::Int(2))
        .unwrap();
    let main = graph.current_state_id();
    let old = graph.read_view_at(20, None).unwrap();
    graph.checkout_branch("other".into()).unwrap();
    assert_eq!(
        graph
            .snapshot_at_timestamp(u64::MAX)
            .unwrap()
            .lineage()
            .commit_id,
        first
    );
    graph
        .writer("test")
        .with_effective_at(Some(30))
        .unset_node_attr(0, "v".into())
        .unwrap();
    let other = graph.current_state_id();
    let allocated = graph.try_add_node().unwrap();
    graph
        .writer("test")
        .set_node_external_id(allocated, "side-only".into())
        .unwrap();
    graph.checkout_branch("main".into()).unwrap();
    assert_eq!(graph.current_state_id(), main);
    let recorded = graph.snapshot_at_timestamp(u64::MAX).unwrap();
    assert_eq!(recorded.lineage().commit_id, main);
    assert_eq!(recorded.node_attr(0, &"v".into()), Some(V::Int(2)));
    assert!(!recorded.node_exists(allocated));
    assert_eq!(
        graph
            .read_view_at(40, None)
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        Some(V::Int(2))
    );
    assert_eq!(
        graph
            .snapshot_at_commit(other)
            .unwrap()
            .node_attr(0, &"v".into()),
        None
    );
    assert_eq!(old.node_attr(0, &"v".into()).unwrap(), Some(V::Int(2)));
    assert_eq!(
        graph.allocated_entity_for_external_id("side-only"),
        Some(groggy::ExternalEntityRef::Node(allocated))
    );
    graph.checkout_branch("empty".into()).unwrap();
    assert!(graph.snapshot_at_timestamp(u64::MAX).is_err());
    assert!(graph.is_empty());
    assert_eq!(graph.try_add_node().unwrap(), allocated + 1);
    assert_eq!(
        graph
            .list_branches()
            .iter()
            .filter(|b| b.is_current)
            .count(),
        1
    );
}
#[test]
fn owned_copy_selected_older_does_not_select_later_retained_descendants_by_timestamp() {
    let mut graph = Graph::new_publishing();
    let first = initial(&mut graph);
    graph
        .writer("test")
        .set_node_attr(0, "v".into(), V::Int(8))
        .unwrap();
    let later = graph.current_state_id();
    let copied = graph
        .owned_snapshot_at_commit(first)
        .unwrap()
        .to_graph()
        .unwrap();
    assert_eq!(
        copied
            .snapshot_at_timestamp(u64::MAX)
            .unwrap()
            .lineage()
            .commit_id,
        first
    );
    assert_eq!(
        copied
            .snapshot_at_commit(later)
            .unwrap()
            .node_attr(0, &"v".into()),
        Some(V::Int(8))
    );
}
fn metadata(seconds: i64) -> Option<PublicationMetadata> {
    Some(PublicationMetadata {
        recorded_at: seconds * 1_000_000,
        effective_at: seconds * 1_000_000,
        change_times: Default::default(),
    })
}
fn change(old: Option<V>, new: V) -> ChangeSet {
    let mut change = ChangeSet::new();
    change.node_attr_changes = vec![(0, "v".into(), old, Some(new))];
    change
}
#[test]
fn recorded_selector_uses_first_parent_merge_lineage_and_exact_timestamp_ties() {
    let mut history = HistoryForest::new();
    let mut base = change(None, V::Int(0));
    base.nodes_added = vec![0];
    let root = history
        .create_commit_with_publication(base, "root".into(), "test".into(), None, metadata(10))
        .unwrap();
    let left = history
        .create_commit_with_publication(
            change(Some(V::Int(0)), V::Int(1)),
            "left".into(),
            "test".into(),
            Some(root),
            metadata(20),
        )
        .unwrap();
    let right = history
        .create_commit_with_publication(
            change(Some(V::Int(0)), V::Int(9)),
            "right".into(),
            "test".into(),
            Some(root),
            metadata(30),
        )
        .unwrap();
    let merged = history
        .create_merge_commit_with_publication(
            change(Some(V::Int(1)), V::Int(2)),
            "resolved".into(),
            "test".into(),
            vec![left, right],
            metadata(40),
        )
        .unwrap();
    assert_eq!(history.commit_at_or_before(35), Some(right));
    assert_eq!(
        history.commit_at_or_before_on_lineage(merged, 35).unwrap(),
        Some(left)
    );
    assert_eq!(
        TemporalSnapshot::at_timestamp_on_lineage(&history, merged, 35)
            .unwrap()
            .node_attr(0, &"v".into()),
        Some(V::Int(1))
    );
    let tie = history
        .create_commit_with_publication(
            change(Some(V::Int(2)), V::Int(3)),
            "tie".into(),
            "test".into(),
            Some(merged),
            metadata(40),
        )
        .unwrap();
    assert_eq!(
        history.commit_at_or_before_on_lineage(tie, 40).unwrap(),
        Some(tie)
    );
    assert_eq!(
        history.commit_at_or_before_on_lineage(merged, 40).unwrap(),
        Some(merged)
    );
    assert_eq!(
        history.commit_at_or_before_on_lineage(0, u64::MAX).unwrap(),
        None
    );
    assert!(history.commit_at_or_before_on_lineage(999, 40).is_err());
    assert!(TemporalSnapshot::at_timestamp_on_lineage(&history, merged, 5).is_err());
}
#[test]
fn duplicate_unknown_and_dirty_branch_operations_preserve_heads_and_pending_values() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    let first = graph.commit("first".into(), "test".into()).unwrap();
    graph.create_branch("other".into()).unwrap();
    assert!(graph.create_branch("other".into()).is_err());
    assert!(graph.checkout_branch("unknown".into()).is_err());
    graph
        .set_node_attr(node, "pending".into(), V::Int(3))
        .unwrap();
    assert!(graph.checkout_branch("other".into()).is_err());
    assert_eq!(graph.current_state_id(), first);
    assert_eq!(
        graph.get_node_attr(node, &"pending".into()).unwrap(),
        Some(V::Int(3))
    );
    graph.reset_hard().unwrap();
    graph.checkout_branch("other".into()).unwrap();
    assert_eq!(graph.current_state_id(), first);
    assert_eq!(graph.get_node_attr(node, &"pending".into()).unwrap(), None);
}
