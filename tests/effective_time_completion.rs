use groggy::{AttrValue as V, BatchNodeRef as N, Graph, GraphBatch, PublicationMetadata};
use std::collections::{BTreeMap, HashMap};
fn initial(graph: &mut Graph) {
    graph
        .apply_batch(
            GraphBatch {
                add_nodes: 2,
                add_edges: vec![
                    (N::Added(0), N::Added(1)),
                    (N::Added(0), N::Added(1)),
                    (N::Added(0), N::Added(0)),
                ],
                node_attrs: HashMap::from([(
                    "name".into(),
                    vec![(N::Added(0), V::Text("original".into()))],
                )]),
                effective_at: Some(10),
                ..Default::default()
            },
            0,
            "initial".into(),
            "test".into(),
        )
        .unwrap();
}
#[test]
fn frozen_group_revisions_keep_source_properties_topology_and_nested_referents() {
    let mut graph = Graph::new_publishing();
    initial(&mut graph);
    let source = graph.read_view().unwrap();
    let first = graph.retain_group(&source, "inner".into()).unwrap();
    let meta = graph.create_meta_node(first).unwrap();
    let born = graph
        .publication_metadata(graph.current_state_id())
        .unwrap()
        .unwrap()
        .effective_at;
    let old = graph.read_view_at(born, None).unwrap();
    assert_eq!(
        old.meta_subgraph(meta)
            .unwrap()
            .node_attr(0, &"name".into())
            .unwrap(),
        Some(V::Text("original".into()))
    );
    graph
        .writer("test")
        .set_node_attr(0, "name".into(), V::Text("changed".into()))
        .unwrap();
    let second = graph
        .retain_group(
            &graph.read_view().unwrap().select_nodes(&[0, 1]).unwrap(),
            "inner".into(),
        )
        .unwrap();
    assert_ne!(first, second);
    graph
        .apply_batch(
            GraphBatch {
                node_attrs: HashMap::from([(
                    "contains_subgraph".into(),
                    vec![(N::Existing(meta), V::SubgraphRef(second))],
                )]),
                effective_at: Some(born + 20),
                ..Default::default()
            },
            graph.current_state_id(),
            "revision".into(),
            "test".into(),
        )
        .unwrap();
    let middle = graph.read_view_at(born + 19, None).unwrap();
    assert_eq!(
        middle
            .meta_subgraph(meta)
            .unwrap()
            .node_attr(0, &"name".into())
            .unwrap(),
        Some(V::Text("original".into()))
    );
    let latest = graph.read_view_at(born + 20, None).unwrap();
    assert_eq!(
        latest
            .meta_subgraph(meta)
            .unwrap()
            .node_attr(0, &"name".into())
            .unwrap(),
        Some(V::Text("changed".into()))
    );
    let outer_group = graph
        .retain_group(&latest.select_nodes(&[meta]).unwrap(), "outer".into())
        .unwrap();
    let outer = graph.create_meta_node(outer_group).unwrap();
    let retained = graph.read_view_at(i64::MAX, None).unwrap();
    graph.remove_node(0).unwrap();
    drop(graph);
    assert_eq!(
        retained
            .meta_subgraph(outer)
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .edge_count(),
        3
    );
    assert_eq!(
        old.meta_subgraph(meta)
            .unwrap()
            .node_attr(0, &"name".into())
            .unwrap(),
        Some(V::Text("original".into()))
    );
}
#[test]
fn legacy_clocks_are_explicit_read_local_and_cannot_override_precise_publications() {
    let mut graph = Graph::new();
    graph.add_node();
    graph.set_node_attr(0, "v".into(), V::Int(1)).unwrap();
    let first = graph.commit("old".into(), "test".into()).unwrap();
    graph.set_node_attr(0, "v".into(), V::Null).unwrap();
    let second = graph.commit("old2".into(), "test".into()).unwrap();
    let times = BTreeMap::from([
        (
            first,
            PublicationMetadata {
                effective_at: 10,
                recorded_at: 100,
                change_times: Default::default(),
            },
        ),
        (
            second,
            PublicationMetadata {
                effective_at: 20,
                recorded_at: 200,
                change_times: Default::default(),
            },
        ),
    ]);
    assert!(graph.read_view_at(15, None).is_err());
    let view = graph
        .read_view_at_with_legacy(15, None, times.clone())
        .unwrap();
    assert_eq!(view.node_attr(0, &"v".into()).unwrap(), Some(V::Int(1)));
    assert_eq!(
        graph
            .read_view_at_with_legacy(25, Some(150), times.clone())
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        Some(V::Int(1))
    );
    assert_eq!(
        graph
            .read_view_at_with_legacy(25, None, times)
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        Some(V::Null)
    );
    assert!(graph.publication_metadata(first).unwrap().is_none());
    assert_eq!(
        view.read_state().read_time().unwrap().legacy_clocks.len(),
        2
    );
    let mut precise = Graph::new_publishing();
    initial(&mut precise);
    assert!(precise
        .prepare_read_at_with_legacy(
            15,
            None,
            BTreeMap::from([(
                1,
                PublicationMetadata {
                    effective_at: 0,
                    recorded_at: 0,
                    change_times: Default::default()
                }
            )])
        )
        .is_err());
}
#[test]
fn durable_frozen_nested_groups_reopen_and_backup_without_the_source_graph() {
    let mut source = Graph::new_publishing();
    initial(&mut source);
    let group = source
        .retain_group(&source.read_view().unwrap(), "inner".into())
        .unwrap();
    let meta = source.create_meta_node(group).unwrap();
    let frozen = source.read_view().unwrap().select_nodes(&[meta]).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("group.ggj");
    let backup = dir.path().join("backup.ggj");
    let mut store = groggy::DurableGraph::create(&path, groggy::GraphConfig::new(), None).unwrap();
    let group = store.retain_group(&frozen, "outer".into()).unwrap();
    let head = store.graph().unwrap().current_state_id();
    let outer = store.create_meta_node(group, head, Some(50)).unwrap();
    store.backup(&backup).unwrap();
    let view = store.read_view_at(50, None).unwrap();
    drop(store);
    drop(source);
    for path in [&path, &backup] {
        let store = groggy::DurableGraph::open(path).unwrap();
        let view = store.read_view_at(50, None).unwrap();
        assert_eq!(
            view.meta_subgraph(outer)
                .unwrap()
                .meta_subgraph(meta)
                .unwrap()
                .node_attr(0, &"name".into())
                .unwrap(),
            Some(V::Text("original".into()))
        );
    }
    assert_eq!(
        view.meta_subgraph(outer)
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .edge_count(),
        3
    );
}
#[test]
fn resolved_merge_replays_first_parent_and_resolution_not_sibling_changes() {
    use groggy::state::{ChangeSet, HistoryForest};
    let time = |at| {
        Some(PublicationMetadata {
            effective_at: at,
            recorded_at: at + 100,
            change_times: Default::default(),
        })
    };
    let mut history = HistoryForest::new();
    let mut base = ChangeSet::new();
    base.nodes_added = vec![0];
    let base = history
        .create_commit_with_publication(base, "base".into(), "test".into(), None, time(0))
        .unwrap();
    let mut left = ChangeSet::new();
    left.node_attr_changes = vec![(0, "v".into(), None, V::Int(1))];
    let left = history
        .create_commit_with_publication(left, "left".into(), "test".into(), Some(base), time(10))
        .unwrap();
    let mut right = ChangeSet::new();
    right.nodes_added = vec![99];
    right.node_attr_changes = vec![(0, "v".into(), None, V::Int(9))];
    let right = history
        .create_commit_with_publication(right, "right".into(), "test".into(), Some(base), time(10))
        .unwrap();
    let mut resolution = ChangeSet::new();
    resolution.node_attr_changes = vec![(0, "v".into(), Some(V::Int(1)), V::Int(2))];
    let merged = history
        .create_merge_commit_with_publication(
            resolution,
            "resolved".into(),
            "test".into(),
            vec![left, right],
            time(20),
        )
        .unwrap();
    let view = history
        .prepare_effective_read(merged, 25, None, BTreeMap::new())
        .unwrap()
        .read_view()
        .unwrap();
    assert_eq!(view.node_ids(), vec![0]);
    assert_eq!(view.node_attr(0, &"v".into()).unwrap(), Some(V::Int(2)));
    assert_eq!(
        history
            .prepare_effective_read(merged, 15, None, BTreeMap::new())
            .unwrap()
            .read_view()
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        Some(V::Int(1))
    );
}

#[test]
fn membership_only_legacy_groups_require_explicit_source_and_roundtrip_aliases() {
    use groggy::state::effective_read::EffectiveReadOptions;
    let mut graph = Graph::new_publishing();
    initial(&mut graph);
    let source = graph.read_view_at(15, None).unwrap();
    let legacy = graph
        .pool_mut()
        .store_subgraph(source.nodes().clone(), source.edges().clone(), "old".into())
        .unwrap();
    let meta = graph.create_meta_node_at(legacy, Some(20)).unwrap();
    assert!(graph.read_view_at(25, None).is_err());
    let options = EffectiveReadOptions {
        group_sources: BTreeMap::from([(legacy, source.clone())]),
        ..Default::default()
    };
    let view = graph
        .prepare_effective_read(25, options)
        .unwrap()
        .read_view()
        .unwrap();
    assert_eq!(view.meta_subgraph(meta).unwrap().edge_count(), 3);
    assert_eq!(
        view.node_attr(meta, &"contains_subgraph".into()).unwrap(),
        Some(V::SubgraphRef(legacy))
    );
    assert_eq!(
        view.read_state().read_time().unwrap().group_sources.len(),
        1
    );
    assert!(graph
        .prepare_effective_read(
            25,
            EffectiveReadOptions {
                group_sources: BTreeMap::from([(legacy, source.select_nodes(&[0]).unwrap())]),
                ..Default::default()
            }
        )
        .is_err());
    let pinned = graph.pin_legacy_group(legacy, &source).unwrap();
    assert_ne!(legacy, pinned);
    assert!(graph
        .prepare_effective_read(
            25,
            EffectiveReadOptions {
                group_sources: BTreeMap::from([(pinned, source)]),
                ..Default::default()
            }
        )
        .is_err());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy-pins.ggj");
    let mut store = groggy::DurableGraph::create(&path, groggy::GraphConfig::new(), None).unwrap();
    let group = store
        .retain_group(&view.select_nodes(&[meta]).unwrap(), "pinned legacy".into())
        .unwrap();
    let outer = store.create_meta_node(group, 0, Some(30)).unwrap();
    drop(store);
    drop(graph);
    let store = groggy::DurableGraph::open(&path).unwrap();
    assert_eq!(
        store
            .read_view_at(30, None)
            .unwrap()
            .meta_subgraph(outer)
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .edge_count(),
        3
    );
}

#[test]
fn unknown_group_references_reject_before_ids_and_cannot_be_fixed_by_later_pool_writes() {
    let mut graph = Graph::new_publishing();
    let batch = GraphBatch {
        add_nodes: 1,
        node_attrs: HashMap::from([("group".into(), vec![(N::Added(0), V::SubgraphRef(777))])]),
        effective_at: Some(10),
        ..Default::default()
    };
    assert!(graph
        .apply_batch(batch, 0, "invalid".into(), "test".into())
        .is_err());
    assert_eq!(graph.current_state_id(), 0);
    assert_eq!(graph.insert_nodes(1).unwrap(), vec![0]);
}
