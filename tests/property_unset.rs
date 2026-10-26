use groggy::{
    AttrValue as V, BatchChangeRef as C, BatchEdgeRef as E, BatchNodeRef as N, Graph, GraphBatch,
};
use std::collections::HashMap;
fn publish(graph: &mut Graph, batch: GraphBatch) -> usize {
    graph
        .apply_batch(
            batch,
            graph.current_state_id(),
            "test".into(),
            "test".into(),
        )
        .unwrap()
        .state_id as usize
}
fn initial(graph: &mut Graph) {
    publish(
        graph,
        GraphBatch {
            add_nodes: 2,
            add_edges: vec![(N::Added(0), N::Added(1))],
            node_attrs: HashMap::from([("v".into(), vec![(N::Added(0), V::Null)])]),
            edge_attrs: HashMap::from([(
                "v".into(),
                vec![(E::Added(0), V::Float64Vec(vec![1.0, 2.0]))],
            )]),
            effective_at: Some(10),
            ..Default::default()
        },
    );
}
#[test]
fn absence_is_a_timed_event_distinct_from_null_and_retains_old_views() {
    let mut graph = Graph::new_publishing();
    initial(&mut graph);
    let before = graph.read_view_at(10, None).unwrap();
    let state = publish(
        &mut graph,
        GraphBatch {
            unset_node_attrs: HashMap::from([("v".into(), vec![N::Existing(0)])]),
            unset_edge_attrs: HashMap::from([("v".into(), vec![E::Existing(0)])]),
            effective_at: Some(20),
            change_times: vec![(C::EdgeAttribute(E::Existing(0), "v".into()), 30)],
            ..Default::default()
        },
    ) as u64;
    assert_eq!(before.node_attr(0, &"v".into()).unwrap(), Some(V::Null));
    assert_eq!(
        graph
            .read_view_at(19, None)
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        Some(V::Null)
    );
    assert_eq!(
        graph
            .read_view_at(20, None)
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        None
    );
    assert_eq!(
        graph
            .read_view_at(29, None)
            .unwrap()
            .edge_attr(0, &"v".into())
            .unwrap(),
        Some(V::Float64Vec(vec![1.0, 2.0]))
    );
    assert_eq!(
        graph
            .read_view_at(30, None)
            .unwrap()
            .edge_attr(0, &"v".into())
            .unwrap(),
        None
    );
    let index = graph.build_temporal_index().unwrap();
    assert_eq!(
        index.node_attr_history(0, &"v".into(), 1, state),
        vec![(1, Some(V::Null)), (state, None)]
    );
    assert_eq!(index.node_attr_at_commit(0, &"v".into(), state), None);
    assert_eq!(
        graph
            .snapshot_at_commit(state)
            .unwrap()
            .read_state()
            .snapshot()
            .node_attributes
            .get(&0)
            .and_then(|row| row.get("v")),
        None
    );
    graph
        .writer("test")
        .with_effective_at(Some(40))
        .set_node_attr(0, "v".into(), V::Int(9))
        .unwrap();
    assert_eq!(
        graph
            .read_view_at(39, None)
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        None
    );
    assert_eq!(
        graph
            .read_view_at(40, None)
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        Some(V::Int(9))
    );
}
#[test]
fn bulk_scalar_and_unset_coalesce_in_operation_order_in_manual_mode() {
    let mut graph = Graph::new_manual();
    let node = graph.add_node().unwrap();
    graph.set_node_attr(node, "v".into(), V::Int(1)).unwrap();
    graph.commit("initial".into(), "test".into()).unwrap();
    graph
        .set_node_attrs(HashMap::from([("v".into(), vec![(node, V::Int(2))])]))
        .unwrap();
    graph
        .set_node_attrs(HashMap::from([("v".into(), vec![(node, V::Int(3))])]))
        .unwrap();
    graph.unset_node_attr(node, "v".into()).unwrap();
    graph.set_node_attr(node, "v".into(), V::Null).unwrap();
    let state = graph.commit("changes".into(), "test".into()).unwrap();
    let index = graph.build_temporal_index().unwrap();
    assert_eq!(
        index.node_attr_history(node, &"v".into(), state, state),
        vec![(state, Some(V::Null))]
    );
    assert_eq!(
        graph
            .snapshot_at_commit(state)
            .unwrap()
            .read_state()
            .snapshot()
            .node_attributes[&node]["v"],
        V::Null
    );
    graph.create_branch("before_unset".into()).unwrap();
    graph.unset_node_attr(node, "v".into()).unwrap();
    let removed = graph.commit("unset".into(), "test".into()).unwrap();
    graph.checkout_branch("before_unset".into()).unwrap();
    assert_eq!(
        graph.get_node_attr(node, &"v".into()).unwrap(),
        Some(V::Null)
    );
    graph.checkout_branch("main".into()).unwrap();
    assert_eq!(graph.current_state_id(), removed);
    assert_eq!(graph.get_node_attr(node, &"v".into()).unwrap(), None);
}
#[test]
fn mixed_invalid_unsets_and_assignments_are_atomic_and_allocate_no_ids() {
    let mut graph = Graph::new_publishing();
    initial(&mut graph);
    let head = graph.current_state_id();
    for batch in [
        GraphBatch {
            add_nodes: 1,
            unset_node_attrs: HashMap::from([("v".into(), vec![N::Existing(0), N::Existing(99)])]),
            ..Default::default()
        },
        GraphBatch {
            node_attrs: HashMap::from([("v".into(), vec![(N::Existing(0), V::Int(2))])]),
            unset_node_attrs: HashMap::from([("v".into(), vec![N::Existing(0)])]),
            ..Default::default()
        },
        GraphBatch {
            unset_node_attrs: HashMap::from([("entity_type".into(), vec![N::Existing(0)])]),
            ..Default::default()
        },
    ] {
        assert!(graph
            .apply_batch(batch, head, "bad".into(), "test".into())
            .is_err());
        assert_eq!(graph.current_state_id(), head);
        assert_eq!(graph.get_node_attr(0, &"v".into()).unwrap(), Some(V::Null));
        assert!(!graph.has_uncommitted_changes());
    }
    assert_eq!(graph.try_add_node().unwrap(), 2);
}
#[test]
fn later_backdated_absence_uses_accepted_order_and_explicit_recorded_cutoff() {
    let mut graph = Graph::new_publishing();
    initial(&mut graph);
    let first = graph.publication_metadata(1).unwrap().unwrap().recorded_at;
    graph
        .writer("test")
        .with_effective_at(Some(10))
        .unset_node_attr(0, "v".into())
        .unwrap();
    assert_eq!(
        graph
            .read_view_at(10, None)
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        None
    );
    assert_eq!(
        graph
            .read_view_at(10, Some(first))
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        Some(V::Null)
    );
    // An absent current mapping must still record a backdated absence event.
    graph
        .writer("test")
        .with_effective_at(Some(40))
        .set_node_attr(0, "v".into(), V::Int(4))
        .unwrap();
    graph
        .writer("test")
        .with_effective_at(Some(20))
        .unset_node_attr(0, "v".into())
        .unwrap();
    graph
        .writer("test")
        .with_effective_at(Some(30))
        .unset_node_attr(0, "v".into())
        .unwrap();
    assert_eq!(
        graph
            .read_view_at(35, None)
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        None
    );
    assert_eq!(
        graph
            .read_view_at(40, None)
            .unwrap()
            .node_attr(0, &"v".into())
            .unwrap(),
        Some(V::Int(4))
    );
}

#[test]
fn every_property_representation_keeps_typed_history_after_unset() {
    use groggy::types::{CalendarDate, CompactString, CompressedData, FixedDuration, UtcInstant};
    let mut graph = Graph::new_publishing();
    initial(&mut graph);
    let group = graph
        .retain_group(&graph.read_view().unwrap(), "source".into())
        .unwrap();
    let values = vec![
        V::Null,
        V::Int(i64::MAX),
        V::SmallInt(-7),
        V::Float(f32::from_bits(0xffc00001)),
        V::Float64(f64::from_bits(0xfff8000000000001)),
        V::Bool(true),
        V::Text("index_0".into()),
        V::CompactText(CompactString::new("compact")),
        V::CompressedText(CompressedData::compress_text(&"text".repeat(100))),
        V::FloatVec(vec![1.0, -0.0]),
        V::Float64Vec(vec![1.0, -0.0]),
        V::CompressedFloatVec(CompressedData::compress_float_vec(&[1.0, -0.0])),
        V::IntVec(vec![i64::MIN]),
        V::TextVec(vec!["a".into()]),
        V::BoolVec(vec![true, false]),
        V::Bytes(vec![0, 255]),
        V::Json("{\"v\":null}".into()),
        V::NodeArray(vec![0, 1]),
        V::EdgeArray(vec![0]),
        V::SubgraphRef(group),
        V::UtcInstant(UtcInstant::new(-1, 999999).unwrap()),
        V::Date(CalendarDate::new(2026, 10, 25).unwrap()),
        V::Duration(FixedDuration::new(-1, 0, 7).unwrap()),
    ];
    let columns: HashMap<_, _> = values
        .iter()
        .enumerate()
        .map(|(i, v)| (format!("p{i}"), vec![(N::Existing(0), v.clone())]))
        .collect();
    publish(
        &mut graph,
        GraphBatch {
            node_attrs: columns,
            effective_at: Some(20),
            ..Default::default()
        },
    );
    let old = graph.read_view_at(20, None).unwrap();
    let removed = publish(
        &mut graph,
        GraphBatch {
            unset_node_attrs: values
                .iter()
                .enumerate()
                .map(|(i, _)| (format!("p{i}"), vec![N::Existing(0)]))
                .collect(),
            effective_at: Some(30),
            ..Default::default()
        },
    ) as u64;
    let index = graph.build_temporal_index().unwrap();
    for (i, value) in values.iter().enumerate() {
        let name = format!("p{i}");
        assert_eq!(old.node_attr(0, &name).unwrap(), Some(value.clone()));
        assert_eq!(
            graph
                .read_view_at(30, None)
                .unwrap()
                .node_attr(0, &name)
                .unwrap(),
            None
        );
        assert_eq!(
            index.node_attr_history(0, &name, 1, removed),
            vec![(2, Some(value.clone())), (removed, None)]
        );
    }
}

#[test]
fn assignment_only_delta_hashes_preserve_the_existing_hash_format() {
    use groggy::state::{change_tracker::ChangeSet, history::Delta};
    use std::hash::{Hash, Hasher};
    let mut changes = ChangeSet::new();
    changes.nodes_added = vec![0];
    changes.node_attr_changes = vec![(0, "v".into(), None, Some(V::Null))];
    let delta = Delta::from_changes(changes.clone());
    let mut old = std::collections::hash_map::DefaultHasher::new();
    changes.nodes_added.hash(&mut old);
    changes.nodes_removed.hash(&mut old);
    changes.edges_added.hash(&mut old);
    changes.edges_removed.hash(&mut old);
    vec![(0usize, "v".to_string(), None::<V>, V::Null)].hash(&mut old);
    Vec::<(usize, String, Option<V>, V)>::new().hash(&mut old);
    assert_eq!(&delta.content_hash[..8], &old.finish().to_le_bytes());
    changes.node_attr_changes[0].3 = None;
    assert_ne!(
        Delta::from_changes(changes).content_hash,
        delta.content_hash
    );
}
