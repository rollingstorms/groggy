use groggy::{
    query::ActivationInterval as I, AttrValue as V, BatchEdgeRef as E, BatchNodeRef as N,
    ChangeKey as K, Graph, GraphBatch, GraphPublicationOptions, StateRestoreMode,
};
fn publish(g: &mut Graph, b: GraphBatch) -> u64 {
    g.apply_batch(b, g.current_state_id(), "test".into(), "test".into())
        .unwrap()
        .state_id
}
fn fixture() -> (Graph, u64) {
    let mut g = Graph::new();
    let first = publish(
        &mut g,
        GraphBatch {
            add_nodes: 2,
            add_edges: vec![(N::Added(0), N::Added(1))],
            node_attrs: [("x".into(), vec![(N::Added(0), V::Int(1))])].into(),
            effective_at: Some(10),
            ..Default::default()
        },
    );
    (g, first)
}
fn update(g: &mut Graph, t: i64, v: V) -> u64 {
    publish(
        g,
        GraphBatch {
            node_attrs: [("x".into(), vec![(N::Existing(0), v)])].into(),
            effective_at: Some(t),
            ..Default::default()
        },
    )
}
fn restore(g: &mut Graph, target: u64, at: i64) {
    let out = g
        .prepare_state_restore(target, g.current_state_id(), StateRestoreMode::Tracked)
        .unwrap()
        .execute()
        .unwrap();
    g.finish_state_restore_with_options(
        out,
        GraphPublicationOptions {
            effective_at: Some(at),
            ..Default::default()
        },
        "restore".into(),
        "test".into(),
        None,
    )
    .unwrap();
}
#[test]
fn effective_index_reuses_canonical_reads_and_matches_metadata_window_reference() {
    let (mut g, first) = fixture();
    let mut ids = vec![first];
    for (at, value) in [(30, V::Int(3)), (20, V::Int(2)), (20, V::Null)] {
        ids.push(update(&mut g, at, value));
    }
    ids.push(publish(
        &mut g,
        GraphBatch {
            unset_node_attrs: [("x".into(), vec![N::Existing(0)])].into(),
            effective_at: Some(25),
            ..Default::default()
        },
    ));
    let index = g.build_effective_index().unwrap();
    assert_eq!(index.captured_head(), g.current_state_id());
    assert_eq!(index.branch(), "main");
    let mut expected = Vec::new();
    for &state in &ids {
        let m = g.publication_metadata(state).unwrap().unwrap();
        for (key, &at) in &m.change_times {
            expected.push((at, state, key.clone(), m.recorded_at));
        }
    }
    assert_eq!(index.event_count(), expected.len());
    let cutoffs = [
        None,
        Some(i64::MIN),
        Some(g.publication_metadata(first).unwrap().unwrap().recorded_at),
        Some(i64::MAX),
    ];
    for cutoff in cutoffs {
        for start in [i64::MIN, 10, 20, 25, 40] {
            for end in [start, 30, i64::MAX]
                .into_iter()
                .filter(|end| *end >= start)
            {
                let actual = index.events_between(start, end, cutoff).unwrap();
                let mut rows: Vec<_> = actual
                    .iter()
                    .map(|e| (e.effective_at, e.state_id, e.change.clone(), e.recorded_at))
                    .collect();
                rows.sort();
                let mut reference: Vec<_> = expected
                    .iter()
                    .filter(|(at, _, _, recorded)| {
                        *at >= start && *at <= end && cutoff.is_none_or(|u| *recorded <= u)
                    })
                    .cloned()
                    .collect();
                reference.sort();
                assert_eq!(rows, reference);
                assert_eq!(
                    index.node_events(0, start, end, cutoff).unwrap(),
                    actual
                        .iter()
                        .filter(|e| matches!(
                            e.change,
                            K::NodeAdded(0) | K::NodeRemoved(0) | K::NodeAttribute(0, _)
                        ))
                        .cloned()
                        .collect::<Vec<_>>()
                );
            }
        }
        for at in [i64::MIN, 9, 10, 19, 20, 24, 25, 30, i64::MAX] {
            let a = index.read_view_at(at, cutoff).unwrap();
            let b = g.read_view_at(at, cutoff).unwrap();
            assert_eq!(a.node_ids(), b.node_ids());
            assert_eq!(a.edge_ids(), b.edge_ids());
            assert_eq!(
                a.read_state().snapshot().node_attributes,
                b.read_state().snapshot().node_attributes
            );
            assert_eq!(a.read_state().read_time(), b.read_state().read_time());
            assert!(a.write_state_id().is_none());
        }
    }
    let old = index.events_between(i64::MIN, i64::MAX, None).unwrap();
    update(&mut g, 20, V::Int(99));
    assert_eq!(index.events_between(i64::MIN, i64::MAX, None).unwrap(), old);
    assert_ne!(
        g.build_effective_index().unwrap().event_count(),
        index.event_count()
    );
    assert!(index.events_between(1, 0, None).is_err());
    assert!(index.node_events(999, 1, 0, None).is_err());
}
#[test]
fn activation_intervals_handle_restoration_same_time_ties_and_recorded_cutoffs() {
    for restore_at in [40, 50] {
        let (mut g, first) = fixture();
        let dead_edge = publish(
            &mut g,
            GraphBatch {
                remove_edges: vec![E::Existing(0)],
                effective_at: Some(35),
                ..Default::default()
            },
        );
        publish(
            &mut g,
            GraphBatch {
                remove_nodes: vec![N::Existing(0)],
                effective_at: Some(40),
                ..Default::default()
            },
        );
        let before = g
            .publication_metadata(dead_edge)
            .unwrap()
            .unwrap()
            .recorded_at;
        restore(&mut g, first, restore_at);
        let index = g.build_effective_index().unwrap();
        let expected = if restore_at == 40 {
            vec![I {
                start: 10,
                end: None,
            }]
        } else {
            vec![
                I {
                    start: 10,
                    end: Some(40),
                },
                I {
                    start: 50,
                    end: None,
                },
            ]
        };
        assert_eq!(index.node_activation_intervals(0, None), expected);
        assert_eq!(
            index.edge_activation_intervals(0, None),
            vec![
                I {
                    start: 10,
                    end: Some(35)
                },
                I {
                    start: restore_at,
                    end: None
                }
            ]
        );
        assert_eq!(
            index.node_activation_intervals(0, Some(before)),
            vec![I {
                start: 10,
                end: None
            }]
        );
        assert!(index
            .node_activation_intervals(0, Some(i64::MIN))
            .is_empty());
        assert!(index.edge_activation_intervals(999, None).is_empty());
        for time in [9, 10, 34, 35, 39, 40, 49, 50, 60] {
            let view = index.read_view_at(time, None).unwrap();
            assert_eq!(
                view.node_ids().contains(&0),
                expected
                    .iter()
                    .any(|i| i.start <= time && i.end.is_none_or(|end| time < end))
            );
        }
    }
}
#[test]
fn captures_are_branch_scoped_thread_safe_and_builds_reject_pending_manual_changes() {
    let (mut g, _) = fixture();
    g.create_branch("sibling".into()).unwrap();
    let index = g.build_effective_index().unwrap();
    update(&mut g, 20, V::Int(2));
    let newer = g.build_effective_index().unwrap();
    g.checkout_branch("sibling".into()).unwrap();
    assert_eq!(
        g.build_effective_index()
            .unwrap()
            .read_view_at(25, None)
            .unwrap()
            .node_attr(0, &"x".into())
            .unwrap(),
        Some(V::Int(1))
    );
    drop(g);
    std::thread::spawn(move || {
        assert_eq!(
            index
                .read_view_at(25, None)
                .unwrap()
                .node_attr(0, &"x".into())
                .unwrap(),
            Some(V::Int(1))
        );
        assert_eq!(
            newer
                .read_view_at(25, None)
                .unwrap()
                .node_attr(0, &"x".into())
                .unwrap(),
            Some(V::Int(2))
        );
    })
    .join()
    .unwrap();
    let mut manual = Graph::new_manual();
    manual.add_node().unwrap();
    assert!(manual.build_effective_index().is_err());
    let empty = Graph::new().build_effective_index().unwrap();
    assert_eq!(empty.event_count(), 0);
    assert!(empty
        .read_view_at(i64::MIN, None)
        .unwrap()
        .node_ids()
        .is_empty());
}

#[test]
fn property_intervals_match_point_reads_across_null_absence_ties_and_lifetimes() {
    let (mut g, first) = fixture();
    update(&mut g, 30, V::Int(3));
    update(&mut g, 20, V::Int(2));
    update(&mut g, 20, V::Null);
    publish(
        &mut g,
        GraphBatch {
            unset_node_attrs: [("x".into(), vec![N::Existing(0)])].into(),
            effective_at: Some(25),
            ..Default::default()
        },
    );
    publish(
        &mut g,
        GraphBatch {
            remove_nodes: vec![N::Existing(0)],
            effective_at: Some(40),
            ..Default::default()
        },
    );
    restore(&mut g, first, 50);
    let index = g.build_effective_index().unwrap();
    let rows = index.node_property_intervals(0, "x", None).unwrap();
    let tuples: Vec<_> = rows
        .iter()
        .map(|r| (r.start, r.end, r.value.clone()))
        .collect();
    assert_eq!(
        tuples,
        vec![
            (10, Some(20), Some(V::Int(1))),
            (20, Some(25), Some(V::Null)),
            (25, Some(30), None),
            (30, Some(40), Some(V::Int(3))),
            (50, None, Some(V::Int(1))),
        ]
    );
    let cutoffs = [
        None,
        Some(i64::MIN),
        Some(g.publication_metadata(first).unwrap().unwrap().recorded_at),
    ];
    for cutoff in cutoffs {
        for node in [true, false] {
            for name in ["x", "never_assigned"] {
                let rows = if node {
                    index.node_property_intervals(0, name, cutoff)
                } else {
                    index.edge_property_intervals(0, name, cutoff)
                }
                .unwrap();
                for at in [
                    i64::MIN,
                    9,
                    10,
                    19,
                    20,
                    24,
                    25,
                    29,
                    30,
                    39,
                    40,
                    49,
                    50,
                    i64::MAX,
                ] {
                    let view = index.read_view_at(at, cutoff).unwrap();
                    let found = rows
                        .iter()
                        .find(|r| r.start <= at && r.end.is_none_or(|end| at < end));
                    let active = if node {
                        view.nodes().contains(&0)
                    } else {
                        view.edges().contains(&0)
                    };
                    assert_eq!(found.is_some(), active);
                    if let Some(row) = found {
                        let value = if node {
                            view.node_attr(0, &name.into())
                        } else {
                            view.edge_attr(0, &name.into())
                        }
                        .unwrap();
                        assert_eq!(row.value, value);
                    }
                }
            }
        }
    }
    assert!(index
        .node_property_intervals(999, "x", None)
        .unwrap()
        .is_empty());
    assert!(index
        .edge_property_intervals(999, "x", None)
        .unwrap()
        .is_empty());
}

#[test]
fn property_intervals_ignore_inactive_writes_and_preserve_exact_float_payloads() {
    use groggy::BatchChangeRef as C;
    let mut g = Graph::new();
    publish(
        &mut g,
        GraphBatch {
            add_nodes: 1,
            node_attrs: [("x".into(), vec![(N::Added(0), V::Int(1))])].into(),
            effective_at: Some(i64::MIN),
            change_times: vec![
                (C::NodeAdded(N::Added(0)), 10),
                (C::NodeAttribute(N::Added(0), "x".into()), i64::MIN),
            ],
            ..Default::default()
        },
    );
    let absent = g
        .build_effective_index()
        .unwrap()
        .node_property_intervals(0, "x", None)
        .unwrap();
    assert_eq!(absent.len(), 1);
    assert_eq!(
        (absent[0].start, absent[0].end, &absent[0].value),
        (10, None, &None)
    );
    let values = [
        V::Float64(0.),
        V::Float64(-0.),
        V::Float(0.),
        V::Float64(f64::from_bits(0x7ff8_0000_0000_0001)),
        V::Float64(f64::from_bits(0x7ff8_0000_0000_0002)),
    ];
    for (i, value) in values.iter().enumerate() {
        update(&mut g, 20 + i as i64, value.clone());
    }
    let index = g.build_effective_index().unwrap();
    let rows = index.node_property_intervals(0, "x", None).unwrap();
    assert_eq!(rows.len(), 6);
    for (row, value) in rows[1..].iter().zip(&values) {
        match (&row.value, value) {
            (Some(V::Float64(a)), V::Float64(b)) => assert_eq!(a.to_bits(), b.to_bits()),
            (Some(V::Float(a)), V::Float(b)) => assert_eq!(a.to_bits(), b.to_bits()),
            _ => panic!("Unexpected type"),
        }
    }
}

#[test]
fn property_intervals_coalesce_equal_time_reactivation_and_extreme_bounds() {
    for restored_at in [40, 50] {
        let (mut g, first) = fixture();
        publish(
            &mut g,
            GraphBatch {
                remove_nodes: vec![N::Existing(0)],
                effective_at: Some(40),
                ..Default::default()
            },
        );
        restore(&mut g, first, restored_at);
        let index = g.build_effective_index().unwrap();
        let rows = index.node_property_intervals(0, "x", None).unwrap();
        assert_eq!(rows.len(), if restored_at == 40 { 1 } else { 2 });
        assert_eq!(rows[0].end, if restored_at == 40 { None } else { Some(40) });
        let missing = index.node_property_intervals(0, "missing", None).unwrap();
        assert_eq!(missing.len(), rows.len());
        assert!(missing.iter().all(|r| r.value.is_none()));
    }
    let mut g = Graph::new();
    publish(
        &mut g,
        GraphBatch {
            add_nodes: 1,
            effective_at: Some(i64::MIN),
            ..Default::default()
        },
    );
    update(&mut g, i64::MAX, V::Int(7));
    let index = g.build_effective_index().unwrap();
    let rows = index.node_property_intervals(0, "x", None).unwrap();
    assert_eq!((rows[0].start, rows[0].end), (i64::MIN, Some(i64::MAX)));
    assert_eq!((rows[1].start, rows[1].end), (i64::MAX, None));
}
