use groggy::{AttrValue as V, ExternalEntityRef, Graph};
fn fixture() -> (Graph, usize, usize, usize, u64) {
    let mut g = Graph::new();
    g.add_node().unwrap();
    g.writer("test")
        .set_node_external_id(0, "keep".into())
        .unwrap();
    g.create_branch("discard".into()).unwrap();
    g.checkout_branch("discard".into()).unwrap();
    let node = g.add_node().unwrap();
    let edge = g.add_edge(0, node).unwrap();
    g.writer("test")
        .set_node_external_id(node, "orphan-node".into())
        .unwrap();
    g.writer("test")
        .set_edge_external_id(edge, "orphan-edge".into())
        .unwrap();
    g.set_node_attr(node, "payload".into(), V::Text("retained by pin".into()))
        .unwrap();
    let group = g
        .retain_group(&g.read_view().unwrap(), "orphan source".into())
        .unwrap();
    let gone = g.current_state_id();
    g.checkout_branch("main".into()).unwrap();
    (g, node, edge, group, gone)
}
#[test]
fn explicit_purge_preserves_retained_history_until_pruned_and_never_recycles_ids() {
    let (mut g, node, edge, group, gone) = fixture();
    let preview = g.preview_purge().unwrap();
    assert_eq!(preview.purged_nodes, 0);
    assert_eq!(preview.purged_edges, 0);
    assert!(preview.removed_groups.contains(&group));
    // Unattached registrations can be purged, while all entity history is retained.
    g.purge_unreferenced().unwrap();
    assert!(g.snapshot_at_commit(gone).is_ok());
    assert!(g.is_node_allocated(node));
    assert!(g.is_edge_allocated(edge));
    let pin = g.snapshot_at_commit(gone).unwrap();
    let owned = g.owned_snapshot_at_commit(gone).unwrap();
    g.prune_branch("discard").unwrap();
    let state = g.current_state_id();
    let before = g.preview_purge().unwrap();
    let plan = g.prepare_purge().unwrap();
    let outcome = std::thread::spawn(move || plan.execute().unwrap())
        .join()
        .unwrap();
    assert_eq!(outcome.report(), &before);
    let report = g.finish_purge(outcome).unwrap();
    assert_eq!(report, before);
    assert_eq!(report.purged_nodes, 1);
    assert_eq!(report.purged_edges, 1);
    assert_eq!(report.node_ranges, vec![(node, node + 1)]);
    assert_eq!(report.edge_ranges, vec![(edge, edge + 1)]);
    assert_eq!(g.current_state_id(), state);
    assert!(!g.is_node_allocated(node));
    assert!(!g.is_edge_allocated(edge));
    assert_eq!(g.pool().get_edge_endpoints(edge), None);
    assert_eq!(g.allocated_entity_for_external_id("orphan-node"), None);
    assert_eq!(g.allocated_entity_for_external_id("orphan-edge"), None);
    assert_eq!(
        g.allocated_entity_for_external_id("keep"),
        Some(ExternalEntityRef::Node(0))
    );
    assert_eq!(
        pin.node_attr(node, &"payload".into()),
        Some(V::Text("retained by pin".into()))
    );
    assert!(owned.to_graph().unwrap().snapshot_at_commit(gone).is_ok());
    let again = g.purge_unreferenced().unwrap();
    assert_eq!(again.purged_nodes, 0);
    assert_eq!(again.purged_edges, 0);
    let mut copy = g.prepare_archive().unwrap().restore().unwrap();
    assert!(!copy.is_node_allocated(node));
    assert!(!copy.is_edge_allocated(edge));
    let next = copy.add_node().unwrap();
    assert_eq!(next, node + 1);
    assert_eq!(copy.add_edge(0, next).unwrap(), edge + 1);
    copy.writer("test")
        .set_node_external_id(next, "orphan-node".into())
        .unwrap();
    assert_eq!(copy.node_by_external_id("orphan-node", None), Some(next));
    copy.check_integrity().unwrap();
}
#[test]
fn frozen_nested_sources_survive_owner_registration_purge_and_id_domains_do_not_pin_owner_ids() {
    let (mut g, node, edge, group, gone) = fixture();
    // Freeze a source with owner-side IDs, then keep it through a meta node on main.
    let source = groggy::GraphView::all(g.snapshot_at_commit(gone).unwrap().read_state().clone());
    let group2 = g.retain_group(&source, "used source".into()).unwrap();
    let meta = g.create_meta_node(group2).unwrap();
    let pin = g.read_view().unwrap();
    g.prune_branch("discard").unwrap();
    let report = g.purge_unreferenced().unwrap();
    assert_eq!(report.purged_nodes, 1);
    assert_eq!(report.purged_edges, 1);
    assert!(report.removed_groups.contains(&group));
    assert!(!report.removed_groups.contains(&group2));
    assert!(!g.is_node_allocated(node));
    assert!(!g.is_edge_allocated(edge));
    assert!(g.has_node(meta));
    assert_eq!(
        g.read_view()
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .node_attr(node, &"payload".into())
            .unwrap(),
        Some(V::Text("retained by pin".into()))
    );
    assert_eq!(
        pin.meta_subgraph(meta)
            .unwrap()
            .edge_endpoints(edge)
            .unwrap(),
        (0, node)
    );
    let copy = g.prepare_archive().unwrap().restore().unwrap();
    assert_eq!(
        copy.read_view()
            .unwrap()
            .meta_subgraph(meta)
            .unwrap()
            .node_attr(node, &"payload".into())
            .unwrap(),
        Some(V::Text("retained by pin".into()))
    );
    copy.check_integrity().unwrap();
}
#[test]
fn previews_do_not_mutate_and_stale_foreign_or_dirty_candidates_reject() {
    let (mut g, _, _, _, _) = fixture();
    g.prune_branch("discard").unwrap();
    let before = g.check_integrity().unwrap();
    assert_eq!(g.preview_purge().unwrap(), g.preview_purge().unwrap());
    assert_eq!(g.check_integrity().unwrap(), before);
    let result = g.prepare_purge().unwrap().execute().unwrap();
    let mut foreign = g.prepare_archive().unwrap().restore().unwrap();
    assert!(foreign.finish_purge(result).is_err());
    let result = g.prepare_purge().unwrap().execute().unwrap();
    g.add_node().unwrap();
    assert!(g.finish_purge(result).is_err());
    let mut manual = Graph::new_manual();
    manual.add_node().unwrap();
    assert!(manual.preview_purge().is_err());
    assert!(manual.purge_unreferenced().is_err());
}

#[test]
fn nested_group_payloads_remain_readable_after_their_owner_registration_is_purged() {
    let mut g = Graph::new();
    g.add_node().unwrap();
    g.create_branch("discard".into()).unwrap();
    g.checkout_branch("discard".into()).unwrap();
    let child_node = g.add_node().unwrap();
    g.set_node_attr(child_node, "v".into(), V::Int(42)).unwrap();
    let child = g
        .retain_group(&g.read_view().unwrap(), "child".into())
        .unwrap();
    let child_meta = g.create_meta_node(child).unwrap();
    let parent = g
        .retain_group(&g.read_view().unwrap(), "parent".into())
        .unwrap();
    g.checkout_branch("main".into()).unwrap();
    let parent_meta = g.create_meta_node(parent).unwrap();
    g.prune_branch("discard").unwrap();
    let pin = g.read_view().unwrap();
    let report = g.purge_unreferenced().unwrap();
    assert!(report.removed_groups.contains(&child));
    assert!(!report.removed_groups.contains(&parent));
    assert!(!g.is_node_allocated(child_node));
    assert!(!g.is_node_allocated(child_meta));
    for view in [
        g.read_view().unwrap(),
        pin,
        g.prepare_archive()
            .unwrap()
            .restore()
            .unwrap()
            .read_view()
            .unwrap(),
    ] {
        assert_eq!(
            view.meta_subgraph(parent_meta)
                .unwrap()
                .meta_subgraph(child_meta)
                .unwrap()
                .node_attr(child_node, &"v".into())
                .unwrap(),
            Some(V::Int(42))
        );
    }
    g.check_integrity().unwrap();
}

#[test]
fn only_aliases_absent_from_all_retained_identity_changes_are_released() {
    let mut g = Graph::new();
    g.add_node().unwrap();
    g.writer("test")
        .set_node_external_id(0, "original".into())
        .unwrap();
    g.writer("test")
        .set_node_external_id(0, "current".into())
        .unwrap();
    g.create_branch("discard".into()).unwrap();
    g.checkout_branch("discard".into()).unwrap();
    g.writer("test")
        .set_node_external_id(0, "ephemeral".into())
        .unwrap();
    let pin = g.read_view().unwrap();
    g.checkout_branch("main".into()).unwrap();
    assert!(!g
        .purge_unreferenced()
        .unwrap()
        .released_external_ids
        .contains(&"ephemeral".into()));
    g.prune_branch("discard").unwrap();
    let report = g.purge_unreferenced().unwrap();
    assert_eq!(report.released_external_ids, vec!["ephemeral"]);
    assert_eq!(
        g.allocated_entity_for_external_id("original"),
        Some(ExternalEntityRef::Node(0))
    );
    assert_eq!(
        g.allocated_entity_for_external_id("current"),
        Some(ExternalEntityRef::Node(0))
    );
    assert_eq!(g.allocated_entity_for_external_id("ephemeral"), None);
    assert_eq!(
        pin.node_attr(0, &"external_id".into()).unwrap(),
        Some(V::Text("ephemeral".into()))
    );
    let node = g.add_node().unwrap();
    g.writer("test")
        .set_node_external_id(node, "ephemeral".into())
        .unwrap();
    g.check_integrity().unwrap();
}

#[test]
fn unreferenced_recorded_versions_protect_entities_until_explicit_history_collection() {
    let mut g = Graph::new();
    g.add_node().unwrap();
    let root = g.current_state_id();
    let node = g.add_node().unwrap();
    let future = g.current_state_id();
    let mut detached = g
        .owned_snapshot_at_commit(root)
        .unwrap()
        .to_graph()
        .unwrap();
    assert_eq!(detached.purge_unreferenced().unwrap().purged_nodes, 0);
    assert!(detached.snapshot_at_commit(future).is_ok());
    assert!(detached.is_node_allocated(node));
    assert_eq!(detached.gc_history().unwrap(), 1);
    assert_eq!(detached.purge_unreferenced().unwrap().purged_nodes, 1);
    assert!(!detached.is_node_allocated(node));
    detached.check_integrity().unwrap();
}

#[test]
fn numeric_id_collections_preserve_their_values_without_becoming_hidden_ownership_pins() {
    let mut g = Graph::new();
    g.add_node().unwrap();
    g.create_branch("side".into()).unwrap();
    g.checkout_branch("side".into()).unwrap();
    let node = g.add_node().unwrap();
    let edge = g.add_edge(0, node).unwrap();
    g.checkout_branch("main".into()).unwrap();
    g.set_node_attr(0, "node_ids".into(), V::NodeArray(vec![node, 999]))
        .unwrap();
    g.set_node_attr(0, "edge_ids".into(), V::EdgeArray(vec![edge, 999]))
        .unwrap();
    g.prune_branch("side").unwrap();
    g.purge_unreferenced().unwrap();
    assert!(!g.is_node_allocated(node));
    assert!(!g.is_edge_allocated(edge));
    assert_eq!(
        g.get_node_attr(0, &"node_ids".into()).unwrap(),
        Some(V::NodeArray(vec![node, 999]))
    );
    assert_eq!(
        g.get_node_attr(0, &"edge_ids".into()).unwrap(),
        Some(V::EdgeArray(vec![edge, 999]))
    );
    g.check_integrity().unwrap();
}
