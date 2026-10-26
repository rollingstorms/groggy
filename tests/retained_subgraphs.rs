use groggy::{AttrValue, Graph};
use std::collections::HashSet;

#[test]
fn deactivation_preserves_stored_membership_and_retained_pool_values() {
    let mut g = Graph::new_manual();
    let nodes = g.add_nodes(2).unwrap();
    let edge = g.add_edge(nodes[0], nodes[1]).unwrap();
    g.set_node_attr(nodes[1], "name".into(), AttrValue::Text("Lin".into()))
        .unwrap();
    let old_index = g
        .space()
        .get_node_attr_index(nodes[1], &"name".into())
        .unwrap();
    let selection = g
        .pool_mut()
        .store_subgraph(
            nodes.iter().copied().collect(),
            HashSet::from([edge]),
            "group".into(),
        )
        .unwrap();
    let meta = g.create_meta_node(selection).unwrap();
    let first = g.commit("group".into(), "test".into()).unwrap();
    g.create_branch("original".into()).unwrap();
    g.remove_node(nodes[1]).unwrap();
    g.commit("remove".into(), "test".into()).unwrap();
    assert!(!g.has_node(nodes[1]));
    assert!(!g.has_edge(edge));
    let (retained_nodes, retained_edges, _) = g.pool().get_subgraph(selection).unwrap();
    assert_eq!(retained_nodes, nodes.iter().copied().collect());
    assert_eq!(retained_edges, HashSet::from([edge]));
    g.checkout_branch("original".into()).unwrap();
    assert_eq!(g.current_state_id(), first);
    assert!(g.has_node(nodes[1]));
    assert!(g.has_edge(edge));
    assert_eq!(
        g.get_node_attr(meta, &"contains_subgraph".into()).unwrap(),
        Some(AttrValue::SubgraphRef(selection))
    );
    assert_eq!(g.pool().get_subgraph(selection).unwrap().0, retained_nodes);
    // Indices owned by retained history do not become dangling during checkout.
    assert_eq!(
        g.pool()
            .get_attr_by_index(&"name".into(), old_index, true)
            .and_then(AttrValue::get_text),
        Some("Lin".into())
    );
    g.checkout_branch("main".into()).unwrap();
    assert!(!g.has_node(nodes[1]));
    assert_eq!(g.pool().get_subgraph(selection).unwrap().1, retained_edges);
}

#[test]
fn reset_keeps_other_branch_selections_endpoints_and_allocation() {
    let mut g = Graph::new_manual();
    let base = g.add_node().unwrap();
    g.commit("base".into(), "test".into()).unwrap();
    g.create_branch("base".into()).unwrap();
    let later = g.add_node().unwrap();
    let edge = g.add_edge(base, later).unwrap();
    let selection = g
        .pool_mut()
        .store_subgraph(
            HashSet::from([base, later]),
            HashSet::from([edge]),
            "group".into(),
        )
        .unwrap();
    let meta = g.create_meta_node(selection).unwrap();
    g.commit("later group".into(), "test".into()).unwrap();
    g.checkout_branch("base".into()).unwrap();
    assert!(!g.has_node(meta));
    assert_eq!(
        g.pool().get_subgraph(selection).unwrap().0,
        HashSet::from([base, later])
    );
    assert_eq!(g.pool().get_edge_endpoints(edge), Some((base, later)));
    let dirty = g.add_node().unwrap();
    g.reset_hard().unwrap();
    assert!(!g.has_node(dirty));
    assert_eq!(
        g.pool().get_subgraph(selection).unwrap().1,
        HashSet::from([edge])
    );
    assert_eq!(g.add_node().unwrap(), dirty + 1);
    g.reset_hard().unwrap();
    g.checkout_branch("main".into()).unwrap();
    assert!(g.has_node(meta));
    assert_eq!(g.edge_endpoints(edge).unwrap(), (base, later));
}

#[test]
fn missing_referents_reject_before_creation_and_detached_materialization() {
    let mut g = Graph::new_manual();
    assert!(g.create_meta_node(123).is_err());
    assert_eq!(g.node_count(), 0);
    assert!(!g.has_uncommitted_changes());
    let n = g.add_node().unwrap();
    assert_eq!(n, 0);
    let selection = g
        .pool_mut()
        .store_subgraph(HashSet::from([n]), HashSet::new(), "group".into())
        .unwrap();
    let meta = g.create_meta_node(selection).unwrap();
    let state = g.commit("group".into(), "test".into()).unwrap();
    let snapshot = g.snapshot_at_commit(state).unwrap();
    assert_eq!(
        snapshot.node_attr(meta, &"contains_subgraph".into()),
        Some(AttrValue::SubgraphRef(selection))
    );
    // Until detached pool transfer is implemented, reject rather than return broken references.
    assert!(snapshot.as_subgraph().is_err());
}
