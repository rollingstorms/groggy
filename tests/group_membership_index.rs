use groggy::{query::GroupReference as R, AttrValue as V, Graph};

fn fixture() -> (Graph, usize, usize) {
    let mut source = Graph::new();
    source.add_nodes(2).unwrap();
    source.set_node_attr(0, "source".into(), V::Int(7)).unwrap();
    source.add_edge(0, 1).unwrap();
    source.add_edge(0, 1).unwrap();
    source.add_edge(0, 0).unwrap();
    let mut parent = Graph::new();
    let child = parent
        .retain_group(&source.read_view().unwrap(), "child".into())
        .unwrap();
    parent.create_meta_node(child).unwrap();
    let mut root = Graph::new();
    let outer = root
        .retain_group(&parent.read_view().unwrap(), "parent".into())
        .unwrap();
    root.retain_group(&source.read_view().unwrap(), "child".into())
        .unwrap();
    root.create_meta_node(outer).unwrap();
    root.create_meta_node(child).unwrap();
    root.set_node_attr(0, "also".into(), V::SubgraphRef(child))
        .unwrap();
    let edge = root.add_edge(0, 1).unwrap();
    root.set_edge_attr(edge, "evidence".into(), V::SubgraphRef(child))
        .unwrap();
    (root, outer, child)
}
fn reference(entity_id: usize, property: &str) -> R {
    R {
        entity_id,
        property: property.into(),
    }
}

#[test]
fn group_index_tracks_typed_references_nested_links_and_source_scoped_membership() {
    let (root, parent, child) = fixture();
    let head = root.current_state_id();
    let index = root.build_group_index().unwrap();
    assert_eq!(index.group_ids().len(), 2);
    assert_eq!(index.root_groups().len(), 2);
    assert_eq!(index.children(parent).unwrap(), vec![child]);
    assert_eq!(index.parents(child).unwrap(), vec![parent]);
    assert_eq!(
        index.node_references(child, false).unwrap(),
        vec![reference(0, "also"), reference(1, "contains_subgraph")]
    );
    assert_eq!(
        index.node_references(child, true).unwrap(),
        vec![
            reference(0, "also"),
            reference(0, "contains_subgraph"),
            reference(1, "contains_subgraph")
        ]
    );
    assert_eq!(
        index.edge_references(child, true).unwrap(),
        vec![reference(0, "evidence")]
    );
    for edge in 0..3 {
        assert!(index.contains_edge(child, edge).unwrap());
    }
    assert!(!index.contains_edge(parent, 0).unwrap());
    assert!(index.contains_node(child, 1).unwrap());
    assert!(!index.contains_node(parent, 1).unwrap());
    assert!(!index.contains_node(child, usize::MAX).unwrap());
    assert_eq!(
        index
            .group_view(child)
            .unwrap()
            .node_attr(0, &"source".into())
            .unwrap(),
        Some(V::Int(7))
    );
    assert_eq!(root.current_state_id(), head);
    assert!(index.group_view(0).is_err());
    assert!(index.node_references(0, true).is_err());
}

#[test]
fn selected_group_indexes_exclude_unselected_roots_and_unreferenced_revisions() {
    let (mut root, parent, child) = fixture();
    root.retain_group(&Graph::new().read_view().unwrap(), "unreferenced".into())
        .unwrap();
    let selected = root.read_view().unwrap().select_nodes(&[1]).unwrap();
    let index = selected.build_group_index().unwrap();
    assert_eq!(index.root_groups(), vec![child]);
    assert_eq!(index.group_ids(), vec![child]);
    assert!(index.group_view(parent).is_err());
    assert!(index.edge_references(child, true).unwrap().is_empty());
    assert_eq!(
        index.node_references(child, true).unwrap(),
        vec![reference(1, "contains_subgraph")]
    );
    let empty = selected
        .select_nodes(&[])
        .unwrap()
        .build_group_index()
        .unwrap();
    assert!(empty.group_ids().is_empty());
    let parent_only = root
        .read_view()
        .unwrap()
        .select_nodes(&[0])
        .unwrap()
        .build_group_index()
        .unwrap();
    assert_eq!(parent_only.group_ids().len(), 2);
    assert_eq!(parent_only.node_references(child, true).unwrap().len(), 2);
}

#[test]
fn group_indexes_survive_mutation_rebuild_branch_switch_and_owner_drop() {
    let (mut root, parent, child) = fixture();
    let index = root.build_group_index().unwrap();
    root.create_branch("old".into()).unwrap();
    root.remove_node(1).unwrap();
    let newer = index.rebuild(root.read_view().unwrap()).unwrap();
    assert_eq!(
        newer.node_references(child, false).unwrap(),
        vec![reference(0, "also")]
    );
    assert!(newer.edge_references(child, false).unwrap().is_empty());
    root.checkout_branch("old".into()).unwrap();
    assert_eq!(
        root.build_group_index()
            .unwrap()
            .node_references(child, false)
            .unwrap()
            .len(),
        2
    );
    drop(root);
    std::thread::spawn(move || {
        assert_eq!(index.node_references(child, false).unwrap().len(), 2);
        assert_eq!(newer.children(parent).unwrap(), vec![child]);
        assert!(index.group_view(child).unwrap().write_state_id().is_none());
    })
    .join()
    .unwrap();
}
