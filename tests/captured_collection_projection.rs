//! Bulk projection must preserve independent captures, ordering and presence semantics.
use groggy::{storage::table::Table, AttrValue as V, Graph, GraphView};

#[test]
fn independent_sources_and_empty_projections_preserve_input_order() {
    let mut graph = Graph::new_manual();
    graph.add_nodes(2).unwrap();
    graph.add_edge(0, 1).unwrap();
    graph.set_edge_attr(0, "x".into(), V::Int(8)).unwrap();
    let old = graph.read_view().unwrap();
    graph.set_node_attr(1, "x".into(), V::Null).unwrap();
    let new = graph.read_view().unwrap();
    let empty = new.select_nodes(&[]).unwrap();
    let views = [new, empty, old];
    let name = "x".to_owned();
    let values = GraphView::project_attribute_observations(&views, &name);
    assert_eq!(values[0].as_ref().unwrap().data(), &vec![V::Null]);
    assert!(values[1].is_none());
    assert_eq!(values[2].as_ref().unwrap().data(), &vec![V::Int(8)]);
    let tables = GraphView::project_node_columns(&views, &[name.clone(), name]).unwrap();
    assert_eq!(tables.len(), 3);
    assert_eq!(
        tables[0].column("x").unwrap().data(),
        &vec![V::Null, V::Null]
    );
    assert!(tables[1].column("x").unwrap().is_empty());
    assert_eq!(
        tables[2].column("x").unwrap().data(),
        &vec![V::Null, V::Null]
    );
    assert_eq!(
        GraphView::project_node_columns(&views, &[]).unwrap().len(),
        3
    );
    assert!(GraphView::project_node_columns(&[], &["x".into()])
        .unwrap()
        .is_empty());
    assert!(GraphView::project_attribute_observations(&[], &"x".into()).is_empty());
}

#[test]
fn parallel_groups_match_scalar_readers_after_owner_changes_and_removal() {
    let mut graph = Graph::new_manual();
    let count = 8192;
    let groups = 64;
    graph.add_nodes(count).unwrap();
    for id in 0..count {
        graph
            .set_node_attr(id, "group".into(), V::Int((id % groups) as i64))
            .unwrap();
        if id % 3 != 2 {
            graph
                .set_node_attr(
                    id,
                    "x".into(),
                    if id % 3 == 0 {
                        V::Int(id as i64)
                    } else {
                        V::Null
                    },
                )
                .unwrap();
        }
        graph
            .set_node_attr(id, "y".into(), V::Float64(id as f64 + 0.5))
            .unwrap();
    }
    let root = graph.read_view().unwrap();
    let mut views = root.group_by_nodes(&"group".into()).unwrap();
    views.reverse();
    graph.set_node_attr(0, "x".into(), V::Int(-1)).unwrap();
    graph.remove_node(1).unwrap();
    let columns = ["x".into(), "y".into(), "missing".into()];
    let tables = GraphView::project_node_columns(&views, &columns).unwrap();
    let observations = GraphView::project_attribute_observations(&views, &"x".into());
    assert_eq!(tables.len(), groups);
    for ((view, table), observed) in views.iter().zip(tables).zip(observations) {
        let nodes = view.node_accessor();
        for name in &columns {
            assert_eq!(
                table.column(name).unwrap().data(),
                nodes.column(name).data()
            );
        }
        assert_eq!(
            observed.unwrap().data(),
            nodes.present_column(&"x".into()).data()
        );
    }
}
