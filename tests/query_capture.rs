use groggy::{
    query::{AttributeFilter as A, EdgeFilter as E, NodeFilter as N, QueryOptions as O},
    AttrValue as V, Graph,
};

fn fixture() -> Graph {
    let mut graph = Graph::new();
    graph.add_nodes(4).unwrap();
    for (node, value) in [
        (0, V::Int(9_007_199_254_740_993)),
        (1, V::Float64(9_007_199_254_740_992.)),
        (2, V::Null),
    ] {
        graph.set_node_attr(node, "score".into(), value).unwrap();
    }
    for (a, b, weight) in [(0, 1, 3), (0, 1, 1), (1, 1, 2)] {
        let edge = graph.add_edge(a, b).unwrap();
        graph
            .set_edge_attr(edge, "weight".into(), V::Int(weight))
            .unwrap();
    }
    graph
}
#[test]
fn query_capture_is_composable_and_stays_frozen_after_owner_changes() {
    let mut graph = fixture();
    let view = graph.query().unwrap();
    let all = N::NodeSet([0, 1, 2, 3].into());
    let options = O {
        order_by: Some("score".into()),
        ..Default::default()
    };
    assert_eq!(view.query_nodes(&all, &options).unwrap(), vec![1, 0, 2, 3]);
    assert_eq!(
        view.query_nodes(
            &all,
            &O {
                descending: true,
                ..options.clone()
            }
        )
        .unwrap(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        view.query_edges(
            &E::And(vec![]),
            &O {
                order_by: Some("weight".into()),
                offset: 1,
                limit: Some(1),
                ..Default::default()
            }
        )
        .unwrap(),
        vec![2]
    );
    let selected = view.filter_nodes(&N::NodeSet([0, 1].into())).unwrap();
    assert_eq!(selected.edge_count(), 3);
    graph.remove_node(0).unwrap();
    graph.set_node_attr(1, "score".into(), V::Int(-1)).unwrap();
    assert_eq!(view.query_nodes(&all, &options).unwrap(), vec![1, 0, 2, 3]);
    assert_eq!(
        selected
            .query_edges(&E::And(vec![]), &O::default())
            .unwrap(),
        vec![0, 1, 2]
    );
    assert_eq!(
        graph.query().unwrap().query_nodes(&all, &options).unwrap(),
        vec![1, 2, 3]
    );
    drop(graph);
    std::thread::spawn(move || {
        assert_eq!(view.query_nodes(&all, &options).unwrap(), vec![1, 0, 2, 3])
    })
    .join()
    .unwrap();
}
#[test]
fn scan_and_indexed_pages_share_exact_comparison_scopes_and_validation() {
    let graph = fixture();
    let view = graph.query().unwrap().select_edges(&[0, 2]).unwrap();
    let index = view
        .build_index(&["score".into()], &["weight".into()], true)
        .unwrap();
    for descending in [true, false] {
        for offset in [0, 1, 3, usize::MAX] {
            for limit in [None, Some(0), Some(1), Some(usize::MAX)] {
                let options = O {
                    order_by: Some("score".into()),
                    descending,
                    offset,
                    limit,
                };
                let filter = N::NodeSet([0, 1, 2, 3].into());
                assert_eq!(
                    view.query_nodes(&filter, &options).unwrap(),
                    index.query_nodes(&filter, &options, false).unwrap()
                );
                let options = O {
                    order_by: Some("weight".into()),
                    ..options
                };
                assert_eq!(
                    view.query_edges(&E::And(vec![]), &options).unwrap(),
                    index.query_edges(&E::And(vec![]), &options, false).unwrap()
                );
            }
        }
    }
    assert_eq!(
        view.query_nodes(
            &N::AttributeFilter {
                name: "score".into(),
                filter: A::GreaterThan(V::Float64(9_007_199_254_740_992.))
            },
            &O::default()
        )
        .unwrap(),
        vec![0]
    );
    assert!(view
        .query_nodes(
            &N::And(vec![]),
            &O {
                order_by: Some(" ".into()),
                ..Default::default()
            }
        )
        .is_err());
}
#[test]
fn scalar_ordering_rejects_invalid_matching_values_even_for_empty_pages() {
    let mut graph = fixture();
    graph
        .set_node_attr(0, "score".into(), V::Float64(f64::NAN))
        .unwrap();
    let view = graph.query().unwrap();
    let options = O {
        order_by: Some("score".into()),
        limit: Some(0),
        ..Default::default()
    };
    assert!(view
        .query_nodes(&N::NodeSet([0, 1].into()), &options)
        .is_err());
    assert!(view
        .query_nodes(&N::NodeSet([1].into()), &options)
        .unwrap()
        .is_empty());
}
