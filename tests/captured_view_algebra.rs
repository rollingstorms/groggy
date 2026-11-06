//! Algebra never changes capture identity, invents edges or leaves dangling endpoints.
use groggy::{
    subgraphs::{SimilarityMetric, Subgraph},
    AttrValue, Graph, GraphView,
};
use std::{cell::RefCell, collections::HashSet, rc::Rc};

fn fixture() -> (Graph, GraphView) {
    let mut graph = Graph::new_manual();
    graph.add_nodes(4).unwrap();
    for (s, t) in [(0, 1), (0, 1), (1, 2), (0, 0)] {
        graph.add_edge(s, t).unwrap();
    }
    let view = graph.read_view().unwrap();
    (graph, view)
}
fn exact(root: &GraphView, nodes: &[usize], edges: &[usize]) -> GraphView {
    GraphView::from_membership(
        root.read_state().clone(),
        nodes.iter().copied().collect(),
        edges.iter().copied().collect(),
    )
    .unwrap()
}
#[test]
fn all_valid_small_selections_obey_exact_set_algebra_and_closure() {
    let (_, root) = fixture();
    let mut choices = Vec::new();
    for nmask in 0..16 {
        let nodes: HashSet<_> = (0..4).filter(|n| nmask & (1 << n) != 0).collect();
        for emask in 0..16 {
            let edges: HashSet<_> = (0..4).filter(|e| emask & (1 << e) != 0).collect();
            if let Ok(view) =
                GraphView::from_membership(root.read_state().clone(), nodes.clone(), edges)
            {
                choices.push(view);
            }
        }
    }
    for left in &choices {
        for right in &choices {
            let union = left.merge_with(right).unwrap();
            let intersection = left.intersect_with(right).unwrap();
            let difference = left.subtract_from(right).unwrap();
            assert_eq!(
                union.nodes(),
                &left.nodes().union(right.nodes()).copied().collect()
            );
            assert_eq!(
                union.edges(),
                &left.edges().union(right.edges()).copied().collect()
            );
            assert_eq!(
                intersection.nodes(),
                &left.nodes().intersection(right.nodes()).copied().collect()
            );
            assert_eq!(
                intersection.edges(),
                &left.edges().intersection(right.edges()).copied().collect()
            );
            assert_eq!(
                difference.nodes(),
                &left.nodes().difference(right.nodes()).copied().collect()
            );
            for result in [union, intersection, difference] {
                assert!(std::ptr::eq(
                    root.read_state().snapshot(),
                    result.read_state().snapshot()
                ));
                for edge in result.edge_ids() {
                    let (s, t) = result.edge_endpoints(edge).unwrap();
                    assert!(result.contains_node(s) && result.contains_node(t));
                }
            }
        }
    }
}
#[test]
fn union_preserves_exact_instances_and_difference_prunes_incident_edges() {
    let (_, root) = fixture();
    let left = exact(&root, &[0, 1, 3], &[0]);
    let right = exact(&root, &[0, 1, 2], &[1, 2]);
    assert_eq!(left.merge_with(&right).unwrap().edge_ids(), vec![0, 1, 2]); // loop is not induced
    assert!(left.intersect_with(&right).unwrap().edge_ids().is_empty());
    let remove = exact(&root, &[1], &[]);
    let difference = left.subtract_from(&remove).unwrap();
    assert_eq!(difference.node_ids(), vec![0, 3]);
    assert!(difference.edge_ids().is_empty());
    assert_eq!(
        left.calculate_similarity(&right, SimilarityMetric::Jaccard)
            .unwrap(),
        0.5
    );
    assert_eq!(
        left.calculate_similarity(&right, SimilarityMetric::Dice)
            .unwrap(),
        2. / 3.
    );
    assert_eq!(
        left.calculate_similarity(&right, SimilarityMetric::Overlap)
            .unwrap(),
        2. / 3.
    );
    assert!(
        (left
            .calculate_similarity(&right, SimilarityMetric::Cosine)
            .unwrap()
            - 2. / 3.)
            .abs()
            < 1e-15
    );
    let empty = root.select_nodes(&[]).unwrap();
    assert_eq!(
        empty
            .calculate_similarity(&empty, SimilarityMetric::Jaccard)
            .unwrap(),
        0.
    );
}
#[test]
fn independent_captures_and_unrelated_owners_reject_even_when_ids_match() {
    let (graph, root) = fixture();
    let second = graph.read_view().unwrap(); // weak payload reuse does not make capture identity equal
    assert!(std::ptr::eq(
        root.read_state().snapshot(),
        second.read_state().snapshot()
    ));
    let (_, other) = fixture();
    for other in [second, other] {
        assert!(root.merge_with(&other).is_err());
        assert!(root.intersect_with(&other).is_err());
        assert!(root.subtract_from(&other).is_err());
        assert!(root
            .calculate_similarity(&other, SimilarityMetric::Jaccard)
            .is_err());
        assert!(root.find_overlaps(&[root.clone(), other]).is_err());
    }
}
#[test]
fn grouping_uses_typed_order_and_frozen_values_without_expanding_edge_scope() {
    let (mut graph, _) = fixture();
    graph
        .set_node_attr(0, "group".into(), AttrValue::Int(9_007_199_254_740_993))
        .unwrap();
    graph
        .set_node_attr(1, "group".into(), AttrValue::Int(9_007_199_254_740_993))
        .unwrap();
    graph
        .set_node_attr(2, "group".into(), AttrValue::Int(9_007_199_254_740_992))
        .unwrap();
    graph
        .set_node_attr(3, "group".into(), AttrValue::Null)
        .unwrap();
    graph
        .set_edge_attr(0, "group".into(), AttrValue::Text("x".into()))
        .unwrap();
    graph
        .set_edge_attr(1, "group".into(), AttrValue::Null)
        .unwrap();
    graph
        .set_edge_attr(2, "group".into(), AttrValue::Text("x".into()))
        .unwrap();
    let root = graph.read_view().unwrap();
    let selected = exact(&root, &[0, 1, 2, 3], &[0, 2]);
    let groups = selected.group_by_nodes(&"group".into()).unwrap();
    assert_eq!(
        groups.iter().map(GraphView::node_ids).collect::<Vec<_>>(),
        vec![vec![3], vec![2], vec![0, 1]]
    );
    assert_eq!(groups[2].edge_ids(), vec![0]);
    let edges = selected.group_by_edges(&"group".into()).unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].edge_ids(), vec![0, 2]);
    graph
        .set_node_attr(0, "group".into(), AttrValue::Null)
        .unwrap();
    graph.remove_node(1).unwrap();
    drop(graph);
    assert_eq!(
        selected.group_by_nodes(&"group".into()).unwrap()[2].node_ids(),
        vec![0, 1]
    );
    assert!(selected
        .group_by_nodes(&"absent".into())
        .unwrap()
        .is_empty());
    std::thread::spawn(move || {
        assert_eq!(
            selected.group_by_edges(&"group".into()).unwrap()[0].edge_ids(),
            vec![0, 2]
        )
    })
    .join()
    .unwrap();
}
#[test]
fn connected_adapters_reuse_canonical_membership_and_reject_other_owners() {
    let (graph, _) = fixture();
    let graph = Rc::new(RefCell::new(graph));
    let make = |nodes: &[usize], edges: &[usize]| {
        Subgraph::new(
            graph.clone(),
            nodes.iter().copied().collect(),
            edges.iter().copied().collect(),
            "test".into(),
        )
    };
    let left = make(&[0, 1, 3], &[0]);
    let right = make(&[1], &[]);
    let difference = left.subtract_from(&right).unwrap();
    assert_eq!(difference.node_count(), 2);
    assert_eq!(difference.edge_count(), 0);
    let (other, _) = fixture();
    let other = Subgraph::new(
        Rc::new(RefCell::new(other)),
        HashSet::from([1]),
        HashSet::new(),
        "other".into(),
    );
    assert!(left.merge_with(&other).is_err());
    assert!(left.intersect_with(&other).is_err());
    assert!(left.subtract_from(&other).is_err());
    assert!(left
        .calculate_similarity(&other, SimilarityMetric::Jaccard)
        .is_err());
}

#[test]
fn collection_union_preserves_capture_and_exact_instances_without_filler_members() {
    let (_, root) = fixture();
    let left = exact(&root, &[0, 1], &[0, 3]);
    let right = exact(&root, &[0, 1, 2], &[1, 2]);
    let union = GraphView::merge_all(&[left.clone(), right, left])
        .unwrap()
        .unwrap();
    assert_eq!(union.node_ids(), vec![0, 1, 2]);
    assert_eq!(union.edge_ids(), vec![0, 1, 2, 3]);
    assert!(root.merge_with(&union).is_ok());
    assert!(GraphView::merge_all(&[]).unwrap().is_none());
    let (_, unrelated) = fixture();
    assert!(GraphView::merge_all(&[union, unrelated]).is_err());
}
