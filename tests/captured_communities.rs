//! Community kernels consume the same frozen topology/values as other classic reads.
use groggy::algorithms::{community::*, Algorithm, Context};
use groggy::ownership::OwnerCell;
use groggy::{subgraphs::Subgraph, types::GraphType, AttrValue, Graph, GraphView};
use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn fixture(directed: bool) -> (Arc<OwnerCell<Graph>>, Subgraph, GraphView) {
    let mut graph = Graph::new_with_type_manual(if directed {
        GraphType::Directed
    } else {
        GraphType::Undirected
    });
    graph.add_nodes(9).unwrap();
    graph.remove_node(1).unwrap(); // sparse IDs
    for (u, v) in [
        (0, 2),
        (2, 3),
        (3, 0),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 4),
        (0, 2),
        (2, 2),
        (6, 8),
    ] {
        let edge = graph.add_edge(u, v).unwrap();
        graph
            .set_edge_attr(edge, "length".into(), AttrValue::Float64(1.0))
            .unwrap();
    }
    for node in [0, 2, 3, 4, 5, 6, 7] {
        graph
            .set_node_attr(
                node,
                "seed".into(),
                AttrValue::Text(if node < 4 { "left" } else { "right" }.into()),
            )
            .unwrap();
    }
    // Exclude an external endpoint and one parallel instance; retain the isolate.
    let nodes = HashSet::from([0, 2, 3, 4, 5, 6, 7]);
    let edges = HashSet::from([0, 1, 2, 3, 4, 5, 6, 8]);
    let view = GraphView::from_membership(
        graph.read_view().unwrap().read_state().clone(),
        nodes.clone(),
        edges.clone(),
    )
    .unwrap();
    let graph = Arc::new(OwnerCell::new(graph));
    let selected = Subgraph::new(graph.clone(), nodes, edges, "communities".into());
    (graph, selected, view)
}

fn numeric(view: &GraphView, kind: usize, ctx: &mut Context) -> Vec<(usize, usize)> {
    let input = view.algorithm_input().unwrap();
    match kind {
        0 => input.louvain(&Louvain::new(20, 10, 1.0, "result".into()).unwrap(), ctx),
        1 => input.leiden(
            &Leiden::new(20, 10, 1.0, Some(19), "result".into()).unwrap(),
            ctx,
        ),
        2 => input.infomap(
            &Infomap::new(0.15, 3, 30, Some(19), "result".into()).unwrap(),
            ctx,
        ),
        _ => input.girvan_newman(
            &GirvanNewman::new(Some(4), None, Some("length".into()), "result".into()),
            ctx,
        ),
    }
    .unwrap()
}

#[test]
fn captured_numeric_communities_match_connected_kernels_and_never_publish() {
    for kind in 0..4 {
        let (graph, selected, view) = fixture(false);
        let before = graph.borrow().current_state_id();
        let expected = numeric(&view, kind, &mut Context::new());
        assert_eq!(graph.borrow().current_state_id(), before);
        for &(node, _) in &expected {
            assert_eq!(
                graph
                    .borrow()
                    .get_node_attr(node, &"result".into())
                    .unwrap(),
                None
            );
        }
        let algorithm: Box<dyn Algorithm> = match kind {
            0 => Box::new(Louvain::new(20, 10, 1.0, "result".into()).unwrap()),
            1 => Box::new(Leiden::new(20, 10, 1.0, Some(19), "result".into()).unwrap()),
            2 => Box::new(Infomap::new(0.15, 3, 30, Some(19), "result".into()).unwrap()),
            _ => Box::new(GirvanNewman::new(
                Some(4),
                None,
                Some("length".into()),
                "result".into(),
            )),
        };
        let mut ctx = Context::new();
        ctx.set_persist_results(true);
        algorithm.execute(&mut ctx, selected).unwrap();
        for (node, label) in expected {
            assert_eq!(
                graph
                    .borrow()
                    .get_node_attr(node, &"result".into())
                    .unwrap(),
                Some(AttrValue::Int(label as i64)),
                "kind {kind}, node {node}"
            );
        }
    }
}

#[test]
fn captured_seed_labels_match_connected_and_survive_owner_removal_and_drop() {
    for directed in [false, true] {
        let (graph, selected, view) = fixture(directed);
        let algorithm = LabelPropagation::new(20, 0.0, "result".into(), Some("seed".into()))
            .unwrap()
            .with_seed(Some(23));
        let expected = view
            .algorithm_input()
            .unwrap()
            .label_propagation(&algorithm, &mut Context::new())
            .unwrap();
        let mut ctx = Context::new();
        ctx.set_persist_results(true);
        algorithm.execute(&mut ctx, selected.clone()).unwrap();
        for (node, label) in &expected {
            assert_eq!(
                graph
                    .borrow()
                    .get_node_attr(*node, &"result".into())
                    .unwrap(),
                Some(label.clone())
            );
        }
        graph
            .borrow_mut()
            .set_node_attr(0, "seed".into(), AttrValue::Null)
            .unwrap();
        graph
            .borrow_mut()
            .set_edge_attr(0, "length".into(), AttrValue::Float64(-1.0))
            .unwrap();
        graph.borrow_mut().remove_node(2).unwrap();
        drop(selected);
        drop(graph);
        let input = view.algorithm_input().unwrap();
        assert_eq!(
            input
                .label_propagation(&algorithm, &mut Context::new())
                .unwrap(),
            expected
        );
        std::thread::spawn(move || {
            assert_eq!(
                input
                    .label_propagation(&algorithm, &mut Context::new())
                    .unwrap(),
                expected
            )
        })
        .join()
        .unwrap();
    }
}

#[test]
fn numeric_communities_survive_owner_mutation_and_worker_transfer() {
    let (graph, selected, view) = fixture(false);
    let expected: Vec<_> = (0..4)
        .map(|kind| numeric(&view, kind, &mut Context::new()))
        .collect();
    graph.borrow_mut().remove_node(3).unwrap();
    graph
        .borrow_mut()
        .set_edge_attr(0, "length".into(), AttrValue::Float64(-1.0))
        .unwrap();
    drop(selected);
    drop(graph);
    std::thread::spawn(move || {
        for (kind, rows) in expected.into_iter().enumerate() {
            assert_eq!(numeric(&view, kind, &mut Context::new()), rows);
        }
    })
    .join()
    .unwrap();
}

#[test]
fn empty_isolate_direction_and_cancellation_are_explicit() {
    for count in [0, 1] {
        let mut graph = Graph::new_manual();
        graph.add_nodes(count).unwrap();
        let view = graph.read_view().unwrap();
        assert_eq!(view.louvain(20, 10, 1.0).unwrap().len(), count);
        assert_eq!(view.leiden(20, 10, 1.0, None, 0.01).unwrap().len(), count);
        assert_eq!(view.infomap(0.15, 3, 20, 10000, None).unwrap().len(), count);
        assert_eq!(view.girvan_newman(None, None, None).unwrap().len(), count);
        assert_eq!(
            view.label_propagation(20, 0.0, None, None).unwrap().len(),
            count
        );
        let input = view.algorithm_input().unwrap();
        let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        assert!(input
            .louvain(&Louvain::new(20, 10, 1.0, "x".into()).unwrap(), &mut ctx)
            .is_err());
        assert!(input
            .leiden(
                &Leiden::new(20, 10, 1.0, None, "x".into()).unwrap(),
                &mut ctx
            )
            .is_err());
        assert!(input
            .infomap(
                &Infomap::new(0.15, 3, 20, None, "x".into()).unwrap(),
                &mut ctx
            )
            .is_err());
        assert!(input
            .girvan_newman(&GirvanNewman::new(None, None, None, "x".into()), &mut ctx)
            .is_err());
        assert!(input
            .label_propagation(
                &LabelPropagation::new(20, 0.0, "x".into(), None).unwrap(),
                &mut ctx
            )
            .is_err());
    }
    let (_, _, view) = fixture(true);
    assert!(view.louvain(20, 10, 1.0).is_err());
    assert!(view.leiden(20, 10, 1.0, None, 0.01).is_err());
    assert!(view.girvan_newman(None, None, None).is_err());
    assert_eq!(
        view.infomap(0.15, 3, 20, 10000, None).unwrap().len(),
        view.node_count()
    );
}

#[test]
fn invalid_parameters_and_selected_weight_domains_fail_without_writes() {
    let (graph, _, view) = fixture(false);
    assert!(view.louvain(0, 10, 1.0).is_err());
    assert!(view.leiden(20, 10, 1.0, None, f64::NAN).is_err());
    assert!(view.infomap(0.15, 3, 20, 0, None).is_err());
    assert!(view.girvan_newman(None, Some(f64::NAN), None).is_err());
    assert!(view.label_propagation(20, f64::NAN, None, None).is_err());
    // Bad excluded parallel edge is not part of the captured weighted problem.
    graph
        .borrow_mut()
        .set_edge_attr(7, "length".into(), AttrValue::Float64(-1.0))
        .unwrap();
    let all = graph.borrow().read_view().unwrap();
    assert!(all
        .girvan_newman(Some(0), None, Some("length".into()))
        .is_err());
    assert!(view
        .girvan_newman(Some(0), None, Some("length".into()))
        .is_ok());
}
