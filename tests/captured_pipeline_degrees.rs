//! Degree table computations use explicit immutable topology, never the owner.
use groggy::ownership::OwnerCell;
use groggy::{
    algorithms::{
        steps::{
            ensure_core_steps_registered, global_step_registry, StepScope, StepSpec, StepValue,
            StepVariables,
        },
        AlgorithmParams, Context,
    },
    storage::table::Table,
    subgraphs::Subgraph,
    AttrValue, Graph, GraphView,
};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
fn apply(
    view: GraphView,
    ctx: &mut Context,
    scope: &Subgraph,
) -> anyhow::Result<groggy::storage::table::BaseTable> {
    ensure_core_steps_registered();
    let mut params = AlgorithmParams::new();
    params.set_text("source", "input");
    params.set_text("output", "result");
    let step = global_step_registry().instantiate(&StepSpec {
        id: "graph.degrees_table".into(),
        params,
        inputs: vec![],
        outputs: vec![],
    })?;
    let mut vars = StepVariables::default();
    vars.set_value("input", StepValue::GraphView(view));
    step.apply(ctx, &mut StepScope::new(scope, &mut vars))?;
    Ok(vars.table("result")?.clone())
}
#[test]
fn selected_sparse_directed_and_undirected_instances_use_canonical_degree_definitions() {
    for directed in [false, true] {
        let mut g = if directed {
            Graph::new_directed()
        } else {
            Graph::new()
        };
        g.add_nodes(7).unwrap();
        g.remove_nodes(&[1, 3, 4]).unwrap();
        g.add_edges(&[(0, 2), (0, 2), (2, 2), (5, 0), (2, 5)])
            .unwrap();
        g.set_node_attr(0, "degree".into(), AttrValue::Int(999))
            .unwrap();
        let view = GraphView::from_membership(
            g.read_view().unwrap().read_state().clone(),
            [0, 2, 5, 6].into(),
            [0, 1, 2, 3].into(),
        )
        .unwrap();
        let g = Arc::new(OwnerCell::new(g));
        let scope = Subgraph::from_frozen_view(g.clone(), view.clone(), "old".into());
        g.borrow_mut().remove_node(2).unwrap();
        let table = apply(view.clone(), &mut Context::new(), &scope).unwrap();
        assert_eq!(table.source_entity_ids(), Some([0, 2, 5, 6].as_slice()));
        assert_eq!(
            table.column_names(),
            &["node_id", "degree", "in_degree", "out_degree"]
        );
        let values = |name| {
            table
                .column(name)
                .unwrap()
                .data()
                .iter()
                .map(|v| {
                    if let AttrValue::Int(v) = v {
                        *v
                    } else {
                        panic!("int")
                    }
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(values("degree"), vec![3, 3, 1, 0]);
        assert_eq!(
            values("in_degree"),
            if directed {
                vec![1, 3, 0, 0]
            } else {
                vec![3, 3, 1, 0]
            }
        );
        assert_eq!(
            values("out_degree"),
            if directed {
                vec![2, 1, 1, 0]
            } else {
                vec![3, 3, 1, 0]
            }
        );
        assert_eq!(
            view.node_attr(0, &"degree".into()).unwrap(),
            Some(AttrValue::Int(999))
        );
        drop(scope);
        drop(view);
        drop(g);
        std::thread::spawn(move || {
            assert_eq!(table.shape(), (4, 4));
            assert_eq!(table.source_views()[0].edge_endpoints(0).unwrap(), (0, 2));
        })
        .join()
        .unwrap();
    }
}
#[test]
fn empty_and_cancelled_computations_do_not_install_partial_results() {
    let g = Arc::new(OwnerCell::new(Graph::new()));
    let view = g.borrow().read_view().unwrap();
    let scope = Subgraph::from_frozen_view(g, view.clone(), "empty".into());
    assert_eq!(
        apply(view.clone(), &mut Context::new(), &scope)
            .unwrap()
            .shape(),
        (0, 4)
    );
    assert!(apply(
        view,
        &mut Context::with_cancel_token(Arc::new(AtomicBool::new(true))),
        &scope
    )
    .is_err());
}

#[test]
fn parallel_degree_columns_keep_sparse_identity_and_three_distinct_count_modes_aligned() {
    for directed in [false, true] {
        let mut g = if directed {
            Graph::new_directed()
        } else {
            Graph::new()
        };
        g.add_nodes(10_000).unwrap();
        g.remove_nodes(&(1..10_000).step_by(2).collect::<Vec<_>>())
            .unwrap();
        let mut edges: Vec<_> = (2..10_000).step_by(2).map(|id| (0, id)).collect();
        edges.extend([(0, 2), (2, 2)]);
        g.add_edges(&edges).unwrap();
        let view = g.read_view().unwrap();
        let g = Arc::new(OwnerCell::new(g));
        let scope = Subgraph::from_frozen_view(g, view.clone(), "large".into());
        let table = apply(view, &mut Context::new(), &scope).unwrap();
        let ids = table.source_entity_ids().unwrap();
        assert_eq!(ids.len(), 5_000);
        for (row, &id) in ids.iter().enumerate() {
            assert_eq!(id, row * 2);
            let degree = if id == 0 {
                5000
            } else if id == 2 {
                3
            } else {
                1
            };
            let incoming = if !directed {
                degree
            } else if id == 0 {
                0
            } else if id == 2 {
                3
            } else {
                1
            };
            let outgoing = if !directed {
                degree
            } else if id == 0 {
                5000
            } else if id == 2 {
                1
            } else {
                0
            };
            for (name, expected) in [
                ("degree", degree),
                ("in_degree", incoming),
                ("out_degree", outgoing),
            ] {
                assert_eq!(
                    table.column(name).unwrap().data()[row],
                    AttrValue::Int(expected)
                );
            }
        }
    }
}
