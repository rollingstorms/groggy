use groggy::{
    algorithms::{
        steps::{
            ensure_core_steps_registered, global_step_registry, StepScope, StepSpec, StepVariables,
        },
        AlgorithmParamValue as V, AlgorithmParams, Context,
    },
    subgraphs::Subgraph,
    Graph,
};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};
fn fixture() -> Subgraph {
    let mut graph = Graph::new();
    graph.add_nodes(3).unwrap();
    graph.add_edge(0, 2).unwrap();
    graph.add_edge(2, 0).unwrap();
    Subgraph::new(
        Rc::new(RefCell::new(graph)),
        HashSet::from([0, 2]),
        HashSet::from([0, 1]),
        "power".into(),
    )
}
fn apply(
    variables: &mut StepVariables,
    selected: &Subgraph,
    ctx: &mut Context,
) -> anyhow::Result<()> {
    ensure_core_steps_registered();
    let mut params = AlgorithmParams::new();
    params.set_text("left", "a");
    params.set_text("right", "b");
    params.set_text("target", "out");
    let step = global_step_registry().instantiate(&StepSpec {
        id: "core.pow".into(),
        params,
        inputs: vec![],
        outputs: vec![],
    })?;
    assert!(step.supports_atomic_properties());
    step.apply(ctx, &mut StepScope::new(selected, variables))
}
#[test]
fn power_uses_checked_exact_expression_semantics_on_node_and_edge_maps() {
    let selected = fixture();
    for edges in [false, true] {
        for (a, b, expected) in [
            (V::Int(3), V::Int(34), V::Int(16_677_181_699_666_569)),
            (V::Int(-2), V::Int(3), V::Int(-8)),
            (V::Int(0), V::Int(0), V::Int(1)),
            (V::Int(2), V::Int(-3), V::Float(0.125)),
            (V::Float(9.), V::Float(0.5), V::Float(3.)),
        ] {
            let mut vars = StepVariables::default();
            let keys = if edges { [0, 1] } else { [0, 2] };
            let left = keys.into_iter().map(|k| (k, a.clone())).collect();
            let right = keys.into_iter().map(|k| (k, b.clone())).collect();
            if edges {
                vars.set_edge_map("a", left);
                vars.set_edge_map("b", right);
            } else {
                vars.set_node_map("a", left);
                vars.set_node_map("b", right);
            }
            apply(&mut vars, &selected, &mut Context::new()).unwrap();
            let result = if edges {
                vars.edge_map("out")
            } else {
                vars.node_map("out")
            }
            .unwrap();
            assert_eq!(
                result,
                &keys.into_iter().map(|k| (k, expected.clone())).collect()
            );
        }
    }
}
#[test]
fn power_preserves_scalar_side_and_rejects_invalid_values_without_replacing_outputs() {
    let selected = fixture();
    for reverse in [false, true] {
        let mut vars = StepVariables::default();
        vars.set_node_map(
            if reverse { "b" } else { "a" },
            HashMap::from([(0, V::Int(3))]),
        );
        vars.set_scalar(if reverse { "a" } else { "b" }, V::Int(2));
        apply(&mut vars, &selected, &mut Context::new()).unwrap();
        assert_eq!(
            vars.node_map("out").unwrap()[&0],
            V::Int(if reverse { 8 } else { 9 })
        );
    }
    for (a, b) in [
        (V::Int(2), V::Int(63)),
        (V::Int(0), V::Int(-1)),
        (V::Int(-2), V::Float(0.5)),
        (V::Float(f64::NAN), V::Int(2)),
        (V::Bool(true), V::Int(2)),
        (V::Text("2".into()), V::Int(2)),
    ] {
        let mut vars = StepVariables::default();
        vars.set_node_map("a", HashMap::from([(0, a)]));
        vars.set_scalar("b", b);
        vars.set_scalar("out", V::Int(99));
        assert!(apply(&mut vars, &selected, &mut Context::new()).is_err());
        assert_eq!(vars.scalar("out").unwrap(), &V::Int(99));
    }
    let mut vars = StepVariables::default();
    vars.set_node_map("a", HashMap::from([(0, V::Int(2))]));
    vars.set_edge_map("b", HashMap::from([(0, V::Int(3))]));
    assert!(apply(&mut vars, &selected, &mut Context::new()).is_err());
    vars.set_node_map("b", HashMap::new());
    assert!(apply(&mut vars, &selected, &mut Context::new()).is_err());
    vars.set_node_map("a", HashMap::new());
    apply(&mut vars, &selected, &mut Context::new()).unwrap();
    assert!(vars.node_map("out").unwrap().is_empty());
}

#[test]
fn scalar_power_stays_scalar_and_cancellation_preserves_existing_outputs() {
    let selected = fixture();
    let mut vars = StepVariables::default();
    vars.set_scalar("a", V::Int(3));
    vars.set_scalar("b", V::Int(4));
    apply(&mut vars, &selected, &mut Context::new()).unwrap();
    assert_eq!(vars.scalar("out").unwrap(), &V::Int(81));
    let mut ctx = Context::with_cancel_token(std::sync::Arc::new(
        std::sync::atomic::AtomicBool::new(true),
    ));
    assert!(apply(&mut vars, &selected, &mut ctx).is_err());
    assert_eq!(vars.scalar("out").unwrap(), &V::Int(81));
}
