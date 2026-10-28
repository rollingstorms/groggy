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
fn selected() -> Subgraph {
    let mut graph = Graph::new();
    graph.add_nodes(2).unwrap();
    graph.add_edge(0, 1).unwrap();
    Subgraph::new(
        Rc::new(RefCell::new(graph)),
        HashSet::from([0, 1]),
        HashSet::from([0]),
        "math".into(),
    )
}
fn spec(id: &str, base: Option<V>) -> StepSpec {
    let mut params = AlgorithmParams::new();
    params.set_text("source", "in");
    params.set_text("target", "out");
    if let Some(v) = base {
        params.insert("base", v);
    }
    StepSpec {
        id: id.into(),
        params,
        inputs: vec![],
        outputs: vec![],
    }
}
#[test]
fn unary_math_preserves_scalar_node_and_edge_kinds_and_checked_values() {
    ensure_core_steps_registered();
    let selected = selected();
    for (id, input, base, expected) in [
        (
            "core.abs",
            V::Int(-9_007_199_254_740_993),
            None,
            V::Int(9_007_199_254_740_993),
        ),
        ("core.sqrt", V::Int(9), None, V::Float(3.)),
        ("core.exp", V::Int(0), None, V::Float(1.)),
        ("core.log", V::Int(16), Some(V::Int(4)), V::Float(2.)),
    ] {
        for kind in 0..3 {
            let step = global_step_registry()
                .instantiate(&spec(id, base.clone()))
                .unwrap();
            assert!(step.supports_atomic_properties());
            let mut vars = StepVariables::default();
            match kind {
                0 => vars.set_scalar("in", input.clone()),
                1 => vars.set_node_map("in", HashMap::from([(0, input.clone())])),
                _ => vars.set_edge_map("in", HashMap::from([(0, input.clone())])),
            }
            step.apply(
                &mut Context::new(),
                &mut StepScope::new(&selected, &mut vars),
            )
            .unwrap();
            let actual = match kind {
                0 => vars.scalar("out").unwrap(),
                1 => &vars.node_map("out").unwrap()[&0],
                _ => &vars.edge_map("out").unwrap()[&0],
            };
            assert_eq!(actual, &expected);
        }
    }
}
#[test]
fn unary_math_validates_controls_and_preserves_outputs_on_failure_or_cancellation() {
    ensure_core_steps_registered();
    let selected = selected();
    for value in [
        V::Int(0),
        V::Int(1),
        V::Int(-1),
        V::Float(f64::INFINITY),
        V::Bool(true),
    ] {
        assert!(global_step_registry()
            .instantiate(&spec("core.log", Some(value)))
            .is_err());
    }
    for (id, value) in [
        ("core.abs", V::Int(i64::MIN)),
        ("core.sqrt", V::Int(-1)),
        ("core.exp", V::Int(1000)),
        ("core.log", V::Int(0)),
        ("core.abs", V::Bool(true)),
        ("core.sqrt", V::Float(f64::NAN)),
    ] {
        let step = global_step_registry().instantiate(&spec(id, None)).unwrap();
        let mut vars = StepVariables::default();
        vars.set_node_map("in", HashMap::from([(0, value)]));
        vars.set_scalar("out", V::Int(99));
        assert!(step
            .apply(
                &mut Context::new(),
                &mut StepScope::new(&selected, &mut vars)
            )
            .is_err());
        assert_eq!(vars.scalar("out").unwrap(), &V::Int(99));
    }
    let step = global_step_registry()
        .instantiate(&spec("core.sqrt", None))
        .unwrap();
    let mut vars = StepVariables::default();
    vars.set_node_map("in", HashMap::new());
    step.apply(
        &mut Context::new(),
        &mut StepScope::new(&selected, &mut vars),
    )
    .unwrap();
    assert!(vars.node_map("out").unwrap().is_empty());
    vars.set_scalar("out", V::Int(99));
    let mut ctx = Context::with_cancel_token(std::sync::Arc::new(
        std::sync::atomic::AtomicBool::new(true),
    ));
    assert!(step
        .apply(&mut ctx, &mut StepScope::new(&selected, &mut vars))
        .is_err());
    assert_eq!(vars.scalar("out").unwrap(), &V::Int(99));
}

#[test]
fn elementwise_extrema_preserve_exact_mixed_numbers_left_ties_and_map_kind() {
    ensure_core_steps_registered();
    let selected = selected();
    for id in ["core.min", "core.max"] {
        let mut params = AlgorithmParams::new();
        params.set_text("left", "a");
        params.set_text("right", "b");
        params.set_text("target", "out");
        let step = global_step_registry()
            .instantiate(&StepSpec {
                id: id.into(),
                params,
                inputs: vec![],
                outputs: vec![],
            })
            .unwrap();
        let mut vars = StepVariables::default();
        vars.set_node_map("a", HashMap::from([(0, V::Int(9_007_199_254_740_993))]));
        vars.set_scalar("b", V::Float(9_007_199_254_740_992.));
        step.apply(
            &mut Context::new(),
            &mut StepScope::new(&selected, &mut vars),
        )
        .unwrap();
        assert_eq!(
            vars.node_map("out").unwrap()[&0],
            if id == "core.min" {
                V::Float(9_007_199_254_740_992.)
            } else {
                V::Int(9_007_199_254_740_993)
            }
        );
        vars.set_scalar("a", V::Int(7));
        vars.set_scalar("b", V::Float(7.));
        step.apply(
            &mut Context::new(),
            &mut StepScope::new(&selected, &mut vars),
        )
        .unwrap();
        assert_eq!(vars.scalar("out").unwrap(), &V::Int(7));
        vars.set_edge_map("a", HashMap::from([(0, V::Int(2))]));
        vars.set_scalar("b", V::Int(3));
        step.apply(
            &mut Context::new(),
            &mut StepScope::new(&selected, &mut vars),
        )
        .unwrap();
        assert_eq!(
            vars.edge_map("out").unwrap()[&0],
            V::Int(if id == "core.min" { 2 } else { 3 })
        );
        vars.set_scalar("out", V::Int(99));
        vars.set_scalar("b", V::Bool(true));
        assert!(step
            .apply(
                &mut Context::new(),
                &mut StepScope::new(&selected, &mut vars)
            )
            .is_err());
        assert_eq!(vars.scalar("out").unwrap(), &V::Int(99));
    }
}
