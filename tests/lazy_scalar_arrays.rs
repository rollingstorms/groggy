use groggy::{
    storage::array::{BatchQueryEvaluator, LazyArrayIterator, QueryEvaluator},
    AttrValue as V,
};
use std::collections::HashSet;

#[test]
fn boolean_precedence_quoted_operators_and_parity_use_compiled_queries() {
    for (query, input, expected) in [
        (
            "value > 10 AND (value < 20 OR value == 30)",
            V::Int(15),
            true,
        ),
        (
            "value > 10 AND (value < 20 OR value == 30)",
            V::Int(25),
            false,
        ),
        ("NOT (value < 10 OR value > 20)", V::Int(12), true),
        ("value == 'CANDOR > 3'", V::Text("CANDOR > 3".into()), true),
        (
            "value == 'true OR false'",
            V::Text("true OR false".into()),
            true,
        ),
        ("value == 'a\\\"b'", V::Text("a\"b".into()), true),
        ("value % 2 == 0 AND value > 10", V::Int(12), true),
        ("value % 2 == 0 OR value > 10", V::Int(9), false),
        ("false OR (true AND value >= 5)", V::Int(5), true),
        ("value == bare_text", V::Text("bare_text".into()), true),
        (
            "value >= 9007199254740993",
            V::Float64(9007199254740992.0),
            false,
        ),
        ("value == 9007199254740993", V::Int(9007199254740993), true),
        ("value == null", V::Null, true),
        ("value != 1", V::Text("1".into()), false),
        ("value > 'b'", V::Text("c".into()), true),
        ("length >= 2", V::IntVec(vec![1, 2]), true),
        ("length == 2", V::Text("é".into()), true),
    ] {
        let evaluator = QueryEvaluator::new(query);
        assert_eq!(evaluator.evaluate(&input).unwrap(), expected, "{query}");
    }
}

#[test]
fn invalid_queries_reject_even_with_empty_inputs_and_short_circuited_branches() {
    for query in [
        "",
        "   ",
        "unknown > 2",
        "value = 1",
        "value >",
        "value == 'unterminated",
        "value > 1 garbage",
        "true OR unknown == 1",
        "false AND value >",
        "value % 3 == 0",
        "__array_even == true",
        "value > 9223372036854775808",
    ] {
        let evaluator = QueryEvaluator::new(query);
        assert!(evaluator.validate().is_err(), "{query}");
        assert!(
            BatchQueryEvaluator::new(query)
                .filter_values(vec![])
                .is_err(),
            "{query}"
        );
        assert!(
            LazyArrayIterator::<V>::new(vec![])
                .filter(query)
                .collect()
                .is_err(),
            "{query}"
        );
        assert!(
            LazyArrayIterator::new(vec![V::Int(1)])
                .take(0)
                .filter(query)
                .collect()
                .is_err(),
            "{query}"
        );
    }
}

#[test]
fn filters_take_and_skip_match_eager_statement_order_exhaustively() {
    #[derive(Clone, Copy)]
    enum Op {
        Filter,
        OrFilter,
        Take(usize),
        Skip(usize),
    }
    let operations = [
        Op::Filter,
        Op::OrFilter,
        Op::Take(0),
        Op::Take(3),
        Op::Take(usize::MAX),
        Op::Skip(1),
        Op::Skip(3),
        Op::Skip(usize::MAX),
    ];
    for a in operations {
        for b in operations {
            for c in operations {
                let mut expected = (-3_i64..9).collect::<Vec<_>>();
                let mut lazy = LazyArrayIterator::new(expected.clone());
                for op in [a, b, c] {
                    match op {
                        Op::Filter => {
                            expected.retain(|&value| value > 0);
                            lazy = lazy.filter("value > 0");
                        }
                        Op::OrFilter => {
                            expected.retain(|&value| !(0..=5).contains(&value));
                            lazy = lazy.filter("value < 0 OR value > 5");
                        }
                        Op::Take(n) => {
                            expected.truncate(n);
                            lazy = lazy.take(n);
                        }
                        Op::Skip(n) => {
                            expected = expected.into_iter().skip(n).collect();
                            lazy = lazy.skip(n);
                        }
                    }
                }
                assert!(lazy.estimated_len() >= expected.len());
                assert_eq!(lazy.collect().unwrap(), expected);
            }
        }
    }
}

#[test]
fn numeric_lazy_filters_preserve_source_dtype_and_exact_values() {
    assert_eq!(
        LazyArrayIterator::new(vec![1_i32, 3, 5])
            .filter("value >= 3")
            .collect()
            .unwrap(),
        vec![3_i32, 5]
    );
    assert_eq!(
        LazyArrayIterator::new(vec![1_f32, 3., 5.])
            .filter("value >= 3")
            .collect()
            .unwrap(),
        vec![3_f32, 5.]
    );
    assert_eq!(
        LazyArrayIterator::new(vec![1_f64, 3., 5.])
            .filter("value >= 3")
            .collect()
            .unwrap(),
        vec![3_f64, 5.]
    );
    assert_eq!(
        LazyArrayIterator::new(vec![true, false])
            .filter("value == true")
            .collect()
            .unwrap(),
        vec![true]
    );
    assert_eq!(
        LazyArrayIterator::new(vec![0_usize, 3])
            .filter("value > 0")
            .collect()
            .unwrap(),
        vec![3_usize]
    );
    assert!(LazyArrayIterator::new(vec![usize::MAX])
        .filter("value > 0")
        .collect()
        .is_err());
    let array = LazyArrayIterator::new(vec![V::Int(1), V::Int(3)])
        .filter("value > 1")
        .collect_as_array()
        .unwrap();
    assert_eq!(array.to_vec(), vec![V::Int(3)]);
    let one = LazyArrayIterator::new(vec![V::Int(1)]).filter("true");
    assert_eq!(one.estimated_len(), 1);
    assert!(!one.is_empty());
}

#[test]
fn seeded_sampling_is_local_repeatable_and_obeys_chain_order() {
    let run = || {
        LazyArrayIterator::new((0_i64..1000).collect())
            .take(100)
            .sample_seeded(10, 42)
            .filter("value >= 5")
            .collect()
            .unwrap()
    };
    let first = run();
    fastrand::seed(123);
    for _ in 0..100 {
        let _ = fastrand::u64(..);
    }
    assert_eq!(run(), first);
    assert!(first.iter().all(|&id| (5..100).contains(&id)));
    assert_eq!(first.iter().collect::<HashSet<_>>().len(), first.len());
    assert_eq!(
        LazyArrayIterator::new(vec![1, 2, 3])
            .sample_seeded(8, 42)
            .collect()
            .unwrap(),
        vec![1, 2, 3]
    );
    assert!(LazyArrayIterator::new(vec![1, 2, 3])
        .sample_seeded(0, 42)
        .collect()
        .unwrap()
        .is_empty());
}

#[test]
fn unsupported_graph_and_type_changing_operations_never_fabricate_results() {
    use groggy::storage::array::{NodeIdLike, SubgraphLike};
    #[derive(Clone)]
    struct Node;
    impl NodeIdLike for Node {}
    #[derive(Clone)]
    struct Selection;
    impl SubgraphLike for Selection {}
    assert!(LazyArrayIterator::new(vec![Node])
        .get_neighbors()
        .take(0)
        .collect()
        .is_err());
    assert!(LazyArrayIterator::new(vec![Node])
        .to_subgraph()
        .collect()
        .is_err());
    assert!(LazyArrayIterator::new(vec![Node])
        .filter_by_degree(2)
        .collect()
        .is_err());
    assert!(LazyArrayIterator::new(vec![Selection])
        .filter_nodes("x > 0")
        .collect()
        .is_err());
    assert!(LazyArrayIterator::new(vec![Selection])
        .filter_edges("x > 0")
        .collect()
        .is_err());
    assert!(LazyArrayIterator::new(Vec::<Selection>::new())
        .collapse(Default::default())
        .collect()
        .is_err());
    assert!(LazyArrayIterator::new(vec!["text".to_string()])
        .filter("true")
        .collect()
        .is_err());
}

#[test]
fn graph_and_array_numeric_predicates_share_exact_literal_and_value_semantics() {
    use groggy::{
        query::{query_parser::QueryParser, QueryOptions},
        Graph,
    };
    let values = vec![
        V::Int(i64::MIN),
        V::Int(i64::MAX),
        V::Int(9007199254740993),
        V::Float64(9007199254740992.0),
        V::Float64(f64::NAN),
        V::Float64(f64::INFINITY),
        V::Float64(-0.0),
        V::Float(f32::from_bits(1)),
        V::SmallInt(1),
    ];
    let mut graph = Graph::new();
    graph.add_nodes(values.len()).unwrap();
    graph
        .set_node_attrs(std::collections::HashMap::from([(
            "score".into(),
            values.iter().cloned().enumerate().collect(),
        )]))
        .unwrap();
    let view = graph.query().unwrap();
    for operator in ["<", "<=", ">", ">="] {
        for literal in [
            "-9223372036854775808",
            "9223372036854775807",
            "9007199254740993",
            "9007199254740992.0",
            "0",
            "1.0e-45f32",
        ] {
            let array = QueryEvaluator::new(&format!("value {operator} {literal}"));
            let expected = values
                .iter()
                .enumerate()
                .filter_map(|(id, value)| array.evaluate(value).unwrap().then_some(id))
                .collect::<Vec<_>>();
            let predicate = QueryParser::new()
                .parse_node_query(&format!("score {operator} {literal}"))
                .unwrap();
            assert_eq!(
                view.query_nodes(&predicate, &QueryOptions::default())
                    .unwrap(),
                expected,
                "{operator} {literal}"
            );
        }
    }
}

#[test]
fn scalar_extensions_do_not_silently_extend_graph_query_language() {
    use groggy::query::query_parser::QueryParser;
    assert!(QueryParser::new()
        .parse_node_query("score % 2 == 0")
        .is_err());
    assert!(QueryParser::new()
        .parse_edge_query("score % 2 == 0")
        .is_err());
    assert!(QueryEvaluator::new("value != NaN")
        .evaluate(&V::Int(2))
        .unwrap());
    assert!(QueryEvaluator::new("value == inf")
        .evaluate(&V::Float64(f64::INFINITY))
        .unwrap());
    assert!(QueryEvaluator::new("value > -inf")
        .evaluate(&V::Int(2))
        .unwrap());
    assert!(QueryParser::new().parse_node_query("score == inf").is_err());
    assert!(QueryEvaluator::new("value == 'NaN'")
        .evaluate(&V::Text("NaN".into()))
        .unwrap());
}

#[test]
fn typed_numeric_lowering_matches_generic_predicates_bit_for_bit() {
    let integers = vec![
        i64::MIN,
        -9007199254740993,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        9007199254740993,
        i64::MAX,
    ];
    let floats = vec![
        f64::NEG_INFINITY,
        -9007199254740992.0,
        -2f64.powi(63),
        2f64.powi(63),
        -2.5,
        -2.0,
        -0.0,
        0.0,
        f64::from_bits(1),
        2.0,
        2.5,
        9007199254740992.0,
        f64::INFINITY,
        f64::from_bits(0x7ff8000000000042),
    ];
    let queries = [
        "true",
        "false",
        "value",
        "length",
        "length == 0",
        "length > -1",
        "value == 2.0",
        "value == 9223372036854775807.0",
        "value >= 9223372036854775807",
        "value > -9223372036854775808.0",
        "value == 9007199254740993.0",
        "value == 1.0000000000000002",
        "value == -0.0",
        "value != 2",
        "value >= 9007199254740993",
        "value < -9007199254740993.0",
        "value % 2 == 0",
        "NOT (value % 2 == 0)",
        "value > 0 AND value <= 3",
        "false OR (value % 2 == 0 AND NOT value == 2)",
        "value == '2'",
        "value != true",
        "value == null",
        "value == inf",
        "value != NaN",
        "value >= -inf",
        "NOT value > 0",
    ];
    for query in queries {
        let generic = QueryEvaluator::new(query);
        let expected = integers
            .iter()
            .copied()
            .filter(|&value| generic.evaluate(&V::Int(value)).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            LazyArrayIterator::new(integers.clone())
                .filter(query)
                .collect()
                .unwrap(),
            expected,
            "{query}"
        );
        let expected = floats
            .iter()
            .copied()
            .filter(|&value| generic.evaluate(&V::Float64(value)).unwrap())
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        let actual = LazyArrayIterator::new(floats.clone())
            .filter(query)
            .collect()
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{query}");
        let narrow = [0x00000001, 0x80000000, 0x3f800001, 0x7f800000, 0x7fc00042]
            .map(f32::from_bits)
            .to_vec();
        let expected = narrow
            .iter()
            .copied()
            .filter(|&value| generic.evaluate(&V::Float(value)).unwrap())
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let actual = LazyArrayIterator::new(narrow)
            .filter(query)
            .collect()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "f32 {query}");
        let mut heterogeneous = integers.iter().copied().map(V::Int).collect::<Vec<_>>();
        heterogeneous.extend(floats.iter().copied().map(V::Float64));
        heterogeneous.extend([
            V::Bool(true),
            V::Bool(false),
            V::Null,
            V::Text("2".into()),
            V::IntVec(vec![1, 2]),
        ]);
        let expected = heterogeneous
            .iter()
            .filter(|value| generic.evaluate(value).unwrap())
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>();
        let actual = LazyArrayIterator::new(heterogeneous)
            .filter(query)
            .collect()
            .unwrap()
            .iter()
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{query}");
    }
}
