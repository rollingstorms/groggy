use groggy::viz::projection::{
    algorithms::TSNEProjection, ProjectionConfig, ProjectionEngine, ProjectionEngineFactory,
    ProjectionMethod,
};
use groggy::{Graph, GraphMatrix};
fn input(scale: f64) -> GraphMatrix {
    GraphMatrix::from_row_major_data(
        [0., 0., 1., 0., 0., 2., 3., 1.]
            .iter()
            .map(|v| v * scale)
            .collect(),
        4,
        2,
        None,
    )
    .unwrap()
}
#[test]
fn seeded_output_preserves_identity_ownership_and_uniform_scale() {
    let engine = TSNEProjection::new(1.7, 50, 100., 1.).with_seed(19);
    let mut matrix = input(1.0);
    matrix.set_row_labels(Some(vec![
        "node_9".into(),
        "node_3".into(),
        "node_1".into(),
        "node_7".into(),
    ]));
    let before = matrix.to_vec().unwrap();
    fastrand::seed(73);
    let next = fastrand::u64(..);
    fastrand::seed(73);
    let output = engine.project_matrix(&matrix).unwrap();
    assert_eq!(fastrand::u64(..), next);
    assert_eq!(output.row_labels(), matrix.row_labels());
    assert_eq!(output.column_names(), ["tsne_x", "tsne_y"]);
    assert_eq!(matrix.to_vec().unwrap(), before);
    assert_eq!(
        output.to_vec().unwrap(),
        engine.project_matrix(&matrix).unwrap().to_vec().unwrap()
    );
    for scale in [1e-200, 1e200] {
        let other = engine.project_matrix(&input(scale)).unwrap();
        for (a, b) in output.to_vec().unwrap().iter().zip(other.to_vec().unwrap()) {
            assert!((a - b).abs() < 1e-8);
        }
    }
    let positions = engine.project_embedding(&matrix, &Graph::new()).unwrap();
    for (row, position) in positions.iter().enumerate() {
        assert_eq!(position.x, output.get(row, 0).unwrap());
        assert_eq!(position.y, output.get(row, 1).unwrap());
    }
    let config = ProjectionConfig {
        method: ProjectionMethod::TSNE {
            perplexity: 1.7,
            iterations: 50,
            learning_rate: 100.,
            early_exaggeration: 1.,
        },
        seed: Some(19),
        ..Default::default()
    };
    let factory = ProjectionEngineFactory::create_engine(&config)
        .unwrap()
        .project_embedding(&matrix, &Graph::new())
        .unwrap();
    for (a, b) in factory.iter().zip(positions) {
        assert_eq!((a.x, a.y), (b.x, b.y));
    }
}
#[test]
fn invalid_parameters_shapes_and_nonfinite_inputs_fail_even_without_updates() {
    for p in [0., -1., 4., f64::NAN, f64::INFINITY] {
        assert!(TSNEProjection::new(p, 0, 100., 1.)
            .project_matrix(&input(1.))
            .is_err());
    }
    for lr in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(TSNEProjection::new(1.7, 0, lr, 1.)
            .project_matrix(&input(1.))
            .is_err());
    }
    for ex in [0., 0.5, f64::NAN, f64::INFINITY] {
        assert!(TSNEProjection::new(1.7, 0, 100., ex)
            .project_matrix(&input(1.))
            .is_err());
    }
    let engine = TSNEProjection::new(1., 0, 100., 1.).with_seed(0);
    for shape in [(0, 2), (1, 2), (3, 0), (3, 1)] {
        assert!(engine
            .project_matrix(&GraphMatrix::zeros(shape.0, shape.1))
            .is_err());
    }
    for value in [f64::NAN, f64::INFINITY] {
        let mut matrix = input(1.);
        matrix.set(0, 0, value).unwrap();
        assert!(engine.project_matrix(&matrix).is_err());
    }
    let mut matrix = input(1.);
    matrix.set_row_labels(Some(vec!["short".into()]));
    assert!(engine.project_matrix(&matrix).is_err());
    assert!(engine.project_matrix(&GraphMatrix::zeros(4, 2)).is_err());
    let constant = TSNEProjection::new(3., 0, 100., 1.)
        .project_matrix(&GraphMatrix::zeros(4, 2))
        .unwrap();
    assert!(constant.to_vec().unwrap().iter().all(|x| x.is_finite()));
    let two = engine.project_matrix(&GraphMatrix::zeros(2, 2)).unwrap();
    assert_eq!(two.shape(), (2, 2));
}
