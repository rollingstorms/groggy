use groggy::viz::projection::{
    algorithms::UMAPProjection, ProjectionConfig, ProjectionEngine, ProjectionEngineFactory,
    ProjectionMethod,
};
use groggy::{Graph, GraphMatrix};
fn input(scale: f64) -> GraphMatrix {
    GraphMatrix::from_row_major_data(
        [0., 0., 1., 0., 3., 0.].iter().map(|x| x * scale).collect(),
        3,
        2,
        None,
    )
    .unwrap()
}
#[test]
fn first_step_matches_balanced_neighbor_forces_and_preserves_centroid() {
    let matrix = input(1.0);
    let before = UMAPProjection::new(1, 0.0, 0, 0.0)
        .with_seed(19)
        .project_matrix(&matrix)
        .unwrap();
    let after = UMAPProjection::new(1, 0.0, 1, 0.0)
        .with_seed(19)
        .project_matrix(&matrix)
        .unwrap();
    // Exact kNN springs: 0->1, 1->0 and 2->1; all local weights equal one.
    for axis in 0..2 {
        let x: Vec<_> = (0..3).map(|row| before.get(row, axis).unwrap()).collect();
        let gradient = [
            2. * (x[0] - x[1]),
            2. * (x[1] - x[0]) + x[1] - x[2],
            x[2] - x[1],
        ];
        for row in 0..3 {
            assert!(
                (after.get(row, axis).unwrap() - (x[row] - 0.01 * gradient[row])).abs() < 1e-14
            );
        }
        assert!(
            ((0..3).map(|row| after.get(row, axis).unwrap()).sum::<f64>() - x.iter().sum::<f64>())
                .abs()
                < 1e-14
        );
    }
    // At very large min_dist there is no attractive penalty and no negatives.
    let flat = UMAPProjection::new(1, 100.0, 5, 0.0)
        .with_seed(19)
        .project_matrix(&matrix)
        .unwrap();
    assert_eq!(flat.to_vec().unwrap(), before.to_vec().unwrap());
    let fractional = UMAPProjection::new(1, 0.0, 1, 0.2)
        .with_seed(19)
        .project_matrix(&matrix)
        .unwrap();
    assert_eq!(fractional.to_vec().unwrap(), after.to_vec().unwrap()); // floor(3*0.2)=0
    let negative = UMAPProjection::new(1, 0.0, 1, 0.4)
        .with_seed(19)
        .project_matrix(&matrix)
        .unwrap();
    assert_ne!(negative.to_vec().unwrap(), after.to_vec().unwrap()); // one nonself sample
}
#[test]
fn owned_identity_seed_factory_and_uniform_extreme_scaling_are_consistent() {
    let mut matrix = input(1.0);
    matrix.set_row_labels(Some(vec![
        "node_8".into(),
        "node_2".into(),
        "node_5".into(),
    ]));
    let before = matrix.to_vec().unwrap();
    let engine = UMAPProjection::new(2, 0.1, 20, 5.0).with_seed(53);
    fastrand::seed(81);
    let expected = fastrand::u64(..);
    fastrand::seed(81);
    let output = engine.project_matrix(&matrix).unwrap();
    assert_eq!(fastrand::u64(..), expected);
    assert_eq!(output.row_labels(), matrix.row_labels());
    assert_eq!(output.column_names(), ["umap_x", "umap_y"]);
    assert_eq!(matrix.to_vec().unwrap(), before);
    assert_eq!(
        output.to_vec().unwrap(),
        engine.project_matrix(&matrix).unwrap().to_vec().unwrap()
    );
    for scale in [1e-200, 1e200] {
        let other = engine.project_matrix(&input(scale)).unwrap();
        for (a, b) in output.to_vec().unwrap().iter().zip(other.to_vec().unwrap()) {
            assert!((a - b).abs() < 1e-12);
        }
    }
    let config = ProjectionConfig {
        method: ProjectionMethod::UMAP {
            n_neighbors: 2,
            min_dist: 0.1,
            n_epochs: 20,
            negative_sample_rate: 5.0,
        },
        seed: Some(53),
        ..Default::default()
    };
    let factory = ProjectionEngineFactory::create_engine(&config)
        .unwrap()
        .project_embedding(&matrix, &Graph::new_manual())
        .unwrap();
    for (row, p) in factory.iter().enumerate() {
        assert_eq!(p.x, output.get(row, 0).unwrap());
        assert_eq!(p.y, output.get(row, 1).unwrap());
    }
    assert!(!engine.supports_incremental());
}
#[test]
fn invalid_controls_nonfinite_distances_and_sample_overflow_error_without_panics() {
    assert!(UMAPProjection::new(0, 0.1, 0, 5.0)
        .project_matrix(&input(1.))
        .is_err());
    for value in [-1., f64::NAN, f64::INFINITY] {
        assert!(UMAPProjection::new(1, value, 0, 5.0)
            .project_matrix(&input(1.))
            .is_err());
        assert!(UMAPProjection::new(1, 0.1, 0, value)
            .project_matrix(&input(1.))
            .is_err());
    }
    assert!(UMAPProjection::new(1, 0.1, 0, 1e300)
        .project_matrix(&input(1.))
        .is_err());
    let engine = UMAPProjection::new(usize::MAX, 0.1, 5, 0.0).with_seed(0);
    for value in [f64::NAN, f64::INFINITY] {
        let mut matrix = input(1.);
        matrix.set(0, 0, value).unwrap();
        assert!(engine.project_matrix(&matrix).is_err());
    }
    for shape in [(0, 2), (3, 0), (3, 1)] {
        assert!(engine
            .project_matrix(&GraphMatrix::zeros(shape.0, shape.1))
            .is_err());
    }
    let mut matrix = input(1.);
    matrix.set_row_labels(Some(vec!["short".into()]));
    assert!(engine.project_matrix(&matrix).is_err());
    assert_eq!(
        engine
            .project_matrix(&GraphMatrix::zeros(1, 2))
            .unwrap()
            .to_vec()
            .unwrap(),
        [0., 0.]
    );
    assert!(engine
        .project_matrix(&GraphMatrix::zeros(4, 2))
        .unwrap()
        .to_vec()
        .unwrap()
        .iter()
        .all(|x| x.is_finite()));
    assert!(engine.project_matrix(&input(1.)).is_ok()); // clamp k to N-1
}
