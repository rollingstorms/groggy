use groggy::viz::projection::{algorithms::PCAProjection, ProjectionEngine};
use groggy::{Graph, GraphMatrix};
fn matrix(data: &[f64], rows: usize, cols: usize) -> GraphMatrix {
    GraphMatrix::from_row_major_data(data.to_vec(), rows, cols, None).unwrap()
}
#[test]
fn uniform_extreme_scaling_preserves_projected_geometry() {
    let base = [3., 3., 0., -3., -3., 0., 1., -1., 0., -1., 1., 0.];
    for scale in [1e-200, 1., 1e200] {
        for standardize in [false, true] {
            let data: Vec<_> = base.iter().map(|v| v * scale).collect();
            let input = matrix(&data, 4, 3);
            let output = PCAProjection::new(true, standardize)
                .project_matrix(&input)
                .unwrap();
            assert_eq!(input.to_vec().unwrap(), data);
            let expected_scale = if standardize { 1. } else { scale };
            for i in 0..4 {
                for j in 0..4 {
                    let want: f64 = (0..3)
                        .map(|k| {
                            let divisor = if standardize && k < 2 {
                                5f64.sqrt()
                            } else {
                                1.
                            };
                            ((base[i * 3 + k] - base[j * 3 + k]) / divisor).powi(2)
                        })
                        .sum();
                    let actual: f64 = (0..2)
                        .map(|k| {
                            ((output.get(i, k).unwrap() - output.get(j, k).unwrap())
                                / expected_scale)
                                .powi(2)
                        })
                        .sum();
                    assert!(
                        (actual - want).abs() < 1e-10,
                        "scale={scale}, standardize={standardize}, pair={i},{j}"
                    );
                }
            }
        }
    }
}
#[test]
fn singleton_constant_and_uncentered_projection_are_defined() {
    let singleton = matrix(&[3., 4., 0.], 1, 3);
    for center in [false, true] {
        for standardize in [false, true] {
            let output = PCAProjection::new(center, standardize)
                .project_matrix(&singleton)
                .unwrap();
            let norm: f64 = (0..2).map(|c| output.get(0, c).unwrap().powi(2)).sum();
            assert!((norm - if center { 0. } else { 25. }).abs() < 1e-10);
        }
    }
    let constants = matrix(&[1., 1., 1., 1., 1., 1.], 3, 2);
    let out = PCAProjection::new(true, true)
        .project_matrix(&constants)
        .unwrap();
    assert_eq!(out.to_vec().unwrap(), vec![0.; 6]);
}
#[test]
fn projection_preserves_row_identity_and_trait_agrees_with_matrix_scores() {
    let mut input = matrix(&[1., 0., 0., 0., 2., 0., 0., 0., 3.], 3, 3);
    input.set_row_labels(Some(vec![
        "node_9".into(),
        "node_2".into(),
        "node_5".into(),
    ]));
    let engine = PCAProjection::new(false, false);
    let out = engine.project_matrix(&input).unwrap();
    assert_eq!(out.row_labels(), input.row_labels());
    assert_eq!(out.column_names(), ["pc_1", "pc_2"]);
    let positions = engine.project_embedding(&input, &Graph::new()).unwrap();
    for (i, p) in positions.iter().enumerate() {
        assert_eq!(p.x, out.get(i, 0).unwrap());
        assert_eq!(p.y, out.get(i, 1).unwrap());
    }
    let gram = |i, j| {
        out.get(i, 0).unwrap() * out.get(j, 0).unwrap()
            + out.get(i, 1).unwrap() * out.get(j, 1).unwrap()
    };
    assert!(gram(0, 0).abs() < 1e-12);
    assert!((gram(1, 1) - 4.).abs() < 1e-12);
    assert!((gram(2, 2) - 9.).abs() < 1e-12);
}
#[test]
fn invalid_shapes_nonfinite_inputs_and_unrepresentable_scores_reject() {
    for (data, rows, cols) in [
        (vec![], 0, 2),
        (vec![1., 2.], 2, 1),
        (vec![f64::NAN, 1.], 1, 2),
        (vec![f64::INFINITY, 1.], 1, 2),
    ] {
        assert!(PCAProjection::new(true, true)
            .project_matrix(&matrix(&data, rows, cols))
            .is_err());
    }
    let too_large = matrix(&[f64::MAX, f64::MAX], 1, 2);
    assert!(PCAProjection::new(false, false)
        .project_matrix(&too_large)
        .is_err());
}
