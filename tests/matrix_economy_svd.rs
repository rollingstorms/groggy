use groggy::storage::GraphMatrix;
fn matrix(data: Vec<f64>, m: usize, n: usize) -> GraphMatrix<f64> {
    GraphMatrix::from_row_major_data(data, m, n, None).unwrap()
}
#[test]
fn economy_factors_match_full_requested_columns_and_reconstruct_rectangles() {
    for (m, n) in [(7, 3), (3, 7), (3, 3), (0, 4), (4, 0)] {
        for zero in [false, true] {
            let a = matrix(
                (0..m * n)
                    .map(|i| {
                        if zero {
                            0.
                        } else {
                            ((i * 13 % 29) as f64 - 14.) / 19.
                        }
                    })
                    .collect(),
                m,
                n,
            );
            let (u, s, vt) = a.svd_thin().unwrap();
            let (fu, fs, fvt) = a.svd().unwrap();
            let k = m.min(n);
            assert_eq!((u.shape(), vt.shape()), ((m, k), (k, n)));
            assert_eq!(s, fs);
            for row in 0..m {
                for col in 0..k {
                    assert_eq!(u.get(row, col), fu.get(row, col));
                }
            }
            for row in 0..k {
                for col in 0..n {
                    assert_eq!(vt.get(row, col), fvt.get(row, col));
                }
            }
            for row in 0..m {
                for col in 0..n {
                    let actual: f64 = (0..k)
                        .map(|i| u.get(row, i).unwrap() * s[i] * vt.get(i, col).unwrap())
                        .sum();
                    assert!((actual - a.get(row, col).unwrap()).abs() < 1e-12);
                }
            }
        }
    }
}
#[test]
fn tall_rank_condition_least_squares_and_thin_tracking_avoid_square_null_bases() {
    let m = 10000;
    let a = matrix((0..m).flat_map(|i| [1., (i % 17) as f64]).collect(), m, 2);
    let (u, s, vt) = a.svd_thin().unwrap();
    assert_eq!((u.shape(), s.len(), vt.shape()), ((m, 2), 2, (2, 2)));
    assert_eq!(a.rank().unwrap(), 2);
    assert!(a.condition_number().unwrap().is_finite());
    let b = matrix((0..m).map(|i| 2. - (i % 17) as f64).collect(), m, 1);
    let x = a.least_squares(&b, None).unwrap();
    assert!((x.get(0, 0).unwrap() - 2.).abs() < 1e-12);
    assert!((x.get(1, 0).unwrap() + 1.).abs() < 1e-12);
    let a = a.requires_grad(true);
    let (_, mut s, _) = a.svd_tracked().unwrap();
    s.backward().unwrap();
    assert_eq!(a.grad().unwrap().shape(), (m, 2));
}
