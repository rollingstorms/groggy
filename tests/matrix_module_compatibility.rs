//! Old and canonical imports must name the same implementation, not wrappers.
use groggy::storage::{advanced_matrix as legacy, matrix};
use std::any::TypeId;
#[test]
fn canonical_and_legacy_matrix_types_traits_and_nested_modules_are_identical() {
    assert_eq!(
        TypeId::of::<matrix::UnifiedMatrix<f64>>(),
        TypeId::of::<legacy::UnifiedMatrix<f64>>()
    );
    assert_eq!(
        TypeId::of::<matrix::autodiff::Operation>(),
        TypeId::of::<legacy::neural::autodiff::Operation>()
    );
    assert_eq!(
        TypeId::of::<matrix::autodiff::ComputationGraph<f64>>(),
        TypeId::of::<matrix::neural::autodiff::ComputationGraph<f64>>()
    );
    assert_eq!(
        TypeId::of::<matrix::adjacency::MatrixFormat>(),
        TypeId::of::<groggy::storage::adjacency::MatrixFormat>()
    );
    let lower = legacy::unified_matrix::Matrix64::from_data(vec![1., 2., 3., 4.], 2, 2).unwrap();
    let canonical: matrix::UnifiedMatrix<f64> = lower;
    let graph = matrix::GraphMatrix::try_from_storage(canonical).unwrap();
    assert_eq!(graph.to_vec().unwrap(), vec![1., 2., 3., 4.]);
    fn legacy_numeric<T: legacy::numeric_type::NumericType>(value: T) -> T {
        value
    }
    fn canonical_numeric<T: matrix::NumericType>(value: T) -> T {
        legacy_numeric(value)
    }
    assert_eq!(canonical_numeric(7i64), 7);
    assert_eq!(
        TypeId::of::<legacy::backend::BackendHint>(),
        TypeId::of::<matrix::backend::BackendHint>()
    );
    assert_eq!(
        TypeId::of::<legacy::backends::BlasBackend>(),
        TypeId::of::<matrix::backends::BlasBackend>()
    );
    assert_eq!(
        TypeId::of::<legacy::memory::MatrixLayout>(),
        TypeId::of::<matrix::memory::MatrixLayout>()
    );
    assert_eq!(
        TypeId::of::<legacy::operations::MatrixStats>(),
        TypeId::of::<matrix::operations::MatrixStats>()
    );
}
