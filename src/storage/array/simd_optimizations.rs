//! Compatibility entry points sharing checked reference reductions.
//! Acceleration is deferred until it preserves these semantics and is measured.
use super::numeric_reductions as kernels;
/// Compensated reference sum behind the historical SIMD name.
pub fn simd_sum(data: &[f64]) -> f64 {
    kernels::sum(data)
}
/// Shared reference mean.
pub fn simd_mean(data: &[f64]) -> Option<f64> {
    kernels::mean(data)
}
/// Sample moment around the caller's explicitly supplied center.
pub fn simd_variance(data: &[f64], mean: f64) -> f64 {
    kernels::variance_with_mean(data, mean)
}
/// Shared sample standard deviation.
pub fn simd_std_dev(data: &[f64]) -> Option<f64> {
    kernels::std_dev(data)
}
/// Shared minimum with NaN propagation.
pub fn simd_min(data: &[f64]) -> Option<f64> {
    kernels::extremum(data, true)
}
/// Shared maximum with NaN propagation.
pub fn simd_max(data: &[f64]) -> Option<f64> {
    kernels::extremum(data, false)
}
/// In-place selection using bounded-stack partitioning and a safe midpoint.
pub fn quickselect_median(data: &mut [f64]) -> Option<f64> {
    kernels::select_median(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = simd_sum(&data);
        let expected: f64 = data.iter().sum();
        assert!((result - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simd_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = simd_mean(&data).unwrap();
        assert!((result - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simd_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = simd_mean(&data).unwrap();
        let result = simd_variance(&data, mean);

        // Expected variance for this dataset
        let expected = 2.5; // Sample variance
        assert!((result - expected).abs() < 0.001);
    }

    #[test]
    fn test_simd_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

        let min_result = simd_min(&data).unwrap();
        let max_result = simd_max(&data).unwrap();

        assert_eq!(min_result, 1.0);
        assert_eq!(max_result, 9.0);
    }

    #[test]
    fn test_quickselect_median() {
        let mut data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let result = quickselect_median(&mut data).unwrap();

        // Median should be 3.0
        assert_eq!(result, 3.0);

        // Test even length
        let mut data2 = vec![1.0, 2.0, 3.0, 4.0];
        let result2 = quickselect_median(&mut data2).unwrap();
        assert_eq!(result2, 2.5); // (2 + 3) / 2
    }
}
