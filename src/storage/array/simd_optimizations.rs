//! SIMD Optimizations for NumArray Statistical Operations
//!
//! This module provides vectorized implementations of statistical operations
//! for improved performance on numerical data arrays.

use wide::f64x4; // 4-way SIMD for f64

/// SIMD-optimized sum calculation
pub fn simd_sum(data: &[f64]) -> f64 {
    if data.len() < 4 {
        // Fall back to scalar for small arrays
        return data.iter().sum();
    }

    let chunks = data.len() / 4;
    let _remainder = data.len() % 4;

    // Process 4 elements at a time with SIMD
    let mut simd_sum = f64x4::ZERO;

    for i in 0..chunks {
        let chunk_start = i * 4;
        let chunk = f64x4::new([
            data[chunk_start],
            data[chunk_start + 1],
            data[chunk_start + 2],
            data[chunk_start + 3],
        ]);
        simd_sum += chunk;
    }

    // Sum the SIMD register and add remainder
    let simd_total = simd_sum.to_array().iter().sum::<f64>();
    let remainder_sum: f64 = data[(chunks * 4)..].iter().sum();

    simd_total + remainder_sum
}

/// SIMD-optimized mean calculation  
pub fn simd_mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    Some(simd_sum(data) / data.len() as f64)
}

/// SIMD-optimized variance calculation
pub fn simd_variance(data: &[f64], mean: f64) -> f64 {
    if data.len() <= 1 {
        return 0.0;
    }

    if data.len() < 4 {
        // Fall back to scalar for small arrays
        return data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>()
            / (data.len() - 1) as f64;
    }

    let chunks = data.len() / 4;
    let _remainder = data.len() % 4;

    let mean_vec = f64x4::splat(mean);
    let mut variance_sum = f64x4::ZERO;

    for i in 0..chunks {
        let chunk_start = i * 4;
        let chunk = f64x4::new([
            data[chunk_start],
            data[chunk_start + 1],
            data[chunk_start + 2],
            data[chunk_start + 3],
        ]);

        let diff = chunk - mean_vec;
        let squared_diff = diff * diff;
        variance_sum += squared_diff;
    }

    // Sum the SIMD register and add remainder
    let simd_variance = variance_sum.to_array().iter().sum::<f64>();
    let remainder_variance: f64 = data[(chunks * 4)..]
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum();

    (simd_variance + remainder_variance) / (data.len() - 1) as f64
}

/// SIMD-optimized standard deviation calculation
pub fn simd_std_dev(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let mean = simd_mean(data)?;
    Some(simd_variance(data, mean).sqrt())
}

/// SIMD-optimized min calculation
pub fn simd_min(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    if data.len() < 4 {
        return data.iter().cloned().reduce(f64::min);
    }

    let chunks = data.len() / 4;
    let _remainder = data.len() % 4;

    // Initialize with first chunk
    let mut min_vec = f64x4::new([data[0], data[1], data[2], data[3]]);

    // Process remaining chunks
    for i in 1..chunks {
        let chunk_start = i * 4;
        let chunk = f64x4::new([
            data[chunk_start],
            data[chunk_start + 1],
            data[chunk_start + 2],
            data[chunk_start + 3],
        ]);

        // Element-wise min
        min_vec = min_vec.min(chunk);
    }

    // Find min in the SIMD register
    let mut min_val = min_vec.to_array().iter().cloned().reduce(f64::min).unwrap();

    // Process remainder
    for &val in &data[(chunks * 4)..] {
        min_val = min_val.min(val);
    }

    Some(min_val)
}

/// SIMD-optimized max calculation
pub fn simd_max(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    if data.len() < 4 {
        return data.iter().cloned().reduce(f64::max);
    }

    let chunks = data.len() / 4;
    let _remainder = data.len() % 4;

    // Initialize with first chunk
    let mut max_vec = f64x4::new([data[0], data[1], data[2], data[3]]);

    // Process remaining chunks
    for i in 1..chunks {
        let chunk_start = i * 4;
        let chunk = f64x4::new([
            data[chunk_start],
            data[chunk_start + 1],
            data[chunk_start + 2],
            data[chunk_start + 3],
        ]);

        // Element-wise max
        max_vec = max_vec.max(chunk);
    }

    // Find max in the SIMD register
    let mut max_val = max_vec.to_array().iter().cloned().reduce(f64::max).unwrap();

    // Process remainder
    for &val in &data[(chunks * 4)..] {
        max_val = max_val.max(val);
    }

    Some(max_val)
}

/// Optimized partial sorting for median calculation
/// Uses quickselect algorithm which is O(n) average case vs O(n log n) for full sort
pub fn quickselect_median(data: &mut [f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    let len = data.len();

    if len % 2 == 1 {
        // Odd length - find middle element
        let median_idx = len / 2;
        Some(quickselect(data, median_idx))
    } else {
        // Even length - average of two middle elements
        let mid_high = len / 2;
        let mid_low = mid_high - 1;

        // Find both middle elements
        let low_val = quickselect(data, mid_low);
        let high_val = quickselect(data, mid_high);

        Some((low_val + high_val) / 2.0)
    }
}

/// Quickselect implementation for finding kth smallest element
fn quickselect(data: &mut [f64], k: usize) -> f64 {
    if data.len() == 1 {
        return data[0];
    }

    let pivot_idx = partition(data);

    if k == pivot_idx {
        data[k]
    } else if k < pivot_idx {
        quickselect(&mut data[0..pivot_idx], k)
    } else {
        quickselect(&mut data[pivot_idx + 1..], k - pivot_idx - 1)
    }
}

/// Partition function for quickselect
fn partition(data: &mut [f64]) -> usize {
    let pivot_idx = data.len() / 2;
    data.swap(pivot_idx, data.len() - 1);

    let pivot = data[data.len() - 1];
    let mut i = 0;

    for j in 0..data.len() - 1 {
        if data[j] <= pivot {
            data.swap(i, j);
            i += 1;
        }
    }

    data.swap(i, data.len() - 1);
    i
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
