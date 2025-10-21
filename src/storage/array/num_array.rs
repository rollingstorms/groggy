use super::simd_optimizations;
use super::BaseArray;
use std::ops::{Add, Mul};

/// NumArray extends BaseArray with statistical and numerical operations.
/// This is used by TableArray and MatrixArray for numerical computations.
#[derive(Debug, Clone)]
pub struct NumArray<T> {
    pub base: BaseArray<T>,
}

impl<T> NumArray<T> {
    /// Create a new NumArray from a vector
    pub fn new(data: Vec<T>) -> Self {
        Self {
            base: BaseArray::new(data),
        }
    }

    /// Create a NumArray from an existing BaseArray
    pub fn from_base(base: BaseArray<T>) -> Self {
        Self { base }
    }

    // Delegate all basic operations to BaseArray

    /// Get the number of elements in the array
    pub fn len(&self) -> usize {
        self.base.len()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.base.is_empty()
    }

    /// Get a reference to an element at the given index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.base.get(index)
    }

    /// Get an iterator over the elements
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.base.iter()
    }

    /// Get the first element
    pub fn first(&self) -> Option<&T> {
        self.base.first()
    }

    /// Get the last element
    pub fn last(&self) -> Option<&T> {
        self.base.last()
    }
}

// Statistical operations - only available for numerical types
impl<T> NumArray<T>
where
    T: Copy + Add<Output = T> + Mul<f64, Output = T> + PartialOrd + Into<f64>,
{
    /// Calculate the mean (average) of all elements
    pub fn mean(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        let sum: f64 = self.iter().map(|&x| x.into()).sum();
        Some(sum / self.len() as f64)
    }

    /// Calculate the sum of all elements
    pub fn sum(&self) -> f64 {
        self.iter().map(|&x| x.into()).sum()
    }

    /// Find the minimum value
    pub fn min(&self) -> Option<f64> {
        self.iter()
            .map(|&x| x.into())
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Find the maximum value
    pub fn max(&self) -> Option<f64> {
        self.iter()
            .map(|&x| x.into())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Calculate the standard deviation (sample standard deviation)
    pub fn std_dev(&self) -> Option<f64> {
        let mean = self.mean()?;
        if self.len() <= 1 {
            return Some(0.0);
        }

        let variance = self
            .iter()
            .map(|&x| {
                let diff = x.into() - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.len() - 1) as f64;

        Some(variance.sqrt())
    }

    /// Calculate the variance (sample variance)
    pub fn variance(&self) -> Option<f64> {
        let mean = self.mean()?;
        if self.len() <= 1 {
            return Some(0.0);
        }

        let variance = self
            .iter()
            .map(|&x| {
                let diff = x.into() - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.len() - 1) as f64;

        Some(variance)
    }

    /// Calculate the median value
    pub fn median(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }

        let mut sorted: Vec<f64> = self.iter().map(|&x| x.into()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            Some((sorted[mid - 1] + sorted[mid]) / 2.0)
        } else {
            Some(sorted[mid])
        }
    }

    /// Calculate the percentile (0.0 to 1.0)
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.is_empty() || !(0.0..=1.0).contains(&p) {
            return None;
        }

        let mut sorted: Vec<f64> = self.iter().map(|&x| x.into()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if p == 0.0 {
            return Some(sorted[0]);
        }
        if p == 1.0 {
            return Some(sorted[sorted.len() - 1]);
        }

        let index = p * (sorted.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            Some(sorted[lower])
        } else {
            let weight = index - lower as f64;
            Some(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
        }
    }

    /// Calculate correlation with another NumArray
    pub fn correlate(&self, other: &Self) -> Option<f64> {
        if self.len() != other.len() || self.is_empty() {
            return None;
        }

        let self_mean = self.mean()?;
        let other_mean = other.mean()?;

        let numerator: f64 = self
            .iter()
            .zip(other.iter())
            .map(|(&x, &y)| (x.into() - self_mean) * (y.into() - other_mean))
            .sum();

        let self_var: f64 = self.iter().map(|&x| (x.into() - self_mean).powi(2)).sum();

        let other_var: f64 = other.iter().map(|&y| (y.into() - other_mean).powi(2)).sum();

        let denominator = (self_var * other_var).sqrt();
        if denominator == 0.0 {
            None
        } else {
            Some(numerator / denominator)
        }
    }

    /// Element-wise addition with another NumArray
    pub fn add(&self, other: &Self) -> Option<NumArray<T>>
    where
        T: Add<Output = T>,
    {
        if self.len() != other.len() {
            return None;
        }

        let result_data: Vec<T> = self
            .iter()
            .zip(other.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Some(NumArray::new(result_data))
    }

    /// Multiply all elements by a scalar
    pub fn multiply(&self, scalar: f64) -> NumArray<T>
    where
        T: Mul<f64, Output = T>,
    {
        let result_data: Vec<T> = self.iter().map(|&x| x * scalar).collect();

        NumArray::new(result_data)
    }

    /// Calculate descriptive statistics summary
    pub fn describe(&self) -> StatsSummary {
        StatsSummary {
            count: self.len(),
            mean: self.mean(),
            std_dev: self.std_dev(),
            min: self.min(),
            percentile_25: self.percentile(0.25),
            median: self.median(),
            percentile_75: self.percentile(0.75),
            max: self.max(),
        }
    }
}

/// Summary statistics structure
#[derive(Debug, Clone)]
pub struct StatsSummary {
    pub count: usize,
    pub mean: Option<f64>,
    pub std_dev: Option<f64>,
    pub min: Option<f64>,
    pub percentile_25: Option<f64>,
    pub median: Option<f64>,
    pub percentile_75: Option<f64>,
    pub max: Option<f64>,
}

impl StatsSummary {
    pub fn is_valid(&self) -> bool {
        self.count > 0
    }
}

impl<T> From<Vec<T>> for NumArray<T> {
    fn from(data: Vec<T>) -> Self {
        Self::new(data)
    }
}

impl<T> From<BaseArray<T>> for NumArray<T> {
    fn from(base: BaseArray<T>) -> Self {
        Self::from_base(base)
    }
}

impl<T> From<NumArray<T>> for BaseArray<T> {
    fn from(num_array: NumArray<T>) -> Self {
        num_array.base
    }
}

// SIMD-optimized implementations specifically for f64 arrays
impl NumArray<f64> {
    /// SIMD-optimized sum for f64 arrays
    pub fn sum_simd(&self) -> f64 {
        simd_optimizations::simd_sum(self.base.as_slice())
    }

    /// SIMD-optimized mean for f64 arrays
    pub fn mean_simd(&self) -> Option<f64> {
        simd_optimizations::simd_mean(self.base.as_slice())
    }

    /// SIMD-optimized standard deviation for f64 arrays
    pub fn std_dev_simd(&self) -> Option<f64> {
        simd_optimizations::simd_std_dev(self.base.as_slice())
    }

    /// SIMD-optimized variance for f64 arrays
    pub fn variance_simd(&self) -> Option<f64> {
        self.mean_simd()
            .map(|mean| simd_optimizations::simd_variance(self.base.as_slice(), mean))
    }

    /// SIMD-optimized min for f64 arrays
    pub fn min_simd(&self) -> Option<f64> {
        simd_optimizations::simd_min(self.base.as_slice())
    }

    /// SIMD-optimized max for f64 arrays
    pub fn max_simd(&self) -> Option<f64> {
        simd_optimizations::simd_max(self.base.as_slice())
    }

    /// Optimized median using quickselect algorithm for f64 arrays
    pub fn median_optimized(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }

        // Make a mutable copy for quickselect
        let mut data: Vec<f64> = self.base.as_slice().to_vec();
        simd_optimizations::quickselect_median(&mut data)
    }

    /// Auto-choose between regular and SIMD implementations based on array size
    pub fn sum_auto(&self) -> f64 {
        if self.len() >= 8 {
            self.sum_simd()
        } else {
            self.sum()
        }
    }

    /// Auto-choose mean implementation based on array size
    pub fn mean_auto(&self) -> Option<f64> {
        if self.len() >= 8 {
            self.mean_simd()
        } else {
            self.mean()
        }
    }

    /// Auto-choose median implementation based on array size
    pub fn median_auto(&self) -> Option<f64> {
        if self.len() >= 100 {
            // Use quickselect for larger arrays
            self.median_optimized()
        } else {
            // Use regular sort for smaller arrays
            self.median()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let num_array = NumArray::new(data);

        assert_eq!(num_array.len(), 5);
        assert_eq!(num_array.mean(), Some(3.0));
        assert_eq!(num_array.sum(), 15.0);
        assert_eq!(num_array.min(), Some(1.0));
        assert_eq!(num_array.max(), Some(5.0));
        assert_eq!(num_array.median(), Some(3.0));
    }

    #[test]
    fn test_correlation() {
        let x = NumArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = NumArray::new(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let correlation = x.correlate(&y);
        assert!(correlation.is_some());
        // Perfect positive correlation should be close to 1.0
        assert!((correlation.unwrap() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_element_wise_operations() {
        let a = NumArray::new(vec![1.0, 2.0, 3.0]);
        let b = NumArray::new(vec![4.0, 5.0, 6.0]);

        let sum = a.add(&b);
        assert!(sum.is_some());
        assert_eq!(sum.unwrap().base.clone_vec(), vec![5.0, 7.0, 9.0]);

        let scaled = a.multiply(2.0);
        assert_eq!(scaled.base.clone_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_empty_num_array() {
        let empty: NumArray<f64> = NumArray::new(vec![]);

        assert!(empty.is_empty());
        assert_eq!(empty.mean(), None);
        assert_eq!(empty.min(), None);
        assert_eq!(empty.max(), None);
    }

    #[test]
    fn test_describe() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let num_array = NumArray::new(data);
        let summary = num_array.describe();

        assert_eq!(summary.count, 5);
        assert!(summary.is_valid());
        assert_eq!(summary.mean, Some(3.0));
        assert_eq!(summary.median, Some(3.0));
    }
}
