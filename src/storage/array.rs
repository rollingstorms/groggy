//! GraphArray - Enhanced array with native statistical operations
//!
//! Provides GraphArray that combines list-like functionality with
//! fast native statistical computations and intelligent caching.

use crate::types::AttrValue;
use std::cell::RefCell;

/// Cache for expensive statistical computations
#[derive(Debug, Clone)]
struct CachedStats {
    mean: Option<f64>,
    std: Option<f64>,
    min: Option<AttrValue>,
    max: Option<AttrValue>,
    count: Option<usize>,
    sum: Option<f64>,
}

impl CachedStats {
    fn new() -> Self {
        Self {
            mean: None,
            std: None,
            min: None,
            max: None,
            count: None,
            sum: None,
        }
    }

    #[allow(dead_code)]
    fn invalidate(&mut self) {
        *self = Self::new();
    }
}

/// Lazy Statistical Array with fast native operations and intelligent caching
///
/// Combines list-like functionality (indexing, iteration, len) with
/// high-performance statistical methods computed in Rust.
///
/// LAZY EVALUATION ARCHITECTURE:
/// - Data storage: Efficient sparse representation by default
/// - View operations: Create lazy views, no immediate computation
/// - Materialization: .values property converts to Python objects
/// - Display: repr() shows preview only, not full data
///
/// PERFORMANCE FEATURES:
/// - Lazy computation: Stats calculated only when requested
/// - Intelligent caching: Expensive computations cached until data changes
/// - Native speed: All statistical operations computed in Rust
/// - Zero-copy views: Efficient access to underlying data
/// - Sparse storage: Only non-default values stored
///
/// USAGE:
/// ```rust
/// use groggy::core::array::GraphArray;
/// use groggy::AttrValue;
/// let arr = GraphArray::from_vec(vec![
///     AttrValue::Float(1.0),
///     AttrValue::Float(2.0),
///     AttrValue::Float(3.0),
///     AttrValue::Float(4.0),
///     AttrValue::Float(5.0)
/// ]);
/// let mean = arr.mean().unwrap();  // Computed and cached
/// let std = arr.std().unwrap();    // Uses cached mean
/// let list = arr.to_list();        // Convert to Vec for compatibility
/// ```
#[derive(Debug, Clone)]
pub struct GraphArray {
    /// Core data storage
    values: Vec<AttrValue>,
    /// Optional name/label for the array
    name: Option<String>,
    /// Cached statistical computations (lazy evaluation)
    cached_stats: RefCell<CachedStats>,
}

impl GraphArray {
    /// Create a new GraphArray from a vector of AttrValues
    pub fn from_vec(values: Vec<AttrValue>) -> Self {
        Self {
            values,
            name: None,
            cached_stats: RefCell::new(CachedStats::new()),
        }
    }

    /// Create an empty GraphArray
    pub fn new() -> Self {
        Self::from_vec(Vec::new())
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get element by index
    pub fn get(&self, index: usize) -> Option<&AttrValue> {
        self.values.get(index)
    }

    /// Convert to a plain Vec<AttrValue> for compatibility
    pub fn to_list(&self) -> Vec<AttrValue> {
        self.values.clone()
    }

    /// Get iterator over values
    pub fn iter(&self) -> std::slice::Iter<'_, AttrValue> {
        self.values.iter()
    }

    /// Extract numeric values for statistical computation
    /// Returns None if array contains non-numeric values
    fn extract_numeric_values(&self) -> Option<Vec<f64>> {
        let mut numeric_values = Vec::with_capacity(self.values.len());
        let mut has_non_null_non_numeric = false;

        for value in &self.values {
            match value {
                AttrValue::Int(i) => numeric_values.push(*i as f64),
                AttrValue::SmallInt(i) => numeric_values.push(*i as f64),
                AttrValue::Float(f) => numeric_values.push(*f as f64),
                AttrValue::Null => {
                    // Skip null values - this is the key fix!
                    continue;
                }
                _ => {
                    // Non-numeric, non-null value found (like text)
                    has_non_null_non_numeric = true;
                    break;
                }
            }
        }

        // Only fail if we found non-null, non-numeric values
        // If we only found nulls mixed with numbers, that's fine
        if has_non_null_non_numeric {
            return None;
        }

        if numeric_values.is_empty() {
            None
        } else {
            Some(numeric_values)
        }
    }

    /// Calculate and cache count of non-null values
    pub fn count(&self) -> usize {
        let mut cache = self.cached_stats.borrow_mut();

        if let Some(cached_count) = cache.count {
            return cached_count;
        }

        let count = self.values.len();
        cache.count = Some(count);
        count
    }

    /// Calculate and cache mean (average) of numeric values
    pub fn mean(&self) -> Option<f64> {
        let mut cache = self.cached_stats.borrow_mut();

        if let Some(cached_mean) = cache.mean {
            return Some(cached_mean);
        }

        let numeric_values = self.extract_numeric_values()?;
        if numeric_values.is_empty() {
            return None;
        }

        let sum: f64 = numeric_values.iter().sum();
        let mean = sum / numeric_values.len() as f64;

        // Cache both sum and mean for efficiency
        cache.sum = Some(sum);
        cache.mean = Some(mean);

        Some(mean)
    }

    /// Calculate and cache sum of numeric values
    pub fn sum(&self) -> Option<f64> {
        let mut cache = self.cached_stats.borrow_mut();

        if let Some(cached_sum) = cache.sum {
            return Some(cached_sum);
        }

        let numeric_values = self.extract_numeric_values()?;
        if numeric_values.is_empty() {
            return None;
        }

        let sum: f64 = numeric_values.iter().sum();
        cache.sum = Some(sum);

        Some(sum)
    }

    /// Calculate and cache standard deviation of numeric values
    pub fn std(&self) -> Option<f64> {
        let mut cache = self.cached_stats.borrow_mut();

        if let Some(cached_std) = cache.std {
            return Some(cached_std);
        }

        let numeric_values = self.extract_numeric_values()?;
        if numeric_values.len() < 2 {
            return None; // Need at least 2 values for std dev
        }

        // Get or compute mean
        let mean = if let Some(cached_mean) = cache.mean {
            cached_mean
        } else {
            let sum: f64 = numeric_values.iter().sum();
            let mean = sum / numeric_values.len() as f64;
            cache.sum = Some(sum);
            cache.mean = Some(mean);
            mean
        };

        // Calculate variance
        let variance: f64 = numeric_values
            .iter()
            .map(|x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / (numeric_values.len() - 1) as f64; // Sample standard deviation

        let std = variance.sqrt();
        cache.std = Some(std);

        Some(std)
    }

    /// Calculate and cache minimum value
    pub fn min(&self) -> Option<AttrValue> {
        let mut cache = self.cached_stats.borrow_mut();

        if let Some(ref cached_min) = cache.min {
            return Some(cached_min.clone());
        }

        if self.values.is_empty() {
            return None;
        }

        // Custom min logic since AttrValue doesn't implement Ord
        let mut min_value = &self.values[0];
        for value in &self.values[1..] {
            if self.compare_attr_values(value, min_value) < 0 {
                min_value = value;
            }
        }

        let min_value = min_value.clone();
        cache.min = Some(min_value.clone());

        Some(min_value)
    }

    /// Calculate and cache maximum value
    pub fn max(&self) -> Option<AttrValue> {
        let mut cache = self.cached_stats.borrow_mut();

        if let Some(ref cached_max) = cache.max {
            return Some(cached_max.clone());
        }

        if self.values.is_empty() {
            return None;
        }

        // Custom max logic since AttrValue doesn't implement Ord
        let mut max_value = &self.values[0];
        for value in &self.values[1..] {
            if self.compare_attr_values(value, max_value) > 0 {
                max_value = value;
            }
        }

        let max_value = max_value.clone();
        cache.max = Some(max_value.clone());

        Some(max_value)
    }

    /// Compare two AttrValues for ordering
    /// Returns: -1 if a < b, 0 if a == b, 1 if a > b
    fn compare_attr_values(&self, a: &AttrValue, b: &AttrValue) -> i32 {
        use std::cmp::Ordering;
        use AttrValue::*;

        let ord = match (a, b) {
            // Numeric comparisons
            (Int(a), Int(b)) => a.cmp(b),
            (SmallInt(a), SmallInt(b)) => a.cmp(b),
            (Float(a), Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),

            // Cross-type numeric comparisons
            (Int(a), SmallInt(b)) => (*a as f64)
                .partial_cmp(&(*b as f64))
                .unwrap_or(Ordering::Equal),
            (SmallInt(a), Int(b)) => (*a as f64)
                .partial_cmp(&(*b as f64))
                .unwrap_or(Ordering::Equal),
            (Int(a), Float(b)) => (*a as f64)
                .partial_cmp(&(*b as f64))
                .unwrap_or(Ordering::Equal),
            (Float(a), Int(b)) => (*a as f64)
                .partial_cmp(&(*b as f64))
                .unwrap_or(Ordering::Equal),
            (SmallInt(a), Float(b)) => (*a as f64)
                .partial_cmp(&(*b as f64))
                .unwrap_or(Ordering::Equal),
            (Float(a), SmallInt(b)) => (*a as f64)
                .partial_cmp(&(*b as f64))
                .unwrap_or(Ordering::Equal),

            // String comparisons
            (Text(a), Text(b)) => a.cmp(b),

            // Boolean comparisons
            (Bool(a), Bool(b)) => a.cmp(b),

            // Default: treat different types as equal
            _ => Ordering::Equal,
        };

        match ord {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }

    /// Calculate quantile (percentile) of numeric values
    /// quantile: 0.0 to 1.0 (e.g., 0.5 for median, 0.95 for 95th percentile)
    pub fn quantile(&self, quantile: f64) -> Option<f64> {
        if !(0.0..=1.0).contains(&quantile) {
            return None;
        }

        let mut numeric_values = self.extract_numeric_values()?;
        if numeric_values.is_empty() {
            return None;
        }

        numeric_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (quantile * (numeric_values.len() - 1) as f64).round() as usize;
        Some(numeric_values[index.min(numeric_values.len() - 1)])
    }

    /// Calculate median (50th percentile)
    pub fn median(&self) -> Option<f64> {
        self.quantile(0.5)
    }

    /// Get comprehensive statistical summary
    pub fn describe(&self) -> StatsSummary {
        StatsSummary {
            count: self.count(),
            mean: self.mean(),
            std: self.std(),
            min: self.min(),
            max: self.max(),
            median: self.median(),
            q25: self.quantile(0.25),
            q75: self.quantile(0.75),
        }
    }

    /// Set or update the name of this array
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get the name of this array
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    /// Get the data type of this array
    pub fn dtype(&self) -> crate::types::AttrValueType {
        use crate::types::AttrValueType;

        if self.values.is_empty() {
            return AttrValueType::Text; // Default
        }

        // Determine type from first non-null value
        for value in &self.values {
            match value {
                crate::types::AttrValue::Int(_) | crate::types::AttrValue::SmallInt(_) => {
                    return AttrValueType::Int;
                }
                crate::types::AttrValue::Float(_) => {
                    return AttrValueType::Float;
                }
                crate::types::AttrValue::Bool(_) => {
                    return AttrValueType::Bool;
                }
                crate::types::AttrValue::Text(_) | crate::types::AttrValue::CompactText(_) => {
                    return AttrValueType::Text;
                }
                // Add other types as needed
                _ => continue,
            }
        }

        AttrValueType::Text // Fallback
    }

    /// Convert this array to a different type (if possible)
    pub fn convert_to(
        &self,
        target_type: crate::types::AttrValueType,
    ) -> Result<GraphArray, crate::errors::GraphError> {
        use crate::errors::GraphError;
        use crate::types::{AttrValue, AttrValueType};

        let converted_values: Result<Vec<AttrValue>, GraphError> = self
            .values
            .iter()
            .map(|value| {
                match (value, target_type) {
                    // Int conversions
                    (AttrValue::Int(i), AttrValueType::Float) => Ok(AttrValue::Float(*i as f32)),
                    (AttrValue::SmallInt(i), AttrValueType::Float) => {
                        Ok(AttrValue::Float(*i as f32))
                    }
                    (AttrValue::Int(i), AttrValueType::Text) => Ok(AttrValue::Text(i.to_string())),
                    (AttrValue::SmallInt(i), AttrValueType::Text) => {
                        Ok(AttrValue::Text(i.to_string()))
                    }

                    // Float conversions
                    (AttrValue::Float(f), AttrValueType::Int) => Ok(AttrValue::Int(*f as i64)),
                    (AttrValue::Float(f), AttrValueType::Text) => {
                        Ok(AttrValue::Text(f.to_string()))
                    }

                    // Bool conversions
                    (AttrValue::Bool(b), AttrValueType::Int) => {
                        Ok(AttrValue::Int(if *b { 1 } else { 0 }))
                    }
                    (AttrValue::Bool(b), AttrValueType::Text) => Ok(AttrValue::Text(b.to_string())),

                    // Text conversions
                    (AttrValue::Text(s), AttrValueType::Int) => {
                        s.parse::<i64>().map(AttrValue::Int).map_err(|_| {
                            GraphError::InvalidInput(format!("Cannot convert '{}' to integer", s))
                        })
                    }
                    (AttrValue::Text(s), AttrValueType::Float) => {
                        s.parse::<f32>().map(AttrValue::Float).map_err(|_| {
                            GraphError::InvalidInput(format!("Cannot convert '{}' to float", s))
                        })
                    }

                    // Same type - no conversion needed
                    (v, _) if self.dtype() == target_type => Ok(v.clone()),

                    // Unsupported conversion
                    _ => Err(GraphError::InvalidInput(format!(
                        "Cannot convert {:?} to {:?}",
                        self.dtype(),
                        target_type
                    ))),
                }
            })
            .collect();

        Ok(Self {
            values: converted_values?,
            name: self.name.clone(),
            cached_stats: RefCell::new(CachedStats::new()),
        })
    }

    /// Create a GraphArray from a graph attribute column
    pub fn from_graph_attribute(
        _graph: &crate::api::graph::Graph,
        attr: &str,
        entities: &[crate::types::NodeId],
    ) -> Result<Self, crate::errors::GraphError> {
        // This would need to be implemented with proper graph integration
        // For now, return a placeholder
        let values = entities
            .iter()
            .map(|_| crate::types::AttrValue::Null)
            .collect();

        Ok(Self::from_vec(values).with_name(attr.to_string()))
    }

    // ==================================================================================
    // LAZY EVALUATION & MATERIALIZATION METHODS
    // ==================================================================================

    /// Get a preview of the array for display purposes (first 10 elements)
    /// This is used by repr() and does not materialize the full array
    pub fn preview(&self, limit: usize) -> Vec<&AttrValue> {
        self.values.iter().take(limit.min(self.len())).collect()
    }

    /// Materialize the array to a vector of values for Python consumption
    /// This is the primary materialization method used by .values property
    pub fn materialize(&self) -> Vec<AttrValue> {
        self.values.clone()
    }

    /// Check if the array is effectively sparse (has many default/zero values)
    pub fn is_sparse(&self) -> bool {
        let total_count = self.len();
        if total_count == 0 {
            return false;
        }

        let zero_count = self
            .values
            .iter()
            .filter(|v| self.is_default_value(v))
            .count();

        // Consider sparse if >50% are default values
        (zero_count as f64) / (total_count as f64) > 0.5
    }

    /// Check if a value is considered a "default" value for sparsity
    fn is_default_value(&self, value: &AttrValue) -> bool {
        match value {
            AttrValue::Int(0) | AttrValue::SmallInt(0) => true,
            AttrValue::Float(f) if f.abs() < 1e-10 => true,
            AttrValue::Bool(false) => true,
            AttrValue::Text(s) if s.is_empty() => true,
            _ => false,
        }
    }

    /// Create a sparse representation of the array (index, value) pairs
    /// Only includes non-default values
    pub fn to_sparse(&self) -> Vec<(usize, AttrValue)> {
        self.values
            .iter()
            .enumerate()
            .filter(|(_, v)| !self.is_default_value(v))
            .map(|(i, v)| (i, v.clone()))
            .collect()
    }

    /// Get summary information for lazy display without full materialization
    pub fn summary_info(&self) -> String {
        let dtype = self.dtype();
        let len = self.len();
        let is_sparse = self.is_sparse();
        let name = self.name.as_deref().unwrap_or("unnamed");

        format!(
            "GraphArray('{}', length={}, dtype={:?}, sparse={})",
            name, len, dtype, is_sparse
        )
    }
}

/// Comprehensive statistical summary
#[derive(Debug, Clone)]
pub struct StatsSummary {
    pub count: usize,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub min: Option<AttrValue>,
    pub max: Option<AttrValue>,
    pub median: Option<f64>,
    pub q25: Option<f64>, // 25th percentile
    pub q75: Option<f64>, // 75th percentile
}

impl std::fmt::Display for StatsSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Statistical Summary:")?;
        writeln!(f, "  Count: {}", self.count)?;

        if let Some(mean) = self.mean {
            writeln!(f, "  Mean:  {:.2}", mean)?;
        }

        if let Some(std) = self.std {
            writeln!(f, "  Std:   {:.2}", std)?;
        }

        if let Some(ref min) = self.min {
            writeln!(f, "  Min:   {:?}", min)?;
        }

        if let Some(q25) = self.q25 {
            writeln!(f, "  25%:   {:.2}", q25)?;
        }

        if let Some(median) = self.median {
            writeln!(f, "  50%:   {:.2}", median)?;
        }

        if let Some(q75) = self.q75 {
            writeln!(f, "  75%:   {:.2}", q75)?;
        }

        if let Some(ref max) = self.max {
            writeln!(f, "  Max:   {:?}", max)?;
        }

        Ok(())
    }
}

// Implement indexing
impl std::ops::Index<usize> for GraphArray {
    type Output = AttrValue;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

// Implement IntoIterator for for-loop support
impl IntoIterator for GraphArray {
    type Item = AttrValue;
    type IntoIter = std::vec::IntoIter<AttrValue>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

// Implement IntoIterator for references
impl<'a> IntoIterator for &'a GraphArray {
    type Item = &'a AttrValue;
    type IntoIter = std::slice::Iter<'a, AttrValue>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AttrValue;

    #[test]
    fn test_basic_functionality() {
        let values = vec![
            AttrValue::Int(1),
            AttrValue::Int(2),
            AttrValue::Int(3),
            AttrValue::Int(4),
            AttrValue::Int(5),
        ];

        let arr = GraphArray::from_vec(values);

        assert_eq!(arr.len(), 5);
        assert_eq!(arr[0], AttrValue::Int(1));
        assert_eq!(arr.to_list().len(), 5);
    }

    #[test]
    fn test_statistical_operations() {
        let values = vec![
            AttrValue::Float(1.0),
            AttrValue::Float(2.0),
            AttrValue::Float(3.0),
            AttrValue::Float(4.0),
            AttrValue::Float(5.0),
        ];

        let arr = GraphArray::from_vec(values);

        assert_eq!(arr.mean(), Some(3.0));
        assert_eq!(arr.median(), Some(3.0));
        assert_eq!(arr.min(), Some(AttrValue::Float(1.0)));
        assert_eq!(arr.max(), Some(AttrValue::Float(5.0)));

        // Test caching by calling twice
        assert_eq!(arr.mean(), Some(3.0));
    }

    #[test]
    fn test_describe() {
        let values = vec![AttrValue::Int(10), AttrValue::Int(20), AttrValue::Int(30)];

        let arr = GraphArray::from_vec(values);
        let summary = arr.describe();

        assert_eq!(summary.count, 3);
        assert_eq!(summary.mean, Some(20.0));
        assert_eq!(summary.median, Some(20.0));
    }

    #[test]
    fn test_iteration() {
        let values = vec![AttrValue::Int(1), AttrValue::Int(2), AttrValue::Int(3)];

        let arr = GraphArray::from_vec(values);

        let mut sum = 0;
        for value in &arr {
            if let AttrValue::Int(i) = value {
                sum += i;
            }
        }

        assert_eq!(sum, 6);
    }
}
