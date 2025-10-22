//! ArrayArray - Array of Arrays with aggregation support
//!
//! This module provides the ArrayArray type, which is a collection of arrays
//! that supports aggregation operations. It's used as an intermediate type when
//! extracting columns from TableArray or attributes from SubgraphArray.
//!
//! # Example
//! ```
//! use groggy::storage::array::ArrayArray;
//! use groggy::storage::array::BaseArray;
//!
//! // Create array of arrays
//! let arrays = vec![
//!     BaseArray::from(vec![1.0, 2.0, 3.0]),
//!     BaseArray::from(vec![4.0, 5.0, 6.0]),
//!     BaseArray::from(vec![7.0, 8.0, 9.0]),
//! ];
//! let arr_arr = ArrayArray::new(arrays);
//!
//! // Calculate mean of each array
//! let means = arr_arr.mean();
//! assert_eq!(means, vec![2.0, 5.0, 8.0]);
//! ```

use super::base_array::BaseArray;
use crate::errors::GraphError;
use crate::storage::table::BaseTable;
use std::fmt;

/// Array of arrays with aggregation support
#[derive(Clone, Debug)]
pub struct ArrayArray<T> {
    /// Inner arrays
    arrays: Vec<BaseArray<T>>,

    /// Optional keys for each array (e.g., group keys from group_by)
    keys: Option<Vec<String>>,

    /// Optional column name to use when materialising keys into a table
    key_name: Option<String>,
}

impl<T> ArrayArray<T>
where
    T: Clone + Default + fmt::Display,
{
    /// Create new ArrayArray from vector of arrays
    pub fn new(arrays: Vec<BaseArray<T>>) -> Self {
        Self {
            arrays,
            keys: None,
            key_name: None,
        }
    }

    /// Create new ArrayArray with associated keys
    pub fn with_keys(arrays: Vec<BaseArray<T>>, keys: Vec<String>) -> Result<Self, GraphError> {
        Self::with_named_keys(arrays, keys, "group_key")
    }

    /// Create new ArrayArray with associated keys and explicit key column name
    pub fn with_named_keys(
        arrays: Vec<BaseArray<T>>,
        keys: Vec<String>,
        key_name: impl Into<String>,
    ) -> Result<Self, GraphError> {
        if arrays.len() != keys.len() {
            return Err(GraphError::InvalidInput(format!(
                "Number of arrays ({}) must match number of keys ({})",
                arrays.len(),
                keys.len()
            )));
        }

        Ok(Self {
            arrays,
            keys: Some(keys),
            key_name: Some(key_name.into()),
        })
    }

    /// Get the number of arrays
    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    /// Get array at index
    pub fn get(&self, index: usize) -> Option<&BaseArray<T>> {
        self.arrays.get(index)
    }

    /// Get keys if available
    pub fn keys(&self) -> Option<&Vec<String>> {
        self.keys.as_ref()
    }

    /// Get the key column name if available
    pub fn key_name(&self) -> Option<&String> {
        self.key_name.as_ref()
    }

    /// Convert to vector of arrays
    pub fn into_arrays(self) -> Vec<BaseArray<T>> {
        self.arrays
    }

    /// Iterate over arrays
    pub fn iter(&self) -> impl Iterator<Item = &BaseArray<T>> {
        self.arrays.iter()
    }
}

impl ArrayArray<f64> {
    /// Calculate mean of each array
    ///
    /// Returns a vector where each element is the mean of the corresponding array.
    /// Empty arrays return 0.0.
    pub fn mean(&self) -> Vec<f64> {
        self.arrays
            .iter()
            .map(|arr| {
                if arr.is_empty() {
                    0.0
                } else {
                    arr.iter().sum::<f64>() / arr.len() as f64
                }
            })
            .collect()
    }

    /// Calculate sum of each array
    pub fn sum(&self) -> Vec<f64> {
        self.arrays.iter().map(|arr| arr.iter().sum()).collect()
    }

    /// Calculate minimum of each array
    ///
    /// Returns a vector where each element is the minimum of the corresponding array.
    /// Empty arrays return 0.0.
    pub fn min(&self) -> Vec<f64> {
        self.arrays
            .iter()
            .map(|arr| {
                if arr.is_empty() {
                    0.0
                } else {
                    arr.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b))
                }
            })
            .collect()
    }

    /// Calculate maximum of each array
    ///
    /// Returns a vector where each element is the maximum of the corresponding array.
    /// Empty arrays return 0.0.
    pub fn max(&self) -> Vec<f64> {
        self.arrays
            .iter()
            .map(|arr| {
                if arr.is_empty() {
                    0.0
                } else {
                    arr.iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b))
                }
            })
            .collect()
    }

    /// Calculate standard deviation of each array
    ///
    /// Uses sample standard deviation (n-1 denominator).
    /// Arrays with < 2 elements return 0.0.
    pub fn std(&self) -> Vec<f64> {
        self.arrays
            .iter()
            .map(|arr| {
                if arr.len() < 2 {
                    return 0.0;
                }

                let mean = arr.iter().sum::<f64>() / arr.len() as f64;
                let variance =
                    arr.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (arr.len() - 1) as f64;

                variance.sqrt()
            })
            .collect()
    }

    /// Count elements in each array
    pub fn count(&self) -> Vec<usize> {
        self.arrays.iter().map(|arr| arr.len()).collect()
    }

    /// Convert to BaseTable with results and group keys
    ///
    /// This is used to automatically package aggregation results into a table.
    /// If keys are present, creates a table with 'group_key' and 'value' columns.
    /// If no keys, creates a table with just 'value' column.
    ///
    /// # Example
    /// ```ignore
    /// let arr_arr = ArrayArray::with_keys(arrays, vec!["A", "B", "C"]);
    /// let means = arr_arr.mean();
    /// let table = arr_arr.to_table_with_aggregation("mean", means);
    /// // Table has columns: ['group_key', 'mean']
    /// ```ignore
    pub fn to_table_with_aggregation(
        &self,
        agg_name: &str,
        values: Vec<f64>,
    ) -> Result<BaseTable, GraphError> {
        use crate::types::AttrValue;

        if self.len() != values.len() {
            return Err(GraphError::InvalidInput(format!(
                "Number of values ({}) must match number of arrays ({})",
                values.len(),
                self.len()
            )));
        }

        let mut columns = std::collections::HashMap::new();

        if let Some(keys) = &self.keys {
            // Create table with group keys
            let key_col: Vec<AttrValue> = keys.iter().map(|s| AttrValue::Text(s.clone())).collect();
            let name = self
                .key_name
                .as_ref()
                .cloned()
                .unwrap_or_else(|| "group_key".to_string());
            columns.insert(name, BaseArray::from_attr_values(key_col));
        }

        // Add aggregation column
        let value_col: Vec<AttrValue> =
            values.iter().map(|&v| AttrValue::Float(v as f32)).collect();
        columns.insert(agg_name.to_string(), BaseArray::from_attr_values(value_col));

        BaseTable::from_columns(columns)
    }
}

impl<T: Clone + Default + fmt::Display> fmt::Display for ArrayArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ArrayArray({} arrays", self.arrays.len())?;
        if let Some(keys) = &self.keys {
            write!(f, ", with keys: {:?}", keys)?;
            if let Some(name) = &self.key_name {
                write!(f, ", key_name: {}", name)?;
            }
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_array_creation() {
        let arrays = vec![
            BaseArray::from(vec![1.0, 2.0, 3.0]),
            BaseArray::from(vec![4.0, 5.0]),
        ];
        let arr_arr = ArrayArray::new(arrays);

        assert_eq!(arr_arr.len(), 2);
        assert!(!arr_arr.is_empty());
    }

    #[test]
    fn test_array_array_with_keys() {
        let arrays = vec![
            BaseArray::from(vec![1.0, 2.0, 3.0]),
            BaseArray::from(vec![4.0, 5.0]),
        ];
        let keys = vec!["group_a".to_string(), "group_b".to_string()];

        let arr_arr = ArrayArray::with_keys(arrays, keys).unwrap();
        assert_eq!(arr_arr.keys().unwrap().len(), 2);
    }

    #[test]
    fn test_mean_aggregation() {
        let arrays = vec![
            BaseArray::from(vec![1.0, 2.0, 3.0]), // mean = 2.0
            BaseArray::from(vec![4.0, 5.0, 6.0]), // mean = 5.0
            BaseArray::from(vec![7.0, 8.0, 9.0]), // mean = 8.0
        ];
        let arr_arr = ArrayArray::new(arrays);

        let means = arr_arr.mean();
        assert_eq!(means, vec![2.0, 5.0, 8.0]);
    }

    #[test]
    fn test_sum_aggregation() {
        let arrays = vec![
            BaseArray::from(vec![1.0, 2.0, 3.0]), // sum = 6.0
            BaseArray::from(vec![4.0, 5.0]),      // sum = 9.0
        ];
        let arr_arr = ArrayArray::new(arrays);

        let sums = arr_arr.sum();
        assert_eq!(sums, vec![6.0, 9.0]);
    }

    #[test]
    fn test_min_max_aggregation() {
        let arrays = vec![
            BaseArray::from(vec![1.0, 5.0, 3.0]),
            BaseArray::from(vec![4.0, 2.0, 6.0]),
        ];
        let arr_arr = ArrayArray::new(arrays);

        let mins = arr_arr.min();
        let maxs = arr_arr.max();

        assert_eq!(mins, vec![1.0, 2.0]);
        assert_eq!(maxs, vec![5.0, 6.0]);
    }

    #[test]
    fn test_std_aggregation() {
        let arrays = vec![BaseArray::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])];
        let arr_arr = ArrayArray::new(arrays);

        let stds = arr_arr.std();
        // Sample std of [1,2,3,4,5] ≈ 1.58
        assert!((stds[0] - 1.58).abs() < 0.01);
    }

    #[test]
    fn test_count_aggregation() {
        let arrays = vec![
            BaseArray::from(vec![1.0, 2.0, 3.0]),
            BaseArray::from(vec![4.0]),
            BaseArray::from(vec![5.0, 6.0]),
        ];
        let arr_arr = ArrayArray::new(arrays);

        let counts = arr_arr.count();
        assert_eq!(counts, vec![3, 1, 2]);
    }

    #[test]
    fn test_empty_array_handling() {
        let arrays = vec![
            BaseArray::from(vec![1.0, 2.0]),
            BaseArray::from(vec![]), // Empty
            BaseArray::from(vec![3.0]),
        ];
        let arr_arr = ArrayArray::new(arrays);

        let means = arr_arr.mean();
        assert_eq!(means, vec![1.5, 0.0, 3.0]);

        let counts = arr_arr.count();
        assert_eq!(counts, vec![2, 0, 1]);
    }

    #[test]
    fn test_to_table_with_keys() {
        let arrays = vec![
            BaseArray::from(vec![1.0, 2.0, 3.0]),
            BaseArray::from(vec![4.0, 5.0, 6.0]),
        ];
        let keys = vec!["A".to_string(), "B".to_string()];
        let arr_arr = ArrayArray::with_keys(arrays, keys).unwrap();

        let means = arr_arr.mean();
        let table = arr_arr.to_table_with_aggregation("mean", means).unwrap();

        use crate::storage::table::traits::Table;
        assert!(table.has_column("group_key"));
        assert!(table.has_column("mean"));
        assert_eq!(table.nrows(), 2);
    }

    #[test]
    fn test_to_table_with_named_keys() {
        let arrays = vec![
            BaseArray::from(vec![1.0, 2.0, 3.0]),
            BaseArray::from(vec![4.0, 5.0, 6.0]),
        ];
        let keys = vec!["alpha".to_string(), "beta".to_string()];
        let arr_arr = ArrayArray::with_named_keys(arrays, keys, "label").unwrap();

        let means = arr_arr.mean();
        let table = arr_arr.to_table_with_aggregation("mean", means).unwrap();

        use crate::storage::table::traits::Table;
        assert!(table.has_column("label"));
        assert!(table.has_column("mean"));
        assert_eq!(table.nrows(), 2);
    }

    #[test]
    fn test_to_table_without_keys() {
        let arrays = vec![
            BaseArray::from(vec![1.0, 2.0, 3.0]),
            BaseArray::from(vec![4.0, 5.0, 6.0]),
        ];
        let arr_arr = ArrayArray::new(arrays);

        let means = arr_arr.mean();
        let table = arr_arr.to_table_with_aggregation("mean", means).unwrap();

        use crate::storage::table::traits::Table;
        assert!(table.has_column("mean"));
        assert_eq!(table.nrows(), 2);
    }
}
