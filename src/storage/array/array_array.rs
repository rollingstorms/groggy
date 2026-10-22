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
        if self.len() != values.len() {
            return Err(GraphError::InvalidInput(format!(
                "Number of values ({}) must match number of arrays ({})",
                values.len(),
                self.len()
            )));
        }

        let mut columns = self.key_columns();
        if columns.contains_key(agg_name) {
            return Err(GraphError::InvalidInput(format!(
                "Aggregation column conflicts with group key: {agg_name}"
            )));
        }
        columns.insert(agg_name.to_string(), Self::stored_aggregation(values)?);

        BaseTable::from_columns(columns)
    }
    fn materialize_aggregations(
        &self,
        spec: &[(String, String)],
        values: impl Fn(&str) -> Result<Vec<f64>, GraphError>,
    ) -> Result<BaseTable, GraphError> {
        let mut columns = self.key_columns();
        for (name, function) in spec {
            if columns.contains_key(name) {
                return Err(GraphError::InvalidInput(format!(
                    "Duplicate or group-key-conflicting aggregation column: {name}"
                )));
            }
            validate_reduction(function)?;
            columns.insert(name.clone(), Self::stored_aggregation(values(function)?)?);
        }
        BaseTable::from_columns(columns)
    }

    // Table AttrValue floats are f32: reject finite overflow rather than create infinity.
    fn stored_aggregation(
        values: Vec<f64>,
    ) -> Result<BaseArray<crate::types::AttrValue>, GraphError> {
        let values = values
            .into_iter()
            .map(|value| {
                let stored = value as f32;
                if value.is_finite() && !stored.is_finite() {
                    Err(GraphError::InvalidInput(
                        "Finite aggregation result exceeds f32 table storage".into(),
                    ))
                } else {
                    Ok(crate::types::AttrValue::Float(stored))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BaseArray::from_attr_values(values))
    }

    fn key_columns(&self) -> std::collections::HashMap<String, BaseArray<crate::types::AttrValue>> {
        let mut columns = std::collections::HashMap::new();
        if let Some(keys) = &self.keys {
            columns.insert(
                self.key_name.as_deref().unwrap_or("group_key").to_string(),
                BaseArray::from_attr_values(
                    keys.iter()
                        .cloned()
                        .map(crate::types::AttrValue::Text)
                        .collect(),
                ),
            );
        }
        columns
    }
}

#[derive(Clone, Copy)]
enum AttributeNumber {
    Integer(i64),
    Float(f64),
}
impl super::numeric_reductions::Scalar for AttributeNumber {
    fn as_f64(self) -> f64 {
        match self {
            Self::Integer(v) => v as f64,
            Self::Float(v) => v,
        }
    }
    fn integer(self) -> Option<i128> {
        match self {
            Self::Integer(v) => Some(i128::from(v)),
            Self::Float(_) => None,
        }
    }
}
fn attribute_number(value: &crate::types::AttrValue) -> Option<AttributeNumber> {
    use crate::types::AttrValue;
    match value {
        AttrValue::Int(v) => Some(AttributeNumber::Integer(*v)),
        AttrValue::SmallInt(v) => Some(AttributeNumber::Integer(i64::from(*v))),
        AttrValue::Bool(v) => Some(AttributeNumber::Integer(i64::from(*v))),
        AttrValue::Float(v) => Some(AttributeNumber::Float(f64::from(*v))),
        _ => None,
    }
}
fn validate_reduction(function: &str) -> Result<(), GraphError> {
    match function {
        "mean" | "sum" | "min" | "max" | "std" | "count" => Ok(()),
        _ => Err(GraphError::InvalidInput(format!(
            "Unknown aggregation function: {function}"
        ))),
    }
}
fn reduce_numeric<S: super::numeric_reductions::Scalar>(
    values: &[S],
    function: &str,
) -> Result<f64, GraphError> {
    use super::numeric_reductions as k;
    Ok(match function {
        "sum" => k::sum(values),
        "mean" => k::mean(values).unwrap_or(0.0),
        "min" => k::extremum(values, true).unwrap_or(0.0),
        "max" => k::extremum(values, false).unwrap_or(0.0),
        "std" => k::std_dev(values).unwrap_or(0.0),
        "count" => values.len() as f64,
        _ => {
            return Err(GraphError::InvalidInput(format!(
                "Unknown aggregation function: {function}"
            )));
        }
    })
}

impl ArrayArray<crate::types::AttrValue> {
    /// Reduce numeric values while retaining exact integer hints until f64 output.
    /// Legacy null/text/collection skipping and bool-as-zero/one remain explicit.
    pub fn numeric_reduce(&self, function: &str) -> Result<Vec<f64>, GraphError> {
        validate_reduction(function)?;
        self.arrays
            .iter()
            .map(|array| {
                let values: Vec<_> = array.iter().filter_map(attribute_number).collect();
                reduce_numeric(&values, function)
            })
            .collect()
    }
    /// Count projected numeric scalars without converting integers or copying values.
    pub fn numeric_counts(&self) -> Vec<usize> {
        self.arrays
            .iter()
            .map(|a| a.iter().filter(|v| attribute_number(v).is_some()).count())
            .collect()
    }
    /// Typed numeric multi-aggregation with shared table/key validation.
    pub fn aggregate_numeric(&self, spec: &[(String, String)]) -> Result<BaseTable, GraphError> {
        self.materialize_aggregations(spec, |function| self.numeric_reduce(function))
    }

    /// Legacy numeric projection: retain scalar numbers/bools, skip null/text/collections.
    /// Retains keys and their column name. Integer values convert
    /// to f64 here; this projection does not promise exact i64 statistics.
    pub fn numeric_arrays(&self) -> Result<ArrayArray<f64>, GraphError> {
        use crate::types::AttrValue;
        let arrays = self
            .arrays
            .iter()
            .map(|array| {
                BaseArray::new(
                    array
                        .iter()
                        .filter_map(|value| match value {
                            AttrValue::Float(v) => Some(f64::from(*v)),
                            AttrValue::Int(v) => Some(*v as f64),
                            AttrValue::SmallInt(v) => Some(f64::from(*v)),
                            AttrValue::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
                            _ => None,
                        })
                        .collect(),
                )
            })
            .collect();
        match &self.keys {
            Some(keys) => ArrayArray::with_named_keys(
                arrays,
                keys.clone(),
                self.key_name.as_deref().unwrap_or("group_key"),
            ),
            None => Ok(ArrayArray::new(arrays)),
        }
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
            .map(|a| super::numeric_reductions::mean(a.as_slice()).unwrap_or(0.0))
            .collect()
    }

    /// Sum each array with shared compensated/scaled kernels; empty=0.
    pub fn sum(&self) -> Vec<f64> {
        self.arrays
            .iter()
            .map(|a| super::numeric_reductions::sum(a.as_slice()))
            .collect()
    }

    /// Minimum of each array; empty=0, NaNs propagate.
    pub fn min(&self) -> Vec<f64> {
        self.arrays
            .iter()
            .map(|a| super::numeric_reductions::extremum(a.as_slice(), true).unwrap_or(0.0))
            .collect()
    }

    /// Maximum of each array; empty=0, NaNs propagate.
    pub fn max(&self) -> Vec<f64> {
        self.arrays
            .iter()
            .map(|a| super::numeric_reductions::extremum(a.as_slice(), false).unwrap_or(0.0))
            .collect()
    }

    /// Sample standard deviation; empty/finite singleton=0, nonfinite=NaN.
    pub fn std(&self) -> Vec<f64> {
        self.arrays
            .iter()
            .map(|a| super::numeric_reductions::std_dev(a.as_slice()).unwrap_or(0.0))
            .collect()
    }

    /// Count elements in each array
    pub fn count(&self) -> Vec<usize> {
        self.arrays.iter().map(|arr| arr.len()).collect()
    }

    /// Materialize named numeric aggregations, retaining keys and checked f32 table results.
    /// Reject unknown functions and duplicate/key-conflicting output names.
    pub fn aggregate(&self, spec: &[(String, String)]) -> Result<BaseTable, GraphError> {
        self.materialize_aggregations(spec, |function| {
            self.arrays
                .iter()
                .map(|array| reduce_numeric(array.as_slice(), function))
                .collect()
        })
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
