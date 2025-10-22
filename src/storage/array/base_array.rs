use super::{ArrayIterator, ArrayOps};
use crate::errors::GraphResult;
use crate::types::AttrValue;
use crate::viz::VizModule;
use std::sync::Arc;

/// BaseArray provides fundamental array operations for all array types.
/// This is the foundation that all specialized arrays delegate to for basic functionality.
#[derive(Debug, Clone)]
pub struct BaseArray<T> {
    pub inner: Arc<Vec<T>>,
}

impl<T> BaseArray<T> {
    /// Create a new BaseArray from a vector
    pub fn new(data: Vec<T>) -> Self {
        Self {
            inner: Arc::new(data),
        }
    }

    /// Get the number of elements in the array
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get a reference to an element at the given index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    /// Get an iterator over the elements
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter()
    }

    /// Clone the internal vector (use sparingly for performance)
    pub fn clone_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        (*self.inner).clone()
    }

    /// Get access to underlying data as a slice (for SIMD operations)
    pub fn as_slice(&self) -> &[T] {
        &self.inner
    }

    /// Get the first element
    pub fn first(&self) -> Option<&T> {
        self.inner.first()
    }

    /// Get the last element
    pub fn last(&self) -> Option<&T> {
        self.inner.last()
    }

    /// Check if the array contains an element (requires PartialEq)
    pub fn contains(&self, item: &T) -> bool
    where
        T: PartialEq,
    {
        self.inner.contains(item)
    }

    /// Map over elements to create a new BaseArray with a different type
    pub fn map<U, F>(self, f: F) -> BaseArray<U>
    where
        F: Fn(T) -> U,
        T: Clone,
    {
        let mapped_data: Vec<U> = self.inner.iter().cloned().map(f).collect();
        BaseArray::new(mapped_data)
    }

    /// Filter elements to create a new BaseArray with same type
    pub fn filter<F>(self, predicate: F) -> BaseArray<T>
    where
        F: Fn(&T) -> bool,
        T: Clone,
    {
        let filtered_data: Vec<T> = self
            .inner
            .iter()
            .filter(|item| predicate(item))
            .cloned()
            .collect();
        BaseArray::new(filtered_data)
    }
}

impl<T> From<Vec<T>> for BaseArray<T> {
    fn from(data: Vec<T>) -> Self {
        Self::new(data)
    }
}

impl<T> From<BaseArray<T>> for Vec<T>
where
    T: Clone,
{
    fn from(base_array: BaseArray<T>) -> Vec<T> {
        base_array.clone_vec()
    }
}

// Specialized implementation for AttrValue (used by table system)
impl BaseArray<AttrValue> {
    /// Create a BaseArray from a vector of AttrValues
    pub fn from_attr_values(values: Vec<AttrValue>) -> Self {
        Self::new(values)
    }

    /// Get the underlying data (for compatibility with old table code)
    pub fn data(&self) -> &Vec<AttrValue> {
        &self.inner
    }

    /// Set a value at a specific index (mutable operation)
    pub fn set(&mut self, index: usize, value: AttrValue) -> Result<(), crate::errors::GraphError> {
        if index >= self.len() {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Index {} out of bounds for array of length {}",
                index,
                self.len()
            )));
        }

        // Since we use Arc, we need to make the data mutable
        if let Some(data) = Arc::get_mut(&mut self.inner) {
            data[index] = value;
            Ok(())
        } else {
            // If there are other references, we need to clone
            let mut data = (*self.inner).clone();
            data[index] = value;
            self.inner = Arc::new(data);
            Ok(())
        }
    }

    /// Create a slice of the array (for table operations)
    pub fn slice(&self, start: usize, length: usize) -> Self {
        let end = (start + length).min(self.len());
        let sliced_data = self.inner[start..end].to_vec();
        Self::new(sliced_data)
    }

    /// Get sorted indices (for table sorting)
    pub fn sort_indices(&self, ascending: bool) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.len()).collect();

        indices.sort_by(|&a, &b| {
            let ordering = self.inner[a].cmp(&self.inner[b]);
            if ascending {
                ordering
            } else {
                ordering.reverse()
            }
        });

        indices
    }

    /// Take values at specified indices (for table operations)
    pub fn take_indices(&self, indices: &[usize]) -> Result<Self, crate::errors::GraphError> {
        let selected_data: Vec<AttrValue> = indices
            .iter()
            .filter_map(|&i| self.inner.get(i).cloned())
            .collect();
        Ok(Self::new(selected_data))
    }

    /// Filter by boolean mask (for table operations)
    pub fn filter_by_mask(&self, mask: &[bool]) -> Result<Self, crate::errors::GraphError> {
        let filtered_data: Vec<AttrValue> = self
            .inner
            .iter()
            .zip(mask.iter())
            .filter_map(|(value, &keep)| if keep { Some(value.clone()) } else { None })
            .collect();
        Ok(Self::new(filtered_data))
    }

    /// Create BaseArray from NodeIds (specialized for nodes table)
    pub fn from_node_ids(node_ids: Vec<crate::types::NodeId>) -> Self {
        let values: Vec<AttrValue> = node_ids
            .into_iter()
            .map(|id| AttrValue::Int(id as i64))
            .collect();
        Self::new(values)
    }

    /// Convert to NodeIds (specialized for nodes table)
    pub fn as_node_ids(&self) -> Result<Vec<crate::types::NodeId>, crate::errors::GraphError> {
        let mut node_ids = Vec::new();

        for value in self.inner.iter() {
            match value {
                AttrValue::Int(id) => {
                    if *id >= 0 {
                        node_ids.push(*id as crate::types::NodeId);
                    } else {
                        return Err(crate::errors::GraphError::InvalidInput(format!(
                            "Invalid node ID: {}",
                            id
                        )));
                    }
                }
                _ => {
                    return Err(crate::errors::GraphError::InvalidInput(
                        "Column contains non-integer values, cannot convert to NodeIds".to_string(),
                    ));
                }
            }
        }

        Ok(node_ids)
    }

    /// Convert to NodeIds with filtering (specialized for edges table)
    /// Returns (valid_node_ids, valid_indices)
    pub fn as_node_ids_filtered(&self) -> (Vec<crate::types::NodeId>, Vec<usize>) {
        let mut node_ids = Vec::new();
        let mut valid_indices = Vec::new();

        for (idx, value) in self.inner.iter().enumerate() {
            if let AttrValue::Int(id) = value {
                if *id >= 0 {
                    node_ids.push(*id as crate::types::NodeId);
                    valid_indices.push(idx);
                }
            }
        }

        (node_ids, valid_indices)
    }

    /// Get unique values in the array
    pub fn unique_values(&self) -> Result<Vec<AttrValue>, crate::errors::GraphError> {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let mut unique_vals = Vec::new();

        for value in self.inner.iter() {
            if seen.insert(value.clone()) {
                unique_vals.push(value.clone());
            }
        }

        Ok(unique_vals)
    }

    /// Create BaseArray from EdgeIds (specialized for edges table)
    pub fn from_edge_ids(edge_ids: Vec<crate::types::EdgeId>) -> Self {
        let values: Vec<AttrValue> = edge_ids
            .into_iter()
            .map(|id| AttrValue::Int(id as i64))
            .collect();
        Self::new(values)
    }

    /// Convert to EdgeIds with filtering (specialized for edges table)
    /// Returns (valid_edge_ids, valid_indices)
    pub fn as_edge_ids_filtered(&self) -> (Vec<crate::types::EdgeId>, Vec<usize>) {
        let mut edge_ids = Vec::new();
        let mut valid_indices = Vec::new();

        for (idx, value) in self.inner.iter().enumerate() {
            if let AttrValue::Int(id) = value {
                if *id >= 0 {
                    edge_ids.push(*id as crate::types::EdgeId);
                    valid_indices.push(idx);
                }
            }
        }

        (edge_ids, valid_indices)
    }

    /// Launch interactive visualization for this AttrValue array
    ///
    /// Creates a single-column table from this array and delegates to
    /// BaseTable.interactive() following the delegation pattern.
    pub fn interactive(&self) -> GraphResult<VizModule> {
        use crate::storage::table::BaseTable;
        use std::collections::HashMap;

        // Create a single-column table from this array
        let mut columns = HashMap::new();
        let column_name = "values".to_string();
        columns.insert(column_name.clone(), self.clone());

        // Create BaseTable from the column
        let table = BaseTable::from_columns(columns)?;

        // Delegate to BaseTable's interactive method
        table.interactive(None)
    }

    // ==================================================================================
    // COLUMN MANAGEMENT OPERATIONS FOR TABLE CONVERSION
    // ==================================================================================

    /// Convert array to single-column table with specified column name
    pub fn to_table_with_name(
        &self,
        column_name: &str,
    ) -> GraphResult<crate::storage::table::BaseTable> {
        use crate::storage::table::BaseTable;
        use std::collections::HashMap;

        let mut columns = HashMap::new();
        columns.insert(column_name.to_string(), self.clone());

        BaseTable::from_columns(columns)
    }

    /// Convert array to single-column table with prefix added to default column name
    pub fn to_table_with_prefix(
        &self,
        prefix: &str,
    ) -> GraphResult<crate::storage::table::BaseTable> {
        let column_name = format!("{}values", prefix);
        self.to_table_with_name(&column_name)
    }

    /// Convert array to single-column table with suffix added to default column name
    pub fn to_table_with_suffix(
        &self,
        suffix: &str,
    ) -> GraphResult<crate::storage::table::BaseTable> {
        let column_name = format!("values{}", suffix);
        self.to_table_with_name(&column_name)
    }

    // ==================================================================================
    // PHASE 2.2: ELEMENT OPERATIONS (Array equivalent of row operations)
    // ==================================================================================

    /// Append a single element to the array
    pub fn append_element(&self, element: AttrValue) -> Self {
        let mut new_data = self.clone_vec();
        new_data.push(element);
        Self::new(new_data)
    }

    /// Extend array with multiple elements
    pub fn extend_elements(&self, elements: Vec<AttrValue>) -> Self {
        let mut new_data = self.clone_vec();
        new_data.extend(elements);
        Self::new(new_data)
    }

    /// Drop elements by indices
    pub fn drop_elements(&self, indices: &[usize]) -> GraphResult<Self> {
        if indices.is_empty() {
            return Ok(self.clone());
        }

        // Validate indices
        for &idx in indices {
            if idx >= self.len() {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Index {} out of range for array with {} elements",
                    idx,
                    self.len()
                )));
            }
        }

        // Create a set of indices to drop for efficient lookup
        let drop_set: std::collections::HashSet<usize> = indices.iter().cloned().collect();

        let new_data: Vec<AttrValue> = self
            .inner
            .iter()
            .enumerate()
            .filter(|(i, _)| !drop_set.contains(i))
            .map(|(_, val)| val.clone())
            .collect();

        Ok(Self::new(new_data))
    }

    /// Drop duplicate elements
    pub fn drop_duplicates_elements(&self) -> Self {
        let mut seen = std::collections::HashSet::new();
        let mut unique_elements = Vec::new();

        for val in self.inner.iter() {
            // Convert AttrValue to a hashable representation
            let key = match val {
                AttrValue::Int(x) => format!("i:{}", x),
                AttrValue::SmallInt(x) => format!("si:{}", x),
                AttrValue::Float(x) => format!("f:{}", x),
                AttrValue::Text(x) => format!("t:{}", x),
                AttrValue::CompactText(x) => format!("ct:{:?}", x),
                AttrValue::Bool(x) => format!("b:{}", x),
                AttrValue::Null => "null".to_string(),
                AttrValue::FloatVec(v) => format!("fv:{:?}", v),
                AttrValue::Bytes(b) => format!("by:{:?}", b),
                _ => format!("other:{:?}", val),
            };

            if seen.insert(key) {
                unique_elements.push(val.clone());
            }
        }

        Self::new(unique_elements)
    }

    /// Random sampling with comprehensive options
    pub fn sample(
        &self,
        n: Option<usize>,
        fraction: Option<f64>,
        weights: Option<Vec<f64>>,
        replace: bool,
    ) -> GraphResult<Self> {
        use fastrand;

        // Determine sample size
        let sample_size = match (n, fraction) {
            (Some(n_val), None) => n_val,
            (None, Some(frac)) => {
                if !(0.0..=1.0).contains(&frac) {
                    return Err(crate::errors::GraphError::InvalidInput(
                        "Fraction must be between 0.0 and 1.0".to_string(),
                    ));
                }
                (self.len() as f64 * frac).round() as usize
            }
            (Some(_), Some(_)) => {
                return Err(crate::errors::GraphError::InvalidInput(
                    "Cannot specify both n and fraction".to_string(),
                ));
            }
            (None, None) => {
                return Err(crate::errors::GraphError::InvalidInput(
                    "Must specify either n or fraction".to_string(),
                ));
            }
        };

        if sample_size == 0 {
            return Ok(Self::new(Vec::new()));
        }

        if !replace && sample_size > self.len() {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Cannot sample {} elements without replacement from array of length {}",
                sample_size,
                self.len()
            )));
        }

        // Generate sample indices
        let sampled_indices = if let Some(ref weight_vec) = weights {
            // Weighted sampling
            if weight_vec.len() != self.len() {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Weights length ({}) must match array length ({})",
                    weight_vec.len(),
                    self.len()
                )));
            }

            // Validate weights
            for &w in weight_vec {
                if w < 0.0 || !w.is_finite() {
                    return Err(crate::errors::GraphError::InvalidInput(
                        "All weights must be non-negative and finite".to_string(),
                    ));
                }
            }

            let total_weight: f64 = weight_vec.iter().sum();
            if total_weight <= 0.0 {
                return Err(crate::errors::GraphError::InvalidInput(
                    "Total weight must be positive".to_string(),
                ));
            }

            if replace {
                self.weighted_sample_with_replacement(sample_size, weight_vec, total_weight)?
            } else {
                self.weighted_sample_without_replacement(sample_size, weight_vec, total_weight)?
            }
        } else {
            // Uniform sampling
            if replace {
                (0..sample_size)
                    .map(|_| fastrand::usize(0..self.len()))
                    .collect()
            } else {
                let mut indices: Vec<usize> = (0..self.len()).collect();
                let mut sampled = Vec::new();

                for _ in 0..sample_size.min(indices.len()) {
                    let idx = fastrand::usize(0..indices.len());
                    sampled.push(indices.swap_remove(idx));
                }

                sampled
            }
        };

        // Extract sampled elements
        let sampled_data: Vec<AttrValue> = sampled_indices
            .iter()
            .map(|&i| self.inner[i].clone())
            .collect();

        Ok(Self::new(sampled_data))
    }

    /// Helper for weighted sampling with replacement
    fn weighted_sample_with_replacement(
        &self,
        sample_size: usize,
        weights: &[f64],
        total_weight: f64,
    ) -> GraphResult<Vec<usize>> {
        let mut sampled = Vec::with_capacity(sample_size);

        for _ in 0..sample_size {
            let mut target = fastrand::f64() * total_weight;

            for (i, &weight) in weights.iter().enumerate() {
                target -= weight;
                if target <= 0.0 {
                    sampled.push(i);
                    break;
                }
            }
        }

        Ok(sampled)
    }

    /// Helper for weighted sampling without replacement
    fn weighted_sample_without_replacement(
        &self,
        sample_size: usize,
        weights: &[f64],
        _total_weight: f64,
    ) -> GraphResult<Vec<usize>> {
        let mut available_weights = weights.to_vec();
        let mut available_indices: Vec<usize> = (0..self.len()).collect();
        let mut sampled = Vec::with_capacity(sample_size);

        for _ in 0..sample_size.min(available_indices.len()) {
            let current_total: f64 = available_weights.iter().sum();
            if current_total <= 0.0 {
                break;
            }

            let mut target = fastrand::f64() * current_total;
            let mut selected_pos = 0;

            for (pos, &weight) in available_weights.iter().enumerate() {
                target -= weight;
                if target <= 0.0 {
                    selected_pos = pos;
                    break;
                }
            }

            sampled.push(available_indices.swap_remove(selected_pos));
            available_weights.swap_remove(selected_pos);
        }

        Ok(sampled)
    }

    /// Calculate sum of numeric values in the array
    pub fn sum(&self) -> crate::errors::GraphResult<AttrValue> {
        let mut sum_int: i64 = 0;
        let mut sum_float: f64 = 0.0;
        let mut has_float = false;
        let mut count = 0;

        for val in self.inner.iter() {
            match val {
                AttrValue::Int(i) => {
                    if has_float {
                        sum_float += *i as f64;
                    } else {
                        sum_int += i;
                    }
                    count += 1;
                }
                AttrValue::SmallInt(i) => {
                    if has_float {
                        sum_float += *i as f64;
                    } else {
                        sum_int += *i as i64;
                    }
                    count += 1;
                }
                AttrValue::Float(f) => {
                    if !has_float {
                        // Convert previous int sum to float
                        sum_float = sum_int as f64 + *f as f64;
                        has_float = true;
                    } else {
                        sum_float += *f as f64;
                    }
                    count += 1;
                }
                AttrValue::Null => {} // Skip null values
                _ => {
                    return Err(crate::errors::GraphError::InvalidInput(format!(
                        "Cannot compute sum of non-numeric data type: {:?}",
                        val
                    )));
                }
            }
        }

        if count == 0 {
            return Err(crate::errors::GraphError::InvalidInput(
                "Cannot compute sum of empty array".to_string(),
            ));
        }

        if has_float {
            Ok(AttrValue::Float(sum_float as f32))
        } else {
            Ok(AttrValue::Int(sum_int))
        }
    }

    /// Calculate mean of numeric values in the array
    pub fn mean(&self) -> crate::errors::GraphResult<f64> {
        let mut sum: f64 = 0.0;
        let mut count = 0;

        for val in self.inner.iter() {
            match val {
                AttrValue::Int(i) => {
                    sum += *i as f64;
                    count += 1;
                }
                AttrValue::SmallInt(i) => {
                    sum += *i as f64;
                    count += 1;
                }
                AttrValue::Float(f) => {
                    sum += *f as f64;
                    count += 1;
                }
                AttrValue::Null => {} // Skip null values
                _ => {
                    return Err(crate::errors::GraphError::InvalidInput(format!(
                        "Cannot compute mean of non-numeric data type: {:?}",
                        val
                    )));
                }
            }
        }

        if count == 0 {
            return Err(crate::errors::GraphError::InvalidInput(
                "Cannot compute mean of empty array".to_string(),
            ));
        }

        Ok(sum / count as f64)
    }

    /// Calculate median of numeric values in the array
    pub fn median(&self) -> crate::errors::GraphResult<f64> {
        let mut numeric_values: Vec<f64> = Vec::new();

        for val in self.inner.iter() {
            match val {
                AttrValue::Int(i) => numeric_values.push(*i as f64),
                AttrValue::SmallInt(i) => numeric_values.push(*i as f64),
                AttrValue::Float(f) => numeric_values.push(*f as f64),
                AttrValue::Null => {} // Skip null values
                _ => {
                    return Err(crate::errors::GraphError::InvalidInput(format!(
                        "Cannot compute median of non-numeric data type: {:?}",
                        val
                    )));
                }
            }
        }

        if numeric_values.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(
                "Cannot compute median of empty array".to_string(),
            ));
        }

        numeric_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = numeric_values.len();
        if len.is_multiple_of(2) {
            // Even number of values - average of middle two
            let mid1 = numeric_values[len / 2 - 1];
            let mid2 = numeric_values[len / 2];
            Ok((mid1 + mid2) / 2.0)
        } else {
            // Odd number of values - middle value
            Ok(numeric_values[len / 2])
        }
    }

    /// Calculate standard deviation of numeric values in the array
    pub fn std(&self) -> crate::errors::GraphResult<f64> {
        let variance = self.var()?;
        Ok(variance.sqrt())
    }

    /// Calculate variance of numeric values in the array
    pub fn var(&self) -> crate::errors::GraphResult<f64> {
        let mean = self.mean()?;
        let mut sum_squared_diff: f64 = 0.0;
        let mut count = 0;

        for val in self.inner.iter() {
            match val {
                AttrValue::Int(i) => {
                    let diff = (*i as f64) - mean;
                    sum_squared_diff += diff * diff;
                    count += 1;
                }
                AttrValue::SmallInt(i) => {
                    let diff = (*i as f64) - mean;
                    sum_squared_diff += diff * diff;
                    count += 1;
                }
                AttrValue::Float(f) => {
                    let diff = (*f as f64) - mean;
                    sum_squared_diff += diff * diff;
                    count += 1;
                }
                AttrValue::Null => {} // Skip null values
                _ => {
                    return Err(crate::errors::GraphError::InvalidInput(format!(
                        "Cannot compute variance of non-numeric data type: {:?}",
                        val
                    )));
                }
            }
        }

        if count <= 1 {
            return Err(crate::errors::GraphError::InvalidInput(
                "Cannot compute variance with less than 2 values".to_string(),
            ));
        }

        // Sample variance (n-1 denominator)
        Ok(sum_squared_diff / (count - 1) as f64)
    }

    /// Find minimum value in the array (numeric only)
    pub fn min(&self) -> crate::errors::GraphResult<AttrValue> {
        let mut min_val: Option<(AttrValue, f64)> = None;

        for val in self.inner.iter() {
            if matches!(val, AttrValue::Null) {
                continue;
            }

            let numeric = match Self::extract_numeric(val) {
                Some(value) => value,
                None => {
                    return Err(crate::errors::GraphError::InvalidInput(format!(
                        "Cannot compute min of non-numeric data type: {:?}",
                        val
                    )));
                }
            };

            match &mut min_val {
                Some((_current_attr, current_numeric)) if numeric >= *current_numeric => {}
                _ => {
                    min_val = Some((val.clone(), numeric));
                }
            }
        }

        min_val.map(|(value, _)| value).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(
                "Cannot find minimum of empty array".to_string(),
            )
        })
    }

    /// Find maximum value in the array (numeric only)
    pub fn max(&self) -> crate::errors::GraphResult<AttrValue> {
        let mut max_val: Option<(AttrValue, f64)> = None;

        for val in self.inner.iter() {
            if matches!(val, AttrValue::Null) {
                continue;
            }

            let numeric = match Self::extract_numeric(val) {
                Some(value) => value,
                None => {
                    return Err(crate::errors::GraphError::InvalidInput(format!(
                        "Cannot compute max of non-numeric data type: {:?}",
                        val
                    )));
                }
            };

            match &mut max_val {
                Some((_current_attr, current_numeric)) if numeric <= *current_numeric => {}
                _ => {
                    max_val = Some((val.clone(), numeric));
                }
            }
        }

        max_val.map(|(value, _)| value).ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(
                "Cannot find maximum of empty array".to_string(),
            )
        })
    }

    /// Count non-null values in the array
    pub fn count(&self) -> usize {
        self.inner
            .iter()
            .filter(|val| !matches!(val, AttrValue::Null))
            .count()
    }

    /// Count unique non-null values in the array
    pub fn nunique(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for val in self.inner.iter() {
            if !matches!(val, AttrValue::Null) {
                seen.insert(val);
            }
        }
        seen.len()
    }

    /// Extract a numeric value for comparison-based operations
    fn extract_numeric(value: &AttrValue) -> Option<f64> {
        match value {
            AttrValue::Int(i) => Some(*i as f64),
            AttrValue::SmallInt(i) => Some(*i as f64),
            AttrValue::Float(f) => Some(*f as f64),
            _ => None,
        }
    }

    /// Detect missing/null values in the array
    /// Returns a boolean array where True indicates a null value
    /// Similar to pandas Series.isna()
    pub fn isna(&self) -> BaseArray<bool> {
        let null_mask: Vec<bool> = self
            .inner
            .iter()
            .map(|val| matches!(val, AttrValue::Null))
            .collect();
        BaseArray::new(null_mask)
    }

    /// Detect non-missing/non-null values in the array
    /// Returns a boolean array where True indicates a non-null value
    /// Similar to pandas Series.notna()
    pub fn notna(&self) -> BaseArray<bool> {
        let not_null_mask: Vec<bool> = self
            .inner
            .iter()
            .map(|val| !matches!(val, AttrValue::Null))
            .collect();
        BaseArray::new(not_null_mask)
    }

    /// Remove missing/null values from the array
    /// Returns a new array with null values filtered out
    /// Similar to pandas Series.dropna()
    pub fn dropna(&self) -> BaseArray<AttrValue> {
        let non_null_values: Vec<AttrValue> = self
            .inner
            .iter()
            .filter(|val| !matches!(val, AttrValue::Null))
            .cloned()
            .collect();
        BaseArray::new(non_null_values)
    }

    /// Check if the array contains any null values
    /// Similar to pandas Series.hasnans
    pub fn has_nulls(&self) -> bool {
        self.inner.iter().any(|val| matches!(val, AttrValue::Null))
    }

    /// Count the number of null values in the array
    pub fn null_count(&self) -> usize {
        self.inner
            .iter()
            .filter(|val| matches!(val, AttrValue::Null))
            .count()
    }

    /// Fill null values with a specified value
    /// Returns a new array with nulls replaced by the fill value
    /// Similar to pandas Series.fillna()
    pub fn fillna(&self, fill_value: AttrValue) -> BaseArray<AttrValue> {
        let filled_values: Vec<AttrValue> = self
            .inner
            .iter()
            .map(|val| {
                if matches!(val, AttrValue::Null) {
                    fill_value.clone()
                } else {
                    val.clone()
                }
            })
            .collect();
        BaseArray::new(filled_values)
    }

    /// Get string accessor for text operations
    /// Provides pandas-like string operations (.str.upper(), .str.contains(), etc.)
    pub fn str(&self) -> StringAccessor<'_> {
        StringAccessor::new(self)
    }

    /// Check if values in this array are in a provided set (pandas-style isin)
    ///
    /// # Arguments
    /// * `values` - Vector of values to check membership against
    ///
    /// # Returns
    /// Boolean array indicating which elements match the values
    ///
    /// # Examples
    /// ```ignore
    /// use groggy::storage::array::BaseArray;
    /// use groggy::types::AttrValue;
    ///
    /// let array = BaseArray::from_attr_values(vec![
    ///     AttrValue::Text("Engineering".to_string()),
    ///     AttrValue::Text("Marketing".to_string()),
    ///     AttrValue::Text("Sales".to_string())
    /// ]);
    ///
    /// let check_values = vec![
    ///     AttrValue::Text("Engineering".to_string()),
    ///     AttrValue::Text("Marketing".to_string())
    /// ];
    ///
    /// let mask = array.isin(check_values)?;
    /// // mask will be [true, true, false]
    /// ```ignore
    pub fn isin(&self, values: Vec<AttrValue>) -> crate::errors::GraphResult<BaseArray<AttrValue>> {
        let mut mask_data = Vec::with_capacity(self.len());

        for attr_val in self.inner.iter() {
            // Use simple iteration instead of HashSet for now to debug
            let mut matches = false;
            for check_val in &values {
                if attr_val == check_val {
                    matches = true;
                    break;
                }
            }
            mask_data.push(AttrValue::Bool(matches));
        }

        Ok(BaseArray::new(mask_data))
    }

    /// Count the frequency of each unique value in the array (pandas-style value_counts)
    ///
    /// # Arguments
    /// * `sort` - Whether to sort the results by count (default: true)
    /// * `ascending` - Sort order when sort=true (default: false, most frequent first)
    /// * `dropna` - Whether to exclude null values (default: true)
    ///
    /// # Returns
    /// Table with 'value' and 'count' columns showing frequency of each unique value
    ///
    /// # Examples
    /// ```ignore
    /// use groggy::storage::array::BaseArray;
    /// use groggy::types::AttrValue;
    ///
    /// let array = BaseArray::from_attr_values(vec![
    ///     AttrValue::Text("A".to_string()),
    ///     AttrValue::Text("B".to_string()),
    ///     AttrValue::Text("A".to_string()),
    ///     AttrValue::Text("C".to_string()),
    ///     AttrValue::Text("A".to_string()),
    /// ]);
    ///
    /// let counts = array.value_counts(true, false, true)?;
    /// // Returns table with:
    /// // value | count
    /// // "A"   | 3
    /// // "B"   | 1
    /// // "C"   | 1
    /// ```ignore
    pub fn value_counts(
        &self,
        sort: bool,
        ascending: bool,
        dropna: bool,
    ) -> crate::errors::GraphResult<crate::storage::table::BaseTable> {
        use crate::types::AttrValue;
        use std::collections::HashMap;

        // Count frequencies
        let mut counts: HashMap<String, (AttrValue, usize)> = HashMap::new();

        for value in self.inner.iter() {
            // Skip null values if dropna is true
            if dropna && matches!(value, AttrValue::Null) {
                continue;
            }

            let key = format!("{:?}", value);
            let entry = counts.entry(key).or_insert((value.clone(), 0));
            entry.1 += 1;
        }

        // Convert to vectors for sorting/table creation
        let mut results: Vec<(AttrValue, usize)> = counts.into_values().collect();

        // Sort if requested
        if sort {
            if ascending {
                results.sort_by(|a, b| a.1.cmp(&b.1));
            } else {
                results.sort_by(|a, b| b.1.cmp(&a.1));
            }
        }

        // Create table columns
        let mut value_data = Vec::with_capacity(results.len());
        let mut count_data = Vec::with_capacity(results.len());

        for (value, count) in results {
            value_data.push(value);
            count_data.push(AttrValue::Int(count as i64));
        }

        // Create table
        let mut columns = HashMap::new();
        columns.insert("value".to_string(), BaseArray::from_attr_values(value_data));
        columns.insert("count".to_string(), BaseArray::from_attr_values(count_data));

        let table = crate::storage::table::BaseTable::with_column_order(
            columns,
            vec!["value".to_string(), "count".to_string()],
        )?;

        Ok(table)
    }

    /// Apply a function to each element in the array (pandas-style apply)
    ///
    /// This method takes a function that operates on AttrValue and returns a new AttrValue,
    /// creating a new array with the transformed values.
    ///
    /// # Arguments
    /// * `func` - Function to apply to each element
    ///
    /// # Returns
    /// New BaseArray with transformed values
    ///
    /// # Examples
    /// ```ignore
    /// use groggy::storage::array::BaseArray;
    /// use groggy::types::AttrValue;
    ///
    /// let array = BaseArray::from_attr_values(vec![
    ///     AttrValue::Int(1),
    ///     AttrValue::Int(2),
    ///     AttrValue::Int(3)
    /// ]);
    ///
    /// // Square each number
    /// let squared = array.apply(|x| match x {
    ///     AttrValue::Int(n) => AttrValue::Int(n * n),
    ///     _ => x.clone()
    /// });
    /// ```ignore
    pub fn apply<F>(&self, func: F) -> Self
    where
        F: Fn(&AttrValue) -> AttrValue,
    {
        let transformed_data: Vec<AttrValue> = self.inner.iter().map(func).collect();

        BaseArray::from_attr_values(transformed_data)
    }

    /// Compute quantiles for the array (pandas-style quantile)
    ///
    /// # Arguments
    /// * `q` - Quantile to compute (0.0 to 1.0), or vector of quantiles
    /// * `interpolation` - Method for interpolation ("linear", "lower", "higher", "midpoint", "nearest")
    ///
    /// # Returns
    /// AttrValue (single quantile) or BaseArray (multiple quantiles)
    ///
    /// # Examples
    /// ```ignore
    /// use groggy::storage::array::BaseArray;
    /// use groggy::types::AttrValue;
    ///
    /// let array = BaseArray::from_attr_values(vec![
    ///     AttrValue::Int(1), AttrValue::Int(2), AttrValue::Int(3),
    ///     AttrValue::Int(4), AttrValue::Int(5)
    /// ]);
    ///
    /// // Single quantile
    /// let median = array.quantile(0.5, "linear")?; // AttrValue::Float(3.0)
    ///
    /// // Multiple quantiles
    /// let quartiles = array.quantiles(&[0.25, 0.5, 0.75], "linear")?;
    /// ```ignore
    pub fn quantile(&self, q: f64, interpolation: &str) -> crate::errors::GraphResult<AttrValue> {
        if !(0.0..=1.0).contains(&q) {
            return Err(crate::errors::GraphError::InvalidInput(
                "Quantile must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Extract numeric values, filtering out non-numeric and null values
        let mut numeric_values: Vec<f64> = Vec::new();
        for value in self.inner.iter() {
            match value {
                AttrValue::Int(i) => numeric_values.push(*i as f64),
                AttrValue::SmallInt(i) => numeric_values.push(*i as f64),
                AttrValue::Float(f) => numeric_values.push(*f as f64),
                AttrValue::Null => continue, // Skip nulls
                _ => continue,               // Skip non-numeric values
            }
        }

        if numeric_values.is_empty() {
            return Ok(AttrValue::Null);
        }

        // Sort the values
        numeric_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = numeric_values.len();
        if n == 1 {
            return Ok(AttrValue::Float(numeric_values[0] as f32));
        }

        // Calculate the position
        let position = q * (n - 1) as f64;
        let lower_index = position.floor() as usize;
        let upper_index = position.ceil() as usize;

        // Handle interpolation
        let result = match interpolation {
            "linear" => {
                if lower_index == upper_index {
                    numeric_values[lower_index]
                } else {
                    let fraction = position - lower_index as f64;
                    let lower_value = numeric_values[lower_index];
                    let upper_value = numeric_values[upper_index];
                    lower_value + fraction * (upper_value - lower_value)
                }
            }
            "lower" => numeric_values[lower_index],
            "higher" => numeric_values[upper_index],
            "midpoint" => {
                if lower_index == upper_index {
                    numeric_values[lower_index]
                } else {
                    (numeric_values[lower_index] + numeric_values[upper_index]) / 2.0
                }
            }
            "nearest" => {
                let fraction = position - lower_index as f64;
                if fraction < 0.5 {
                    numeric_values[lower_index]
                } else {
                    numeric_values[upper_index]
                }
            }
            _ => {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Invalid interpolation method: {}",
                    interpolation
                )))
            }
        };

        Ok(AttrValue::Float(result as f32))
    }

    /// Compute multiple quantiles for the array
    ///
    /// # Arguments
    /// * `quantiles` - Slice of quantiles to compute (each 0.0 to 1.0)
    /// * `interpolation` - Method for interpolation
    ///
    /// # Returns
    /// BaseArray containing the computed quantiles
    pub fn quantiles(
        &self,
        quantiles: &[f64],
        interpolation: &str,
    ) -> crate::errors::GraphResult<BaseArray<AttrValue>> {
        let mut results = Vec::with_capacity(quantiles.len());

        for &q in quantiles {
            let quantile_value = self.quantile(q, interpolation)?;
            results.push(quantile_value);
        }

        Ok(BaseArray::from_attr_values(results))
    }

    /// Compute percentiles for the array (equivalent to quantile * 100)
    ///
    /// # Arguments
    /// * `percentiles` - Percentile to compute (0.0 to 100.0), or vector of percentiles
    /// * `interpolation` - Method for interpolation
    ///
    /// # Examples
    /// ```ignore
    /// let median = array.percentile(50.0, "linear")?; // 50th percentile = median
    /// let quartiles = array.percentiles(&[25.0, 50.0, 75.0], "linear")?;
    /// ```ignore
    pub fn get_percentile(
        &self,
        percentile: f64,
        interpolation: &str,
    ) -> crate::errors::GraphResult<AttrValue> {
        if !(0.0..=100.0).contains(&percentile) {
            return Err(crate::errors::GraphError::InvalidInput(
                "Percentile must be between 0.0 and 100.0".to_string(),
            ));
        }

        self.quantile(percentile / 100.0, interpolation)
    }

    /// Compute multiple percentiles for the array
    pub fn percentiles(
        &self,
        percentiles: &[f64],
        interpolation: &str,
    ) -> crate::errors::GraphResult<BaseArray<AttrValue>> {
        // Validate all percentiles first
        for &p in percentiles {
            if !(0.0..=100.0).contains(&p) {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Percentile {} must be between 0.0 and 100.0",
                    p
                )));
            }
        }

        // Convert percentiles to quantiles
        let quantiles: Vec<f64> = percentiles.iter().map(|&p| p / 100.0).collect();
        self.quantiles(&quantiles, interpolation)
    }

    /// Compute correlation coefficient with another array (pandas-style corr)
    ///
    /// # Arguments
    /// * `other` - Another BaseArray to compute correlation with
    /// * `method` - Correlation method ("pearson", "spearman", "kendall")
    ///
    /// # Returns
    /// Correlation coefficient as AttrValue::Float, or AttrValue::Null if calculation fails
    ///
    /// # Examples
    /// ```ignore
    /// use groggy::storage::array::BaseArray;
    /// use groggy::types::AttrValue;
    ///
    /// let array1 = BaseArray::from_attr_values(vec![
    ///     AttrValue::Int(1), AttrValue::Int(2), AttrValue::Int(3)
    /// ]);
    /// let array2 = BaseArray::from_attr_values(vec![
    ///     AttrValue::Int(2), AttrValue::Int(4), AttrValue::Int(6)
    /// ]);
    ///
    /// // Perfect positive correlation
    /// let corr = array1.corr(&array2, "pearson")?; // AttrValue::Float(1.0)
    /// ```ignore
    pub fn corr(
        &self,
        other: &BaseArray<AttrValue>,
        method: &str,
    ) -> crate::errors::GraphResult<AttrValue> {
        // Extract numeric values from both arrays, filtering out non-numeric and null values
        let mut values1: Vec<f64> = Vec::new();
        let mut values2: Vec<f64> = Vec::new();

        let min_len = std::cmp::min(self.inner.len(), other.inner.len());

        for i in 0..min_len {
            let val1 = self.inner.get(i).unwrap_or(&AttrValue::Null);
            let val2 = other.inner.get(i).unwrap_or(&AttrValue::Null);

            let num1 = match val1 {
                AttrValue::Int(n) => Some(*n as f64),
                AttrValue::SmallInt(n) => Some(*n as f64),
                AttrValue::Float(f) => Some(*f as f64),
                AttrValue::Null => None,
                _ => None,
            };

            let num2 = match val2 {
                AttrValue::Int(n) => Some(*n as f64),
                AttrValue::SmallInt(n) => Some(*n as f64),
                AttrValue::Float(f) => Some(*f as f64),
                AttrValue::Null => None,
                _ => None,
            };

            if let (Some(n1), Some(n2)) = (num1, num2) {
                values1.push(n1);
                values2.push(n2);
            }
        }

        if values1.len() < 2 {
            return Ok(AttrValue::Null);
        }

        let correlation = match method {
            "pearson" => self.pearson_correlation(&values1, &values2)?,
            "spearman" => {
                // Convert to ranks and compute Pearson correlation of ranks
                let ranks1 = self.compute_ranks(&values1);
                let ranks2 = self.compute_ranks(&values2);
                self.pearson_correlation(&ranks1, &ranks2)?
            }
            "kendall" => {
                // Kendall's tau correlation
                self.kendall_correlation(&values1, &values2)?
            }
            _ => {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Invalid correlation method: {}. Use 'pearson', 'spearman', or 'kendall'",
                    method
                )))
            }
        };

        Ok(AttrValue::Float(correlation as f32))
    }

    /// Compute covariance with another array (pandas-style cov)
    ///
    /// # Arguments
    /// * `other` - Another BaseArray to compute covariance with
    /// * `ddof` - Delta degrees of freedom (default: 1 for sample covariance)
    ///
    /// # Returns
    /// Covariance as AttrValue::Float, or AttrValue::Null if calculation fails
    ///
    /// # Examples
    /// ```ignore
    /// let cov = array1.cov(&array2, 1)?; // Sample covariance
    /// let cov_pop = array1.cov(&array2, 0)?; // Population covariance
    /// ```ignore
    pub fn cov(
        &self,
        other: &BaseArray<AttrValue>,
        ddof: i32,
    ) -> crate::errors::GraphResult<AttrValue> {
        // Extract numeric values from both arrays
        let mut values1: Vec<f64> = Vec::new();
        let mut values2: Vec<f64> = Vec::new();

        let min_len = std::cmp::min(self.inner.len(), other.inner.len());

        for i in 0..min_len {
            let val1 = self.inner.get(i).unwrap_or(&AttrValue::Null);
            let val2 = other.inner.get(i).unwrap_or(&AttrValue::Null);

            let num1 = match val1 {
                AttrValue::Int(n) => Some(*n as f64),
                AttrValue::SmallInt(n) => Some(*n as f64),
                AttrValue::Float(f) => Some(*f as f64),
                AttrValue::Null => None,
                _ => None,
            };

            let num2 = match val2 {
                AttrValue::Int(n) => Some(*n as f64),
                AttrValue::SmallInt(n) => Some(*n as f64),
                AttrValue::Float(f) => Some(*f as f64),
                AttrValue::Null => None,
                _ => None,
            };

            if let (Some(n1), Some(n2)) = (num1, num2) {
                values1.push(n1);
                values2.push(n2);
            }
        }

        if values1.len() <= ddof as usize {
            return Ok(AttrValue::Null);
        }

        // Calculate means
        let mean1: f64 = values1.iter().sum::<f64>() / values1.len() as f64;
        let mean2: f64 = values2.iter().sum::<f64>() / values2.len() as f64;

        // Calculate covariance
        let covariance: f64 = values1
            .iter()
            .zip(values2.iter())
            .map(|(x1, x2)| (x1 - mean1) * (x2 - mean2))
            .sum::<f64>()
            / (values1.len() as f64 - ddof as f64);

        Ok(AttrValue::Float(covariance as f32))
    }

    /// Helper function to compute Pearson correlation coefficient
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> crate::errors::GraphResult<f64> {
        let n = x.len() as f64;
        if n < 2.0 {
            return Ok(0.0);
        }

        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Helper function to compute ranks for Spearman correlation
    fn compute_ranks(&self, values: &[f64]) -> Vec<f64> {
        let mut indexed_values: Vec<(usize, f64)> =
            values.iter().enumerate().map(|(i, &v)| (i, v)).collect();

        // Sort by value
        indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0; values.len()];
        let mut i = 0;
        while i < indexed_values.len() {
            let current_value = indexed_values[i].1;
            let start = i;

            // Find all equal values
            while i < indexed_values.len() && indexed_values[i].1 == current_value {
                i += 1;
            }

            // Assign average rank to all equal values
            let avg_rank = (start + i + 1) as f64 / 2.0;
            for j in start..i {
                ranks[indexed_values[j].0] = avg_rank;
            }
        }

        ranks
    }

    /// Helper function to compute Kendall's tau correlation
    fn kendall_correlation(&self, x: &[f64], y: &[f64]) -> crate::errors::GraphResult<f64> {
        let n = x.len();
        if n < 2 {
            return Ok(0.0);
        }

        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            for j in i + 1..n {
                let x_diff = x[i] - x[j];
                let y_diff = y[i] - y[j];

                if (x_diff > 0.0 && y_diff > 0.0) || (x_diff < 0.0 && y_diff < 0.0) {
                    concordant += 1;
                } else if (x_diff > 0.0 && y_diff < 0.0) || (x_diff < 0.0 && y_diff > 0.0) {
                    discordant += 1;
                }
                // If either difference is 0, it's a tie and doesn't count
            }
        }

        let total_pairs = n * (n - 1) / 2;
        if total_pairs == 0 {
            Ok(0.0)
        } else {
            Ok((concordant as f64 - discordant as f64) / total_pairs as f64)
        }
    }

    /// Rolling window operation with specified window size
    ///
    /// # Arguments
    /// * `window` - Window size for rolling operations
    /// * `operation` - Function to apply to each window (e.g., "mean", "sum", "min", "max", "std")
    ///
    /// # Returns
    /// New BaseArray with rolling operation results
    pub fn rolling(&self, window: usize, operation: &str) -> GraphResult<Self> {
        if window == 0 {
            return Err(crate::errors::GraphError::InvalidInput(
                "Window size must be greater than 0".to_string(),
            ));
        }

        let mut result_values = Vec::new();

        for i in 0..self.data().len() {
            if i + 1 < window {
                // Not enough data for window, use null
                result_values.push(AttrValue::Null);
            } else {
                // Extract window data
                let start_idx = if i + 1 >= window { i + 1 - window } else { 0 };
                let window_data: Vec<f64> = self.data()[start_idx..=i]
                    .iter()
                    .filter_map(|val| match val {
                        AttrValue::Int(i) => Some(*i as f64),
                        AttrValue::Float(f) => Some(*f as f64),
                        _ => None,
                    })
                    .collect();

                if window_data.is_empty() {
                    result_values.push(AttrValue::Null);
                } else {
                    let result = match operation {
                        "mean" => window_data.iter().sum::<f64>() / window_data.len() as f64,
                        "sum" => window_data.iter().sum::<f64>(),
                        "min" => window_data.iter().cloned().fold(f64::INFINITY, f64::min),
                        "max" => window_data
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max),
                        "std" => {
                            let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                            let variance =
                                window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                                    / window_data.len() as f64;
                            variance.sqrt()
                        }
                        _ => {
                            return Err(crate::errors::GraphError::InvalidInput(format!(
                                "Unsupported rolling operation: {}",
                                operation
                            )));
                        }
                    };
                    result_values.push(AttrValue::Float(result as f32));
                }
            }
        }

        Ok(BaseArray::from_attr_values(result_values))
    }

    /// Expanding window operation (cumulative from start)
    ///
    /// # Arguments
    /// * `operation` - Function to apply to expanding window (e.g., "mean", "sum", "min", "max", "std")
    ///
    /// # Returns
    /// New BaseArray with expanding operation results
    pub fn expanding(&self, operation: &str) -> GraphResult<Self> {
        let mut result_values = Vec::new();

        for i in 0..self.data().len() {
            // Extract data from start to current position
            let window_data: Vec<f64> = self.data()[0..=i]
                .iter()
                .filter_map(|val| match val {
                    AttrValue::Int(i) => Some(*i as f64),
                    AttrValue::Float(f) => Some(*f as f64),
                    _ => None,
                })
                .collect();

            if window_data.is_empty() {
                result_values.push(AttrValue::Null);
            } else {
                let result = match operation {
                    "mean" => window_data.iter().sum::<f64>() / window_data.len() as f64,
                    "sum" => window_data.iter().sum::<f64>(),
                    "min" => window_data.iter().cloned().fold(f64::INFINITY, f64::min),
                    "max" => window_data
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max),
                    "std" => {
                        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                        let variance = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                            / window_data.len() as f64;
                        variance.sqrt()
                    }
                    _ => {
                        return Err(crate::errors::GraphError::InvalidInput(format!(
                            "Unsupported expanding operation: {}",
                            operation
                        )));
                    }
                };
                result_values.push(AttrValue::Float(result as f32));
            }
        }

        Ok(BaseArray::from_attr_values(result_values))
    }

    /// Cumulative sum operation
    ///
    /// # Returns
    /// New BaseArray with cumulative sum values
    pub fn cumsum(&self) -> GraphResult<Self> {
        let mut result_values = Vec::new();
        let mut running_sum = 0.0;

        for val in self.data() {
            match val {
                AttrValue::Int(i) => {
                    running_sum += *i as f64;
                    result_values.push(AttrValue::Float(running_sum as f32));
                }
                AttrValue::Float(f) => {
                    running_sum += *f as f64;
                    result_values.push(AttrValue::Float(running_sum as f32));
                }
                _ => {
                    // For non-numeric values, keep the running sum unchanged
                    result_values.push(AttrValue::Float(running_sum as f32));
                }
            }
        }

        Ok(BaseArray::from_attr_values(result_values))
    }

    /// Cumulative minimum operation
    ///
    /// # Returns
    /// New BaseArray with cumulative minimum values
    pub fn cummin(&self) -> GraphResult<Self> {
        let mut result_values = Vec::new();
        let mut running_min = f64::INFINITY;

        for val in self.data() {
            match val {
                AttrValue::Int(i) => {
                    running_min = running_min.min(*i as f64);
                    result_values.push(AttrValue::Float(running_min as f32));
                }
                AttrValue::Float(f) => {
                    running_min = running_min.min(*f as f64);
                    result_values.push(AttrValue::Float(running_min as f32));
                }
                _ => {
                    // For non-numeric values, keep current running minimum
                    if running_min != f64::INFINITY {
                        result_values.push(AttrValue::Float(running_min as f32));
                    } else {
                        result_values.push(AttrValue::Null);
                    }
                }
            }
        }

        Ok(BaseArray::from_attr_values(result_values))
    }

    /// Cumulative maximum operation
    ///
    /// # Returns
    /// New BaseArray with cumulative maximum values
    pub fn cummax(&self) -> GraphResult<Self> {
        let mut result_values = Vec::new();
        let mut running_max = f64::NEG_INFINITY;

        for val in self.data() {
            match val {
                AttrValue::Int(i) => {
                    running_max = running_max.max(*i as f64);
                    result_values.push(AttrValue::Float(running_max as f32));
                }
                AttrValue::Float(f) => {
                    running_max = running_max.max(*f as f64);
                    result_values.push(AttrValue::Float(running_max as f32));
                }
                _ => {
                    // For non-numeric values, keep current running maximum
                    if running_max != f64::NEG_INFINITY {
                        result_values.push(AttrValue::Float(running_max as f32));
                    } else {
                        result_values.push(AttrValue::Null);
                    }
                }
            }
        }

        Ok(BaseArray::from_attr_values(result_values))
    }

    /// Shift operation - shift values by specified periods
    ///
    /// # Arguments
    /// * `periods` - Number of periods to shift (positive = shift right, negative = shift left)
    /// * `fill_value` - Value to use for filling gaps (default: Null)
    ///
    /// # Returns
    /// New BaseArray with shifted values
    pub fn shift(&self, periods: i32, fill_value: Option<AttrValue>) -> GraphResult<Self> {
        let fill = fill_value.unwrap_or(AttrValue::Null);
        let mut result_values = vec![fill.clone(); self.data().len()];

        if periods == 0 {
            return Ok(BaseArray::from_attr_values(self.data().clone()));
        }

        if periods > 0 {
            // Shift right
            let shift = periods as usize;
            result_values[shift..self.data().len()]
                .clone_from_slice(&self.data()[..(self.data().len() - shift)]);
        } else {
            // Shift left
            let shift = (-periods) as usize;
            result_values[..(self.data().len() - shift)].clone_from_slice(&self.data()[shift..]);
        }

        Ok(BaseArray::from_attr_values(result_values))
    }

    /// Percentage change operation
    ///
    /// # Arguments
    /// * `periods` - Number of periods to use for comparison (default: 1)
    ///
    /// # Returns
    /// New BaseArray with percentage change values
    pub fn pct_change(&self, periods: Option<usize>) -> GraphResult<Self> {
        let periods = periods.unwrap_or(1);
        let mut result_values = Vec::new();

        for i in 0..self.data().len() {
            if i < periods {
                result_values.push(AttrValue::Null);
            } else {
                let current = match &self.data()[i] {
                    AttrValue::Int(val) => *val as f64,
                    AttrValue::Float(val) => *val as f64,
                    _ => {
                        result_values.push(AttrValue::Null);
                        continue;
                    }
                };

                let previous = match &self.data()[i - periods] {
                    AttrValue::Int(val) => *val as f64,
                    AttrValue::Float(val) => *val as f64,
                    _ => {
                        result_values.push(AttrValue::Null);
                        continue;
                    }
                };

                if previous == 0.0 {
                    result_values.push(AttrValue::Null);
                } else {
                    let pct_change = (current - previous) / previous;
                    result_values.push(AttrValue::Float(pct_change as f32));
                }
            }
        }

        Ok(BaseArray::from_attr_values(result_values))
    }
}

/// String accessor for BaseArray<AttrValue> providing text processing operations
/// Similar to pandas Series.str accessor
pub struct StringAccessor<'a> {
    array: &'a BaseArray<AttrValue>,
}

impl<'a> StringAccessor<'a> {
    /// Create a new StringAccessor
    pub fn new(array: &'a BaseArray<AttrValue>) -> Self {
        Self { array }
    }

    /// Convert all string values to uppercase
    /// Non-string values remain unchanged
    pub fn upper(&self) -> BaseArray<AttrValue> {
        let transformed_values: Vec<AttrValue> = self
            .array
            .data()
            .iter()
            .map(|val| match val {
                AttrValue::Text(s) => AttrValue::Text(s.to_uppercase()),
                AttrValue::CompactText(s) => AttrValue::Text(s.as_str().to_uppercase()),
                other => other.clone(),
            })
            .collect();
        BaseArray::new(transformed_values)
    }

    /// Convert all string values to lowercase
    /// Non-string values remain unchanged
    pub fn lower(&self) -> BaseArray<AttrValue> {
        let transformed_values: Vec<AttrValue> = self
            .array
            .data()
            .iter()
            .map(|val| match val {
                AttrValue::Text(s) => AttrValue::Text(s.to_lowercase()),
                AttrValue::CompactText(s) => AttrValue::Text(s.as_str().to_lowercase()),
                other => other.clone(),
            })
            .collect();
        BaseArray::new(transformed_values)
    }

    /// Get the length of each string value
    /// Non-string values return 0
    pub fn len(&self) -> BaseArray<AttrValue> {
        let lengths: Vec<AttrValue> = self
            .array
            .data()
            .iter()
            .map(|val| match val {
                AttrValue::Text(s) => AttrValue::Int(s.len() as i64),
                AttrValue::CompactText(s) => AttrValue::Int(s.as_str().len() as i64),
                AttrValue::Null => AttrValue::Null,
                _ => AttrValue::Int(0i64),
            })
            .collect();
        BaseArray::new(lengths)
    }

    /// Strip whitespace from both ends of string values
    /// Non-string values remain unchanged
    pub fn strip(&self) -> BaseArray<AttrValue> {
        let stripped_values: Vec<AttrValue> = self
            .array
            .data()
            .iter()
            .map(|val| match val {
                AttrValue::Text(s) => AttrValue::Text(s.trim().to_string()),
                AttrValue::CompactText(s) => AttrValue::Text(s.as_str().trim().to_string()),
                other => other.clone(),
            })
            .collect();
        BaseArray::new(stripped_values)
    }

    /// Check if string values contain a substring
    /// Returns boolean array
    pub fn contains(&self, pattern: &str) -> BaseArray<AttrValue> {
        let contains_results: Vec<AttrValue> = self
            .array
            .data()
            .iter()
            .map(|val| match val {
                AttrValue::Text(s) => AttrValue::Bool(s.contains(pattern)),
                AttrValue::CompactText(s) => AttrValue::Bool(s.as_str().contains(pattern)),
                AttrValue::Null => AttrValue::Null,
                _ => AttrValue::Bool(false),
            })
            .collect();
        BaseArray::new(contains_results)
    }

    /// Check if string values start with a prefix
    /// Returns boolean array
    pub fn startswith(&self, prefix: &str) -> BaseArray<AttrValue> {
        let startswith_results: Vec<AttrValue> = self
            .array
            .data()
            .iter()
            .map(|val| match val {
                AttrValue::Text(s) => AttrValue::Bool(s.starts_with(prefix)),
                AttrValue::CompactText(s) => AttrValue::Bool(s.as_str().starts_with(prefix)),
                AttrValue::Null => AttrValue::Null,
                _ => AttrValue::Bool(false),
            })
            .collect();
        BaseArray::new(startswith_results)
    }

    /// Check if string values end with a suffix
    /// Returns boolean array
    pub fn endswith(&self, suffix: &str) -> BaseArray<AttrValue> {
        let endswith_results: Vec<AttrValue> = self
            .array
            .data()
            .iter()
            .map(|val| match val {
                AttrValue::Text(s) => AttrValue::Bool(s.ends_with(suffix)),
                AttrValue::CompactText(s) => AttrValue::Bool(s.as_str().ends_with(suffix)),
                AttrValue::Null => AttrValue::Null,
                _ => AttrValue::Bool(false),
            })
            .collect();
        BaseArray::new(endswith_results)
    }

    /// Replace occurrences of a pattern with replacement text
    /// Non-string values remain unchanged
    pub fn replace(&self, from: &str, to: &str) -> BaseArray<AttrValue> {
        let replaced_values: Vec<AttrValue> = self
            .array
            .data()
            .iter()
            .map(|val| match val {
                AttrValue::Text(s) => AttrValue::Text(s.replace(from, to)),
                AttrValue::CompactText(s) => AttrValue::Text(s.as_str().replace(from, to)),
                other => other.clone(),
            })
            .collect();
        BaseArray::new(replaced_values)
    }

    /// Split string values by a delimiter and return the first part
    /// For more complex splitting, use split_expand()
    pub fn split(&self, delimiter: &str) -> BaseArray<AttrValue> {
        let split_values: Vec<AttrValue> = self
            .array
            .data()
            .iter()
            .map(|val| match val {
                AttrValue::Text(s) => {
                    if let Some(first_part) = s.split(delimiter).next() {
                        AttrValue::Text(first_part.to_string())
                    } else {
                        AttrValue::Text(s.clone())
                    }
                }
                AttrValue::CompactText(s) => {
                    let s_str = s.as_str();
                    if let Some(first_part) = s_str.split(delimiter).next() {
                        AttrValue::Text(first_part.to_string())
                    } else {
                        AttrValue::Text(s_str.to_string())
                    }
                }
                other => other.clone(),
            })
            .collect();
        BaseArray::new(split_values)
    }

    /// Get substring by position (start_index, length)
    /// Similar to pandas str.slice()
    pub fn slice(&self, start: usize, length: Option<usize>) -> BaseArray<AttrValue> {
        let sliced_values: Vec<AttrValue> = self
            .array
            .data()
            .iter()
            .map(|val| match val {
                AttrValue::Text(s) => {
                    let chars: Vec<char> = s.chars().collect();
                    let start_idx = start.min(chars.len());
                    let end_idx = if let Some(len) = length {
                        (start_idx + len).min(chars.len())
                    } else {
                        chars.len()
                    };

                    if start_idx < chars.len() {
                        let substring: String = chars[start_idx..end_idx].iter().collect();
                        AttrValue::Text(substring)
                    } else {
                        AttrValue::Text(String::new())
                    }
                }
                AttrValue::CompactText(s) => {
                    let s_str = s.as_str();
                    let chars: Vec<char> = s_str.chars().collect();
                    let start_idx = start.min(chars.len());
                    let end_idx = if let Some(len) = length {
                        (start_idx + len).min(chars.len())
                    } else {
                        chars.len()
                    };

                    if start_idx < chars.len() {
                        let substring: String = chars[start_idx..end_idx].iter().collect();
                        AttrValue::Text(substring)
                    } else {
                        AttrValue::Text(String::new())
                    }
                }
                other => other.clone(),
            })
            .collect();
        BaseArray::new(sliced_values)
    }

    /// Pad strings with specified character to reach target length
    /// Similar to pandas str.pad()
    pub fn pad(&self, width: usize, side: &str, fillchar: char) -> BaseArray<AttrValue> {
        let padded_values: Vec<AttrValue> = self
            .array
            .data()
            .iter()
            .map(|val| match val {
                AttrValue::Text(s) => {
                    if s.len() >= width {
                        AttrValue::Text(s.clone())
                    } else {
                        let padding = width - s.len();
                        let fill_str = fillchar.to_string().repeat(padding);

                        let result = match side {
                            "left" => format!("{}{}", fill_str, s),
                            "right" => format!("{}{}", s, fill_str),
                            "both" => {
                                let left_pad = padding / 2;
                                let right_pad = padding - left_pad;
                                format!(
                                    "{}{}{}",
                                    fillchar.to_string().repeat(left_pad),
                                    s,
                                    fillchar.to_string().repeat(right_pad)
                                )
                            }
                            _ => s.clone(), // Invalid side, return original
                        };
                        AttrValue::Text(result)
                    }
                }
                AttrValue::CompactText(s) => {
                    let s_str = s.as_str();
                    if s_str.len() >= width {
                        AttrValue::Text(s_str.to_string())
                    } else {
                        let padding = width - s_str.len();
                        let fill_str = fillchar.to_string().repeat(padding);

                        let result = match side {
                            "left" => format!("{}{}", fill_str, s_str),
                            "right" => format!("{}{}", s_str, fill_str),
                            "both" => {
                                let left_pad = padding / 2;
                                let right_pad = padding - left_pad;
                                format!(
                                    "{}{}{}",
                                    fillchar.to_string().repeat(left_pad),
                                    s_str,
                                    fillchar.to_string().repeat(right_pad)
                                )
                            }
                            _ => s_str.to_string(), // Invalid side, return original
                        };
                        AttrValue::Text(result)
                    }
                }
                other => other.clone(),
            })
            .collect();
        BaseArray::new(padded_values)
    }
}

// Implement IntoIterator for convenient usage
impl<T> IntoIterator for BaseArray<T>
where
    T: Clone,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.clone_vec().into_iter()
    }
}

// Implement ArrayOps trait for BaseArray<T>
impl<T> ArrayOps<T> for BaseArray<T>
where
    T: Clone + 'static,
{
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    fn iter(&self) -> ArrayIterator<T> {
        ArrayIterator::new(self.clone_vec())
    }

    fn to_vec(&self) -> Vec<T> {
        self.clone_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let data = vec![1, 2, 3, 4, 5];
        let base_array = BaseArray::new(data.clone());

        assert_eq!(base_array.len(), 5);
        assert!(!base_array.is_empty());
        assert_eq!(base_array.get(0), Some(&1));
        assert_eq!(base_array.get(10), None);
        assert_eq!(base_array.first(), Some(&1));
        assert_eq!(base_array.last(), Some(&5));
        assert!(base_array.contains(&3));
        assert!(!base_array.contains(&10));
    }

    #[test]
    fn test_map_and_filter() {
        let data = vec![1, 2, 3, 4, 5];
        let base_array = BaseArray::new(data);

        // Test map
        let doubled = base_array.clone().map(|x| x * 2);
        assert_eq!(doubled.clone_vec(), vec![2, 4, 6, 8, 10]);

        // Test filter
        let evens = base_array.filter(|&x| x % 2 == 0);
        assert_eq!(evens.clone_vec(), vec![2, 4]);
    }

    #[test]
    fn test_empty_array() {
        let empty: BaseArray<i32> = BaseArray::new(vec![]);

        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.first(), None);
        assert_eq!(empty.last(), None);
    }
}
