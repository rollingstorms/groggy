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
            let val_a = &self.inner[a];
            let val_b = &self.inner[b];

            // Simple comparison based on AttrValue
            use crate::types::AttrValue;
            let ordering = match (val_a, val_b) {
                (AttrValue::Int(a), AttrValue::Int(b)) => a.cmp(b),
                (AttrValue::Float(a), AttrValue::Float(b)) => {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                }
                (AttrValue::Text(a), AttrValue::Text(b)) => a.cmp(b),
                (AttrValue::Bool(a), AttrValue::Bool(b)) => a.cmp(b),
                _ => std::cmp::Ordering::Equal, // Default for mixed types
            };

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
            match value {
                AttrValue::Int(id) => {
                    if *id >= 0 {
                        node_ids.push(*id as crate::types::NodeId);
                        valid_indices.push(idx);
                    }
                }
                _ => {}
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
            match value {
                AttrValue::Int(id) => {
                    if *id >= 0 {
                        edge_ids.push(*id as crate::types::EdgeId);
                        valid_indices.push(idx);
                    }
                }
                _ => {}
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
                if frac < 0.0 || frac > 1.0 {
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

    /// Find minimum value in the array
    pub fn min(&self) -> crate::errors::GraphResult<AttrValue> {
        let mut min_val: Option<AttrValue> = None;

        for val in self.inner.iter() {
            match val {
                AttrValue::Null => continue, // Skip null values
                _ => match &min_val {
                    None => min_val = Some(val.clone()),
                    Some(current_min) => {
                        if Self::compare_values(val, current_min)? < 0 {
                            min_val = Some(val.clone());
                        }
                    }
                },
            }
        }

        min_val.ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(
                "Cannot find minimum of empty array".to_string(),
            )
        })
    }

    /// Find maximum value in the array
    pub fn max(&self) -> crate::errors::GraphResult<AttrValue> {
        let mut max_val: Option<AttrValue> = None;

        for val in self.inner.iter() {
            match val {
                AttrValue::Null => continue, // Skip null values
                _ => match &max_val {
                    None => max_val = Some(val.clone()),
                    Some(current_max) => {
                        if Self::compare_values(val, current_max)? > 0 {
                            max_val = Some(val.clone());
                        }
                    }
                },
            }
        }

        max_val.ok_or_else(|| {
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

    /// Helper function to compare two AttrValues for ordering
    fn compare_values(a: &AttrValue, b: &AttrValue) -> crate::errors::GraphResult<i32> {
        use std::cmp::Ordering;

        let ordering = match (a, b) {
            // Numeric comparisons
            (AttrValue::Int(a), AttrValue::Int(b)) => a.cmp(b),
            (AttrValue::SmallInt(a), AttrValue::SmallInt(b)) => a.cmp(b),
            (AttrValue::Float(a), AttrValue::Float(b)) => {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            }

            // Mixed numeric comparisons
            (AttrValue::Int(a), AttrValue::SmallInt(b)) => (*a).cmp(&(*b as i64)),
            (AttrValue::SmallInt(a), AttrValue::Int(b)) => (*a as i64).cmp(b),
            (AttrValue::Int(a), AttrValue::Float(b)) => {
                (*a as f32).partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (AttrValue::Float(a), AttrValue::Int(b)) => {
                a.partial_cmp(&(*b as f32)).unwrap_or(Ordering::Equal)
            }
            (AttrValue::SmallInt(a), AttrValue::Float(b)) => {
                (*a as f32).partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (AttrValue::Float(a), AttrValue::SmallInt(b)) => {
                a.partial_cmp(&(*b as f32)).unwrap_or(Ordering::Equal)
            }

            // String comparisons
            (AttrValue::Text(a), AttrValue::Text(b)) => a.cmp(b),
            (AttrValue::CompactText(a), AttrValue::CompactText(b)) => a.as_str().cmp(b.as_str()),
            (AttrValue::Text(a), AttrValue::CompactText(b)) => a.as_str().cmp(b.as_str()),
            (AttrValue::CompactText(a), AttrValue::Text(b)) => a.as_str().cmp(b.as_str()),

            // Boolean comparisons
            (AttrValue::Bool(a), AttrValue::Bool(b)) => a.cmp(b),

            // Incompatible types
            _ => {
                return Err(crate::errors::GraphError::InvalidInput(format!(
                    "Cannot compare incompatible types: {:?} and {:?}",
                    a, b
                )));
            }
        };

        Ok(match ordering {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        })
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
    pub fn str(&self) -> StringAccessor {
        StringAccessor::new(self)
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
