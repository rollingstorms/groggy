//! Legacy typed/name metadata adapter over the canonical BaseArray<AttrValue>.

use super::base_array::BaseArray as Column;
use crate::storage::array::{ArrayIterator, ArrayOps};
use crate::types::{AttrValue, AttrValueType};

/// Compatibility metadata adapter; values use the canonical shared column.
/// This replaces and extends the current PyGraphArray functionality
#[derive(Clone)]
pub struct BaseArray {
    /// Columnar data storage - optimized for AttrValue collections  
    values: Column<AttrValue>,
    /// Data type for type consistency and optimization
    dtype: AttrValueType,
    /// Optional column name for debugging and display
    name: Option<String>,
}

impl BaseArray {
    /// Wrap an existing canonical column without copying its values.
    pub fn from_column(values: Column<AttrValue>, dtype: AttrValueType) -> Self {
        Self {
            values,
            dtype,
            name: None,
        }
    }
    /// Access canonical immutable column storage.
    pub fn as_column(&self) -> &Column<AttrValue> {
        &self.values
    }
    /// Consume the metadata adapter and return its canonical column.
    pub fn into_column(self) -> Column<AttrValue> {
        self.values
    }

    /// Create a new BaseArray from AttrValue vector
    pub fn new(data: Vec<AttrValue>, dtype: AttrValueType) -> Self {
        Self {
            values: Column::new(data),
            dtype,
            name: None,
        }
    }

    /// Create a new BaseArray with a name
    pub fn with_name(data: Vec<AttrValue>, dtype: AttrValueType, name: String) -> Self {
        Self {
            values: Column::new(data),
            dtype,
            name: Some(name),
        }
    }

    /// Get the data type of this array
    pub fn dtype(&self) -> &AttrValueType {
        &self.dtype
    }

    /// Get the name of this array (if any)
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    /// Set the name of this array
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    // =================================================================
    // Type-specific constructors
    // =================================================================

    /// Create BaseArray from NodeId vector
    pub fn try_from_node_ids(ids: Vec<crate::types::NodeId>) -> crate::errors::GraphResult<Self> {
        Column::try_from_node_ids(ids).map(|values| Self::from_column(values, AttrValueType::Int))
    }
    /// Legacy infallible ID constructor; use try_from_node_ids for errors.
    pub fn from_node_ids(ids: Vec<crate::types::NodeId>) -> Self {
        Self::try_from_node_ids(ids).unwrap_or_else(|error| {
            panic!("Legacy BaseArray ID construction failed: {error}; use try_from_node_ids")
        })
    }

    /// Create BaseArray from EdgeId vector
    pub fn try_from_edge_ids(ids: Vec<crate::types::EdgeId>) -> crate::errors::GraphResult<Self> {
        Column::try_from_edge_ids(ids).map(|values| Self::from_column(values, AttrValueType::Int))
    }
    /// Legacy infallible ID constructor; use try_from_edge_ids for errors.
    pub fn from_edge_ids(ids: Vec<crate::types::EdgeId>) -> Self {
        Self::try_from_edge_ids(ids).unwrap_or_else(|error| {
            panic!("Legacy BaseArray ID construction failed: {error}; use try_from_edge_ids")
        })
    }

    /// Create BaseArray from AttrValue vector
    pub fn from_attr_values(attr_values: Vec<AttrValue>) -> Self {
        let dtype = if attr_values.is_empty() {
            AttrValueType::Text
        } else {
            match &attr_values[0] {
                AttrValue::Text(_) => AttrValueType::Text,
                AttrValue::Int(_) => AttrValueType::Int,
                AttrValue::Float(_) => AttrValueType::Float,
                AttrValue::Float64(_) => AttrValueType::Float64,
                AttrValue::UtcInstant(_) => AttrValueType::UtcInstant,
                AttrValue::Date(_) => AttrValueType::Date,
                AttrValue::Duration(_) => AttrValueType::Duration,
                AttrValue::Float64Vec(_) => AttrValueType::Float64Vec,
                AttrValue::Bool(_) => AttrValueType::Bool,
                _ => AttrValueType::Text,
            }
        };

        Self::new(attr_values, dtype)
    }

    // =================================================================
    // Type-specific extractors
    // =================================================================

    /// Extract as NodeId vector (assumes Int type)
    pub fn as_node_ids(&self) -> crate::errors::GraphResult<Vec<crate::types::NodeId>> {
        self.values.as_node_ids()
    }

    /// Extract as EdgeId vector (assumes Int type)
    pub fn as_edge_ids(&self) -> crate::errors::GraphResult<Vec<crate::types::EdgeId>> {
        self.values.as_node_ids()
    }

    /// Extract as NodeId vector, filtering out rows with invalid values
    pub fn as_node_ids_filtered(&self) -> (Vec<crate::types::NodeId>, Vec<usize>) {
        self.values.as_node_ids_filtered()
    }

    /// Extract as EdgeId vector, filtering out rows with invalid values  
    pub fn as_edge_ids_filtered(&self) -> (Vec<crate::types::EdgeId>, Vec<usize>) {
        self.values.as_edge_ids_filtered()
    }

    // =================================================================
    // Sorting and indexing operations
    // =================================================================

    /// Get sort indices for this array
    pub fn sort_indices(&self, ascending: bool) -> Vec<usize> {
        self.values.sort_indices(ascending)
    }

    /// Take values at specified indices
    pub fn take_indices(&self, indices: &[usize]) -> crate::errors::GraphResult<Self> {
        self.values
            .take_indices(indices)
            .map(|values| Self::from_column(values, self.dtype))
    }

    // =================================================================
    // Filtering operations
    // =================================================================

    /// Create equality mask for string comparison
    pub fn eq_string(&self, value: &str) -> Vec<bool> {
        self.values
            .data()
            .iter()
            .map(|val| match val {
                AttrValue::Text(s) => s == value,
                _ => false,
            })
            .collect()
    }

    /// Filter by boolean mask
    pub fn filter_by_mask(&self, mask: &[bool]) -> crate::errors::GraphResult<Self> {
        self.values
            .filter_by_mask(mask)
            .map(|values| Self::from_column(values, self.dtype))
    }

    /// Slice this array
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let end = end.min(self.values.len());
        let start = start.min(end);
        Self::from_column(self.values.slice(start, end - start), self.dtype)
    }

    // =================================================================
    // Statistical operations
    // =================================================================

    /// Get unique values in this array
    pub fn unique_values(&self) -> crate::errors::GraphResult<Vec<AttrValue>> {
        self.values.unique_values()
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &Vec<AttrValue> {
        self.values.data()
    }
}

impl BaseArray {
    /// Set a value at a specific index
    pub fn set(&mut self, index: usize, value: AttrValue) -> crate::errors::GraphResult<()> {
        if index >= self.values.data().len() {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Index {} out of bounds for array with length {}",
                index,
                self.values.data().len()
            )));
        }

        self.values.set(index, value)
    }

    /// Get mutable access to the data for batch operations
    pub fn data_mut(&mut self) -> &mut Vec<AttrValue> {
        self.values.data_mut()
    }
}

// Continue with ArrayOps implementation - remaining methods
impl ArrayOps<AttrValue> for BaseArray {
    fn len(&self) -> usize {
        self.values.data().len()
    }

    fn get(&self, index: usize) -> Option<&AttrValue> {
        self.values.data().get(index)
    }

    fn iter(&self) -> ArrayIterator<AttrValue>
    where
        AttrValue: Clone + 'static,
    {
        ArrayIterator::new(self.values.data().clone())
    }
}

// =============================================================================
// Statistical operations (preserve existing PyGraphArray functionality)
// =============================================================================

impl BaseArray {
    /// Get the first n elements (head operation)
    pub fn head(&self, n: usize) -> BaseArray {
        let head_data: Vec<AttrValue> = self.values.data().iter().take(n).cloned().collect();

        BaseArray::new(head_data, self.dtype)
    }

    /// Get the last n elements (tail operation)
    pub fn tail(&self, n: usize) -> BaseArray {
        let start_idx = if self.values.data().len() > n {
            self.values.data().len() - n
        } else {
            0
        };
        let tail_data: Vec<AttrValue> =
            self.values.data().iter().skip(start_idx).cloned().collect();

        BaseArray::new(tail_data, self.dtype)
    }

    /// Get unique values in the array
    pub fn unique(&self) -> BaseArray {
        let mut seen = std::collections::HashSet::new();
        let unique_data: Vec<AttrValue> = self
            .values
            .data()
            .iter()
            .filter(|item| seen.insert((*item).clone()))
            .cloned()
            .collect();

        BaseArray::new(unique_data, self.dtype)
    }

    /// Count occurrences of each value
    pub fn value_counts(&self) -> std::collections::HashMap<AttrValue, usize> {
        let mut counts = std::collections::HashMap::new();
        for value in self.values.data() {
            *counts.entry(value.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Checked finite numeric sum shared with the public generic BaseArray.
    pub fn sum(&self) -> crate::errors::GraphResult<AttrValue> {
        self.values.sum()
    }

    /// Compensated finite numeric mean shared with the public generic BaseArray.
    pub fn mean(&self) -> crate::errors::GraphResult<f64> {
        self.values.mean()
    }

    /// First minimum with exact numeric ordering; retain legacy text/bool support.
    pub fn min(&self) -> crate::errors::GraphResult<AttrValue> {
        super::reductions::legacy_extremum(self.values.data(), true)
    }

    /// First maximum with exact numeric ordering; retain legacy text/bool support.
    pub fn max(&self) -> crate::errors::GraphResult<AttrValue> {
        super::reductions::legacy_extremum(self.values.data(), false)
    }

    /// Count non-null values in the array
    pub fn count(&self) -> usize {
        self.values.count()
    }

    /// Count unique non-null values in the array
    pub fn nunique(&self) -> usize {
        self.values.nunique()
    }

    /// Calculate basic statistics for numeric data
    pub fn describe(&self) -> std::collections::HashMap<String, f64> {
        let mut stats = std::collections::HashMap::new();

        // Convert to numbers where possible
        let numbers: Vec<f64> = self
            .values
            .data()
            .iter()
            .filter_map(|val| match val {
                AttrValue::Int(i) => Some(*i as f64),
                AttrValue::Float(f) => Some(*f as f64),
                AttrValue::Float64(f) => Some(*f),
                _ => None,
            })
            .collect();

        if numbers.is_empty() {
            return stats;
        }

        let count = numbers.len() as f64;
        let sum: f64 = numbers.iter().sum();
        let mean = sum / count;

        stats.insert("count".to_string(), count);
        stats.insert("sum".to_string(), sum);
        stats.insert("mean".to_string(), mean);

        if !numbers.is_empty() {
            let min = numbers.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = numbers.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            stats.insert("min".to_string(), min);
            stats.insert("max".to_string(), max);
        }

        stats
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
    /// use groggy::types::{AttrValue, AttrValueType};
    ///
    /// let array = BaseArray::new(vec![
    ///     AttrValue::Text("Engineering".to_string()),
    ///     AttrValue::Text("Marketing".to_string()),
    ///     AttrValue::Text("Sales".to_string())
    /// ], AttrValueType::Text);
    ///
    /// let check_values = vec![
    ///     AttrValue::Text("Engineering".to_string()),
    ///     AttrValue::Text("Marketing".to_string())
    /// ];
    ///
    /// let mask = array.isin(check_values)?;
    /// // mask will be [true, true, false]
    /// ```ignore
    pub fn isin(&self, values: Vec<AttrValue>) -> crate::errors::GraphResult<BaseArray> {
        // Create a set for efficient lookup - use direct AttrValue comparison
        let value_set: std::collections::HashSet<&AttrValue> = values.iter().collect();

        let mut mask_data = Vec::with_capacity(self.values.data().len());

        for attr_val in self.values.data() {
            let matches = value_set.contains(attr_val);
            mask_data.push(AttrValue::Bool(matches));
        }

        Ok(BaseArray::new(mask_data, AttrValueType::Bool))
    }
}

// =============================================================================
// Display and Debug implementations
// =============================================================================

impl std::fmt::Display for BaseArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name_str = self.name.as_deref().unwrap_or("BaseArray");
        write!(f, "{}[{}] (dtype: {:?})", name_str, self.len(), self.dtype)?;

        if !self.values.data().is_empty() {
            write!(f, "\nFirst 5 elements: [")?;
            for (i, item) in self.values.data().iter().take(5).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", item)?;
            }
            if self.values.data().len() > 5 {
                write!(f, ", ... ({} more)", self.values.data().len() - 5)?;
            }
            write!(f, "]")?;
        }

        Ok(())
    }
}

impl std::fmt::Debug for BaseArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BaseArray")
            .field("len", &self.len())
            .field("dtype", &self.dtype)
            .field("name", &self.name)
            .field(
                "data_sample",
                &self.values.data().iter().take(3).collect::<Vec<_>>(),
            )
            .finish()
    }
}
