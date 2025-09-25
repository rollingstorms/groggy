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
