//! Advanced indexing system for arrays
//!
//! This module provides a unified indexing interface that supports:
//! - Single integer indexing: `array[5]`
//! - Integer list indexing: `array[[0, 2, 5]]`
//! - Boolean indexing: `array[bool_mask]`
//! - Range/slice indexing: `array[:5]`, `array[::2]`

use crate::errors::{GraphError, GraphResult};
use crate::storage::array::{BaseArray, NumArray};
use crate::storage::BoolArray;
use crate::types::AttrValue;

/// Unified slice index type supporting all indexing operations
#[derive(Debug, Clone)]
pub enum SliceIndex {
    /// Single integer index: `array[5]`
    Single(i64),
    /// Range slice: `array[start:end:step]`
    Range {
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    },
    /// Integer list indexing: `array[[0, 2, 5]]`
    List(Vec<i64>),
    /// Boolean array indexing: `array[bool_mask]`
    BoolArray(BoolArray),
}

impl SliceIndex {
    /// Validate length-independent slice controls before language-boundary dispatch.
    pub fn validate(&self) -> GraphResult<()> {
        if let SliceIndex::Range { step, .. } = self {
            validate_step(step.unwrap_or(1))?;
        }
        Ok(())
    }
    /// Convert Python slice notation to indices
    pub fn resolve_indices(&self, length: usize) -> GraphResult<Vec<usize>> {
        match self {
            SliceIndex::Single(idx) => {
                let resolved = resolve_negative_index(*idx, length)?;
                Ok(vec![resolved])
            }
            SliceIndex::Range { start, stop, step } => {
                resolve_range_indices(*start, *stop, *step, length)
            }
            SliceIndex::List(indices) => {
                let mut resolved = Vec::with_capacity(indices.len());
                for &idx in indices {
                    resolved.push(resolve_negative_index(idx, length)?);
                }
                Ok(resolved)
            }
            SliceIndex::BoolArray(mask) => {
                if mask.len() != length {
                    return Err(GraphError::InvalidInput(format!(
                        "Boolean mask length {} doesn't match array length {}",
                        mask.len(),
                        length
                    )));
                }
                Ok(mask.nonzero())
            }
        }
    }
}

/// Resolve negative indices to positive indices
fn resolve_negative_index(index: i64, length: usize) -> GraphResult<usize> {
    let len = i128::try_from(length)
        .map_err(|_| GraphError::InvalidInput("Array length exceeds index working range".into()))?;
    let index = i128::from(index);
    let resolved = if index < 0 { len + index } else { index };
    if resolved < 0 || resolved >= len {
        return Err(GraphError::InvalidInput(format!(
            "Index {index} out of range for array of length {length}"
        )));
    }
    usize::try_from(resolved)
        .map_err(|_| GraphError::InvalidInput("Resolved index exceeds platform range".into()))
}
fn validate_step(step: i64) -> GraphResult<()> {
    if step == 0 {
        Err(GraphError::InvalidInput("Slice step cannot be zero".into()))
    } else {
        Ok(())
    }
}
/// Resolve raw slice bounds once against the actual length, including omitted
/// reverse bounds. i128 working positions prevent i64 step-addition overflow.
fn resolve_range_indices(
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
    length: usize,
) -> GraphResult<Vec<usize>> {
    let step = step.unwrap_or(1);
    validate_step(step)?;
    let step = i128::from(step);
    let len = i128::try_from(length)
        .map_err(|_| GraphError::InvalidInput("Array length exceeds slice working range".into()))?;
    let (lower, upper) = if step > 0 { (0, len) } else { (-1, len - 1) };
    let bound = |value: i64| {
        let value = i128::from(value);
        let value = if value < 0 { len + value } else { value };
        value.clamp(lower, upper)
    };
    let start = start
        .map(bound)
        .unwrap_or(if step > 0 { 0 } else { len - 1 });
    let stop = stop.map(bound).unwrap_or(if step > 0 { len } else { -1 });
    let span = if step > 0 { stop - start } else { start - stop };
    let count = if span <= 0 {
        0
    } else {
        (span - 1) / step.abs() + 1
    };
    let count = usize::try_from(count).map_err(|_| {
        GraphError::InvalidInput("Slice result length exceeds platform range".into())
    })?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(count)
        .map_err(|_| GraphError::InvalidInput("Slice index allocation failed".into()))?;
    let mut current = start;
    for position in 0..count {
        indices.push(
            usize::try_from(current).map_err(|_| {
                GraphError::InvalidInput("Slice produced an invalid position".into())
            })?,
        );
        if position + 1 < count {
            current = current.checked_add(step).ok_or_else(|| {
                GraphError::InvalidInput("Slice position exceeds working range".into())
            })?;
        }
    }
    Ok(indices)
}

/// Trait for types that support advanced indexing
pub trait AdvancedIndexing<T> {
    /// Get elements using advanced indexing
    fn get_slice(&self, index: &SliceIndex) -> GraphResult<Self>
    where
        Self: Sized;

    /// Get a single element by index
    fn get_single(&self, index: usize) -> GraphResult<T>;

    /// Get length for index validation
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Implementation for BaseArray<AttrValue>
impl AdvancedIndexing<AttrValue> for BaseArray<AttrValue> {
    fn get_slice(&self, index: &SliceIndex) -> GraphResult<Self> {
        let indices = index.resolve_indices(self.len())?;
        let mut result_data = Vec::with_capacity(indices.len());

        for &idx in &indices {
            if let Some(value) = self.get(idx) {
                result_data.push(value.clone());
            } else {
                return Err(GraphError::InvalidInput(format!(
                    "Index {} out of bounds",
                    idx
                )));
            }
        }

        Ok(BaseArray::new(result_data))
    }

    fn get_single(&self, index: usize) -> GraphResult<AttrValue> {
        self.get(index)
            .cloned()
            .ok_or_else(|| GraphError::InvalidInput(format!("Index {} out of bounds", index)))
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// Implementation for NumArray<T>
impl<T> AdvancedIndexing<T> for NumArray<T>
where
    T: Clone + Copy + std::fmt::Debug,
{
    fn get_slice(&self, index: &SliceIndex) -> GraphResult<Self> {
        let indices = index.resolve_indices(self.len())?;
        let mut result_data = Vec::with_capacity(indices.len());

        for &idx in &indices {
            if let Some(&value) = self.get(idx) {
                result_data.push(value);
            } else {
                return Err(GraphError::InvalidInput(format!(
                    "Index {} out of bounds",
                    idx
                )));
            }
        }

        Ok(NumArray::new(result_data))
    }

    fn get_single(&self, index: usize) -> GraphResult<T> {
        self.get(index)
            .copied()
            .ok_or_else(|| GraphError::InvalidInput(format!("Index {} out of bounds", index)))
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// Implementation for BoolArray
impl AdvancedIndexing<bool> for BoolArray {
    fn get_slice(&self, index: &SliceIndex) -> GraphResult<Self> {
        let indices = index.resolve_indices(self.len())?;
        let mut result_data = Vec::with_capacity(indices.len());

        for &idx in &indices {
            if let Some(value) = self.get(idx) {
                result_data.push(value);
            } else {
                return Err(GraphError::InvalidInput(format!(
                    "Index {} out of bounds",
                    idx
                )));
            }
        }

        Ok(BoolArray::new(result_data))
    }

    fn get_single(&self, index: usize) -> GraphResult<bool> {
        self.get(index)
            .ok_or_else(|| GraphError::InvalidInput(format!("Index {} out of bounds", index)))
    }

    fn len(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_index_resolution() {
        let idx = SliceIndex::Single(2);
        let indices = idx.resolve_indices(10).unwrap();
        assert_eq!(indices, vec![2]);

        // Test negative index
        let idx = SliceIndex::Single(-1);
        let indices = idx.resolve_indices(10).unwrap();
        assert_eq!(indices, vec![9]);
    }

    #[test]
    fn test_range_index_resolution() {
        // Test simple range
        let idx = SliceIndex::Range {
            start: Some(1),
            stop: Some(4),
            step: Some(1),
        };
        let indices = idx.resolve_indices(10).unwrap();
        assert_eq!(indices, vec![1, 2, 3]);

        // Test step size
        let idx = SliceIndex::Range {
            start: Some(0),
            stop: Some(10),
            step: Some(2),
        };
        let indices = idx.resolve_indices(10).unwrap();
        assert_eq!(indices, vec![0, 2, 4, 6, 8]);

        // Test None defaults
        let idx = SliceIndex::Range {
            start: None,
            stop: Some(3),
            step: None,
        };
        let indices = idx.resolve_indices(10).unwrap();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_list_index_resolution() {
        let idx = SliceIndex::List(vec![0, 2, 4, -1]);
        let indices = idx.resolve_indices(10).unwrap();
        assert_eq!(indices, vec![0, 2, 4, 9]);
    }

    #[test]
    fn test_bool_array_indexing() {
        let mask_data = vec![true, false, true, false, true];
        let mask = BoolArray::new(mask_data);
        let idx = SliceIndex::BoolArray(mask);
        let indices = idx.resolve_indices(5).unwrap();
        assert_eq!(indices, vec![0, 2, 4]);
    }

    #[test]
    fn test_advanced_indexing_base_array() {
        use crate::types::AttrValue;

        let data = vec![
            AttrValue::SmallInt(10),
            AttrValue::SmallInt(20),
            AttrValue::SmallInt(30),
            AttrValue::SmallInt(40),
            AttrValue::SmallInt(50),
        ];
        let array = BaseArray::new(data);

        // Test list indexing
        let list_idx = SliceIndex::List(vec![0, 2, 4]);
        let sliced = array.get_slice(&list_idx).unwrap();
        assert_eq!(sliced.len(), 3);

        // Test range indexing
        let range_idx = SliceIndex::Range {
            start: Some(1),
            stop: Some(4),
            step: Some(1),
        };
        let sliced = array.get_slice(&range_idx).unwrap();
        assert_eq!(sliced.len(), 3);
    }
}
