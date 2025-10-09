//! BoolArray - Efficient boolean operations and indexing
//!
//! This module provides a specialized boolean array implementation based on NumArray<bool>
//! for efficient boolean operations, masking, and indexing operations.

use super::NumArray;
use crate::errors::GraphResult;
use std::ops::{BitAnd, BitOr, Not};

/// Specialized boolean array for efficient boolean operations and indexing
#[derive(Debug, Clone)]
pub struct BoolArray {
    inner: NumArray<bool>,
    length: usize,
}

impl BoolArray {
    /// Create a new BoolArray from a vector of booleans
    pub fn new(data: Vec<bool>) -> Self {
        let length = data.len();
        Self {
            inner: NumArray::new(data),
            length,
        }
    }

    /// Create a new BoolArray from a slice of booleans
    pub fn from_slice(data: &[bool]) -> Self {
        Self::new(data.to_vec())
    }

    /// Get the length of the boolean array
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get an element at the specified index
    pub fn get(&self, index: usize) -> Option<bool> {
        self.inner.get(index).copied()
    }

    /// Get an iterator over the boolean values
    pub fn iter(&self) -> impl Iterator<Item = &bool> {
        self.inner.iter()
    }

    /// Count the number of True values
    pub fn count(&self) -> usize {
        self.inner.iter().filter(|&&x| x).count()
    }

    /// Count the number of False values
    pub fn count_false(&self) -> usize {
        self.length - self.count()
    }

    /// Check if any value is True
    pub fn any(&self) -> bool {
        self.inner.iter().any(|&x| x)
    }

    /// Check if all values are True
    pub fn all(&self) -> bool {
        self.inner.iter().all(|&x| x)
    }

    /// Get percentage of True values
    pub fn percentage(&self) -> f64 {
        if self.length == 0 {
            0.0
        } else {
            (self.count() as f64 / self.length as f64) * 100.0
        }
    }

    /// Convert to indices where value is True
    /// Returns a Vec<usize> of indices where the boolean is True
    pub fn nonzero(&self) -> Vec<usize> {
        self.inner
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val { Some(i) } else { None })
            .collect()
    }

    /// Convert to indices where value is True (alias for nonzero)
    pub fn to_indices(&self) -> Vec<usize> {
        self.nonzero()
    }

    /// Convert to indices where value is False
    pub fn false_indices(&self) -> Vec<usize> {
        self.inner
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if !val { Some(i) } else { None })
            .collect()
    }

    /// Apply this boolean mask to a slice of data
    /// Returns elements where the mask is True
    pub fn apply_mask<T: Clone>(&self, data: &[T]) -> GraphResult<Vec<T>> {
        if data.len() != self.length {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Array length mismatch: expected {}, got {}",
                self.length,
                data.len()
            )));
        }

        Ok(data
            .iter()
            .zip(self.inner.iter())
            .filter_map(|(item, &mask)| if mask { Some(item.clone()) } else { None })
            .collect())
    }

    /// Get access to underlying NumArray<bool>
    pub fn as_num_array(&self) -> &NumArray<bool> {
        &self.inner
    }

    /// Convert to underlying NumArray<bool> (consuming)
    pub fn into_num_array(self) -> NumArray<bool> {
        self.inner
    }

    /// Convert to a Vec<bool>
    pub fn to_vec(&self) -> Vec<bool> {
        self.inner.iter().copied().collect()
    }
}

// Implement boolean operations using bitwise operators (following NumPy/pandas convention)
impl BitAnd for BoolArray {
    type Output = BoolArray;

    fn bitand(self, other: BoolArray) -> BoolArray {
        if self.length != other.length {
            // Return empty array on length mismatch for now
            // TODO: Better error handling
            return BoolArray::new(vec![]);
        }

        let result_data: Vec<bool> = self
            .inner
            .iter()
            .zip(other.inner.iter())
            .map(|(&a, &b)| a && b)
            .collect();

        BoolArray::new(result_data)
    }
}

impl BitAnd for &BoolArray {
    type Output = BoolArray;

    fn bitand(self, other: &BoolArray) -> BoolArray {
        if self.length != other.length {
            return BoolArray::new(vec![]);
        }

        let result_data: Vec<bool> = self
            .inner
            .iter()
            .zip(other.inner.iter())
            .map(|(&a, &b)| a && b)
            .collect();

        BoolArray::new(result_data)
    }
}

impl BitOr for BoolArray {
    type Output = BoolArray;

    fn bitor(self, other: BoolArray) -> BoolArray {
        if self.length != other.length {
            return BoolArray::new(vec![]);
        }

        let result_data: Vec<bool> = self
            .inner
            .iter()
            .zip(other.inner.iter())
            .map(|(&a, &b)| a || b)
            .collect();

        BoolArray::new(result_data)
    }
}

impl BitOr for &BoolArray {
    type Output = BoolArray;

    fn bitor(self, other: &BoolArray) -> BoolArray {
        if self.length != other.length {
            return BoolArray::new(vec![]);
        }

        let result_data: Vec<bool> = self
            .inner
            .iter()
            .zip(other.inner.iter())
            .map(|(&a, &b)| a || b)
            .collect();

        BoolArray::new(result_data)
    }
}

impl Not for BoolArray {
    type Output = BoolArray;

    fn not(self) -> BoolArray {
        let result_data: Vec<bool> = self.inner.iter().map(|&x| !x).collect();

        BoolArray::new(result_data)
    }
}

impl Not for &BoolArray {
    type Output = BoolArray;

    fn not(self) -> BoolArray {
        let result_data: Vec<bool> = self.inner.iter().map(|&x| !x).collect();

        BoolArray::new(result_data)
    }
}

// Display implementation
impl std::fmt::Display for BoolArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Short array display
        if self.length <= 10 {
            write!(f, "BoolArray([")?;
            for (i, &val) in self.inner.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", val)?;
            }
            writeln!(f, "])")?;

            // Show indexed values
            for (i, &val) in self.inner.iter().enumerate() {
                writeln!(f, "[{}] {}", i, val)?;
            }
        } else {
            // Large array display with truncation
            write!(f, "BoolArray([")?;
            for (i, &val) in self.inner.iter().take(3).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", val)?;
            }
            write!(f, ", ..., ")?;
            for (i, &val) in self.inner.iter().skip(self.length - 2).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", val)?;
            }
            writeln!(f, "], length={})", self.length)?;
        }

        // Statistics
        let count = self.count();
        let percentage = self.percentage();
        write!(f, "Count: {}/{} ({:.1}%)", count, self.length, percentage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_array_creation() {
        let data = vec![true, false, true, false];
        let bool_array = BoolArray::new(data);

        assert_eq!(bool_array.len(), 4);
        assert_eq!(bool_array.get(0), Some(true));
        assert_eq!(bool_array.get(1), Some(false));
        assert_eq!(bool_array.get(2), Some(true));
        assert_eq!(bool_array.get(3), Some(false));
    }

    #[test]
    fn test_bool_array_statistics() {
        let data = vec![true, false, true, true, false];
        let bool_array = BoolArray::new(data);

        assert_eq!(bool_array.count(), 3);
        assert_eq!(bool_array.count_false(), 2);
        assert!(bool_array.any());
        assert!(!bool_array.all());
        assert!((bool_array.percentage() - 60.0).abs() < 0.1);
    }

    #[test]
    fn test_bool_array_indices() {
        let data = vec![true, false, true, false, true];
        let bool_array = BoolArray::new(data);

        assert_eq!(bool_array.nonzero(), vec![0, 2, 4]);
        assert_eq!(bool_array.to_indices(), vec![0, 2, 4]);
        assert_eq!(bool_array.false_indices(), vec![1, 3]);
    }

    #[test]
    fn test_bool_array_operations() {
        let data1 = vec![true, false, true, false];
        let data2 = vec![false, true, true, false];
        let bool_array1 = BoolArray::new(data1);
        let bool_array2 = BoolArray::new(data2);

        // Test AND operation
        let and_result = &bool_array1 & &bool_array2;
        assert_eq!(and_result.to_vec(), vec![false, false, true, false]);

        // Test OR operation
        let or_result = &bool_array1 | &bool_array2;
        assert_eq!(or_result.to_vec(), vec![true, true, true, false]);

        // Test NOT operation
        let not_result = !&bool_array1;
        assert_eq!(not_result.to_vec(), vec![false, true, false, true]);
    }

    #[test]
    fn test_apply_mask() {
        let mask_data = vec![true, false, true, false, true];
        let mask = BoolArray::new(mask_data);
        let data = vec!["a", "b", "c", "d", "e"];

        let filtered = mask.apply_mask(&data).unwrap();
        assert_eq!(filtered, vec!["a", "c", "e"]);
    }

    #[test]
    fn test_empty_array() {
        let bool_array = BoolArray::new(vec![]);
        assert!(bool_array.is_empty());
        assert_eq!(bool_array.count(), 0);
        assert!(!bool_array.any());
        assert!(bool_array.all()); // Vacuous truth
        assert_eq!(bool_array.percentage(), 0.0);
    }
}
