//! BaseArray - Foundation columnar array with chaining support

use crate::types::{AttrValue, AttrValueType};
use crate::storage::array::{ArrayOps, ArrayIterator};

/// BaseArray - Foundation columnar storage with statistical operations and chaining
/// This replaces and extends the current PyGraphArray functionality
#[derive(Clone)]
pub struct BaseArray {
    /// Columnar data storage - optimized for AttrValue collections  
    data: Vec<AttrValue>,
    /// Data type for type consistency and optimization
    dtype: AttrValueType,
    /// Optional column name for debugging and display
    name: Option<String>,
}

impl BaseArray {
    /// Create a new BaseArray from AttrValue vector
    pub fn new(data: Vec<AttrValue>, dtype: AttrValueType) -> Self {
        Self {
            data,
            dtype,
            name: None,
        }
    }
    
    /// Create a new BaseArray with a name
    pub fn with_name(data: Vec<AttrValue>, dtype: AttrValueType, name: String) -> Self {
        Self {
            data,
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
    pub fn from_node_ids(node_ids: Vec<crate::types::NodeId>) -> Self {
        let data = node_ids.into_iter().map(|id| AttrValue::Int(id as i64)).collect();
        Self::new(data, AttrValueType::Int)
    }
    
    /// Create BaseArray from EdgeId vector
    pub fn from_edge_ids(edge_ids: Vec<crate::types::EdgeId>) -> Self {
        let data = edge_ids.into_iter().map(|id| AttrValue::Int(id as i64)).collect();
        Self::new(data, AttrValueType::Int)
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
        self.data.iter().map(|val| {
            match val {
                AttrValue::Int(i) => Ok(*i as crate::types::NodeId),
                _ => Err(crate::errors::GraphError::InvalidInput(
                    "Cannot convert non-integer to NodeId".to_string()
                ))
            }
        }).collect()
    }
    
    /// Extract as EdgeId vector (assumes Int type)
    pub fn as_edge_ids(&self) -> crate::errors::GraphResult<Vec<crate::types::EdgeId>> {
        self.data.iter().map(|val| {
            match val {
                AttrValue::Int(i) => Ok(*i as crate::types::EdgeId),
                _ => Err(crate::errors::GraphError::InvalidInput(
                    "Cannot convert non-integer to EdgeId".to_string()
                ))
            }
        }).collect()
    }
    
    /// Extract as NodeId vector, filtering out rows with invalid values
    pub fn as_node_ids_filtered(&self) -> (Vec<crate::types::NodeId>, Vec<usize>) {
        let mut node_ids = Vec::new();
        let mut valid_indices = Vec::new();
        
        for (index, val) in self.data.iter().enumerate() {
            match val {
                AttrValue::Int(i) => {
                    node_ids.push(*i as crate::types::NodeId);
                    valid_indices.push(index);
                }
                _ => {} // Skip invalid values
            }
        }
        
        (node_ids, valid_indices)
    }
    
    /// Extract as EdgeId vector, filtering out rows with invalid values  
    pub fn as_edge_ids_filtered(&self) -> (Vec<crate::types::EdgeId>, Vec<usize>) {
        let mut edge_ids = Vec::new();
        let mut valid_indices = Vec::new();
        
        for (index, val) in self.data.iter().enumerate() {
            match val {
                AttrValue::Int(i) => {
                    edge_ids.push(*i as crate::types::EdgeId);
                    valid_indices.push(index);
                }
                _ => {} // Skip invalid values
            }
        }
        
        (edge_ids, valid_indices)
    }
    
    // =================================================================
    // Sorting and indexing operations
    // =================================================================
    
    /// Get sort indices for this array
    pub fn sort_indices(&self, ascending: bool) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.data.len()).collect();
        
        if ascending {
            indices.sort_by(|&a, &b| self.data[a].partial_cmp(&self.data[b]).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            indices.sort_by(|&a, &b| self.data[b].partial_cmp(&self.data[a]).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        indices
    }
    
    /// Take values at specified indices
    pub fn take_indices(&self, indices: &[usize]) -> crate::errors::GraphResult<Self> {
        let mut new_data = Vec::with_capacity(indices.len());
        
        for &idx in indices {
            if idx >= self.data.len() {
                return Err(crate::errors::GraphError::InvalidInput(
                    format!("Index {} out of bounds for array of length {}", idx, self.data.len())
                ));
            }
            new_data.push(self.data[idx].clone());
        }
        
        Ok(Self::new(new_data, self.dtype.clone()))
    }
    
    // =================================================================
    // Filtering operations
    // =================================================================
    
    /// Create equality mask for string comparison
    pub fn eq_string(&self, value: &str) -> Vec<bool> {
        self.data.iter().map(|val| {
            match val {
                AttrValue::Text(s) => s == value,
                _ => false,
            }
        }).collect()
    }
    
    /// Filter by boolean mask
    pub fn filter_by_mask(&self, mask: &[bool]) -> crate::errors::GraphResult<Self> {
        if mask.len() != self.data.len() {
            return Err(crate::errors::GraphError::InvalidInput(
                format!("Mask length {} doesn't match array length {}", mask.len(), self.data.len())
            ));
        }
        
        let filtered_data: Vec<AttrValue> = self.data.iter()
            .zip(mask.iter())
            .filter_map(|(val, &keep)| if keep { Some(val.clone()) } else { None })
            .collect();
        
        Ok(Self::new(filtered_data, self.dtype.clone()))
    }
    
    /// Slice this array 
    pub fn slice(&self, start: usize, end: usize) -> Self {
        let end = std::cmp::min(end, self.data.len());
        let start = std::cmp::min(start, end);
        
        let sliced_data = self.data[start..end].to_vec();
        Self::new(sliced_data, self.dtype.clone())
    }
    
    // =================================================================
    // Statistical operations
    // =================================================================
    
    /// Get unique values in this array
    pub fn unique_values(&self) -> crate::errors::GraphResult<Vec<AttrValue>> {
        let mut unique_set = std::collections::HashSet::new();
        let mut unique_list = Vec::new();
        
        for val in &self.data {
            if unique_set.insert(val.clone()) {
                unique_list.push(val.clone());
            }
        }
        
        Ok(unique_list)
    }
    
    /// Get a reference to the underlying data
    pub fn data(&self) -> &Vec<AttrValue> {
        &self.data
    }
}

// Implement ArrayOps for BaseArray
impl ArrayOps<AttrValue> for BaseArray {
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn get(&self, index: usize) -> Option<&AttrValue> {
        self.data.get(index)
    }
    
    fn iter(&self) -> ArrayIterator<AttrValue> 
    where 
        AttrValue: Clone + 'static 
    {
        ArrayIterator::new(self.data.clone())
    }
}

// =============================================================================
// Statistical operations (preserve existing PyGraphArray functionality)
// =============================================================================

impl BaseArray {
    /// Get the first n elements (head operation)
    pub fn head(&self, n: usize) -> BaseArray {
        let head_data: Vec<AttrValue> = self.data
            .iter()
            .take(n)
            .cloned()
            .collect();
            
        BaseArray::new(head_data, self.dtype.clone())
    }
    
    /// Get the last n elements (tail operation)
    pub fn tail(&self, n: usize) -> BaseArray {
        let start_idx = if self.data.len() > n { self.data.len() - n } else { 0 };
        let tail_data: Vec<AttrValue> = self.data
            .iter()
            .skip(start_idx)
            .cloned()
            .collect();
            
        BaseArray::new(tail_data, self.dtype.clone())
    }
    
    /// Get unique values in the array
    pub fn unique(&self) -> BaseArray {
        let mut seen = std::collections::HashSet::new();
        let unique_data: Vec<AttrValue> = self.data
            .iter()
            .filter(|item| seen.insert((*item).clone()))
            .cloned()
            .collect();
            
        BaseArray::new(unique_data, self.dtype.clone())
    }
    
    /// Count occurrences of each value
    pub fn value_counts(&self) -> std::collections::HashMap<AttrValue, usize> {
        let mut counts = std::collections::HashMap::new();
        for value in &self.data {
            *counts.entry(value.clone()).or_insert(0) += 1;
        }
        counts
    }
    
    /// Calculate basic statistics for numeric data
    pub fn describe(&self) -> std::collections::HashMap<String, f64> {
        let mut stats = std::collections::HashMap::new();
        
        // Convert to numbers where possible
        let numbers: Vec<f64> = self.data
            .iter()
            .filter_map(|val| match val {
                AttrValue::Int(i) => Some(*i as f64),
                AttrValue::Float(f) => Some(*f as f64),
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
}


// =============================================================================
// Display and Debug implementations
// =============================================================================

impl std::fmt::Display for BaseArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name_str = self.name.as_deref().unwrap_or("BaseArray");
        write!(f, "{}[{}] (dtype: {:?})", name_str, self.len(), self.dtype)?;
        
        if !self.data.is_empty() {
            write!(f, "\nFirst 5 elements: [")?;
            for (i, item) in self.data.iter().take(5).enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{:?}", item)?;
            }
            if self.data.len() > 5 {
                write!(f, ", ... ({} more)", self.data.len() - 5)?;
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
            .field("data_sample", &self.data.iter().take(3).collect::<Vec<_>>())
            .finish()
    }
}