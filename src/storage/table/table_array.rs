//! TableArray - Core implementation for arrays of tables
//!
//! This enables powerful groupby operations where groupby() returns a TableArray
//! and operations like .sum(), .agg() work on the entire array of tables.

use super::{traits::Table, BaseTable};
use crate::errors::GraphResult;
use crate::types::AttrValue;
use std::collections::HashMap;

/// Core TableArray - an array of BaseTable instances
///
/// This is the foundation for advanced table operations like groupby().sum()
/// where operations are applied across multiple tables and aggregated.
#[derive(Debug, Clone)]
pub struct TableArray {
    /// The tables in this array
    pub tables: Vec<BaseTable>,
    /// Optional group keys for each table (used by groupby)
    pub group_keys: Option<Vec<HashMap<String, AttrValue>>>,
}

impl TableArray {
    /// Create a new empty TableArray
    pub fn new() -> Self {
        Self {
            tables: Vec::new(),
            group_keys: None,
        }
    }

    /// Create a TableArray from a vector of tables
    pub fn from_tables(tables: Vec<BaseTable>) -> Self {
        Self {
            tables,
            group_keys: None,
        }
    }

    /// Create a TableArray with group keys (for groupby results)
    pub fn from_tables_with_keys(
        tables: Vec<BaseTable>,
        group_keys: Vec<HashMap<String, AttrValue>>,
    ) -> Self {
        Self {
            tables,
            group_keys: Some(group_keys),
        }
    }

    /// Get the number of tables in the array
    pub fn len(&self) -> usize {
        self.tables.len()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    /// Get a reference to a table by index
    pub fn get(&self, index: usize) -> Option<&BaseTable> {
        self.tables.get(index)
    }

    /// Get a mutable reference to a table by index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut BaseTable> {
        self.tables.get_mut(index)
    }

    /// Add a table to the array
    pub fn push(&mut self, table: BaseTable) {
        self.tables.push(table);
    }

    /// Add a table with a group key
    pub fn push_with_key(&mut self, table: BaseTable, key: HashMap<String, AttrValue>) {
        self.tables.push(table);
        if let Some(ref mut keys) = self.group_keys {
            keys.push(key);
        } else {
            self.group_keys = Some(vec![key]);
        }
    }

    /// Apply sum aggregation across all tables
    /// Returns a new table with summed values for each column
    pub fn sum(&self) -> GraphResult<BaseTable> {
        if self.tables.is_empty() {
            return Ok(BaseTable::new());
        }

        // Start with the first table's structure
        let first_table = &self.tables[0];
        let mut result_columns: HashMap<String, Vec<AttrValue>> = HashMap::new();

        // Initialize result columns
        for col_name in first_table.column_order() {
            result_columns.insert(col_name.clone(), Vec::new());
        }

        // Add group key columns if they exist
        if let Some(ref group_keys) = self.group_keys {
            for (table_idx, table) in self.tables.iter().enumerate() {
                if let Some(group_key) = group_keys.get(table_idx) {
                    // Add group key values to result
                    for (key_name, key_value) in group_key {
                        if !result_columns.contains_key(key_name) {
                            result_columns.insert(key_name.clone(), Vec::new());
                        }
                        result_columns
                            .get_mut(key_name)
                            .unwrap()
                            .push(key_value.clone());
                    }
                }

                // Calculate sum for each numeric column
                for col_name in first_table.column_order() {
                    if let Some(column) = table.columns().get(col_name) {
                        let column_sum = column.sum()?;
                        result_columns.get_mut(col_name).unwrap().push(column_sum);
                    }
                }
            }
        } else {
            // No group keys, just aggregate all tables into one row
            let mut row_data = HashMap::new();

            for col_name in first_table.column_order() {
                let mut total_sum = AttrValue::Int(0);
                let mut is_float = false;
                let mut float_sum = 0.0;
                let mut int_sum: i64 = 0;

                for table in &self.tables {
                    if let Some(column) = table.columns().get(col_name) {
                        let column_sum = column.sum()?;
                        match (&total_sum, &column_sum) {
                            (_, AttrValue::Float(f)) => {
                                if !is_float {
                                    float_sum = int_sum as f64;
                                    is_float = true;
                                }
                                float_sum += *f as f64;
                            }
                            (_, AttrValue::Int(i)) => {
                                if is_float {
                                    float_sum += *i as f64;
                                } else {
                                    int_sum += i;
                                }
                            }
                            _ => {} // Skip non-numeric values
                        }
                    }
                }

                total_sum = if is_float {
                    AttrValue::Float(float_sum as f32)
                } else {
                    AttrValue::Int(int_sum)
                };

                row_data.insert(col_name.clone(), vec![total_sum]);
            }

            result_columns = row_data;
        }

        // Convert to BaseArray columns
        let mut base_columns = HashMap::new();
        for (col_name, values) in result_columns {
            if !values.is_empty() {
                base_columns.insert(
                    col_name,
                    crate::storage::array::BaseArray::from_attr_values(values),
                );
            }
        }

        // Create result table with proper column order
        if let Some(ref group_keys) = self.group_keys {
            if let Some(first_key) = group_keys.first() {
                let mut column_order = first_key.keys().cloned().collect::<Vec<_>>();
                column_order.extend(first_table.column_order().iter().cloned());
                let result = BaseTable::with_column_order(base_columns, column_order)?;
                return Ok(result);
            }
        }

        // Default case: use first table's column order
        let result =
            BaseTable::with_column_order(base_columns, first_table.column_order().to_vec())?;

        Ok(result)
    }

    /// Apply mean aggregation across all tables
    pub fn mean(&self) -> GraphResult<BaseTable> {
        if self.tables.is_empty() {
            return Ok(BaseTable::new());
        }

        // Similar to sum but calculate means
        let first_table = &self.tables[0];
        let mut result_columns: HashMap<String, Vec<AttrValue>> = HashMap::new();

        // Initialize result columns
        for col_name in first_table.column_order() {
            result_columns.insert(col_name.clone(), Vec::new());
        }

        if let Some(ref group_keys) = self.group_keys {
            for (table_idx, table) in self.tables.iter().enumerate() {
                if let Some(group_key) = group_keys.get(table_idx) {
                    // Add group key values to result
                    for (key_name, key_value) in group_key {
                        if !result_columns.contains_key(key_name) {
                            result_columns.insert(key_name.clone(), Vec::new());
                        }
                        result_columns
                            .get_mut(key_name)
                            .unwrap()
                            .push(key_value.clone());
                    }
                }

                // Calculate mean for each numeric column
                for col_name in first_table.column_order() {
                    if let Some(column) = table.columns().get(col_name) {
                        let column_mean = AttrValue::Float(column.mean()? as f32);
                        result_columns.get_mut(col_name).unwrap().push(column_mean);
                    }
                }
            }
        }

        // Convert to BaseArray columns
        let mut base_columns = HashMap::new();
        for (col_name, values) in result_columns {
            if !values.is_empty() {
                base_columns.insert(
                    col_name,
                    crate::storage::array::BaseArray::from_attr_values(values),
                );
            }
        }

        let result = BaseTable::from_columns(base_columns)?;

        if let Some(ref group_keys) = self.group_keys {
            if let Some(first_key) = group_keys.first() {
                let mut column_order = first_key.keys().cloned().collect::<Vec<_>>();
                column_order.extend(first_table.column_order().iter().cloned());
                // column_order will be set using with_column_order
            }
        }

        Ok(result)
    }

    /// Apply count aggregation across all tables
    pub fn count(&self) -> GraphResult<BaseTable> {
        if self.tables.is_empty() {
            return Ok(BaseTable::new());
        }

        let first_table = &self.tables[0];
        let mut result_columns: HashMap<String, Vec<AttrValue>> = HashMap::new();

        // Initialize result columns
        for col_name in first_table.column_order() {
            result_columns.insert(col_name.clone(), Vec::new());
        }

        if let Some(ref group_keys) = self.group_keys {
            for (table_idx, table) in self.tables.iter().enumerate() {
                if let Some(group_key) = group_keys.get(table_idx) {
                    // Add group key values
                    for (key_name, key_value) in group_key {
                        if !result_columns.contains_key(key_name) {
                            result_columns.insert(key_name.clone(), Vec::new());
                        }
                        result_columns
                            .get_mut(key_name)
                            .unwrap()
                            .push(key_value.clone());
                    }
                }

                // Count non-null values for each column
                for col_name in first_table.column_order() {
                    if let Some(column) = table.columns().get(col_name) {
                        let column_count = AttrValue::Int(column.count() as i64);
                        result_columns.get_mut(col_name).unwrap().push(column_count);
                    }
                }
            }
        }

        // Convert to BaseArray columns
        let mut base_columns = HashMap::new();
        for (col_name, values) in result_columns {
            if !values.is_empty() {
                base_columns.insert(
                    col_name,
                    crate::storage::array::BaseArray::from_attr_values(values),
                );
            }
        }

        let result = BaseTable::from_columns(base_columns)?;

        if let Some(ref group_keys) = self.group_keys {
            if let Some(first_key) = group_keys.first() {
                let mut column_order = first_key.keys().cloned().collect::<Vec<_>>();
                column_order.extend(first_table.column_order().iter().cloned());
                // column_order will be set using with_column_order
            }
        }

        Ok(result)
    }

    /// Apply generic aggregation with custom functions
    /// agg_functions maps column names to aggregation function names
    pub fn agg(&self, agg_functions: HashMap<String, String>) -> GraphResult<BaseTable> {
        if self.tables.is_empty() {
            return Ok(BaseTable::new());
        }

        let first_table = &self.tables[0];
        let mut result_columns: HashMap<String, Vec<AttrValue>> = HashMap::new();

        if let Some(ref group_keys) = self.group_keys {
            for (table_idx, table) in self.tables.iter().enumerate() {
                if let Some(group_key) = group_keys.get(table_idx) {
                    // Add group key values
                    for (key_name, key_value) in group_key {
                        if !result_columns.contains_key(key_name) {
                            result_columns.insert(key_name.clone(), Vec::new());
                        }
                        result_columns
                            .get_mut(key_name)
                            .unwrap()
                            .push(key_value.clone());
                    }
                }

                // Apply aggregation functions
                for (col_name, agg_func) in &agg_functions {
                    if let Some(column) = table.columns().get(col_name) {
                        let result_value = match agg_func.as_str() {
                            "sum" => column.sum()?,
                            "mean" => AttrValue::Float(column.mean()? as f32),
                            "count" => AttrValue::Int(column.count() as i64),
                            "min" => column.min()?,
                            "max" => column.max()?,
                            "nunique" => AttrValue::Int(column.nunique() as i64),
                            _ => {
                                return Err(crate::errors::GraphError::InvalidInput(format!(
                                    "Unknown aggregation function: {}",
                                    agg_func
                                )))
                            }
                        };

                        if !result_columns.contains_key(col_name) {
                            result_columns.insert(col_name.clone(), Vec::new());
                        }
                        result_columns.get_mut(col_name).unwrap().push(result_value);
                    }
                }
            }
        }

        // Convert to BaseArray columns
        let mut base_columns = HashMap::new();
        for (col_name, values) in result_columns {
            if !values.is_empty() {
                base_columns.insert(
                    col_name,
                    crate::storage::array::BaseArray::from_attr_values(values),
                );
            }
        }

        let result = BaseTable::from_columns(base_columns)?;

        if let Some(ref group_keys) = self.group_keys {
            if let Some(first_key) = group_keys.first() {
                let mut column_order = first_key.keys().cloned().collect::<Vec<_>>();
                column_order.extend(agg_functions.keys().cloned());
                // column_order will be set using with_column_order
            }
        }

        Ok(result)
    }

    /// Get an iterator over the tables
    pub fn iter(&self) -> impl Iterator<Item = &BaseTable> {
        self.tables.iter()
    }

    /// Get a mutable iterator over the tables
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut BaseTable> {
        self.tables.iter_mut()
    }

    /// Filter all tables in the array using a query predicate
    /// Returns a new TableArray with only tables that match the predicate
    pub fn filter<F>(&self, predicate: F) -> TableArray
    where
        F: Fn(&BaseTable) -> bool,
    {
        let filtered_tables: Vec<BaseTable> = self
            .tables
            .iter()
            .filter(|table| predicate(table))
            .cloned()
            .collect();

        let filtered_keys = if let Some(ref keys) = self.group_keys {
            let filtered_keys: Vec<HashMap<String, AttrValue>> = self
                .tables
                .iter()
                .zip(keys.iter())
                .filter(|(table, _)| predicate(table))
                .map(|(_, key)| key.clone())
                .collect();
            Some(filtered_keys)
        } else {
            None
        };

        if let Some(keys) = filtered_keys {
            TableArray::from_tables_with_keys(filtered_tables, keys)
        } else {
            TableArray::from_tables(filtered_tables)
        }
    }

    /// Take the first n tables from the array
    pub fn take(&self, n: usize) -> TableArray {
        let taken_tables = self.tables.iter().take(n).cloned().collect();

        let taken_keys = if let Some(ref keys) = self.group_keys {
            Some(keys.iter().take(n).cloned().collect())
        } else {
            None
        };

        if let Some(keys) = taken_keys {
            TableArray::from_tables_with_keys(taken_tables, keys)
        } else {
            TableArray::from_tables(taken_tables)
        }
    }

    /// Skip the first n tables in the array
    pub fn skip(&self, n: usize) -> TableArray {
        let skipped_tables = self.tables.iter().skip(n).cloned().collect();

        let skipped_keys = if let Some(ref keys) = self.group_keys {
            Some(keys.iter().skip(n).cloned().collect())
        } else {
            None
        };

        if let Some(keys) = skipped_keys {
            TableArray::from_tables_with_keys(skipped_tables, keys)
        } else {
            TableArray::from_tables(skipped_tables)
        }
    }

    /// Join this TableArray with another TableArray
    /// Performs element-wise joins between corresponding tables
    pub fn join(&self, other: &TableArray, on: &str, join_type: &str) -> GraphResult<TableArray> {
        if self.tables.len() != other.tables.len() {
            return Err(crate::errors::GraphError::InvalidInput(
                "TableArrays must have same length for join".to_string(),
            ));
        }

        let mut joined_tables = Vec::new();
        let mut joined_keys = Vec::new();

        for (i, (left_table, right_table)) in
            self.tables.iter().zip(other.tables.iter()).enumerate()
        {
            // Perform join between corresponding tables
            let joined_table = match join_type {
                "inner" => left_table.inner_join(right_table, on, on)?,
                "left" => left_table.left_join(right_table, on, on)?,
                _ => {
                    return Err(crate::errors::GraphError::InvalidInput(format!(
                        "Unsupported join type: {}",
                        join_type
                    )))
                }
            };

            joined_tables.push(joined_table);

            // Combine group keys if they exist
            if let (Some(ref left_keys), Some(ref right_keys)) =
                (&self.group_keys, &other.group_keys)
            {
                if let (Some(left_key), Some(right_key)) = (left_keys.get(i), right_keys.get(i)) {
                    let mut combined_key = left_key.clone();
                    for (key, value) in right_key {
                        combined_key.insert(format!("right_{}", key), value.clone());
                    }
                    joined_keys.push(combined_key);
                }
            }
        }

        if !joined_keys.is_empty() {
            Ok(TableArray::from_tables_with_keys(
                joined_tables,
                joined_keys,
            ))
        } else {
            Ok(TableArray::from_tables(joined_tables))
        }
    }

    /// Apply a custom function to each table in the array
    pub fn map<F>(&self, func: F) -> GraphResult<TableArray>
    where
        F: Fn(&BaseTable) -> GraphResult<BaseTable>,
    {
        let mut mapped_tables = Vec::new();

        for table in &self.tables {
            let mapped_table = func(table)?;
            mapped_tables.push(mapped_table);
        }

        if let Some(ref keys) = self.group_keys {
            Ok(TableArray::from_tables_with_keys(
                mapped_tables,
                keys.clone(),
            ))
        } else {
            Ok(TableArray::from_tables(mapped_tables))
        }
    }

    /// Reduce all tables in the array to a single value using a binary operation
    pub fn reduce<F, T>(&self, init: T, func: F) -> GraphResult<T>
    where
        F: Fn(T, &BaseTable) -> GraphResult<T>,
    {
        let mut accumulator = init;
        for table in &self.tables {
            accumulator = func(accumulator, table)?;
        }
        Ok(accumulator)
    }

    /// Get the shape (number of tables, average rows per table)
    pub fn shape(&self) -> (usize, f64) {
        let num_tables = self.tables.len();
        if num_tables == 0 {
            return (0, 0.0);
        }

        let total_rows: usize = self.tables.iter().map(|t| t.nrows()).sum();
        let avg_rows = total_rows as f64 / num_tables as f64;

        (num_tables, avg_rows)
    }

    /// Concatenate all tables in the array into a single table
    /// This is useful for operations like .union() or flattening grouped data
    pub fn concat(&self) -> GraphResult<BaseTable> {
        if self.tables.is_empty() {
            return Ok(BaseTable::new());
        }

        let first_table = &self.tables[0];
        let mut all_columns: HashMap<String, Vec<AttrValue>> = HashMap::new();

        // Initialize columns from the first table
        for col_name in first_table.column_order() {
            all_columns.insert(col_name.clone(), Vec::new());
        }

        // Concatenate data from all tables
        for table in &self.tables {
            for col_name in first_table.column_order() {
                if let Some(column) = table.columns().get(col_name) {
                    all_columns
                        .get_mut(col_name)
                        .unwrap()
                        .extend(column.data().iter().cloned());
                } else {
                    // Fill missing columns with nulls
                    let null_count = table.nrows();
                    for _ in 0..null_count {
                        all_columns.get_mut(col_name).unwrap().push(AttrValue::Null);
                    }
                }
            }
        }

        // Convert to BaseArray columns
        let mut base_columns = HashMap::new();
        for (col_name, values) in all_columns {
            base_columns.insert(
                col_name,
                crate::storage::array::BaseArray::from_attr_values(values),
            );
        }

        let result = BaseTable::from_columns(base_columns)?;
        // column_order will be set using with_column_order

        Ok(result)
    }
}

impl Default for TableArray {
    fn default() -> Self {
        Self::new()
    }
}
