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

        let _first_table = &self.tables[0];
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

        let taken_keys = self
            .group_keys
            .as_ref()
            .map(|keys| keys.iter().take(n).cloned().collect());

        if let Some(keys) = taken_keys {
            TableArray::from_tables_with_keys(taken_tables, keys)
        } else {
            TableArray::from_tables(taken_tables)
        }
    }

    /// Skip the first n tables in the array
    pub fn skip(&self, n: usize) -> TableArray {
        let skipped_tables = self.tables.iter().skip(n).cloned().collect();

        let skipped_keys = self
            .group_keys
            .as_ref()
            .map(|keys| keys.iter().skip(n).cloned().collect());

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

    /// Get first n rows from each table in the array
    /// Returns a new TableArray with head(n) applied to each table
    pub fn head(&self, n: usize) -> TableArray {
        let head_tables: Vec<BaseTable> = self.tables.iter().map(|table| table.head(n)).collect();

        if let Some(ref keys) = self.group_keys {
            TableArray::from_tables_with_keys(head_tables, keys.clone())
        } else {
            TableArray::from_tables(head_tables)
        }
    }

    /// Get last n rows from each table in the array
    /// Returns a new TableArray with tail(n) applied to each table
    pub fn tail(&self, n: usize) -> TableArray {
        let tail_tables: Vec<BaseTable> = self.tables.iter().map(|table| table.tail(n)).collect();

        if let Some(ref keys) = self.group_keys {
            TableArray::from_tables_with_keys(tail_tables, keys.clone())
        } else {
            TableArray::from_tables(tail_tables)
        }
    }

    /// Sample n rows from each table in the array
    /// Returns a new TableArray with sample(n) applied to each table
    pub fn sample(&self, n: usize) -> GraphResult<TableArray> {
        let mut sampled_tables = Vec::new();

        for table in &self.tables {
            // Call sample with proper parameters: n, fraction=None, weights=None, subset=None, class_weights=None, replace=false
            let sampled_table = table.sample(Some(n), None, None, None, None, false)?;
            sampled_tables.push(sampled_table);
        }

        if let Some(ref keys) = self.group_keys {
            Ok(TableArray::from_tables_with_keys(
                sampled_tables,
                keys.clone(),
            ))
        } else {
            Ok(TableArray::from_tables(sampled_tables))
        }
    }

    /// Select specific columns from each table in the array
    /// Returns a new TableArray with select(columns) applied to each table
    pub fn select(&self, columns: &[String]) -> GraphResult<TableArray> {
        let mut selected_tables = Vec::new();

        for table in &self.tables {
            let selected_table = table.select(columns)?;
            selected_tables.push(selected_table);
        }

        if let Some(ref keys) = self.group_keys {
            Ok(TableArray::from_tables_with_keys(
                selected_tables,
                keys.clone(),
            ))
        } else {
            Ok(TableArray::from_tables(selected_tables))
        }
    }

    /// Sort each table in the array by a column
    /// Returns a new TableArray with sort_by applied to each table
    pub fn sort_by(&self, column: &str, ascending: bool) -> GraphResult<TableArray> {
        let mut sorted_tables = Vec::new();

        for table in &self.tables {
            let sorted_table = table.sort_by(column, ascending)?;
            sorted_tables.push(sorted_table);
        }

        if let Some(ref keys) = self.group_keys {
            Ok(TableArray::from_tables_with_keys(
                sorted_tables,
                keys.clone(),
            ))
        } else {
            Ok(TableArray::from_tables(sorted_tables))
        }
    }

    /// Drop columns from each table in the array
    /// Returns a new TableArray with drop_columns applied to each table
    pub fn drop_columns(&self, columns: &[String]) -> GraphResult<TableArray> {
        let mut tables_with_dropped_columns = Vec::new();

        for table in &self.tables {
            let table_with_dropped_columns = table.drop_columns(columns)?;
            tables_with_dropped_columns.push(table_with_dropped_columns);
        }

        if let Some(ref keys) = self.group_keys {
            Ok(TableArray::from_tables_with_keys(
                tables_with_dropped_columns,
                keys.clone(),
            ))
        } else {
            Ok(TableArray::from_tables(tables_with_dropped_columns))
        }
    }

    /// Rename columns in each table in the array
    /// Returns a new TableArray with rename applied to each table
    pub fn rename(&self, mapping: &HashMap<String, String>) -> GraphResult<TableArray> {
        let mut renamed_tables = Vec::new();

        for table in &self.tables {
            let renamed_table = table.rename(mapping.clone())?;
            renamed_tables.push(renamed_table);
        }

        if let Some(ref keys) = self.group_keys {
            Ok(TableArray::from_tables_with_keys(
                renamed_tables,
                keys.clone(),
            ))
        } else {
            Ok(TableArray::from_tables(renamed_tables))
        }
    }

    /// Apply a function to each table in the array
    /// Returns a new TableArray with apply applied to each table
    pub fn apply<F>(&self, func: F) -> GraphResult<TableArray>
    where
        F: Fn(&BaseTable) -> GraphResult<BaseTable>,
    {
        let mut applied_tables = Vec::new();

        for table in &self.tables {
            let applied_table = func(table)?;
            applied_tables.push(applied_table);
        }

        if let Some(ref keys) = self.group_keys {
            Ok(TableArray::from_tables_with_keys(
                applied_tables,
                keys.clone(),
            ))
        } else {
            Ok(TableArray::from_tables(applied_tables))
        }
    }

    /// Apply a function to each table and return a vector of results
    /// This allows the function to return any type T
    pub fn apply_to_vec<F, T>(&self, func: F) -> GraphResult<Vec<T>>
    where
        F: Fn(&BaseTable) -> GraphResult<T>,
    {
        let mut results = Vec::new();

        for table in &self.tables {
            let result = func(table)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Apply a function to each table and return an array of the results
    /// The function can return any AttrValue-compatible type
    pub fn apply_to_array<F>(
        &self,
        func: F,
    ) -> GraphResult<crate::storage::array::BaseArray<AttrValue>>
    where
        F: Fn(&BaseTable) -> GraphResult<AttrValue>,
    {
        let mut values = Vec::new();

        for table in &self.tables {
            let value = func(table)?;
            values.push(value);
        }

        Ok(crate::storage::array::BaseArray::from_attr_values(values))
    }

    /// Apply a function to each table and reduce to a single value
    /// This is useful for operations like sum of all table statistics
    pub fn apply_reduce<F, T, R>(
        &self,
        func: F,
        init: R,
        reduce: impl Fn(R, T) -> R,
    ) -> GraphResult<R>
    where
        F: Fn(&BaseTable) -> GraphResult<T>,
    {
        let mut accumulator = init;

        for table in &self.tables {
            let value = func(table)?;
            accumulator = reduce(accumulator, value);
        }

        Ok(accumulator)
    }

    /// Get total row count across all tables
    /// Returns the sum of rows across all tables in the array
    pub fn total_count(&self) -> usize {
        self.tables.iter().map(|table| table.nrows()).sum()
    }

    /// Get detailed shape information (num_tables, total_rows, num_cols)
    /// Returns a tuple with number of tables, total rows, and columns (if consistent)
    pub fn shape_detailed(&self) -> (usize, usize, Option<usize>) {
        let num_tables = self.tables.len();
        let total_rows = self.total_count();

        // Check if all tables have the same number of columns
        let num_cols = if let Some(first_table) = self.tables.first() {
            let first_ncols = first_table.ncols();
            if self.tables.iter().all(|t| t.ncols() == first_ncols) {
                Some(first_ncols)
            } else {
                None // Tables have different column counts
            }
        } else {
            Some(0) // Empty array
        };

        (num_tables, total_rows, num_cols)
    }

    /// Describe statistics for each table
    /// Returns a new TableArray with describe() applied to each table
    pub fn describe(&self) -> GraphResult<TableArray> {
        let mut described_tables = Vec::new();

        for table in &self.tables {
            let described_table = table.describe()?;
            described_tables.push(described_table);
        }

        if let Some(ref keys) = self.group_keys {
            Ok(TableArray::from_tables_with_keys(
                described_tables,
                keys.clone(),
            ))
        } else {
            Ok(TableArray::from_tables(described_tables))
        }
    }

    /// Sum values in a column across all tables
    /// Returns the total sum of the specified column across all tables
    pub fn sum_column(&self, column: &str) -> GraphResult<AttrValue> {
        let total_sum = AttrValue::Int(0);
        let mut is_float = false;
        let mut float_sum = 0.0;
        let mut int_sum: i64 = 0;

        for table in &self.tables {
            if let Some(col_array) = table.columns().get(column) {
                let table_sum = col_array.sum()?;
                match (&total_sum, &table_sum) {
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
                    _ => {
                        return Err(crate::errors::GraphError::InvalidInput(format!(
                            "Cannot sum non-numeric values in column '{}'",
                            column
                        )))
                    }
                }
            }
        }

        if is_float {
            Ok(AttrValue::Float(float_sum as f32))
        } else {
            Ok(AttrValue::Int(int_sum))
        }
    }

    /// Mean of values in a column across all tables
    /// Returns the average of the specified column across all tables
    pub fn mean_column(&self, column: &str) -> GraphResult<f64> {
        let mut total_sum = 0.0;
        let mut total_count = 0;

        for table in &self.tables {
            if let Some(col_array) = table.columns().get(column) {
                let table_mean = col_array.mean()?;
                let table_count = col_array.count();

                total_sum += table_mean * table_count as f64;
                total_count += table_count;
            }
        }

        if total_count == 0 {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "No values found in column '{}'",
                column
            )));
        }

        Ok(total_sum / total_count as f64)
    }

    /// Minimum value in a column across all tables
    /// Returns the minimum value of the specified column across all tables
    pub fn min_column(&self, column: &str) -> GraphResult<AttrValue> {
        let mut global_min: Option<AttrValue> = None;

        for table in &self.tables {
            if let Some(col_array) = table.columns().get(column) {
                let table_min = col_array.min()?;

                match &global_min {
                    None => global_min = Some(table_min),
                    Some(current_min) => {
                        if self.compare_attr_values(&table_min, current_min)? < 0 {
                            global_min = Some(table_min);
                        }
                    }
                }
            }
        }

        global_min.ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!(
                "No values found in column '{}'",
                column
            ))
        })
    }

    /// Maximum value in a column across all tables
    /// Returns the maximum value of the specified column across all tables
    pub fn max_column(&self, column: &str) -> GraphResult<AttrValue> {
        let mut global_max: Option<AttrValue> = None;

        for table in &self.tables {
            if let Some(col_array) = table.columns().get(column) {
                let table_max = col_array.max()?;

                match &global_max {
                    None => global_max = Some(table_max),
                    Some(current_max) => {
                        if self.compare_attr_values(&table_max, current_max)? > 0 {
                            global_max = Some(table_max);
                        }
                    }
                }
            }
        }

        global_max.ok_or_else(|| {
            crate::errors::GraphError::InvalidInput(format!(
                "No values found in column '{}'",
                column
            ))
        })
    }

    /// Standard deviation of values in a column across all tables
    /// Returns the standard deviation of the specified column across all tables
    pub fn std_column(&self, column: &str) -> GraphResult<f64> {
        let mean = self.mean_column(column)?;
        let mut sum_squared_diffs = 0.0;
        let mut total_count = 0;

        for table in &self.tables {
            if let Some(col_array) = table.columns().get(column) {
                for value in col_array.data().iter() {
                    match value {
                        AttrValue::Int(i) => {
                            let diff = *i as f64 - mean;
                            sum_squared_diffs += diff * diff;
                            total_count += 1;
                        }
                        AttrValue::SmallInt(i) => {
                            let diff = *i as f64 - mean;
                            sum_squared_diffs += diff * diff;
                            total_count += 1;
                        }
                        AttrValue::Float(f) => {
                            let diff = *f as f64 - mean;
                            sum_squared_diffs += diff * diff;
                            total_count += 1;
                        }
                        AttrValue::Null => {} // Skip null values
                        _ => {
                            return Err(crate::errors::GraphError::InvalidInput(format!(
                            "Cannot compute standard deviation of non-numeric data in column '{}'",
                            column
                        )))
                        }
                    }
                }
            }
        }

        if total_count <= 1 {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Cannot compute standard deviation with {} values in column '{}'",
                total_count, column
            )));
        }

        Ok((sum_squared_diffs / (total_count - 1) as f64).sqrt())
    }

    /// Helper method to compare AttrValues for min/max operations
    fn compare_attr_values(&self, a: &AttrValue, b: &AttrValue) -> GraphResult<i32> {
        use std::cmp::Ordering;

        let ordering = match (a, b) {
            (AttrValue::Int(a), AttrValue::Int(b)) => a.cmp(b),
            (AttrValue::SmallInt(a), AttrValue::SmallInt(b)) => a.cmp(b),
            (AttrValue::Float(a), AttrValue::Float(b)) => {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            }
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
            (AttrValue::SmallInt(a), AttrValue::Int(b)) => (*a as i64).cmp(b),
            (AttrValue::Int(a), AttrValue::SmallInt(b)) => a.cmp(&(*b as i64)),
            _ => {
                return Err(crate::errors::GraphError::InvalidInput(
                    "Cannot compare non-numeric values".to_string(),
                ))
            }
        };

        Ok(match ordering {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        })
    }
}

impl Default for TableArray {
    fn default() -> Self {
        Self::new()
    }
}
