//! TableArray - Core implementation for arrays of tables
//!
//! This enables powerful groupby operations where groupby() returns a TableArray
//! and operations like .sum(), .agg() work on the entire array of tables.

use super::{traits::Table, BaseTable};
use crate::errors::GraphResult;
use crate::types::AttrValue;
use std::collections::HashMap;

#[path = "collection_reductions.rs"]
mod collection_reductions;

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
        if !matches!(join_type, "inner" | "left" | "right" | "outer" | "cross") {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Unsupported join type: {join_type}"
            )));
        }
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
                "right" => left_table.right_join(right_table, on, on)?,
                "outer" => left_table.outer_join(right_table, on, on)?,
                "cross" => left_table.cross_join(right_table)?,
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
        let values = self.column_values(column);
        if values.clone().next().is_none() {
            return Ok(AttrValue::Int(0));
        }
        crate::storage::array::reductions::sum_iter(values)
    }

    fn column_values<'a>(&'a self, column: &'a str) -> impl Iterator<Item = &'a AttrValue> + Clone {
        self.tables
            .iter()
            .filter_map(move |table| table.columns().get(column))
            .flat_map(|column| column.data().iter())
    }

    /// Mean across selected rows, with exact integer accumulation and no flattening copy.
    pub fn mean_column(&self, column: &str) -> GraphResult<f64> {
        crate::storage::array::reductions::mean_iter(self.column_values(column))
    }

    /// Minimum value in a column across all tables
    /// Returns the minimum value of the specified column across all tables
    pub fn min_column(&self, column: &str) -> GraphResult<AttrValue> {
        crate::storage::array::reductions::numeric_extremum_iter(self.column_values(column), true)
    }

    /// Maximum value in a column across all tables
    /// Returns the maximum value of the specified column across all tables
    pub fn max_column(&self, column: &str) -> GraphResult<AttrValue> {
        crate::storage::array::reductions::numeric_extremum_iter(self.column_values(column), false)
    }

    /// Standard deviation of values in a column across all tables
    /// Returns the standard deviation of the specified column across all tables
    pub fn std_column(&self, column: &str) -> GraphResult<f64> {
        crate::storage::array::reductions::std_iter(self.column_values(column))
    }
}

impl Default for TableArray {
    fn default() -> Self {
        Self::new()
    }
}
