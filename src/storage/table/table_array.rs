//! TableArray - Core implementation for arrays of tables
//!
//! This enables powerful groupby operations where groupby() returns a TableArray
//! and operations like .sum(), .agg() work on the entire array of tables.

use super::{traits::Table, BaseTable};
use crate::errors::GraphResult;
use crate::types::AttrValue;
use std::collections::HashMap;

#[path = "collection_context.rs"]
mod collection_context;
#[path = "collection_reductions.rs"]
mod collection_reductions;
use collection_context::CollectionContext;
use std::sync::Arc;

/// Core TableArray - an array of BaseTable instances
///
/// This is the foundation for advanced table operations like groupby().sum()
/// where operations are applied across multiple tables and aggregated.
/// Construct with new/from_tables/from_tables_with_keys; private context retains
/// empty-result schema and immutable source ownership across supported transforms.
#[derive(Debug, Clone)]
pub struct TableArray {
    context: Option<Arc<CollectionContext>>,
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
            context: None,
        }
    }

    /// Create a TableArray from a vector of tables
    pub fn from_tables(tables: Vec<BaseTable>) -> Self {
        Self {
            tables,
            group_keys: None,
            context: None,
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
            context: None,
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
        let mut tables = Vec::new();
        let mut selected_keys = self.group_keys.as_ref().map(|_| Vec::new());
        for (i, table) in self.tables.iter().enumerate() {
            if predicate(table) {
                tables.push(table.clone());
                if let (Some(keys), Some(selected)) = (&self.group_keys, &mut selected_keys) {
                    if let Some(key) = keys.get(i) {
                        selected.push(key.clone());
                    }
                }
            }
        }
        self.inherit(tables, selected_keys)
    }

    /// Take the first n tables from the array
    pub fn take(&self, n: usize) -> TableArray {
        let taken_tables = self.tables.iter().take(n).cloned().collect();

        let taken_keys = self
            .group_keys
            .as_ref()
            .map(|keys| keys.iter().take(n).cloned().collect());

        self.inherit(taken_tables, taken_keys)
    }

    /// Skip the first n tables in the array
    pub fn skip(&self, n: usize) -> TableArray {
        let skipped_tables = self.tables.iter().skip(n).cloned().collect();

        let skipped_keys = self
            .group_keys
            .as_ref()
            .map(|keys| keys.iter().skip(n).cloned().collect());

        self.inherit(skipped_tables, skipped_keys)
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

        self.key_names()?;
        other.key_names()?;
        let joined_tables = self
            .tables
            .iter()
            .zip(&other.tables)
            .map(|(left, right)| left.join_on(right, &[on], &[on], join_type))
            .collect::<GraphResult<Vec<_>>>()?;
        self.joined_collection(other, on, join_type, joined_tables)
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

        Ok(self.inherit(mapped_tables, self.group_keys.clone()))
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

        self.inherit(head_tables, self.group_keys.clone())
    }

    /// Get last n rows from each table in the array
    /// Returns a new TableArray with tail(n) applied to each table
    pub fn tail(&self, n: usize) -> TableArray {
        let tail_tables: Vec<BaseTable> = self.tables.iter().map(|table| table.tail(n)).collect();

        self.inherit(tail_tables, self.group_keys.clone())
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

        Ok(self.inherit(sampled_tables, self.group_keys.clone()))
    }

    /// Select specific columns from each table in the array
    /// Returns a new TableArray with select(columns) applied to each table
    pub fn select(&self, columns: &[String]) -> GraphResult<TableArray> {
        self.select_schema(columns)
    }

    /// Sort each table in the array by a column
    /// Returns a new TableArray with sort_by applied to each table
    pub fn sort_by(&self, column: &str, ascending: bool) -> GraphResult<TableArray> {
        self.empty_schema().sort_by(column, ascending)?;
        self.map(|table| table.sort_by(column, ascending))
    }

    /// Drop columns from each table in the array
    /// Returns a new TableArray with drop_columns applied to each table
    pub fn drop_columns(&self, columns: &[String]) -> GraphResult<TableArray> {
        self.drop_schema(columns)
    }

    /// Rename columns in each table in the array
    /// Returns a new TableArray with rename applied to each table
    pub fn rename(&self, mapping: &HashMap<String, String>) -> GraphResult<TableArray> {
        self.rename_schema(mapping)
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

        Ok(self.inherit(applied_tables, self.group_keys.clone()))
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

        Ok(self.inherit(described_tables, self.group_keys.clone()))
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
