//! Core table traits for the unified table system

use crate::errors::GraphResult;
use crate::storage::array::BaseArray;
use crate::types::AttrValue;
use std::collections::HashMap;

/// Core table operations trait - foundation for all table types
/// All tables are composed of BaseArray columns and support unified operations
pub trait Table {
    // =================================================================
    // Basic table info
    // =================================================================

    /// Get the number of rows in the table
    fn nrows(&self) -> usize;

    /// Get the number of columns in the table
    fn ncols(&self) -> usize;

    /// Get the column names
    fn column_names(&self) -> &[String];

    /// Get the shape (rows, cols) of the table
    fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// Check whether the table contains any rows
    fn is_empty(&self) -> bool {
        self.nrows() == 0
    }

    // =================================================================
    // Column access
    // =================================================================

    /// Get a column by name
    fn column(&self, name: &str) -> Option<&BaseArray<AttrValue>>;

    /// Get a column by index
    fn column_by_index(&self, index: usize) -> Option<&BaseArray<AttrValue>>;

    /// Check if a column exists
    fn has_column(&self, name: &str) -> bool;

    // =================================================================
    // Data access patterns
    // =================================================================

    /// Get the first n rows
    fn head(&self, n: usize) -> Self
    where
        Self: Sized;

    /// Get the last n rows  
    fn tail(&self, n: usize) -> Self
    where
        Self: Sized;

    /// Get a slice of rows [start, end)
    fn slice(&self, start: usize, end: usize) -> Self
    where
        Self: Sized;

    // =================================================================
    // DataFrame-like operations
    // =================================================================

    /// Sort the table by a column
    fn sort_by(&self, column: &str, ascending: bool) -> GraphResult<Self>
    where
        Self: Sized;

    /// Sort the table by multiple columns with mixed ascending/descending order
    /// Pandas-style multi-column sorting with priority order
    fn sort_values(&self, columns: Vec<String>, ascending: Vec<bool>) -> GraphResult<Self>
    where
        Self: Sized;

    /// Filter rows using a query expression
    fn filter(&self, predicate: &str) -> GraphResult<Self>
    where
        Self: Sized;

    /// Group by columns and return grouped tables
    fn group_by(&self, columns: &[String]) -> GraphResult<Vec<Self>>
    where
        Self: Sized;

    /// Select specific columns to create a new table
    fn select(&self, column_names: &[String]) -> GraphResult<Self>
    where
        Self: Sized;

    /// Add a new column to the table
    fn with_column(&self, name: String, column: BaseArray<AttrValue>) -> GraphResult<Self>
    where
        Self: Sized;

    /// Drop columns from the table
    fn drop_columns(&self, column_names: &[String]) -> GraphResult<Self>
    where
        Self: Sized;

    // =================================================================
    // Data reshaping operations
    // =================================================================

    /// Pandas-style pivot table operation for data reshaping
    ///
    /// Creates a pivot table that spreads unique values from one column (columns_col)
    /// into new columns, grouping by index columns (index_cols), and aggregating
    /// values (values_col) using the specified aggregation function.
    fn pivot_table(
        &self,
        index_cols: &[String],
        columns_col: &str,
        values_col: &str,
        agg_func: &str,
    ) -> GraphResult<Self>
    where
        Self: Sized;

    /// Pandas-style melt operation for data unpivoting
    ///
    /// Transforms wide-format data to long-format by unpivoting columns into rows.
    /// - id_vars: Column(s) to use as identifier variables (remain unchanged)
    /// - value_vars: Column(s) to unpivot (if None, uses all columns except id_vars)
    /// - var_name: Name for the new variable column (default: "variable")
    /// - value_name: Name for the new value column (default: "value")
    fn melt(
        &self,
        id_vars: Option<&[String]>,
        value_vars: Option<&[String]>,
        var_name: Option<String>,
        value_name: Option<String>,
    ) -> GraphResult<Self>
    where
        Self: Sized;

    // =================================================================
    // Chaining support - the key integration point!
    // =================================================================

    /// Iterate over rows in the table.
    /// The iterator yields `TableRow` handles that provide access to values by column name.
    fn iter(&self) -> TableRowIterator<'_, Self>
    where
        Self: Sized,
    {
        TableRowIterator::new(self)
    }
}
/// Iterator over table rows yielding lightweight row handles.
pub struct TableRowIterator<'a, T: Table> {
    table: &'a T,
    index: usize,
    len: usize,
}

impl<'a, T: Table> TableRowIterator<'a, T> {
    pub fn new(table: &'a T) -> Self {
        Self {
            table,
            index: 0,
            len: table.nrows(),
        }
    }
}

impl<'a, T: Table> Iterator for TableRowIterator<'a, T> {
    type Item = TableRow<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            return None;
        }

        let row = TableRow {
            table: self.table,
            index: self.index,
        };
        self.index += 1;
        Some(row)
    }
}

/// Lightweight accessor for a single table row.
pub struct TableRow<'a, T: Table> {
    table: &'a T,
    index: usize,
}

impl<'a, T: Table> TableRow<'a, T> {
    /// Row index within the table.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get a value for the column, if present.
    pub fn get(&self, column: &str) -> Option<&'a AttrValue> {
        self.table
            .column(column)
            .and_then(|col| col.data().get(self.index))
    }

    /// Borrow the column names for this table.
    pub fn column_names(&self) -> &'a [String] {
        self.table.column_names()
    }

    /// Clone the current row into a HashMap for ergonomic consumption.
    pub fn to_hash_map(&self) -> HashMap<String, AttrValue> {
        let mut row = HashMap::new();
        for column in self.table.column_names() {
            if let Some(value) = self.get(column) {
                row.insert(column.clone(), value.clone());
            }
        }
        row
    }
}
