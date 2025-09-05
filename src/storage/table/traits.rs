//! Core table traits for the unified table system

use crate::storage::array::BaseArray;
use crate::errors::GraphResult;

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
    
    // =================================================================
    // Column access
    // =================================================================
    
    /// Get a column by name
    fn column(&self, name: &str) -> Option<&BaseArray>;
    
    /// Get a column by index
    fn column_by_index(&self, index: usize) -> Option<&BaseArray>;
    
    /// Check if a column exists
    fn has_column(&self, name: &str) -> bool;
    
    // =================================================================
    // Data access patterns
    // =================================================================
    
    /// Get the first n rows
    fn head(&self, n: usize) -> Self where Self: Sized;
    
    /// Get the last n rows  
    fn tail(&self, n: usize) -> Self where Self: Sized;
    
    /// Get a slice of rows [start, end)
    fn slice(&self, start: usize, end: usize) -> Self where Self: Sized;
    
    // =================================================================
    // DataFrame-like operations
    // =================================================================
    
    /// Sort the table by a column
    fn sort_by(&self, column: &str, ascending: bool) -> GraphResult<Self> where Self: Sized;
    
    /// Filter rows using a query expression
    fn filter(&self, predicate: &str) -> GraphResult<Self> where Self: Sized;
    
    /// Group by columns and return grouped tables
    fn group_by(&self, columns: &[String]) -> GraphResult<Vec<Self>> where Self: Sized;
    
    /// Select specific columns to create a new table
    fn select(&self, column_names: &[String]) -> GraphResult<Self> where Self: Sized;
    
    /// Add a new column to the table
    fn with_column(&self, name: String, column: BaseArray) -> GraphResult<Self> where Self: Sized;
    
    /// Drop columns from the table
    fn drop_columns(&self, column_names: &[String]) -> GraphResult<Self> where Self: Sized;
    
    // =================================================================
    // Chaining support - the key integration point!
    // =================================================================
    
    /// Enable fluent chaining with .iter() method
    /// Returns a TableIterator that supports method chaining
    fn iter(&self) -> TableIterator<Self> where Self: Sized + Clone;
}

/// Universal iterator for table operations with chaining support
/// This mirrors the ArrayIterator pattern for tables
#[derive(Clone)]
pub struct TableIterator<T: Table> {
    /// The table being iterated over
    table: T,
    /// Optional transformation operations to apply
    operations: Vec<TableOperation>,
}

/// Operations that can be chained on tables
#[derive(Clone, Debug)]
pub enum TableOperation {
    Head(usize),
    Tail(usize),
    Slice(usize, usize),
    SortBy(String, bool),
    Filter(String),
    Select(Vec<String>),
    DropColumns(Vec<String>),
}

impl<T: Table + Clone> TableIterator<T> {
    /// Create a new TableIterator
    pub fn new(table: T) -> Self {
        Self {
            table,
            operations: Vec::new(),
        }
    }
    
    /// Add a head operation to the chain
    pub fn head(mut self, n: usize) -> Self {
        self.operations.push(TableOperation::Head(n));
        self
    }
    
    /// Add a tail operation to the chain
    pub fn tail(mut self, n: usize) -> Self {
        self.operations.push(TableOperation::Tail(n));
        self
    }
    
    /// Add a slice operation to the chain
    pub fn slice(mut self, start: usize, end: usize) -> Self {
        self.operations.push(TableOperation::Slice(start, end));
        self
    }
    
    /// Add a sort operation to the chain
    pub fn sort_by(mut self, column: &str, ascending: bool) -> Self {
        self.operations.push(TableOperation::SortBy(column.to_string(), ascending));
        self
    }
    
    /// Add a filter operation to the chain
    pub fn filter(mut self, predicate: &str) -> Self {
        self.operations.push(TableOperation::Filter(predicate.to_string()));
        self
    }
    
    /// Add a select operation to the chain
    pub fn select(mut self, column_names: &[String]) -> Self {
        self.operations.push(TableOperation::Select(column_names.to_vec()));
        self
    }
    
    /// Add a drop columns operation to the chain
    pub fn drop_columns(mut self, column_names: &[String]) -> Self {
        self.operations.push(TableOperation::DropColumns(column_names.to_vec()));
        self
    }
    
    /// Execute all chained operations and return the result
    pub fn collect(self) -> GraphResult<T> {
        let mut result = self.table;
        
        for operation in self.operations {
            result = match operation {
                TableOperation::Head(n) => result.head(n),
                TableOperation::Tail(n) => result.tail(n),
                TableOperation::Slice(start, end) => result.slice(start, end),
                TableOperation::SortBy(column, ascending) => result.sort_by(&column, ascending)?,
                TableOperation::Filter(predicate) => result.filter(&predicate)?,
                TableOperation::Select(columns) => result.select(&columns)?,
                TableOperation::DropColumns(columns) => result.drop_columns(&columns)?,
            };
        }
        
        Ok(result)
    }
    
    /// Get the current table without executing operations (for inspection)
    pub fn current_table(&self) -> &T {
        &self.table
    }
}