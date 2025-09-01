//! Query Parser FFI Bindings
//!
//! Python bindings for the core Rust query parser. This eliminates the
//! circular dependency by providing direct access to Rust parsing functionality.

use crate::ffi::core::query::{PyEdgeFilter, PyNodeFilter};
use groggy::core::query_parser::QueryParser;
use pyo3::prelude::*;

/// Python wrapper for the core Rust QueryParser
#[pyclass(name = "QueryParser")]
pub struct PyQueryParser {
    inner: QueryParser,
}

#[pymethods]
impl PyQueryParser {
    /// Create a new query parser instance
    #[new]
    pub fn new() -> Self {
        Self {
            inner: QueryParser::new(),
        }
    }

    /// Parse a node query string into a NodeFilter
    pub fn parse_node_query(&mut self, query: &str) -> PyResult<PyNodeFilter> {
        let filter = self.inner.parse_node_query(query).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
        })?;
        Ok(PyNodeFilter { inner: filter })
    }

    /// Parse an edge query string into an EdgeFilter  
    pub fn parse_edge_query(&mut self, query: &str) -> PyResult<PyEdgeFilter> {
        let filter = self.inner.parse_edge_query(query).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
        })?;
        Ok(PyEdgeFilter { inner: filter })
    }

    /// Parse a node query and return the internal representation (for debugging)
    pub fn parse_node_query_debug(&mut self, query: &str) -> PyResult<String> {
        match self.inner.parse_node_query(query) {
            Ok(filter) => Ok(format!("{:#?}", filter)),
            Err(e) => Ok(format!("Error: {}", e)),
        }
    }

    /// Parse an edge query and return the internal representation (for debugging)
    pub fn parse_edge_query_debug(&mut self, query: &str) -> PyResult<String> {
        match self.inner.parse_edge_query(query) {
            Ok(filter) => Ok(format!("{:#?}", filter)),
            Err(e) => Ok(format!("Error: {}", e)),
        }
    }

    /// Check if a query string is syntactically valid for nodes
    pub fn validate_node_query(&mut self, query: &str) -> bool {
        self.inner.parse_node_query(query).is_ok()
    }

    /// Check if a query string is syntactically valid for edges
    pub fn validate_edge_query(&mut self, query: &str) -> bool {
        self.inner.parse_edge_query(query).is_ok()
    }

    /// Get detailed error information for a failing node query
    pub fn get_node_query_error(&mut self, query: &str) -> Option<String> {
        match self.inner.parse_node_query(query) {
            Ok(_) => None,
            Err(e) => Some(e.to_string()),
        }
    }

    /// Get detailed error information for a failing edge query
    pub fn get_edge_query_error(&mut self, query: &str) -> Option<String> {
        match self.inner.parse_edge_query(query) {
            Ok(_) => None,
            Err(e) => Some(e.to_string()),
        }
    }

    /// Reset the parser state (useful for reusing parser instances)
    pub fn reset(&mut self) {
        self.inner = QueryParser::new();
    }

    /// Get parser information/version for debugging
    pub fn info(&self) -> String {
        format!("Groggy Rust Core Query Parser v{}", groggy::VERSION)
    }

    /// String representation
    pub fn __str__(&self) -> String {
        "QueryParser(Rust Core)".to_string()
    }

    /// Debug representation
    pub fn __repr__(&self) -> String {
        format!("PyQueryParser(version={})", groggy::VERSION)
    }
}

impl Default for PyQueryParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for parsing node queries (stateless)
#[pyfunction]
pub fn parse_node_query(query: &str) -> PyResult<PyNodeFilter> {
    let mut parser = QueryParser::new();
    let filter = parser.parse_node_query(query).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
    })?;
    Ok(PyNodeFilter { inner: filter })
}

/// Convenience function for parsing edge queries (stateless)
#[pyfunction]
pub fn parse_edge_query(query: &str) -> PyResult<PyEdgeFilter> {
    let mut parser = QueryParser::new();
    let filter = parser.parse_edge_query(query).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
    })?;
    Ok(PyEdgeFilter { inner: filter })
}

// Note: Removed duplicate validation and error functions as they were unused
// The PyQueryParser class methods provide the same functionality if needed

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_query_parser() {
        let mut parser = PyQueryParser::new();

        // Test node query parsing
        let result = parser.parse_node_query("salary > 120000");
        assert!(result.is_ok());

        let result = parser.parse_node_query("department == 'Engineering'");
        assert!(result.is_ok());

        // Test complex query
        let result = parser
            .parse_node_query("(salary > 120000 AND department == 'Engineering') OR age < 25");
        assert!(result.is_ok());

        // Test error case
        let result = parser.parse_node_query("salary >");
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_methods() {
        let mut parser = PyQueryParser::new();
        assert!(parser.validate_node_query("salary > 100000"));
        assert!(parser.validate_node_query("name == 'test'"));
        assert!(!parser.validate_node_query("invalid >>"));
        assert!(!parser.validate_node_query(""));
    }

    #[test]
    fn test_error_reporting_methods() {
        let mut parser = PyQueryParser::new();
        let error = parser.get_node_query_error("salary >");
        assert!(error.is_some());
        assert!(error.unwrap().contains("Expected"));

        let error = parser.get_node_query_error("salary > 100000");
        assert!(error.is_none());
    }

    #[test]
    fn test_edge_queries() {
        let mut parser = PyQueryParser::new();

        let result = parser.parse_edge_query("weight > 0.5");
        assert!(result.is_ok());

        let result = parser.parse_edge_query("type == 'friendship'");
        assert!(result.is_ok());
    }

    #[test]
    fn test_debug_methods() {
        let mut parser = PyQueryParser::new();

        let debug = parser.parse_node_query_debug("salary > 100000");
        assert!(debug.is_ok());
        assert!(debug.unwrap().contains("AttributeFilter"));

        let debug = parser.parse_node_query_debug("invalid >>");
        assert!(debug.is_ok());
        assert!(debug.unwrap().contains("Error:"));
    }
}
