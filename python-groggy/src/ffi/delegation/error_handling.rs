//! Error handling for the delegation system
//!
//! Provides robust error propagation and handling for delegation operations,
//! including try_map and Result propagation patterns.

use pyo3::{PyErr, PyResult};
use std::fmt;

/// Result type for delegation operations
pub type DelegationResult<T> = Result<T, DelegationError>;

/// Errors that can occur during delegation operations
#[derive(Debug)]
pub enum DelegationError {
    /// Error during method forwarding
    ForwardingError(String),
    /// Error during type conversion
    ConversionError(String),
    /// Error during array operation
    ArrayOperationError(String),
    /// Error during iterator operation
    IteratorError(String),
    /// Python-specific error
    PythonError(PyErr),
    /// Unsupported operation error
    UnsupportedOperation(String),
}

impl fmt::Display for DelegationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DelegationError::ForwardingError(msg) => write!(f, "Forwarding error: {}", msg),
            DelegationError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
            DelegationError::ArrayOperationError(msg) => {
                write!(f, "Array operation error: {}", msg)
            }
            DelegationError::IteratorError(msg) => write!(f, "Iterator error: {}", msg),
            DelegationError::PythonError(err) => write!(f, "Python error: {}", err),
            DelegationError::UnsupportedOperation(msg) => {
                write!(f, "Unsupported operation: {}", msg)
            }
        }
    }
}

impl std::error::Error for DelegationError {}

impl From<PyErr> for DelegationError {
    fn from(err: PyErr) -> Self {
        DelegationError::PythonError(err)
    }
}

impl From<&str> for DelegationError {
    fn from(msg: &str) -> Self {
        DelegationError::ForwardingError(msg.to_string())
    }
}

impl From<String> for DelegationError {
    fn from(msg: String) -> Self {
        DelegationError::ForwardingError(msg)
    }
}

impl From<DelegationError> for PyErr {
    fn from(err: DelegationError) -> Self {
        match err {
            DelegationError::PythonError(py_err) => py_err,
            _ => pyo3::exceptions::PyRuntimeError::new_err(err.to_string()),
        }
    }
}

/// Trait for try-mapping operations that can fail
pub trait TryMapOps<T> {
    type Output;

    /// Try to map over elements, collecting successes and handling errors gracefully
    fn try_map<U, F, E>(&self, f: F) -> DelegationResult<Vec<U>>
    where
        F: Fn(&T) -> Result<U, E>,
        E: Into<DelegationError>;

    /// Try to map over elements, stopping at first error
    fn try_map_fail_fast<U, F, E>(&self, f: F) -> DelegationResult<Vec<U>>
    where
        F: Fn(&T) -> Result<U, E>,
        E: Into<DelegationError>;

    /// Try to map over elements, collecting both successes and errors
    fn try_map_collect_errors<U, F, E>(&self, f: F) -> (Vec<U>, Vec<DelegationError>)
    where
        F: Fn(&T) -> Result<U, E>,
        E: Into<DelegationError>;
}

impl<T> TryMapOps<T> for Vec<T> {
    type Output = Vec<T>;

    fn try_map<U, F, E>(&self, f: F) -> DelegationResult<Vec<U>>
    where
        F: Fn(&T) -> Result<U, E>,
        E: Into<DelegationError>,
    {
        let mut results = Vec::new();
        for item in self.iter() {
            match f(item) {
                Ok(result) => results.push(result),
                Err(e) => return Err(e.into()),
            }
        }
        Ok(results)
    }

    fn try_map_fail_fast<U, F, E>(&self, f: F) -> DelegationResult<Vec<U>>
    where
        F: Fn(&T) -> Result<U, E>,
        E: Into<DelegationError>,
    {
        self.try_map(f) // Same as try_map for Vec
    }

    fn try_map_collect_errors<U, F, E>(&self, f: F) -> (Vec<U>, Vec<DelegationError>)
    where
        F: Fn(&T) -> Result<U, E>,
        E: Into<DelegationError>,
    {
        let mut successes = Vec::new();
        let mut errors = Vec::new();

        for item in self.iter() {
            match f(item) {
                Ok(result) => successes.push(result),
                Err(e) => errors.push(e.into()),
            }
        }

        (successes, errors)
    }
}

/// Extension trait for PyResult to provide delegation-specific error handling
pub trait DelegationResultExt<T> {
    /// Convert PyResult to DelegationResult
    fn into_delegation_result(self) -> DelegationResult<T>;

    /// Map error with additional context
    fn map_delegation_error<F>(self, f: F) -> DelegationResult<T>
    where
        F: FnOnce(PyErr) -> DelegationError;
}

impl<T> DelegationResultExt<T> for PyResult<T> {
    fn into_delegation_result(self) -> DelegationResult<T> {
        self.map_err(DelegationError::from)
    }

    fn map_delegation_error<F>(self, f: F) -> DelegationResult<T>
    where
        F: FnOnce(PyErr) -> DelegationError,
    {
        self.map_err(f)
    }
}

/// Helper trait for safe delegation operations
pub trait SafeDelegation<T> {
    /// Safely delegate operation with error recovery
    fn safe_delegate<U, F, R>(&self, operation: F, recovery: R) -> DelegationResult<U>
    where
        F: Fn(&T) -> DelegationResult<U>,
        R: Fn(&DelegationError) -> Option<U>;

    /// Chain delegation operations with error propagation
    fn chain_delegate<U, V, F1, F2>(&self, op1: F1, op2: F2) -> DelegationResult<V>
    where
        F1: Fn(&T) -> DelegationResult<U>,
        F2: Fn(U) -> DelegationResult<V>;
}

impl<T> SafeDelegation<T> for T {
    fn safe_delegate<U, F, R>(&self, operation: F, recovery: R) -> DelegationResult<U>
    where
        F: Fn(&T) -> DelegationResult<U>,
        R: Fn(&DelegationError) -> Option<U>,
    {
        match operation(self) {
            Ok(result) => Ok(result),
            Err(error) => match recovery(&error) {
                Some(recovered) => Ok(recovered),
                None => Err(error),
            },
        }
    }

    fn chain_delegate<U, V, F1, F2>(&self, op1: F1, op2: F2) -> DelegationResult<V>
    where
        F1: Fn(&T) -> DelegationResult<U>,
        F2: Fn(U) -> DelegationResult<V>,
    {
        let intermediate = op1(self)?;
        op2(intermediate)
    }
}

/// Macro for creating delegation error with context
#[macro_export]
macro_rules! delegation_error {
    ($variant:ident, $msg:expr) => {
        DelegationError::$variant($msg.to_string())
    };
    ($variant:ident, $fmt:expr, $($arg:tt)*) => {
        DelegationError::$variant(format!($fmt, $($arg)*))
    };
}

/// Macro for try-mapping over collections with error handling
#[macro_export]
macro_rules! try_map_delegate {
    ($collection:expr, |$item:ident| $operation:expr) => {
        $collection.try_map(|$item| $operation)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_map_success() {
        let numbers = vec![1, 2, 3, 4, 5];
        let result = numbers.try_map(|x| Ok::<i32, String>(x * 2));

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_try_map_error() {
        let numbers = vec![1, 2, 0, 4, 5]; // 0 will cause division error
        let result = numbers.try_map(|x| {
            if *x == 0 {
                Err("Division by zero")
            } else {
                Ok(10 / x)
            }
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_try_map_collect_errors() {
        let numbers = vec![1, 2, 0, 4, 0];
        let (successes, errors) = numbers.try_map_collect_errors(|x| {
            if *x == 0 {
                Err("Division by zero")
            } else {
                Ok(10 / x)
            }
        });

        assert_eq!(successes, vec![10, 5, 2]);
        assert_eq!(errors.len(), 2);
    }
}
