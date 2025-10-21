//! Experimental delegation system for prototype trait methods.
//!
//! This module provides infrastructure for rapid prototyping of new graph operations
//! behind the `experimental-delegation` feature flag. Methods can be developed and tested
//! without committing to the stable API surface.
//!
//! # Workflow
//! 1. Add new trait method to `src/traits/` with implementation
//! 2. Register it in the experimental registry (guarded by feature flag)
//! 3. Call via `graph.experimental("method_name", *args, **kwargs)` in Python
//! 4. Once stable, add explicit wrapper in PyGraph/PySubgraph
//! 5. Remove from experimental registry
//!
//! # Feature Flag
//! Enable with:
//! - Rust: `cargo build --features experimental-delegation`
//! - Python: `GROGGY_EXPERIMENTAL=1 python script.py`
//! - Maturin: `maturin develop --features experimental-delegation`

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;

/// Registry of experimental methods available for PyGraph.
///
/// When the `experimental-delegation` feature is enabled, this registry
/// populates with prototype methods that can be called dynamically via
/// `graph.experimental("method_name", *args, **kwargs)`.
pub struct ExperimentalRegistry {
    methods: HashMap<String, ExperimentalMethod>,
}

/// Metadata for an experimental method.
#[allow(dead_code)]
pub struct ExperimentalMethod {
    pub name: String,
    pub description: String,
    pub handler: fn(PyObject, Python, &PyTuple, Option<&PyDict>) -> PyResult<PyObject>,
}

impl ExperimentalRegistry {
    /// Create new experimental registry.
    ///
    /// When `experimental-delegation` feature is disabled, returns empty registry.
    /// When enabled, populates with available experimental methods.
    pub fn new() -> Self {
        let methods = HashMap::new();

        #[cfg(feature = "experimental-delegation")]
        {
            // Example: PageRank algorithm (not yet in stable API)
            methods.insert(
                "pagerank".to_string(),
                ExperimentalMethod {
                    name: "pagerank".to_string(),
                    description: "Calculate PageRank scores (experimental)".to_string(),
                    handler: experimental_pagerank,
                },
            );

            // Example: Community detection (not yet in stable API)
            methods.insert(
                "detect_communities".to_string(),
                ExperimentalMethod {
                    name: "detect_communities".to_string(),
                    description: "Detect communities using Louvain method (experimental)"
                        .to_string(),
                    handler: experimental_detect_communities,
                },
            );
        }

        Self { methods }
    }

    /// Check if a method is registered.
    #[allow(dead_code)]
    pub fn has_method(&self, name: &str) -> bool {
        self.methods.contains_key(name)
    }

    /// Get list of all registered experimental methods.
    pub fn list_methods(&self) -> Vec<&str> {
        self.methods.keys().map(|s| s.as_str()).collect()
    }

    /// Get method description.
    pub fn describe(&self, name: &str) -> Option<&str> {
        self.methods.get(name).map(|m| m.description.as_str())
    }

    /// Call an experimental method.
    #[allow(dead_code)]
    pub fn call(
        &self,
        name: &str,
        obj: PyObject,
        py: Python,
        args: &PyTuple,
        kwargs: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        match self.methods.get(name) {
            Some(method) => (method.handler)(obj, py, args, kwargs),
            None => Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "Experimental method '{}' not found. Available: {:?}",
                name,
                self.list_methods()
            ))),
        }
    }
}

/// Global experimental registry (lazy-initialized with thread-safe access).
use std::sync::OnceLock;
static REGISTRY: OnceLock<ExperimentalRegistry> = OnceLock::new();

/// Get the global experimental registry.
pub fn get_registry() -> &'static ExperimentalRegistry {
    REGISTRY.get_or_init(ExperimentalRegistry::new)
}

// =============================================================================
// EXPERIMENTAL METHOD IMPLEMENTATIONS
// =============================================================================

#[cfg(feature = "experimental-delegation")]
fn experimental_pagerank(
    _obj: PyObject,
    py: Python,
    args: &PyTuple,
    kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    use pyo3::exceptions::PyNotImplementedError;

    // Extract arguments
    let damping: f64 = if args.len() > 0 {
        args.get_item(0)?.extract()?
    } else if let Some(kw) = kwargs {
        kw.get_item("damping")
            .and_then(|item| item.map(|v| v.extract().ok()).flatten())
            .unwrap_or(0.85)
    } else {
        0.85
    };

    // TODO: Implement actual PageRank algorithm
    // For now, return NotImplementedError with a helpful message
    Err(PyNotImplementedError::new_err(format!(
        "PageRank algorithm not yet implemented. Damping factor would be: {}. \
        This is an experimental prototype - contribute an implementation in src/algorithms/centrality/!",
        damping
    )))
}

#[cfg(feature = "experimental-delegation")]
fn experimental_detect_communities(
    _obj: PyObject,
    py: Python,
    _args: &PyTuple,
    _kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    use pyo3::exceptions::PyNotImplementedError;

    // TODO: Implement community detection (Louvain, etc.)
    Err(PyNotImplementedError::new_err(
        "Community detection not yet implemented. This is an experimental prototype - \
        contribute an implementation in src/algorithms/community/!",
    ))
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_initialization() {
        let registry = ExperimentalRegistry::new();

        #[cfg(feature = "experimental-delegation")]
        {
            assert!(registry.has_method("pagerank"));
            assert!(registry.has_method("detect_communities"));
            assert_eq!(registry.list_methods().len(), 2);
        }

        #[cfg(not(feature = "experimental-delegation"))]
        {
            assert!(!registry.has_method("pagerank"));
            assert_eq!(registry.list_methods().len(), 0);
        }
    }

    #[test]
    fn test_method_descriptions() {
        let registry = ExperimentalRegistry::new();

        #[cfg(feature = "experimental-delegation")]
        {
            let desc = registry.describe("pagerank");
            assert!(desc.is_some());
            assert!(desc.unwrap().contains("PageRank"));
        }

        #[cfg(not(feature = "experimental-delegation"))]
        {
            let desc = registry.describe("pagerank");
            assert!(desc.is_none());
        }
    }
}
