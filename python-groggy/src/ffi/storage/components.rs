//! Components Array FFI Bindings
//!
//! Lazy array of connected components - only materializes PySubgraphs on access

use groggy::api::graph::Graph;
use groggy::subgraphs::Subgraph;
use groggy::{EdgeId, NodeId};
use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::ffi::subgraphs::subgraph::PySubgraph;

/// Lazy array of connected components - avoids creating hundreds of PySubgraphs immediately
#[pyclass(name = "ComponentsArray", unsendable)]
pub struct PyComponentsArray {
    // Store only the lightweight metadata, not materialized subgraphs!
    components_data: Vec<(HashSet<NodeId>, HashSet<EdgeId>)>,
    graph_ref: Rc<RefCell<Graph>>,

    // Lazy cache for materialized components (only created when accessed)
    materialized_cache: RefCell<HashMap<usize, PySubgraph>>,
}

impl PyComponentsArray {
    /// Get the number of components (public accessor for internal use)
    pub fn len(&self) -> usize {
        self.components_data.len()
    }

    /// Create new ComponentsArray from trait objects (internal use)
    pub fn from_components(
        components: Vec<Box<dyn groggy::traits::SubgraphOperations>>,
        graph_ref: Rc<RefCell<Graph>>,
    ) -> Self {
        // Extract lightweight data from trait objects
        let components_data: Vec<(HashSet<NodeId>, HashSet<EdgeId>)> = components
            .into_iter()
            .map(|comp| (comp.node_set().clone(), comp.edge_set().clone()))
            .collect();

        Self {
            components_data,
            graph_ref,
            materialized_cache: RefCell::new(HashMap::new()),
        }
    }
}

#[pymethods]
impl PyComponentsArray {
    /// Get the number of components
    fn __len__(&self) -> usize {
        self.components_data.len()
    }

    /// Lazy access to individual components - only materializes on demand
    fn __getitem__(&self, index: isize) -> PyResult<PySubgraph> {
        let len = self.components_data.len() as isize;

        // Handle negative indexing
        let actual_index = if index < 0 {
            (len + index) as usize
        } else {
            index as usize
        };

        if actual_index >= self.components_data.len() {
            return Err(PyIndexError::new_err(format!(
                "Component index {} out of range (0-{})",
                index,
                self.components_data.len() - 1
            )));
        }

        // Check cache first
        if let Some(cached) = self.materialized_cache.borrow().get(&actual_index) {
            return Ok(cached.clone());
        }

        // Lazy materialization - create PySubgraph on-demand
        let (nodes, edges) = &self.components_data[actual_index];
        let subgraph = Subgraph::new(
            self.graph_ref.clone(),
            nodes.clone(),
            edges.clone(),
            format!("component_{}", actual_index),
        );

        let py_subgraph = PySubgraph::from_core_subgraph(subgraph).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create component subgraph: {}", e))
        })?;

        // Cache for next time
        self.materialized_cache
            .borrow_mut()
            .insert(actual_index, py_subgraph.clone());

        Ok(py_subgraph)
    }

    /// Iteration support
    fn __iter__(slf: PyRef<Self>) -> PyResult<PyComponentsArrayIterator> {
        Ok(PyComponentsArrayIterator {
            components: slf.into(),
            index: 0,
        })
    }

    /// String representation showing count
    fn __repr__(&self) -> String {
        format!("ComponentsArray({} components)", self.components_data.len())
    }

    /// Convert to regular Python list (materializes all components)
    fn to_list(&self) -> PyResult<Vec<PySubgraph>> {
        let mut result = Vec::new();
        for i in 0..self.components_data.len() {
            result.push(self.__getitem__(i as isize)?);
        }
        Ok(result)
    }

    /// Get component sizes without materializing subgraphs
    fn sizes(&self) -> Vec<(usize, usize)> {
        self.components_data
            .iter()
            .map(|(nodes, edges)| (nodes.len(), edges.len()))
            .collect()
    }

    /// Get largest component without materializing others
    fn largest_component(&self) -> PyResult<PySubgraph> {
        if self.components_data.is_empty() {
            return Err(PyRuntimeError::new_err("No components available"));
        }

        // Find largest by node count
        let largest_index = self
            .components_data
            .iter()
            .enumerate()
            .max_by_key(|(_, (nodes, _))| nodes.len())
            .map(|(i, _)| i)
            .unwrap();

        self.__getitem__(largest_index as isize)
    }
}

/// Iterator for ComponentsArray
#[pyclass]
pub struct PyComponentsArrayIterator {
    components: Py<PyComponentsArray>,
    index: usize,
}

#[pymethods]
impl PyComponentsArrayIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PySubgraph>> {
        let components = self.components.borrow(py);
        if self.index < components.components_data.len() {
            let result = components.__getitem__(self.index as isize)?;
            self.index += 1;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
}
