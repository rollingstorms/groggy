use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use anyhow::{anyhow, Result};

use super::pipeline::AlgorithmSpec;
use super::{Algorithm, AlgorithmMetadata};

/// Factory signature used for algorithm registration.
pub type AlgorithmFactory = dyn Fn(&AlgorithmSpec) -> Result<Box<dyn Algorithm>> + Send + Sync;

struct AlgorithmEntry {
    factory: Arc<AlgorithmFactory>,
    metadata: AlgorithmMetadata,
}

/// Thread-safe registry of algorithms keyed by identifier.
#[derive(Default)]
pub struct Registry {
    entries: RwLock<HashMap<String, AlgorithmEntry>>,
}

impl Registry {
    /// Register a new algorithm factory. Returns an error if the id already exists.
    pub fn register_factory<F>(&self, id: &str, factory: F) -> Result<()>
    where
        F: Fn(&AlgorithmSpec) -> Result<Box<dyn Algorithm>> + Send + Sync + 'static,
    {
        let mut guard = self.entries.write().expect("registry poisoned");
        if guard.contains_key(id) {
            return Err(anyhow!("algorithm '{id}' is already registered"));
        }

        let metadata = AlgorithmMetadata {
            id: id.to_string(),
            name: id.to_string(),
            ..AlgorithmMetadata::default()
        };

        guard.insert(
            id.to_string(),
            AlgorithmEntry {
                factory: Arc::new(factory),
                metadata,
            },
        );
        Ok(())
    }

    /// Register with explicit metadata.
    pub fn register_with_metadata<F>(
        &self,
        id: &str,
        metadata: AlgorithmMetadata,
        factory: F,
    ) -> Result<()>
    where
        F: Fn(&AlgorithmSpec) -> Result<Box<dyn Algorithm>> + Send + Sync + 'static,
    {
        let mut guard = self.entries.write().expect("registry poisoned");
        if guard.contains_key(id) {
            return Err(anyhow!("algorithm '{id}' is already registered"));
        }

        guard.insert(
            id.to_string(),
            AlgorithmEntry {
                factory: Arc::new(factory),
                metadata,
            },
        );
        Ok(())
    }

    /// Instantiate an algorithm from a spec.
    pub fn instantiate(&self, spec: &AlgorithmSpec) -> Result<Box<dyn Algorithm>> {
        let guard = self.entries.read().expect("registry poisoned");
        let entry = guard
            .get(&spec.id)
            .ok_or_else(|| anyhow!("algorithm '{}' is not registered", spec.id))?;
        (entry.factory)(spec)
    }

    /// Whether the registry contains the given id.
    pub fn contains(&self, id: &str) -> bool {
        let guard = self.entries.read().expect("registry poisoned");
        guard.contains_key(id)
    }

    /// Retrieve metadata for a registered algorithm.
    pub fn metadata(&self, id: &str) -> Option<AlgorithmMetadata> {
        let guard = self.entries.read().expect("registry poisoned");
        guard.get(id).map(|entry| entry.metadata.clone())
    }

    /// List metadata for all registered algorithms.
    pub fn list(&self) -> Vec<AlgorithmMetadata> {
        let guard = self.entries.read().expect("registry poisoned");
        guard.values().map(|entry| entry.metadata.clone()).collect()
    }
}

static GLOBAL_REGISTRY: OnceLock<Registry> = OnceLock::new();

/// Global registry accessor used by high-level APIs.
pub fn global_registry() -> &'static Registry {
    GLOBAL_REGISTRY.get_or_init(Registry::default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::{Algorithm, AlgorithmParams, Context};
    use crate::api::graph::Graph;
    use crate::subgraphs::Subgraph;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    struct Dummy;

    impl Algorithm for Dummy {
        fn id(&self) -> &'static str {
            "dummy"
        }

        fn execute(&self, _ctx: &mut Context, subgraph: Subgraph) -> anyhow::Result<Subgraph> {
            Ok(subgraph)
        }
    }

    #[test]
    fn register_and_instantiate() {
        let registry = Registry::default();
        registry
            .register_factory("dummy", |_spec| Ok(Box::new(Dummy)))
            .unwrap();

        assert!(registry.contains("dummy"));

        let spec = AlgorithmSpec {
            id: "dummy".to_string(),
            params: AlgorithmParams::new(),
        };
        let algo = registry.instantiate(&spec).unwrap();

        let mut graph = Graph::new();
        let node = graph.add_node();
        let mut nodes = HashSet::new();
        nodes.insert(node);
        let sg =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".to_string()).unwrap();

        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, sg).unwrap();
        assert!(result.has_node(node));
    }
}
