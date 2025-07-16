// src_new/graph/collections/base.rs
//! Shared trait and default logic for node/edge collections in Groggy graphs.

/// Trait for shared collection interface for nodes and edges.
/// Provides a common API for batch operations, filtering, attribute access, and iteration.
/// Designed for agent/LLM friendliness and modular backend delegation.
pub trait BaseCollection {
    /// The type used for element IDs (e.g., NodeId, EdgeId).
    type IdType: Clone + Eq + std::hash::Hash;
    /// The type used for the attribute manager.
    type AttrMgr;

    /// Add one or more elements to the collection.
    /// Returns Ok(()) if all elements were added, or an error message otherwise.
    fn add(&mut self, items: Vec<Self::IdType>) -> Result<(), String>;

    /// Remove one or more elements from the collection.
    /// Returns Ok(()) if all elements were removed, or an error message otherwise.
    fn remove(&mut self, ids: &[Self::IdType]) -> Result<(), String>;

    /// Filter elements in the collection, returning matching IDs.
    /// The predicate receives a reference to each ID and returns true if it should be included.
    fn filter<F>(&self, predicate: F) -> Vec<Self::IdType>
    where
        F: Fn(&Self::IdType) -> bool;

    /// Get the number of elements in the collection.
    fn size(&self) -> usize;

    /// Get all element IDs in the collection.
    fn ids(&self) -> Vec<Self::IdType>;

    /// Check if an element exists in the collection.
    fn has(&self, id: &Self::IdType) -> bool;

    /// Get the attribute manager for this collection.
    fn attr(&self) -> &Self::AttrMgr;

    /// Get an iterator over the collection.
    fn iter(&self) -> Box<dyn Iterator<Item = Self::IdType> + '_>;

    /// Get a reference to an element by ID, if it exists.
    fn get(&self, id: &Self::IdType) -> Option<&Self::IdType>;
}


/// Default implementation for shared collection logic (can be composed or used via delegation)
/// Placeholder for future logic and composition patterns.
use std::collections::HashSet;

/// Generic, batch-optimized implementation for node/edge collections.
/// Use batch operations as the fast path; single operations wrap batch.
pub struct BaseCollectionImpl<IdType, AttrMgr>
where
    IdType: Clone + Eq + std::hash::Hash,
{
    ids: HashSet<IdType>,
    attribute_manager: AttrMgr,
}

impl<IdType, AttrMgr> BaseCollectionImpl<IdType, AttrMgr>
where
    IdType: Clone + Eq + std::hash::Hash,
{
    /// Create a new collection with an attribute manager.
    pub fn new(attribute_manager: AttrMgr) -> Self {
        Self {
            ids: HashSet::new(),
            attribute_manager,
        }
    }

    /// Add a single element (wraps batch add for efficiency).
    pub fn add_one(&mut self, item: IdType) -> Result<(), String> {
        self.add(vec![item])
    }

    /// Remove a single element (wraps batch remove for efficiency).
    pub fn remove_one(&mut self, id: &IdType) -> Result<(), String> {
        self.remove(&[*id.clone()])
    }
}

impl<IdType, AttrMgr> super::BaseCollection for BaseCollectionImpl<IdType, AttrMgr>
where
    IdType: Clone + Eq + std::hash::Hash,
{
    type IdType = IdType;
    type AttrMgr = AttrMgr;

    /// Batch add: optimized, always the fast path.
    fn add(&mut self, items: Vec<Self::IdType>) -> Result<(), String> {
        let mut duplicates = Vec::new();
        for id in items {
            if !self.ids.insert(id.clone()) {
                duplicates.push(id);
            }
        }
        if duplicates.is_empty() {
            Ok(())
        } else {
            Err(format!("Duplicate IDs: {:?}", duplicates))
        }
    }

    /// Batch remove: optimized, always the fast path.
    fn remove(&mut self, ids: &[Self::IdType]) -> Result<(), String> {
        let mut missing = Vec::new();
        for id in ids {
            if !self.ids.remove(id) {
                missing.push(id.clone());
            }
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!("IDs not found: {:?}", missing))
        }
    }

    fn filter<F>(&self, predicate: F) -> Vec<Self::IdType>
    where
        F: Fn(&Self::IdType) -> bool,
    {
        self.ids.iter().filter(|id| predicate(id)).cloned().collect()
    }

    fn size(&self) -> usize {
        self.ids.len()
    }

    fn ids(&self) -> Vec<Self::IdType> {
        self.ids.iter().cloned().collect()
    }

    fn has(&self, id: &Self::IdType) -> bool {
        self.ids.contains(id)
    }

    fn attr(&self) -> &Self::AttrMgr {
        &self.attribute_manager
    }

    fn iter(&self) -> Box<dyn Iterator<Item = Self::IdType> + '_> {
        Box::new(self.ids.iter().cloned())
    }

    fn get(&self, id: &Self::IdType) -> Option<&Self::IdType> {
        self.ids.get(id)
    }
}

