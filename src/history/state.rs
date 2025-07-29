//! State objects and metadata for graph history tracking.

use std::sync::Arc;
use crate::types::StateId;
use crate::core::delta::DeltaObject;
use crate::util::timestamp_now;

/// Immutable state object - a point in the graph's history
#[derive(Debug, Clone)]
pub struct StateObject {
    /// Parent state (None for root)
    pub parent: Option<StateId>,
    /// Changes from parent
    pub delta: Arc<DeltaObject>,
    /// Metadata
    pub metadata: Arc<StateMetadata>,
}

/// Metadata associated with a state
#[derive(Debug, Clone)]
pub struct StateMetadata {
    /// Human-readable label
    pub label: String,
    /// When this state was created (Unix timestamp)
    pub timestamp: u64,
    /// Who created this state
    pub author: String,
    /// Content hash for verification/deduplication
    pub hash: [u8; 32],
    /// Optional commit message
    pub message: Option<String>,
    /// Tags associated with this state
    pub tags: Vec<String>,
}

impl StateObject {
    /// Create a new state object
    pub fn new(
        parent: Option<StateId>,
        delta: DeltaObject,
        label: String,
        author: String,
        message: Option<String>,
    ) -> Self {
        let metadata = StateMetadata {
            label,
            timestamp: timestamp_now(),
            author,
            hash: delta.content_hash,
            message,
            tags: Vec::new(),
        };

        Self {
            parent,
            delta: Arc::new(delta),
            metadata: Arc::new(metadata),
        }
    }

    /// Create a root state (no parent)
    pub fn new_root(delta: DeltaObject, label: String, author: String) -> Self {
        Self::new(None, delta, label, author, None)
    }

    /// Get the parent state ID
    pub fn parent(&self) -> Option<StateId> {
        self.parent
    }

    /// Get the delta object
    pub fn delta(&self) -> &DeltaObject {
        &self.delta
    }

    /// Get the metadata
    pub fn metadata(&self) -> &StateMetadata {
        &self.metadata
    }

    /// Check if this is a root state (no parent)
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Get the content hash
    pub fn content_hash(&self) -> [u8; 32] {
        self.metadata.hash
    }

    /// Get the timestamp
    pub fn timestamp(&self) -> u64 {
        self.metadata.timestamp
    }

    /// Get the author
    pub fn author(&self) -> &str {
        &self.metadata.author
    }

    /// Get the label
    pub fn label(&self) -> &str {
        &self.metadata.label
    }

    /// Get the commit message
    pub fn message(&self) -> Option<&str> {
        self.metadata.message.as_deref()
    }

    /// Get tags
    pub fn tags(&self) -> &[String] {
        &self.metadata.tags
    }

    /// Add a tag to this state's metadata
    pub fn add_tag(&mut self, tag: String) {
        // Since metadata is Arc, we need to clone and modify
        let mut metadata = (*self.metadata).clone();
        metadata.tags.push(tag);
        self.metadata = Arc::new(metadata);
    }

    /// Remove a tag from this state's metadata
    pub fn remove_tag(&mut self, tag: &str) {
        let mut metadata = (*self.metadata).clone();
        metadata.tags.retain(|t| t != tag);
        self.metadata = Arc::new(metadata);
    }

    /// Check if this state has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.metadata.tags.contains(&tag.to_string())
    }

    /// Update the label
    pub fn set_label(&mut self, label: String) {
        let mut metadata = (*self.metadata).clone();
        metadata.label = label;
        self.metadata = Arc::new(metadata);
    }

    /// Update the message
    pub fn set_message(&mut self, message: Option<String>) {
        let mut metadata = (*self.metadata).clone();
        metadata.message = message;
        self.metadata = Arc::new(metadata);
    }

    /// Get the size of this state's delta in terms of change count
    pub fn delta_size(&self) -> usize {
        self.delta.change_count()
    }

    /// Check if this state represents an empty delta
    pub fn is_empty_delta(&self) -> bool {
        self.delta.is_empty()
    }
}

impl StateMetadata {
    /// Create new metadata
    pub fn new(label: String, author: String, hash: [u8; 32]) -> Self {
        Self {
            label,
            timestamp: timestamp_now(),
            author,
            hash,
            message: None,
            tags: Vec::new(),
        }
    }

    /// Create metadata with a message
    pub fn with_message(label: String, author: String, hash: [u8; 32], message: String) -> Self {
        Self {
            label,
            timestamp: timestamp_now(),
            author,
            hash,
            message: Some(message),
            tags: Vec::new(),
        }
    }

    /// Get a human-readable timestamp
    pub fn timestamp_string(&self) -> String {
        // Convert Unix timestamp to readable format
        // This is a simplified implementation
        format!("timestamp:{}", self.timestamp)
    }

    /// Get a short hash representation
    pub fn short_hash(&self) -> String {
        format!("{:02x}{:02x}{:02x}{:02x}", 
                self.hash[0], self.hash[1], self.hash[2], self.hash[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::delta::DeltaObject;

    #[test]
    fn test_state_object_creation() {
        let delta = DeltaObject::empty();
        let state = StateObject::new_root(
            delta,
            "Initial state".to_string(),
            "test_user".to_string(),
        );

        assert!(state.is_root());
        assert_eq!(state.label(), "Initial state");
        assert_eq!(state.author(), "test_user");
        assert!(state.is_empty_delta());
    }

    #[test]
    fn test_state_tags() {
        let delta = DeltaObject::empty();
        let mut state = StateObject::new_root(
            delta,
            "Tagged state".to_string(),
            "test_user".to_string(),
        );

        state.add_tag("important".to_string());
        state.add_tag("milestone".to_string());

        assert!(state.has_tag("important"));
        assert!(state.has_tag("milestone"));
        assert!(!state.has_tag("nonexistent"));

        state.remove_tag("important");
        assert!(!state.has_tag("important"));
        assert!(state.has_tag("milestone"));
    }
}
