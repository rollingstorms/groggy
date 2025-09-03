//! Reference Management System - Git-like branch and tag management.
//!
//! ARCHITECTURE ROLE:
//! This module provides Git-like reference management for organizing and
//! navigating the graph's history. It manages branches (mutable references)
//! and tags (immutable references) that point to specific states.
//!
//! DESIGN PHILOSOPHY:
//! - Lightweight references (just pointers to states)
//! - Git-like workflow (branch, merge, tag operations)
//! - Metadata-rich references (creation time, author, description)
//! - Concurrent-safe operations

use crate::errors::{GraphError, GraphResult};
use crate::types::{BranchName, StateId};
use std::collections::{HashMap, HashSet};

/*
=== REFERENCE SYSTEM OVERVIEW ===

The reference system provides:
1. BRANCH MANAGEMENT: Mutable pointers to evolving history lines
2. TAG MANAGEMENT: Immutable markers for important states
3. CURRENT CONTEXT: Track which branch is currently checked out
4. MERGE OPERATIONS: Combine history from different branches
5. GARBAGE COLLECTION: Clean up references to non-existent states

KEY DESIGN DECISIONS:
- Branches are lightweight (just state pointers + metadata)
- Tags are immutable once created
- Always maintain a "current branch" for context
- Support branch descriptions and creation metadata
- Automatic cleanup of invalid references
*/

/// A branch pointer to a specific state in the graph history
///
/// DESIGN: Branches are mutable references that can be updated to point
/// to new states as development progresses. They carry metadata about
/// their creation and purpose.
///
/// LIFECYCLE:
/// 1. Created pointing to an initial state
/// 2. Updated as new commits are made on the branch
/// 3. Can be merged with other branches
/// 4. Can be deleted when no longer needed
#[derive(Debug, Clone)]
pub struct Branch {
    /// Human-readable branch name (must be unique)
    /// EXAMPLES: "main", "feature/user-auth", "hotfix/security-patch"
    pub name: BranchName,

    /// Current head state (most recent commit on this branch)
    /// This is the state that new commits will build upon
    pub head: StateId,

    /// Optional human-readable description
    /// USAGE: Explain the purpose of this branch
    pub description: Option<String>,

    /// When this branch was created (Unix timestamp)
    /// USAGE: For sorting, cleanup, and auditing
    pub created_at: u64,

    /// Who created this branch
    /// USAGE: For auditing and contact information
    pub created_by: String,
}

impl Branch {
    /// Create a new branch pointing to a specific state
    pub fn new(name: BranchName, head: StateId, created_by: String) -> Self {
        Self {
            name,
            head,
            description: None,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            created_by,
        }
    }

    /// Create a branch with a description
    pub fn with_description(
        name: BranchName,
        head: StateId,
        created_by: String,
        description: String,
    ) -> Self {
        Self {
            name,
            head,
            description: Some(description),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            created_by,
        }
    }

    /// Update the head of this branch to a new state
    ///
    /// USAGE: Called when new commits are made on this branch
    pub fn update_head(&mut self, new_head: StateId) {
        self.head = new_head;
    }

    /// Set or update the description
    pub fn set_description(&mut self, description: Option<String>) {
        self.description = description;
    }

    /// Check if this branch is older than a certain number of days
    pub fn is_older_than_days(&self, days: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let threshold = days * 24 * 60 * 60; // Convert to seconds
        now.saturating_sub(self.created_at) > threshold
    }
}

/// Manages all branches and tags in the graph system
///
/// RESPONSIBILITIES:
/// - Maintain the collection of all branches and tags
/// - Track which branch is currently checked out
/// - Provide operations for creating, deleting, and updating references
/// - Handle branch switching and merging
/// - Clean up stale references
///
/// NOT RESPONSIBLE FOR:
/// - Actually storing graph states (that's HistoryForest's job)
/// - Performing merge operations (that's Graph's job)
/// - Validating state existence (that's HistoryForest's job)
#[derive(Debug)]
pub struct RefManager {
    /*
    === BRANCH MANAGEMENT ===
    */
    /// All branches indexed by name
    /// INVARIANT: Always contains at least the default branch
    branches: HashMap<BranchName, Branch>,

    /// Currently checked out branch
    /// INVARIANT: Must always exist in the branches map
    current_branch: BranchName,

    /// Default branch name (usually "main" or "master")
    /// INVARIANT: This branch cannot be deleted
    default_branch: BranchName,

    /*
    === TAG MANAGEMENT ===
    */
    /// Tags (immutable refs to specific states)
    /// DESIGN: Simple map from tag name to state ID
    /// IMMUTABILITY: Tags cannot be moved once created
    tags: HashMap<String, StateId>,
}

impl RefManager {
    /// Create a new reference manager with a default branch
    pub fn new() -> Self {
        let default_branch = "main".to_string();
        let mut branches = HashMap::new();

        // Create the default branch pointing to state 0 (empty state)
        let default_branch_obj = Branch::new(
            default_branch.clone(),
            0, // StateId is just usize
            "system".to_string(),
        );
        branches.insert(default_branch.clone(), default_branch_obj);

        Self {
            branches,
            current_branch: default_branch.clone(),
            default_branch,
            tags: HashMap::new(),
        }
    }

    /// Create a new reference manager with a custom default branch name
    pub fn with_default_branch(default_name: BranchName) -> Self {
        let mut branches = HashMap::new();

        // Create the default branch pointing to state 0 (empty state)
        let default_branch_obj = Branch::new(
            default_name.clone(),
            0, // StateId is just usize
            "system".to_string(),
        );
        branches.insert(default_name.clone(), default_branch_obj);

        Self {
            branches,
            current_branch: default_name.clone(),
            default_branch: default_name,
            tags: HashMap::new(),
        }
    }

    /*
    === BRANCH OPERATIONS ===
    */

    /// Create a new branch pointing to a specific state
    ///
    /// ALGORITHM:
    /// 1. Check that branch name doesn't already exist
    /// 2. Validate that the target state exists (optional check)
    /// 3. Create Branch object with metadata
    /// 4. Add to branches map
    pub fn create_branch(
        &mut self,
        name: BranchName,
        start_state: StateId,
        created_by: String,
        description: Option<String>,
    ) -> GraphResult<()> {
        if self.branches.contains_key(&name) {
            return Err(GraphError::BranchAlreadyExists {
                branch_name: name.clone(),
                existing_head: self.branches[&name].head,
            });
        }

        let branch = if let Some(desc) = description {
            Branch::with_description(name.clone(), start_state, created_by, desc)
        } else {
            Branch::new(name.clone(), start_state, created_by)
        };

        self.branches.insert(name, branch);
        Ok(())
    }

    /// Switch to a different branch (checkout)
    ///
    /// ALGORITHM:
    /// 1. Verify the target branch exists
    /// 2. Update current_branch pointer
    /// 3. Return the state ID that should be loaded
    pub fn checkout_branch(&mut self, name: &BranchName) -> GraphResult<StateId> {
        if !self.branches.contains_key(name) {
            return Err(GraphError::BranchNotFound {
                branch_name: name.clone(),
                operation: "checkout".to_string(),
                available_branches: self.branches.keys().cloned().collect(),
            });
        }

        self.current_branch = name.clone();
        Ok(self.branches[name].head)
    }

    /// Delete a branch
    ///
    /// RESTRICTIONS:
    /// - Cannot delete the default branch
    /// - Cannot delete the currently checked out branch
    /// - Branch must exist
    pub fn delete_branch(&mut self, name: &BranchName) -> GraphResult<()> {
        if name == &self.default_branch {
            return Err(GraphError::InvalidInput(format!(
                "Cannot delete default branch '{}': The default branch cannot be deleted",
                name
            )));
        }

        if name == &self.current_branch {
            return Err(GraphError::CannotDeleteCurrentBranch {
                branch_name: name.clone(),
            });
        }

        if !self.branches.contains_key(name) {
            return Err(GraphError::BranchNotFound {
                branch_name: name.clone(),
                operation: "delete".to_string(),
                available_branches: self.branches.keys().cloned().collect(),
            });
        }

        self.branches.remove(name);
        Ok(())
    }

    /// List all branches with their metadata
    pub fn list_branches(&self) -> Vec<BranchInfo> {
        self.branches
            .values()
            .map(|branch| BranchInfo {
                name: branch.name.clone(),
                head: branch.head,
                description: branch.description.clone(),
                created_at: branch.created_at,
                created_by: branch.created_by.clone(),
                is_current: branch.name == self.current_branch,
                is_default: branch.name == self.default_branch,
            })
            .collect()
    }

    /// Get the currently checked out branch
    pub fn get_current_branch(&self) -> GraphResult<&Branch> {
        self.branches
            .get(&self.current_branch)
            .ok_or_else(|| GraphError::BranchNotFound {
                branch_name: self.current_branch.clone(),
                operation: "get current".to_string(),
                available_branches: self.branches.keys().cloned().collect(),
            })
    }

    /// Get a specific branch by name
    pub fn get_branch(&self, name: &BranchName) -> GraphResult<&Branch> {
        self.branches
            .get(name)
            .ok_or_else(|| GraphError::BranchNotFound {
                branch_name: name.clone(),
                operation: "get branch".to_string(),
                available_branches: self.branches.keys().cloned().collect(),
            })
    }

    /// Update the head of the current branch
    ///
    /// USAGE: Called after making a new commit
    pub fn update_current_branch_head(&mut self, new_head: StateId) -> GraphResult<()> {
        let current_name = self.current_branch.clone();
        if let Some(branch) = self.branches.get_mut(&current_name) {
            branch.update_head(new_head);
            Ok(())
        } else {
            Err(GraphError::BranchNotFound {
                branch_name: current_name,
                operation: "update head".to_string(),
                available_branches: self.branches.keys().cloned().collect(),
            })
        }
    }

    /// Update the head of a specific branch
    pub fn update_branch_head(
        &mut self,
        branch_name: &BranchName,
        new_head: StateId,
    ) -> GraphResult<()> {
        if let Some(branch) = self.branches.get_mut(branch_name) {
            branch.update_head(new_head);
            Ok(())
        } else {
            Err(GraphError::BranchNotFound {
                branch_name: branch_name.clone(),
                operation: "update head".to_string(),
                available_branches: self.branches.keys().cloned().collect(),
            })
        }
    }

    /*
    === TAG OPERATIONS ===
    */

    /// Create a new tag pointing to a specific state
    ///
    /// IMMUTABILITY: Tags cannot be moved once created
    pub fn create_tag(&mut self, tag_name: String, state_id: StateId) -> GraphResult<()> {
        if self.tags.contains_key(&tag_name) {
            return Err(GraphError::InvalidInput(format!(
                "Tag '{}' already exists pointing to state {}",
                tag_name, self.tags[&tag_name]
            )));
        }

        self.tags.insert(tag_name, state_id);
        Ok(())
    }

    /// Delete a tag
    pub fn delete_tag(&mut self, tag_name: &str) -> GraphResult<()> {
        if !self.tags.contains_key(tag_name) {
            return Err(GraphError::InvalidInput(format!(
                "Tag '{}' not found. Available tags: {}",
                tag_name,
                self.tags.keys().cloned().collect::<Vec<_>>().join(", ")
            )));
        }

        self.tags.remove(tag_name);
        Ok(())
    }

    /// List all tags
    pub fn list_tags(&self) -> Vec<TagInfo> {
        self.tags
            .iter()
            .map(|(name, &state_id)| TagInfo {
                name: name.clone(),
                state_id,
            })
            .collect()
    }

    /// Get the state ID for a specific tag
    pub fn get_tag(&self, tag_name: &str) -> Option<StateId> {
        self.tags.get(tag_name).copied()
    }

    /*
    === UTILITY AND MAINTENANCE OPERATIONS ===
    */

    /// Get all state IDs referenced by branches and tags
    ///
    /// USAGE: For garbage collection - these states should not be deleted
    pub fn get_referenced_states(&self) -> Vec<StateId> {
        let mut states = Vec::new();

        // Add branch heads
        for branch in self.branches.values() {
            states.push(branch.head);
        }

        // Add tag states
        for &state_id in self.tags.values() {
            states.push(state_id);
        }

        // Remove duplicates and sort
        states.sort();
        states.dedup();
        states
    }

    /// Clean up branches that point to non-existent states
    ///
    /// USAGE: After garbage collection in the history system
    /// RETURNS: Number of branches that were removed
    pub fn prune_invalid_branches(&mut self, valid_states: &[StateId]) -> usize {
        let valid_set: HashSet<_> = valid_states.iter().collect();
        let mut removed_count = 0;

        let branch_names: Vec<_> = self.branches.keys().cloned().collect();
        for branch_name in branch_names {
            if let Some(branch) = self.branches.get(&branch_name) {
                if !valid_set.contains(&branch.head) && branch_name != self.default_branch {
                    self.branches.remove(&branch_name);
                    removed_count += 1;
                }
            }
        }

        removed_count
    }

    /// Clean up tags that point to non-existent states
    ///
    /// RETURNS: Number of tags that were removed
    pub fn prune_invalid_tags(&mut self, valid_states: &[StateId]) -> usize {
        let valid_set: HashSet<_> = valid_states.iter().collect();
        let mut removed_count = 0;

        let tag_names: Vec<_> = self.tags.keys().cloned().collect();
        for tag_name in tag_names {
            if let Some(&state_id) = self.tags.get(&tag_name) {
                if !valid_set.contains(&state_id) {
                    self.tags.remove(&tag_name);
                    removed_count += 1;
                }
            }
        }

        removed_count
    }

    /// Get basic information about the reference manager
    pub fn statistics(&self) -> RefStatistics {
        RefStatistics {
            branch_count: self.branches.len(),
            tag_count: self.tags.len(),
            current_branch: self.current_branch.clone(),
            default_branch: self.default_branch.clone(),
        }
    }

    /*
    === HELPER METHODS ===
    */

    /// Get list of all branch names (for error messages)
    #[allow(dead_code)]
    fn list_branch_names(&self) -> Vec<BranchName> {
        self.branches.keys().cloned().collect()
    }

    /// Get the current branch name
    pub fn current_branch_name(&self) -> &BranchName {
        &self.current_branch
    }

    /// Get the default branch name
    pub fn default_branch_name(&self) -> &BranchName {
        &self.default_branch
    }

    /// Check if a branch exists
    pub fn has_branch(&self, name: &BranchName) -> bool {
        self.branches.contains_key(name)
    }

    /// Check if a tag exists
    pub fn has_tag(&self, name: &str) -> bool {
        self.tags.contains_key(name)
    }
}

impl Default for RefManager {
    fn default() -> Self {
        Self::new()
    }
}

/*
=== SUPPORTING DATA STRUCTURES ===
*/

/// Information about a branch for listing and display
#[derive(Debug, Clone)]
pub struct BranchInfo {
    pub name: BranchName,
    pub head: StateId,
    pub description: Option<String>,
    pub created_at: u64,
    pub created_by: String,
    pub is_current: bool,
    pub is_default: bool,
}

impl BranchInfo {
    /// Get a human-readable description of this branch
    pub fn display_name(&self) -> String {
        let mut result = self.name.clone();
        if self.is_current {
            result.push_str(" *");
        }
        if self.is_default {
            result.push_str(" (default)");
        }
        result
    }

    /// Get the age of this branch in days
    pub fn age_days(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        (now.saturating_sub(self.created_at)) / (24 * 60 * 60)
    }
}

/// Information about a tag for listing and display
#[derive(Debug, Clone)]
pub struct TagInfo {
    pub name: String,
    pub state_id: StateId,
}

/// Statistics about the reference manager
#[derive(Debug, Clone)]
pub struct RefStatistics {
    pub branch_count: usize,
    pub tag_count: usize,
    pub current_branch: BranchName,
    pub default_branch: BranchName,
}

/*
=== COMPREHENSIVE TEST SUITE ===
*/

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_ref_manager_creation() {
        // TODO: Uncomment when RefManager is implemented
        /*
        let ref_manager = RefManager::new();

        // Should have default branch
        assert_eq!(ref_manager.branch_count(), 1);
        assert_eq!(ref_manager.current_branch_name(), &"main".to_string());
        assert_eq!(ref_manager.default_branch_name(), &"main".to_string());
        assert!(ref_manager.has_branch(&"main".to_string()));
        */
    }

    #[test]
    fn test_branch_creation_and_deletion() {
        // TODO: Test branch lifecycle
        /*
        let mut ref_manager = RefManager::new();

        // Create a new branch
        ref_manager.create_branch(
            "feature".to_string(),
            StateId(1),
            "user".to_string(),
            Some("Feature branch".to_string()),
        ).unwrap();

        assert!(ref_manager.has_branch(&"feature".to_string()));
        assert_eq!(ref_manager.branch_count(), 2);

        let branch = ref_manager.get_branch(&"feature".to_string()).unwrap();
        assert_eq!(branch.head, StateId(1));
        assert_eq!(branch.description, Some("Feature branch".to_string()));

        // Delete the branch
        ref_manager.delete_branch(&"feature".to_string()).unwrap();
        assert!(!ref_manager.has_branch(&"feature".to_string()));
        assert_eq!(ref_manager.branch_count(), 1);
        */
    }

    #[test]
    fn test_branch_checkout() {
        // TODO: Test branch switching
        /*
        let mut ref_manager = RefManager::new();

        ref_manager.create_branch(
            "dev".to_string(),
            StateId(2),
            "user".to_string(),
            None
        ).unwrap();

        let head = ref_manager.checkout_branch(&"dev".to_string()).unwrap();
        assert_eq!(head, StateId(2));
        assert_eq!(ref_manager.current_branch_name(), &"dev".to_string());
        */
    }

    #[test]
    fn test_tag_operations() {
        // TODO: Test tag lifecycle
        /*
        let mut ref_manager = RefManager::new();

        // Create tag
        ref_manager.create_tag("v1.0".to_string(), StateId(5)).unwrap();

        assert!(ref_manager.has_tag("v1.0"));
        assert_eq!(ref_manager.get_tag("v1.0"), Some(StateId(5)));

        // List tags
        let tags = ref_manager.list_tags();
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].name, "v1.0");
        assert_eq!(tags[0].state_id, StateId(5));

        // Delete tag
        ref_manager.delete_tag("v1.0").unwrap();
        assert!(!ref_manager.has_tag("v1.0"));
        */
    }

    #[test]
    fn test_reference_cleanup() {
        // TODO: Test cleanup of invalid references
        /*
        let mut ref_manager = RefManager::new();

        // Create branches and tags
        ref_manager.create_branch("temp".to_string(), StateId(999), "user".to_string(), None).unwrap();
        ref_manager.create_tag("old_tag".to_string(), StateId(888)).unwrap();

        // Prune invalid references
        let valid_states = vec![StateId(0), StateId(1), StateId(2)];
        let removed_branches = ref_manager.prune_invalid_branches(&valid_states);
        let removed_tags = ref_manager.prune_invalid_tags(&valid_states);

        assert_eq!(removed_branches, 1); // temp branch removed
        assert_eq!(removed_tags, 1); // old_tag removed
        */
    }

    #[test]
    fn test_error_handling() {
        // TODO: Test error conditions
        /*
        let mut ref_manager = RefManager::new();

        // Try to create duplicate branch
        let result = ref_manager.create_branch(
            "main".to_string(),
            StateId(1),
            "user".to_string(),
            None
        );
        assert!(result.is_err());

        // Try to checkout non-existent branch
        let result = ref_manager.checkout_branch(&"nonexistent".to_string());
        assert!(result.is_err());

        // Try to delete default branch
        let result = ref_manager.delete_branch(&"main".to_string());
        assert!(result.is_err());

        // Try to delete current branch
        ref_manager.create_branch("test".to_string(), StateId(1), "user".to_string(), None).unwrap();
        ref_manager.checkout_branch(&"test".to_string()).unwrap();
        let result = ref_manager.delete_branch(&"test".to_string());
        assert!(result.is_err());
        */
    }
}

/*
=== IMPLEMENTATION NOTES ===

PERFORMANCE CHARACTERISTICS:
- Branch operations: O(1) for most operations (HashMap lookups)
- Tag operations: O(1) for most operations
- Listing operations: O(n) where n = number of branches/tags
- Cleanup operations: O(n) where n = number of references

MEMORY USAGE:
- Very lightweight - just metadata and pointers
- No duplication of state data
- Efficient HashMap storage for fast lookups

CONCURRENCY:
- Thread-safe for read operations
- Mutations require exclusive access
- No internal locking (handled at higher level)

INTEGRATION WITH HISTORY:
- RefManager doesn't validate state existence
- HistoryForest is responsible for state management
- Cleanup operations bridge the two systems

FUTURE ENHANCEMENTS:
- Branch permissions and access control
- Branch-specific configuration
- Automatic branch cleanup policies
- Integration with external version control systems
*/
