//! Error handling system for the graph library.
//!
//! DESIGN PHILOSOPHY:
//! - Comprehensive error types that provide actionable information
//! - Context-rich errors that help with debugging
//! - Performance-aware (errors should be cheap to create)
//! - Extensible for future error scenarios

/*
=== ERROR HANDLING STRATEGY ===

This module defines all the ways operations can fail in the graph system.
Good error handling is critical for:

1. Debugging - Users need to understand what went wrong
2. Recovery - Some errors can be handled programmatically
3. User Experience - Clear error messages improve usability
4. System Reliability - Proper error handling prevents crashes

DESIGN DECISIONS:
- Use Result<T, GraphError> for all fallible operations
- Provide detailed context in error messages
- Include suggested recovery actions where possible
- Separate user errors from system/internal errors
*/

use crate::types::{AttrName, BranchName, EdgeId, NodeId, StateId};
use std::collections::HashMap;

/// The main error type for all graph operations
///
/// DESIGN: Use an enum to represent all possible error conditions.
/// Each variant includes relevant context and should provide actionable information.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /*
    === ENTITY NOT FOUND ERRORS ===
    The most common errors - trying to operate on non-existent entities
    */
    /// Attempted to operate on a node that doesn't exist
    NodeNotFound {
        node_id: NodeId,
        operation: String,
        suggestion: String,
    },

    /// Attempted to operate on an edge that doesn't exist
    EdgeNotFound {
        edge_id: EdgeId,
        operation: String,
        suggestion: String,
    },

    /// Attempted to operate on a state that doesn't exist in history
    StateNotFound {
        state_id: StateId,
        operation: String,
        available_states: Vec<StateId>,
    },

    /// Attempted to operate on a branch that doesn't exist
    BranchNotFound {
        branch_name: BranchName,
        operation: String,
        available_branches: Vec<BranchName>,
    },

    /*
    === INVALID OPERATION ERRORS ===
    Operations that are syntactically valid but semantically invalid
    */
    /// Attempted to create an edge with invalid endpoints
    InvalidEdgeEndpoints {
        source: NodeId,
        target: NodeId,
        reason: String,
    },

    /// Attempted to perform an operation that requires no uncommitted changes
    UncommittedChanges {
        operation: String,
        change_count: usize,
        suggestion: String,
    },

    /// Attempted to create a branch that already exists
    BranchAlreadyExists {
        branch_name: BranchName,
        existing_head: StateId,
    },

    /// Attempted to delete a branch that is currently checked out
    CannotDeleteCurrentBranch { branch_name: BranchName },

    /// Attempted an operation on an empty graph that requires data
    EmptyGraph { operation: String },

    /*
    === ATTRIBUTE ERRORS ===
    Problems with attribute operations
    */
    /// Attempted to access an attribute that doesn't exist
    AttributeNotFound {
        entity_type: EntityType,
        entity_id: u64,
        attribute_name: AttrName,
        available_attributes: Vec<AttrName>,
    },

    /// Attempted to set an attribute with an incompatible type
    AttributeTypeMismatch {
        entity_type: EntityType,
        entity_id: u64,
        attribute_name: AttrName,
        expected_type: String,
        provided_type: String,
        provided_value: String,
    },

    /// Attribute name is invalid (empty, too long, contains invalid characters)
    InvalidAttributeName {
        attribute_name: String,
        reason: String,
    },

    /*
    === HISTORY/VERSION CONTROL ERRORS ===
    Problems with commits, branches, and version control
    */
    /// Attempted to commit when there are no changes
    NoChangesToCommit,

    /// History operation failed due to corrupted state
    CorruptedHistory {
        state_id: StateId,
        corruption_type: String,
        recovery_suggestion: String,
    },

    /// Merge operation failed due to conflicts
    MergeConflict {
        conflicts: Vec<MergeConflictDetail>,
        resolution_suggestions: Vec<String>,
    },

    /// Attempted to access a state that exists but is unreachable
    UnreachableState { state_id: StateId, reason: String },

    /*
    === CAPACITY/RESOURCE ERRORS ===
    System limits exceeded
    */
    /// Too many nodes/edges (implementation limit exceeded)
    CapacityExceeded {
        resource_type: String,
        current_count: usize,
        maximum_allowed: usize,
    },

    /// Operation would use too much memory
    MemoryExhausted {
        requested_bytes: usize,
        available_bytes: usize,
        operation: String,
    },

    /// History has grown too large
    HistoryTooLarge {
        current_size: usize,
        maximum_size: usize,
        suggestion: String,
    },

    /*
    === CONCURRENCY ERRORS ===
    Problems with concurrent access (future use)
    */
    /// Another operation is already in progress
    OperationInProgress {
        conflicting_operation: String,
        estimated_completion: Option<u64>,
    },

    /// Lock acquisition failed
    LockContentionError { resource: String, timeout_ms: u64 },

    /*
    === I/O AND PERSISTENCE ERRORS ===
    File system and serialization problems
    */
    /// Failed to read from or write to disk
    IoError {
        operation: String,
        path: String,
        underlying_error: String,
    },

    /// Failed to serialize or deserialize data
    SerializationError {
        data_type: String,
        operation: String, // "serialize" or "deserialize"
        underlying_error: String,
    },

    /// File format is not supported or corrupted
    InvalidFileFormat {
        path: String,
        expected_format: String,
        detected_format: Option<String>,
    },

    /*
    === QUERY ERRORS ===
    Problems with filtering and query operations
    */
    /// Query syntax is invalid
    InvalidQuery {
        query: String,
        error_position: Option<usize>,
        error_message: String,
        suggestion: String,
    },

    /// Query is valid but cannot be executed efficiently
    QueryTooComplex {
        query: String,
        complexity_score: f64,
        maximum_allowed: f64,
        suggestion: String,
    },

    /// Query timed out
    QueryTimeout {
        query: String,
        timeout_ms: u64,
        partial_results: Option<usize>,
    },

    /*
    === CONFIGURATION ERRORS ===
    Problems with settings and configuration
    */
    /// Configuration value is invalid
    InvalidConfiguration {
        setting: String,
        value: String,
        reason: String,
        valid_values: Vec<String>,
    },

    /// General invalid input error  
    InvalidInput(String),

    /// Required configuration is missing
    MissingConfiguration {
        setting: String,
        description: String,
    },

    /*
    === INTERNAL/SYSTEM ERRORS ===
    Unexpected conditions that indicate bugs
    */
    /// An internal invariant was violated (this indicates a bug)
    InternalError {
        message: String,
        location: String,
        context: HashMap<String, String>,
    },

    /// Operation not yet implemented
    NotImplemented {
        feature: String,
        tracking_issue: Option<String>,
    },

    /// Unexpected state that should never occur
    UnexpectedState {
        expected: String,
        actual: String,
        operation: String,
    },
}

impl GraphError {
    /*
    === ERROR CONSTRUCTION HELPERS ===
    Convenience methods for creating common errors with good context
    */

    /// Create a NodeNotFound error with helpful context
    pub fn node_not_found(node_id: NodeId, operation: &str) -> Self {
        Self::NodeNotFound {
            node_id,
            operation: operation.to_string(),
            suggestion: format!(
                "Check that node {} exists before trying to {}",
                node_id, operation
            ),
        }
    }

    /// Create an EdgeNotFound error with helpful context
    pub fn edge_not_found(edge_id: EdgeId, operation: &str) -> Self {
        Self::EdgeNotFound {
            edge_id,
            operation: operation.to_string(),
            suggestion: format!(
                "Check that edge {} exists before trying to {}",
                edge_id, operation
            ),
        }
    }

    /// Create a StateNotFound error with available alternatives
    pub fn state_not_found(state_id: StateId, operation: &str, available: Vec<StateId>) -> Self {
        Self::StateNotFound {
            state_id,
            operation: operation.to_string(),
            available_states: available,
        }
    }

    /// Create a BranchNotFound error with available alternatives
    pub fn branch_not_found(
        branch_name: BranchName,
        operation: &str,
        available: Vec<BranchName>,
    ) -> Self {
        Self::BranchNotFound {
            branch_name,
            operation: operation.to_string(),
            available_branches: available,
        }
    }

    /// Create an UncommittedChanges error with helpful context
    pub fn uncommitted_changes(operation: &str, change_count: usize) -> Self {
        Self::UncommittedChanges {
            operation: operation.to_string(),
            change_count,
            suggestion:
                "Commit your changes with graph.commit() or reset them with graph.reset_hard()"
                    .to_string(),
        }
    }

    /// Create an AttributeNotFound error with available alternatives
    pub fn attribute_not_found(
        entity_type: EntityType,
        entity_id: u64,
        attr_name: AttrName,
        available: Vec<AttrName>,
    ) -> Self {
        Self::AttributeNotFound {
            entity_type,
            entity_id,
            attribute_name: attr_name,
            available_attributes: available,
        }
    }

    /// Create an InternalError for unexpected conditions
    pub fn internal(message: &str, location: &str) -> Self {
        Self::InternalError {
            message: message.to_string(),
            location: location.to_string(),
            context: HashMap::new(),
        }
    }

    /// Create an InternalError with additional context
    pub fn internal_with_context(
        message: &str,
        location: &str,
        context: HashMap<String, String>,
    ) -> Self {
        Self::InternalError {
            message: message.to_string(),
            location: location.to_string(),
            context,
        }
    }

    /*
    === ERROR ANALYSIS METHODS ===
    Methods to help understand and categorize errors
    */

    /// Check if this error indicates a user mistake (vs system problem)
    pub fn is_user_error(&self) -> bool {
        match self {
            // User errors - user provided invalid input or tried invalid operation
            GraphError::NodeNotFound { .. }
            | GraphError::EdgeNotFound { .. }
            | GraphError::StateNotFound { .. }
            | GraphError::BranchNotFound { .. }
            | GraphError::InvalidEdgeEndpoints { .. }
            | GraphError::UncommittedChanges { .. }
            | GraphError::BranchAlreadyExists { .. }
            | GraphError::AttributeNotFound { .. }
            | GraphError::AttributeTypeMismatch { .. }
            | GraphError::InvalidAttributeName { .. }
            | GraphError::NoChangesToCommit
            | GraphError::EmptyGraph { .. }
            | GraphError::InvalidQuery { .. }
            | GraphError::InvalidConfiguration { .. }
            | GraphError::InvalidInput(_) => true,

            // System errors - something went wrong internally
            GraphError::CorruptedHistory { .. }
            | GraphError::MemoryExhausted { .. }
            | GraphError::IoError { .. }
            | GraphError::SerializationError { .. }
            | GraphError::InternalError { .. }
            | GraphError::UnexpectedState { .. } => false,

            // Ambiguous cases - could be either
            _ => false,
        }
    }

    /// Check if this error might be recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Definitely recoverable - user can fix and retry
            GraphError::NodeNotFound { .. }
            | GraphError::EdgeNotFound { .. }
            | GraphError::UncommittedChanges { .. }
            | GraphError::AttributeNotFound { .. }
            | GraphError::NoChangesToCommit
            | GraphError::InvalidQuery { .. } => true,

            // Definitely not recoverable - system is in bad state
            GraphError::CorruptedHistory { .. }
            | GraphError::MemoryExhausted { .. }
            | GraphError::InternalError { .. } => false,

            // Might be recoverable depending on circumstances
            _ => false,
        }
    }

    /// Get the error category for grouping similar errors
    pub fn category(&self) -> ErrorCategory {
        match self {
            GraphError::NodeNotFound { .. }
            | GraphError::EdgeNotFound { .. }
            | GraphError::StateNotFound { .. }
            | GraphError::BranchNotFound { .. }
            | GraphError::AttributeNotFound { .. } => ErrorCategory::NotFound,

            GraphError::InvalidEdgeEndpoints { .. }
            | GraphError::InvalidAttributeName { .. }
            | GraphError::InvalidQuery { .. }
            | GraphError::InvalidConfiguration { .. }
            | GraphError::InvalidInput(_) => ErrorCategory::InvalidInput,

            GraphError::UncommittedChanges { .. }
            | GraphError::BranchAlreadyExists { .. }
            | GraphError::CannotDeleteCurrentBranch { .. }
            | GraphError::NoChangesToCommit => ErrorCategory::InvalidOperation,

            GraphError::CorruptedHistory { .. } | GraphError::UnreachableState { .. } => {
                ErrorCategory::DataCorruption
            }

            GraphError::CapacityExceeded { .. }
            | GraphError::MemoryExhausted { .. }
            | GraphError::HistoryTooLarge { .. } => ErrorCategory::ResourceExhausted,

            GraphError::IoError { .. }
            | GraphError::SerializationError { .. }
            | GraphError::InvalidFileFormat { .. } => ErrorCategory::Persistence,

            GraphError::QueryTimeout { .. } | GraphError::QueryTooComplex { .. } => {
                ErrorCategory::Query
            }

            GraphError::OperationInProgress { .. } | GraphError::LockContentionError { .. } => {
                ErrorCategory::Concurrency
            }

            GraphError::InternalError { .. }
            | GraphError::UnexpectedState { .. }
            | GraphError::NotImplemented { .. } => ErrorCategory::Internal,

            _ => ErrorCategory::Other,
        }
    }

    /// Get a short description suitable for logging
    pub fn short_description(&self) -> String {
        match self {
            GraphError::NodeNotFound {
                node_id, operation, ..
            } => format!("Node {} not found during {}", node_id, operation),
            GraphError::EdgeNotFound {
                edge_id, operation, ..
            } => format!("Edge {} not found during {}", edge_id, operation),
            GraphError::UncommittedChanges { change_count, .. } => {
                format!("{} uncommitted changes", change_count)
            }
            GraphError::InternalError { message, .. } => format!("Internal error: {}", message),
            // TODO: Add cases for other error types
            _ => format!("{:?}", self),
        }
    }

    /// Get actionable suggestions for resolving this error
    pub fn suggestions(&self) -> Vec<String> {
        match self {
            GraphError::NodeNotFound { suggestion, .. }
            | GraphError::EdgeNotFound { suggestion, .. }
            | GraphError::UncommittedChanges { suggestion, .. } => vec![suggestion.clone()],

            GraphError::StateNotFound {
                available_states, ..
            } => {
                if available_states.is_empty() {
                    vec!["No states available in history".to_string()]
                } else {
                    vec![format!("Available states: {:?}", available_states)]
                }
            }

            GraphError::BranchNotFound {
                available_branches, ..
            } => {
                if available_branches.is_empty() {
                    vec!["No branches available".to_string()]
                } else {
                    vec![format!("Available branches: {:?}", available_branches)]
                }
            }

            GraphError::MergeConflict {
                resolution_suggestions,
                ..
            } => resolution_suggestions.clone(),

            // TODO: Add suggestions for other error types
            _ => vec![],
        }
    }
}

/*
=== SUPPORTING TYPES ===
*/

/// Categories for grouping similar errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorCategory {
    NotFound,
    InvalidInput,
    InvalidOperation,
    DataCorruption,
    ResourceExhausted,
    Persistence,
    Query,
    Concurrency,
    Internal,
    Other,
}

/// Entity types for error reporting
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityType {
    Node,
    Edge,
    State,
    Branch,
    Attribute,
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntityType::Node => write!(f, "node"),
            EntityType::Edge => write!(f, "edge"),
            EntityType::State => write!(f, "state"),
            EntityType::Branch => write!(f, "branch"),
            EntityType::Attribute => write!(f, "attribute"),
        }
    }
}

/// Detailed information about a merge conflict
#[derive(Debug, Clone, PartialEq)]
pub struct MergeConflictDetail {
    pub entity_type: EntityType,
    pub entity_id: u64,
    pub attribute: Option<AttrName>,
    pub our_value: String,
    pub their_value: String,
    pub common_ancestor_value: Option<String>,
}

/*
=== RESULT TYPE ALIAS ===
*/

/// Standard result type for graph operations
/// This saves typing and ensures consistency across the codebase
pub type GraphResult<T> = Result<T, GraphError>;

/*
=== ERROR FORMATTING ===
*/

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::NodeNotFound {
                node_id,
                operation,
                suggestion,
            } => {
                write!(
                    f,
                    "Node {} not found while attempting to {}. {}",
                    node_id, operation, suggestion
                )
            }

            GraphError::EdgeNotFound {
                edge_id,
                operation,
                suggestion,
            } => {
                write!(
                    f,
                    "Edge {} not found while attempting to {}. {}",
                    edge_id, operation, suggestion
                )
            }

            GraphError::StateNotFound {
                state_id,
                operation,
                available_states,
            } => {
                write!(
                    f,
                    "State {} not found while attempting to {}. Available states: {:?}",
                    state_id, operation, available_states
                )
            }

            GraphError::BranchNotFound {
                branch_name,
                operation,
                available_branches,
            } => {
                write!(
                    f,
                    "Branch '{}' not found while attempting to {}. Available branches: {:?}",
                    branch_name, operation, available_branches
                )
            }

            GraphError::UncommittedChanges {
                operation,
                change_count,
                suggestion,
            } => {
                write!(
                    f,
                    "Cannot {} with {} uncommitted changes. {}",
                    operation, change_count, suggestion
                )
            }

            GraphError::BranchAlreadyExists {
                branch_name,
                existing_head,
            } => {
                write!(
                    f,
                    "Branch '{}' already exists (currently points to state {})",
                    branch_name, existing_head
                )
            }

            GraphError::AttributeNotFound {
                entity_type,
                entity_id,
                attribute_name,
                available_attributes,
            } => {
                write!(
                    f,
                    "Attribute '{}' not found on {} {}. Available attributes: {:?}",
                    attribute_name, entity_type, entity_id, available_attributes
                )
            }

            GraphError::NoChangesToCommit => {
                write!(f, "No changes to commit. Make some modifications first.")
            }

            GraphError::InternalError {
                message,
                location,
                context,
            } => {
                write!(
                    f,
                    "Internal error at {}: {}. Context: {:?}",
                    location, message, context
                )
            }

            // TODO: Add display implementations for all error variants
            _ => write!(f, "{:?}", self),
        }
    }
}

impl std::error::Error for GraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // TODO: For errors that wrap other errors (like IoError), return the underlying error
        None
    }
}

/*
=== UTILITY FUNCTIONS ===
*/

/// Convert a standard I/O error to a GraphError
pub fn io_error_to_graph_error(err: std::io::Error, operation: &str, path: &str) -> GraphError {
    GraphError::IoError {
        operation: operation.to_string(),
        path: path.to_string(),
        underlying_error: err.to_string(),
    }
}

/// Create an internal error with file and line information
/// Usage: internal_error!("Something unexpected happened")
#[macro_export]
macro_rules! internal_error {
    ($message:expr) => {
        GraphError::internal($message, &format!("{}:{}", file!(), line!()))
    };
    ($message:expr, $($context_key:expr => $context_value:expr),+) => {
        {
            let mut context = std::collections::HashMap::new();
            $(
                context.insert($context_key.to_string(), $context_value.to_string());
            )+
            GraphError::internal_with_context($message, &format!("{}:{}", file!(), line!()), context)
        }
    };
}

/// Create a user-friendly error result
/// Usage: user_error!(NodeNotFound, node_id, "get neighbors")
#[macro_export]
macro_rules! user_error {
    (NodeNotFound, $node_id:expr, $operation:expr) => {
        Err(GraphError::node_not_found($node_id, $operation))
    };
    (EdgeNotFound, $edge_id:expr, $operation:expr) => {
        Err(GraphError::edge_not_found($edge_id, $operation))
    }; // TODO: Add macros for other common error types
}

/*
=== IMPLEMENTATION NOTES ===

ERROR DESIGN PRINCIPLES:
1. Every error should provide actionable information
2. Include context about what operation was being attempted
3. Suggest concrete steps for resolution when possible
4. Distinguish between user errors and system errors
5. Make errors easy to match on for programmatic handling

PERFORMANCE CONSIDERATIONS:
- Errors should be cheap to create (avoid expensive string formatting until display)
- Use Cow<str> for strings that might be static
- Consider boxing large error variants to keep Result size small

EXTENSIBILITY:
- Easy to add new error variants without breaking existing code
- Context fields allow adding information without new variants
- Error categories help with high-level error handling

INTEGRATION POINTS:
- All Result types in the codebase should use GraphResult<T>
- Error creation should use the convenience methods when possible
- Logging systems should use short_description() for structured logs
- User interfaces should use Display trait for error messages

TESTING STRATEGY:
- Test that errors contain expected information
- Test error categorization and recovery hints
- Test error formatting for user-friendliness
- Test that internal errors include sufficient debugging context
*/
