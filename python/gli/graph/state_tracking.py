"""
State tracking module for GLI.

This module provides functionality to track changes over states for:
- Graphs
- Subgraphs  
- Nodes
- Edges

This enables version control-like capabilities for graph data structures,
allowing users to track how their graphs evolve through different states.
"""

from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict


class ChangeType(Enum):
    """Types of changes that can be tracked."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


@dataclass
class Change:
    """Represents a single change to a graph element."""
    element_type: str  # 'node', 'edge', 'graph', 'subgraph'
    element_id: str
    change_type: ChangeType
    timestamp: float
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class State:
    """Represents a state of a graph or subgraph."""
    state_id: str
    parent_state_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    changes: List[Change] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_change(self, change: Change):
        """Add a change to this state."""
        self.changes.append(change)


class StateTracker:
    """
    Tracks changes over states for graph elements.
    
    This class provides the core functionality for state management,
    allowing users to:
    - Create new states
    - Track changes between states
    - Query historical changes
    - Navigate between states
    """
    
    def __init__(self):
        """Initialize a new StateTracker."""
        self.states: Dict[str, State] = {}
        self.current_state_id: Optional[str] = None
        self.state_counter = 0
        
        # Indexes for efficient querying
        self.element_changes: Dict[str, List[Change]] = defaultdict(list)
        self.state_tree: Dict[str, List[str]] = defaultdict(list)  # parent -> children
    
    def create_state(self, parent_state_id: Optional[str] = None, 
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new state.
        
        Args:
            parent_state_id: ID of the parent state (None for root state)
            metadata: Additional metadata for this state
            
        Returns:
            The ID of the newly created state
        """
        self.state_counter += 1
        state_id = f"state_{self.state_counter}"
        
        state = State(
            state_id=state_id,
            parent_state_id=parent_state_id,
            metadata=metadata or {}
        )
        
        self.states[state_id] = state
        
        if parent_state_id:
            self.state_tree[parent_state_id].append(state_id)
        
        self.current_state_id = state_id
        return state_id
    
    def record_change(self, element_type: str, element_id: str, 
                     change_type: ChangeType, old_value: Any = None, 
                     new_value: Any = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a change to an element.
        
        Args:
            element_type: Type of element ('node', 'edge', 'graph', 'subgraph')
            element_id: ID of the element that changed
            change_type: Type of change (added, removed, modified)
            old_value: Previous value (for modifications)
            new_value: New value (for additions and modifications)
            metadata: Additional metadata about this change
        """
        if not self.current_state_id:
            self.create_state()
        
        change = Change(
            element_type=element_type,
            element_id=element_id,
            change_type=change_type,
            timestamp=time.time(),
            old_value=old_value,
            new_value=new_value,
            metadata=metadata or {}
        )
        
        self.states[self.current_state_id].add_change(change)
        self.element_changes[element_id].append(change)
    
    def get_state(self, state_id: str) -> Optional[State]:
        """Get a state by ID."""
        return self.states.get(state_id)
    
    def get_current_state(self) -> Optional[State]:
        """Get the current state."""
        if self.current_state_id:
            return self.states.get(self.current_state_id)
        return None
    
    def set_current_state(self, state_id: str):
        """Set the current state."""
        if state_id in self.states:
            self.current_state_id = state_id
        else:
            raise ValueError(f"State {state_id} not found")
    
    def get_element_history(self, element_id: str) -> List[Change]:
        """Get the complete change history for an element."""
        return self.element_changes[element_id].copy()
    
    def get_changes_between_states(self, from_state_id: str, 
                                  to_state_id: str) -> List[Change]:
        """Get all changes between two states."""
        # This is a simplified implementation
        # A more sophisticated version would traverse the state tree
        changes = []
        
        from_state = self.states.get(from_state_id)
        to_state = self.states.get(to_state_id)
        
        if not from_state or not to_state:
            return changes
        
        # For now, just collect all changes in states between the timestamps
        for state in self.states.values():
            if from_state.timestamp < state.timestamp <= to_state.timestamp:
                changes.extend(state.changes)
        
        return changes
    
    def get_state_children(self, state_id: str) -> List[str]:
        """Get the child states of a given state."""
        return self.state_tree[state_id].copy()
    
    def get_state_lineage(self, state_id: str) -> List[str]:
        """Get the lineage (path from root) to a given state."""
        lineage = []
        current_id = state_id
        
        while current_id:
            lineage.insert(0, current_id)
            state = self.states.get(current_id)
            if state:
                current_id = state.parent_state_id
            else:
                break
        
        return lineage
    
    def query_changes(self, element_type: Optional[str] = None,
                     change_type: Optional[ChangeType] = None,
                     element_id: Optional[str] = None,
                     from_timestamp: Optional[float] = None,
                     to_timestamp: Optional[float] = None) -> List[Change]:
        """
        Query changes with various filters.
        
        Args:
            element_type: Filter by element type
            change_type: Filter by change type
            element_id: Filter by element ID
            from_timestamp: Filter changes after this timestamp
            to_timestamp: Filter changes before this timestamp
            
        Returns:
            List of changes matching the filters
        """
        results = []
        
        # If element_id is specified, use the index for efficiency
        if element_id:
            candidates = self.element_changes[element_id]
        else:
            # Otherwise, check all changes in all states
            candidates = []
            for state in self.states.values():
                candidates.extend(state.changes)
        
        for change in candidates:
            # Apply filters
            if element_type and change.element_type != element_type:
                continue
            if change_type and change.change_type != change_type:
                continue
            if from_timestamp and change.timestamp < from_timestamp:
                continue
            if to_timestamp and change.timestamp > to_timestamp:
                continue
                
            results.append(change)
        
        return results


# Global state tracker instance
_global_state_tracker = StateTracker()


def get_state_tracker() -> StateTracker:
    """Get the global state tracker instance."""
    return _global_state_tracker


def create_state(parent_state_id: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to create a new state."""
    return _global_state_tracker.create_state(parent_state_id, metadata)


def record_change(element_type: str, element_id: str, 
                 change_type: ChangeType, old_value: Any = None,
                 new_value: Any = None, metadata: Optional[Dict[str, Any]] = None):
    """Convenience function to record a change."""
    _global_state_tracker.record_change(
        element_type, element_id, change_type, old_value, new_value, metadata
    )
