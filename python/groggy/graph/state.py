"""
State management methods for Graph class
"""

from typing import Dict, List, Any, Optional


class StateMixin:
    """Mixin class providing state management functionality"""
    
    def save_state(self, message: str = None) -> str:
        """
        Save the current graph state to storage
        
        Args:
            message: Optional message describing the changes
            
        Returns:
            Hash of the saved state
        """
        if self.use_rust:
            commit_msg = message or f"state_{len(self._rust_store.list_branches())}"
            state_hash = self._rust_store.store_current_graph(self._rust_core, commit_msg)
            
            # Track this commit in auto_states
            if hasattr(self, 'auto_states') and state_hash not in self.auto_states:
                self.auto_states.append(state_hash)
            
            # Update current_hash
            self.current_hash = state_hash
            
            return state_hash
        else:
            raise NotImplementedError("State saving only supported with Rust backend")
    
    # Backward compatibility alias
    def commit(self, message: str = None) -> str:
        """Legacy alias for save_state"""
        return self.save_state(message)
    
    def create_branch(self, branch_name: str, from_hash: str = None, switch=False) -> str:
        """Create a new branch (delegated to Rust)"""
        if self.use_rust:
            self._rust_store.create_branch(branch_name, from_hash)
            if switch:
                self.switch_branch(branch_name)
            return branch_name
        else:
            raise NotImplementedError("Branching only supported with Rust backend")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics (delegated to Rust)"""
        if self.use_rust:
            return self._rust_store.get_stats()
        else:
            return {"backend": "python_fallback"}
    
    def load_state(self, state_hash: str) -> bool:
        """
        Load a previous state of the graph
        
        Args:
            state_hash: Hash of the state to load
            
        Returns:
            True if successful, False otherwise
        """
        if self.use_rust:
            try:
                # Get the graph state from the store
                restored_graph = self._rust_store.get_graph_from_state(state_hash)
                
                # Replace the current graph core with the restored one
                self._rust_core = restored_graph
                
                # Update current hash from the backend and invalidate cache
                self.current_hash = self._rust_store.get_current_hash()
                self._invalidate_cache()
                
                return True
            except Exception as e:
                print(f"Error loading state {state_hash}: {e}")
                return False
        else:
            raise NotImplementedError("State loading only supported with Rust backend")
    
    def get_state_info(self, state_hash: str = None) -> Dict[str, Any]:
        """Get detailed information about a specific state or current state
        
        Args:
            state_hash: Hash of the state to inspect (None for current state)
            
        Returns:
            Dictionary with state information
        """
        if self.use_rust:
            if state_hash is None:
                state_hash = self._rust_store.get_current_hash()
            
            # Find which branches point to this state
            branches_for_state = []
            for branch_name, branch_hash in self.branches.items():
                if branch_hash == state_hash:
                    branches_for_state.append(branch_name)
            
            return {
                'hash': state_hash,
                'is_current': state_hash == self._rust_store.get_current_hash(),
                'branches': branches_for_state,
                'in_auto_states': state_hash in self.auto_states if hasattr(self, 'auto_states') else False
            }
        else:
            raise NotImplementedError("State info only supported with Rust backend")
    
    def switch_branch(self, branch_name: str) -> bool:
        """
        Switch to a different branch
        
        Args:
            branch_name: Name of the branch to switch to
            
        Returns:
            True if successful, False otherwise
        """
        if self.use_rust:
            # Check if branch exists
            branches = self.branches
            if branch_name not in branches:
                print(f"Branch '{branch_name}' does not exist. Available branches: {list(branches.keys())}")
                return False
            
            # Get the hash for this branch
            branch_hash = branches[branch_name]
            
            # Load the state for this branch
            success = self.load_state(branch_hash)
            if success:
                self.current_branch = branch_name
                print(f"Switched to branch '{branch_name}' (state: {branch_hash})")
                return True
            else:
                print(f"Failed to switch to branch '{branch_name}'")
                return False
        else:
            raise NotImplementedError("Branch switching only supported with Rust backend")
