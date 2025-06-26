"""
Graph store implementation with Rust backend support
"""

import time
import weakref
from collections import deque
from typing import Dict, List, Any, Optional
from .graph import Graph
from .delta import CompactGraphDelta

# Import detection for backend
try:
    from . import _core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class Branch:
    """Represents a named branch in the graph store"""
    
    def __init__(self, name: str, current_hash: str, created_from: str, 
                 description: str = "", is_subgraph: bool = False):
        self.name = name
        self.current_hash = current_hash
        self.created_from = created_from
        self.created_at = time.time()
        self.description = description
        self.is_subgraph = is_subgraph


class GraphStore:
    """High-performance Graph Store with Rust backend support"""
    
    def __init__(self, use_rust=None, max_auto_states: int = 10):
        # Backend selection
        self.use_rust = (use_rust if use_rust is not None else RUST_AVAILABLE)
        
        if self.use_rust:
            self._rust_store = _core.GraphStore()
            self.current_hash = None
        else:
            # Python fallback implementation
            self.states = {}  # hash -> GraphState
            self.auto_states = deque(maxlen=max_auto_states)
            self.commits = {}  # hash -> commit info
            self.current_hash = None
        
        self.current_graph = Graph.empty(self, use_rust=self.use_rust)
        
        # Branching support
        self.branches = {}  # branch_name -> Branch
        self.current_branch = "main"
        self.branch_heads = {}  # branch_name -> current_hash
        
        # Caching
        self._reconstructed_cache = {}  # hash -> reconstructed Graph (weak refs)
        
        # Initialize with empty state
        self._create_initial_state()
    
    def _create_initial_state(self):
        """Create initial empty state"""
        if self.use_rust:
            # Rust backend handles initialization
            initial_hash = self._rust_store.store_graph(self.current_graph._rust_core, "initialize")
            self.current_hash = initial_hash
        else:
            # Python implementation
            from .state import GraphState
            initial_state = GraphState(
                hash="initial",
                nodes={},
                edges={},
                graph_attributes={},
                operation="initialize"
            )
            self.states["initial"] = initial_state
            self.current_hash = "initial"
            self.auto_states.append("initial")
        
        # Initialize main branch
        self.branches["main"] = Branch(
            name="main",
            current_hash=self.current_hash,
            created_from=self.current_hash,
            description="Main branch"
        )
        self.branch_heads["main"] = self.current_hash
    
    def get_current_graph(self) -> Graph:
        """Get current graph state"""
        if self.use_rust:
            # Reconstruct from Rust store
            if self.current_hash:
                rust_graph = self._rust_store.reconstruct_graph(self.current_hash)
                if rust_graph:
                    graph = Graph.empty(self, use_rust=True)
                    graph._rust_core = rust_graph
                    return graph
            return self.current_graph
        else:
            # Python implementation
            if self.current_hash in self.states:
                return self._reconstruct_graph_from_state(self.current_hash)
            return self.current_graph
    
    def update_graph(self, new_graph: Graph, operation: str = "update") -> str:
        """Update current graph state"""
        if self.use_rust:
            # Use Rust backend
            new_hash = self._rust_store.store_graph(new_graph._rust_core, operation)
            self.current_hash = new_hash
            self.current_graph = new_graph
        else:
            # Python implementation
            snapshot_graph = new_graph.snapshot()
            new_hash = self._create_snapshot_state(snapshot_graph, operation)
            self.current_hash = new_hash
            self.auto_states.append(new_hash)
            self.current_graph = snapshot_graph
        
        # Update branch head
        if self.current_branch in self.branches:
            self.branches[self.current_branch].current_hash = new_hash
            self.branch_heads[self.current_branch] = new_hash
        
        return new_hash
    
    def create_branch(self, branch_name: str, from_hash: str = None, 
                     description: str = "") -> str:
        """Create a new branch"""
        if branch_name in self.branches:
            raise ValueError(f"Branch '{branch_name}' already exists")
        
        base_hash = from_hash or self.current_hash
        
        if self.use_rust:
            # Use Rust backend
            self._rust_store.create_branch(branch_name, base_hash)
        
        # Create branch object
        branch = Branch(
            name=branch_name,
            current_hash=base_hash,
            created_from=base_hash,
            description=description
        )
        
        self.branches[branch_name] = branch
        self.branch_heads[branch_name] = base_hash
        
        return branch_name
    
    def switch_branch(self, branch_name: str) -> Graph:
        """Switch to specified branch"""
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' does not exist")
        
        old_branch = self.current_branch
        self.current_branch = branch_name
        branch = self.branches[branch_name]
        
        # Update current state
        self.current_hash = branch.current_hash
        self.current_graph = self.get_current_graph()
        self.current_graph.branch_name = branch_name
        
        return self.current_graph
    
    def list_branches(self) -> List[Dict[str, Any]]:
        """List all branches with metadata"""
        branches = []
        for name, branch in self.branches.items():
            branches.append({
                'name': name,
                'current': name == self.current_branch,
                'hash': branch.current_hash,
                'created_from': branch.created_from,
                'created_at': branch.created_at,
                'description': branch.description,
                'is_subgraph': branch.is_subgraph,
            })
        return branches
    
    def commit(self, message: str = "") -> str:
        """Explicitly commit current state"""
        if self.current_hash:
            if not self.use_rust:
                self.commits[self.current_hash] = {
                    'message': message,
                    'timestamp': time.time()
                }
        return self.current_hash
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about storage usage"""
        if self.use_rust:
            return self._rust_store.get_stats()
        else:
            return {
                'total_states': len(self.states),
                'cached_reconstructions': len(self._reconstructed_cache),
                'auto_states_length': len(self.auto_states),
                'committed_states': len(self.commits)
            }
    
    # Python backend methods (fallback implementation)
    def _create_snapshot_state(self, graph: Graph, operation: str) -> str:
        """Create full snapshot state (Python backend)"""
        # This would be the full Python implementation
        # For now, simplified version
        import hashlib
        import json
        
        state_data = {
            'nodes': {nid: {'id': node.id, 'attributes': node.attributes} 
                     for nid, node in graph.nodes.items()},
            'edges': {eid: {'source': edge.source, 'target': edge.target, 
                           'attributes': edge.attributes} for eid, edge in graph.edges.items()},
            'graph_attributes': graph.graph_attributes
        }
        
        content = json.dumps(state_data, sort_keys=True)
        state_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        from .state import GraphState
        state = GraphState(
            hash=state_hash,
            nodes={},  # Would store content hashes
            edges={},  # Would store content hashes  
            graph_attributes=graph.graph_attributes.copy(),
            operation=operation
        )
        
        self.states[state_hash] = state
        return state_hash
    
    def _reconstruct_graph_from_state(self, state_hash: str) -> Graph:
        """Reconstruct graph from state (Python backend)"""
        # Check cache first
        if state_hash in self._reconstructed_cache:
            cached_ref = self._reconstructed_cache[state_hash]()
            if cached_ref is not None:
                return cached_ref
        
        # For now, return empty graph (would implement full reconstruction)
        graph = Graph.empty(self, use_rust=False)
        
        # Cache with weak reference
        self._reconstructed_cache[state_hash] = weakref.ref(graph)
        return graph
