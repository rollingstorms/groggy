"""
Graph state management
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from .delta import CompactGraphDelta


@dataclass
class GraphState:
    """Optimized graph state using deltas and content addressing"""
    hash: str
    parent_hash: Optional[str] = None
    operation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    branch_name: Optional[str] = None  # Track which branch this state belongs to
    
    # For root states: full content
    nodes: Optional[Dict[str, str]] = None  # node_id -> content_hash
    edges: Optional[Dict[str, str]] = None  # edge_id -> content_hash
    graph_attributes: Optional[Dict[str, Any]] = None
    
    # For incremental states: delta only
    delta: Optional[CompactGraphDelta] = None
    
    # Subgraph support
    is_subgraph_state: bool = False
    subgraph_metadata: Optional[Dict[str, Any]] = None
    
    def is_root(self) -> bool:
        return self.parent_hash is None
