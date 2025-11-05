"""
IR Node type system for algorithm optimization.

Defines a typed, domain-aware intermediate representation that replaces
the simple JSON step list with a structured graph suitable for analysis
and optimization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class IRDomain(Enum):
    """Domain classification for IR nodes."""
    CORE = "core"           # Arithmetic, reductions, conditionals
    GRAPH = "graph"         # Topology operations, neighbor aggregation
    ATTR = "attr"           # Attribute load/store
    CONTROL = "control"     # Loops, convergence checks
    UNKNOWN = "unknown"


@dataclass
class IRNode(ABC):
    """
    Base class for all IR nodes.
    
    Each node represents a single operation in the algorithm with:
    - Domain classification (core, graph, attr, control)
    - Operation type within that domain
    - Input and output variable names
    - Metadata for optimization and code generation
    """
    
    # Unique identifier for this node
    id: str
    
    # Domain this operation belongs to
    domain: IRDomain
    
    # Operation type within the domain (e.g., "add", "neighbor_agg")
    op_type: str
    
    # Input variable names
    inputs: List[str] = field(default_factory=list)
    
    # Output variable name
    output: Optional[str] = None
    
    # Additional metadata (parameters, options, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary for JSON/FFI serialization."""
        pass
    
    @abstractmethod
    def to_step(self) -> Dict[str, Any]:
        """Convert to legacy step format for backward compatibility."""
        pass
    
    def __repr__(self) -> str:
        """Pretty representation for debugging."""
        inputs_str = ", ".join(self.inputs) if self.inputs else ""
        output_str = f" -> {self.output}" if self.output else ""
        return f"{self.domain.value}.{self.op_type}({inputs_str}){output_str}"


@dataclass
class CoreIRNode(IRNode):
    """
    IR node for core arithmetic/scalar operations.
    
    Covers:
    - Binary arithmetic: add, sub, mul, div, pow
    - Unary operations: neg, recip, sqrt, abs
    - Reductions: sum, mean, min, max, norm
    - Conditionals: compare, where
    - Broadcasting and scalar operations
    """
    
    def __init__(self, node_id: str, op_type: str, inputs: List[str], 
                 output: str, **metadata):
        super().__init__(
            id=node_id,
            domain=IRDomain.CORE,
            op_type=op_type,
            inputs=inputs,
            output=output,
            metadata=metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain.value,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "output": self.output,
            "metadata": self.metadata
        }
    
    def to_step(self) -> Dict[str, Any]:
        """Convert to legacy step format."""
        # Handle special case: init_nodes_unique -> init_nodes_with_index
        if self.op_type == "init_nodes_unique":
            return {
                "type": "init_nodes_with_index",
                "output": self.output
            }
        
        step = {
            "type": f"core.{self.op_type}",
            "output": self.output
        }
        
        # Map inputs to legacy parameter names
        if self.op_type in ["add", "sub", "mul", "div", "pow"]:
            if len(self.inputs) >= 1:
                step["a"] = self.inputs[0]
            if len(self.inputs) >= 2:
                step["b"] = self.inputs[1]
        elif self.op_type in ["neg", "recip", "sqrt", "abs"]:
            step["input"] = self.inputs[0]
        elif self.op_type in ["sum", "mean", "min", "max"]:
            step["input"] = self.inputs[0]
        elif self.op_type == "where":
            step["mask"] = self.inputs[0]
            step["if_true"] = self.inputs[1] if len(self.inputs) > 1 else self.metadata.get("if_true", 0.0)
            step["if_false"] = self.inputs[2] if len(self.inputs) > 2 else self.metadata.get("if_false", 0.0)
        elif self.op_type == "compare":
            step["a"] = self.inputs[0]
            step["b"] = self.inputs[1] if len(self.inputs) > 1 else self.metadata.get("b", 0.0)
            step["op"] = self.metadata.get("op", "eq")
        
        # Add any additional metadata
        for key, value in self.metadata.items():
            if key not in step:
                step[key] = value
        
        return step


@dataclass
class GraphIRNode(IRNode):
    """
    IR node for graph topology operations.
    
    Covers:
    - Structural queries: degree, neighbors, subgraph
    - Aggregation: neighbor_agg (sum, mean, min, max)
    - Traversal: BFS, DFS, shortest paths
    - Normalization: normalize adjacency matrix
    """
    
    def __init__(self, node_id: str, op_type: str, inputs: List[str],
                 output: str, **metadata):
        super().__init__(
            id=node_id,
            domain=IRDomain.GRAPH,
            op_type=op_type,
            inputs=inputs,
            output=output,
            metadata=metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain.value,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "output": self.output,
            "metadata": self.metadata
        }
    
    def to_step(self) -> Dict[str, Any]:
        """Convert to legacy step format."""
        step = {
            "type": f"graph.{self.op_type}",
            "output": self.output
        }
        
        # Map inputs to legacy parameter names
        if self.op_type == "degree":
            # No inputs typically
            pass
        elif self.op_type == "neighbor_agg":
            if len(self.inputs) >= 1:
                step["source"] = self.inputs[0]
            step["agg"] = self.metadata.get("agg", "sum")
            if "weights" in self.metadata:
                step["weights"] = self.metadata["weights"]
        elif self.op_type == "subgraph":
            step["mask"] = self.inputs[0]
        
        # Add any additional metadata
        for key, value in self.metadata.items():
            if key not in step:
                step[key] = value
        
        return step


@dataclass
class AttrIRNode(IRNode):
    """
    IR node for attribute operations.
    
    Covers:
    - Loading: load attribute by name
    - Storing: attach computed values as attributes
    - Aggregation: groupby, join operations
    """
    
    def __init__(self, node_id: str, op_type: str, inputs: List[str],
                 output: Optional[str], **metadata):
        super().__init__(
            id=node_id,
            domain=IRDomain.ATTR,
            op_type=op_type,
            inputs=inputs,
            output=output,
            metadata=metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain.value,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "output": self.output,
            "metadata": self.metadata
        }
    
    def to_step(self) -> Dict[str, Any]:
        """Convert to legacy step format."""
        step = {
            "type": f"attr.{self.op_type}",
        }
        
        if self.output:
            step["output"] = self.output
        
        # Map specific operations
        if self.op_type == "load":
            step["name"] = self.metadata.get("name", "")
            step["default"] = self.metadata.get("default", 0.0)
        elif self.op_type == "attach":
            step["name"] = self.metadata.get("name", "")
            if len(self.inputs) >= 1:
                step["source"] = self.inputs[0]
        
        # Add any additional metadata
        for key, value in self.metadata.items():
            if key not in step:
                step[key] = value
        
        return step


@dataclass
class ControlIRNode(IRNode):
    """
    IR node for control flow operations.
    
    Covers:
    - Loops: fixed iteration, convergence-based
    - Conditionals: if/else branches
    - Barriers: synchronization points
    - Scope: variable lifetime management
    """
    
    def __init__(self, node_id: str, op_type: str, inputs: List[str],
                 output: Optional[str], **metadata):
        super().__init__(
            id=node_id,
            domain=IRDomain.CONTROL,
            op_type=op_type,
            inputs=inputs,
            output=output,
            metadata=metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain.value,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "output": self.output,
            "metadata": self.metadata
        }
    
    def to_step(self) -> Dict[str, Any]:
        """Convert to legacy step format."""
        step = {
            "type": f"control.{self.op_type}",
        }
        
        if self.output:
            step["output"] = self.output
        
        # Map specific operations
        if self.op_type == "loop":
            step["count"] = self.metadata.get("count", 1)
            step["body"] = self.metadata.get("body", [])
        elif self.op_type == "until_converged":
            step["tol"] = self.metadata.get("tol", 1e-6)
            step["max_iter"] = self.metadata.get("max_iter", 100)
            step["body"] = self.metadata.get("body", [])
        
        # Add any additional metadata
        for key, value in self.metadata.items():
            if key not in step:
                step[key] = value
        
        return step


# Convenience type for any IR node
AnyIRNode = Union[CoreIRNode, GraphIRNode, AttrIRNode, ControlIRNode]
