"""
Algorithm Builder DSL for composing custom algorithms from steps.

This module provides a high-level interface for building custom algorithms
by composing pre-registered steps, without requiring Rust compilation.

Example:
    >>> from groggy.builder import AlgorithmBuilder
    >>> 
    >>> builder = AlgorithmBuilder("my_algorithm")
    >>> nodes = builder.init_nodes(default=0.0)
    >>> degrees = builder.node_degrees(nodes)
    >>> normalized = builder.normalize(degrees)
    >>> builder.attach_as("degree_normalized", normalized)
    >>> 
    >>> # Use the built algorithm
    >>> algo = builder.build()
    >>> result = subgraph.apply(algo)

Custom algorithms built with this builder execute via the Rust step
interpreter registered as `builder.step_pipeline`.
"""

from typing import Any, Dict, Optional
from groggy import _groggy
from groggy.algorithms.base import AlgorithmHandle


class VarHandle:
    """
    Handle representing a variable in the algorithm builder.
    
    Variables track intermediate results as the algorithm is built.
    """
    
    def __init__(self, name: str, builder: 'AlgorithmBuilder'):
        """
        Create a variable handle.
        
        Args:
            name: Variable name
            builder: Parent builder
        """
        self.name = name
        self.builder = builder
    
    def __repr__(self) -> str:
        return f"VarHandle('{self.name}')"


class AlgorithmBuilder:
    """Builder for composing custom algorithms from step primitives."""
    
    def __init__(self, name: str):
        """
        Create a new algorithm builder.
        
        Args:
            name: Name for the algorithm
        """
        self.name = name
        self.steps = []
        self.variables = {}
        self._var_counter = 0
    
    def _new_var(self, prefix: str = "var") -> VarHandle:
        """Create a new variable handle."""
        var_name = f"{prefix}_{self._var_counter}"
        self._var_counter += 1
        handle = VarHandle(var_name, self)
        self.variables[var_name] = handle
        return handle
    
    def init_nodes(self, default: Any = 0.0) -> VarHandle:
        """
        Initialize node values.
        
        Args:
            default: Default value for all nodes
            
        Returns:
            Variable handle for initialized values
        """
        var = self._new_var("nodes")
        self.steps.append({
            "type": "init_nodes",
            "output": var.name,
            "default": default
        })
        return var
    
    def node_degrees(self, nodes: VarHandle) -> VarHandle:
        """
        Compute node degrees.
        
        Args:
            nodes: Input node variable
            
        Returns:
            Variable handle for degrees
        """
        var = self._new_var("degrees")
        self.steps.append({
            "type": "node_degree",
            "input": nodes.name,
            "output": var.name
        })
        return var
    
    def normalize(self, values: VarHandle, method: str = "sum") -> VarHandle:
        """
        Normalize values.
        
        Args:
            values: Input values
            method: Normalization method ("sum", "max", "minmax")
            
        Returns:
            Variable handle for normalized values
        """
        var = self._new_var("normalized")
        self.steps.append({
            "type": "normalize",
            "input": values.name,
            "output": var.name,
            "method": method
        })
        return var
    
    def attach_as(self, attr_name: str, values: VarHandle):
        """
        Attach values as a node attribute.
        
        Args:
            attr_name: Name for the output attribute
            values: Values to attach
        """
        self.steps.append({
            "type": "attach_attr",
            "input": values.name,
            "attr_name": attr_name
        })
    
    def build(self) -> 'BuiltAlgorithm':
        """
        Build the algorithm from the composed steps.
        
        Returns:
            Built algorithm ready for use
        """
        return BuiltAlgorithm(self.name, self.steps)


class BuiltAlgorithm(AlgorithmHandle):
    """Algorithm built from composed steps."""
    
    def __init__(self, name: str, steps: list):
        """
        Create a built algorithm.
        
        Args:
            name: Algorithm name
            steps: List of step specifications
        """
        self._id = f"custom.{name}"
        self._name = name
        self._steps = steps
    
    @property
    def id(self) -> str:
        """Get the algorithm identifier."""
        return self._id
    
    def to_spec(self) -> Dict[str, Any]:
        """Convert to a pipeline spec entry usable by the Rust executor."""

        pipeline_definition = {
            "name": self._name,
            "steps": [self._encode_step(step) for step in self._steps],
        }

        return {
            "id": "builder.step_pipeline",
            "params": {
                "name": _groggy.AttrValue(self._name),
                "steps": _groggy.AttrValue(pipeline_definition),
            },
        }

    def _encode_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        step_type = step.get("type")

        if step_type == "init_nodes":
            params: Dict[str, Any] = {"target": step["output"]}
            default = step.get("default")
            if default is not None:
                params["value"] = default
            return {"id": "core.init_nodes", "params": params}

        if step_type == "node_degree":
            return {
                "id": "core.node_degree",
                "params": {"target": step["output"]},
            }

        if step_type == "normalize":
            params = {
                "source": step["input"],
                "target": step["output"],
                "method": step.get("method", "sum"),
                "epsilon": 1e-9,
            }
            return {"id": "core.normalize_node_values", "params": params}

        if step_type == "attach_attr":
            return {
                "id": "core.attach_node_attr",
                "params": {"source": step["input"], "attr": step["attr_name"]},
            }

        raise ValueError(f"Unsupported builder step type: {step_type}")
    
    def __repr__(self) -> str:
        return f"BuiltAlgorithm('{self._name}', {len(self._steps)} steps)"


def builder(name: str) -> AlgorithmBuilder:
    """
    Create a new algorithm builder.
    
    Args:
        name: Name for the algorithm
        
    Returns:
        Algorithm builder instance
        
    Example:
        >>> from groggy import builder
        >>> b = builder("my_algo")
        >>> nodes = b.init_nodes()
        >>> degrees = b.node_degrees(nodes)
        >>> b.attach_as("degree", degrees)
        >>> algo = b.build()
    """
    return AlgorithmBuilder(name)
