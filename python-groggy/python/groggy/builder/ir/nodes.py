"""
IR Node type system for algorithm optimization.

Defines a typed, domain-aware intermediate representation that replaces
the simple JSON step list with a structured graph suitable for analysis
and optimization.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class IRDomain(Enum):
    """Domain classification for IR nodes."""

    CORE = "core"  # Arithmetic, reductions, conditionals
    GRAPH = "graph"  # Topology operations, neighbor aggregation
    ATTR = "attr"  # Attribute load/store
    CONTROL = "control"  # Loops, convergence checks
    EXECUTION = "execution"  # Execution blocks (message-pass, streaming)
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

    def __init__(
        self, node_id: str, op_type: str, inputs: List[str], output: str, **metadata
    ):
        super().__init__(
            id=node_id,
            domain=IRDomain.CORE,
            op_type=op_type,
            inputs=inputs,
            output=output,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain.value,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "output": self.output,
            "metadata": self.metadata,
        }

    def to_step(self) -> Dict[str, Any]:
        """Convert to legacy step format."""
        # Handle special case: init_nodes_unique -> init_nodes_with_index
        if self.op_type == "init_nodes_unique":
            return {"type": "init_nodes_with_index", "output": self.output}

        step = {"type": f"core.{self.op_type}"}

        # Fused operations use "target" instead of "output"
        if "fused" not in self.op_type:
            step["output"] = self.output

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
            step["if_true"] = (
                self.inputs[1]
                if len(self.inputs) > 1
                else self.metadata.get("if_true", 0.0)
            )
            step["if_false"] = (
                self.inputs[2]
                if len(self.inputs) > 2
                else self.metadata.get("if_false", 0.0)
            )
        elif self.op_type == "compare":
            step["a"] = self.inputs[0]
            step["b"] = (
                self.inputs[1] if len(self.inputs) > 1 else self.metadata.get("b", 0.0)
            )
            step["op"] = self.metadata.get("op", "eq")
        elif self.op_type == "reduce_scalar":
            # Reduce scalar operations expect a single source input
            if self.inputs:
                step["source"] = self.inputs[0]
            step["op"] = self.metadata.get("op", "sum")
        elif self.op_type == "normalize_sum":
            if self.inputs:
                step["input"] = self.inputs[0]
        elif self.op_type == "collect_neighbor_values":
            if self.inputs:
                step["source"] = self.inputs[0]
            step["include_self"] = self.metadata.get("include_self", True)
        elif self.op_type == "mode_list":
            if self.inputs:
                step["source"] = self.inputs[0]
            step["tie_break"] = self.metadata.get("tie_break", "lowest")
        elif self.op_type == "histogram":
            if self.inputs:
                step["source"] = self.inputs[0]
            step["bins"] = self.metadata.get("bins", 10)
        elif self.op_type == "map_nodes":
            step["type"] = "map_nodes"
            step["fn"] = self.metadata.get("fn", "")
            step["inputs"] = self.metadata.get("map_inputs", {})
            step["async_update"] = self.metadata.get("async_update", False)
        elif self.op_type == "fused_axpy":
            # Fused AXPY: result = a * x + b * y
            # Rust signature: FusedAXPY(a, x, b, y, target)
            fused_inputs = self.metadata.get("fused_inputs")
            if fused_inputs:
                step["a"] = fused_inputs.get("a")
                step["x"] = fused_inputs.get("x")
                step["b"] = fused_inputs.get("b")
                step["y"] = fused_inputs.get("y")
            elif len(self.inputs) >= 4:
                step["a"] = self.inputs[0]
                step["x"] = self.inputs[1]
                step["b"] = self.inputs[2]
                step["y"] = self.inputs[3]
            step["target"] = self.output
        elif self.op_type == "fused_madd":
            # Fused MADD: result = a * b + c
            # Rust signature: FusedMADD(a, b, c, target)
            fused_inputs = self.metadata.get("fused_inputs")
            if fused_inputs:
                step["a"] = fused_inputs.get("a")
                step["b"] = fused_inputs.get("b")
                step["c"] = fused_inputs.get("c")
            elif len(self.inputs) >= 3:
                step["a"] = self.inputs[0]
                step["b"] = self.inputs[1]
                step["c"] = self.inputs[2]
            step["target"] = self.output
        elif self.op_type == "broadcast_scalar":
            # Broadcast scalar: inputs = [scalar, reference]
            if len(self.inputs) >= 2:
                step["scalar"] = self.inputs[0]
                step["reference"] = self.inputs[1]

        # Add any additional metadata
        for key, value in self.metadata.items():
            if key in {"fused_inputs", "map_inputs"}:
                continue  # internal bookkeeping only
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

    def __init__(
        self, node_id: str, op_type: str, inputs: List[str], output: str, **metadata
    ):
        super().__init__(
            id=node_id,
            domain=IRDomain.GRAPH,
            op_type=op_type,
            inputs=inputs,
            output=output,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain.value,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "output": self.output,
            "metadata": self.metadata,
        }

    def to_step(self) -> Dict[str, Any]:
        """Convert to legacy step format."""
        step = {"type": f"graph.{self.op_type}"}

        # Fused operations use "target" instead of "output"
        if "fused" not in self.op_type:
            step["output"] = self.output

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
            if "direction" in self.metadata:
                step["direction"] = self.metadata["direction"]
        elif self.op_type == "fused_neighbor_mul_agg":
            # Fused neighbor aggregation with multiplication
            # Rust signature: FusedNeighborMulAgg(values, scalars, target)
            if len(self.inputs) >= 2:
                step["values"] = self.inputs[0]
                step["scalars"] = self.inputs[1]
            step["target"] = self.output
            step["direction"] = self.metadata.get("direction", "in")
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

    def __init__(
        self,
        node_id: str,
        op_type: str,
        inputs: List[str],
        output: Optional[str],
        **metadata,
    ):
        super().__init__(
            id=node_id,
            domain=IRDomain.ATTR,
            op_type=op_type,
            inputs=inputs,
            output=output,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain.value,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "output": self.output,
            "metadata": self.metadata,
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

    def __init__(
        self,
        node_id: str,
        op_type: str,
        inputs: List[str],
        output: Optional[str],
        **metadata,
    ):
        super().__init__(
            id=node_id,
            domain=IRDomain.CONTROL,
            op_type=op_type,
            inputs=inputs,
            output=output,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain.value,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "output": self.output,
            "metadata": self.metadata,
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


@dataclass
class LoopIRNode(IRNode):
    """
    IR node for structured fixed-iteration loops.

    Stores the serialized body so that the builder can preserve the original
    operations without unrolling them across the IR or FFI boundary.
    """

    def __init__(
        self,
        node_id: str,
        iterations: int,
        body: List[Dict[str, Any]],
        loop_vars: Optional[List[str]] = None,
        batch_plan: Optional[Dict[str, Any]] = None,
    ):
        metadata = {
            "iterations": iterations,
            "body": body,
            "loop_vars": loop_vars or [],
        }
        if batch_plan is not None:
            metadata["batch_plan"] = batch_plan

        super().__init__(
            id=node_id,
            domain=IRDomain.CONTROL,
            op_type="iter_loop",
            inputs=[],
            output=None,
            metadata=metadata,
        )

    @property
    def iterations(self) -> int:
        return int(self.metadata.get("iterations", 1))

    @property
    def body(self) -> List[Dict[str, Any]]:
        return self.metadata.get("body", [])

    @property
    def loop_vars(self) -> List[str]:
        return self.metadata.get("loop_vars", [])

    @property
    def batch_plan(self) -> Optional[Dict[str, Any]]:
        return self.metadata.get("batch_plan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain.value,
            "op_type": self.op_type,
            "iterations": self.iterations,
            "body": deepcopy(self.body),
            "loop_vars": list(self.loop_vars),
        }

    def to_step(self) -> Dict[str, Any]:
        step = {
            "type": "iter.loop",
            "iterations": self.iterations,
            "body": deepcopy(self.body),
        }
        if self.loop_vars:
            step["loop_vars"] = list(self.loop_vars)
        if self.batch_plan is not None:
            step["batch_plan"] = deepcopy(self.batch_plan)
        return step

    def is_batch_compatible(self) -> bool:
        """
        Check if this loop body can be compiled to a batch plan.

        Returns True if all operations in the body are supported by
        the batch executor, False otherwise.

        Unsupported operations:
        - Nested loops
        - Execution blocks (TODO: may support in future)
        - Conditionals (if/where - TODO)
        - Complex graph operations (custom neighbor functions)
        """
        supported_ops = {
            # Core arithmetic
            "core.add",
            "core.sub",
            "core.mul",
            "core.div",
            "core.constant",
            # Graph operations
            "graph.neighbor_sum",
            "graph.neighbor_mean",
            "graph.neighbor_min",
            "graph.neighbor_max",
            # Loads/stores
            "init_nodes_with_index",
            "attach_attr",
            "alias",  # No-op in batch execution
        }

        for step in self.body:
            step_type = step.get("type", "")

            # Check for unsupported control flow
            if step_type.startswith("iter.") or step_type.startswith("control."):
                return False  # Nested loops not supported

            if step_type == "core.execution_block":
                return False  # Execution blocks not supported yet

            # Check if operation is in supported set
            if step_type not in supported_ops:
                # Unknown operation - not batch compatible
                return False

        return True


@dataclass
class ExecutionBlockNode(IRNode):
    """
    IR node for structured execution blocks (message-passing, streaming, etc.).

    Execution blocks encapsulate a sub-computation with specific semantics:
    - Message-Pass: Gauss-Seidel style neighbor updates with in-place writes
    - Streaming: Incremental/transactional updates (future)

    The block contains its own mini-DAG of operations that are executed
    with special semantics (e.g., ordered traversal, in-place updates).
    """

    def __init__(self, node_id: str, mode: str, target: str, **metadata):
        """
        Create an execution block node.

        Args:
            node_id: Unique identifier for this block
            mode: Execution mode ("message_pass", "streaming", etc.)
            target: Variable being updated by the block
            **metadata: Block configuration (include_self, ordered, tie_break, etc.)
        """
        super().__init__(
            id=node_id,
            domain=IRDomain.EXECUTION,
            op_type=f"block_{mode}",
            inputs=[],  # Inputs tracked in body nodes
            output=target,
            metadata={
                "mode": mode,
                "target": target,
                "body_nodes": [],  # List of IRNode dicts representing block body
                **metadata,
            },
        )

    @property
    def mode(self) -> str:
        """Get the execution mode of this block."""
        return self.metadata["mode"]

    @property
    def target(self) -> str:
        """Get the target variable being updated."""
        return self.metadata["target"]

    @property
    def body_nodes(self) -> List[Dict[str, Any]]:
        """Get the body operations of this block."""
        return self.metadata.get("body_nodes", [])

    def add_body_node(self, node: IRNode) -> None:
        """Add an operation to the block body."""
        if "body_nodes" not in self.metadata:
            self.metadata["body_nodes"] = []
        self.metadata["body_nodes"].append(node.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain.value,
            "op_type": self.op_type,
            "mode": self.mode,
            "target": self.target,
            "metadata": self.metadata,
        }

    def to_step(self) -> Dict[str, Any]:
        """Convert to legacy step format (for FFI serialization)."""
        return {
            "type": "core.execution_block",
            "mode": self.mode,
            "target": self.target,
            "options": {
                k: v
                for k, v in self.metadata.items()
                if k not in ["mode", "target", "body_nodes"]
            },
            "body": {"nodes": self.body_nodes},
        }

    def expand_to_steps(self) -> List[Dict[str, Any]]:
        """
        Expand execution block into a flat list of steps (fallback mode).

        This provides a lowering path for execution blocks when the runtime
        doesn't support native execution block handling. The block operations
        are expanded into regular sequential steps.

        Returns:
            List of step dictionaries representing the body operations
        """
        steps = []

        # For message-pass blocks, we need to reconstruct the loop structure
        # For now, just emit the body operations in sequence
        for node_dict in self.body_nodes:
            # Reconstruct the step from the node dictionary
            domain = node_dict.get("domain", "unknown")
            op_type = node_dict.get("op_type", "unknown")
            inputs = node_dict.get("inputs", [])
            metadata = node_dict.get("metadata", {})
            output = node_dict.get("output")

            # Create step with type and output
            step = {
                "type": f"{domain}.{op_type}",
            }

            if output:
                step["output"] = output

            # Map inputs to parameter names based on operation type
            # This covers the common operations used in execution blocks
            if op_type in ["add", "sub", "mul", "div", "compare"]:
                # Binary operations: a, b
                if len(inputs) >= 1:
                    step["a"] = inputs[0]
                if len(inputs) >= 2:
                    step["b"] = inputs[1]
                # Add metadata like comparison operator
                if "op" in metadata:
                    step["op"] = metadata["op"]

            elif op_type == "where":
                # Conditional: condition, if_true, if_false
                if len(inputs) >= 1:
                    step["condition"] = inputs[0]
                if len(inputs) >= 2:
                    step["if_true"] = inputs[1]
                if len(inputs) >= 3:
                    step["if_false"] = inputs[2]

            elif op_type == "collect_neighbor_values":
                # Neighbor collection: source
                if len(inputs) >= 1:
                    step["source"] = inputs[0]
                step["include_self"] = metadata.get("include_self", True)
                if "direction" in metadata:
                    step["direction"] = metadata["direction"]

            elif op_type == "mode" or op_type == "mode_list":
                # Mode computation: source (list of values)
                if len(inputs) >= 1:
                    step["source"] = inputs[0]
                if "tie_break" in metadata:
                    step["tie_break"] = metadata["tie_break"]

            elif op_type == "constant":
                # Constant value
                if "value" in metadata:
                    step["value"] = metadata["value"]

            elif op_type == "init_nodes":
                # Node initialization
                step["target"] = output or "temp"
                if "value" in metadata:
                    step["value"] = metadata["value"]

            else:
                # Generic fallback: use inputs as-is and add all metadata
                for i, inp in enumerate(inputs):
                    step[f"input_{i}"] = inp
                step.update(metadata)

            steps.append(step)

        return steps


# Convenience type for any IR node
AnyIRNode = Union[
    CoreIRNode,
    GraphIRNode,
    AttrIRNode,
    ControlIRNode,
    LoopIRNode,
    ExecutionBlockNode,
]
