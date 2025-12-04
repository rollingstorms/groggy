"""
Core arithmetic and value operations.

This module provides CoreOps, containing pure value-space operations
like arithmetic, reductions, conditionals, and scalar operations.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from groggy.builder.algorithm_builder import AlgorithmBuilder

# Import VarHandle from the new location
from groggy.builder.varhandle import VarHandle


class CoreOps:
    """Namespace for core step primitives."""

    def __init__(self, builder: "AlgorithmBuilder"):
        """
        Initialize core operations.

        Args:
            builder: Parent algorithm builder
        """
        self.builder = builder

    def _add_op(
        self, op_type: str, inputs: list, metadata: Optional[Dict] = None
    ) -> VarHandle:
        """
        Add an operation (supports both IR and step modes).

        Args:
            op_type: Operation type (e.g., 'add', 'mul')
            inputs: List of input variable names
            metadata: Optional metadata dict

        Returns:
            VarHandle for the result
        """
        var = self.builder._new_var(op_type)

        if self.builder.use_ir and self.builder.ir_graph is not None:
            # Import here to avoid circular dependency
            from groggy.builder.ir.nodes import CoreIRNode, IRDomain

            node = CoreIRNode(
                node_id=f"node_{len(self.builder.ir_graph.nodes)}",
                op_type=op_type,
                inputs=inputs,
                output=var.name,
                **(metadata or {}),
            )
            self.builder._add_ir_node(node)
        else:
            # Legacy step mode (without domain prefix - added by encoder)
            step = {"type": op_type, "output": var.name}
            step.update(metadata or {})
            if len(inputs) >= 1:
                step["left"] = inputs[0]
            if len(inputs) >= 2:
                step["right"] = inputs[1]
            for i, inp in enumerate(inputs[2:], start=2):
                step[f"input_{i}"] = inp
            self.builder.steps.append(step)

        return var

    def _ensure_var(self, value: Union[VarHandle, float, int]) -> str:
        """Convert value to variable name, creating scalar constant if needed."""
        if isinstance(value, VarHandle):
            return value.name
        else:
            return self.constant(value).name

    def constant(self, value: Union[float, int]) -> VarHandle:
        """
        Create a constant scalar value.

        Args:
            value: Constant value

        Returns:
            VarHandle for the constant
        """
        var = self.builder._new_var("const")

        if self.builder.use_ir and self.builder.ir_graph is not None:
            from groggy.builder.ir.nodes import CoreIRNode, IRDomain

            node = CoreIRNode(
                node_id=f"node_{len(self.builder.ir_graph.nodes)}",
                op_type="constant",
                inputs=[],
                output=var.name,
                value=value,
            )
            self.builder._add_ir_node(node)
        else:
            self.builder.steps.append(
                {"type": "init_scalar", "output": var.name, "value": value}
            )

        return var

    def add(
        self, left: Union[VarHandle, float], right: Union[VarHandle, float]
    ) -> VarHandle:
        """
        Element-wise addition or scalar addition.

        Args:
            left: Left operand (variable or scalar)
            right: Right operand (variable or scalar)

        Returns:
            VarHandle for the result

        Example:
            >>> result = builder.core.add(values, 1.0)
            >>> combined = builder.core.add(values1, values2)
        """
        # Optimize: if both are scalar constants, compute at build time
        if not isinstance(left, VarHandle) and not isinstance(right, VarHandle):
            result_value = left + right
            return self.constant(result_value)

        left_var = self._ensure_var(left)
        right_var = self._ensure_var(right)
        return self._add_op("add", [left_var, right_var])

    def sub(
        self, left: Union[VarHandle, float], right: Union[VarHandle, float]
    ) -> VarHandle:
        """
        Element-wise subtraction or scalar subtraction.

        Args:
            left: Left operand (variable or scalar)
            right: Right operand (variable or scalar)

        Returns:
            VarHandle for the result
        """
        # Optimize: if both are scalar constants, compute at build time
        if not isinstance(left, VarHandle) and not isinstance(right, VarHandle):
            result_value = left - right
            return self.constant(result_value)

        left_var = self._ensure_var(left)
        right_var = self._ensure_var(right)
        return self._add_op("sub", [left_var, right_var])

    def mul(
        self, left: Union[VarHandle, float], right: Union[VarHandle, float]
    ) -> VarHandle:
        """
        Element-wise multiplication or scalar multiplication.

        Args:
            left: Left operand (variable or scalar)
            right: Right operand (variable or scalar)

        Returns:
            VarHandle for the result

        Example:
            >>> scaled = builder.core.mul(values, 0.85)
        """
        # Optimize: if both are scalar constants, compute at build time
        if not isinstance(left, VarHandle) and not isinstance(right, VarHandle):
            result_value = left * right
            return self.constant(result_value)

        left_var = self._ensure_var(left)
        right_var = self._ensure_var(right)
        return self._add_op("mul", [left_var, right_var])

    def div(
        self, left: Union[VarHandle, float], right: Union[VarHandle, float]
    ) -> VarHandle:
        """
        Element-wise division or scalar division.

        Args:
            left: Left operand (variable or scalar)
            right: Right operand (variable or scalar)

        Returns:
            VarHandle for the result
        """
        # Optimize: if both are scalar constants, compute at build time
        if not isinstance(left, VarHandle) and not isinstance(right, VarHandle):
            result_value = left / right
            return self.constant(result_value)

        left_var = self._ensure_var(left)
        right_var = self._ensure_var(right)
        return self._add_op("div", [left_var, right_var])

    def recip(self, values: VarHandle, epsilon: float = 1e-10) -> VarHandle:
        """
        Element-wise reciprocal (1/x) with safe zero handling.

        Computes 1 / (x + epsilon) to avoid division by zero.

        Args:
            values: Input values
            epsilon: Small value added to prevent division by zero (default: 1e-10)

        Returns:
            VarHandle for reciprocal values

        Example:
            >>> degrees = builder.node_degrees()
            >>> inv_degrees = builder.core.recip(degrees, epsilon=1e-10)
        """
        var = self.builder._new_var("recip")
        self.builder.steps.append(
            {
                "type": "core.recip",
                "source": values.name,
                "epsilon": epsilon,
                "output": var.name,
            }
        )
        return var

    def compare(
        self, left: VarHandle, op: str, right: Union[VarHandle, float]
    ) -> VarHandle:
        """
        Element-wise comparison producing 0.0/1.0 mask.

        Compares left operand with right operand using the specified operator.
        Returns 1.0 where condition is true, 0.0 where false.

        Args:
            left: Left operand (variable)
            op: Comparison operator: 'eq', 'ne', 'lt', 'le', 'gt', 'ge'
            right: Right operand (variable or scalar)

        Returns:
            VarHandle for mask values (0.0 or 1.0)

        Example:
            >>> degrees = builder.node_degrees()
            >>> is_sink = builder.core.compare(degrees, "eq", 0.0)
            >>> is_high_degree = builder.core.compare(degrees, "gt", 10.0)
        """
        right_var = self._ensure_var(right)
        return self._add_op("compare", [left.name, right_var], {"op": op})

    def where(
        self,
        condition: VarHandle,
        if_true: Union[VarHandle, float],
        if_false: Union[VarHandle, float],
    ) -> VarHandle:
        """
        Element-wise conditional selection (ternary operator).

        For each element, returns if_true value where condition is non-zero,
        otherwise returns if_false value.

        Args:
            condition: Mask variable (typically from compare)
            if_true: Value to use where condition is true (variable or scalar)
            if_false: Value to use where condition is false (variable or scalar)

        Returns:
            VarHandle for selected values

        Example:
            >>> degrees = builder.node_degrees()
            >>> is_sink = builder.core.compare(degrees, "eq", 0.0)
            >>> sink_ranks = builder.core.where(is_sink, ranks, 0.0)
        """
        true_var = self._ensure_var(if_true)
        false_var = self._ensure_var(if_false)
        return self._add_op("where", [condition.name, true_var, false_var])

    def reduce_scalar(self, values: VarHandle, op: str = "sum") -> VarHandle:
        """
        Reduce node/edge map to a single scalar value.

        Aggregates all values in the map to a single number.

        Args:
            values: Map to reduce
            op: Reduction operation: 'sum', 'mean', 'min', 'max'

        Returns:
            VarHandle for scalar result

        Example:
            >>> is_sink = builder.core.compare(degrees, "eq", 0.0)
            >>> sink_ranks = builder.core.where(is_sink, ranks, 0.0)
            >>> sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
        """
        var = self.builder._new_var("scalar")
        self.builder.steps.append(
            {
                "type": "core.reduce_scalar",
                "source": values.name,
                "op": op,
                "output": var.name,
            }
        )
        return var

    def broadcast_scalar(self, scalar: VarHandle, reference: VarHandle) -> VarHandle:
        """
        Broadcast a scalar value to all nodes/edges.

        Creates a map where every entry has the same scalar value.
        Uses a reference map to determine which keys to create.

        Args:
            scalar: Scalar variable to broadcast
            reference: Reference map (defines which nodes/edges to create)

        Returns:
            VarHandle for broadcasted map

        Example:
            >>> sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
            >>> sink_contrib = builder.core.broadcast_scalar(sink_mass, ranks)
            >>> # Now sink_contrib has the same value for every node
        """
        var = self.builder._new_var("broadcast")

        if self.builder.use_ir and self.builder.ir_graph is not None:
            from groggy.builder.ir.nodes import CoreIRNode

            node = CoreIRNode(
                node_id=f"node_{len(self.builder.ir_graph.nodes)}",
                op_type="broadcast_scalar",
                inputs=[scalar.name, reference.name],
                output=var.name,
                scalar=scalar.name,
                reference=reference.name,
            )
            self.builder._add_ir_node(node)
        else:
            self.builder.steps.append(
                {
                    "type": "core.broadcast_scalar",
                    "scalar": scalar.name,
                    "reference": reference.name,
                    "output": var.name,
                }
            )

        return var

    def neighbor_agg(
        self, values: VarHandle, agg: str = "sum", weights: Optional[VarHandle] = None
    ) -> VarHandle:
        """
        Aggregate neighbor values for each node.

        .. deprecated::
            Use ``builder.graph_ops.neighbor_agg()`` instead.
            This method moved to GraphOps trait for better organization.

        Args:
            values: Node values to aggregate
            agg: Aggregation type: 'sum', 'mean', 'min', 'max', 'mode'
            weights: Optional weights to apply to neighbor values

        Returns:
            VarHandle for aggregated neighbor values
        """
        import warnings

        warnings.warn(
            "core.neighbor_agg() is deprecated, use builder.graph_ops.neighbor_agg() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        # Delegate to GraphOps
        return self.builder.graph_ops.neighbor_agg(values, agg, weights)

    def collect_neighbor_values(
        self, values: VarHandle, include_self: bool = True
    ) -> VarHandle:
        """
        Collect neighbor values into lists for each node.

        .. deprecated::
            Use ``builder.graph_ops.collect_neighbor_values()`` instead.
            This method moved to GraphOps trait for better organization.

        Args:
            values: Node values to collect
            include_self: Whether to include the node's own value in the list

        Returns:
            VarHandle for lists of neighbor values
        """
        import warnings

        warnings.warn(
            "core.collect_neighbor_values() is deprecated, use builder.graph_ops.collect_neighbor_values() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        # Delegate to GraphOps
        return self.builder.graph_ops.collect_neighbor_values(values, include_self)

    def mode(self, lists: VarHandle, tie_break: str = "lowest") -> VarHandle:
        """
        Find the most frequent value in each list (mode).

        For each node's list, finds the most common value. Used in LPA
        to adopt the most common neighbor label.

        Args:
            lists: Node lists (from collect_neighbor_values)
            tie_break: How to break ties: 'lowest', 'highest', or 'keep' (first)

        Returns:
            VarHandle for mode values

        Example:
            >>> # LPA: find most common neighbor label
            >>> labels = builder.init_nodes(unique=True)
            >>> neighbor_labels = builder.core.collect_neighbor_values(labels, include_self=True)
            >>> new_labels = builder.core.mode(neighbor_labels, tie_break="lowest")
        """
        var = self.builder._new_var("mode")

        if self.builder.use_ir and self.builder.ir_graph is not None:
            from groggy.builder.ir.nodes import CoreIRNode

            node = CoreIRNode(
                node_id=f"node_{len(self.builder.ir_graph.nodes)}",
                op_type="mode_list",
                inputs=[lists.name],
                output=var.name,
                tie_break=tie_break,
                source=lists.name,
            )
            self.builder._add_ir_node(node)
        else:
            self.builder.steps.append(
                {
                    "type": "core.mode_list",
                    "source": lists.name,
                    "tie_break": tie_break,
                    "output": var.name,
                }
            )

        return var

    def update_in_place(
        self, source: VarHandle, target: VarHandle, ordered: bool = False
    ) -> VarHandle:
        """
        Update a target map in-place with values from source.

        This enables "async" update semantics where each node's update
        is immediately visible during traversal. When ordered=True,
        nodes are processed in sorted order for determinism (essential for LPA).

        Args:
            source: Map with new values to apply
            target: Map to update in place (gets mutated)
            ordered: If True, process nodes in sorted order

        Returns:
            VarHandle for updated target (same as target param)

        Example:
            >>> # LPA async update
            >>> labels = builder.init_nodes(unique=True)
            >>> with builder.iterate(10):
            ...     neighbor_labels = builder.core.collect_neighbor_values(labels)
            ...     new_labels = builder.core.mode(neighbor_labels)
            ...     labels = builder.core.update_in_place(new_labels, target=labels, ordered=True)
        """
        self.builder.steps.append(
            {
                "type": "core.update_in_place",
                "source": source.name,
                "target": target.name,
                "ordered": ordered,
                "output": target.name,  # Updates in place, returns target
            }
        )
        return target

    def neighbor_mode_update(
        self,
        target: VarHandle,
        include_self: bool = True,
        tie_break: str = "lowest",
        ordered: bool = True,
    ) -> VarHandle:
        """
        Update labels in-place by taking the mode of neighbor labels.

        .. deprecated::
            Use ``builder.graph_ops.neighbor_mode_update()`` instead.
            This method moved to GraphOps trait for better organization.

        Args:
            target: Label map to update in place
            include_self: Whether to include the node's current label in the vote
            tie_break: 'lowest', 'highest', or 'keep' for first occurrence
            ordered: Process nodes in sorted order if True (recommended for determinism)

        Returns:
            VarHandle for updated labels (same as target)
        """
        import warnings

        warnings.warn(
            "core.neighbor_mode_update() is deprecated, use builder.graph_ops.neighbor_mode_update() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        # Delegate to GraphOps
        return self.builder.graph_ops.neighbor_mode_update(
            target, include_self, tie_break, ordered
        )

    def normalize_sum(self, values: VarHandle) -> VarHandle:
        """
        Normalize values so they sum to 1.0.

        Args:
            values: Values to normalize

        Returns:
            VarHandle for normalized values

        Example:
            >>> normalized = builder.core.normalize_sum(ranks)
        """
        var = self.builder._new_var("normalized")
        self.builder.steps.append(
            {"type": "normalize_sum", "input": values.name, "output": var.name}
        )
        return var

    def histogram(self, values: VarHandle, bins: int = 10) -> VarHandle:
        """
        Compute histogram of node values.

        Divides the value range into bins and counts how many values
        fall into each bin. Returns a map where keys are bin indices
        and values are counts.

        Args:
            values: Node values to histogram
            bins: Number of bins (default: 10)

        Returns:
            VarHandle for histogram (bin_index -> count)

        Example:
            >>> # Compute degree distribution
            >>> degrees = builder.node_degrees()
            >>> hist = builder.core.histogram(degrees, bins=20)
        """
        var = self.builder._new_var("histogram")
        self.builder.steps.append(
            {
                "type": "core.histogram",
                "source": values.name,
                "bins": bins,
                "output": var.name,
            }
        )
        return var

    def clip(
        self,
        values: VarHandle,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> VarHandle:
        """
        Clip values to min/max bounds.

        Constrains each value to lie within [min_value, max_value].
        At least one of min_value or max_value must be provided.

        Args:
            values: Node or edge values to clip
            min_value: Minimum value (optional)
            max_value: Maximum value (optional)

        Returns:
            VarHandle for clipped values

        Example:
            >>> # Clip degrees to avoid division issues
            >>> degrees = builder.node_degrees()
            >>> safe_deg = builder.core.clip(degrees, min_value=1.0)
            >>>
            >>> # Cap values at both ends
            >>> normalized = builder.core.clip(values, min_value=0.0, max_value=1.0)
        """
        if min_value is None and max_value is None:
            raise ValueError(
                "core.clip requires at least one of min_value or max_value"
            )

        var = self.builder._new_var("clipped")
        step = {
            "type": "core.clip",
            "source": values.name,
            "target": var.name,
        }
        if min_value is not None:
            step["min_value"] = min_value
        if max_value is not None:
            step["max_value"] = max_value

        self.builder.steps.append(step)
        return var

    # Additional mathematical operations

    def pow(
        self, base: Union[VarHandle, float], exponent: Union[VarHandle, float]
    ) -> VarHandle:
        """
        Element-wise power operation.

        Args:
            base: Base values
            exponent: Exponent (can be variable or scalar)

        Returns:
            VarHandle for base^exponent

        Example:
            >>> # Square values
            >>> squared = builder.core.pow(values, 2.0)
            >>>
            >>> # Variable exponent
            >>> result = builder.core.pow(values, exponents)
        """
        var = self.builder._new_var("pow")
        self.builder.steps.append(
            {
                "type": "core.pow",
                "base": self._ensure_var(base),
                "exponent": self._ensure_var(exponent),
                "output": var.name,
            }
        )
        return var

    def abs(self, values: VarHandle) -> VarHandle:
        """
        Element-wise absolute value.

        Args:
            values: Input values

        Returns:
            VarHandle for absolute values

        Example:
            >>> abs_vals = builder.core.abs(differences)
        """
        var = self.builder._new_var("abs")
        self.builder.steps.append(
            {"type": "core.abs", "source": values.name, "output": var.name}
        )
        return var

    def sqrt(self, values: VarHandle) -> VarHandle:
        """
        Element-wise square root.

        Args:
            values: Input values (should be non-negative)

        Returns:
            VarHandle for square roots

        Example:
            >>> roots = builder.core.sqrt(values)
        """
        var = self.builder._new_var("sqrt")
        self.builder.steps.append(
            {"type": "core.sqrt", "source": values.name, "output": var.name}
        )
        return var

    def exp(self, values: VarHandle) -> VarHandle:
        """
        Element-wise exponential (e^x).

        Args:
            values: Input values

        Returns:
            VarHandle for e^values

        Example:
            >>> exponentials = builder.core.exp(log_values)
        """
        var = self.builder._new_var("exp")
        self.builder.steps.append(
            {"type": "core.exp", "source": values.name, "output": var.name}
        )
        return var

    def log(self, values: VarHandle, base: Optional[float] = None) -> VarHandle:
        """
        Element-wise logarithm.

        Args:
            values: Input values (should be positive)
            base: Optional logarithm base (default: natural log)

        Returns:
            VarHandle for logarithms

        Example:
            >>> # Natural log
            >>> ln_vals = builder.core.log(values)
            >>>
            >>> # Log base 10
            >>> log10_vals = builder.core.log(values, base=10.0)
        """
        var = self.builder._new_var("log")
        step = {"type": "core.log", "source": values.name, "output": var.name}
        if base is not None:
            step["base"] = base
        self.builder.steps.append(step)
        return var

    def min(
        self, left: Union[VarHandle, float], right: Union[VarHandle, float]
    ) -> VarHandle:
        """
        Element-wise minimum.

        Args:
            left: First operand
            right: Second operand

        Returns:
            VarHandle for element-wise min(left, right)

        Example:
            >>> clamped = builder.core.min(values, 1.0)
        """
        var = self.builder._new_var("min")
        self.builder.steps.append(
            {
                "type": "core.min",
                "left": self._ensure_var(left),
                "right": self._ensure_var(right),
                "output": var.name,
            }
        )
        return var

    def max(
        self, left: Union[VarHandle, float], right: Union[VarHandle, float]
    ) -> VarHandle:
        """
        Element-wise maximum.

        Args:
            left: First operand
            right: Second operand

        Returns:
            VarHandle for element-wise max(left, right)

        Example:
            >>> clamped = builder.core.max(values, 0.0)
        """
        var = self.builder._new_var("max")
        self.builder.steps.append(
            {
                "type": "core.max",
                "left": self._ensure_var(left),
                "right": self._ensure_var(right),
                "output": var.name,
            }
        )
        return var
