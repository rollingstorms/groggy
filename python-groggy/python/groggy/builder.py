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

from typing import Any, Dict, Optional, Union

from groggy import _groggy
from groggy.algorithms.base import AlgorithmHandle


class LoopContext:
    """Context manager for loop body."""

    def __init__(self, builder: "AlgorithmBuilder", iterations: int):
        """
        Create a loop context.

        Args:
            builder: Parent algorithm builder
            iterations: Number of times to repeat loop body
        """
        self.builder = builder
        self.iterations = iterations
        self.start_step = None
        self.loop_vars = {}

    def __enter__(self):
        """Mark start of loop body."""
        # Mark where loop body starts
        self.start_step = len(self.builder.steps)

        # Snapshot current variables
        self.loop_vars = dict(self.builder.variables)

        return self

    def __exit__(self, *args):
        """Unroll the loop."""
        self.builder._finalize_loop(self.start_step, self.iterations, self.loop_vars)


class VarHandle:
    """
    Handle representing a variable in the algorithm builder.

    Variables track intermediate results as the algorithm is built.
    """

    def __init__(self, name: str, builder: "AlgorithmBuilder"):
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


class SubgraphHandle:
    """
    Handle representing a reference to the input subgraph.

    This allows algorithms to explicitly reference the input subgraph
    when needed, though most operations implicitly work on the input.
    """

    def __init__(self, name: str, builder: "AlgorithmBuilder"):
        """
        Create a subgraph handle.

        Args:
            name: Reference name for the subgraph
            builder: Parent builder
        """
        self.name = name
        self.builder = builder

    def __repr__(self) -> str:
        return f"SubgraphHandle('{self.name}')"


class CoreOps:
    """Namespace for core step primitives."""

    def __init__(self, builder: "AlgorithmBuilder"):
        """
        Initialize core operations.

        Args:
            builder: Parent algorithm builder
        """
        self.builder = builder

    def _ensure_var(self, value: Union[VarHandle, float, int]) -> str:
        """Convert value to variable name, creating scalar constant if needed."""
        if isinstance(value, VarHandle):
            return value.name
        else:
            # Create a scalar constant variable
            const_var = self.builder._new_var("scalar")
            self.builder.steps.append(
                {"type": "init_scalar", "output": const_var.name, "value": value}
            )
            return const_var.name

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
        var = self.builder._new_var("add")
        self.builder.steps.append(
            {
                "type": "core.add",
                "left": self._ensure_var(left),
                "right": self._ensure_var(right),
                "output": var.name,
            }
        )
        return var

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
        var = self.builder._new_var("sub")
        self.builder.steps.append(
            {
                "type": "core.sub",
                "left": self._ensure_var(left),
                "right": self._ensure_var(right),
                "output": var.name,
            }
        )
        return var

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
        var = self.builder._new_var("mul")
        self.builder.steps.append(
            {
                "type": "core.mul",
                "left": self._ensure_var(left),
                "right": self._ensure_var(right),
                "output": var.name,
            }
        )
        return var

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
        var = self.builder._new_var("div")
        self.builder.steps.append(
            {
                "type": "core.div",
                "left": self._ensure_var(left),
                "right": self._ensure_var(right),
                "output": var.name,
            }
        )
        return var

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
        var = self.builder._new_var("compare")
        self.builder.steps.append(
            {
                "type": "core.compare",
                "left": left.name,
                "op": op,
                "right": self._ensure_var(right),
                "output": var.name,
            }
        )
        return var

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
        var = self.builder._new_var("where")
        self.builder.steps.append(
            {
                "type": "core.where",
                "condition": condition.name,
                "if_true": self._ensure_var(if_true),
                "if_false": self._ensure_var(if_false),
                "output": var.name,
            }
        )
        return var

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

        For each node, aggregates values from its neighbors. Optionally
        applies weights to neighbor contributions.

        Args:
            values: Node values to aggregate
            agg: Aggregation type: 'sum', 'mean', 'min', 'max', 'mode'
            weights: Optional weights to apply to neighbor values

        Returns:
            VarHandle for aggregated neighbor values

        Example:
            >>> # PageRank: sum(ranks[neighbors(node)] / out_degree[neighbors(node)])
            >>> inv_degrees = builder.core.recip(degrees)
            >>> weighted_ranks = builder.core.mul(ranks, inv_degrees)
            >>> neighbor_sum = builder.core.neighbor_agg(weighted_ranks, agg="sum")
            >>>
            >>> # Or with weights parameter:
            >>> neighbor_sum = builder.core.neighbor_agg(ranks, agg="sum", weights=inv_degrees)
        """
        var = self.builder._new_var("neighbor_agg")
        step = {
            "type": "core.neighbor_agg",
            "source": values.name,
            "agg": agg,
            "output": var.name,
        }
        if weights is not None:
            step["weights"] = weights.name
        self.builder.steps.append(step)
        return var

    def collect_neighbor_values(
        self, values: VarHandle, include_self: bool = True
    ) -> VarHandle:
        """
        Collect neighbor values into lists for each node.

        For each node, gathers all neighbor values into a list. Used for
        algorithms like LPA that need to find the most common neighbor value.

        Args:
            values: Node values to collect
            include_self: Whether to include the node's own value in the list

        Returns:
            VarHandle for lists of neighbor values

        Example:
            >>> # LPA: collect neighbor labels (including self)
            >>> labels = builder.init_nodes_with_index()
            >>> neighbor_labels = builder.core.collect_neighbor_values(labels, include_self=True)
            >>> most_common = builder.core.mode(neighbor_labels)
        """
        var = self.builder._new_var("neighbor_values")
        self.builder.steps.append(
            {
                "type": "core.collect_neighbor_values",
                "source": values.name,
                "include_self": include_self,
                "output": var.name,
            }
        )
        return var

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

        This mirrors asynchronous LPA semantics where nodes update sequentially
        using the most common neighbor label (with optional self inclusion) and
        deterministic ordering.

        Args:
            target: Label map to update in place
            include_self: Whether to include the node's current label in the vote
            tie_break: 'lowest', 'highest', or 'keep' for first occurrence
            ordered: Process nodes in sorted order if True (recommended for determinism)

        Returns:
            VarHandle for updated labels (same as target)
        """
        self.builder.steps.append(
            {
                "type": "core.neighbor_mode_update",
                "target": target.name,
                "include_self": include_self,
                "tie_break": tie_break,
                "ordered": ordered,
                "output": target.name,
            }
        )
        return target

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
        self._input_ref = None
        self.core = CoreOps(self)

    def _new_var(self, prefix: str = "var") -> VarHandle:
        """Create a new variable handle."""
        var_name = f"{prefix}_{self._var_counter}"
        self._var_counter += 1
        handle = VarHandle(var_name, self)
        self.variables[var_name] = handle
        return handle

    def auto_var(self, prefix: str = "var") -> VarHandle:
        """
        Create a unique variable name with the given prefix.

        This is the public interface to the variable name generator,
        useful when you need to create temporary variables.

        Args:
            prefix: Prefix for the variable name

        Returns:
            VarHandle with a unique name

        Example:
            >>> temp = builder.auto_var("temp")
            >>> print(temp.name)  # "temp_0"
        """
        return self._new_var(prefix)

    def input(self, name: str = "subgraph") -> SubgraphHandle:
        """
        Create a reference to the input subgraph.

        This allows algorithms to explicitly reference the input subgraph,
        though most operations implicitly work on the input by default.

        Args:
            name: Name for the input reference (default: "subgraph")

        Returns:
            SubgraphHandle for the input

        Example:
            >>> sg = builder.input("graph")
            >>> # Most operations don't need explicit input reference
            >>> # but it's available for advanced use cases
        """
        if self._input_ref is None:
            self._input_ref = SubgraphHandle(name, self)
        return self._input_ref

    def init_nodes(self, default: Any = 0.0, *, unique: bool = False) -> VarHandle:
        """
        Initialize node values.

        Args:
            default: Default value for all nodes (ignored if unique=True)
            unique: If True, initialize with sequential indices (0, 1, 2, ...)
                    Useful for algorithms like LPA that need unique labels

        Returns:
            Variable handle for initialized values

        Example:
            >>> # Initialize with constant
            >>> ranks = builder.init_nodes(default=1.0)
            >>>
            >>> # Initialize with unique indices (for LPA)
            >>> labels = builder.init_nodes(unique=True)
        """
        var = self._new_var("nodes")

        if unique:
            self.steps.append({"type": "init_nodes_with_index", "output": var.name})
        else:
            self.steps.append(
                {"type": "init_nodes", "output": var.name, "default": default}
            )

        return var

    def load_attr(self, attr: str, default: Any = 0.0) -> VarHandle:
        """
        Load node attribute values into a variable.

        This allows algorithms to read existing node attributes from the graph.

        Args:
            attr: Name of the node attribute to load
            default: Default value for nodes without the attribute

        Returns:
            Variable handle containing the attribute values

        Example:
            >>> # Load existing node weights
            >>> weights = builder.load_attr("weight", default=1.0)
            >>> scaled = builder.core.mul(weights, 2.0)
        """
        var = self._new_var("attr")
        self.steps.append(
            {
                "type": "load_attr",
                "attr_name": attr,
                "default": default,
                "output": var.name,
            }
        )
        return var

    def graph_node_count(self) -> VarHandle:
        """
        Get the number of nodes in the current subgraph as a scalar variable.

        This returns a scalar that can be used in arithmetic operations
        (e.g., computing teleport probability in PageRank as 1/N).

        Returns:
            Scalar variable handle containing the node count

        Example:
            >>> n = builder.graph_node_count()
            >>> teleport_prob = builder.core.div(1.0, n)
            >>> uniform = builder.core.div(1.0, n)
        """
        var = self._new_var("n")
        self.steps.append({"type": "graph_node_count", "output": var.name})
        return var

    def graph_edge_count(self) -> VarHandle:
        """
        Get the number of edges in the current subgraph as a scalar variable.

        This returns a scalar that can be used in arithmetic operations.

        Returns:
            Scalar variable handle containing the edge count

        Example:
            >>> m = builder.graph_edge_count()
            >>> density = builder.core.div(m, builder.core.mul(n, n))
        """
        var = self._new_var("m")
        self.steps.append({"type": "graph_edge_count", "output": var.name})
        return var

    def load_edge_attr(self, attr: str, default: Any = 0.0) -> VarHandle:
        """
        Load edge attribute values into a variable.

        This creates a mapping from (source, target) pairs to attribute values.

        Args:
            attr: Name of the edge attribute to load
            default: Default value for edges without the attribute

        Returns:
            Variable handle containing the edge attribute values

        Example:
            >>> # Load edge weights for weighted aggregation
            >>> edge_weights = builder.load_edge_attr("weight", default=1.0)
        """
        var = self._new_var("edge_attr")
        self.steps.append(
            {
                "type": "load_edge_attr",
                "attr_name": attr,
                "default": default,
                "output": var.name,
            }
        )
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
        self.steps.append(
            {"type": "node_degree", "source": nodes.name, "output": var.name}
        )
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
        self.steps.append(
            {
                "type": "normalize",
                "input": values.name,
                "output": var.name,
                "method": method,
            }
        )
        return var

    def iterate(self, count: int) -> LoopContext:
        """
        Create a loop that repeats for `count` iterations.

        Steps added within the context will be repeated, with variables
        properly tracked across iterations.

        Args:
            count: Number of iterations

        Returns:
            Loop context manager

        Example:
            >>> with builder.iterate(20):
            ...     sums = builder.map_nodes(
            ...         "sum(ranks[neighbors(node)])",
            ...         inputs={"ranks": ranks}
            ...     )
            ...     ranks = builder.var("ranks", builder.core.mul(sums, 0.85))
        """
        return LoopContext(self, count)

    def _finalize_loop(
        self, start_step: int, iterations: int, loop_vars: Dict[str, VarHandle]
    ):
        """
        Unroll loop by repeating steps.

        Args:
            start_step: Index where loop body starts
            iterations: Number of times to repeat
            loop_vars: Variables at loop start
        """
        # Extract loop body
        loop_body = self.steps[start_step:]

        # Remove loop body from main steps
        self.steps = self.steps[:start_step]

        # Track variable renames across iterations
        var_mapping = {v.name: v.name for v in loop_vars.values()}

        def remap_value(val):
            """Recursively remap variable references using current iteration mapping."""
            if isinstance(val, str):
                return var_mapping.get(val, val)
            if isinstance(val, list):
                return [remap_value(item) for item in val]
            if isinstance(val, dict):
                return {key: remap_value(value) for key, value in val.items()}
            return val

        # Repeat body N times
        for iteration in range(iterations):
            for step in loop_body:
                # Clone step
                new_step = step.copy()
                step_type = new_step.get("type")

                # Remap all variable references except for fields we handle explicitly
                for key, value in list(new_step.items()):
                    if key == "type":
                        continue
                    if key == "output":
                        continue  # handled below with iteration suffix
                    if step_type == "alias" and key == "target":
                        continue  # logical name should remain stable
                    new_step[key] = remap_value(value)

                # Generate unique output name for this iteration
                if "output" in new_step:
                    original_output = new_step["output"]
                    new_output = f"{original_output}_iter{iteration}"
                    new_step["output"] = new_output

                    # Update mapping
                    var_mapping[original_output] = new_output
                    # Update paired target fields to use the new output name
                    if (
                        step_type != "alias"
                        and "target" in new_step
                        and isinstance(new_step["target"], str)
                    ):
                        new_step["target"] = new_output

                # Handle alias target - update mapping immediately after processing.
                # new_step["source"] has already been remapped for this iteration.
                if new_step.get("type") == "alias" and "target" in new_step:
                    target = new_step.get("target")
                    remapped_source = new_step.get("source")  # Already remapped
                    if target and remapped_source:
                        # Map the target name to the remapped source
                        var_mapping[target] = remapped_source

                self.steps.append(new_step)

        # Add final aliases to restore original variable names for any variables
        # that were reassigned during the loop (exist in builder.variables)
        for original, final in var_mapping.items():
            if original != final and original in self.variables:
                self.steps.append(
                    {"type": "alias", "source": final, "target": original}
                )

                # Update variable handle to point to final name
                self.variables[original].name = final

    def map_nodes(
        self,
        fn: str,
        inputs: Optional[Dict[str, VarHandle]] = None,
        async_update: bool = False,
    ) -> VarHandle:
        """
        Map expression over nodes with access to neighbors.

        This allows aggregating neighbor values or applying expressions to each node.

        Args:
            fn: Expression string (e.g., "sum(ranks[neighbors(node)])")
            inputs: Variable context for the expression
            async_update: If True, updates are visible to subsequent nodes in same iteration (for LPA)

        Returns:
            VarHandle for the result

        Example:
            >>> # Sum neighbor values
            >>> sums = builder.map_nodes(
            ...     "sum(ranks[neighbors(node)])",
            ...     inputs={"ranks": ranks}
            ... )
            >>>
            >>> # LPA with async updates (nodes see earlier updates)
            >>> labels = builder.map_nodes(
            ...     "mode(labels[neighbors(node)])",
            ...     inputs={"labels": labels},
            ...     async_update=True
            ... )
        """
        # Build inputs dict with variable names
        input_vars = {}
        if inputs:
            for key, val in inputs.items():
                input_vars[key] = val.name

        if async_update:
            if not input_vars:
                raise ValueError(
                    "async_update=True requires at least one input variable"
                )
            # Use the first input variable as both source and output so the step operates in place.
            first_input_name = next(iter(input_vars.values()))
            output_name = first_input_name
            handle = VarHandle(output_name, self)
        else:
            handle = self._new_var("mapped")
            output_name = handle.name

        self.steps.append(
            {
                "type": "map_nodes",
                "fn": fn,
                "inputs": input_vars,
                "output": output_name,
                "async_update": async_update,
            }
        )

        return handle

    def var(self, name: str, value: VarHandle) -> VarHandle:
        """
        Create or reassign a variable.

        This is useful for updating variables in loops, allowing you to
        reuse the same variable name across iterations.

        Args:
            name: Variable name
            value: Value to assign

        Returns:
            VarHandle with the given name

        Example:
            >>> ranks = builder.init_nodes(1.0)
            >>> with builder.iterate(10):
            >>>     updated = builder.core.mul(ranks, 0.85)
            >>>     ranks = builder.var("ranks", updated)
        """
        # Create handle with requested name
        handle = VarHandle(name, self)
        self.variables[name] = handle

        # Track the assignment (no actual step needed, just variable tracking)
        # The value's name will be used where this variable is referenced
        if value.name != name:
            # Store mapping for step generation
            self.steps.append({"type": "alias", "source": value.name, "target": name})

        return handle

    def attach_as(self, attr_name: str, values: VarHandle):
        """
        Attach values as a node attribute.

        Args:
            attr_name: Name for the output attribute
            values: Values to attach
        """
        self.steps.append(
            {"type": "attach_attr", "input": values.name, "attr_name": attr_name}
        )

    def build(self, validate: bool = True) -> "BuiltAlgorithm":
        """
        Build the algorithm from the composed steps.

        Args:
            validate: If True, validate pipeline before building (default: True)

        Returns:
            Built algorithm ready for use

        Raises:
            ValidationError: If validation fails and validate=True
        """
        algo = BuiltAlgorithm(self.name, self.steps)

        if validate:
            errors, warnings = algo._validate()

            # Print warnings if any
            if warnings:
                import warnings as warn_module

                for warning in warnings:
                    warn_module.warn(f"Pipeline validation: {warning}", UserWarning)

            # Raise error if validation failed
            if errors:
                from groggy.errors import ValidationError

                raise ValidationError(errors, warnings)

            algo._validated = True

        return algo


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
        self._validated = False

    @property
    def id(self) -> str:
        """Get the algorithm identifier."""
        return self._id

    def to_spec(self) -> Dict[str, Any]:
        """Convert to a pipeline spec entry usable by the Rust executor."""

        alias_map: Dict[str, str] = {}
        encoded_steps: list[Dict[str, Any]] = []

        for step in self._steps:
            if step.get("type") == "alias":
                source = step.get("source")
                target = step.get("target")
                if source and target:
                    resolved_source = self._resolve_with_alias(source, alias_map)
                    alias_map[target] = resolved_source
                continue

            encoded = self._encode_step(step, alias_map)
            if encoded is not None:
                encoded_steps.append(encoded)

        pipeline_definition = {
            "name": self._name,
            "steps": encoded_steps,
        }

        return {
            "id": "builder.step_pipeline",
            "params": {
                "name": _groggy.AttrValue(self._name),
                "steps": _groggy.AttrValue(pipeline_definition),
            },
        }

    def _resolve_with_alias(self, value: Any, alias_map: Dict[str, str]) -> Any:
        if not isinstance(value, str):
            return value

        resolved = value
        visited = set()
        while resolved in alias_map and resolved not in visited:
            visited.add(resolved)
            resolved = alias_map[resolved]

        return resolved

    def _resolve_operand(self, value: Any, alias_map: Dict[str, str]) -> Any:
        """Resolve operands that reference aliased variables."""
        return self._resolve_with_alias(value, alias_map)

    def _encode_step(
        self,
        step: Dict[str, Any],
        alias_map: Dict[str, str],
    ) -> Dict[str, Any]:
        step_type = step.get("type")

        if step_type == "init_nodes":
            params: Dict[str, Any] = {"target": step["output"]}
            default = step.get("default")
            if default is not None:
                params["value"] = default
            return {"id": "core.init_nodes", "params": params}

        if step_type == "init_nodes_with_index":
            return {
                "id": "core.init_nodes_with_index",
                "params": {"target": step["output"]},
            }

        if step_type == "init_scalar":
            params: Dict[str, Any] = {"target": step["output"]}
            value = step.get("value")
            if value is not None:
                params["value"] = value
            return {"id": "core.init_scalar", "params": params}

        if step_type == "graph_node_count":
            return {"id": "core.graph_node_count", "params": {"target": step["output"]}}

        if step_type == "graph_edge_count":
            return {"id": "core.graph_edge_count", "params": {"target": step["output"]}}

        if step_type == "node_degree":
            params = {
                "target": step["output"],
            }
            if "source" in step:
                params["source"] = self._resolve_operand(step["source"], alias_map)
            return {
                "id": "core.node_degree",
                "params": params,
            }

        if step_type == "normalize":
            params = {
                "source": self._resolve_operand(step["input"], alias_map),
                "target": step["output"],
                "method": step.get("method", "sum"),
                "epsilon": 1e-9,
            }
            return {"id": "core.normalize_node_values", "params": params}

        if step_type == "attach_attr":
            # Resolve variable name through aliases
            resolved_input = self._resolve_with_alias(step["input"], alias_map)
            return {
                "id": "core.attach_node_attr",
                "params": {"source": resolved_input, "attr": step["attr_name"]},
            }

        if step_type == "core.add":
            return {
                "id": "core.add",
                "params": {
                    "left": self._resolve_operand(step["left"], alias_map),
                    "right": self._resolve_operand(step["right"], alias_map),
                    "target": step["output"],
                },
            }

        if step_type == "core.sub":
            return {
                "id": "core.sub",
                "params": {
                    "left": self._resolve_operand(step["left"], alias_map),
                    "right": self._resolve_operand(step["right"], alias_map),
                    "target": step["output"],
                },
            }

        if step_type == "core.mul":
            return {
                "id": "core.mul",
                "params": {
                    "left": self._resolve_operand(step["left"], alias_map),
                    "right": self._resolve_operand(step["right"], alias_map),
                    "target": step["output"],
                },
            }

        if step_type == "core.div":
            return {
                "id": "core.div",
                "params": {
                    "left": self._resolve_operand(step["left"], alias_map),
                    "right": self._resolve_operand(step["right"], alias_map),
                    "target": step["output"],
                },
            }

        if step_type == "core.recip":
            return {
                "id": "core.recip",
                "params": {
                    "source": self._resolve_operand(step["source"], alias_map),
                    "target": step["output"],
                    "epsilon": step.get("epsilon", 1e-10),
                },
            }

        if step_type == "core.compare":
            return {
                "id": "core.compare",
                "params": {
                    "left": self._resolve_operand(step["left"], alias_map),
                    "op": step["op"],
                    "right": self._resolve_operand(step["right"], alias_map),
                    "target": step["output"],
                },
            }

        if step_type == "core.where":
            return {
                "id": "core.where",
                "params": {
                    "condition": self._resolve_operand(step["condition"], alias_map),
                    "if_true": self._resolve_operand(step["if_true"], alias_map),
                    "if_false": self._resolve_operand(step["if_false"], alias_map),
                    "target": step["output"],
                },
            }

        if step_type == "core.reduce_scalar":
            return {
                "id": "core.reduce_scalar",
                "params": {
                    "source": self._resolve_operand(step["source"], alias_map),
                    "op": step["op"],
                    "target": step["output"],
                },
            }

        if step_type == "core.broadcast_scalar":
            return {
                "id": "core.broadcast_scalar",
                "params": {
                    "scalar": self._resolve_operand(step["scalar"], alias_map),
                    "reference": self._resolve_operand(step["reference"], alias_map),
                    "target": step["output"],
                },
            }

        if step_type == "core.neighbor_agg":
            params = {
                "source": self._resolve_operand(step["source"], alias_map),
                "agg": step.get("agg", "sum"),
                "target": step["output"],
            }
            if "weights" in step:
                params["weights"] = self._resolve_operand(step["weights"], alias_map)
            return {"id": "core.neighbor_agg", "params": params}

        if step_type == "core.collect_neighbor_values":
            return {
                "id": "core.collect_neighbor_values",
                "params": {
                    "source": self._resolve_operand(step["source"], alias_map),
                    "include_self": step.get("include_self", True),
                    "target": step["output"],
                },
            }

        if step_type == "core.mode_list":
            return {
                "id": "core.mode_list",
                "params": {
                    "source": self._resolve_operand(step["source"], alias_map),
                    "tie_break": step.get("tie_break", "lowest"),
                    "target": step["output"],
                },
            }

        if step_type == "core.update_in_place":
            params = {
                "source": self._resolve_operand(step["source"], alias_map),
                "target": self._resolve_operand(step["target"], alias_map),
                "ordered": step.get("ordered", False),
            }
            # If output is specified and different from target, include it
            if "output" in step:
                params["output"] = step["output"]
            return {"id": "core.update_in_place", "params": params}

        if step_type == "core.neighbor_mode_update":
            params = {
                "target": self._resolve_operand(step["target"], alias_map),
                "include_self": step.get("include_self", True),
                "tie_break": step.get("tie_break", "lowest"),
                "ordered": step.get("ordered", True),
            }
            if "output" in step:
                params["output"] = step["output"]
            return {"id": "core.neighbor_mode_update", "params": params}

        if step_type == "normalize_sum":
            return {
                "id": "core.normalize_values",
                "params": {
                    "source": self._resolve_operand(step["input"], alias_map),
                    "target": step["output"],
                    "method": "sum",
                    "epsilon": 1e-9,
                },
            }

        if step_type == "map_nodes":
            # Parse expression string to Expr JSON
            from groggy.expr_parser import parse_expression

            expr_str = step["fn"]
            inputs = step.get("inputs", {})
            resolved_inputs = {
                key: self._resolve_operand(val, alias_map)
                for key, val in inputs.items()
            }

            # For now, we need to identify which variable the expression uses
            # The source is the first (and typically only) variable in inputs
            source = list(resolved_inputs.values())[0] if resolved_inputs else "input"

            # Parse the expression
            expr_json = parse_expression(expr_str)

            # Rewrite variable names in expression based on inputs mapping
            def rewrite_vars(expr_node):
                if isinstance(expr_node, dict):
                    if expr_node.get("type") == "var":
                        # Check if this variable name is in the inputs mapping
                        var_name = expr_node.get("name")
                        if var_name in resolved_inputs:
                            # Replace with actual variable name
                            expr_node["name"] = resolved_inputs[var_name]
                    # Recursively process args
                    if "args" in expr_node:
                        for arg in expr_node["args"]:
                            rewrite_vars(arg)
                    # Recursively process left/right
                    if "left" in expr_node:
                        rewrite_vars(expr_node["left"])
                    if "right" in expr_node:
                        rewrite_vars(expr_node["right"])
                return expr_node

            rewrite_vars(expr_json)

            params = {"source": source, "target": step["output"], "expr": expr_json}

            # Add async_update flag if present
            if step.get("async_update", False):
                params["async_update"] = True

            return {"id": "core.map_nodes", "params": params}

        if step_type == "alias":
            # Alias is just variable tracking, no Rust step needed
            # But we'll skip it in step generation
            return None

        if step_type == "load_attr":
            params: Dict[str, Any] = {
                "attr": step["attr_name"],
                "target": step["output"],
            }
            default = step.get("default")
            if default is not None:
                params["default"] = default
            return {"id": "core.load_node_attr", "params": params}

        if step_type == "load_edge_attr":
            params: Dict[str, Any] = {
                "attr": step["attr_name"],
                "target": step["output"],
            }
            default = step.get("default")
            if default is not None:
                params["default"] = default
            return {"id": "core.load_edge_attr", "params": params}

        if step_type == "core.histogram":
            return {
                "id": "core.histogram",
                "params": {
                    "source": self._resolve_operand(step["source"], alias_map),
                    "bins": step.get("bins", 10),
                    "target": step["output"],
                },
            }

        if step_type == "core.clip":
            params = {
                "source": self._resolve_operand(step["source"], alias_map),
                "target": step["target"],
            }
            if "min_value" in step:
                params["min_value"] = step["min_value"]
            if "max_value" in step:
                params["max_value"] = step["max_value"]
            return {"id": "core.clip", "params": params}

        # Fused operations
        if step_type == "graph.fused_neighbor_mul_agg":
            params = {
                "values": self._resolve_operand(step["values"], alias_map),
                "scalars": self._resolve_operand(step["scalars"], alias_map),
                "target": step["target"],
            }
            if "direction" in step:
                params["direction"] = step["direction"]
            return {"id": "graph.fused_neighbor_mul_agg", "params": params}

        if step_type == "core.fused_axpy":
            return {
                "id": "core.fused_axpy",
                "params": {
                    "a": self._resolve_operand(step["a"], alias_map),
                    "x": self._resolve_operand(step["x"], alias_map),
                    "b": self._resolve_operand(step["b"], alias_map),
                    "y": self._resolve_operand(step["y"], alias_map),
                    "target": step["target"],
                },
            }

        if step_type == "core.fused_madd":
            return {
                "id": "core.fused_madd",
                "params": {
                    "a": self._resolve_operand(step["a"], alias_map),
                    "b": self._resolve_operand(step["b"], alias_map),
                    "c": self._resolve_operand(step["c"], alias_map),
                    "target": step["target"],
                },
            }

        raise ValueError(f"Unsupported builder step type: {step_type}")

    def _validate(self) -> tuple[list[str], list[str]]:
        """
        Validate the pipeline.

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        defined_vars = set()

        for i, step in enumerate(self._steps):
            step_type = step.get("type")

            # Check for undefined variables in inputs
            if "input" in step:
                input_var = step["input"]
                if isinstance(input_var, str) and input_var not in defined_vars:
                    errors.append(
                        f"Step {i} ({step_type}): references undefined variable '{input_var}'"
                    )

            # Check for undefined variables in source field
            if "source" in step:
                source_var = step["source"]
                if isinstance(source_var, str) and source_var not in defined_vars:
                    errors.append(
                        f"Step {i} ({step_type}): references undefined variable '{source_var}'"
                    )

            # Check for undefined variables in left/right operands
            for operand in ["left", "right"]:
                if operand in step:
                    val = step[operand]
                    if isinstance(val, str) and val not in defined_vars:
                        errors.append(
                            f"Step {i} ({step_type}): references undefined variable '{val}'"
                        )

            # Check for undefined variables in map_nodes inputs
            if "inputs" in step and isinstance(step["inputs"], dict):
                for key, val in step["inputs"].items():
                    if val not in defined_vars:
                        errors.append(
                            f"Step {i} ({step_type}): input '{key}' references undefined variable '{val}'"
                        )

            # Track defined output variables (both "output" and "target")
            # Exceptions for redefinition warnings:
            #  - update_in_place / neighbor_mode_update intentionally redefine (in-place semantics)
            #  - alias steps generated by loop unrolling intentionally redefine to track iteration state
            in_place_steps = {"core.update_in_place", "core.neighbor_mode_update"}

            if "output" in step:
                output_var = step["output"]
                # Check if output name looks like it was generated by loop unrolling
                is_iteration_output = "_iter" in output_var
                if (
                    output_var in defined_vars
                    and step_type not in in_place_steps
                    and not is_iteration_output
                ):
                    warnings.append(
                        f"Step {i} ({step_type}): redefines variable '{output_var}'"
                    )
                defined_vars.add(output_var)

            if "target" in step:
                target_var = step["target"]
                # Don't warn about alias steps - they're meant to reassign variable names
                # This includes both loop-generated aliases and explicit var() reassignments
                if (
                    target_var in defined_vars
                    and step_type not in in_place_steps
                    and step_type != "alias"
                ):
                    warnings.append(
                        f"Step {i} ({step_type}): redefines variable '{target_var}'"
                    )
                defined_vars.add(target_var)

            # Special check: ensure alias steps are valid
            if step_type == "alias":
                source = step.get("source")
                if source and source not in defined_vars:
                    errors.append(
                        f"Step {i} (alias): source variable '{source}' is not defined"
                    )
                # Alias also creates its target variable
                target = step.get("target")
                if target:
                    defined_vars.add(target)

        # Check for empty pipelines
        if not self._steps:
            warnings.append("Pipeline has no steps")

        # Check if anything is attached
        has_attach = any(s.get("type") == "attach_attr" for s in self._steps)
        if not has_attach:
            warnings.append("Pipeline doesn't attach any output attributes")

        return errors, warnings

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
