"""
Algorithm Builder orchestrator with trait-based operations.

This module provides the main AlgorithmBuilder class that coordinates
domain-specific trait classes (CoreOps, GraphOps, AttrOps, IterOps).
"""

import importlib.util
# Import the original implementation temporarily during refactor
# This will be fully replaced as we migrate functionality
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from groggy import _groggy
from groggy.algorithms.base import AlgorithmHandle
from groggy.builder.execution import MessagePassContext
from groggy.builder.ir import IRGraph, IRNode
from groggy.builder.traits.attr import AttrOps
from groggy.builder.traits.core import CoreOps
from groggy.builder.traits.graph import GraphOps
from groggy.builder.traits.iter import IterOps
from groggy.builder.varhandle import GraphHandle, SubgraphHandle, VarHandle

# Load the original builder module for LoopContext and BuiltAlgorithm
builder_original_path = Path(__file__).parent.parent / "builder_original.py"
spec = importlib.util.spec_from_file_location(
    "builder_original",
    str(builder_original_path),
)
builder_original = importlib.util.module_from_spec(spec)
sys.modules["builder_original"] = builder_original
spec.loader.exec_module(builder_original)

# Import remaining classes from original
LoopContext = builder_original.LoopContext
BuiltAlgorithm = builder_original.BuiltAlgorithm


class AlgorithmBuilder:
    """
    Orchestrator for building algorithms via domain traits.

    This class coordinates trait namespaces (core, graph, attr, iter) and
    provides methods for algorithm composition with operator overloading support.

    Example (new style with operators):
        >>> builder = AlgorithmBuilder("pagerank")
        >>> G = builder.graph()
        >>> ranks = G.nodes(1.0 / G.N)
        >>> with builder.iter.loop(10):
        ...     neighbor_sum = G @ (ranks / (ranks.degrees() + 1e-9))
        ...     ranks = builder.var("ranks", 0.85 * neighbor_sum + 0.15 / G.N)

    Example (original style, backward compatible):
        >>> builder = AlgorithmBuilder("my_algo")
        >>> nodes = builder.init_nodes(1.0)
        >>> degrees = builder.node_degrees(nodes)
        >>> result = builder.core.add(nodes, degrees)
    """

    def __init__(self, name: str, use_ir: bool = True):
        """
        Create a new algorithm builder.

        Args:
            name: Name for the algorithm
            use_ir: If True, use typed IR graph (default); if False, use legacy steps list
        """
        self.name = name
        self.use_ir = use_ir

        # New IR-based representation
        self.ir_graph = IRGraph(name) if use_ir else None

        # Legacy step list (maintained for backward compatibility)
        self.steps = []

        self.variables = {}
        self._var_counter = 0
        self._input_ref = None
        self._graph_handle = None

        # Execution context tracking
        self._active_exec_context = None

        # Register trait namespaces
        self.core = CoreOps(self)
        self.graph_ops = GraphOps(self)
        self.attr = AttrOps(self)
        self.iter = IterOps(self)

    def graph(self) -> GraphHandle:
        """
        Get handle to the input graph.

        Returns:
            GraphHandle with topology methods

        Example:
            >>> G = builder.graph()
            >>> ranks = G.nodes(1.0 / G.N)
            >>> neighbor_sum = G @ ranks
        """
        if self._graph_handle is None:
            self._graph_handle = GraphHandle(self)
            # Allow access to builder from graph handle for convenience
            self._graph_handle.builder = self
        return self._graph_handle

    def message_pass(
        self,
        target: VarHandle,
        include_self: bool = True,
        ordered: bool = True,
        name: Optional[str] = None,
        **options,
    ) -> MessagePassContext:
        """
        Create a message-passing execution context for Gauss-Seidel style updates.

        Operations performed inside this context are captured and executed with
        special semantics: in-place updates, ordered traversal, and efficient
        neighbor aggregation.

        Args:
            target: Variable being updated in this block
            include_self: Whether to include self-loops in neighbor aggregation
            ordered: Whether to process nodes in order (Gauss-Seidel) or parallel (Jacobi)
            name: Optional name for this block (for debugging/profiling)
            **options: Additional options (tie_break, direction, etc.)

        Returns:
            MessagePassContext that can be used as a context manager

        Example:
            >>> labels = G.nodes(unique=True)
            >>> with builder.message_pass(target=labels, include_self=True, ordered=True) as mp:
            ...     msgs = mp.pull(labels)
            ...     update = builder.core.mode(msgs, tie_break="lowest")
            ...     mp.apply(update)
        """
        return MessagePassContext(
            self,
            target=target,
            include_self=include_self,
            ordered=ordered,
            name=name,
            **options,
        )

    def _new_var(self, prefix: str = "var") -> VarHandle:
        """
        Create a new unique variable.

        Args:
            prefix: Prefix for the variable name

        Returns:
            VarHandle with unique name
        """
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

    def var(self, name: str, value: VarHandle) -> VarHandle:
        """
        Create a named alias for a variable (for loop reassignment).

        Args:
            name: Logical name for the variable
            value: VarHandle to alias

        Returns:
            VarHandle with the logical name

        Example:
            >>> ranks = G.nodes(1.0)
            >>> with builder.iterate(10):
            ...     new_ranks = compute_update(ranks)
            ...     ranks = builder.var("ranks", new_ranks)  # Reassign for next iteration
        """
        self.steps.append({"type": "alias", "source": value.name, "target": name})
        # Return VarHandle that references the logical name
        if name not in self.variables:
            self.variables[name] = VarHandle(name, self)
        return self.variables[name]

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

    # Convenience methods (delegate to implementations or will move to traits)
    def constant(self, value: Any) -> VarHandle:
        """
        Create a constant value variable.

        Args:
            value: The constant value

        Returns:
            Variable handle for the constant

        Example:
            >>> damping = builder.constant(0.85)
            >>> ranks = ranks * damping
        """
        var = self._new_var("const")

        if self.use_ir and self.ir_graph is not None:
            from .ir.nodes import CoreIRNode

            node = CoreIRNode(
                node_id=f"node_{len(self.ir_graph.nodes)}",
                op_type="constant",
                inputs=[],
                output=var.name,
                value=value,
            )
            self._add_ir_node(node)
        else:
            self.steps.append({"type": "constant", "output": var.name, "value": value})

        return var

    def init_nodes(self, default: Any = 0.0, *, unique: bool = False) -> VarHandle:
        """
        Initialize node values.

        Args:
            default: Default value for all nodes (ignored if unique=True)
            unique: If True, initialize with sequential indices (0, 1, 2, ...)

        Returns:
            Variable handle for initialized values

        Example:
            >>> ranks = builder.init_nodes(default=1.0)
            >>> labels = builder.init_nodes(unique=True)  # For LPA
        """
        var = self._new_var("nodes")

        if self.use_ir and self.ir_graph is not None:
            from .ir.nodes import CoreIRNode

            op_type = "init_nodes_unique" if unique else "init_nodes"
            node = CoreIRNode(
                node_id=f"node_{len(self.ir_graph.nodes)}",
                op_type=op_type,
                inputs=[],
                output=var.name,
                **{"default": default} if not unique else {},
            )
            self._add_ir_node(node)
        else:
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

        Args:
            attr: Name of the node attribute to load
            default: Default value for nodes without the attribute

        Returns:
            Variable handle containing the attribute values

        Example:
            >>> weights = builder.load_attr("weight", default=1.0)
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

        Returns:
            Scalar variable handle containing the node count

        Example:
            >>> n = builder.graph_node_count()
            >>> uniform = builder.core.div(1.0, n)
        """
        var = self._new_var("n")

        if self.use_ir and self.ir_graph is not None:
            from .ir.nodes import GraphIRNode

            node = GraphIRNode(
                node_id=f"node_{len(self.ir_graph.nodes)}",
                op_type="graph_node_count",
                inputs=[],
                output=var.name,
            )
            self._add_ir_node(node)
        else:
            self.steps.append({"type": "graph_node_count", "output": var.name})

        return var

    def graph_edge_count(self) -> VarHandle:
        """
        Get the number of edges in the current subgraph as a scalar variable.

        Returns:
            Scalar variable handle containing the edge count

        Example:
            >>> m = builder.graph_edge_count()
        """
        var = self._new_var("m")

        if self.use_ir and self.ir_graph is not None:
            from .ir.nodes import GraphIRNode

            node = GraphIRNode(
                node_id=f"node_{len(self.ir_graph.nodes)}",
                op_type="graph_edge_count",
                inputs=[],
                output=var.name,
            )
            self._add_ir_node(node)
        else:
            self.steps.append({"type": "graph_edge_count", "output": var.name})

        return var

    def load_edge_attr(self, attr: str, default: Any = 0.0) -> VarHandle:
        """
        Load edge attribute values into a variable.

        Args:
            attr: Name of the edge attribute to load
            default: Default value for edges without the attribute

        Returns:
            Variable handle containing the edge attribute values

        Example:
            >>> edge_weights = builder.load_edge_attr("weight", default=1.0)
        """
        var = self._new_var("edge_attr")

        if self.use_ir and self.ir_graph is not None:
            from .ir.nodes import GraphIRNode

            node = GraphIRNode(
                node_id=f"node_{len(self.ir_graph.nodes)}",
                op_type="load_edge_attr",
                inputs=[],
                output=var.name,
                attr_name=attr,
                default=default,
            )
            self._add_ir_node(node)
        else:
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

        Example:
            >>> nodes = builder.init_nodes(1.0)
            >>> degrees = builder.node_degrees(nodes)
        """
        var = self._new_var("degrees")

        if self.use_ir and self.ir_graph is not None:
            from .ir.nodes import GraphIRNode

            node = GraphIRNode(
                node_id=f"node_{len(self.ir_graph.nodes)}",
                op_type="degree",
                inputs=[nodes.name],
                output=var.name,
            )
            self._add_ir_node(node)
        else:
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

        Example:
            >>> normalized = builder.normalize(values, method="sum")
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
            ...     neighbor_sum = builder.core.neighbor_agg(ranks, "sum")
            ...     ranks = builder.var("ranks", neighbor_sum)
        """
        return LoopContext(self, count)

    def _finalize_loop(
        self, start_step: int, iterations: int, loop_vars: Dict[str, VarHandle]
    ):
        """
        Unroll loop by repeating steps AND IR nodes.

        This is called by LoopContext.__exit__() to expand the loop body.

        Args:
            start_step: Index where loop body starts
            iterations: Number of times to repeat
            loop_vars: Variables at loop start
        """
        # First, unroll the steps using the original implementation
        from builder_original import AlgorithmBuilder as OriginalBuilder

        temp_builder = OriginalBuilder(self.name)
        temp_builder.steps = self.steps
        temp_builder.variables = self.variables
        temp_builder._var_counter = self._var_counter

        temp_builder._finalize_loop(start_step, iterations, loop_vars)

        # Copy back the modified state
        self.steps = temp_builder.steps
        self.variables = temp_builder.variables
        self._var_counter = temp_builder._var_counter

        # Now rebuild IR graph to reflect unrolled steps
        if self.use_ir:
            self._rebuild_ir_from_steps()

    def _unroll_ir_loop(
        self, start_step: int, iterations: int, loop_vars: Dict[str, VarHandle]
    ):
        """
        Unroll IR nodes to match the unrolled steps.

        The steps have already been unrolled by _finalize_loop. Now we need to
        replicate the IR nodes and update their variable references to match.

        SIMPLE APPROACH: Just rebuild the IR from the final (unrolled) steps.
        This works because steps are the source of truth after _finalize_loop.

        Args:
            start_step: Index where loop body started (in original steps)
            iterations: Number of iterations
            loop_vars: Variables at loop start
        """
        from groggy.builder.ir.graph import IRGraph
        from groggy.builder.ir.nodes import (ControlIRNode, CoreIRNode,
                                             GraphIRNode)

        # Rebuild the IR from updated steps
        self._rebuild_ir_from_steps()

    def attach_as(self, attr_name: str, values: VarHandle):
        """
        Attach computed values as a node attribute.

        Args:
            attr_name: Name for the output attribute
            values: VarHandle to attach

        Example:
            >>> ranks = compute_pagerank(...)
            >>> builder.attach_as("pagerank", ranks)
        """
        self.steps.append(
            {"type": "attach_attr", "input": values.name, "attr_name": attr_name}
        )

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

    def build(self, validate: bool = True, optimize: bool = True) -> AlgorithmHandle:
        """
        Build the algorithm from accumulated steps.

        Args:
            validate: If True, validate pipeline before building (default: True)
            optimize: If True, apply IR optimization passes (default: True)

        Returns:
            AlgorithmHandle ready for execution

        Raises:
            ValidationError: If validation fails and validate=True
        """
        # Apply IR optimization if enabled
        steps = self.steps
        has_aliases = any(step.get("type") == "alias" for step in self.steps)

        if optimize and self.use_ir and self.ir_graph is not None:
            from groggy.builder.ir.optimizer import optimize_ir

            # Ensure IR reflects latest step list before optimization
            self._rebuild_ir_from_steps()

            # Run optimization passes (returns (ir_graph, variable_renames))
            self.ir_graph, variable_renames = optimize_ir(
                self.ir_graph, passes=None, max_iterations=3, return_renames=True
            )

            # Regenerate steps from optimized IR
            ir_steps = self.ir_graph.to_steps()

            # Ensure scalar operands are materialized as constants where required
            ir_steps = self._materialize_scalar_operands(ir_steps)

            # Extract alias steps from original steps
            alias_steps = [s for s in self.steps if s.get("type") == "alias"]
            ir_managed_side_effects = {"iter.loop", "core.execution_block"}
            side_effect_steps = [
                s
                for s in self.steps
                if s.get("type") != "alias"
                and s.get("type") not in ir_managed_side_effects
                and not s.get("output")
            ]

            # Update alias steps to track through fusion transformations
            expanded_renames = None
            if alias_steps and variable_renames:
                # Expand variable_renames to include iteration-specific mappings
                # e.g., if mul_3 -> add_5, also add mul_3_iter0 -> add_5_iter0
                expanded_renames = self._expand_renames_for_iterations(
                    variable_renames, alias_steps
                )
                alias_steps = self._apply_renames_to_aliases(
                    alias_steps, expanded_renames
                )
            if side_effect_steps and variable_renames:
                if expanded_renames is None:
                    expanded_renames = self._expand_renames_for_iterations(
                        variable_renames, alias_steps
                    )
                side_effect_steps = [
                    self._apply_renames_to_step_fields(step, expanded_renames)
                    for step in side_effect_steps
                ]
            if expanded_renames:
                for handle in self.variables.values():
                    handle.name = self._resolve_rename(handle.name, expanded_renames)

            # Merge IR steps and alias steps in dependency order
            steps = self._merge_steps_topologically(ir_steps, alias_steps)
            steps.extend(side_effect_steps)

        algo = BuiltAlgorithm(self.name, steps)

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

    def _expand_renames_for_iterations(self, variable_renames, alias_steps):
        """
        Expand variable renames to include iteration-specific variants.

        When loops are unrolled, variables get iteration suffixes (e.g., mul_3_iter0).
        But the optimizer only knows about base renames (e.g., mul_3 -> add_5).
        This method creates iteration-specific mappings for all iteration variants
        found in the alias steps.

        Args:
            variable_renames: Base variable rename dict (e.g., {'mul_3': 'add_5'})
            alias_steps: List of alias steps that may reference iteration variants

        Returns:
            Expanded rename dict including iteration-specific mappings
        """
        import re

        expanded = dict(variable_renames)  # Start with base renames

        # Pattern to match iteration-specific variables: varname_iter123
        iter_pattern = re.compile(r"^(.+)_iter(\d+)$")

        # Collect all iteration-specific variables from alias steps
        iter_vars = set()
        for step in alias_steps:
            source = step.get("source")
            if source:
                match = iter_pattern.match(source)
                if match:
                    iter_vars.add(source)

        # For each iteration variable, check if we have a base rename
        for iter_var in iter_vars:
            match = iter_pattern.match(iter_var)
            if match:
                base_var = match.group(1)  # e.g., 'mul_3' from 'mul_3_iter0'
                iter_num = match.group(2)  # e.g., '0' from 'mul_3_iter0'

                # If we have a base rename, create the iteration-specific rename
                if base_var in variable_renames:
                    renamed_base = variable_renames[base_var]  # e.g., 'add_5'
                    renamed_iter = (
                        f"{renamed_base}_iter{iter_num}"  # e.g., 'add_5_iter0'
                    )
                    expanded[iter_var] = renamed_iter

        return expanded

    def _materialize_scalar_operands(self, steps):
        """
        Ensure operations that expect variable operands don't receive raw literals.

        Some optimizer passes (e.g., constant folding) can inline literal values
        into fields such as ``if_true``/``if_false`` on ``core.where`` steps.
        Rust step implementations expect these operands to reference variables,
        so we insert constant- or broadcast-producing steps and rewrite the
        operands to point at generated variables.
        """
        materialized = []
        constant_cache: Dict[float, str] = {}
        broadcast_cache: Dict[tuple, str] = {}

        # Track which step defines each variable (before inserting new ones)
        definition_map: Dict[str, Dict] = {}
        for step in steps:
            output = step.get("output")
            if output is not None:
                definition_map[output] = step

        i = 0
        while i < len(steps):
            step = steps[i]
            step_type = step.get("type")

            if step_type in ("core.where", "where"):
                # Make sure condition references a map. If the condition was
                # constant-folded to a scalar, broadcast it to match operand size.
                cond_key = "condition" if "condition" in step else "mask"
                cond_value = step.get(cond_key)
                if isinstance(cond_value, str):
                    defining_step = definition_map.get(cond_value)
                    if defining_step and defining_step.get("type") in {
                        "core.constant",
                        "init_scalar",
                        "core.init_scalar",
                    }:
                        # Determine a reference map (prefer if_true, otherwise if_false)
                        ref_var = None
                        for candidate in (step.get("if_true"), step.get("if_false")):
                            if isinstance(candidate, str):
                                ref_def = definition_map.get(candidate)
                                if ref_def and ref_def.get("type") not in {
                                    "core.constant",
                                    "init_scalar",
                                    "core.init_scalar",
                                }:
                                    ref_var = candidate
                                    break
                        if ref_var is None:
                            # Fallback to whatever the step produces; not ideal but keeps pipeline valid.
                            if isinstance(step.get("output"), str):
                                ref_var = step["output"]

                        if ref_var is not None:
                            cache_key = (cond_value, ref_var)
                            broadcast_name = broadcast_cache.get(cache_key)
                            if broadcast_name is None:
                                broadcast_handle = self._new_var("broadcast_inline")
                                broadcast_name = broadcast_handle.name
                                broadcast_cache[cache_key] = broadcast_name
                                broadcast_step = {
                                    "type": "core.broadcast_scalar",
                                    "output": broadcast_name,
                                    "scalar": cond_value,
                                    "reference": ref_var,
                                }
                                materialized.append((i, broadcast_step))
                                definition_map[broadcast_name] = broadcast_step
                            step[cond_key] = broadcast_name

                for field in ("if_true", "if_false"):
                    value = step.get(field)
                    if isinstance(value, (int, float)):
                        cache_key = float(value)
                        const_name = constant_cache.get(cache_key)
                        if const_name is None:
                            const_handle = self._new_var("const_inline")
                            const_name = const_handle.name
                            constant_cache[cache_key] = const_name
                            const_step = {
                                "type": "core.constant",
                                "output": const_name,
                                "value": value,
                            }
                            materialized.append((i, const_step))
                            definition_map[const_name] = const_step
                        step[field] = const_name
            i += 1

        # Insert constant steps before their use, adjusting for prior insertions.
        offset = 0
        for index, const_step in materialized:
            steps.insert(index + offset, const_step)
            offset += 1

        return steps

    def _apply_renames_to_aliases(self, steps, variable_renames):
        """
        Update alias steps to use renamed variables after fusion.

        When fusion combines operations, intermediate variables are eliminated.
        Alias steps that reference those variables need to be updated to point
        to the final output of the fused operation.

        Args:
            steps: List of step dictionaries
            variable_renames: Dict mapping old variable names to new names

        Returns:
            Updated list of steps
        """

        # Follow rename chains to get final variable names
        def follow_rename_chain(var_name):
            """Follow variable renames to the final name."""
            visited = set()
            current = var_name
            while current in variable_renames and current not in visited:
                visited.add(current)
                current = variable_renames[current]
            return current

        updated_steps = []
        for step in steps:
            if step.get("type") == "alias":
                # Update the 'source' field if it was renamed
                source_var = step.get("source")
                if source_var:
                    final_var = self._resolve_rename(source_var, variable_renames)
                    if final_var != source_var:
                        step = step.copy()
                        step["source"] = final_var
            updated_steps.append(step)
        return updated_steps

    def _apply_renames_to_step_fields(self, step, variable_renames):
        """
        Apply variable renames to common step fields (input/source/mask/etc.).
        """
        updated = step.copy()
        for key in (
            "input",
            "source",
            "mask",
            "condition",
            "reference",
            "scalar",
            "values",
        ):
            if key in updated and isinstance(updated[key], str):
                updated[key] = self._resolve_rename(updated[key], variable_renames)

        if "inputs" in updated and isinstance(updated["inputs"], dict):
            updated["inputs"] = {
                k: (
                    self._resolve_rename(v, variable_renames)
                    if isinstance(v, str)
                    else v
                )
                for k, v in updated["inputs"].items()
            }

        return updated

    def _resolve_rename(self, name: str, variable_renames: Dict[str, str]) -> str:
        """Follow rename chain to final variable name."""
        current = name
        visited = set()
        while current in variable_renames and current not in visited:
            visited.add(current)
            current = variable_renames[current]
        return current

    def _rebuild_ir_from_steps(self):
        """
        Reconstruct the IR graph from the current step list.
        """
        if not self.use_ir:
            return

        import copy

        from groggy.builder.ir.graph import IRGraph
        from groggy.builder.ir.nodes import (AttrIRNode, ControlIRNode,
                                             CoreIRNode, ExecutionBlockNode,
                                             GraphIRNode, LoopIRNode)

        graph_name = self.ir_graph.name if self.ir_graph is not None else self.name
        new_ir = IRGraph(graph_name)

        for step in self.steps:
            step_type = step.get("type", "")
            if step_type == "alias":
                continue

            if step_type == "iter.loop":
                iterations = step.get("iterations") or step.get("count") or 1
                body = copy.deepcopy(step.get("body", []))
                loop_vars = step.get("loop_vars")
                batch_plan = step.get("batch_plan")
                node = LoopIRNode(
                    node_id=f"node_{len(new_ir.nodes)}",
                    iterations=int(iterations),
                    body=body,
                    loop_vars=list(loop_vars) if loop_vars else None,
                    batch_plan=copy.deepcopy(batch_plan) if batch_plan else None,
                )
                new_ir.add_node(node)
                continue

            if step_type == "core.execution_block":
                node_id = f"node_{len(new_ir.nodes)}"
                target = step.get("target") or step.get("output")
                mode = step.get("mode", "message_pass")
                options = step.get("options", {})
                node = ExecutionBlockNode(
                    node_id=node_id,
                    mode=mode,
                    target=target,
                    **options,
                )
                body_nodes = step.get("body", {}).get("nodes", [])
                for body_node in body_nodes:
                    node.metadata.setdefault("body_nodes", []).append(
                        copy.deepcopy(body_node)
                    )
                new_ir.add_node(node)
                continue

            output = step.get("output")
            if not output:
                continue

            if "." in step_type:
                domain_str, op_type = step_type.split(".", 1)
            else:
                domain_str = "core"
                op_type = step_type

            inputs = []
            for field in [
                "input",
                "source",
                "a",
                "b",
                "left",
                "right",
                "scalar",
                "reference",
                "condition",
                "if_true",
                "if_false",
                "weights",
                "values",
            ]:
                value = step.get(field)
                if isinstance(value, str):
                    inputs.append(value)
            if op_type == "map_nodes":
                for value in step.get("inputs", {}).values():
                    if isinstance(value, str):
                        inputs.append(value)

            metadata = {}
            skip_fields = {
                "type",
                "output",
                "input",
                "source",
                "a",
                "b",
                "left",
                "right",
                "scalar",
                "reference",
                "condition",
                "if_true",
                "if_false",
                "weights",
                "values",
                "inputs",
            }
            for key, value in step.items():
                if key not in skip_fields:
                    metadata[key] = value
            if op_type == "map_nodes":
                metadata["map_inputs"] = step.get("inputs", {})
                metadata["fn"] = step.get("fn", "")
                metadata["async_update"] = step.get("async_update", False)

            node_id = f"node_{len(new_ir.nodes)}"
            if domain_str == "graph":
                node = GraphIRNode(
                    node_id=node_id,
                    op_type=op_type,
                    inputs=inputs,
                    output=output,
                    **metadata,
                )
            elif domain_str == "control":
                node = ControlIRNode(
                    node_id=node_id,
                    op_type=op_type,
                    inputs=inputs,
                    output=output,
                    **metadata,
                )
            elif domain_str == "attr":
                node = AttrIRNode(
                    node_id=node_id,
                    op_type=op_type,
                    inputs=inputs,
                    output=output,
                    **metadata,
                )
            else:
                node = CoreIRNode(
                    node_id=node_id,
                    op_type=op_type,
                    inputs=inputs,
                    output=output,
                    **metadata,
                )

            new_ir.add_node(node)

        self.ir_graph = new_ir

    def _merge_steps_topologically(self, ir_steps, alias_steps):
        """
        Merge IR and alias steps in dependency order.

        Ensures each alias step comes immediately after the step that defines
        its source variable, maintaining correct execution order.

        Args:
            ir_steps: List of IR-generated steps
            alias_steps: List of alias steps with updated sources

        Returns:
            Merged list of steps in correct order
        """
        if not alias_steps:
            return ir_steps

        from collections import defaultdict

        alias_by_source = defaultdict(list)
        fallback_aliases = []

        # Preserve original alias order when multiple map to same source
        for alias in alias_steps:
            source = alias.get("source")
            if source:
                alias_by_source[source].append(alias)
            else:
                fallback_aliases.append(alias)

        merged = []
        for step in ir_steps:
            merged.append(step)
            output = step.get("output")
            if output and output in alias_by_source:
                merged.extend(alias_by_source.pop(output))

        # Append any aliases whose sources were not part of the optimized IR
        for leftover in alias_by_source.values():
            merged.extend(leftover)
        merged.extend(fallback_aliases)

        return merged

    # ====================================================================
    # IR Support Methods (Phase 1: IR Foundation)
    # ====================================================================

    def _add_ir_node(self, node: IRNode):
        """
        Add an IR node to the graph (if using IR mode).

        Also adds a legacy step for backward compatibility.

        Args:
            node: IR node to add
        """
        context = getattr(self, "_active_exec_context", None)

        if self.use_ir and self.ir_graph is not None:
            self.ir_graph.add_node(node)
            if context is not None:
                context.capture_node(node)

        # Only record legacy steps when not inside a structured execution block
        if context is None:
            self.steps.append(node.to_step())

        return node

    def _remove_ir_nodes(self, nodes: List[IRNode]) -> None:
        """
        Remove a collection of IR nodes (used for execution block capture).
        """
        if not self.ir_graph or not nodes:
            return

        remove_ids = {node.id for node in nodes}
        if not remove_ids:
            return

        self.ir_graph.nodes = [
            node for node in self.ir_graph.nodes if node.id not in remove_ids
        ]
        for node_id in remove_ids:
            self.ir_graph.node_map.pop(node_id, None)

        # Rebuild tracking to drop stale references
        self.ir_graph.rebuild_var_tracking()

    def get_ir_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the IR graph.

        Returns:
            Dictionary with node counts, variable counts, and domain distribution
        """
        if not self.use_ir or self.ir_graph is None:
            return {"use_ir": False, "total_steps": len(self.steps)}

        return self.ir_graph.stats()

    def visualize_ir(self, format: str = "text") -> str:
        """
        Visualize the IR graph.

        Args:
            format: Output format - "text" for pretty-print, "dot" for Graphviz

        Returns:
            String representation of the IR

        Example:
            >>> print(builder.visualize_ir("text"))
            >>> # Or save DOT for visualization
            >>> with open("algorithm.dot", "w") as f:
            ...     f.write(builder.visualize_ir("dot"))
        """
        if not self.use_ir or self.ir_graph is None:
            return "IR mode not enabled"

        if format == "dot":
            return self.ir_graph.to_dot()
        elif format == "text":
            return self.ir_graph.pretty_print()
        else:
            raise ValueError(f"Unknown format: {format}. Use 'text' or 'dot'")

    def _get_steps_from_ir(self) -> list:
        """
        Generate legacy steps list from IR graph.

        Returns:
            List of step dictionaries for FFI serialization
        """
        if self.use_ir and self.ir_graph is not None:
            return self.ir_graph.to_steps()
        return self.steps


def builder(name: str) -> AlgorithmBuilder:
    """
    Create a new algorithm builder (convenience function).

    Args:
        name: Algorithm name

    Returns:
        New AlgorithmBuilder instance

    Example:
        >>> b = builder("my_algo")
    """
    return AlgorithmBuilder(name)
