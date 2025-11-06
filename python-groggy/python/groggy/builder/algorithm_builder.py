"""
Algorithm Builder orchestrator with trait-based operations.

This module provides the main AlgorithmBuilder class that coordinates
domain-specific trait classes (CoreOps, GraphOps, AttrOps, IterOps).
"""

from typing import Any, Dict, Optional
from groggy import _groggy
from groggy.algorithms.base import AlgorithmHandle
from groggy.builder.varhandle import VarHandle, SubgraphHandle, GraphHandle
from groggy.builder.traits.core import CoreOps
from groggy.builder.traits.graph import GraphOps
from groggy.builder.traits.attr import AttrOps
from groggy.builder.traits.iter import IterOps
from groggy.builder.ir import IRGraph, IRNode

# Import the original implementation temporarily during refactor
# This will be fully replaced as we migrate functionality
import sys
import importlib.util

# Load the original builder module for LoopContext and BuiltAlgorithm
spec = importlib.util.spec_from_file_location(
    "builder_original",
    "/Users/michaelroth/Documents/Code/groggy/python-groggy/python/groggy/builder_original.py"
)
builder_original = importlib.util.module_from_spec(spec)
sys.modules['builder_original'] = builder_original
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
        self.steps.append({
            "type": "alias",
            "source": value.name,
            "target": name
        })
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
                value=value
            )
            self._add_ir_node(node)
        else:
            self.steps.append({
                "type": "constant",
                "output": var.name,
                "value": value
            })
        
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
                **{"default": default} if not unique else {}
            )
            self._add_ir_node(node)
        else:
            if unique:
                self.steps.append({
                    "type": "init_nodes_with_index",
                    "output": var.name
                })
            else:
                self.steps.append({
                    "type": "init_nodes",
                    "output": var.name,
                    "default": default
                })
        
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
        self.steps.append({
            "type": "load_attr",
            "attr_name": attr,
            "default": default,
            "output": var.name
        })
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
                output=var.name
            )
            self._add_ir_node(node)
        else:
            self.steps.append({
                "type": "graph_node_count",
                "output": var.name
            })
        
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
                output=var.name
            )
            self._add_ir_node(node)
        else:
            self.steps.append({
                "type": "graph_edge_count",
                "output": var.name
            })
        
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
                default=default
            )
            self._add_ir_node(node)
        else:
            self.steps.append({
                "type": "load_edge_attr",
                "attr_name": attr,
                "default": default,
                "output": var.name
            })
        
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
                output=var.name
            )
            self._add_ir_node(node)
        else:
            self.steps.append({
                "type": "node_degree",
                "source": nodes.name,
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
            
        Example:
            >>> normalized = builder.normalize(values, method="sum")
        """
        var = self._new_var("normalized")
        self.steps.append({
            "type": "normalize",
            "input": values.name,
            "output": var.name,
            "method": method
        })
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
    
    def _finalize_loop(self, start_step: int, iterations: int, loop_vars: Dict[str, VarHandle]):
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
        
        # Now unroll the IR graph if we're using IR
        if self.use_ir and self.ir_graph is not None:
            self._unroll_ir_loop(start_step, iterations, loop_vars)
    
    def _unroll_ir_loop(self, start_step: int, iterations: int, loop_vars: Dict[str, VarHandle]):
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
        from groggy.builder.ir.nodes import CoreIRNode, GraphIRNode, ControlIRNode
        
        # Create new IR graph from scratch
        new_ir = IRGraph(self.ir_graph.name)
        
        # Convert each step into an IR node
        for step in self.steps:
            # Skip steps that don't create outputs (like alias, attach_attr)
            output = step.get("output")
            if not output:
                continue
            
            step_type = step.get("type", "")
            
            # Skip alias steps - they don't create new values in IR
            if step_type == "alias":
                continue
            
            # Determine domain and op_type
            if "." in step_type:
                domain_str, op_type = step_type.split(".", 1)
            else:
                # Legacy types - default to core
                domain_str = "core"
                op_type = step_type
            
            # Collect input variable names
            inputs = []
            for field in ["input", "source", "a", "b", "left", "right", "scalar", "reference", 
                          "condition", "if_true", "if_false", "weights", "values"]:
                value = step.get(field)
                if value and isinstance(value, str):
                    inputs.append(value)
            
            # Collect metadata - all non-standard fields
            metadata = {}
            skip_fields = {"type", "output", "input", "source", "a", "b", "left", "right", 
                          "scalar", "reference", "condition", "if_true", "if_false", "weights", "values"}
            for key, value in step.items():
                if key not in skip_fields:
                    metadata[key] = value
            
            # Create appropriate IR node type
            node_id = f"node_{len(new_ir.nodes)}"
            
            if domain_str == "control":
                node = ControlIRNode(
                    node_id=node_id,
                    op_type=op_type,
                    inputs=inputs,
                    output=output,
                    **metadata
                )
            elif domain_str == "graph":
                node = GraphIRNode(
                    node_id=node_id,
                    op_type=op_type,
                    inputs=inputs,
                    output=output,
                    **metadata
                )
            else:  # core or unknown
                node = CoreIRNode(
                    node_id=node_id,
                    op_type=op_type,
                    inputs=inputs,
                    output=output,
                    **metadata
                )
            
            new_ir.add_node(node)
        
        # Replace the IR graph
        self.ir_graph = new_ir
    
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
        self.steps.append({
            "type": "attach_attr",
            "input": values.name,
            "attr_name": attr_name
        })
    
    def map_nodes(
        self,
        fn: str,
        inputs: Optional[Dict[str, VarHandle]] = None,
        async_update: bool = False
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
                raise ValueError("async_update=True requires at least one input variable")
            # Use the first input variable as both source and output so the step operates in place.
            first_input_name = next(iter(input_vars.values()))
            output_name = first_input_name
            handle = VarHandle(output_name, self)
        else:
            handle = self._new_var("mapped")
            output_name = handle.name
        
        self.steps.append({
            "type": "map_nodes",
            "fn": fn,
            "inputs": input_vars,
            "output": output_name,
            "async_update": async_update
        })
        
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
            
            # Run optimization passes (returns (ir_graph, variable_renames))
            self.ir_graph, variable_renames = optimize_ir(
                self.ir_graph, passes=None, max_iterations=3, return_renames=True
            )
            
            # Regenerate steps from optimized IR
            ir_steps = self.ir_graph.to_steps()
            
            # Extract alias steps from original steps
            alias_steps = [s for s in self.steps if s.get("type") == "alias"]
            
            # Update alias steps to track through fusion transformations
            if alias_steps and variable_renames:
                alias_steps = self._apply_renames_to_aliases(alias_steps, variable_renames)
            
            # Merge IR steps and alias steps in dependency order
            steps = self._merge_steps_topologically(ir_steps, alias_steps)
        
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
                    final_var = follow_rename_chain(source_var)
                    if final_var != source_var:
                        step = step.copy()
                        step["source"] = final_var
            updated_steps.append(step)
        return updated_steps
    
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
        
        # Build map: variable name -> step index that defines it
        var_to_step_idx = {}
        for i, step in enumerate(ir_steps):
            output = step.get('output')
            if output:
                var_to_step_idx[output] = i
        
        # Insert alias steps after their source variable is defined
        merged = list(ir_steps)
        inserted_count = 0
        
        for alias in alias_steps:
            source = alias.get('source')
            if source and source in var_to_step_idx:
                # Insert immediately after the step that defines this variable
                insert_pos = var_to_step_idx[source] + 1 + inserted_count
                merged.insert(insert_pos, alias)
                inserted_count += 1
            else:
                # Source not found in IR steps, append at end
                merged.append(alias)
        
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
        if self.use_ir and self.ir_graph is not None:
            self.ir_graph.add_node(node)
        
        # Always maintain legacy steps list for backward compatibility
        self.steps.append(node.to_step())
    
    def get_ir_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the IR graph.
        
        Returns:
            Dictionary with node counts, variable counts, and domain distribution
        """
        if not self.use_ir or self.ir_graph is None:
            return {
                "use_ir": False,
                "total_steps": len(self.steps)
            }
        
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
