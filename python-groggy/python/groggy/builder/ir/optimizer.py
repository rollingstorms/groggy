"""
IR optimization passes for the builder DSL.

Implements dead code elimination, constant folding, common subexpression
elimination, and operation fusion to reduce IR size and improve execution efficiency.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .graph import IRGraph
from .nodes import AnyIRNode, CoreIRNode, IRDomain


class IROptimizer:
    """Applies optimization passes to IR graphs."""

    def __init__(self, ir_graph: IRGraph):
        self.ir = ir_graph
        self.modified = False
        self.variable_renames: Dict[str, str] = (
            {}
        )  # Track variable renames during optimization

    def optimize(self, passes: Optional[List[str]] = None) -> bool:
        """
        Run optimization passes on the IR graph.

        Args:
            passes: List of pass names to run. If None, runs all passes.
                   Available: 'dce', 'constant_fold', 'cse', 'fuse_arithmetic', 'fuse_neighbor'

        Returns:
            True if any modifications were made
        """
        if passes is None:
            # Skip DCE by default - it's too aggressive before outputs are attached
            passes = ["constant_fold", "cse", "fuse_neighbor"]

        total_modified = False

        for pass_name in passes:
            if pass_name == "dce":
                modified = self.dead_code_elimination()
            elif pass_name == "constant_fold":
                modified = self.constant_folding()
            elif pass_name == "cse":
                modified = self.common_subexpression_elimination()
            elif pass_name == "fuse_arithmetic":
                modified = self.fuse_arithmetic()
            elif pass_name == "fuse_neighbor":
                modified = self.fuse_neighbor_operations()
            else:
                raise ValueError(f"Unknown optimization pass: {pass_name}")

            total_modified = total_modified or modified

        return total_modified

    def dead_code_elimination(self) -> bool:
        """
        Remove operations that don't contribute to outputs.

        An operation is dead if:
        1. It has no side effects (not output, attach, etc.)
        2. Its result is never used by any live operation

        Returns:
            True if any nodes were removed
        """
        live_nodes = self._mark_live_nodes()

        dead_nodes = []
        for node in self.ir.nodes:
            if node.id not in live_nodes:
                dead_nodes.append(node)

        if dead_nodes:
            for node in dead_nodes:
                self.ir.nodes.remove(node)
                del self.ir.node_map[node.id]
                if node.output in self.ir.var_defs:
                    del self.ir.var_defs[node.output]
            return True

        return False

    def _mark_live_nodes(self) -> Set[str]:
        """
        Mark all nodes that contribute to outputs or have side effects.
        Uses backward reachability from output nodes.
        """
        live = set()
        worklist = []

        # Start with nodes that have side effects (outputs, attachments)
        for node in self.ir.nodes:
            if self._has_side_effects(node):
                live.add(node.id)
                worklist.append(node)

        # Backward traversal to mark dependencies
        while worklist:
            current = worklist.pop()
            deps = self.ir.get_dependencies(current)
            for dep in deps:
                if dep.id not in live:
                    live.add(dep.id)
                    worklist.append(dep)

        return live

    def _has_side_effects(self, node: AnyIRNode) -> bool:
        """Check if a node has side effects (outputs, attachments, etc.)."""
        # Attr attach operations have side effects
        if node.domain == IRDomain.ATTR and node.op_type in ("attach", "store"):
            return True
        # Control flow operations have side effects
        if node.domain == IRDomain.CONTROL:
            return True
        return False

    def constant_folding(self) -> bool:
        """
        Evaluate constant expressions at compile time.

        Folds operations like:
        - scalar + scalar -> scalar
        - scalar * scalar -> scalar
        - constant broadcasts

        Returns:
            True if any constants were folded
        """
        modified = False

        for node in list(self.ir.nodes):
            folded = self._try_fold_constant(node)
            if folded is not None:
                # Update node to be a constant
                node.op_type = "constant"
                node.inputs = []
                node.metadata["value"] = folded
                modified = True

        return modified

    def _try_fold_constant(self, node: AnyIRNode) -> Optional[Any]:
        """Try to fold a node into a constant value."""
        # Only fold pure arithmetic on core domain
        if node.domain != IRDomain.CORE:
            return None

        if node.op_type not in {"add", "mul", "sub", "div"}:
            return None

        # Check if all inputs are constants
        input_values = []
        for input_var in node.inputs:
            def_node = self.ir.get_defining_node(input_var)
            if def_node and def_node.op_type == "constant":
                input_values.append(def_node.metadata.get("value"))
            else:
                return None  # Not all inputs are constant

        if len(input_values) < 2:
            return None

        # Perform the operation
        try:
            lhs, rhs = input_values[0], input_values[1]
            if node.op_type == "add":
                return lhs + rhs
            elif node.op_type == "mul":
                return lhs * rhs
            elif node.op_type == "sub":
                return lhs - rhs
            elif node.op_type == "div":
                if rhs == 0:
                    return None  # Don't fold division by zero
                return lhs / rhs
        except Exception:
            return None

        return None

    def common_subexpression_elimination(self) -> bool:
        """
        Eliminate redundant computations.

        If two operations compute the same result from the same inputs,
        reuse the first computation and eliminate the second.

        Returns:
            True if any duplicates were eliminated
        """
        modified = False

        # Map from operation signature to output variable
        seen_ops: Dict[str, str] = {}
        replacements: Dict[str, str] = {}  # old_var -> new_var

        # Process nodes in order
        for node in self.ir.nodes:
            # Skip nodes with side effects
            if self._has_side_effects(node):
                continue

            # Skip nodes without outputs
            if not node.output:
                continue

            # Create signature for this operation
            signature = self._operation_signature(node)

            if signature in seen_ops:
                # Found duplicate - mark for replacement
                original_var = seen_ops[signature]
                replacements[node.output] = original_var
                modified = True
            else:
                seen_ops[signature] = node.output

        # Apply replacements
        if replacements:
            self._apply_replacements(replacements)

        return modified

    def _operation_signature(self, node: AnyIRNode) -> str:
        """
        Create a unique signature for an operation.
        Two nodes with the same signature compute the same result.
        """
        # Include domain and operation type
        sig_parts = [f"{node.domain.value}.{node.op_type}"]

        # Include sorted inputs
        input_str = ",".join(sorted(node.inputs))
        sig_parts.append(input_str)

        # Include relevant metadata (skip output names)
        meta_parts = []
        for key in sorted(node.metadata.keys()):
            if key not in {"output", "name", "id", "line"}:
                meta_parts.append(f"{key}={node.metadata[key]}")
        if meta_parts:
            sig_parts.append(",".join(meta_parts))

        return "::".join(sig_parts)

    def _apply_replacements(self, replacements: Dict[str, str]):
        """Replace all uses of old variable names with new ones."""
        # Update all node inputs
        for node in self.ir.nodes:
            node.inputs = [replacements.get(inp, inp) for inp in node.inputs]

        # Update var_defs and var_uses
        for old_var, new_var in replacements.items():
            if old_var in self.ir.var_defs:
                del self.ir.var_defs[old_var]
            if old_var in self.ir.var_uses:
                # Redirect uses
                for use_node in self.ir.var_uses[old_var]:
                    self.ir.var_uses[new_var].append(use_node)
                del self.ir.var_uses[old_var]

    def fuse_arithmetic(self) -> bool:
        """
        Fuse chains of arithmetic operations into single operations.

        Detects patterns like:
        - (a * b) + c -> fused_arithmetic("axpy", a, b, c)
        - (a + b) * c -> fused_arithmetic("mul_add", a, b, c)
        - a / (b + epsilon) -> fused_arithmetic("safe_div", a, b, epsilon)

        Returns:
            True if any operations were fused
        """
        modified = False

        # Look for fusable arithmetic patterns
        for node in list(self.ir.nodes):
            if node.domain != IRDomain.CORE:
                continue

            # Pattern: (a op1 b) op2 c -> fused
            fused = self._try_fuse_binary_chain(node)
            if fused:
                modified = True
                continue

            # Pattern: where(mask, a op b, c) -> fused conditional
            if node.op_type == "where":
                fused = self._try_fuse_conditional(node)
                if fused:
                    modified = True

        return modified

    def _try_fuse_binary_chain(self, node: AnyIRNode) -> bool:
        """Try to fuse a chain of binary operations."""
        if node.op_type not in {"add", "mul", "sub", "div"}:
            return False

        # Check if we can fuse with an input operation
        # For now, detect AXPY pattern: (a * b) + c
        if node.op_type == "add" and len(node.inputs) == 2:
            for i, inp_var in enumerate(node.inputs):
                inp_node = self.ir.get_defining_node(inp_var)
                if inp_node and inp_node.op_type == "mul" and len(inp_node.inputs) == 2:
                    # Found (a * b) + c pattern
                    other_inp = node.inputs[1 - i]

                    # Only fuse if mul result is only used here
                    uses = self.ir.var_uses.get(inp_var, [])
                    if len(uses) == 1:
                        # Create fused node
                        node.op_type = "fused_axpy"
                        node.inputs = list(inp_node.inputs) + [other_inp]
                        node.metadata["pattern"] = "axpy"

                        # Remove the mul node (will be cleaned up by DCE)
                        return True

        return False

    def _try_fuse_conditional(self, node: AnyIRNode) -> bool:
        """Try to fuse arithmetic operations within conditional (where)."""
        if node.op_type != "where" or len(node.inputs) < 3:
            return False

        # Check if true or false branches contain simple operations
        mask_var, true_var, false_var = node.inputs[0], node.inputs[1], node.inputs[2]

        # Pattern: where(mask, 0, a * b) -> fused_where_mul
        true_node = self.ir.get_defining_node(true_var)
        false_node = self.ir.get_defining_node(false_var)

        # Check for zero in one branch
        is_true_zero = (
            true_node
            and true_node.op_type == "constant"
            and true_node.metadata.get("value") == 0
        )
        is_false_zero = (
            false_node
            and false_node.op_type == "constant"
            and false_node.metadata.get("value") == 0
        )

        if is_false_zero and true_node and true_node.op_type in {"mul", "div", "add"}:
            # Fuse where(mask, a op b, 0) -> fused_where_op
            uses = self.ir.var_uses.get(true_var, [])
            if len(uses) == 1:
                node.op_type = f"fused_where_{true_node.op_type}"
                node.inputs = [mask_var] + list(true_node.inputs)
                node.metadata["pattern"] = f"where_{true_node.op_type}"
                return True

        return False

    def fuse_neighbor_operations(self) -> bool:
        """
        Fuse graph neighbor operations with arithmetic.

        Patterns detected:
        1. neighbor_agg(values) -> mul(scalars) -> FusedNeighborMulAgg
        2. mul(a, x) -> add(mul(b, y)) -> FusedAXPY (a*x + b*y)
        3. mul(a, b) -> add(c) -> FusedMADD (a*b + c)

        Returns:
            True if any operations were fused
        """
        modified = False

        # Pattern 1: neighbor_agg -> mul -> fused_neighbor_mul_agg
        if self._fuse_neighbor_mul_pattern():
            modified = True
            # Rebuild tracking immediately so next pass sees updated graph
            self.ir.rebuild_var_tracking()

        # Pattern 2 & 3: arithmetic chains
        if self._fuse_arithmetic_chains():
            modified = True
            # Rebuild tracking after arithmetic fusion too
            self.ir.rebuild_var_tracking()

        return modified

    def _is_scalar_variable(self, var_name: str) -> bool:
        """
        Check if a variable is a scalar (not a node-map).

        A variable is scalar if:
        - It's defined by a constant operation
        - It's defined by a scalar reduction (e.g., sum, mean)
        - It's defined by a recip of a scalar
        """
        def_node = self.ir.var_defs.get(var_name)
        if not def_node:
            return False

        # Check for operations that produce scalars
        scalar_ops = {
            "constant",
            "scalar",
            "sum",
            "mean",
            "min",
            "max",
            "norm",
            "graph_node_count",
            "graph_edge_count",
        }

        if def_node.op_type in scalar_ops:
            return True

        # Recip of a scalar is also scalar
        if def_node.op_type == "recip" and len(def_node.inputs) >= 1:
            return self._is_scalar_variable(def_node.inputs[0])

        # Division/multiplication of scalars produces scalar
        if def_node.op_type in ["div", "mul"] and len(def_node.inputs) >= 2:
            return self._is_scalar_variable(
                def_node.inputs[0]
            ) and self._is_scalar_variable(def_node.inputs[1])

        return False

    def _ensure_node_map(
        self, var_name: str, reference_var: str, insert_before_node: "AnyIRNode"
    ) -> str:
        """
        Ensure a variable is a node-map by inserting a broadcast if needed.

        Args:
            var_name: Variable to check/convert
            reference_var: Reference variable to get node count from
            insert_before_node: Node to insert broadcast before (if needed)

        Returns:
            Variable name (original if already node-map, new broadcast var if scalar)
        """
        if not self._is_scalar_variable(var_name):
            return var_name

        # Need to broadcast scalar to node-map
        # Create new broadcast node
        from .nodes import CoreIRNode

        # Make broadcast variable name unique by including a counter
        # This is critical for loops where the same scalar may be broadcast multiple times
        broadcast_var = f"{var_name}_bcast_{self.ir._node_counter}"

        broadcast_node = CoreIRNode(
            node_id=f"bcast_{self.ir._node_counter}",
            op_type="broadcast_scalar",
            inputs=[var_name, reference_var],
            output=broadcast_var,
            metadata={"broadcast_from": var_name},
        )
        self.ir._node_counter += 1

        # Insert broadcast node right before the node that will use it
        insert_pos = self.ir.nodes.index(insert_before_node)
        self.ir.nodes.insert(insert_pos, broadcast_node)
        self.ir.node_map[broadcast_node.id] = broadcast_node

        # Tracking will be rebuilt after fusion completes

        return broadcast_var

    def _fuse_neighbor_mul_pattern(self) -> bool:
        """
        Detect: neighbor_agg(values) -> mul(result, scalars) -> fused_neighbor_mul_agg(values, scalars)
        This is the critical PageRank/LPA pattern.

        If scalars is a scalar value, insert a broadcast before fusion.
        """
        modified = False

        for node in list(self.ir.nodes):
            if node.domain != IRDomain.GRAPH or node.op_type != "neighbor_agg":
                continue

            if not node.output:
                continue

            # Check if result is used by a multiplication
            uses = self.ir.var_uses.get(node.output, [])
            if len(uses) != 1:
                continue

            mul_node = uses[0]
            if mul_node.domain != IRDomain.CORE or mul_node.op_type != "mul":
                continue

            # mul_node.inputs should be [neighbor_agg_output, scalars]
            # Find the scalar input (the one that's not the neighbor_agg output)
            scalar_var = None
            for inp in mul_node.inputs:
                if inp != node.output:
                    scalar_var = inp
                    break

            if not scalar_var:
                continue

            # CRUCIAL FIX: Ensure scalar operand is a node-map (broadcast if needed)
            # The fused kernel expects both operands to be node-maps
            # Use the VALUES input as reference (it's stable and won't change during fusion)
            reference_var = node.inputs[0] if len(node.inputs) >= 1 else node.output
            scalar_var = self._ensure_node_map(scalar_var, reference_var, node)

            # Update mul_node's inputs to use the broadcast variable
            # This is important so when we remove mul_node later, references are correct
            for i, inp in enumerate(mul_node.inputs):
                if inp != node.output and inp != scalar_var:
                    mul_node.inputs[i] = scalar_var

            # Replace with fused operation
            # Track the rename: old neighbor_agg output -> mul's output
            # This is important for alias steps that reference intermediate variables
            old_output = node.output
            new_output = mul_node.output
            if old_output != new_output:
                self.variable_renames[old_output] = new_output

            # Change neighbor_agg node to fused_neighbor_mul_agg
            node.op_type = "fused_neighbor_mul_agg"
            # inputs[0] is values, add scalars as second input
            if len(node.inputs) >= 1:
                node.inputs = [node.inputs[0], scalar_var]
            node.output = new_output  # Take over mul's output
            node.metadata["fused"] = True
            node.metadata["direction"] = node.metadata.get("direction", "in")

            # Remove the mul node
            if mul_node in self.ir.nodes:
                self.ir.nodes.remove(mul_node)
                if mul_node.id in self.ir.node_map:
                    del self.ir.node_map[mul_node.id]

            modified = True

        return modified

    def _fuse_arithmetic_chains(self) -> bool:
        """
        Detect arithmetic chains that can be fused:
        - a*x + b*y -> FusedAXPY
        - a*b + c -> FusedMADD
        """
        modified = False

        for node in list(self.ir.nodes):
            if node.domain != IRDomain.CORE or node.op_type != "add":
                continue

            if len(node.inputs) < 2:
                continue

            # Get the two inputs to the add
            inp1_var, inp2_var = node.inputs[0], node.inputs[1]
            inp1_node = self.ir.get_defining_node(inp1_var)
            inp2_node = self.ir.get_defining_node(inp2_var)

            # Pattern: mul + mul -> check if it's AXPY (a*x + b*y)
            # But don't match if either mul is already a fused operation
            if (
                inp1_node
                and inp2_node
                and inp1_node.op_type == "mul"
                and inp2_node.op_type == "mul"
                and not inp1_node.metadata.get("fused")
                and not inp2_node.metadata.get("fused")
            ):
                # Check if each mul is only used by this add
                uses1 = self.ir.var_uses.get(inp1_var, [])
                uses2 = self.ir.var_uses.get(inp2_var, [])

                if len(uses1) == 1 and len(uses2) == 1:
                    # Fuse to AXPY: a*x + b*y
                    # inp1_node.inputs = [a, x], inp2_node.inputs = [b, y]
                    if len(inp1_node.inputs) >= 2 and len(inp2_node.inputs) >= 2:
                        # Track renames: mul outputs are eliminated
                        self.variable_renames[inp1_var] = node.output
                        self.variable_renames[inp2_var] = node.output

                        # CRUCIAL FIX: Ensure all operands are node-maps (broadcast scalars if needed)
                        # FusedAXPY expects: a, x, b, y all as node-maps
                        # Find a node-map to use as reference for broadcasting
                        stable_ref = None
                        for candidate in [
                            inp1_node.inputs[0],
                            inp1_node.inputs[1],
                            inp2_node.inputs[0],
                            inp2_node.inputs[1],
                        ]:
                            if not self._is_scalar_variable(candidate):
                                stable_ref = candidate
                                break
                        if not stable_ref:
                            # All inputs are scalars - shouldn't happen, use first as fallback
                            stable_ref = inp1_node.inputs[0]

                        a_var = self._ensure_node_map(
                            inp1_node.inputs[0], stable_ref, node
                        )
                        x_var = self._ensure_node_map(
                            inp1_node.inputs[1], stable_ref, node
                        )
                        b_var = self._ensure_node_map(
                            inp2_node.inputs[0], stable_ref, node
                        )
                        y_var = self._ensure_node_map(
                            inp2_node.inputs[1], stable_ref, node
                        )

                        node.op_type = "fused_axpy"
                        # AXPY params: a, x, b, y
                        node.inputs = [a_var, x_var, b_var, y_var]
                        node.metadata["fused"] = True
                        node.metadata["fused_inputs"] = {
                            "a": a_var,
                            "x": x_var,
                            "b": b_var,
                            "y": y_var,
                        }

                        # Remove the mul nodes
                        for n in [inp1_node, inp2_node]:
                            if n in self.ir.nodes:
                                self.ir.nodes.remove(n)
                                if n.id in self.ir.node_map:
                                    del self.ir.node_map[n.id]

                        modified = True
                        continue

            # Pattern: mul + scalar/vector -> FusedMADD (a*b + c)
            # But don't match if inp1_node is already a fused operation
            if (
                inp1_node
                and inp1_node.op_type == "mul"
                and not inp1_node.metadata.get("fused")
            ):
                uses1 = self.ir.var_uses.get(inp1_var, [])
                if len(uses1) == 1 and len(inp1_node.inputs) >= 2:
                    # Track rename: mul output is eliminated
                    self.variable_renames[inp1_var] = node.output

                    # CRUCIAL FIX: Ensure all operands are node-maps (broadcast scalars if needed)
                    # FusedMADD expects: a (node-map), b (node-map), c (node-map)
                    # Find a node-map to use as reference for broadcasting
                    # Try the MUL's inputs first, then fall back to inp2_var
                    stable_ref = None
                    for candidate in [
                        inp1_node.inputs[0],
                        inp1_node.inputs[1],
                        inp2_var,
                    ]:
                        if not self._is_scalar_variable(candidate):
                            stable_ref = candidate
                            break
                    if not stable_ref:
                        # All inputs are scalars - shouldn't happen, but use first input as fallback
                        stable_ref = inp1_node.inputs[0]

                    a_var = self._ensure_node_map(inp1_node.inputs[0], stable_ref, node)
                    b_var = self._ensure_node_map(inp1_node.inputs[1], stable_ref, node)
                    c_var = self._ensure_node_map(inp2_var, stable_ref, node)

                    # Fuse to MADD: a*b + c
                    node.op_type = "fused_madd"
                    node.inputs = [a_var, b_var, c_var]
                    node.metadata["fused"] = True
                    node.metadata["fused_inputs"] = {
                        "a": a_var,
                        "b": b_var,
                        "c": c_var,
                    }

                    # Remove the mul node
                    if inp1_node in self.ir.nodes:
                        self.ir.nodes.remove(inp1_node)
                        if inp1_node.id in self.ir.node_map:
                            del self.ir.node_map[inp1_node.id]

                    modified = True

        return modified


def optimize_ir(
    ir_graph: IRGraph,
    passes: Optional[List[str]] = None,
    max_iterations: int = 3,
    return_renames: bool = False,
) -> Union[IRGraph, Tuple[IRGraph, Dict[str, str]]]:
    """
    Apply optimization passes to an IR graph iteratively.

    Args:
        ir_graph: The IR graph to optimize
        passes: List of optimization passes to apply
        max_iterations: Maximum number of optimization iterations
        return_renames: If True, also return variable renames from fusion

    Returns:
        Optimized IR graph, or (ir_graph, variable_renames) if return_renames=True
    """
    optimizer = IROptimizer(ir_graph)

    for i in range(max_iterations):
        modified = optimizer.optimize(passes)
        if not modified:
            # Reached fixed point
            break

    if return_renames:
        return ir_graph, optimizer.variable_renames
    return ir_graph
