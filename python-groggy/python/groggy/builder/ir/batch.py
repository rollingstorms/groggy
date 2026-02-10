"""
Batch Execution Plan Generator

Compiles IR graphs into compact batch execution plans that can be sent
to Rust in a single FFI call, eliminating per-operation FFI overhead.
"""

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .graph import IRGraph
from .nodes import ControlIRNode, CoreIRNode, GraphIRNode, IRNode

SUPPORTED_BATCH_OPS: Set[str] = {
    # Core arithmetic
    "core.add",
    "core.sub",
    "core.mul",
    "core.div",
    "core.constant",
    "core.min",
    "core.max",
    "core.abs",
    "core.clip",
    "core.recip",
    "core.sqrt",
    "core.exp",
    "core.log",
    "core.pow",
    "core.compare",
    "core.where",
    "core.reduce_scalar",
    "core.broadcast_scalar",
    "init_scalar",
    "init_nodes",
    "load_attr",
    "load_edge_attr",
    "node_degree",
    "graph_node_count",
    "graph_edge_count",
    "normalize",
    "normalize_sum",
    "core.update_in_place",
    "core.neighbor_mode_update",
    "map_nodes",
    "core.collect_neighbor_values",
    "core.mode_list",
    # Graph operations
    "graph.neighbor_sum",
    "graph.neighbor_mean",
    "graph.neighbor_min",
    "graph.neighbor_max",
    "graph.neighbor_agg",
    # Loads/stores
    "init_nodes_with_index",
    "attach_attr",
    "alias",
}


@dataclass
class BatchExecutionPlan:
    """
    A compiled execution plan that packages multiple operations into a single
    FFI-friendly payload.

    The plan includes:
    - Topologically sorted operations
    - Variable lifetime tracking
    - Memory layout optimization
    - Compact binary or JSON representation
    """

    operations: List[Dict[str, Any]] = field(default_factory=list)
    variable_slots: Dict[str, int] = field(default_factory=dict)
    constant_values: Dict[str, Any] = field(default_factory=dict)
    max_live_variables: int = 0
    execution_order: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON for FFI"""
        # Convert operations to make sure enums are serializable
        serializable_ops = []
        for op in self.operations:
            op_copy = op.copy()
            # Convert enum to string if present
            if "domain" in op_copy and hasattr(op_copy["domain"], "value"):
                op_copy["domain"] = op_copy["domain"].value
            serializable_ops.append(op_copy)

        return json.dumps(
            {
                "operations": serializable_ops,
                "variable_slots": self.variable_slots,
                "constant_values": self.constant_values,
                "max_live_variables": self.max_live_variables,
                "execution_order": self.execution_order,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> "BatchExecutionPlan":
        """Deserialize from JSON"""
        obj = json.loads(data)
        return cls(**obj)

    def to_binary(self) -> bytes:
        """
        Serialize to compact binary format for performance-critical use.

        Format:
        - Header: magic number, version, operation count, variable count
        - Constants table: constant values referenced by operations
        - Operations: opcode, input slots, output slots, metadata
        - Variable slots: variable name → slot index mapping
        """
        # For now, use JSON; binary format is future optimization
        return self.to_json().encode("utf-8")

    @classmethod
    def from_binary(cls, data: bytes) -> "BatchExecutionPlan":
        """Deserialize from binary format"""
        return cls.from_json(data.decode("utf-8"))


class BatchPlanGenerator:
    """
    Generates batch execution plans from optimized IR graphs.

    The generator:
    1. Topologically sorts operations
    2. Assigns variable slots (register allocation)
    3. Tracks variable lifetimes
    4. Packs operations into compact representation
    """

    def __init__(self, ir_graph: IRGraph):
        self.ir_graph = ir_graph
        self.execution_order: List[str] = []
        self.variable_slots: Dict[str, int] = {}
        self.live_ranges: Dict[str, Tuple[int, int]] = (
            {}
        )  # var -> (first_use, last_use)
        self.next_slot = 0

    def generate(self) -> BatchExecutionPlan:
        """
        Generate a batch execution plan from the IR graph.

        Steps:
        1. Topological sort to get execution order
        2. Compute variable lifetimes
        3. Assign variable slots (register allocation)
        4. Pack operations into batch format
        """
        # Step 1: Topological sort
        self._compute_execution_order()

        # Step 2: Compute variable lifetimes
        self._compute_live_ranges()

        # Step 3: Assign variable slots
        self._assign_variable_slots()

        # Step 4: Pack operations
        operations = self._pack_operations()

        # Step 5: Extract constant values
        constant_values = self._extract_constants()

        return BatchExecutionPlan(
            operations=operations,
            variable_slots=self.variable_slots,
            constant_values=constant_values,
            max_live_variables=self.next_slot,
            execution_order=self.execution_order,
        )

    def _compute_execution_order(self):
        """
        Topological sort of IR nodes to determine execution order.

        Uses DFS-based topological sort.
        """
        visited = set()
        temp_mark = set()
        order = []

        def visit(node_id: str):
            if node_id in visited:
                return
            if node_id in temp_mark:
                raise ValueError(f"Cycle detected in IR graph at node {node_id}")

            temp_mark.add(node_id)
            node = self.ir_graph.node_map[node_id]

            # Visit dependencies first (inputs must be computed before this node)
            deps = self.ir_graph.get_dependencies(node)
            for dep_node in deps:
                visit(dep_node.id)

            temp_mark.remove(node_id)
            visited.add(node_id)
            order.append(node_id)

        # Visit all nodes
        for node in self.ir_graph.nodes:
            if node.id not in visited:
                visit(node.id)

        self.execution_order = order

    def _compute_live_ranges(self):
        """
        Compute the live range for each variable (first use to last use).

        This enables dead variable elimination and slot reuse.
        """
        for i, node_id in enumerate(self.execution_order):
            node = self.ir_graph.node_map[node_id]

            # Mark output as first defined at this point
            if node.output:
                output = node.output
                if output not in self.live_ranges:
                    self.live_ranges[output] = (i, i)
                else:
                    # Extend last use
                    first, _ = self.live_ranges[output]
                    self.live_ranges[output] = (first, i)

            # Mark inputs as last used at this point
            for input_var in node.inputs:
                if input_var not in self.live_ranges:
                    # Input defined externally (parameter or constant)
                    self.live_ranges[input_var] = (0, i)
                else:
                    first, _ = self.live_ranges[input_var]
                    self.live_ranges[input_var] = (first, i)

    def _assign_variable_slots(self):
        """
        Assign variable slots using a simple linear scan register allocation.

        Variables with non-overlapping lifetimes can share the same slot.
        """
        # Sort variables by start time
        vars_by_start = sorted(self.live_ranges.items(), key=lambda x: x[1][0])

        # Active variables (currently live) and free slots
        active: List[Tuple[str, int, int]] = []  # (var_name, end_time, slot_id)
        free_slots: List[int] = []

        for var_name, (start, end) in vars_by_start:
            # Remove expired variables and free their slots
            new_active = []
            for v, e, slot in active:
                if e > start:
                    new_active.append((v, e, slot))
                else:
                    free_slots.append(slot)
            active = new_active

            if var_name in self.variable_slots:
                continue

            # Assign slot (reuse first available, else allocate new)
            if free_slots:
                slot = free_slots.pop()
            else:
                slot = self.next_slot
                self.next_slot += 1

            self.variable_slots[var_name] = slot
            active.append((var_name, end, slot))

    def _pack_operations(self) -> List[Dict[str, Any]]:
        """
        Pack IR nodes into batch operation format.

        Each operation includes:
        - op_type: operation identifier
        - inputs: list of input variable slots
        - outputs: list of output variable slots
        - metadata: operation-specific parameters
        """
        operations = []

        for node_id in self.execution_order:
            node = self.ir_graph.node_map[node_id]

            # Build outputs list (single output for most nodes)
            outputs = [self.variable_slots.get(node.output, -1)] if node.output else []

            op = {
                "id": node_id,
                "op_type": node.op_type,
                "domain": node.domain,
                "inputs": [self.variable_slots.get(v, -1) for v in node.inputs],
                "outputs": outputs,
                "metadata": node.metadata.copy(),
            }

            operations.append(op)

        return operations

    def _extract_constants(self) -> Dict[str, Any]:
        """
        Extract constant values referenced by operations.

        Returns a mapping of variable name -> constant value.
        """
        constants = {}

        for node in self.ir_graph.nodes:
            if node.op_type == "constant" and node.output:
                var_name = node.output
                value = node.metadata.get("value")
                constants[var_name] = value

        return constants


def compile_to_batch(ir_graph: IRGraph) -> BatchExecutionPlan:
    """
    Convenience function to compile an IR graph to a batch execution plan.

    Usage:
        plan = compile_to_batch(builder.ir_graph)
        json_payload = plan.to_json()
        # Send json_payload to Rust via FFI
    """
    generator = BatchPlanGenerator(ir_graph)
    return generator.generate()


def estimate_performance(
    plan: BatchExecutionPlan, ffi_overhead_ms: float = 0.25
) -> Dict[str, Any]:
    """
    Estimate the performance improvement from batch execution.

    Args:
        plan: The batch execution plan
        ffi_overhead_ms: Overhead per FFI call in milliseconds

    Returns:
        Dictionary with performance estimates
    """
    num_operations = len(plan.operations)

    # Without batching: one FFI call per operation
    unbatched_ffi_time = num_operations * ffi_overhead_ms

    # With batching: one FFI call total
    batched_ffi_time = ffi_overhead_ms

    # Savings
    ffi_savings = unbatched_ffi_time - batched_ffi_time
    speedup = unbatched_ffi_time / batched_ffi_time if batched_ffi_time > 0 else 1.0

    return {
        "num_operations": num_operations,
        "unbatched_ffi_time_ms": unbatched_ffi_time,
        "batched_ffi_time_ms": batched_ffi_time,
        "ffi_savings_ms": ffi_savings,
        "theoretical_speedup": speedup,
        "max_live_variables": plan.max_live_variables,
    }


# ============================================================================
# New Batch Instruction Compiler (for Tier 1 batch executor)
# ============================================================================


class SlotAllocator:
    """
    Linear scan register allocator for loop bodies.

    Assigns slots (registers) to variables while minimizing slot count
    by reusing slots for variables with non-overlapping lifetimes.
    """

    def __init__(self):
        self.slots: Dict[str, int] = {}  # variable name -> slot id
        self.live_ranges: Dict[str, Tuple[int, int]] = (
            {}
        )  # var -> (first_use, last_use)
        self.next_slot_id = 0

    def compute_lifetimes(self, operations: List[Dict[str, Any]]) -> None:
        """
        Compute variable lifetimes from a sequence of operations.

        Args:
            operations: List of operation dicts with various input/output keys
        """
        for i, op in enumerate(operations):
            # Track output definition
            if "output" in op and op["output"]:
                var = op["output"]
                if var not in self.live_ranges:
                    self.live_ranges[var] = (i, i)
                else:
                    # Extend lifetime (redefinition)
                    first, _ = self.live_ranges[var]
                    self.live_ranges[var] = (min(first, i), i)

            # Track target as output (for alias steps)
            if "target" in op and op["target"] and op.get("type") != "attach_attr":
                var = op["target"]
                if var not in self.live_ranges:
                    self.live_ranges[var] = (i, i)
                else:
                    first, _ = self.live_ranges[var]
                    self.live_ranges[var] = (min(first, i), i)

            # Extract all input variables from the operation
            input_vars = []

            # Standard inputs key
            if "inputs" in op:
                inputs = op["inputs"]
                if isinstance(inputs, dict):
                    input_vars.extend(inputs.values())
                else:
                    input_vars.extend(inputs)

            # Operation keys (a, b, lhs, rhs, source, etc.)
            for key in [
                "a",
                "b",
                "lhs",
                "rhs",
                "left",
                "right",
                "source",
                "input",
                "condition",
                "if_true",
                "if_false",
                "scalar",
                "base",
                "exp",
            ]:
                if key in op and isinstance(op[key], str):
                    input_vars.append(op[key])

            # Track input uses
            for input_var in input_vars:
                if input_var not in self.live_ranges:
                    # First use (parameter or external)
                    self.live_ranges[input_var] = (0, i)
                else:
                    # Extend lifetime to this use
                    first, _ = self.live_ranges[input_var]
                    self.live_ranges[input_var] = (first, i)

    def allocate(self) -> int:
        """
        Perform register allocation using linear scan algorithm.

        Returns:
            Number of slots needed (slot_count)
        """
        if not self.live_ranges:
            return 0

        # Sort variables by start time
        vars_sorted = sorted(self.live_ranges.items(), key=lambda x: x[1][0])

        # Track free slots and active intervals
        free_slots: List[int] = []
        active: Dict[str, Tuple[int, int]] = {}  # var -> (slot, end_time)

        for var, (start, end) in vars_sorted:
            # Expire old intervals and free their slots
            expired = [v for v, (s, e) in active.items() if e < start]
            for expired_var in expired:
                slot, _ = active[expired_var]
                free_slots.append(slot)
                del active[expired_var]

            # Try to reuse a free slot
            if free_slots:
                slot = free_slots.pop(0)
                self.slots[var] = slot
                active[var] = (slot, end)
            else:
                # Allocate new slot
                slot = self.next_slot_id
                self.next_slot_id += 1
                self.slots[var] = slot
                active[var] = (slot, end)

        return self.next_slot_id

    def get_slot(self, var: str) -> int:
        """Get the slot allocated to a variable."""
        return self.slots.get(var, -1)


class IRToBatchCompiler:
    """
    Compiles IR operations to BatchInstruction format for the Rust batch executor.

    This compiler:
    1. Analyzes loop body operations
    2. Allocates slots to variables
    3. Lowers IR operations to BatchInstructions
    4. Detects loop-carried variables (phi nodes)
    """

    def __init__(self):
        self.allocator = SlotAllocator()
        self.instructions: List[Dict[str, Any]] = []
        self.carried_vars: List[Tuple[int, int]] = []  # (from_slot, to_slot)
        self.next_temp_slot: int = 0

    def compile_loop_body(
        self,
        body_steps: List[Dict[str, Any]],
        loop_vars: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Compile a loop body to BatchPlan format.

        Args:
            body_steps: List of step dicts (from LoopIRNode.body)
            loop_vars: Optional list of (initial_var, loop_var) pairs

        Returns:
            BatchPlan dict ready for JSON serialization
        """
        body_steps = self._rewrite_neighbor_mode_update(body_steps)

        # Build mapping from loop body variable names to their storage names
        # loop_vars contains (initial_var, loop_var) where loop_var is used in body
        # and should load/store using loop_var as the property name
        var_to_storage = {}
        if loop_vars:
            for initial_var, loop_var in loop_vars:
                # The loop body uses loop_var, which should load/store as itself
                var_to_storage[loop_var] = loop_var

        # Step 1: Compute lifetimes
        self.allocator.compute_lifetimes(body_steps)

        # Step 2: Allocate slots
        slot_count = self.allocator.allocate()
        self.next_temp_slot = slot_count

        # Step 2.5: Identify external inputs (variables used before they're defined)
        # These need LoadNodeProp instructions at the start
        defined_vars = set()
        external_vars = set()

        for step in body_steps:
            # Check uses BEFORE updating definitions
            for key in [
                "a",
                "b",
                "lhs",
                "rhs",
                "left",
                "right",
                "source",
                "input",
                "condition",
                "if_true",
                "if_false",
                "scalar",
                "base",
                "exp",
            ]:
                if key in step and isinstance(step[key], str):
                    var = step[key]
                    if var not in defined_vars:
                        external_vars.add(var)
            if "inputs" in step:
                inputs = step["inputs"]
                if isinstance(inputs, dict):
                    values = inputs.values()
                else:
                    values = inputs
                for var in values:
                    if var not in defined_vars:
                        external_vars.add(var)

            # Now update definitions
            if step.get("output"):
                defined_vars.add(step.get("output"))
            if step.get("target") and step.get("type") != "attach_attr":
                defined_vars.add(step.get("target"))

        # Emit LoadNodeProp for external variables
        for var in sorted(external_vars):  # Sort for deterministic order
            slot = self.allocator.get_slot(var)
            if slot >= 0:
                # Use the storage name if this is a loop-carried variable
                storage_name = var_to_storage.get(var, var)
                self.instructions.append(
                    {
                        "type": "load_node_prop",
                        "dst": slot,
                        "var_name": storage_name,
                    }
                )

        # Step 3: Lower operations to BatchInstructions
        # Also track alias steps for loop-carried variables
        alias_mappings = {}  # target -> source
        for step in body_steps:
            if step.get("type") == "alias":
                source = step.get("source")
                target = step.get("target")
                if source and target:
                    alias_mappings[target] = source
            else:
                instr = self._lower_step(step)
                if instr:
                    if isinstance(instr, list):
                        self.instructions.extend(instr)
                    else:
                        self.instructions.append(instr)

        # Step 3.5: Emit StoreNodeProp for loop-carried variables (alias targets)
        for target, source in alias_mappings.items():
            src_slot = self.allocator.get_slot(source)
            if src_slot >= 0:
                # Use the storage name if this is a loop-carried variable
                storage_name = var_to_storage.get(target, target)
                self.instructions.append(
                    {
                        "type": "store_node_prop",
                        "src": src_slot,
                        "var_name": storage_name,
                    }
                )

        # Step 4: Handle loop-carried variables
        if loop_vars:
            for initial_var, loop_var in loop_vars:
                from_slot = self.allocator.get_slot(loop_var)
                to_slot = self.allocator.get_slot(initial_var)
                if from_slot >= 0 and to_slot >= 0:
                    self.carried_vars.append((from_slot, to_slot))

        slot_count = max(slot_count, self.next_temp_slot)
        return {
            "instructions": self.instructions,
            "slot_count": slot_count,
            "carried_slots": self.carried_vars,
            "name": "loop_body",
        }

    def _alloc_temp(self) -> int:
        slot = self.next_temp_slot
        self.next_temp_slot += 1
        return slot

    def _rewrite_neighbor_mode_update(
        self, body_steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collapse collect_neighbor_values + mode_list + update_in_place into neighbor_mode_update."""
        rewritten: List[Dict[str, Any]] = []
        i = 0
        while i < len(body_steps):
            step = body_steps[i]
            if (
                step.get("type") == "core.collect_neighbor_values"
                and i + 2 < len(body_steps)
            ):
                mode_step = body_steps[i + 1]
                update_step = body_steps[i + 2]
                if (
                    mode_step.get("type") == "core.mode_list"
                    and update_step.get("type") == "core.update_in_place"
                ):
                    collect_output = step.get("output")
                    mode_source = mode_step.get("source")
                    mode_output = mode_step.get("output")
                    update_source = update_step.get("source")
                    target = update_step.get("target")

                    if (
                        collect_output
                        and mode_source == collect_output
                        and update_source == mode_output
                        and target
                    ):
                        rewritten.append(
                            {
                                "type": "core.neighbor_mode_update",
                                "target": target,
                                "include_self": bool(step.get("include_self", True)),
                                "tie_break": mode_step.get("tie_break", "lowest"),
                                "ordered": bool(update_step.get("ordered", True)),
                                "output": update_step.get("output", target),
                            }
                        )
                        i += 3
                        continue
            rewritten.append(step)
            i += 1
        return rewritten

    def _lower_step(self, step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Lower a single step to a BatchInstruction.

        Returns None if the step cannot be lowered (unsupported operation).
        """
        step_type = step.get("type", "")

        # Handle core arithmetic operations
        if step_type == "core.add":
            return self._lower_binary_op(step, "add")
        elif step_type == "core.sub":
            return self._lower_binary_op(step, "sub")
        elif step_type == "core.mul":
            return self._lower_binary_op(step, "mul")
        elif step_type == "core.div":
            return self._lower_binary_op(step, "div")
        elif step_type == "core.min":
            return self._lower_binary_op(step, "min")
        elif step_type == "core.max":
            return self._lower_binary_op(step, "max")
        elif step_type == "core.pow":
            return self._lower_pow(step)
        elif step_type == "core.abs":
            return self._lower_unary_op(step, "abs")
        elif step_type == "core.clip":
            return self._lower_clip(step)
        elif step_type == "core.recip":
            return self._lower_recip(step)
        elif step_type == "core.sqrt":
            return self._lower_unary_op(step, "sqrt")
        elif step_type == "core.exp":
            return self._lower_unary_op(step, "exp")
        elif step_type == "core.log":
            return self._lower_unary_op(step, "log")
        elif step_type == "core.compare":
            return self._lower_compare(step)
        elif step_type == "core.where":
            return self._lower_where(step)
        elif step_type == "core.reduce_scalar":
            return self._lower_reduce_scalar(step)
        elif step_type == "core.broadcast_scalar":
            return self._lower_broadcast_scalar(step)

        # Handle scalar operations
        elif step_type == "core.constant":
            return self._lower_constant(step)
        elif step_type == "init_scalar":
            return self._lower_constant(step)

        # Handle graph operations
        elif step_type == "graph.neighbor_sum":
            return self._lower_neighbor_aggregate(step, "sum")
        elif step_type == "graph.neighbor_mean":
            return self._lower_neighbor_aggregate(step, "mean")
        elif step_type == "graph.neighbor_min":
            return self._lower_neighbor_aggregate(step, "min")
        elif step_type == "graph.neighbor_max":
            return self._lower_neighbor_aggregate(step, "max")
        elif step_type == "graph.neighbor_agg":
            # Generic neighbor aggregation
            agg_type = step.get("agg", "sum")
            return self._lower_neighbor_aggregate(step, agg_type)

        # Handle alias (copy operation)
        elif step_type == "alias":
            return self._lower_alias(step)

        # Handle loads/stores
        elif step_type == "init_nodes_with_index":
            return self._lower_load(step)
        elif step_type == "attach_attr":
            return self._lower_store(step)
        elif step_type == "init_nodes":
            return self._lower_init_nodes(step)
        elif step_type == "load_attr":
            return self._lower_load_attr(step)
        elif step_type == "load_edge_attr":
            return self._lower_load_edge_attr(step)
        elif step_type == "node_degree":
            return self._lower_node_degree(step)
        elif step_type == "graph_node_count":
            return self._lower_graph_node_count(step)
        elif step_type == "graph_edge_count":
            return self._lower_graph_edge_count(step)
        elif step_type == "normalize":
            return self._lower_normalize(step, method=step.get("method", "sum"))
        elif step_type == "normalize_sum":
            return self._lower_normalize(step, method="sum")
        elif step_type == "core.update_in_place":
            return self._lower_update_in_place(step)
        elif step_type == "core.neighbor_mode_update":
            return self._lower_neighbor_mode_update(step)
        elif step_type == "map_nodes":
            return self._lower_map_nodes(step)
        elif step_type == "core.collect_neighbor_values":
            return self._lower_collect_neighbor_values(step)
        elif step_type == "core.mode_list":
            return self._lower_mode_list(step)

        # Unsupported: return None (will trigger fallback)
        return None

    def _lower_binary_op(self, step: Dict[str, Any], op_name: str) -> Dict[str, Any]:
        """Lower binary arithmetic operation."""
        # Try different key names for inputs (steps use 'a'/'b', not 'lhs'/'rhs')
        lhs = (
            step.get("lhs")
            or step.get("a")
            or step.get("source")
            or step.get("inputs", [None])[0]
        )
        rhs = step.get("rhs") or step.get("b") or step.get("inputs", [None, None])[1]
        dst = step.get("output") or step.get("target")

        return {
            "type": op_name,
            "dst": self.allocator.get_slot(dst),
            "lhs": self.allocator.get_slot(lhs),
            "rhs": self.allocator.get_slot(rhs),
        }

    def _lower_constant(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Lower constant load to LoadScalar."""
        dst = step.get("output") or step.get("target")
        value = step.get("value", 0.0)

        return {
            "type": "load_scalar",
            "dst": self.allocator.get_slot(dst),
            "value": float(value),
        }

    def _lower_unary_op(self, step: Dict[str, Any], op_name: str) -> Dict[str, Any]:
        src = step.get("source") or step.get("input") or step.get("inputs", [None])[0]
        dst = step.get("output") or step.get("target")
        return {
            "type": op_name,
            "dst": self.allocator.get_slot(dst),
            "src": self.allocator.get_slot(src),
        }

    def _lower_pow(self, step: Dict[str, Any]) -> Dict[str, Any]:
        base = step.get("base") or step.get("left") or step.get("source")
        exp = step.get("exp") or step.get("right")
        dst = step.get("output") or step.get("target")
        return {
            "type": "pow",
            "dst": self.allocator.get_slot(dst),
            "base": self.allocator.get_slot(base),
            "exp": self.allocator.get_slot(exp),
        }

    def _lower_clip(self, step: Dict[str, Any]) -> Dict[str, Any]:
        src = step.get("source") or step.get("input") or step.get("inputs", [None])[0]
        dst = step.get("output") or step.get("target")
        return {
            "type": "clip",
            "dst": self.allocator.get_slot(dst),
            "src": self.allocator.get_slot(src),
            "min": float(step.get("min_value", step.get("min", 0.0))),
            "max": float(step.get("max_value", step.get("max", 0.0))),
        }

    def _lower_recip(self, step: Dict[str, Any]) -> Dict[str, Any]:
        src = step.get("source") or step.get("input") or step.get("inputs", [None])[0]
        dst = step.get("output") or step.get("target")
        return {
            "type": "recip",
            "dst": self.allocator.get_slot(dst),
            "src": self.allocator.get_slot(src),
            "epsilon": float(step.get("epsilon", 1e-10)),
        }

    def _lower_compare(self, step: Dict[str, Any]) -> Dict[str, Any]:
        lhs = step.get("left") or step.get("lhs") or step.get("a")
        rhs = step.get("right") or step.get("rhs") or step.get("b")
        dst = step.get("output") or step.get("target")
        return {
            "type": "compare",
            "dst": self.allocator.get_slot(dst),
            "lhs": self.allocator.get_slot(lhs),
            "rhs": self.allocator.get_slot(rhs),
            "op": step.get("op", "eq"),
        }

    def _lower_where(self, step: Dict[str, Any]) -> Dict[str, Any]:
        condition = step.get("condition")
        if_true = step.get("if_true")
        if_false = step.get("if_false")
        dst = step.get("output") or step.get("target")
        return {
            "type": "where",
            "dst": self.allocator.get_slot(dst),
            "condition": self.allocator.get_slot(condition),
            "if_true": self.allocator.get_slot(if_true),
            "if_false": self.allocator.get_slot(if_false),
        }

    def _lower_reduce_scalar(self, step: Dict[str, Any]) -> Dict[str, Any]:
        src = step.get("source") or step.get("input") or step.get("inputs", [None])[0]
        dst = step.get("output") or step.get("target")
        return {
            "type": "reduce_scalar",
            "dst": self.allocator.get_slot(dst),
            "src": self.allocator.get_slot(src),
            "operation": step.get("op", "sum"),
        }

    def _lower_broadcast_scalar(self, step: Dict[str, Any]) -> Dict[str, Any]:
        scalar = step.get("scalar")
        dst = step.get("output") or step.get("target")
        return {
            "type": "broadcast_scalar",
            "dst": self.allocator.get_slot(dst),
            "scalar": self.allocator.get_slot(scalar),
        }

    def _lower_neighbor_aggregate(
        self, step: Dict[str, Any], operation: str
    ) -> Dict[str, Any]:
        """Lower neighbor aggregation to NeighborAggregate."""
        src = step.get("source") or step.get("inputs", [None])[0]
        dst = step.get("output") or step.get("target")
        direction = step.get("direction", "in")

        return {
            "type": "neighbor_aggregate",
            "dst": self.allocator.get_slot(dst),
            "src": self.allocator.get_slot(src),
            "operation": operation,
            "direction": direction,
        }

    def _lower_load(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Lower node property load."""
        dst = step.get("output") or step.get("target")
        var_name = step.get("var_name", dst)

        return {
            "type": "load_node_prop",
            "dst": self.allocator.get_slot(dst),
            "var_name": var_name,
        }

    def _lower_init_nodes(self, step: Dict[str, Any]) -> Dict[str, Any]:
        dst = step.get("output") or step.get("target")
        value = step.get("default", 0.0)
        return {
            "type": "init_nodes",
            "dst": self.allocator.get_slot(dst),
            "value": float(value),
        }

    def _lower_load_attr(self, step: Dict[str, Any]) -> Dict[str, Any]:
        dst = step.get("output") or step.get("target")
        attr_name = step.get("attr_name") or step.get("attr")
        default = step.get("default", 0.0)
        return {
            "type": "load_node_attr",
            "dst": self.allocator.get_slot(dst),
            "attr_name": attr_name,
            "default": float(default),
        }

    def _lower_load_edge_attr(self, step: Dict[str, Any]) -> Dict[str, Any]:
        dst = step.get("output") or step.get("target")
        attr_name = step.get("attr_name") or step.get("attr")
        default = step.get("default", 0.0)
        return {
            "type": "load_edge_attr",
            "dst": self.allocator.get_slot(dst),
            "attr_name": attr_name,
            "default": float(default),
        }

    def _lower_node_degree(self, step: Dict[str, Any]) -> Dict[str, Any]:
        dst = step.get("output") or step.get("target")
        return {
            "type": "node_degree",
            "dst": self.allocator.get_slot(dst),
        }

    def _lower_graph_node_count(self, step: Dict[str, Any]) -> Dict[str, Any]:
        dst = step.get("output") or step.get("target")
        return {
            "type": "graph_node_count",
            "dst": self.allocator.get_slot(dst),
        }

    def _lower_graph_edge_count(self, step: Dict[str, Any]) -> Dict[str, Any]:
        dst = step.get("output") or step.get("target")
        return {
            "type": "graph_edge_count",
            "dst": self.allocator.get_slot(dst),
        }

    def _lower_normalize(self, step: Dict[str, Any], method: str) -> Dict[str, Any]:
        src = step.get("input") or step.get("source")
        dst = step.get("output") or step.get("target")
        return {
            "type": "normalize",
            "dst": self.allocator.get_slot(dst),
            "src": self.allocator.get_slot(src),
            "method": method,
            "epsilon": float(step.get("epsilon", 1e-9)),
        }

    def _lower_update_in_place(self, step: Dict[str, Any]) -> Dict[str, Any]:
        source = step.get("source")
        target = step.get("target")
        return {
            "type": "update_in_place",
            "source": self.allocator.get_slot(source),
            "target": self.allocator.get_slot(target),
            "ordered": bool(step.get("ordered", False)),
        }

    def _lower_neighbor_mode_update(self, step: Dict[str, Any]) -> Dict[str, Any]:
        target = step.get("target")
        return {
            "type": "neighbor_mode_update",
            "target": self.allocator.get_slot(target),
            "include_self": bool(step.get("include_self", True)),
            "tie_break": step.get("tie_break", "lowest"),
            "ordered": bool(step.get("ordered", True)),
        }

    def _lower_map_nodes(self, step: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        if step.get("async_update", False):
            return None

        from groggy.expr_parser import parse_expression

        expr_str = step.get("fn", "")
        inputs = step.get("inputs", {})
        if not inputs:
            return None

        resolved_inputs = {key: val for key, val in inputs.items()}
        expr = parse_expression(expr_str)

        def rewrite_vars(expr_node):
            if isinstance(expr_node, dict):
                if expr_node.get("type") == "var":
                    var_name = expr_node.get("name")
                    if var_name == "value":
                        first_input = next(iter(resolved_inputs.values()))
                        expr_node["name"] = first_input
                    elif var_name in resolved_inputs:
                        expr_node["name"] = resolved_inputs[var_name]
                if "args" in expr_node:
                    for arg in expr_node["args"]:
                        rewrite_vars(arg)
                if "left" in expr_node:
                    rewrite_vars(expr_node["left"])
                if "right" in expr_node:
                    rewrite_vars(expr_node["right"])
            return expr_node

        expr = rewrite_vars(expr)

        instructions: List[Dict[str, Any]] = []

        def lower_expr(node) -> Optional[int]:
            node_type = node.get("type")
            if node_type == "const":
                slot = self._alloc_temp()
                instructions.append(
                    {"type": "load_scalar", "dst": slot, "value": float(node["value"])}
                )
                return slot
            if node_type == "var":
                return self.allocator.get_slot(node["name"])
            if node_type == "binary_op":
                op = node.get("op")
                left = lower_expr(node.get("left"))
                right = lower_expr(node.get("right"))
                if left is None or right is None:
                    return None
                slot = self._alloc_temp()
                instructions.append(
                    {"type": op, "dst": slot, "lhs": left, "rhs": right}
                )
                return slot
            if node_type == "call":
                func = node.get("func")
                args = node.get("args", [])
                if func in {"sum", "mean", "mode", "count"} and args:
                    inner = args[0]
                    if inner.get("type") == "call" and inner.get("func") in {
                        "neighbor_values",
                        "neighbors",
                    }:
                        if func == "count":
                            slot = self._alloc_temp()
                            instructions.append({"type": "node_degree", "dst": slot})
                            return slot
                        if inner.get("func") == "neighbor_values":
                            var_node = inner.get("args", [None])[0]
                            if not var_node or var_node.get("type") != "var":
                                return None
                            src = self.allocator.get_slot(var_node["name"])
                            slot = self._alloc_temp()
                            if func == "mode":
                                instructions.append(
                                    {
                                        "type": "neighbor_mode",
                                        "dst": slot,
                                        "src": src,
                                        "tie_break": "lowest",
                                        "direction": "in",
                                    }
                                )
                            else:
                                instructions.append(
                                    {
                                        "type": "neighbor_aggregate",
                                        "dst": slot,
                                        "src": src,
                                        "operation": func,
                                        "direction": "in",
                                    }
                                )
                            return slot
                    return None
            return None

        result_slot = lower_expr(expr)
        if result_slot is None:
            return None

        target = step.get("output") or step.get("target")
        if target is None:
            return None

        instructions.append(
            {
                "type": "store_node_prop",
                "src": result_slot,
                "var_name": target,
            }
        )
        return instructions

    def _lower_collect_neighbor_values(self, step: Dict[str, Any]) -> Dict[str, Any]:
        src = step.get("source")
        dst = step.get("output") or step.get("target")
        return {
            "type": "collect_neighbor_values",
            "dst": self.allocator.get_slot(dst),
            "src": self.allocator.get_slot(src),
            "include_self": bool(step.get("include_self", True)),
        }

    def _lower_mode_list(self, step: Dict[str, Any]) -> Dict[str, Any]:
        src = step.get("source")
        dst = step.get("output") or step.get("target")
        return {
            "type": "mode_list",
            "dst": self.allocator.get_slot(dst),
            "src": self.allocator.get_slot(src),
            "tie_break": step.get("tie_break", "lowest"),
        }

    def _lower_store(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Lower node property store."""
        src = step.get("source") or step.get("inputs", [None])[0]
        var_name = step.get("name", src)

        return {
            "type": "store_node_prop",
            "src": self.allocator.get_slot(src),
            "var_name": var_name,
        }

    def _lower_alias(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Lower alias (copy) operation. This is a no-op at the instruction level,
        as the register allocator already handles variable renaming."""
        # Alias means target = source. In the register model, this is already
        # handled by slot allocation, but we need to track the copy for loop-carried vars.
        # For now, emit as a comment/metadata or skip it.
        # The slot allocator already knows target and source share a slot or are connected.
        return None  # Skip alias in instruction stream


def _collect_batch_diagnostics(
    body_steps: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    edge_attr_outputs = set()
    for step in body_steps:
        if step.get("type") == "load_edge_attr":
            output = step.get("output") or step.get("target")
            if output:
                edge_attr_outputs.add(output)

    def uses_edge_attr(step: Dict[str, Any]) -> bool:
        for key in [
            "a",
            "b",
            "lhs",
            "rhs",
            "left",
            "right",
            "source",
            "input",
            "condition",
            "if_true",
            "if_false",
            "scalar",
            "base",
            "exp",
        ]:
            if step.get(key) in edge_attr_outputs:
                return True
        inputs = step.get("inputs")
        if isinstance(inputs, dict) and any(v in edge_attr_outputs for v in inputs.values()):
            return True
        if isinstance(inputs, list) and any(v in edge_attr_outputs for v in inputs):
            return True
        return False
    for idx, step in enumerate(body_steps):
        step_type = step.get("type", "")
        if step_type.startswith("iter.") or step_type.startswith("control."):
            diagnostics.append(
                {
                    "step_index": idx,
                    "step_type": step_type,
                    "reason": "nested_control_flow",
                }
            )
            continue
        if step_type == "core.execution_block":
            diagnostics.append(
                {
                    "step_index": idx,
                    "step_type": step_type,
                    "reason": "execution_block",
                }
            )
            continue
        if step_type == "map_nodes" and step.get("async_update", False):
            diagnostics.append(
                {
                    "step_index": idx,
                    "step_type": step_type,
                    "reason": "async_update_not_batchable",
                }
            )
            continue
        if edge_attr_outputs and uses_edge_attr(step):
            diagnostics.append(
                {
                    "step_index": idx,
                    "step_type": step_type,
                    "reason": "edge_attr_usage_not_batchable",
                }
            )
            continue
        if step_type not in SUPPORTED_BATCH_OPS:
            diagnostics.append(
                {
                    "step_index": idx,
                    "step_type": step_type,
                    "reason": "unsupported_step_type",
                }
            )
    return diagnostics


def _instruction_input_slots(instr: Dict[str, Any]) -> List[int]:
    inputs: List[int] = []
    instr_type = instr.get("type")
    if instr_type in {"add", "sub", "mul", "div"}:
        inputs.extend([instr.get("lhs", -1), instr.get("rhs", -1)])
    elif instr_type in {"min", "max"}:
        inputs.extend([instr.get("lhs", -1), instr.get("rhs", -1)])
    elif instr_type in {"pow"}:
        inputs.extend([instr.get("base", -1), instr.get("exp", -1)])
    elif instr_type in {"compare"}:
        inputs.extend([instr.get("lhs", -1), instr.get("rhs", -1)])
    elif instr_type in {"where"}:
        inputs.extend(
            [
                instr.get("condition", -1),
                instr.get("if_true", -1),
                instr.get("if_false", -1),
            ]
        )
    elif instr_type in {"abs", "clip", "recip", "sqrt", "exp", "log"}:
        inputs.append(instr.get("src", -1))
    elif instr_type in {"reduce_scalar"}:
        inputs.append(instr.get("src", -1))
    elif instr_type in {"broadcast_scalar"}:
        inputs.append(instr.get("scalar", -1))
    elif instr_type in {"normalize"}:
        inputs.append(instr.get("src", -1))
    elif instr_type in {"collect_neighbor_values", "mode_list"}:
        inputs.append(instr.get("src", -1))
    elif instr_type == "neighbor_aggregate":
        inputs.append(instr.get("src", -1))
    elif instr_type == "neighbor_mode":
        inputs.append(instr.get("src", -1))
    elif instr_type == "fused_neighbor_mul_agg":
        inputs.extend([instr.get("src", -1), instr.get("multiplier", -1)])
    elif instr_type == "fused_madd":
        inputs.extend([instr.get("a", -1), instr.get("b", -1), instr.get("c", -1)])
    elif instr_type == "fused_axpy":
        inputs.extend([instr.get("alpha", -1), instr.get("x", -1), instr.get("y", -1)])
    elif instr_type == "store_node_prop":
        inputs.append(instr.get("src", -1))
    return [slot for slot in inputs if isinstance(slot, int)]


def _instruction_output_slots(instr: Dict[str, Any]) -> List[int]:
    instr_type = instr.get("type")
    if instr_type in {"load_node_prop", "load_scalar"}:
        return [instr.get("dst", -1)]
    if instr_type in {"init_nodes", "load_node_attr", "node_degree", "graph_node_count", "graph_edge_count"}:
        return [instr.get("dst", -1)]
    if instr_type in {"load_edge_attr"}:
        return [instr.get("dst", -1)]
    if instr_type in {"add", "sub", "mul", "div"}:
        return [instr.get("dst", -1)]
    if instr_type in {"min", "max"}:
        return [instr.get("dst", -1)]
    if instr_type in {"abs", "clip", "recip", "sqrt", "exp", "log", "pow"}:
        return [instr.get("dst", -1)]
    if instr_type in {"compare", "where"}:
        return [instr.get("dst", -1)]
    if instr_type in {"reduce_scalar", "broadcast_scalar"}:
        return [instr.get("dst", -1)]
    if instr_type in {"normalize"}:
        return [instr.get("dst", -1)]
    if instr_type in {"update_in_place", "neighbor_mode_update"}:
        return []
    if instr_type in {"collect_neighbor_values", "mode_list"}:
        return [instr.get("dst", -1)]
    if instr_type in {"neighbor_aggregate", "neighbor_mode"}:
        return [instr.get("dst", -1)]
    if instr_type in {"fused_neighbor_mul_agg", "fused_madd", "fused_axpy"}:
        return [instr.get("dst", -1)]
    return []


def _validate_instruction_slots(
    instructions: Iterable[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    invalid: List[Dict[str, Any]] = []
    for idx, instr in enumerate(instructions):
        for slot in _instruction_input_slots(instr) + _instruction_output_slots(instr):
            if isinstance(slot, int) and slot < 0:
                invalid.append(
                    {
                        "instruction_index": idx,
                        "instruction_type": instr.get("type"),
                        "reason": "unresolved_slot",
                    }
                )
                break
    return invalid or None


def _collect_scalar_slots(instructions: Iterable[Dict[str, Any]]) -> Set[int]:
    scalars: Set[int] = set()
    for instr in instructions:
        if instr.get("type") == "load_scalar":
            slot = instr.get("dst")
            if isinstance(slot, int):
                scalars.add(slot)
    return scalars


def _optimize_instructions(instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    use_counts: Dict[int, int] = {}
    for instr in instructions:
        for slot in _instruction_input_slots(instr):
            use_counts[slot] = use_counts.get(slot, 0) + 1

    scalar_slots = _collect_scalar_slots(instructions)
    optimized: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(instructions):
        instr = instructions[idx]
        if instr.get("type") == "mul" and idx + 1 < len(instructions):
            next_instr = instructions[idx + 1]
            if next_instr.get("type") == "add":
                mul_dst = instr.get("dst")
                add_lhs = next_instr.get("lhs")
                add_rhs = next_instr.get("rhs")
                if mul_dst in {add_lhs, add_rhs} and use_counts.get(mul_dst, 0) == 1:
                    other = add_rhs if add_lhs == mul_dst else add_lhs
                    mul_lhs = instr.get("lhs")
                    mul_rhs = instr.get("rhs")
                    if mul_lhs in scalar_slots or mul_rhs in scalar_slots:
                        alpha = mul_lhs if mul_lhs in scalar_slots else mul_rhs
                        x = mul_rhs if alpha == mul_lhs else mul_lhs
                        optimized.append(
                            {
                                "type": "fused_axpy",
                                "dst": next_instr.get("dst"),
                                "alpha": alpha,
                                "x": x,
                                "y": other,
                            }
                        )
                    else:
                        optimized.append(
                            {
                                "type": "fused_madd",
                                "dst": next_instr.get("dst"),
                                "a": mul_lhs,
                                "b": mul_rhs,
                                "c": other,
                            }
                        )
                    idx += 2
                    continue
        optimized.append(instr)
        idx += 1
    return optimized


def compile_loop_to_batch_plan(
    body_steps: List[Dict[str, Any]], loop_vars: Optional[List[Tuple[str, str]]] = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to compile a loop body to a batch plan.

    Args:
        body_steps: List of step dicts from loop body
        loop_vars: Optional loop-carried variables

    Returns:
        BatchPlan dict ready for JSON serialization to Rust, or None if loop
        contains unsupported operations

    Example:
        >>> body = [{"type": "core.mul", "lhs": "a", "rhs": "b", "output": "c"}]
        >>> plan = compile_loop_to_batch_plan(body)
        >>> if plan:
        ...     import json
        ...     json.dumps(plan)  # Send to Rust BatchExecutor
    """
    if os.environ.get("GROGGY_DISABLE_BATCH"):
        if os.environ.get("GROGGY_DEBUG_BATCH"):
            print(
                json.dumps(
                    {"event": "batch_compile_disabled", "reason": "env_override"},
                    indent=2,
                )
            )
        return None
    diagnostics = _collect_batch_diagnostics(body_steps)
    if diagnostics:
        if os.environ.get("GROGGY_DEBUG_BATCH"):
            print(
                json.dumps(
                    {"event": "batch_compile_skipped", "diagnostics": diagnostics},
                    indent=2,
                )
            )
        return None

    # All operations supported - proceed with compilation
    compiler = IRToBatchCompiler()
    plan = compiler.compile_loop_body(body_steps, loop_vars)
    instructions = plan.get("instructions", [])
    plan["instructions"] = _optimize_instructions(instructions)

    invalid_slots = _validate_instruction_slots(plan["instructions"])
    if invalid_slots:
        if os.environ.get("GROGGY_DEBUG_BATCH"):
            print(
                json.dumps(
                    {"event": "batch_compile_invalid_slots", "details": invalid_slots},
                    indent=2,
                )
            )
        return None
    return plan
