"""
Batch Execution Plan Generator

Compiles IR graphs into compact batch execution plans that can be sent
to Rust in a single FFI call, eliminating per-operation FFI overhead.
"""

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .graph import IRGraph
from .nodes import ControlIRNode, CoreIRNode, GraphIRNode, IRNode


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
                input_vars.extend(op["inputs"])

            # Binary operation keys (a, b, lhs, rhs, source, etc.)
            for key in ["a", "b", "lhs", "rhs", "source"]:
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

        # Step 2.5: Identify external inputs (variables used before they're defined)
        # These need LoadNodeProp instructions at the start
        defined_vars = set()
        external_vars = set()

        for step in body_steps:
            # Check uses BEFORE updating definitions
            for key in ["a", "b", "lhs", "rhs", "source"]:
                if key in step and isinstance(step[key], str):
                    var = step[key]
                    if var not in defined_vars:
                        external_vars.add(var)
            if "inputs" in step:
                for var in step["inputs"]:
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

        return {
            "instructions": self.instructions,
            "slot_count": slot_count,
            "carried_slots": self.carried_vars,
            "name": "loop_body",
        }

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

        # Handle scalar operations
        elif step_type == "core.constant":
            return self._lower_constant(step)

        # Handle graph operations
        elif step_type == "graph.neighbor_sum":
            return self._lower_neighbor_aggregate(step, "sum")
        elif step_type == "graph.neighbor_mean":
            return self._lower_neighbor_aggregate(step, "mean")
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
    # Check if all operations are supported
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
        "graph.neighbor_agg",  # Generic neighbor aggregation
        # Loads/stores
        "init_nodes_with_index",
        "attach_attr",
        "alias",  # No-op in batch execution
    }

    for step in body_steps:
        step_type = step.get("type", "")

        # Check for unsupported control flow
        if step_type.startswith("iter.") or step_type.startswith("control."):
            return None  # Nested loops not supported

        if step_type == "core.execution_block":
            return None  # Execution blocks not supported yet

        # Check if operation is in supported set
        if step_type not in supported_ops:
            # Unsupported operation - cannot batch compile
            return None

    # All operations supported - proceed with compilation
    compiler = IRToBatchCompiler()
    return compiler.compile_loop_body(body_steps, loop_vars)
