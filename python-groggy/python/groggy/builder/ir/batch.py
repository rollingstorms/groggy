"""
Batch Execution Plan Generator

Compiles IR graphs into compact batch execution plans that can be sent
to Rust in a single FFI call, eliminating per-operation FFI overhead.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import json
import struct

from .nodes import IRNode, CoreIRNode, GraphIRNode, ControlIRNode
from .graph import IRGraph


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
        
        return json.dumps({
            "operations": serializable_ops,
            "variable_slots": self.variable_slots,
            "constant_values": self.constant_values,
            "max_live_variables": self.max_live_variables,
            "execution_order": self.execution_order,
        })
    
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
        return self.to_json().encode('utf-8')
    
    @classmethod
    def from_binary(cls, data: bytes) -> "BatchExecutionPlan":
        """Deserialize from binary format"""
        return cls.from_json(data.decode('utf-8'))


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
        self.live_ranges: Dict[str, Tuple[int, int]] = {}  # var -> (first_use, last_use)
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
        
        # Active variables (currently live)
        active: List[Tuple[str, int]] = []  # (var_name, end_time)
        
        for var_name, (start, end) in vars_by_start:
            # Remove expired variables from active list
            active = [(v, e) for v, e in active if e >= start]
            
            # Try to reuse a slot from an expired variable
            if active:
                # Find the slot with the earliest end time
                min_end = min(e for _, e in active)
                for v, e in active:
                    if e == min_end and e < start:
                        # Reuse this slot
                        self.variable_slots[var_name] = self.variable_slots[v]
                        active.append((var_name, end))
                        break
                else:
                    # No reusable slot, allocate new one
                    self.variable_slots[var_name] = self.next_slot
                    self.next_slot += 1
                    active.append((var_name, end))
            else:
                # First variable, allocate slot 0
                self.variable_slots[var_name] = self.next_slot
                self.next_slot += 1
                active.append((var_name, end))
    
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


def estimate_performance(plan: BatchExecutionPlan, ffi_overhead_ms: float = 0.25) -> Dict[str, Any]:
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
