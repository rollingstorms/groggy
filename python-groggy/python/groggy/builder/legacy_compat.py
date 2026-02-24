"""
Legacy compatibility helpers for the builder package.

This module contains the remaining runtime components carried forward from the
monolithic builder implementation while the trait-based package is the public API.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from groggy import _groggy
from groggy.algorithms.base import AlgorithmHandle


class LoopContext:
    """Context manager for loop body."""

    def __init__(self, builder: "Any", iterations: int):
        self.builder = builder
        self.iterations = iterations
        self.start_step = None
        self.loop_vars = {}

    def __enter__(self):
        self.start_step = len(self.builder.steps)
        self.loop_vars = dict(self.builder.variables)
        return self

    def __exit__(self, *args):
        self.builder._finalize_loop(self.start_step, self.iterations, self.loop_vars)


def finalize_legacy_loop_steps(builder: Any, start_step: int, iterations: int, loop_vars: Dict[str, Any]) -> None:
    """Replace a recorded loop body with a structured ``iter.loop`` step."""
    loop_body = builder.steps[start_step:]
    builder.steps = builder.steps[:start_step]

    body_copy = json.loads(json.dumps(loop_body))

    def collect_strings(value, sink):
        if isinstance(value, str):
            sink.add(value)
        elif isinstance(value, list):
            for item in value:
                collect_strings(item, sink)
        elif isinstance(value, dict):
            for sub_val in value.values():
                collect_strings(sub_val, sink)

    vars_defined = set()
    vars_used = set()
    for step in body_copy:
        if not isinstance(step, dict):
            continue
        output = step.get("output")
        if isinstance(output, str):
            vars_defined.add(output)
        step_type = step.get("type")
        if step_type == "alias":
            target = step.get("target")
            if isinstance(target, str):
                vars_defined.add(target)
        for key, value in step.items():
            if key == "type" or key == "output":
                continue
            if step_type == "alias" and key == "target":
                continue
            collect_strings(value, vars_used)

    initial_candidates: list[str] = []
    if loop_vars:
        for _, handle in loop_vars.items():
            if handle and handle.name not in initial_candidates:
                initial_candidates.append(handle.name)

    pre_loop_alias_sources = {}
    for step in builder.steps[:start_step]:
        if isinstance(step, dict) and step.get("type") == "alias":
            source = step.get("source")
            target = step.get("target")
            if isinstance(source, str) and isinstance(target, str):
                pre_loop_alias_sources[target] = source

    alias_mappings: list[list[str]] = []
    paired_initials = set()
    logical_seen = set()

    def _is_loop_candidate(name: str) -> bool:
        return name.startswith(("nodes_", "labels_", "edges_", "values_"))

    for step in body_copy:
        if not isinstance(step, dict) or step.get("type") != "alias":
            continue
        logical_name = step.get("target")
        if not isinstance(logical_name, str) or logical_name in logical_seen:
            continue
        initial_name = pre_loop_alias_sources.get(logical_name)

        if initial_name is None:
            for candidate in initial_candidates:
                if not _is_loop_candidate(candidate):
                    continue
                if candidate in paired_initials or candidate == logical_name:
                    continue
                if candidate not in vars_used or candidate in vars_defined:
                    continue
                initial_name = candidate
                break
        if initial_name:
            alias_mappings.append([initial_name, logical_name])
            paired_initials.add(initial_name)
            logical_seen.add(logical_name)

    if alias_mappings:
        replacement_map = {initial: logical for initial, logical in alias_mappings}

        def remap_values(value):
            if isinstance(value, str):
                return replacement_map.get(value, value)
            if isinstance(value, list):
                return [remap_values(item) for item in value]
            if isinstance(value, dict):
                return {k: remap_values(v) for k, v in value.items()}
            return value

        for step in body_copy:
            if not isinstance(step, dict):
                continue
            step_type = step.get("type")
            for key, value in list(step.items()):
                if key == "type" or key == "output":
                    continue
                if step_type == "alias" and key == "target":
                    continue
                step[key] = remap_values(value)

    loop_step = {
        "type": "iter.loop",
        "iterations": iterations,
        "body": body_copy,
    }
    if alias_mappings:
        loop_step["loop_vars"] = alias_mappings

    try:
        import os
        from groggy.builder.ir.batch import compile_loop_to_batch_plan

        loop_vars_tuples = None
        if alias_mappings:
            loop_vars_tuples = [(initial, logical) for initial, logical in alias_mappings]

        batch_plan = compile_loop_to_batch_plan(body_copy, loop_vars_tuples)

        if os.environ.get("GROGGY_DEBUG_BATCH"):
            print(f"[PYTHON] Batch compilation result: {batch_plan is not None}")
            if batch_plan:
                print(f"[PYTHON] Instructions: {len(batch_plan.get('instructions', []))}")
                print(f"[PYTHON] Batch plan keys: {list(batch_plan.keys())}")

        if batch_plan and batch_plan.get("instructions"):
            loop_step["batch_plan"] = batch_plan
            loop_step["_batch_optimized"] = True
            if os.environ.get("GROGGY_DEBUG_BATCH"):
                print("[PYTHON] ✅ Batch plan added to loop_step!")
    except Exception as e:
        import warnings

        warnings.warn(
            f"Batch compilation failed, using fallback: {e}", RuntimeWarning
        )

        if __import__("os").environ.get("GROGGY_DEBUG_BATCH"):
            import traceback
            print("[PYTHON] ❌ Batch compilation failed:")
            traceback.print_exc()

    builder.steps.append(loop_step)


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

    @property
    def name(self) -> str:
        """Human-readable algorithm name."""
        return self._name

    @property
    def steps(self) -> list:
        """Expose legacy step list for inspection/testing."""
        return self._steps

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

        # Serialize pipeline_definition to JSON string for AttrValue
        pipeline_json = json.dumps(pipeline_definition)

        return {
            "id": "builder.step_pipeline",
            "params": {
                "name": _groggy.AttrValue(self._name),
                "steps": _groggy.AttrValue(pipeline_json),
            },
        }

    def _resolve_with_alias(self, value: Any, alias_map: Dict[str, str]) -> Any:
        # Handle VarHandle objects by extracting their name
        from groggy.builder.varhandle import VarHandle

        if isinstance(value, VarHandle):
            value = value.name

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

        if step_type in ["init_nodes", "core.init_nodes"]:
            params: Dict[str, Any] = {"target": step["output"]}
            default = step.get("default")
            if default is not None:
                # Resolve VarHandle to its name
                params["value"] = self._resolve_operand(default, alias_map)
            return {"id": "core.init_nodes", "params": params}

        if step_type in ["init_nodes_with_index", "core.init_nodes_with_index"]:
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

        if step_type in [
            "graph_node_count",
            "core.graph_node_count",
            "graph.graph_node_count",
        ]:
            return {"id": "core.graph_node_count", "params": {"target": step["output"]}}

        if step_type in [
            "graph_edge_count",
            "core.graph_edge_count",
            "graph.graph_edge_count",
        ]:
            return {"id": "core.graph_edge_count", "params": {"target": step["output"]}}

        if step_type in ["node_degree", "core.node_degree", "graph.degree"]:
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

        if step_type in ["attach_attr", "core.attach_attr", "attr.save", "attr.attach"]:
            # Resolve variable name through aliases
            # Handle different field names: 'source' or 'input'
            source = step.get("source", step.get("input"))
            resolved_input = self._resolve_with_alias(source, alias_map)
            return {
                "id": "core.attach_node_attr",
                "params": {"source": resolved_input, "attr": step["attr_name"]},
            }

        # Handle both prefixed and unprefixed versions for new DSL compatibility
        # Also handle both 'left'/'right' (legacy) and 'a'/'b' (IR) field names
        if step_type in ["core.add", "add"]:
            left = step.get("left", step.get("a"))
            right = step.get("right", step.get("b"))
            return {
                "id": "core.add",
                "params": {
                    "left": self._resolve_operand(left, alias_map),
                    "right": self._resolve_operand(right, alias_map),
                    "target": step["output"],
                },
            }

        if step_type in ["core.sub", "sub"]:
            left = step.get("left", step.get("a"))
            right = step.get("right", step.get("b"))
            return {
                "id": "core.sub",
                "params": {
                    "left": self._resolve_operand(left, alias_map),
                    "right": self._resolve_operand(right, alias_map),
                    "target": step["output"],
                },
            }

        if step_type in ["core.mul", "mul"]:
            left = step.get("left", step.get("a"))
            right = step.get("right", step.get("b"))
            return {
                "id": "core.mul",
                "params": {
                    "left": self._resolve_operand(left, alias_map),
                    "right": self._resolve_operand(right, alias_map),
                    "target": step["output"],
                },
            }

        if step_type in ["core.div", "div"]:
            left = step.get("left", step.get("a"))
            right = step.get("right", step.get("b"))
            return {
                "id": "core.div",
                "params": {
                    "left": self._resolve_operand(left, alias_map),
                    "right": self._resolve_operand(right, alias_map),
                    "target": step["output"],
                },
            }

        if step_type in ["core.recip", "recip"]:
            source = step.get("source", step.get("input"))
            return {
                "id": "core.recip",
                "params": {
                    "source": self._resolve_operand(source, alias_map),
                    "target": step["output"],
                    "epsilon": step.get("epsilon", 1e-10),
                },
            }

        if step_type in ["core.compare", "compare"]:
            left = step.get("left", step.get("a"))
            right = step.get("right", step.get("b"))
            return {
                "id": "core.compare",
                "params": {
                    "left": self._resolve_operand(left, alias_map),
                    "op": step["op"],
                    "right": self._resolve_operand(right, alias_map),
                    "target": step["output"],
                },
            }

        if step_type in ["core.where", "where"]:
            # Handle both 'condition'/'mask' and 'if_true'/'if_false' field names
            condition = step.get("condition", step.get("mask"))
            if_true = step.get("if_true")
            if_false = step.get("if_false")
            return {
                "id": "core.where",
                "params": {
                    "condition": self._resolve_operand(condition, alias_map),
                    "if_true": self._resolve_operand(if_true, alias_map),
                    "if_false": self._resolve_operand(if_false, alias_map),
                    "target": step["output"],
                },
            }

        if step_type in ["core.reduce_scalar", "reduce_scalar"]:
            return {
                "id": "core.reduce_scalar",
                "params": {
                    "source": self._resolve_operand(step["source"], alias_map),
                    "op": step["op"],
                    "target": step["output"],
                },
            }

        if step_type in ["core.constant", "constant", "init_scalar"]:
            return {
                "id": "core.init_scalar",
                "params": {"value": step["value"], "target": step["output"]},
            }

        if step_type in ["core.broadcast_scalar", "broadcast_scalar"]:
            return {
                "id": "core.broadcast_scalar",
                "params": {
                    "scalar": self._resolve_operand(step["scalar"], alias_map),
                    "reference": self._resolve_operand(step["reference"], alias_map),
                    "target": step["output"],
                },
            }

        if step_type in ["core.neighbor_agg", "neighbor_agg", "graph.neighbor_agg"]:
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

        if step_type == "iter.loop":
            iterations = step.get("iterations", 1)
            loop_vars = step.get("loop_vars")
            body_specs = []

            body_alias_map = alias_map.copy()
            for body_step in step.get("body", []):
                if body_step.get("type") == "alias":
                    source = body_step.get("source")
                    target = body_step.get("target")
                    if source and target:
                        resolved_source = self._resolve_with_alias(
                            source, body_alias_map
                        )
                        body_specs.append(
                            {
                                "id": "alias",
                                "params": {"source": resolved_source, "target": target},
                            }
                        )
                        # After executing the alias, future references should use the logical name.
                        body_alias_map.pop(target, None)
                    continue

                encoded_body = self._encode_step(body_step, body_alias_map)
                if encoded_body is not None:
                    body_specs.append(encoded_body)

            # After the loop, variables that were aliased inside should be treated as canonical
            # Remove any backward aliases for loop variables so they're not resolved to old sources
            if loop_vars:
                for _initial_var, loop_var in loop_vars:
                    # The loop variable is now the canonical name; don't resolve it backwards
                    alias_map.pop(loop_var, None)

            params: Dict[str, Any] = {
                "iterations": iterations,
                "body": body_specs,
            }
            if loop_vars:
                params["loop_vars"] = loop_vars

            # Forward batch_plan if compiled (Tier 1 batch execution)
            import os

            if "batch_plan" in step:
                params["batch_plan"] = step["batch_plan"]
                if os.environ.get("GROGGY_DEBUG_BATCH"):
                    print(f"[PYTHON _encode_step] ✅ Added batch_plan to params")
                    print(
                        f"[PYTHON _encode_step] Params keys now: {list(params.keys())}"
                    )
            else:
                if os.environ.get("GROGGY_DEBUG_BATCH"):
                    print(f"[PYTHON _encode_step] ⚠️  No batch_plan in step")
                    print(f"[PYTHON _encode_step] Step keys: {list(step.keys())}")

            return {
                "id": "iter.loop",
                "params": params,
            }

        if step_type in ["normalize_sum", "core.normalize_sum"]:
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

        if step_type in ["load_attr", "core.load_attr"]:
            params: Dict[str, Any] = {
                "attr": step["attr_name"],
                "target": step["output"],
            }
            default = step.get("default")
            if default is not None:
                params["default"] = default
            return {"id": "core.load_node_attr", "params": params}

        if step_type in ["load_edge_attr", "graph.load_edge_attr"]:
            params: Dict[str, Any] = {
                "attr": step["attr_name"],
                "target": step["output"],
            }
            default = step.get("default")
            if default is not None:
                params["default"] = default
            return {"id": "core.load_edge_attr", "params": params}

        if step_type in ["sample_nodes", "core.sample_nodes"]:
            params: Dict[str, Any] = {"target": step["output"]}
            if "fraction" in step:
                params["fraction"] = step["fraction"]
            if "count" in step:
                params["count"] = step["count"]
            if "seed" in step:
                params["seed"] = step["seed"]
            return {"id": "core.sample_nodes", "params": params}

        if step_type in ["sample_edges", "core.sample_edges"]:
            params: Dict[str, Any] = {"target": step["output"]}
            if "fraction" in step:
                params["fraction"] = step["fraction"]
            if "count" in step:
                params["count"] = step["count"]
            if "seed" in step:
                params["seed"] = step["seed"]
            return {"id": "core.sample_edges", "params": params}

        if step_type in ["sample.iterate_nodes", "iterate_nodes", "core.iterate_nodes"]:
            return {
                "id": "sample.iterate_nodes",
                "params": {"target": step["output"]},
            }

        if step_type in ["sample.iterate_edges", "iterate_edges", "core.iterate_edges"]:
            return {
                "id": "sample.iterate_edges",
                "params": {"target": step["output"]},
            }

        if step_type in ["sample.neighbors", "neighbors", "core.neighbors"]:
            source = step.get("input", step.get("source"))
            params = {
                "source": self._resolve_operand(source, alias_map),
                "target": step["output"],
                "hops": step.get("hops", 1),
            }
            return {"id": "sample.neighbors", "params": params}

        if step_type in ["sample.emit_subgraphs", "emit_subgraphs", "core.emit_subgraphs"]:
            source = step.get("input", step.get("source"))
            params = {
                "source": self._resolve_operand(source, alias_map),
                "target": step["output"],
                "mode": step.get("mode", "per_item"),
                "induced": step.get("induced", True),
            }
            return {"id": "sample.emit_subgraphs", "params": params}

        if step_type in ["sample.for_each", "for_each", "core.for_each"]:
            body_steps = step.get("body", [])
            body_alias_map = dict(alias_map)
            body_specs: list[Dict[str, Any]] = []

            for body_step in body_steps:
                if body_step.get("type") == "alias":
                    source = body_step.get("source")
                    target = body_step.get("target")
                    if source and target:
                        resolved_source = self._resolve_with_alias(source, body_alias_map)
                        body_alias_map[target] = resolved_source
                    continue

                encoded_body = self._encode_step(body_step, body_alias_map)
                if encoded_body is not None:
                    body_specs.append(encoded_body)

            params = {
                "source": self._resolve_operand(
                    step.get("input", step.get("source")), alias_map
                ),
                "target": step["output"],
                "body": body_specs,
            }
            return {"id": "sample.for_each", "params": params}

        if step_type == "core.histogram":
            # Handle both 'source' and 'input' field names (IR may change them)
            source_val = step.get(
                "source",
                step.get(
                    "input", step.get("inputs", [None])[0] if "inputs" in step else None
                ),
            )
            return {
                "id": "core.histogram",
                "params": {
                    "source": (
                        self._resolve_operand(source_val, alias_map)
                        if source_val
                        else step["output"]
                    ),
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

        if step_type == "core.execution_block":
            body = step.get("body", {"nodes": []})
            rewritten_nodes = []
            for node in body.get("nodes", []):
                node_copy = dict(node)
                inputs = node_copy.get("inputs") or []
                node_copy["inputs"] = [
                    (
                        self._resolve_with_alias(inp, alias_map)
                        if isinstance(inp, str)
                        else inp
                    )
                    for inp in inputs
                ]
                output = node_copy.get("output")
                if isinstance(output, str):
                    node_copy["output"] = self._resolve_with_alias(output, alias_map)
                rewritten_nodes.append(node_copy)

            return {
                "id": "core.execution_block",
                "params": {
                    "mode": step.get("mode", "message_pass"),
                    "target": self._resolve_with_alias(step["target"], alias_map),
                    "options": step.get("options", {}),
                    "body": {"nodes": rewritten_nodes},
                },
            }

        raise ValueError(f"Unsupported builder step type: {step_type}")

    def _validate(self) -> Tuple[List[str], List[str]]:
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
            if step_type == "iter.loop":
                loop_vars = step.get("loop_vars") or []
                for loop_var in loop_vars:
                    # loop_var can be a string or a [initial, logical] pair
                    if isinstance(loop_var, list):
                        defined_vars.add(loop_var[1])  # Add the logical name
                    else:
                        defined_vars.add(loop_var)

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

    def __str__(self) -> str:
        return self.__repr__()


class BuiltSampler(BuiltAlgorithm):
    """Sampler pipeline built from composed steps."""

    def __init__(self, name: str, steps: list):
        super().__init__(name, steps)
        self._id = f"custom.sample.{name}"

    def to_spec(self) -> Dict[str, Any]:
        alias_map: Dict[str, str] = {}
        encoded_steps: list[Dict[str, Any]] = []
        prev_output: Optional[str] = None
        prev_non_alias_step_type: Optional[str] = None
        implicit_source_steps = {
            "sample.neighbors",
            "neighbors",
            "core.neighbors",
            "sample.emit_subgraphs",
            "emit_subgraphs",
            "core.emit_subgraphs",
            "sample.for_each",
            "for_each",
            "core.for_each",
        }

        for step in self._steps:
            if step.get("type") == "alias":
                source = step.get("source")
                target = step.get("target")
                if source and target:
                    resolved_source = self._resolve_with_alias(source, alias_map)
                    alias_map[target] = resolved_source
                continue

            step_to_encode = step
            step_type = step.get("type")
            if (
                step_type in implicit_source_steps
                and "input" not in step
                and "source" not in step
                and prev_output is not None
            ):
                step_to_encode = dict(step)
                step_to_encode["input"] = prev_output

            # IR sampler lowering may emit `core.neighbors` without an explicit source.
            # If the previous step sampled node/edge IDs, reconstruct the missing
            # iterate step so `sample.neighbors` receives a subgraph array.
            if (
                step_type in {"core.neighbors", "sample.neighbors", "neighbors"}
                and prev_output is not None
                and "input" not in step
                and "source" not in step
            ):
                if prev_non_alias_step_type in {
                    "core.sample_nodes",
                    "sample_nodes",
                    "core.sample_edges",
                    "sample_edges",
                }:
                    bridge_name = f"{prev_output}__seed_subgraphs"
                    encoded_steps.append(
                        {
                            "id": "sample.emit_subgraphs",
                            "params": {
                                "source": prev_output,
                                "target": bridge_name,
                                "mode": "per_item",
                                "induced": True,
                            },
                        }
                    )
                    step_to_encode = dict(step_to_encode)
                    step_to_encode["input"] = bridge_name

            encoded = self._encode_step(step_to_encode, alias_map)
            if encoded is not None:
                encoded_steps.append(encoded)
            if isinstance(step.get("output"), str):
                prev_output = step["output"]
            if isinstance(step_type, str):
                prev_non_alias_step_type = step_type

        pipeline_definition = {
            "name": self._name,
            "steps": encoded_steps,
        }

        pipeline_json = json.dumps(pipeline_definition)

        return {
            "id": "builder.sample_pipeline",
            "params": {
                "name": _groggy.AttrValue(self._name),
                "steps": _groggy.AttrValue(pipeline_json),
            },
        }

    def __repr__(self) -> str:
        return f"BuiltSampler('{self._name}', {len(self._steps)} steps)"
