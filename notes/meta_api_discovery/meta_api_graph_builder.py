#!/usr/bin/env python3
"""
Dynamic Meta API Graph Builder for Groggy.

This script builds the API meta-graph directly from live Groggy objects:
- Nodes: Groggy object types + return types
- Edges: Methods (object_type -> return_type)

No JSON inputs. Runs discovery, tests (optional), and inference in one pass.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import groggy


@dataclass
class MethodInfo:
    object_type: str
    method_name: str
    return_type: str
    return_type_source: str
    parameters: List[Dict[str, Any]]
    signature: str
    docstring: str
    is_property: bool


class MetaAPIGraphBuilder:
    def __init__(self, run_tests: bool = True):
        self.run_tests = run_tests
        self.graph = groggy.Graph()
        self.discovered_types: Set[str] = set()
        self.type_info: Dict[str, Dict[str, Any]] = {}
        self.methods: List[MethodInfo] = []

        self.type_patterns = {
            r"\b(?:dictionary|dict|Dict)\b": "dict",
            r"\b(?:list|List|array|Array)\b": "list",
            r"\b(?:string|str|String)\b": "str",
            r"\b(?:integer|int|Integer)\b": "int",
            r"\b(?:number|float|Float|numeric)\b": "float",
            r"\b(?:boolean|bool|Boolean)\b": "bool",
            r"\bGraph\b": "Graph",
            r"\bSubgraph\b": "Subgraph",
            r"\bBaseTable\b": "BaseTable",
            r"\bNodesTable\b": "NodesTable",
            r"\bEdgesTable\b": "EdgesTable",
            r"\bGraphTable\b": "GraphTable",
            r"\bBaseArray\b": "BaseArray",
            r"\bNodesArray\b": "NodesArray",
            r"\bEdgesArray\b": "EdgesArray",
            r"\bSubgraphArray\b": "SubgraphArray",
            r"\bTableArray\b": "TableArray",
            r"\bComponentsArray\b": "ComponentsArray",
            r"\bMetaNodeArray\b": "MetaNodeArray",
            r"\bStatsArray\b": "StatsArray",
            r"\bNumArray\b": "NumArray",
            r"\bMatrix\b": "Matrix",
            r"\bGraphMatrix\b": "GraphMatrix",
            r"\bNodesAccessor\b": "NodesAccessor",
            r"\bEdgesAccessor\b": "EdgesAccessor",
            r"\bMetaNode\b": "MetaNode",
            r"\bDisplayConfig\b": "DisplayConfig",
            r"\bTableFormatter\b": "TableFormatter",
            r"\bAggregationResult\b": "AggregationResult",
            r"\bGroupedAggregationResult\b": "GroupedAggregationResult",
            r"\bHistoricalView\b": "HistoricalView",
            r"\bHistoryStatistics\b": "HistoryStatistics",
            r"\bBranchInfo\b": "BranchInfo",
            r"\bCommit\b": "Commit",
            r"\bGraphArray\b": "GraphArray",
        }

        self.skip_methods = {
            "clear",
            "reset",
            "close",
            "destroy",
            "delete",
            "remove",
            "pop",
            "shutdown",
            "exit",
            "quit",
        }

    def _bootstrap_graph(self) -> groggy.Graph:
        g = groggy.Graph()
        for i in range(5):
            g.add_node(i, name=f"node_{i}", age=20 + i * 5, category="A" if i % 2 == 0 else "B")
        for i in range(4):
            g.add_edge(i, i + 1, weight=1.0 + i * 0.5, edge_type="connection")
        return g

    def _discover_objects(self) -> Dict[str, Any]:
        core_objects: Dict[str, Any] = {}

        g = self._bootstrap_graph()
        core_objects["Graph"] = g

        # Accessors
        try:
            core_objects["NodesAccessor"] = g.nodes
        except Exception:
            pass
        try:
            core_objects["EdgesAccessor"] = g.edges
        except Exception:
            pass

        # Tables
        try:
            core_objects["GraphTable"] = g.table()
        except Exception:
            pass
        try:
            core_objects["NodesTable"] = g.nodes.table()
        except Exception:
            pass
        try:
            core_objects["EdgesTable"] = g.edges.table()
        except Exception:
            pass

        # Arrays
        try:
            core_objects["NodesArray"] = g.nodes.array()
        except Exception:
            pass
        try:
            core_objects["NumArray"] = g.nodes.ids()
        except Exception:
            pass
        try:
            subgraph_array = g.nodes.group_by("category")
            core_objects["SubgraphArray"] = subgraph_array
            try:
                core_objects["TableArray"] = subgraph_array.table()
            except Exception:
                pass
            try:
                if len(subgraph_array) > 0:
                    core_objects["Subgraph"] = subgraph_array[0]
            except Exception:
                pass
        except Exception:
            pass

        # Matrix types
        try:
            core_objects["GraphMatrix"] = g.to_matrix()
        except Exception:
            pass
        try:
            matrix_obj = g.nodes.matrix()
            matrix_type = type(matrix_obj).__name__
            core_objects[matrix_type] = matrix_obj
        except Exception:
            pass

        # Components
        try:
            core_objects["ComponentsArray"] = g.connected_components()
        except Exception:
            pass

        # BaseTable if available
        try:
            nodes_table = core_objects.get("NodesTable")
            if nodes_table is not None:
                core_objects["BaseTable"] = nodes_table.base_table()
        except Exception:
            pass

        # Discover instantiable classes on groggy module
        for attr_name in dir(groggy):
            if attr_name.startswith("_") or attr_name.islower() or attr_name in core_objects:
                continue
            try:
                attr_obj = getattr(groggy, attr_name)
                if inspect.isclass(attr_obj):
                    try:
                        instance = attr_obj()
                        core_objects[attr_name] = instance
                    except Exception:
                        continue
            except Exception:
                continue

        return core_objects

    def _extract_parameters_info(self, method) -> List[Dict[str, Any]]:
        params = []
        try:
            sig = inspect.signature(method)
            for param_name, param in sig.parameters.items():
                if param_name in ["self", "py"]:
                    continue
                param_info = {
                    "name": param_name,
                    "type": "Any",
                    "default": None,
                    "required": param.default == inspect.Parameter.empty,
                }
                if param.annotation != inspect.Parameter.empty:
                    param_info["type"] = str(param.annotation).replace("<class '", "").replace("'>", "")
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = str(param.default)
                params.append(param_info)
        except Exception:
            pass
        return params

    def _extract_return_type_from_signature(self, method) -> Optional[str]:
        try:
            sig = inspect.signature(method)
            if sig.return_annotation and sig.return_annotation != inspect.Signature.empty:
                annotation = str(sig.return_annotation)
                annotation = annotation.replace("<class '", "").replace("'>", "")
                annotation = annotation.replace("groggy.", "")
                if "." in annotation:
                    annotation = annotation.split(".")[-1]
                self.discovered_types.add(annotation)
                return annotation
        except Exception:
            pass
        return None

    def _extract_return_type_from_docstring(self, docstring: str) -> Optional[str]:
        if not docstring:
            return None

        for type_pattern, canonical_type in self.type_patterns.items():
            word_pattern = r"\b" + type_pattern.strip(r"\b") + r"\b"
            if re.search(word_pattern, docstring, re.IGNORECASE):
                self.discovered_types.add(canonical_type)
                return canonical_type

        return_patterns = [
            r"Returns?:?\s*([A-Z][A-Za-z0-9_]*)",
            r"-> ([A-Z][A-Za-z0-9_]*)",
            r"return[s]?\s+([A-Z][A-Za-z0-9_]*)",
            r"PyResult<([A-Za-z_][A-Za-z0-9_]*)>",
            r"Py([A-Z][A-Za-z0-9_]*)",
        ]

        reject_words = {
            "the",
            "and",
            "this",
            "that",
            "with",
            "from",
            "for",
            "new",
            "old",
            "ing",
            "ion",
            "tion",
            "thon",
            "report",
            "empty",
            "New",
            "a",
            "s",
            "L",
        }

        for pattern in return_patterns:
            match = re.search(pattern, docstring, re.IGNORECASE)
            if match:
                return_type = match.group(1)
                if (
                    len(return_type) <= 2
                    or return_type.lower() in reject_words
                    or not return_type[0].isupper()
                    or not return_type.isalpha()
                ):
                    continue
                return_type = return_type.replace("Py", "")
                if len(return_type) <= 2 or return_type.lower() in reject_words:
                    continue
                if return_type[0].isupper() and len(return_type) >= 3 and return_type.isalnum():
                    self.discovered_types.add(return_type)
                    return return_type
        return None

    def _try_method_call(self, method, method_name: str) -> Optional[str]:
        if method_name in self.skip_methods:
            return None

        try:
            sig = inspect.signature(method)
            params = sig.parameters
            required_params = [
                p for p in params.values() if p.default == inspect.Parameter.empty and p.name not in ["self", "py"]
            ]
            if len(required_params) == 0:
                result = method()
                return type(result).__name__
        except Exception:
            return None

        # Targeted parameter attempts for common safe methods
        param_patterns = {
            "filter": ["node_id > 0", lambda x: True],
            "column": ["node_id", "name", "type"],
            "head": [5],
            "tail": [5],
            "sample": [3],
            "group_by": ["category"],
            "to_csv": ["/tmp/groggy_meta_api_tmp.csv"],
            "to_json": ["/tmp/groggy_meta_api_tmp.json"],
        }
        for key, params_list in param_patterns.items():
            if key in method_name.lower():
                for params in params_list:
                    try:
                        if isinstance(params, (list, tuple)):
                            result = method(*params)
                        else:
                            result = method(params)
                        return type(result).__name__
                    except Exception:
                        continue
        return None

    def _discover_methods(self, obj_name: str, obj_instance: Any) -> List[MethodInfo]:
        methods: List[MethodInfo] = []

        try:
            method_names = [name for name in dir(obj_instance.__class__) if not name.startswith("_")]
        except Exception:
            try:
                method_names = [name for name in dir(obj_instance) if not name.startswith("_")]
            except Exception:
                return methods

        for method_name in method_names:
            if method_name.startswith("_"):
                continue
            try:
                method = getattr(obj_instance, method_name)
                if not callable(method):
                    continue
                docstring = inspect.getdoc(method) or ""

                return_type = None
                return_type_source = "unknown"

                if self.run_tests:
                    tested = self._try_method_call(method, method_name)
                    if tested:
                        return_type = tested
                        return_type_source = "test_execution"

                if return_type is None:
                    return_type = self._extract_return_type_from_signature(method)
                    if return_type:
                        return_type_source = "signature"

                if return_type is None:
                    inferred = self._extract_return_type_from_docstring(docstring)
                    if inferred:
                        return_type = inferred
                        return_type_source = "docstring"

                if return_type is None:
                    return_type = "Unknown"
                    return_type_source = "unknown"

                parameters = self._extract_parameters_info(method)
                signature = str(inspect.signature(method)) if hasattr(inspect, "signature") else "Unknown"
                is_property = isinstance(getattr(obj_instance.__class__, method_name, None), property)

                self.discovered_types.add(obj_name)
                self.discovered_types.add(return_type)

                methods.append(
                    MethodInfo(
                        object_type=obj_name,
                        method_name=method_name,
                        return_type=return_type,
                        return_type_source=return_type_source,
                        parameters=parameters,
                        signature=signature,
                        docstring=docstring,
                        is_property=is_property,
                    )
                )
            except Exception:
                continue

        return methods

    def discover(self) -> None:
        objects = self._discover_objects()

        for obj_name, obj in objects.items():
            obj_type = obj.__class__.__name__
            self.type_info[obj_name] = {
                "type_name": obj_name,
                "class_name": obj_type,
                "module": obj.__class__.__module__,
                "docstring": inspect.getdoc(obj.__class__) or "",
            }
            self.methods.extend(self._discover_methods(obj_name, obj))

    def build_meta_graph(self) -> groggy.Graph:
        nodes_data = []
        for type_name in sorted(self.discovered_types):
            node_data = {
                "type_name": type_name,
                "category": "core" if type_name in self.type_info else "inferred",
                "methods_count": 0,
                "description": "",
            }
            if type_name in self.type_info:
                node_data.update(self.type_info[type_name])
            nodes_data.append(node_data)

        self.graph.add_nodes(nodes_data, uid_key="type_name")

        edges_data = []
        for method in self.methods:
            requires_parameters = [p["name"] for p in method.parameters if p["required"]]
            parameter_types = {p["name"]: p["type"] for p in method.parameters if p["type"] != "Any"}
            enhanced_signature = "(" + ", ".join(
                [
                    f"{p['name']}: {p['type']}" if p["type"] != "Any" else p["name"]
                    for p in method.parameters
                ]
            ) + ")"

            edge_data = {
                "object_type": method.object_type,
                "return_type": method.return_type,
                "name": method.method_name,
                "signature": method.signature,
                "enhanced_signature": enhanced_signature,
                "parameter_types": json.dumps(parameter_types),
                "requires_parameters": json.dumps(requires_parameters),
                "source_object": method.object_type,
                "source_type": method.object_type,
                "is_property": method.is_property,
                "doc": method.docstring,
                "parameters_count": len(method.parameters),
                "method_full_name": f"{method.object_type}.{method.method_name}",
                "relationship": f"{method.object_type} -> {method.return_type}",
                "return_type_source": method.return_type_source,
                "edge_object_type": method.object_type,
                "edge_return_type": method.return_type,
            }
            edges_data.append(edge_data)

        for edge_data in edges_data:
            if "object_type" not in edge_data:
                edge_data["object_type"] = edge_data["edge_object_type"]
            if "return_type" not in edge_data:
                edge_data["return_type"] = edge_data["edge_return_type"]

        self.graph.add_edges(
            edges_data,
            uid_key="type_name",
            source="edge_object_type",
            target="edge_return_type",
        )
        return self.graph

    def export_bundle(self, output_dir: Path) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        graph_table = self.graph.table()
        graph_table.save_bundle(str(output_dir))

        summary = {
            "meta_graph_stats": {
                "nodes": self.graph.node_count(),
                "edges": self.graph.edge_count(),
                "types_discovered": len(self.discovered_types),
            },
            "type_info": self.type_info,
        }

        summary_path = output_dir / "meta_api_graph_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Groggy Meta API Graph dynamically.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip dynamic method calls.")
    parser.add_argument(
        "--output-dir",
        default="notes/meta_api_discovery/meta_api_graph_bundle",
        help="Directory to write the graph bundle.",
    )
    args = parser.parse_args()

    builder = MetaAPIGraphBuilder(run_tests=not args.skip_tests)
    builder.discover()
    builder.build_meta_graph()
    summary = builder.export_bundle(Path(args.output_dir))

    print("Meta API Graph built.")
    print(f"Nodes: {summary['meta_graph_stats']['nodes']}")
    print(f"Edges: {summary['meta_graph_stats']['edges']}")
    print(f"Types discovered: {summary['meta_graph_stats']['types_discovered']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
