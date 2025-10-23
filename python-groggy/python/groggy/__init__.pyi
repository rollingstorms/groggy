from typing import Any

from .pipeline import pipeline, Pipeline, apply
from .builder import builder, AlgorithmBuilder, VarHandle

class Graph:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def add_node(self, *args: Any, **kwargs: Any) -> int: ...
    def add_edge(self, *args: Any, **kwargs: Any) -> int: ...
    def view(self) -> 'Subgraph': ...
    def connected_components(self, *args: Any, **kwargs: Any) -> Any: ...
    nodes: Any
    edges: Any

class Subgraph:
    def apply(self, algorithm_or_pipeline: Any) -> 'Subgraph': ...
    def table(self) -> Any: ...
    def viz(self) -> Any: ...

__all__ = [
    "Graph",
    "Subgraph",
    "pipeline",
    "Pipeline",
    "apply",
    "builder",
    "AlgorithmBuilder",
    "VarHandle",
]
