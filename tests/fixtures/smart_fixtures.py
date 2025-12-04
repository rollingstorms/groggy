"""
Smart Fixture Generation for Groggy Testing

Automatically generates valid test data based on method signatures and parameter types.
This is part of Milestone 1: Graph Core Foundation.

Key Features:
- Automatic parameter provisioning based on type hints and names
- Graph state management for consistent test environments
- Diverse test data generation (edge cases, normal cases, stress cases)
- Reusable patterns for all object types
"""

import inspect
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

try:
    import groggy as gr
except ImportError:
    print("Warning: groggy not available for fixtures")
    gr = None


@dataclass
class TestCase:
    """Represents a single test case with arguments and expected behavior"""

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    description: str
    should_succeed: bool = True
    expected_exception: Optional[Type[Exception]] = None


class FixtureFactory:
    """
    Generate valid test data based on method signatures.

    This factory understands Groggy's parameter patterns and can generate
    appropriate test values for nodes, edges, attributes, and other types.
    """

    def __init__(self, graph: Optional["gr.Graph"] = None):
        """Initialize with an optional graph for context"""
        self.graph = graph or (gr.Graph() if gr else None)
        self.node_ids = []
        self.edge_ids = []
        self.branch_names = []
        self.state_ids = []

        # Initialize with some basic test data
        if self.graph:
            self._setup_basic_data()

    def _setup_basic_data(self):
        """Create basic nodes and edges for testing"""
        if not self.graph:
            return

        # Create test nodes with various attribute types
        self.node_ids = [
            self.graph.add_node(label="Alice", age=29, active=True),
            self.graph.add_node(label="Bob", age=35, active=False),
            self.graph.add_node(label="Carol", age=31, department="Engineering"),
            self.graph.add_node(),  # Node with no attributes
        ]

        # Create test edges
        if len(self.node_ids) >= 2:
            self.edge_ids = [
                self.graph.add_edge(
                    self.node_ids[0],
                    self.node_ids[1],
                    weight=1.0,
                    relationship="friend",
                ),
                self.graph.add_edge(
                    self.node_ids[1],
                    self.node_ids[2],
                    weight=0.5,
                    relationship="colleague",
                ),
            ]

        # Create a branch for testing
        self.state_ids.append(self.graph.commit("Initial test state", "FixtureFactory"))

    def get_fixture_for_param(
        self, param_name: str, param_type: Type, method_name: str = ""
    ) -> Any:
        """
        Return a valid test value for a parameter based on its name and type.

        Args:
            param_name: Name of the parameter
            param_type: Type annotation (if available)
            method_name: Name of the method being tested (for context)

        Returns:
            Appropriate test value
        """

        # Node ID parameters
        if param_name in [
            "node_id",
            "node",
            "source",
            "target",
            "start_node",
            "end_node",
        ]:
            return self.node_ids[0] if self.node_ids else 1

        # Edge ID parameters
        if param_name in ["edge_id", "edge"]:
            return self.edge_ids[0] if self.edge_ids else 1

        # Multiple node/edge parameters
        if param_name in ["node_ids", "nodes"]:
            return self.node_ids[:2] if len(self.node_ids) >= 2 else [1, 2]
        if param_name in ["edge_ids", "edges"]:
            return self.edge_ids[:1] if self.edge_ids else [1]

        # Attribute parameters - use simple values that don't cause FFI issues
        if param_name in ["attrs", "attributes", "attrs_dict"]:
            return {"test_attr": "simple_string", "count": 42}
        if param_name in ["attr_name", "attribute_name"]:
            return "test_attr"
        if param_name in ["attr_value", "value"]:
            return "test_value"

        # Branch and state parameters
        if param_name in ["branch_name", "branch"]:
            return "test_branch"
        if param_name in ["state_id", "commit_id"]:
            return self.state_ids[0] if self.state_ids else 1

        # Commit requires both message and author
        if param_name == "message" and method_name == "commit":
            return "Test commit message"
        if param_name == "author" and method_name == "commit":
            return "Test Author"

        # Query and filter parameters
        if param_name in ["query", "node_query", "edge_query"]:
            return "age > 0"  # Use actual attributes, not computed values like degree
        if param_name in ["node_filter", "edge_filter"]:
            if "node" in param_name:
                # Create a proper NodeFilter object
                try:
                    import groggy as gr

                    if gr:
                        return gr.NodeFilter.attribute_filter(
                            "age", gr.AttributeFilter.greater_than(0)
                        )
                except:
                    pass
                return "age > 0"  # Fallback to string
            else:
                # Create a proper EdgeFilter object
                try:
                    import groggy as gr

                    if gr:
                        return gr.EdgeFilter.attribute_filter(
                            "strength", gr.AttributeFilter.greater_than(0.5)
                        )
                except:
                    pass
                return "strength > 0.5"  # Fallback to string
        if param_name in ["filter_string", "filter_query"]:
            return "age > 0"
        if param_name in ["depth", "max_depth"]:
            return 2
        if param_name in ["limit", "max_results"]:
            return 10

        # File and data parameters
        if param_name in ["path", "file_path", "filename"]:
            return "/tmp/test_graph.json"
        if param_name in ["data", "graph_data"]:
            return {"nodes": [], "edges": []}

        # Collection parameters for bulk operations
        if param_name in ["edges_data", "edges_list"]:
            if self.node_ids and len(self.node_ids) >= 2:
                return [(self.node_ids[0], self.node_ids[1])]
            return [(1, 2)]
        if param_name in ["nodes_data", "nodes_list"]:
            return [{"label": "test_node"}]

        # Boolean parameters
        if param_name in ["inplace", "in_place"]:
            return False  # Safer default for testing
        if param_name in ["directed", "is_directed"]:
            return True

        # Generic string parameters
        if param_name in ["label", "name", "key"]:
            return "test_label"
        if param_name in ["message", "description"]:
            return "test message"

        # Numeric parameters
        if param_name in ["weight", "value", "threshold"]:
            return 1.0
        if param_name in ["count", "num", "size"]:
            return 5

        # Type-based defaults when parameter name doesn't give us a hint
        if param_type:
            if param_type == str:
                return f"test_{param_name}"
            elif param_type == int:
                return 1
            elif param_type == float:
                return 1.0
            elif param_type == bool:
                return True
            elif param_type == list:
                return []
            elif param_type == dict:
                return {}

        # Fallback
        return None

    def generate_test_cases(
        self, obj: Any, method_name: str, num_cases: int = 3
    ) -> List[TestCase]:
        """
        Generate multiple test cases for a method.

        Args:
            obj: Object instance to test
            method_name: Name of the method
            num_cases: Number of test cases to generate

        Returns:
            List of TestCase objects
        """
        method = getattr(obj, method_name)

        try:
            sig = inspect.signature(method)
        except (ValueError, TypeError):
            # Method doesn't have inspectable signature
            return [
                TestCase(args=(), kwargs={}, description=f"Basic call to {method_name}")
            ]

        test_cases = []

        # Generate basic valid case
        args, kwargs = self._generate_args_kwargs(sig, method_name)
        test_cases.append(
            TestCase(
                args=args,
                kwargs=kwargs,
                description=f"Basic valid call to {method_name}",
                should_succeed=True,
            )
        )

        # Generate edge case variations if we have more cases to generate
        for i in range(1, num_cases):
            args, kwargs = self._generate_args_kwargs(sig, method_name, variation=i)
            test_cases.append(
                TestCase(
                    args=args,
                    kwargs=kwargs,
                    description=f"Variation {i} call to {method_name}",
                    should_succeed=True,
                )
            )

        return test_cases

    def _generate_args_kwargs(
        self, sig: inspect.Signature, method_name: str, variation: int = 0
    ) -> Tuple[Tuple, Dict]:
        """Generate args and kwargs for a method signature"""
        args = []
        kwargs = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Get the parameter type
            param_type = (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else None
            )

            # Generate value
            value = self.get_fixture_for_param(param_name, param_type, method_name)

            # Apply variations
            if variation > 0 and value is not None:
                value = self._apply_variation(value, variation, param_name)

            # Decide whether to use as positional or keyword argument
            if param.default == inspect.Parameter.empty:
                # Required parameter
                if param.kind in [
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ]:
                    args.append(value)
                else:
                    kwargs[param_name] = value
            else:
                # Optional parameter - sometimes include it
                if random.choice([True, False]):
                    kwargs[param_name] = value

        return tuple(args), kwargs

    def _apply_variation(self, value: Any, variation: int, param_name: str) -> Any:
        """Apply variations to test values for edge case testing"""
        if value is None:
            return None

        if isinstance(value, str):
            if variation == 1:
                return ""  # Empty string
            elif variation == 2:
                return "a" * 100  # Long string
        elif isinstance(value, int):
            if variation == 1:
                return 0
            elif variation == 2:
                return -1
        elif isinstance(value, list):
            if variation == 1:
                return []  # Empty list
            elif variation == 2 and value:
                return value * 3  # Longer list
        elif isinstance(value, dict):
            if variation == 1:
                return {}  # Empty dict
            elif variation == 2:
                return {**value, "extra_key": "extra_value"}  # Extended dict

        return value


class GraphFixtures:
    """
    Predefined graph structures for testing various scenarios.

    These fixtures provide consistent test environments for different
    testing needs: small graphs, complex graphs, edge cases, etc.
    """

    @staticmethod
    def empty_graph() -> "gr.Graph":
        """Create an empty graph"""
        return gr.Graph() if gr else None

    @staticmethod
    def single_node_graph() -> "gr.Graph":
        """Create a graph with a single node"""
        if not gr:
            return None
        g = gr.Graph()
        g.add_node(label="Lonely", value=42)
        return g

    @staticmethod
    def simple_path_graph(length: int = 3) -> "gr.Graph":
        """Create a simple path graph: 1-2-3-...-n"""
        if not gr:
            return None
        g = gr.Graph()
        nodes = [g.add_node(label=f"Node{i}", position=i) for i in range(length)]
        for i in range(length - 1):
            g.add_edge(nodes[i], nodes[i + 1], weight=1.0)
        return g

    @staticmethod
    def simple_cycle_graph(length: int = 4) -> "gr.Graph":
        """Create a simple cycle graph"""
        if not gr:
            return None
        g = gr.Graph()
        nodes = [g.add_node(label=f"Node{i}") for i in range(length)]
        for i in range(length):
            g.add_edge(nodes[i], nodes[(i + 1) % length], weight=1.0)
        return g

    @staticmethod
    def star_graph(center_degree: int = 5) -> "gr.Graph":
        """Create a star graph with center connected to all other nodes"""
        if not gr:
            return None
        g = gr.Graph()
        center = g.add_node(label="Center", type="hub")
        leaves = [
            g.add_node(label=f"Leaf{i}", type="leaf") for i in range(center_degree)
        ]
        for leaf in leaves:
            g.add_edge(center, leaf, relationship="spoke")
        return g

    @staticmethod
    def complete_graph(n: int = 4) -> "gr.Graph":
        """Create a complete graph where every node connects to every other node"""
        if not gr:
            return None
        g = gr.Graph()
        nodes = [g.add_node(label=f"Node{i}") for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(nodes[i], nodes[j], weight=1.0)
        return g

    @staticmethod
    def multi_component_graph() -> "gr.Graph":
        """Create a graph with multiple disconnected components"""
        if not gr:
            return None
        g = gr.Graph()

        # Component 1: Triangle
        n1 = g.add_node(label="A1", component=1)
        n2 = g.add_node(label="A2", component=1)
        n3 = g.add_node(label="A3", component=1)
        g.add_edge(n1, n2)
        g.add_edge(n2, n3)
        g.add_edge(n3, n1)

        # Component 2: Path
        n4 = g.add_node(label="B1", component=2)
        n5 = g.add_node(label="B2", component=2)
        g.add_edge(n4, n5)

        # Component 3: Isolated node
        g.add_node(label="C1", component=3)

        return g

    @staticmethod
    def attributed_graph() -> "gr.Graph":
        """Create a graph with diverse attribute types for testing attribute operations"""
        if not gr:
            return None
        g = gr.Graph()

        # Nodes with simpler attribute types to avoid FFI issues
        n1 = g.add_node(label="Alice", age=29, salary=75000.50, active=True)
        n2 = g.add_node(label="Bob", age=35, salary=85000.75, active=False)
        n3 = g.add_node(label="Carol", age=31, salary=65000.25, active=True)

        # Edges with simple attributes
        g.add_edge(n1, n2, relationship="colleague", strength=0.8, years_known=3)
        g.add_edge(n2, n3, relationship="mentor", strength=0.9, years_known=1)
        g.add_edge(n1, n3, relationship="friend", strength=0.7, years_known=2)

        return g

    @staticmethod
    def large_graph(num_nodes: int = 100, edge_probability: float = 0.1) -> "gr.Graph":
        """Create a larger random graph for performance testing"""
        if not gr:
            return None
        g = gr.Graph()

        # Add nodes
        nodes = []
        for i in range(num_nodes):
            node = g.add_node(
                label=f"Node{i}",
                value=random.randint(1, 100),
                category=random.choice(["A", "B", "C"]),
            )
            nodes.append(node)

        # Add random edges
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < edge_probability:
                    g.add_edge(nodes[i], nodes[j], weight=random.random())

        return g
