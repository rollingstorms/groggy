"""
Graph Sample Data for Testing

Pre-built graph datasets for consistent testing across modules.
These provide known structures with predictable properties.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import groggy as gr
except ImportError:
    gr = None


def karate_club_data() -> Dict[str, List]:
    """Zachary's Karate Club network data"""
    return {
        "nodes": [
            {"id": i, "label": f"Member{i}", "instructor": i in [0, 33]}
            for i in range(34)
        ],
        "edges": [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 17),
            (0, 19),
            (0, 21),
            (0, 31),
            (1, 2),
            (1, 3),
            (1, 7),
            (1, 13),
            (1, 17),
            (1, 19),
            (1, 21),
            (1, 30),
            (2, 3),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 13),
            (2, 27),
            (2, 28),
            (2, 32),
            (3, 7),
            (3, 12),
            (3, 13),
            (4, 6),
            (4, 10),
            (5, 6),
            (5, 10),
            (5, 16),
            (6, 16),
            (8, 30),
            (8, 32),
            (8, 33),
            (9, 33),
            (13, 33),
            (14, 32),
            (14, 33),
            (15, 32),
            (15, 33),
            (18, 32),
            (18, 33),
            (19, 33),
            (20, 32),
            (20, 33),
            (22, 32),
            (22, 33),
            (23, 25),
            (23, 27),
            (23, 29),
            (23, 32),
            (23, 33),
            (24, 25),
            (24, 27),
            (24, 31),
            (25, 31),
            (26, 29),
            (26, 33),
            (27, 33),
            (28, 31),
            (28, 33),
            (29, 32),
            (29, 33),
            (30, 32),
            (30, 33),
            (31, 32),
            (31, 33),
            (32, 33),
        ],
    }


def small_social_network_data() -> Dict[str, List]:
    """Small social network for testing social analysis features"""
    return {
        "nodes": [
            {
                "id": "alice",
                "name": "Alice",
                "age": 28,
                "location": "NYC",
                "occupation": "Engineer",
            },
            {
                "id": "bob",
                "name": "Bob",
                "age": 32,
                "location": "SF",
                "occupation": "Designer",
            },
            {
                "id": "carol",
                "name": "Carol",
                "age": 25,
                "location": "NYC",
                "occupation": "Engineer",
            },
            {
                "id": "dave",
                "name": "Dave",
                "age": 30,
                "location": "LA",
                "occupation": "Manager",
            },
            {
                "id": "eve",
                "name": "Eve",
                "age": 27,
                "location": "SF",
                "occupation": "Engineer",
            },
        ],
        "edges": [
            ("alice", "bob", {"relationship": "friend", "strength": 0.8, "years": 5}),
            (
                "alice",
                "carol",
                {"relationship": "colleague", "strength": 0.9, "years": 2},
            ),
            ("bob", "eve", {"relationship": "friend", "strength": 0.7, "years": 3}),
            (
                "carol",
                "dave",
                {"relationship": "reports_to", "strength": 0.6, "years": 1},
            ),
            ("dave", "eve", {"relationship": "friend", "strength": 0.5, "years": 1}),
        ],
    }


def create_test_datasets():
    """Create CSV/JSON test datasets in tests/fixtures/data/"""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Karate club dataset
    karate_data = karate_club_data()

    # Save as CSV
    with open(data_dir / "karate_nodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "instructor"])
        writer.writeheader()
        writer.writerows(karate_data["nodes"])

    with open(data_dir / "karate_edges.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        writer.writerows(karate_data["edges"])

    # Save as JSON
    with open(data_dir / "karate_club.json", "w") as f:
        json.dump(karate_data, f, indent=2)

    # Small social network
    social_data = small_social_network_data()

    with open(data_dir / "social_nodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "name", "age", "location", "occupation"]
        )
        writer.writeheader()
        writer.writerows(social_data["nodes"])

    with open(data_dir / "social_edges.csv", "w", newline="") as f:
        fieldnames = ["source", "target", "relationship", "strength", "years"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for source, target, attrs in social_data["edges"]:
            row = {"source": source, "target": target, **attrs}
            writer.writerow(row)

    with open(data_dir / "social_network.json", "w") as f:
        json.dump(social_data, f, indent=2)

    print(f"Test datasets created in {data_dir}")


def load_test_graph(name: str) -> "gr.Graph":
    """
    Load a test graph by name.

    Available graphs:
    - 'empty': Empty graph
    - 'karate': Zachary's Karate Club
    - 'social': Small social network
    - 'path': Simple path (3 nodes)
    - 'cycle': Simple cycle (4 nodes)
    - 'star': Star graph (center + 5 leaves)
    - 'complete': Complete graph (4 nodes)
    """
    if not gr:
        return None

    if name == "empty":
        return gr.Graph()

    elif name == "karate":
        g = gr.Graph()
        data = karate_club_data()
        # Add nodes
        node_map = {}
        for node_data in data["nodes"]:
            node_id = g.add_node(**{k: v for k, v in node_data.items() if k != "id"})
            node_map[node_data["id"]] = node_id
        # Add edges
        for source, target in data["edges"]:
            g.add_edge(node_map[source], node_map[target])
        return g

    elif name == "social":
        g = gr.Graph()
        data = small_social_network_data()
        # Add nodes
        node_map = {}
        for node_data in data["nodes"]:
            node_id = g.add_node(**{k: v for k, v in node_data.items() if k != "id"})
            node_map[node_data["id"]] = node_id
        # Add edges
        for source, target, attrs in data["edges"]:
            g.add_edge(node_map[source], node_map[target], **attrs)
        return g

    elif name == "path":
        from .smart_fixtures import GraphFixtures

        return GraphFixtures.simple_path_graph(3)

    elif name == "cycle":
        from .smart_fixtures import GraphFixtures

        return GraphFixtures.simple_cycle_graph(4)

    elif name == "star":
        from .smart_fixtures import GraphFixtures

        return GraphFixtures.star_graph(5)

    elif name == "complete":
        from .smart_fixtures import GraphFixtures

        return GraphFixtures.complete_graph(4)

    else:
        raise ValueError(f"Unknown test graph: {name}")


if __name__ == "__main__":
    create_test_datasets()
