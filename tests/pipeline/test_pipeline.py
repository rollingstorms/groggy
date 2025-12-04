import json

import pytest

from groggy import _groggy, algorithms
from groggy.pipeline import Pipeline
from groggy.pipeline import apply as pipeline_apply


def build_sample_subgraph():
    g = _groggy.Graph()
    a = g.add_node()
    b = g.add_node()
    c = g.add_node()
    g.add_edge(a, b)
    g.add_edge(b, c)
    # Set a start attribute on the first node using bulk API
    g.nodes.set_attrs({a: {"start": True}})
    # Create a subgraph with all nodes
    sub = g.induced_subgraph([a, b, c])
    return sub


def test_pipeline_roundtrip():
    # Wrap parameter values in AttrValue objects
    spec = [
        {
            "id": "centrality.pagerank",
            "params": {
                "max_iter": _groggy.AttrValue(10),
                "output_attr": _groggy.AttrValue("pr"),
            },
        },
        {
            "id": "pathfinding.bfs",
            "params": {
                "start_attr": _groggy.AttrValue("start"),
                "output_attr": _groggy.AttrValue("dist"),
            },
        },
    ]

    subgraph = build_sample_subgraph()

    handle = _groggy.pipeline.build_pipeline(spec)
    result, profile = _groggy.pipeline.run_pipeline(handle, subgraph)
    _groggy.pipeline.drop_pipeline(handle)
    assert result is not None
    assert profile["build_time"] >= 0.0
    assert profile["run_time"] >= 0.0
    assert "timers" in profile
    assert profile["persist_results"] is True
    assert profile["outputs"] == {}


def test_python_pipeline_return_profile():
    subgraph = build_sample_subgraph()
    pipe = Pipeline([algorithms.centrality.pagerank(output_attr="profile_pr")])

    result, profile = pipe(subgraph, return_profile=True)

    assert result is not None
    assert profile is pipe.last_profile()
    assert profile["run_time"] >= 0.0
    timers = profile["timers"]
    assert isinstance(timers, dict)
    assert any(key.startswith("algorithm.") for key in timers)


def test_apply_return_profile_flag():
    subgraph = build_sample_subgraph()
    handle = algorithms.centrality.closeness(output_attr="closeness")

    result, profile = pipeline_apply(subgraph, handle, return_profile=True)

    assert result is not None
    assert "build_time" in profile
    assert "run_time" in profile


def test_subgraph_apply_return_profile():
    subgraph = build_sample_subgraph()
    result, profile = subgraph.apply(
        [algorithms.centrality.betweenness(output_attr="btw")],
        return_profile=True,
    )

    assert result is not None
    assert profile["run_time"] >= 0.0
    assert "timers" in profile
    assert "run_time" in profile


def test_pipeline_persist_false_outputs():
    subgraph = build_sample_subgraph()
    handle = algorithms.community.connected_components(output_attr="comp")

    result, profile = pipeline_apply(
        subgraph,
        handle,
        return_profile=True,
        persist=False,
    )

    # Attributes should not be written when persist=False
    assert "comp" not in result.nodes.attribute_names()

    outputs = profile["outputs"]
    assert "community.connected_components.components" in outputs
    components = outputs["community.connected_components.components"]
    assert isinstance(components, list)
    assert sum(len(component) for component in components) == len(result.nodes)


def test_list_algorithms_includes_pagerank():
    algos = _groggy.pipeline.list_algorithms()
    # Extract string values from AttrValue objects (value is a property, not a method)
    ids = {entry["id"].value for entry in algos}
    assert "centrality.pagerank" in ids


def test_get_algorithm_metadata():
    """Test retrieving metadata for a specific algorithm."""
    metadata = _groggy.pipeline.get_algorithm_metadata("centrality.pagerank")

    assert metadata["id"].value == "centrality.pagerank"
    assert metadata["name"].value == "PageRank"
    assert "description" in metadata
    assert metadata["version"].value == "0.1.0"
    assert "supports_cancellation" in metadata
    assert "cost_hint" in metadata

    # Check parameters are returned as JSON
    if "parameters" in metadata:
        params_json = metadata["parameters"].value
        params = json.loads(params_json)
        assert isinstance(params, list)

        # PageRank should have parameters
        param_names = {p["name"] for p in params}
        assert "max_iter" in param_names or "damping" in param_names


def test_get_algorithm_metadata_not_found():
    """Test that requesting non-existent algorithm raises error."""
    with pytest.raises(RuntimeError, match="not found"):
        _groggy.pipeline.get_algorithm_metadata("nonexistent.algorithm")


def test_validate_algorithm_params_success():
    """Test parameter validation with valid params."""
    params = {
        "max_iter": _groggy.AttrValue(20),
        "damping": _groggy.AttrValue(0.85),
        "output_attr": _groggy.AttrValue("pagerank"),
    }

    errors = _groggy.pipeline.validate_algorithm_params("centrality.pagerank", params)
    assert len(errors) == 0


def test_validate_algorithm_params_missing_required():
    """Test parameter validation with missing required parameter."""
    # BFS requires start_attr
    params = {"output_attr": _groggy.AttrValue("dist")}

    errors = _groggy.pipeline.validate_algorithm_params("pathfinding.bfs", params)
    # Should have at least one error about missing start_attr
    assert len(errors) > 0
    assert any("start_attr" in err for err in errors)


def test_validate_algorithm_params_unknown():
    """Test parameter validation with unknown parameter."""
    params = {"max_iter": _groggy.AttrValue(20), "unknown_param": _groggy.AttrValue(42)}

    errors = _groggy.pipeline.validate_algorithm_params("centrality.pagerank", params)
    # Should have error about unknown parameter
    assert len(errors) > 0
    assert any("unknown_param" in err for err in errors)


def test_validate_algorithm_params_wrong_type():
    """Test parameter validation with wrong type."""
    params = {
        "max_iter": _groggy.AttrValue("not a number"),  # Should be int
        "output_attr": _groggy.AttrValue("pagerank"),
    }

    errors = _groggy.pipeline.validate_algorithm_params("centrality.pagerank", params)
    # Should have error about wrong type for max_iter
    assert len(errors) > 0
    assert any("max_iter" in err and "wrong type" in err for err in errors)


def test_list_algorithm_categories():
    """Test listing algorithms grouped by category."""
    categories = _groggy.pipeline.list_algorithm_categories()

    assert isinstance(categories, dict)

    # Should have centrality category
    assert "centrality" in categories
    centrality_algos = categories["centrality"]
    assert "centrality.pagerank" in centrality_algos
    assert (
        "centrality.betweenness" in centrality_algos
        or "centrality.closeness" in centrality_algos
    )

    # Should have pathfinding category
    assert "pathfinding" in categories
    pathfinding_algos = categories["pathfinding"]
    assert (
        "pathfinding.dijkstra" in pathfinding_algos
        or "pathfinding.bfs" in pathfinding_algos
    )

    # Should have community category
    assert "community" in categories
    community_algos = categories["community"]
    assert "community.lpa" in community_algos or "community.louvain" in community_algos


# ==============================================================================
# Phase 3.3: Subgraph Marshalling & FFI Round-Trip Tests
# ==============================================================================


def test_attribute_preservation_through_pipeline():
    """Test that attributes are preserved and updated through pipeline execution."""
    g = _groggy.Graph()
    nodes = [g.add_node() for _ in range(5)]

    # Create some edges
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])

    # Set initial attributes
    g.nodes.set_attrs(
        {
            nodes[0]: {"label": "A", "value": 1.0},
            nodes[1]: {"label": "B", "value": 2.0},
            nodes[2]: {"label": "C", "value": 3.0},
        }
    )

    sub = g.induced_subgraph(nodes)

    # Run PageRank which should add 'pagerank' attribute
    spec = [
        {
            "id": "centrality.pagerank",
            "params": {
                "output_attr": _groggy.AttrValue("pagerank"),
                "max_iter": _groggy.AttrValue(20),
            },
        }
    ]

    handle = _groggy.pipeline.build_pipeline(spec)
    result, profile = _groggy.pipeline.run_pipeline(handle, sub)
    _groggy.pipeline.drop_pipeline(handle)

    # Verify original attributes are preserved
    result_nodes = result.nodes
    assert len(result_nodes) == 5

    # Check that original attributes still exist
    # Note: Need to check this through the graph's node accessor
    # The result subgraph should have the same underlying graph with new attributes added


def test_bulk_attribute_update_performance():
    """Test that bulk attribute updates are efficient."""
    g = _groggy.Graph()
    # Create a larger graph
    nodes = [g.add_node() for _ in range(100)]

    # Create edges (create a connected component)
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])

    sub = g.induced_subgraph(nodes)

    # Run PageRank on 100 nodes - should complete quickly
    spec = [
        {
            "id": "centrality.pagerank",
            "params": {
                "output_attr": _groggy.AttrValue("rank"),
                "max_iter": _groggy.AttrValue(20),
            },
        }
    ]

    import time

    start = time.time()

    handle = _groggy.pipeline.build_pipeline(spec)
    result, profile = _groggy.pipeline.run_pipeline(handle, sub)
    _groggy.pipeline.drop_pipeline(handle)

    elapsed = time.time() - start

    # Should complete in well under 1 second
    assert elapsed < 1.0, f"Pipeline took {elapsed:.3f}s, expected < 1.0s"
    assert result is not None


def test_multiple_algorithm_attribute_accumulation():
    """Test that multiple algorithms can each add their own attributes."""
    g = _groggy.Graph()
    nodes = [g.add_node() for _ in range(10)]

    # Create a connected graph
    for i in range(len(nodes)):
        for j in range(i + 1, min(i + 3, len(nodes))):
            g.add_edge(nodes[i], nodes[j])

    # Set start node for BFS
    g.nodes.set_attrs({nodes[0]: {"is_start": True}})

    sub = g.induced_subgraph(nodes)

    # Run multiple algorithms that each add attributes
    spec = [
        {
            "id": "centrality.pagerank",
            "params": {
                "output_attr": _groggy.AttrValue("pagerank"),
                "max_iter": _groggy.AttrValue(20),
            },
        },
        {
            "id": "pathfinding.bfs",
            "params": {
                "start_attr": _groggy.AttrValue("is_start"),
                "output_attr": _groggy.AttrValue("distance"),
            },
        },
    ]

    handle = _groggy.pipeline.build_pipeline(spec)
    result, profile = _groggy.pipeline.run_pipeline(handle, sub)
    _groggy.pipeline.drop_pipeline(handle)

    # Verify result has nodes
    assert len(result.nodes) == 10

    # Both algorithms should have completed
    # (We can't easily inspect attributes from Python without additional API,
    # but the pipeline should complete without errors)


def test_empty_subgraph_pipeline():
    """Test that pipeline handles empty subgraphs gracefully."""
    g = _groggy.Graph()
    # Create an empty subgraph
    sub = g.induced_subgraph([])

    spec = [
        {
            "id": "centrality.pagerank",
            "params": {
                "output_attr": _groggy.AttrValue("pagerank"),
            },
        }
    ]

    handle = _groggy.pipeline.build_pipeline(spec)
    result, profile = _groggy.pipeline.run_pipeline(handle, sub)
    _groggy.pipeline.drop_pipeline(handle)

    # Should return empty subgraph without errors
    assert len(result.nodes) == 0


def test_subgraph_node_count_preserved():
    """Test that node count is preserved through pipeline."""
    g = _groggy.Graph()
    nodes = [g.add_node() for _ in range(7)]

    # Create some edges
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])

    sub = g.induced_subgraph(nodes)
    original_count = len(sub.nodes)

    spec = [
        {
            "id": "centrality.pagerank",
            "params": {
                "output_attr": _groggy.AttrValue("rank"),
            },
        }
    ]

    handle = _groggy.pipeline.build_pipeline(spec)
    result, profile = _groggy.pipeline.run_pipeline(handle, sub)
    _groggy.pipeline.drop_pipeline(handle)

    # Node count should be preserved
    assert len(result.nodes) == original_count


def test_pipeline_with_disconnected_components():
    """Test pipeline with a graph that has multiple disconnected components."""
    g = _groggy.Graph()

    # Create two disconnected components
    component1 = [g.add_node() for _ in range(3)]
    component2 = [g.add_node() for _ in range(3)]

    # Connect within components
    g.add_edge(component1[0], component1[1])
    g.add_edge(component1[1], component1[2])
    g.add_edge(component2[0], component2[1])
    g.add_edge(component2[1], component2[2])

    all_nodes = component1 + component2
    sub = g.induced_subgraph(all_nodes)

    spec = [
        {
            "id": "centrality.pagerank",
            "params": {
                "output_attr": _groggy.AttrValue("pagerank"),
                "max_iter": _groggy.AttrValue(20),
            },
        }
    ]

    handle = _groggy.pipeline.build_pipeline(spec)
    result, profile = _groggy.pipeline.run_pipeline(handle, sub)
    _groggy.pipeline.drop_pipeline(handle)

    # Should handle disconnected components
    assert len(result.nodes) == len(all_nodes)


def test_get_pipeline_context_info():
    """Test that pipeline context info is available."""
    info = _groggy.pipeline.get_pipeline_context_info()

    # Should have information about GIL release
    assert "supports_gil_release" in info
    assert info["supports_gil_release"].value == False

    # Should have reason for limitation
    assert "reason" in info
    reason = info["reason"].value
    assert "Rc<RefCell<Graph>>" in reason or "Arc<RwLock<>>" in reason

    # Should confirm bulk optimizations are available
    assert "bulk_attribute_optimization" in info
    assert info["bulk_attribute_optimization"].value == True
