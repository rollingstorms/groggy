import pytest

from groggy import Graph


def test_single_node_conversion_failure_does_not_allocate():
    graph = Graph()
    with pytest.raises(ValueError) as error:
        graph.add_node(name="valid", bad=object(), also_bad=object())
    assert "property[1].value" in str(error.value)
    assert "property[2].value" in str(error.value)
    assert len(graph.nodes) == 0
    assert graph.add_node() == 0


def test_bulk_node_conversion_reports_every_bad_row_before_allocating():
    graph = Graph()
    with pytest.raises(ValueError) as error:
        graph.add_nodes([{"name": "valid"}, {"name": object()}, {"name": object()}])
    message = str(error.value)
    assert "1:" in message and "2:" in message
    assert len(graph.nodes) == 0
    assert graph.add_node() == 0


def test_missing_or_non_string_external_id_does_not_allocate():
    graph = Graph()
    with pytest.raises(ValueError) as error:
        graph.add_nodes([{"uid": "a"}, {}, {"uid": 7}], uid_key="uid")
    assert "1.uid" in str(error.value) and "2.uid" in str(error.value)
    assert len(graph.nodes) == 0
    mapping = graph.add_nodes([{"uid": "a"}, {"uid": "b"}], uid_key="uid")
    assert mapping == {"a": 0, "b": 1}


@pytest.mark.parametrize("bulk", [False, True])
def test_reserved_node_property_is_rejected_before_allocating(bulk):
    graph = Graph()
    with pytest.raises(ValueError, match="entity_type"):
        if bulk:
            graph.add_nodes([{"name": "valid"}, {"entity_type": "meta"}])
        else:
            graph.add_node(entity_type="meta")
    assert len(graph.nodes) == 0
    assert graph.add_node() == 0


@pytest.mark.parametrize("bulk", [False, True])
def test_edge_conversion_failure_does_not_allocate(bulk):
    graph = Graph()
    source, target = graph.add_nodes(2)
    with pytest.raises(ValueError) as error:
        if bulk:
            graph.add_edges(
                [(source, target), (source, target, {"bad": object(), "also_bad": object()})]
            )
        else:
            graph.add_edge(source, target, bad=object(), also_bad=object())
    message = str(error.value)
    assert "property[0].value" in message and "property[1].value" in message
    assert len(graph.edges) == 0
    assert graph.add_edge(source, target, weight=2.0) == 0


@pytest.mark.parametrize("entity", ["node", "edge"])
def test_bulk_removal_rejects_all_invalid_and_duplicate_targets(entity):
    graph = Graph()
    source, target = graph.add_nodes(2)
    edge = graph.add_edge(source, target)
    live = source if entity == "node" else edge
    remove = graph.remove_nodes if entity == "node" else graph.remove_edges
    with pytest.raises(ValueError) as error:
        remove([live, 100, live, 101])
    message = str(error.value)
    assert f"1: {entity} 100" in message
    assert f"2: duplicate {entity}" in message
    assert f"3: {entity} 101" in message
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1


@pytest.mark.parametrize("entity", ["node", "edge"])
@pytest.mark.parametrize("column", [False, True])
def test_bulk_setter_reports_conversion_errors_without_writing(entity, column):
    graph = Graph()
    source, target = graph.add_nodes(2)
    edge = graph.add_edge(source, target)
    live = source if entity == "node" else edge
    setter = graph.set_node_attrs if entity == "node" else graph.set_edge_attrs
    getter = graph.get_node_attr if entity == "node" else graph.get_edge_attr
    invalid = (
        {"nodes": [live, live], "values": [object(), object()]}
        if column
        else {live: object(), "bad-id": object()}
    )
    with pytest.raises(ValueError) as error:
        setter({"valid": {live: 1}, "bad": invalid, "other_bad": {live: object()}})
    message = str(error.value)
    assert "bad:" in message and "other_bad:" in message
    assert "[0].value" in message and "[1].value" in message
    assert getter(live, "valid") is None


def test_bulk_node_row_errors_are_reported_together():
    graph = Graph()
    with pytest.raises(ValueError) as error:
        graph.add_nodes([{"name": "valid"}, 7, {"name": object()}, None])
    message = str(error.value)
    assert all(f"{position}:" in message for position in [1, 2, 3])
    assert graph.add_node() == 0


def test_creation_combines_conversion_and_reserved_property_errors():
    graph = Graph()
    with pytest.raises(ValueError) as error:
        graph.add_nodes([{"name": object()}, {"entity_type": object()}])
    message = str(error.value)
    assert "0: property[0].value" in message
    assert "1: property[0].value" in message
    assert "1.entity_type" in message
    assert graph.add_node() == 0


@pytest.mark.parametrize("bulk", [False, True])
def test_edge_creation_combines_conversion_and_endpoint_errors(bulk):
    graph = Graph()
    source = graph.add_node()
    with pytest.raises(ValueError) as error:
        if bulk:
            graph.add_edges([(source, 99, {"bad": object()})])
        else:
            graph.add_edge(source, 99, bad=object())
    message = str(error.value)
    assert "99" in message and "property[0].value" in message
    assert len(graph.edges) == 0
    assert graph.add_edge(source, source) == 0


@pytest.mark.parametrize("entity", ["node", "edge"])
def test_bulk_setter_combines_conversion_inactive_and_duplicate_errors(entity):
    graph = Graph()
    source, target = graph.add_nodes(2)
    edge = graph.add_edge(source, target)
    live = source if entity == "node" else edge
    setter = graph.set_node_attrs if entity == "node" else graph.set_edge_attrs
    getter = graph.get_node_attr if entity == "node" else graph.get_edge_attr
    with pytest.raises(ValueError) as error:
        setter({"score": {"nodes": [live, 99, live], "values": [5, object(), 7]}})
    message = str(error.value)
    assert "[1].value" in message
    assert f"score[1]: {entity} 99" in message
    assert f"score[2]: duplicate {entity}" in message
    assert getter(live, "score") is None


@pytest.mark.parametrize("entity", ["node", "edge"])
def test_malformed_ids_do_not_shift_later_error_positions(entity):
    graph = Graph()
    source, target = graph.add_nodes(2)
    edge = graph.add_edge(source, target)
    live = source if entity == "node" else edge
    setter = graph.set_node_attrs if entity == "node" else graph.set_edge_attrs
    with pytest.raises(ValueError) as error:
        setter(
            {"score": {
                "nodes": ["bad-id", live, 99, live],
                "values": [object(), 1, object(), 3],
            }}
        )
    message = str(error.value)
    assert "[0].id" in message and "[0].value" in message
    assert f"score[2]: {entity} 99" in message
    assert f"score[3]: duplicate {entity}" in message
    assert "first at score[1]" in message
