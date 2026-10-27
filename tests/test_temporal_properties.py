"""Native temporal properties retain exact typed values across public surfaces."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json

import pytest
from groggy import AttrValue, BaseArray, BaseTable, DurableGraph, Graph
from groggy import _groggy

UTC = timezone.utc
VALUES = [
    (datetime(1969, 12, 31, 23, 59, 59, 999999, UTC), "utc_instant"),
    (datetime.min.replace(tzinfo=UTC), "utc_instant"),
    (datetime.max.replace(tzinfo=UTC), "utc_instant"),
    (date(2000, 2, 29), "date"),
    (date.min, "date"),
    (date.max, "date"),
    (timedelta(microseconds=-1), "duration"),
    (timedelta.min, "duration"),
    (timedelta.max, "duration"),
]


@pytest.mark.parametrize("value,kind", VALUES)
def test_temporal_direct_bulk_columns_snapshots_and_storage(value, kind, tmp_path):
    wrapped = AttrValue(value)
    assert wrapped.type_name == kind
    assert wrapped.value == value
    assert getattr(AttrValue, kind)(value) == wrapped
    assert hash(wrapped) == hash(AttrValue(value))
    assert wrapped != AttrValue(str(value))
    graph = Graph()
    node = graph.add_node(payload=value)
    edge = graph.add_edge(node, node, payload=value)
    graph.set_node_attrs({"bulk": {node: value}})
    graph.set_edge_attrs({"bulk": {edge: value}})
    state = graph.commit("temporal", "test")
    snapshot = graph.snapshot_at_commit(state)
    assert snapshot.node_attr(node, "payload").type_name == kind
    assert snapshot.edge_attr(edge, "bulk").value == value
    assert graph.nodes["payload"][0] == value
    assert graph.nodes.table().column("payload").to_list() == [value]
    assert graph.edges.table().column("payload").to_list() == [value]
    assert graph.get_node_attr(node, "bulk") == value
    graph.set_node_attr(node, "payload", None)
    assert snapshot.node_attr(node, "payload").value == value
    changed = graph.commit("unset temporal value", "test")
    delta = next(row for row in graph.diff_commits(state, changed)["attribute_changes"]
                 if row["entity_type"] == "node" and row["name"] == "payload")
    assert delta["old"] == value
    assert delta["new"] is None
    path = tmp_path / "temporal.json"
    graph.save_to_path(path)
    loaded = Graph.load_from_path(path)
    assert loaded.get_edge_attr(edge, "payload") == value
    payload = json.loads(path.read_text())["edges"][0]["attributes"]["payload"]
    assert payload["type"] == kind
    assert payload["value"]["version"] == 1
    durable_path = tmp_path / "temporal.ggj"
    with DurableGraph.create(durable_path) as durable:
        stored = durable.writer().add_node(payload=value)
        retained = durable.current_state_id
    with DurableGraph.open(durable_path) as durable:
        assert durable.get_node_attr(stored, "payload") == value
        assert durable.snapshot_at_commit(retained).node_attr(stored, "payload").type_name == kind


def test_aware_datetime_normalizes_to_utc_without_float_epoch_arithmetic():
    local = datetime(2024, 2, 29, 12, 0, 0, 123456,
                     timezone(timedelta(hours=5, minutes=30, microseconds=1)))
    result = AttrValue(local).value
    assert result == local.astimezone(UTC)
    assert result.tzinfo is UTC
    assert result.microsecond == 123455
    from zoneinfo import ZoneInfo
    zone = ZoneInfo("America/New_York")
    early = AttrValue(datetime(2021, 11, 7, 1, 30, tzinfo=zone, fold=0)).value
    late = AttrValue(datetime(2021, 11, 7, 1, 30, tzinfo=zone, fold=1)).value
    assert early == datetime(2021, 11, 7, 5, 30, tzinfo=UTC)
    assert late == datetime(2021, 11, 7, 6, 30, tzinfo=UTC)
    assert early != late


def test_temporal_subclasses_use_stored_base_fields_without_class_spoofing():
    class Calendar(date):
        @property
        def year(self):
            return 1999

    class Instant(datetime):
        @property
        def microsecond(self):
            return 0

    class Duration(timedelta):
        @property
        def days(self):
            return 100

    assert AttrValue(Calendar(2024, 2, 29)).value == date(2024, 2, 29)
    assert AttrValue(Instant(2024, 2, 29, microsecond=123456, tzinfo=UTC)).value == datetime(
        2024, 2, 29, microsecond=123456, tzinfo=UTC
    )
    assert AttrValue(Duration(days=-2, microseconds=3)).value == timedelta(
        days=-2, microseconds=3
    )

    class PretendDate:
        @property
        def __class__(self):
            return date

    with pytest.raises(TypeError):
        AttrValue(PretendDate())


def test_datetime_subclass_cannot_return_an_unrelated_normalized_value():
    class BrokenInstant(datetime):
        def astimezone(self, tz=None):
            return date(2024, 2, 29)

    with pytest.raises(TypeError):
        AttrValue(BrokenInstant(2024, 2, 29, tzinfo=UTC))


def test_naive_datetimes_and_implicit_text_dates_are_not_accepted_as_instants():
    with pytest.raises(TypeError, match="aware datetime"):
        AttrValue(datetime(2024, 1, 1))
    graph = Graph()
    with pytest.raises((TypeError, ValueError), match="aware datetime"):
        graph.add_node(when=datetime(2024, 1, 1))
    assert len(graph.node_ids) == 0
    for method, value in [(AttrValue.date, "2024-01-01"),
                          (AttrValue.date, datetime.now(UTC)),
                          (AttrValue.utc_instant, date(2024, 1, 1)),
                          (AttrValue.duration, 86400)]:
        with pytest.raises(TypeError):
            method(value)
    assert AttrValue("2024-01-01").type_name == "text"
    with pytest.raises(OverflowError):
        AttrValue(datetime.min.replace(tzinfo=timezone(timedelta(hours=1))))


@pytest.mark.parametrize("earlier,later", [
    (datetime(1969, 12, 31, 23, 59, 59, 999999, UTC), datetime(1970, 1, 1, tzinfo=UTC)),
    (date(2000, 2, 29), date(2000, 3, 1)),
    (timedelta(microseconds=-1), timedelta(0)),
])
def test_temporal_array_and_graph_filters_use_typed_chronological_order(earlier, later):
    array = BaseArray([later, earlier])
    assert array.sort(True).to_list() == [earlier, later]
    assert (array > earlier).to_list() == [True, False]
    assert (array <= earlier).to_list() == [False, True]
    assert (array == earlier).to_list() == [False, True]
    assert array.dtype() in ("UtcInstant", "Date", "Duration")
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        array > 0
    graph = Graph()
    a = graph.add_node(when=earlier)
    b = graph.add_node(when=later)
    predicate = _groggy.NodeFilter.attribute_filter(
        "when", _groggy.AttributeFilter.greater_than(earlier))
    assert list(graph.filter_nodes(predicate).node_ids) == [b]
    assert a != b


def test_temporal_schema_is_distinct_and_versioned_fixture_is_lossless(tmp_path):
    graph = Graph.load_from_path(Path(__file__).parent / "data" / "temporal_properties_v1.json")
    expected = [VALUES[0][0], date(2000, 2, 29), timedelta(microseconds=-1)]
    for i, value in enumerate(expected):
        assert graph.get_node_attr(0, f"p{i}") == value
    graph.set_node_attr(0, "labels", ["Event"])
    schema = {"node_labels": {"Event": {
        "p0": {"type": "utc_instant"}, "p1": {"type": "date"},
        "p2": {"type": "duration"},
    }}}
    assert graph.validate_schema(schema) == []
    schema["node_labels"]["Event"]["p1"]["type"] = "utc_instant"
    assert graph.validate_schema(schema)
    table = BaseTable.from_dict({"when": [expected[1]]})
    assert table.validate_schema({"when": "date"}).column("type_match").to_list() == [True]
    assert table.validate_schema({"when": "text"}).column("type_match").to_list() == [False]
    base = json.loads((Path(__file__).parent / "data" / "temporal_properties_v1.json").read_text())
    path = tmp_path / "bad.json"
    for invalid in [{"version": 1, "year": 1900, "month": 2, "day": 29},
                    {"version": 2, "year": 2000, "month": 2, "day": 29},
                    {"version": 1, "year": 2000, "month": 2, "day": 29, "extra": True}]:
        base["nodes"][0]["attributes"]["p1"]["value"] = invalid
        path.write_text(json.dumps(base))
        with pytest.raises(ValueError, match="Invalid stored property"):
            Graph.load_from_path(path)
