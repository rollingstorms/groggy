"""Fixed child selections refresh shared pending values without root expansion."""
import pytest
from groggy import Graph, Pipeline
from tests.test_pending_core_compute import attach, fixture, program, run, step, topology


def each(body, source="seeds", target="mapped"):
    return step("sample.for_each", source=source, target=target, body=body)


def capture(name="child"):
    return [step("graph.capture_staged", output="pin"), step("graph.emit", source="pin", name=name)]


def increment():
    return [step("core.load_node_attr", attr="counter", default=0, target="values"),
            step("core.init_scalar", value=1, target="one"),
            step("core.add", left="values", right="one", target="next"),
            attach("next", "counter"),
            step("core.load_node_attr", attr="counter", target="again"),
            attach("again", "seen")]


def outputs(profile, suffix="child"):
    return [value for name, value in sorted(profile["outputs"].items()) if name.endswith("." + suffix)]


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("directed", [False, True])
def test_per_node_children_keep_exact_membership_and_induced_loops(outer, directed):
    graph = fixture(directed)
    head = graph.current_state_id
    selected = graph.induced_subgraph([0, 2, 4])
    operations = [step("sample.iterate_nodes", target="seeds"), each([
        step("core.node_degree", target="degree"), attach("degree", "local_degree"), *capture()])]
    if outer:
        _, profile = Pipeline([program(*topology()), program(*operations)])(selected, atomic=True, return_profile=True)
    else:
        _, profile = selected.apply(program(*(topology() + operations)), return_profile=True)
    views = outputs(profile)
    assert [sorted(view.node_ids) for view in views] == [[0], [4], [7]]
    assert [sorted(view.edge_ids) for view in views] == [[], [3], []]
    assert [graph.get_node_attr(n, "local_degree") for n in [0, 4, 7]] == [0, 1, 0]
    assert graph.get_node_attr(5, "local_degree") is None
    assert graph.current_state_id == head + 1


@pytest.mark.parametrize("outer", [False, True])
def test_overlapping_edge_frames_reload_sibling_writes_and_keep_emitted_values_frozen(outer):
    graph = fixture(True)
    head = graph.current_state_id
    selected = graph.induced_subgraph([0, 2, 4])
    operations = [step("sample.iterate_edges", target="seeds"), each(increment() + capture())]
    if outer:
        _, profile = Pipeline([program(*topology()), program(*operations)])(selected, atomic=True, return_profile=True)
    else:
        _, profile = selected.apply(program(*(topology() + operations)), return_profile=True)
    views = outputs(profile)
    assert [sorted(view.node_ids) for view in views] == [[0, 4], [4], [0, 7], [4, 7]]
    assert [sorted(view.edge_ids) for view in views] == [[0], [3], [5], [6]]
    assert views[0].node_attr(0, "counter") == 1
    assert views[0].node_attr(4, "counter") == 1
    assert views[1].node_attr(4, "counter") == 2
    assert views[2].node_attr(0, "counter") == 2
    assert views[2].node_attr(7, "counter") == 1
    assert [graph.get_node_attr(n, "counter") for n in [0, 4, 7]] == [2, 3, 2]
    assert [graph.get_node_attr(n, "seen") for n in [0, 4, 7]] == [2, 3, 2]
    assert graph.current_state_id == head + 1
    graph.set_node_attr(0, "counter", 99)
    assert views[0].node_attr(0, "counter") == 1


def test_nested_foreach_and_loops_keep_grandchildren_inside_the_parent_frame():
    graph = fixture(True)
    _, profile = graph.induced_subgraph([0, 2, 4]).apply(program(*(topology() + [
        step("sample.iterate_edges", target="seeds"),
        each([step("sample.iterate_nodes", target="inner_seeds"),
              each([step("iter.loop", iterations=2, body=increment()), *capture()], source="inner_seeds", target="inner")], target="outer")
    ])), return_profile=True)
    views = outputs(profile)
    assert [sorted(view.node_ids) for view in views] == [[0], [4], [4], [0], [7], [4], [7]]
    assert [graph.get_node_attr(n, "counter") for n in [0, 4, 7]] == [4, 6, 4]
    assert graph.get_node_attr(5, "counter") is None


@pytest.mark.parametrize("directed,expected_nodes,expected_edges", [
    (True, [[0, 7], [0, 4], [4, 7]], [[5], [0, 3], [3, 6]]),
    (False, [[0, 4, 7]] * 3, [[0, 3, 5, 6]] * 3),
])
def test_neighborhood_children_use_revised_direction_and_exact_root_edge_scope(directed, expected_nodes, expected_edges):
    graph = fixture(directed)
    _, profile = graph.induced_subgraph([0, 2, 4]).apply(program(*(topology() + [
        step("sample.iterate_nodes", target="singletons"),
        step("sample.neighbors", source="singletons", hops=1, target="seeds"), each(capture())
    ])), return_profile=True)
    views = outputs(profile)
    assert [sorted(view.node_ids) for view in views] == expected_nodes
    assert [sorted(view.edge_ids) for view in views] == expected_edges


@pytest.mark.parametrize("induced,expected", [(False, []), (True, [3, 6])])
def test_emission_can_unify_node_masks_without_implicitly_inducing_edges(induced, expected):
    graph = fixture(True)
    _, profile = graph.induced_subgraph([0, 2, 4]).apply(program(*(topology() + [
        step("core.load_node_attr", attr="node_id", target="ids"), attach("ids", "marker"),
        step("core.filter_nodes_by_attr", attr="marker", target="mask", predicate={"type": "gt", "value": {"type": "Int", "value": 3}}),
        step("sample.emit_subgraphs", source="mask", mode="unified", induced=induced, target="seeds"), each(capture())
    ])), return_profile=True)
    view = outputs(profile)[0]
    assert sorted(view.node_ids) == [4, 7]
    assert sorted(view.edge_ids) == expected


def test_child_table_results_attach_with_aliases_and_explicit_clocks():
    graph = fixture()
    run(graph, [step("sample.iterate_nodes", target="seeds"), each([
        step("graph.capture_staged", output="child"),
        step("graph.degrees_table", source="child", output="metrics"),
        step("table.attach", source="metrics", columns={"degree": "degree"}, effective_at=40)
    ])], outer=True)
    assert [graph.get_node_attr(n, "degree") for n in [0, 4, 7]] == [0, 1, 0]
    times = graph.publication_metadata(graph.current_state_id)["change_times"]
    assert {row["effective_at"] for row in times if row["kind"] == "node_attr" and row["name"] == "degree"} == {40}


def test_property_only_producers_capture_staged_values_and_child_reloads_see_overlap():
    graph = Graph(directed=True)
    graph.add_nodes(3)
    graph.add_edges([(0, 1), (1, 2)])
    _, profile = graph.apply(program(
        step("core.init_nodes", target="initial", value=10), attach("initial", "counter"),
        step("sample.iterate_edges", target="seeds"), each(increment() + capture())
    ), return_profile=True)
    assert [graph.get_node_attr(n, "counter") for n in [0, 1, 2]] == [11, 12, 11]
    views = outputs(profile)
    assert views[0].node_attr(1, "counter") == 11
    assert views[1].node_attr(1, "counter") == 12


def test_failure_in_last_child_discards_topology_and_prior_sibling_assignments():
    graph = fixture()
    head = graph.current_state_id
    changes = topology()
    changes[0]["params"]["attributes"] = {"score": "bad"}
    with pytest.raises(RuntimeError, match="numeric"):
        graph.induced_subgraph([0, 2, 4]).apply(program(*(changes + [
            step("sample.iterate_nodes", target="seeds"), each(increment() + [
                step("core.load_node_attr", attr="score", default=1, target="score"),
                step("core.normalize_values", source="score", target="normalized")])
        ])))
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "counter") is None
    assert graph.edge_endpoints(0) == (0, 2)
    assert graph.add_node() == 7


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("body", [
    [step("graph.add_nodes", count=1, output="unsupported")],
    [step("drot.smooth_strokes")],
])
def test_uncertified_body_effects_reject_before_prefix_even_with_empty_children(empty, body):
    graph = fixture()
    head = graph.current_state_id
    seed = step("core.sample_nodes", count=0 if empty else 1, target="mask", seed=43)
    with pytest.raises(RuntimeError):
        graph.induced_subgraph([0, 2, 4]).apply(program(*(topology() + [seed,
            step("sample.emit_subgraphs", source="mask", target="seeds", mode="per_item", induced=True), each(body)])))
    assert graph.current_state_id == head
    assert graph.add_node() == 7


def test_property_only_loops_with_child_producers_refresh_after_every_staged_write():
    graph = Graph()
    graph.add_nodes(2)
    graph.apply(program(step("iter.loop", iterations=3, body=[
        step("sample.iterate_nodes", target="seeds"), each(increment())])))
    assert [graph.get_node_attr(n, "counter") for n in [0, 1]] == [3, 3]


def test_seed_selection_from_before_removal_is_rejected_before_any_child_writes():
    graph = fixture(True)
    head = graph.current_state_id
    with pytest.raises(RuntimeError, match="outside active selection"):
        graph.induced_subgraph([0, 2, 4]).apply(program(*(
            [step("sample.iterate_nodes", target="seeds")] + topology() + [each(increment())])))
    assert graph.current_state_id == head
    assert graph.get_node_attr(0, "counter") is None
    assert graph.add_node() == 7


def test_property_only_loop_reloads_staged_values_before_producing_children():
    graph = Graph()
    graph.add_nodes(2)
    graph.apply(program(step("iter.loop", iterations=3, body=increment() + [
        step("sample.iterate_nodes", target="seeds"), each([])])))
    assert [graph.get_node_attr(n, "counter") for n in [0, 1]] == [3, 3]


@pytest.mark.parametrize("placement", ["root", "loop", "foreach", "loop_foreach"])
def test_literal_vector_types_are_consistent_across_nested_programs(placement):
    graph = Graph()
    graph.add_nodes(2)
    operations = [step("core.init_nodes", target="values", value=[1, 2]), attach("values", "vector")]
    if "foreach" in placement:
        operations = [step("sample.iterate_nodes", target="seeds"), each(operations)]
    if "loop" in placement:
        operations = [step("iter.loop", iterations=1, body=operations)]
    graph.apply(program(*operations))
    assert [graph.get_node_attr(n, "vector") for n in [0, 1]] == [[1, 2], [1, 2]]


@pytest.mark.parametrize("field,value", [("params", 3), ("inputs", ["x", 3]), ("outputs", 3)])
def test_invalid_nested_body_fields_fail_before_a_mutating_prefix(field, value):
    graph = fixture()
    head = graph.current_state_id
    invalid = step("core.init_scalar", target="unused", value=1)
    invalid[field] = value
    with pytest.raises(RuntimeError):
        graph.induced_subgraph([0, 2, 4]).apply(program(*(topology() + [
            step("sample.iterate_nodes", target="seeds"), each([invalid])
        ])))
    assert graph.current_state_id == head
    assert graph.add_node() == 7
