def test_batch_plan_compiles_neighbor_min():
    from groggy.builder.ir.batch import compile_loop_to_batch_plan

    body = [
        {"type": "graph.neighbor_min", "source": "ranks", "output": "min_rank"},
    ]
    plan = compile_loop_to_batch_plan(body)
    assert plan is not None
    instr_types = [instr["type"] for instr in plan["instructions"]]
    assert "neighbor_aggregate" in instr_types


def test_batch_plan_fuses_madd():
    from groggy.builder.ir.batch import compile_loop_to_batch_plan

    body = [
        {"type": "core.mul", "a": "x", "b": "y", "output": "tmp"},
        {"type": "core.add", "a": "tmp", "b": "z", "output": "out"},
    ]
    plan = compile_loop_to_batch_plan(body)
    assert plan is not None
    instr_types = [instr["type"] for instr in plan["instructions"]]
    assert "fused_madd" in instr_types


def test_batch_plan_rejects_unresolved_slot():
    from groggy.builder.ir.batch import compile_loop_to_batch_plan

    body = [{"type": "core.add", "a": "x", "b": "y"}]
    assert compile_loop_to_batch_plan(body) is None


def test_batch_plan_compiles_compare_where():
    from groggy.builder.ir.batch import compile_loop_to_batch_plan

    body = [
        {"type": "core.compare", "left": "x", "op": "gt", "right": "y", "output": "mask"},
        {
            "type": "core.where",
            "condition": "mask",
            "if_true": "x",
            "if_false": "y",
            "output": "out",
        },
    ]
    plan = compile_loop_to_batch_plan(body)
    assert plan is not None
    instr_types = [instr["type"] for instr in plan["instructions"]]
    assert "compare" in instr_types
    assert "where" in instr_types


def test_batch_plan_compiles_normalize():
    from groggy.builder.ir.batch import compile_loop_to_batch_plan

    body = [
        {"type": "normalize", "input": "values", "output": "normalized", "method": "sum"}
    ]
    plan = compile_loop_to_batch_plan(body)
    assert plan is not None
    instr_types = [instr["type"] for instr in plan["instructions"]]
    assert "normalize" in instr_types


def test_batch_plan_compiles_map_nodes_arith():
    from groggy.builder.ir.batch import compile_loop_to_batch_plan

    body = [
        {
            "type": "map_nodes",
            "fn": "values * 2",
            "inputs": {"values": "values"},
            "output": "out",
        }
    ]
    plan = compile_loop_to_batch_plan(body)
    assert plan is not None
    instr_types = [instr["type"] for instr in plan["instructions"]]
    assert "mul" in instr_types


def test_batch_plan_compiles_map_nodes_neighbor_sum():
    from groggy.builder.ir.batch import compile_loop_to_batch_plan

    body = [
        {
            "type": "map_nodes",
            "fn": "sum(ranks[neighbors(node)])",
            "inputs": {"ranks": "ranks"},
            "output": "out",
        }
    ]
    plan = compile_loop_to_batch_plan(body)
    assert plan is not None
    instr_types = [instr["type"] for instr in plan["instructions"]]
    assert "neighbor_aggregate" in instr_types


def test_batch_plan_compiles_collect_neighbor_values_and_mode():
    from groggy.builder.ir.batch import compile_loop_to_batch_plan

    body = [
        {
            "type": "core.collect_neighbor_values",
            "source": "labels",
            "include_self": True,
            "output": "neighbor_labels",
        },
        {
            "type": "core.mode_list",
            "source": "neighbor_labels",
            "tie_break": "lowest",
            "output": "mode",
        },
    ]
    plan = compile_loop_to_batch_plan(body)
    assert plan is not None
    instr_types = [instr["type"] for instr in plan["instructions"]]
    assert "collect_neighbor_values" in instr_types
    assert "mode_list" in instr_types


def test_batch_plan_compiles_load_edge_attr():
    from groggy.builder.ir.batch import compile_loop_to_batch_plan

    body = [
        {"type": "load_edge_attr", "attr_name": "weight", "default": 1.0, "output": "w"}
    ]
    plan = compile_loop_to_batch_plan(body)
    assert plan is not None
    instr_types = [instr["type"] for instr in plan["instructions"]]
    assert "load_edge_attr" in instr_types
