"""Version restoration keeps floating representation while comparison remains logical."""
import struct

from groggy import AttrValue, DurableGraph, Graph, GraphBatch


def values(negative):
    zero = -0.0 if negative else 0.0
    payload = 0x7FF800000000007B if negative else 0x7FF8000000000001
    nan = struct.unpack("<d", struct.pack("<Q", payload))[0]
    return {"scalar": zero, "vector": [zero, nan],
            "scalar32": AttrValue.float32(zero),
            "compressed32": AttrValue.float_vec([zero] * 30)}


def check_bits(graph, negative):
    zero = -0.0 if negative else 0.0
    payload = 0x7FF800000000007B if negative else 0x7FF8000000000001
    for get in (graph.get_node_attr, graph.get_edge_attr):
        assert struct.pack("<d", get(0, "scalar")) == struct.pack("<d", zero)
        vector = get(0, "vector")
        assert struct.pack("<d", vector[0]) == struct.pack("<d", zero)
        assert struct.unpack("<Q", struct.pack("<d", vector[1]))[0] == payload
        assert struct.pack("<f", get(0, "scalar32")) == struct.pack("<f", zero)
        assert all(struct.pack("<f", item) == struct.pack("<f", zero)
                   for item in get(0, "compressed32"))


def initial(graph):
    batch = GraphBatch(effective_at=10)
    first = batch.add_node(**values(False))
    second = batch.add_node()
    batch.add_edge(first, second, **values(False))
    graph.apply_batch(batch, 0, "positive", "test")
    return graph.current_state_id


def update(graph):
    batch = GraphBatch(effective_at=20)
    for name, value in values(True).items():
        batch.set_node_attr(0, name, value)
        batch.set_edge_attr(0, name, value)
    graph.apply_batch(batch, graph.current_state_id, "negative", "test")
    return graph.current_state_id


def test_repeated_branch_owned_and_archive_restoration_preserves_scalar_vector_bits(tmp_path):
    graph = Graph()
    first = initial(graph)
    graph.create_branch("positive")
    last = update(graph)
    for _ in range(3):
        graph.checkout_branch("positive")
        check_bits(graph, False)
        graph.checkout_branch("main")
        check_bits(graph, True)
    assert graph.current_state_id == last
    check_bits(graph.snapshot_at_commit(last, owned=True).to_graph(), True)
    path = tmp_path / "bits.gga"
    graph.save_archive(path)
    copy = Graph.load_archive(path)
    check_bits(copy, True)
    copy.checkout_branch("positive")
    assert copy.current_state_id == first
    check_bits(copy, False)


def test_durable_switch_checkpoint_reopen_and_backup_preserve_float_bits(tmp_path):
    path = tmp_path / "bits.ggj"
    backup = tmp_path / "backup.ggj"
    with DurableGraph.create(path) as graph:
        first = initial(graph)
        graph.create_branch("positive", first)
        last = update(graph)
        graph.checkout_branch("positive", last)
        check_bits(graph, False)
        graph.checkout_branch("main", first)
        check_bits(graph, True)
        graph.checkpoint_storage()
        graph.backup(backup)
    for path in (path, backup):
        with DurableGraph.open(path) as graph:
            check_bits(graph, True)
            graph.checkout_branch("positive", last)
            check_bits(graph, False)
            graph.checkout_branch("main", first)
            check_bits(graph, True)
