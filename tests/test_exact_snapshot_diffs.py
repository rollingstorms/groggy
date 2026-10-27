"""History differences share the exact comparison used by tracked restoration."""
import math
import struct

from groggy import Graph


def test_signed_zero_differences_are_visible_before_and_after_tracked_restoration():
    g = Graph()
    g.add_node(value=0.0)
    plus = g.current_state_id
    g.set_node_attr(0, "value", -0.0)
    minus = g.current_state_id
    diff = g.diff_commits(plus, minus)
    assert len(diff["attribute_changes"]) == 1
    change = diff["attribute_changes"][0]
    assert math.copysign(1, change["old"]) == 1
    assert math.copysign(1, change["new"]) == -1
    new = g.restore_state(plus, mode="tracked")
    assert len(g.diff_commits(minus, new)["attribute_changes"]) == 1
    assert g.diff_commits(plus, new)["attribute_changes"] == []


def test_floating_vector_nan_payload_changes_remain_exact_in_both_diff_directions():
    nan1 = struct.unpack("d", struct.pack("Q", 0x7FF8000000000001))[0]
    nan2 = struct.unpack("d", struct.pack("Q", 0x7FF8000000000002))[0]
    g = Graph()
    g.add_node(vector=[nan1, 0.0])
    old = g.current_state_id
    g.set_node_attr(0, "vector", [nan2, -0.0])
    new = g.current_state_id
    for left, right, a, b in [(old, new, nan1, nan2), (new, old, nan2, nan1)]:
        changes = g.diff_commits(left, right)["attribute_changes"]
        assert len(changes) == 1
        assert struct.pack("d", changes[0]["old"][0]) == struct.pack("d", a)
        assert struct.pack("d", changes[0]["new"][0]) == struct.pack("d", b)
    assert g.diff_commits(new, new)["attribute_changes"] == []
