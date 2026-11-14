"""The public version belongs to the loaded binary, and exports must resolve."""
import groggy
import pytest

from groggy import _groggy


def test_runtime_version_matches_loaded_extension():
    assert groggy.__version__ == _groggy.__version__


def test_declared_public_exports_resolve():
    assert all(hasattr(groggy, name) for name in groggy.__all__)


@pytest.mark.parametrize("data, kind", [
    ({"data": []}, "array"),
    ({"data": [1, 2]}, "array"),
    ({"data": [[1, 2]]}, "matrix"),
    ({"columns": [], "dtypes": []}, "table"),
])
def test_public_display_helpers_delegate_to_native_formatter(data, kind):
    assert groggy.detect_display_type(data) == kind
    config = groggy.DisplayConfig(use_color=False)
    assert groggy.format_data_structure(data, config=config) == (
        groggy.TableFormatter.format_data_structure(data, None, config)
    )
