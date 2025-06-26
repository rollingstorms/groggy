"""
GLI - Graph Language Interface
High-performance graph manipulation library with Rust backend
"""

from .graph import Graph
from .store import GraphStore
from .utils import create_random_graph
from ._version import __version__

# Try to import Rust core, fallback to Python implementation
try:
    from . import _core
    RUST_BACKEND_AVAILABLE = True
except ImportError:
    RUST_BACKEND_AVAILABLE = False
    import warnings
    warnings.warn(
        "Rust backend not available, falling back to Python implementation. "
        "Install with 'pip install gli[rust]' for better performance.",
        UserWarning
    )

# Backend management
_current_backend = 'rust' if RUST_BACKEND_AVAILABLE else 'python'

def get_available_backends():
    """Get list of available backends."""
    backends = ['python']
    if RUST_BACKEND_AVAILABLE:
        backends.append('rust')
    return backends

def set_backend(backend):
    """Set the backend to use for new Graph instances."""
    global _current_backend
    if backend not in get_available_backends():
        raise ValueError(f"Backend '{backend}' not available. Available backends: {get_available_backends()}")
    _current_backend = backend

def get_current_backend():
    """Get the currently selected backend."""
    return _current_backend

__all__ = [
    'Graph',
    'GraphStore', 
    'create_random_graph',
    '__version__',
    'RUST_BACKEND_AVAILABLE',
    'get_available_backends',
    'set_backend',
    'get_current_backend'
]
