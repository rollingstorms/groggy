# python_new/groggy/__init__.py

def get_available_backends():
    """
    Returns a list of all available backend implementations for the Graph library.
    
    Discovers built-in and user-registered backends. Handles plugin loading and validation.
    Returns:
        List[str]: Names of available backends.
    Raises:
        ImportError: If a backend fails to load.
    """
    # Discover built-in backends
    builtins = ['rust']
    plugins = []
    try:
        import pkg_resources
        for entry in pkg_resources.iter_entry_points('groggy.backends'):
            plugins.append(entry.name)
    except ImportError:
        pass
    return builtins + plugins

def set_backend(backend):
    """
    Sets the backend implementation to use for all new Graph instances.
    
    Validates the backend name, loads the backend module, and updates global state.
    Args:
        backend (str): Name of the backend to use.
    Raises:
        ValueError: If backend is not available.
        ImportError: If backend cannot be loaded.
    """
    global _backend
    available = get_available_backends()
    if backend not in available:
        raise ValueError(f"Backend '{backend}' not available. Choose from: {available}")
    # Attempt to import backend module (simulate for 'rust')
    if backend == 'rust':
        try:
            import groggy._core
        except ImportError as e:
            raise ImportError("Rust backend not installed or failed to import.")
    else:
        try:
            import pkg_resources
            entry = next(e for e in pkg_resources.iter_entry_points('groggy.backends') if e.name == backend)
            entry.load()
        except Exception as e:
            raise ImportError(f"Failed to load backend '{backend}': {e}")
    _backend = backend

def get_current_backend():
    """
    Returns the name of the currently selected backend implementation.
    
    Reads from global state. Used for diagnostics and debugging.
    Returns:
        str: Name of the current backend.
    """
    global _backend
    return _backend if '_backend' in globals() else 'rust'
