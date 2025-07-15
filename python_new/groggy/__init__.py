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
    # TODO: 1. Discover built-in backends; 2. Discover plugins; 3. Validate; 4. Handle ImportError.
    pass

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
    # TODO: 1. Validate backend; 2. Load backend; 3. Update global state; 4. Handle errors.
    pass

def get_current_backend():
    """
    Returns the name of the currently selected backend implementation.
    
    Reads from global state. Used for diagnostics and debugging.
    Returns:
        str: Name of the current backend.
    """
    # TODO: 1. Read from global state; 2. Return backend name.
    pass
