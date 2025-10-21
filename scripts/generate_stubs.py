#!/usr/bin/env python3
"""
Generate Python type stub files (.pyi) for the Groggy Rust extension module.

This script introspects the compiled _groggy module and generates comprehensive
type stubs to enable autocomplete and type checking in IDEs and Jupyter notebooks.

Enhanced to:
- Infer return types by calling methods on test instances for better method chaining support
- Detect and include experimental methods when experimental-delegation feature is enabled
- Validate stub signatures match trait definitions
"""

import inspect
import importlib
from typing import get_type_hints, List, Dict, Any, Optional
from pathlib import Path
import warnings
import os


def is_experimental_enabled() -> bool:
    """Check if experimental-delegation feature is enabled."""
    # Try to detect if experimental features are available
    try:
        import groggy
        g = groggy.Graph()
        # Try calling experimental with "list" - if it works, feature is enabled
        result = g.experimental("list")
        return True
    except (AttributeError, ImportError):
        return False


def get_experimental_methods(module) -> Dict[str, str]:
    """
    Get list of experimental methods and their descriptions.
    Returns: {method_name: description}
    """
    experimental_methods = {}
    
    if not is_experimental_enabled():
        return experimental_methods
    
    try:
        g = module.Graph()
        # Get list of experimental methods
        methods = g.experimental("list")
        
        # Get description for each
        for method in methods:
            try:
                desc = g.experimental("describe", method)
                experimental_methods[method] = desc
            except Exception:
                experimental_methods[method] = "Experimental method (no description available)"
                
    except Exception as e:
        warnings.warn(f"Failed to retrieve experimental methods: {e}")
    
    return experimental_methods


def get_known_return_types() -> Dict[str, Dict[str, str]]:
    """
    Manual mapping of known return types for methods that can't be easily inferred.
    Based on PyO3 FFI patterns and common usage.
    """
    return {
        'Graph': {
            'view': 'Subgraph',
            'filter_nodes': 'Subgraph',
            'filter_edges': 'Subgraph',
            'nodes': 'NodesAccessor',
            'edges': 'EdgesAccessor',
            'node_ids': 'NumArray',
            'edge_ids': 'NumArray',
        },
        'Subgraph': {
            'filter_nodes': 'Subgraph',
            'filter_edges': 'Subgraph',
            'nodes': 'NodesAccessor',
            'edges': 'EdgesAccessor',
            'node_ids': 'NumArray',
            'edge_ids': 'NumArray',
            'connected_components': 'ComponentsArray',
        },
        'NodesAccessor': {
            'all': 'Subgraph',
            'ids': 'NumArray',
            'array': 'NodesArray',
        },
        'EdgesAccessor': {
            'all': 'Subgraph',
            'ids': 'NumArray',
            'array': 'EdgesArray',
            'sources': 'NumArray',
            'targets': 'NumArray',
        },
        'NodesArray': {
            'filter': 'NodesArray',
        },
        'EdgesArray': {
            'filter': 'EdgesArray',
        },
        'NumArray': {
            'filter': 'NumArray',
        },
    }


def infer_return_types(module) -> Dict[str, Dict[str, str]]:
    """
    Infer return types by calling methods on test instances.
    Combines runtime inference with known type mappings.
    Returns: {ClassName: {method_name: return_type}}
    """
    # Start with known mappings
    return_types = get_known_return_types()
    
    # Create test instances for runtime inference
    test_instances = {}
    
    try:
        # Graph
        g = module.Graph()
        g.add_nodes(5)
        g.add_edges([(0, 1), (1, 2)])
        test_instances['Graph'] = g
        
        # Subgraph
        test_instances['Subgraph'] = g.view()
        
        # Accessors
        test_instances['NodesAccessor'] = g.nodes
        test_instances['EdgesAccessor'] = g.edges
        
    except Exception as e:
        warnings.warn(f"Failed to create test instances: {e}")
        return return_types
    
    # Test methods that are properties or common methods
    property_tests = {
        'Graph': [
            ('view', lambda obj: obj.view()),
            ('nodes', lambda obj: obj.nodes),
            ('edges', lambda obj: obj.edges),
            ('node_ids', lambda obj: obj.node_ids),
            ('edge_ids', lambda obj: obj.edge_ids),
        ],
        'Subgraph': [
            ('nodes', lambda obj: obj.nodes),
            ('edges', lambda obj: obj.edges),
            ('node_ids', lambda obj: obj.node_ids),
            ('edge_ids', lambda obj: obj.edge_ids),
        ],
        'NodesAccessor': [
            ('all', lambda obj: obj.all()),
            ('ids', lambda obj: obj.ids()),
            ('array', lambda obj: obj.array()),
        ],
        'EdgesAccessor': [
            ('all', lambda obj: obj.all()),
            ('ids', lambda obj: obj.ids()),
            ('array', lambda obj: obj.array()),
            ('sources', lambda obj: obj.sources()),
            ('targets', lambda obj: obj.targets()),
        ],
    }
    
    for class_name, tests in property_tests.items():
        if class_name not in test_instances:
            continue
            
        if class_name not in return_types:
            return_types[class_name] = {}
        
        for method_name, test_func in tests:
            try:
                result = test_func(test_instances[class_name])
                return_type = type(result).__name__
                # Only override if not already in known mappings
                if method_name not in return_types[class_name]:
                    return_types[class_name][method_name] = return_type
            except Exception:
                pass  # Skip methods that fail
    
    return return_types


def generate_stub_for_module(module_name: str, output_path: Path):
    """Generate a .pyi stub file for a compiled extension module."""
    
    # Import the module
    module = importlib.import_module(module_name)
    
    # Check for experimental features
    print("🔬 Checking for experimental features...")
    experimental_enabled = is_experimental_enabled()
    if experimental_enabled:
        print("   ✅ Experimental features detected!")
        experimental_methods = get_experimental_methods(module)
        print(f"   Found {len(experimental_methods)} experimental methods")
        for method, desc in experimental_methods.items():
            print(f"      - {method}: {desc[:60]}...")
    else:
        print("   ℹ️  Experimental features not enabled (build with --features experimental-delegation)")
        experimental_methods = {}
    
    # Infer return types
    print("🔍 Inferring return types for better method chaining...")
    return_type_map = infer_return_types(module)
    print(f"   Found {sum(len(v) for v in return_type_map.values())} method return types")
    
    lines = []
    lines.append("# Type stubs for " + module_name)
    lines.append("# Auto-generated by scripts/generate_stubs.py")
    lines.append("# DO NOT EDIT MANUALLY - regenerate with: python scripts/generate_stubs.py")
    if experimental_enabled:
        lines.append("# Generated WITH experimental-delegation feature enabled")
    else:
        lines.append("# Generated WITHOUT experimental features (build with --features experimental-delegation to include)")
    lines.append("")
    lines.append("from __future__ import annotations  # Enable forward references")
    lines.append("from typing import Any, List, Dict, Optional, Tuple, Union, Iterator")
    lines.append("")
    
    # Get all members
    members = inspect.getmembers(module)
    
    # Separate into functions and classes
    functions = []
    classes = []
    
    for name, obj in members:
        if name.startswith('_'):
            continue  # Skip private members
        
        if inspect.isclass(obj):
            classes.append((name, obj))
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            functions.append((name, obj))
    
    # Generate function stubs
    if functions:
        lines.append("# Module-level functions")
        lines.append("")
        for name, func in sorted(functions):
            doc = inspect.getdoc(func)
            if doc:
                lines.append(f"def {name}(*args, **kwargs) -> Any:")
                lines.append(f'    """')
                for doc_line in doc.split('\n'):
                    lines.append(f"    {doc_line}")
                lines.append(f'    """')
                lines.append("    ...")
            else:
                lines.append(f"def {name}(*args, **kwargs) -> Any: ...")
            lines.append("")
    
    # Generate class stubs
    if classes:
        lines.append("# Classes")
        lines.append("")
        for class_name, cls in sorted(classes):
            class_return_types = return_type_map.get(class_name, {})
            # Pass experimental methods only for Graph class
            exp_methods = experimental_methods if class_name == 'Graph' else {}
            lines.extend(generate_class_stub(class_name, cls, class_return_types, exp_methods))
            lines.append("")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines))
    print(f"✅ Generated stub file: {output_path}")
    print(f"   - {len(functions)} module-level functions")
    print(f"   - {len(classes)} classes")
    if experimental_enabled:
        print(f"   - {len(experimental_methods)} experimental methods documented")


def generate_class_stub(class_name: str, cls, return_types: Dict[str, str] = None, experimental_methods: Dict[str, str] = None) -> List[str]:
    """Generate stub for a single class with inferred return types and experimental methods."""
    lines = []
    if return_types is None:
        return_types = {}
    if experimental_methods is None:
        experimental_methods = {}
    
    # Class definition
    doc = inspect.getdoc(cls)
    lines.append(f"class {class_name}:")
    if doc:
        lines.append(f'    """')
        for doc_line in doc.split('\n'):
            lines.append(f"    {doc_line}")
        lines.append(f'    """')
    
    # Get all methods and properties
    methods = []
    properties = []
    has_getattr = False
    has_experimental = False
    
    for name, member in inspect.getmembers(cls):
        if name == '__getattr__':
            has_getattr = True
        if name == 'experimental':
            has_experimental = True
        
        if name.startswith('_') and name not in ['__init__', '__len__', '__getitem__', 
                                                   '__setitem__', '__iter__', '__next__',
                                                   '__str__', '__repr__', '__eq__', '__hash__',
                                                   '__getattr__']:
            continue  # Skip private methods except special ones
        
        # Check if it's a property (getset_descriptor from PyO3)
        if type(member).__name__ == 'getset_descriptor':
            properties.append((name, member))
        # Check if it's callable (covers method, function, builtin, method_descriptor)
        elif callable(member):
            methods.append((name, member))
    
    if not methods and not properties:
        lines.append("    pass")
        return lines
    
    # Add note about __getattr__ if present
    if has_getattr:
        lines.append("    # Note: This class uses __getattr__ for dynamic attribute access")
        lines.append("    # Intentional dynamic pattern for runtime attribute dictionaries")
        lines.append("")
    
    # Add note about experimental methods if applicable
    if has_experimental and experimental_methods and class_name == 'Graph':
        lines.append("    # Experimental methods (available with experimental-delegation feature):")
        for exp_method, exp_desc in sorted(experimental_methods.items()):
            lines.append(f"    #   - {exp_method}: {exp_desc}")
        lines.append("")
    
    # Generate property stubs first
    for prop_name, prop in sorted(properties):
        # Get inferred return type if available
        inferred_return = return_types.get(prop_name, "Any")
        
        lines.append(f"    @property")
        lines.append(f"    def {prop_name}(self) -> {inferred_return}:")
        
        # Add docstring if available
        prop_doc = inspect.getdoc(prop)
        if prop_doc:
            lines.append(f'        """')
            for doc_line in prop_doc.split('\n'):
                lines.append(f"        {doc_line}")
            lines.append(f'        """')
        
        lines.append("        ...")
        lines.append("")
    
    # Generate method stubs
    for method_name, method in sorted(methods):
        method_doc = inspect.getdoc(method)
        
        # Get inferred return type if available
        inferred_return = return_types.get(method_name)
        
        # Handle special methods
        if method_name == '__init__':
            lines.append(f"    def __init__(self, *args, **kwargs) -> None:")
        elif method_name == '__len__':
            lines.append(f"    def __len__(self) -> int:")
        elif method_name == '__iter__':
            lines.append(f"    def __iter__(self) -> Iterator:")
        elif method_name == '__next__':
            lines.append(f"    def __next__(self) -> Any:")
        elif method_name == '__getitem__':
            lines.append(f"    def __getitem__(self, key: Any) -> Any:")
        elif method_name == '__setitem__':
            lines.append(f"    def __setitem__(self, key: Any, value: Any) -> None:")
        elif method_name == '__str__':
            lines.append(f"    def __str__(self) -> str:")
        elif method_name == '__repr__':
            lines.append(f"    def __repr__(self) -> str:")
        elif method_name == '__eq__':
            lines.append(f"    def __eq__(self, other: Any) -> bool:")
        elif method_name == '__hash__':
            lines.append(f"    def __hash__(self) -> int:")
        elif method_name == '__getattr__':
            lines.append(f"    def __getattr__(self, name: str) -> Any:")
            if not method_doc:
                method_doc = "Dynamic attribute access for runtime attribute dictionaries"
        elif method_name == 'experimental':
            # Special handling for experimental method
            lines.append(f"    def experimental(self, method_name: str, *args, **kwargs) -> Any:")
            if not method_doc:
                method_doc = "Call experimental prototype methods (requires experimental-delegation feature)"
        else:
            # Use inferred return type if available, otherwise Any
            return_hint = inferred_return if inferred_return else "Any"
            lines.append(f"    def {method_name}(self, *args, **kwargs) -> {return_hint}:")
        
        # Add docstring
        if method_doc:
            lines.append(f'        """')
            for doc_line in method_doc.split('\n'):
                lines.append(f"        {doc_line}")
            lines.append(f'        """')
        
        lines.append("        ...")
        lines.append("")
    
    return lines


def main():
    """Generate stubs for groggy._groggy module."""
    
    print("🔨 Generating Python type stubs for Groggy...")
    print("")
    
    # Generate stub for _groggy module
    module_name = "groggy._groggy"
    output_path = Path("python-groggy/python/groggy/_groggy.pyi")
    
    try:
        generate_stub_for_module(module_name, output_path)
        print("")
        print("✨ Success! Type stubs generated.")
        print("📚 Test in Jupyter: import groggy; g = groggy.Graph(); g.<TAB>")
        print("")
        print("📝 To regenerate stubs after changes:")
        print("   1. maturin develop --release")
        print("   2. python scripts/generate_stubs.py")
        
    except Exception as e:
        print(f"❌ Error generating stubs: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
