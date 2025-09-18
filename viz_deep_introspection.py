#!/usr/bin/env python3
"""
Viz Module Deep Introspection Tool

This tool performs deep introspection to verify what visualization functionality
actually exists vs. what we think exists. No assumptions - only verification.
"""

import sys
import inspect
import types
from typing import Any, Dict, List, Tuple

def analyze_object_methods(obj, obj_name: str) -> Dict[str, Any]:
    """Deep analysis of an object's methods and properties."""
    analysis = {
        'object_name': obj_name,
        'object_type': type(obj).__name__,
        'object_class': type(obj),
        'methods': {},
        'properties': {},
        'attributes': {},
        'callables': {},
        'special_methods': {},
        'documentation': getattr(obj, '__doc__', None)
    }
    
    # Get all attributes
    all_attrs = dir(obj)
    
    for attr_name in all_attrs:
        try:
            attr_value = getattr(obj, attr_name)
            attr_type = type(attr_value)
            
            # Categorize the attribute
            if attr_name.startswith('_'):
                # Special/private methods
                analysis['special_methods'][attr_name] = {
                    'type': attr_type.__name__,
                    'callable': callable(attr_value),
                    'doc': getattr(attr_value, '__doc__', None)
                }
            elif callable(attr_value):
                # Callable methods/functions
                analysis['callables'][attr_name] = {
                    'type': attr_type.__name__,
                    'signature': None,
                    'doc': getattr(attr_value, '__doc__', None),
                    'is_method': isinstance(attr_value, types.MethodType),
                    'is_function': isinstance(attr_value, types.FunctionType),
                    'is_builtin': isinstance(attr_value, types.BuiltinMethodType)
                }
                
                # Try to get signature
                try:
                    sig = inspect.signature(attr_value)
                    analysis['callables'][attr_name]['signature'] = str(sig)
                except Exception as e:
                    analysis['callables'][attr_name]['signature_error'] = str(e)
                    
            elif isinstance(attr_value, property):
                # Properties
                analysis['properties'][attr_name] = {
                    'type': 'property',
                    'fget': attr_value.fget is not None,
                    'fset': attr_value.fset is not None,
                    'fdel': attr_value.fdel is not None,
                    'doc': attr_value.__doc__
                }
            else:
                # Regular attributes
                analysis['attributes'][attr_name] = {
                    'type': attr_type.__name__,
                    'value': repr(attr_value) if len(repr(attr_value)) < 100 else f"<{attr_type.__name__}>",
                    'doc': getattr(attr_value, '__doc__', None)
                }
                
        except Exception as e:
            analysis['attributes'][attr_name] = {
                'type': 'ERROR',
                'error': str(e)
            }
    
    return analysis


def verify_viz_accessor_exists():
    """Step 1: Verify the viz accessor actually exists."""
    print("🔍 STEP 1: Verifying viz accessor exists")
    print("=" * 50)
    
    try:
        import groggy as gr
        
        # Create a test graph
        g = gr.Graph()
        node_a = g.add_node(label="Test A")
        node_b = g.add_node(label="Test B") 
        g.add_edge(node_a, node_b, weight=1.0)
        
        print(f"✓ Test graph created: {g.node_count()} nodes, {g.edge_count()} edges")
        
        # Check if viz is in dir()
        graph_attrs = dir(g)
        has_viz_in_dir = 'viz' in graph_attrs
        print(f"✓ 'viz' in dir(g): {has_viz_in_dir}")
        
        # Check if viz is accessible via hasattr
        has_viz_hasattr = hasattr(g, 'viz')
        print(f"✓ hasattr(g, 'viz'): {has_viz_hasattr}")
        
        # Try to access viz
        try:
            viz_attr = getattr(g, 'viz')
            print(f"✓ getattr(g, 'viz'): {type(viz_attr)} - {viz_attr}")
            
            # Check if it's callable
            is_callable = callable(viz_attr)
            print(f"✓ callable(g.viz): {is_callable}")
            
            if is_callable:
                # Try calling it
                try:
                    viz_result = g.viz()
                    print(f"✓ g.viz() call result: {type(viz_result)} - {viz_result}")
                    return g, viz_result
                except Exception as e:
                    print(f"✗ g.viz() call failed: {e}")
                    return g, None
            else:
                print(f"✓ g.viz is not callable, treating as property: {viz_attr}")
                return g, viz_attr
                
        except Exception as e:
            print(f"✗ Cannot access g.viz: {e}")
            return g, None
            
    except Exception as e:
        print(f"✗ Failed to create test graph: {e}")
        return None, None


def deep_analyze_viz_accessor(viz_accessor):
    """Step 2: Deep analysis of the viz accessor."""
    print("\n🔍 STEP 2: Deep analysis of viz accessor")
    print("=" * 50)
    
    if viz_accessor is None:
        print("✗ No viz accessor to analyze")
        return {}
        
    analysis = analyze_object_methods(viz_accessor, "viz_accessor")
    
    print(f"Object type: {analysis['object_type']}")
    print(f"Object class: {analysis['object_class']}")
    
    print(f"\nCallable methods ({len(analysis['callables'])}):")
    for name, info in analysis['callables'].items():
        sig = info.get('signature', 'No signature')
        doc = info.get('doc', 'No documentation')[:100] + "..." if info.get('doc') and len(info.get('doc', '')) > 100 else info.get('doc', 'No documentation')
        print(f"  ✓ {name}{sig}")
        print(f"    Type: {info['type']}, Doc: {doc}")
    
    print(f"\nProperties ({len(analysis['properties'])}):")
    for name, info in analysis['properties'].items():
        print(f"  ✓ {name}: {info}")
    
    print(f"\nAttributes ({len(analysis['attributes'])}):")
    for name, info in analysis['attributes'].items():
        print(f"  ✓ {name}: {info['type']} = {info['value']}")
    
    return analysis


def verify_expected_methods(viz_accessor, analysis):
    """Step 3: Verify all expected methods exist and work."""
    print("\n🔍 STEP 3: Verifying expected methods")
    print("=" * 50)
    
    expected_methods = [
        'interactive',
        'static', 
        'info',
        'supports_graph_view'
    ]
    
    results = {}
    
    for method_name in expected_methods:
        print(f"\nTesting method: {method_name}")
        
        # Check if method exists
        if method_name in analysis.get('callables', {}):
            print(f"  ✓ Method exists in callables")
            method_info = analysis['callables'][method_name]
            print(f"  ✓ Signature: {method_info.get('signature', 'Unknown')}")
            print(f"  ✓ Documentation: {method_info.get('doc', 'None')[:100]}...")
            
            # Try to call the method
            try:
                method = getattr(viz_accessor, method_name)
                print(f"  ✓ Method retrieved: {type(method)}")
                
                # Test different method calls
                if method_name == 'interactive':
                    try:
                        result = method(auto_open=False)
                        print(f"  ✓ interactive() call successful: {type(result)}")
                        if hasattr(result, 'url'):
                            print(f"    URL: {result.url()}")
                        if hasattr(result, 'port'):
                            print(f"    Port: {result.port()}")
                        if hasattr(result, 'stop'):
                            result.stop()
                            print(f"    ✓ Session stopped")
                        results[method_name] = True
                    except Exception as e:
                        print(f"  ✗ interactive() call failed: {e}")
                        results[method_name] = False
                        
                elif method_name == 'static':
                    try:
                        import tempfile
                        import os
                        with tempfile.TemporaryDirectory() as temp_dir:
                            test_file = os.path.join(temp_dir, "test.svg")
                            result = method(test_file, format="svg")
                            print(f"  ✓ static() call successful: {type(result)}")
                            if hasattr(result, 'file_path'):
                                print(f"    File path: {result.file_path}")
                            results[method_name] = True
                    except Exception as e:
                        print(f"  ✗ static() call failed: {e}")
                        results[method_name] = False
                        
                elif method_name == 'info':
                    try:
                        result = method()
                        print(f"  ✓ info() call successful: {type(result)}")
                        print(f"    Result: {result}")
                        results[method_name] = True
                    except Exception as e:
                        print(f"  ✗ info() call failed: {e}")
                        results[method_name] = False
                        
                elif method_name == 'supports_graph_view':
                    try:
                        result = method()
                        print(f"  ✓ supports_graph_view() call successful: {result}")
                        results[method_name] = True
                    except Exception as e:
                        print(f"  ✗ supports_graph_view() call failed: {e}")
                        results[method_name] = False
                        
            except Exception as e:
                print(f"  ✗ Failed to retrieve method: {e}")
                results[method_name] = False
                
        else:
            print(f"  ✗ Method NOT found in callables")
            # Check if it's in other categories
            if method_name in analysis.get('attributes', {}):
                print(f"  ⚠️ Found as attribute: {analysis['attributes'][method_name]}")
            elif method_name in analysis.get('properties', {}):
                print(f"  ⚠️ Found as property: {analysis['properties'][method_name]}")
            else:
                print(f"  ✗ Method completely missing")
            results[method_name] = False
    
    return results


def verify_module_level_functions():
    """Step 4: Verify module-level convenience functions."""
    print("\n🔍 STEP 4: Verifying module-level functions")
    print("=" * 50)
    
    try:
        import groggy as gr
        
        # Check if gr.viz exists
        if hasattr(gr, 'viz'):
            print("✓ gr.viz module exists")
            viz_module = gr.viz
            
            # Analyze the viz module
            analysis = analyze_object_methods(viz_module, "gr.viz")
            
            print(f"Module type: {analysis['object_type']}")
            print(f"Callables in gr.viz: {list(analysis['callables'].keys())}")
            
            # Test specific functions
            expected_functions = ['interactive', 'static']
            
            for func_name in expected_functions:
                if func_name in analysis['callables']:
                    print(f"  ✓ gr.viz.{func_name} exists")
                    
                    # Test the function
                    try:
                        func = getattr(viz_module, func_name)
                        
                        # Create test graph
                        g = gr.Graph()
                        node_a = g.add_node(label="Test")
                        
                        if func_name == 'interactive':
                            result = func(g, auto_open=False)
                            print(f"    ✓ gr.viz.interactive() works: {type(result)}")
                            if hasattr(result, 'stop'):
                                result.stop()
                                
                        elif func_name == 'static':
                            import tempfile
                            import os
                            with tempfile.TemporaryDirectory() as temp_dir:
                                test_file = os.path.join(temp_dir, "module_test.svg")
                                result = func(g, test_file)
                                print(f"    ✓ gr.viz.static() works: {type(result)}")
                                
                    except Exception as e:
                        print(f"    ✗ gr.viz.{func_name}() failed: {e}")
                        
                else:
                    print(f"  ✗ gr.viz.{func_name} NOT found")
                    
        else:
            print("✗ gr.viz module does not exist")
            
    except Exception as e:
        print(f"✗ Module level verification failed: {e}")


def verify_viz_config():
    """Step 5: Verify VizConfig functionality."""
    print("\n🔍 STEP 5: Verifying VizConfig")
    print("=" * 50)
    
    try:
        import groggy as gr
        
        # Check if VizConfig exists
        if hasattr(gr, 'VizConfig'):
            print("✓ gr.VizConfig exists")
            
            # Create default config
            config = gr.VizConfig()
            print(f"✓ Default config created: {type(config)}")
            
            # Analyze VizConfig
            analysis = analyze_object_methods(config, "VizConfig")
            
            print(f"VizConfig attributes: {list(analysis['attributes'].keys())}")
            print(f"VizConfig properties: {list(analysis['properties'].keys())}")
            print(f"VizConfig callables: {list(analysis['callables'].keys())}")
            
            # Test specific attributes
            expected_attrs = ['port', 'layout', 'theme', 'width', 'height']
            for attr in expected_attrs:
                if attr in analysis['attributes'] or attr in analysis['properties']:
                    try:
                        value = getattr(config, attr)
                        print(f"  ✓ config.{attr} = {value}")
                    except Exception as e:
                        print(f"  ✗ config.{attr} failed: {e}")
                else:
                    print(f"  ✗ config.{attr} not found")
            
            # Test preset methods
            preset_methods = ['publication', 'interactive']
            for method in preset_methods:
                if method in analysis['callables']:
                    try:
                        preset_config = getattr(config, method)()
                        print(f"  ✓ config.{method}() works: {type(preset_config)}")
                    except Exception as e:
                        print(f"  ✗ config.{method}() failed: {e}")
                else:
                    print(f"  ⚠️ config.{method}() not found")
            
        else:
            print("✗ gr.VizConfig does not exist")
            
    except Exception as e:
        print(f"✗ VizConfig verification failed: {e}")


def comprehensive_verification_summary(method_results):
    """Step 6: Comprehensive summary and next steps."""
    print("\n🔍 STEP 6: Comprehensive Verification Summary")
    print("=" * 50)
    
    print("\nMethod Verification Results:")
    total_methods = len(method_results)
    passed_methods = sum(1 for result in method_results.values() if result)
    
    for method, result in method_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {method}")
    
    print(f"\nSummary: {passed_methods}/{total_methods} methods verified")
    
    if passed_methods == total_methods:
        print("\n🎉 ALL METHODS VERIFIED - SYSTEM WORKING!")
        recommendation = "PROCEED"
    elif passed_methods >= total_methods * 0.8:
        print(f"\n⚠️ MOSTLY WORKING - {total_methods - passed_methods} methods need attention")
        recommendation = "CONTINUE_WITH_FIXES"
    else:
        print(f"\n❌ SIGNIFICANT ISSUES - {total_methods - passed_methods} methods failing")
        recommendation = "MAJOR_FIXES_NEEDED"
    
    print(f"\nRecommendation: {recommendation}")
    
    return {
        'total_methods': total_methods,
        'passed_methods': passed_methods,
        'success_rate': passed_methods / total_methods if total_methods > 0 else 0,
        'recommendation': recommendation,
        'method_results': method_results
    }


def main():
    """Run comprehensive viz module verification."""
    print("🚀 Viz Module Deep Introspection & Verification")
    print("=" * 60)
    print("This tool performs deep verification of visualization functionality.")
    print("No assumptions - only thorough testing of what actually exists.\n")
    
    # Step 1: Verify viz accessor exists
    graph, viz_accessor = verify_viz_accessor_exists()
    
    if viz_accessor is None:
        print("\n❌ CRITICAL FAILURE: No viz accessor found")
        print("Cannot proceed with further verification.")
        return 1
    
    # Step 2: Deep analyze viz accessor
    analysis = deep_analyze_viz_accessor(viz_accessor)
    
    # Step 3: Verify expected methods
    method_results = verify_expected_methods(viz_accessor, analysis)
    
    # Step 4: Verify module-level functions
    verify_module_level_functions()
    
    # Step 5: Verify VizConfig
    verify_viz_config()
    
    # Step 6: Summary and recommendations
    summary = comprehensive_verification_summary(method_results)
    
    print(f"\n{'='*60}")
    print("🏁 VERIFICATION COMPLETE")
    print('='*60)
    
    if summary['recommendation'] == "PROCEED":
        print("✅ System verification passed - ready for production use!")
        return 0
    elif summary['recommendation'] == "CONTINUE_WITH_FIXES":
        print("⚠️ System mostly working - minor fixes needed")
        return 0
    else:
        print("❌ System needs significant work before production use")
        return 1


if __name__ == "__main__":
    sys.exit(main())