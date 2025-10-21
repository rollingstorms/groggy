#!/usr/bin/env python3
"""
Comprehensive Groggy Library Testing System

Uses the API meta-graph to systematically test every object, method, and functionality
across the entire library to identify what's working vs. what needs fixing for 0.5.0.
"""

import sys
import os
import json
import time
import traceback
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Iterator
from collections import defaultdict

# Add paths
sys.path.append('python')
sys.path.append('documentation/meta_api_discovery')

# Add groggy root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import groggy as gr
    from api_meta_graph_extractor import APIMetaGraphExtractor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the groggy root directory")
    sys.exit(1)


class ComprehensiveLibraryTester:
    """Comprehensive testing system for the entire Groggy library"""

    def __init__(self):
        self.extractor = APIMetaGraphExtractor()
        self.test_results = {
            'objects': {},
            'methods': {},
            'summary': {},
            'failures': [],
            'successes': []
        }
        self.created_objects = {}
        self.failed_objects = set()

    def create_test_objects(self) -> Dict[str, Any]:
        """Create instances of all testable objects"""
        print("=" * 60)
        print("CREATING TEST OBJECTS")
        print("=" * 60)

        objects = {}

        # === PRIMARY OBJECTS ===
        print("Creating primary objects...")

        # Graph (foundation)
        try:
            g = gr.Graph()
            # Add diverse test data
            for i in range(10):
                g.add_node(i, name=f'node_{i}', age=20+i*3, category='A' if i%2==0 else 'B',
                          value=i*10.5, active=True if i < 7 else False)

            for i in range(9):
                g.add_edge(i, (i+1)%10, weight=1.0+i*0.3, edge_type='connection',
                          strength=0.1*i, directed=i%3==0)

            objects['Graph'] = g
            print("✅ Graph created successfully")
        except Exception as e:
            print(f"❌ Graph creation failed: {e}")
            self.failed_objects.add('Graph')

        # If Graph failed, we can't create most other objects
        if 'Graph' not in objects:
            print("⚠️  Cannot create dependent objects without Graph")
            return objects

        g = objects['Graph']

        # === ACCESSOR OBJECTS ===
        print("\\nCreating accessor objects...")

        try:
            objects['NodesAccessor'] = g.nodes
            print("✅ NodesAccessor created")
        except Exception as e:
            print(f"❌ NodesAccessor failed: {e}")

        try:
            objects['EdgesAccessor'] = g.edges
            print("✅ EdgesAccessor created")
        except Exception as e:
            print(f"❌ EdgesAccessor failed: {e}")

        # === TABLE OBJECTS ===
        print("\\nCreating table objects...")

        table_objects = [
            ('GraphTable', lambda: g.table()),
            ('NodesTable', lambda: g.nodes.table()),
            ('EdgesTable', lambda: g.edges.table()),
        ]

        for name, creator in table_objects:
            try:
                objects[name] = creator()
                print(f"✅ {name} created")
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                self.failed_objects.add(name)

        # === ARRAY OBJECTS ===
        print("\\nCreating array objects...")

        array_objects = [
            ('NodesArray', lambda: g.nodes.array()),
            ('EdgesArray', lambda: g.edges.array()),
            ('NumArray', lambda: g.nodes.ids()),
        ]

        for name, creator in array_objects:
            try:
                objects[name] = creator()
                print(f"✅ {name} created")
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                self.failed_objects.add(name)

        # === SPECIALIZED ARRAYS ===
        print("\\nCreating specialized arrays...")

        # SubgraphArray
        try:
            subgraph_array = g.nodes.group_by('category')
            objects['SubgraphArray'] = subgraph_array
            print("✅ SubgraphArray created")

            # TableArray from SubgraphArray
            try:
                objects['TableArray'] = subgraph_array.table()
                print("✅ TableArray created")
            except Exception as e:
                print(f"❌ TableArray failed: {e}")

            # Individual Subgraph
            try:
                if len(subgraph_array) > 0:
                    objects['Subgraph'] = subgraph_array[0]
                    print("✅ Subgraph created")
            except Exception as e:
                print(f"❌ Subgraph failed: {e}")

        except Exception as e:
            print(f"❌ SubgraphArray creation failed: {e}")
            self.failed_objects.add('SubgraphArray')

        # ComponentsArray
        try:
            objects['ComponentsArray'] = g.connected_components()
            print("✅ ComponentsArray created")
        except Exception as e:
            print(f"❌ ComponentsArray failed: {e}")
            self.failed_objects.add('ComponentsArray')

        # === MATRIX OBJECTS ===
        print("\\nCreating matrix objects...")

        try:
            objects['GraphMatrix'] = g.to_matrix()
            print("✅ GraphMatrix created")
        except Exception as e:
            print(f"❌ GraphMatrix failed: {e}")
            self.failed_objects.add('GraphMatrix')

        # === BUILDER FUNCTIONS ===
        print("\\nTesting builder functions...")

        builders = [
            ('BaseArray_from_builder', lambda: gr.array([1, 2, 'hello', 4.5])),
            ('NumArray_from_builder', lambda: gr.num_array([1.0, 2.0, 3.0, 4.0])),
            ('BaseTable_from_builder', lambda: gr.table({'name': ['Alice', 'Bob'], 'age': [25, 30]})),
            ('GraphMatrix_from_builder', lambda: gr.matrix([[1, 2], [3, 4]])),
        ]

        for name, creator in builders:
            try:
                objects[name] = creator()
                print(f"✅ {name} created")
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                self.failed_objects.add(name)

        # === MODULE OBJECTS ===
        print("\\nChecking module-level objects...")

        # Check for classes available in groggy module
        for attr_name in dir(gr):
            if (not attr_name.startswith('_') and
                not attr_name.islower() and
                attr_name not in objects and
                attr_name not in self.failed_objects):
                try:
                    attr_obj = getattr(gr, attr_name)
                    if hasattr(attr_obj, '__call__') and hasattr(attr_obj, '__name__'):
                        # Try to create instance if it's a class
                        try:
                            # Special handling for BaseTable - create with data
                            if attr_name == 'BaseTable':
                                instance = gr.table({'name': ['test1', 'test2'], 'value': [10, 20]})
                            else:
                                instance = attr_obj()
                            objects[attr_name] = instance
                            print(f"✅ {attr_name} created from module")
                        except:
                            # If instantiation fails, just note it exists
                            print(f"🔍 {attr_name} class found but not instantiable")
                except:
                    pass

        print(f"\\nTotal objects created: {len(objects)}")
        print(f"Failed objects: {len(self.failed_objects)}")

        self.created_objects = objects
        return objects

    def test_method(self, obj: Any, obj_name: str, method_name: str,
                   method_info: Dict = None) -> Tuple[bool, str, Any]:
        """Test a single method with intelligent parameter handling"""

        try:
            if not hasattr(obj, method_name):
                return False, f"Method '{method_name}' not found on {obj_name}", None

            method = getattr(obj, method_name)
            if not callable(method):
                # It's a property, try to access it
                try:
                    result = method
                    return True, f"Property access successful: {type(result).__name__}", result
                except Exception as e:
                    return False, f"Property access failed: {str(e)}", None

            # Try to call method with no parameters first
            try:
                result = method()
                return True, f"No-args call successful: {type(result).__name__}", result
            except TypeError as te:
                # Method needs parameters, try to provide smart defaults
                if method_info and 'parameters' in method_info:
                    params = method_info['parameters']
                    args, kwargs = self.generate_smart_parameters(params, obj_name, method_name)

                    try:
                        if kwargs:
                            result = method(*args, **kwargs)
                        else:
                            result = method(*args)
                        return True, f"With-args call successful: {type(result).__name__}", result
                    except Exception as e2:
                        return False, f"With-args failed: {str(e2)}", None
                else:
                    # Try some common single parameter values
                    common_params = [
                        None, 0, 1, 5, '', 'test', [], {}, True, False,
                        'name', 'age', 'weight', 'value'
                    ]

                    for param in common_params:
                        try:
                            result = method(param)
                            return True, f"Single param ({param}) successful: {type(result).__name__}", result
                        except:
                            continue

                    return False, f"Parameter required but none worked: {str(te)}", None
            except Exception as e:
                return False, f"Method execution failed: {str(e)}", None

        except Exception as e:
            return False, f"Method testing failed: {str(e)}", None

    def generate_smart_parameters(self, params: List[Dict], obj_name: str, method_name: str) -> Tuple[List, Dict]:
        """Generate smart parameters based on method signature and context"""
        args = []
        kwargs = {}

        # Smart parameter generation based on object type and method name
        for param in params:
            param_name = param['name']
            param_type = param.get('type', 'Any')
            is_required = param.get('required', True)

            if not is_required and param.get('default'):
                continue  # Skip optional params with defaults

            # Context-aware parameter generation
            value = self.get_smart_value(param_name, param_type, obj_name, method_name)

            if value is not None:
                if is_required:
                    args.append(value)
                else:
                    kwargs[param_name] = value

        return args, kwargs

    def get_smart_value(self, param_name: str, param_type: str, obj_name: str, method_name: str) -> Any:
        """Get smart parameter value based on context"""

        # === GRAPH-SPECIFIC PARAMETERS ===
        if obj_name == 'Graph':
            # Node ID parameters
            if param_name in ['node', 'source', 'target', 'start', 'node_id']:
                return 0  # Use first node ID

            # Edge ID parameters
            if param_name in ['edge', 'edge_id']:
                return 0  # Use first edge ID

            # Attribute name parameters
            if param_name in ['attr', 'attribute', 'attr_name']:
                if 'edge' in method_name.lower():
                    return 'weight'  # Common edge attribute
                else:
                    return 'name'  # Common node attribute

            # Operation parameters
            if param_name == 'operation':
                return 'sum'  # Common aggregation

            # Message parameters
            if param_name == 'message':
                return 'Test commit message'

            # Author parameters
            if param_name == 'author':
                return 'Test Author'

            # Value parameters (for set operations)
            if param_name == 'value':
                return 'test_value'

            # Multiple nodes/edges parameters
            if param_name in ['nodes', 'edges', 'center_nodes']:
                return [0, 1]  # First couple of IDs

            # Data parameter (for bulk operations)
            if param_name == 'data':
                if 'node' in method_name.lower():
                    return [{'id': 10, 'name': 'new_node'}]
                else:
                    return [{'source': 0, 'target': 1, 'weight': 1.0}]

            # Other graph parameter
            if param_name == 'other':
                # For add_graph, we need another graph
                # This is complex, mark as not implementable by script
                return None

            # Filter parameters
            if param_name in ['filter', 'predicate']:
                if 'edge' in method_name.lower():
                    return "weight > 0.5"  # Filter for edges using edge attributes
                else:
                    return "name == 'node_0'"  # Filter for nodes using node attributes

            # String ID and key parameters
            if param_name == 'string_id':
                return 'node_0'
            if param_name == 'uid_key':
                return 'name'

            # Branch parameters
            if param_name == 'branch':
                return 'main'
            if param_name == 'branch_name':
                return 'test_branch'

            # Aggregation parameters
            if param_name in ['aggregation_attr', 'agg_attr']:
                return 'value'  # Attribute to aggregate

            # Depth/distance parameters
            if param_name in ['depth', 'distance', 'max_depth']:
                return 2

            # Algorithm parameters
            if param_name == 'algorithm':
                return 'dijkstra'

            # Direction parameters
            if param_name == 'direction':
                return 'out'

            # Attribute mapping parameters
            if param_name in ['attrs', 'attributes']:
                return {'new_attr': 'new_value'}

            # Strategy parameters
            if param_name == 'strategy':
                return 'merge'

            # Dictionary parameters for bulk operations
            if param_name == 'attrs_dict':
                if 'edge' in method_name.lower():
                    return {0: {'new_attr': 'edge_value'}}  # Edge attributes dict
                else:
                    return {0: {'new_attr': 'node_value'}}  # Node attributes dict

            # Radius/distance parameters
            if param_name in ['radius', 'max_nodes']:
                return 2

            # Commit/branch parameters
            if param_name == 'commit_id':
                return 0  # Use first commit

            # Path parameters
            if param_name in ['path', 'bundle_path']:
                return '/tmp/test_path'

            # Other parameter
            if param_name == 'other':
                # For add_graph and similar - return None to skip
                return None

        # List parameters (CHECK FIRST - before column substring matching)
        if param_type.startswith('List') or param_name in ['columns', 'attrs']:
            if obj_name in ['NodesTable', 'NodesAccessor']:
                return ['name', 'age']  # Common node attributes
            elif obj_name in ['EdgesTable', 'EdgesAccessor']:
                return ['weight', 'directed']  # Common edge attributes
            else:
                return ['name', 'value']  # Safe defaults

        # Column name parameters (single column)
        if 'column' in param_name.lower() or param_name in ['col', 'key']:
            if obj_name in ['NodesTable', 'NodesAccessor']:
                return 'name'  # Common node attribute
            elif obj_name in ['EdgesTable', 'EdgesAccessor']:
                return 'weight'  # Common edge attribute
            else:
                return 'name'  # Safe default

        # Index/number parameters
        if param_name in ['n', 'index', 'k', 'size', 'limit'] or 'count' in param_name:
            return 3

        # String parameters
        if param_type == 'str' or 'name' in param_name or 'query' in param_name:
            if 'query' in param_name:
                return "name == 'node_1'"
            return 'test'

        # Boolean parameters
        if param_type == 'bool' or param_name in ['ascending', 'directed', 'inplace']:
            return True

        # Dict parameters
        if param_type.startswith('Dict') or 'mapping' in param_name:
            return {'old_name': 'new_name'}

        # Function parameters
        if 'func' in param_name or 'callable' in param_type.lower():
            return lambda x: x  # Identity function

        # Default fallbacks
        if param_type == 'int':
            return 1
        elif param_type == 'float':
            return 1.0
        elif param_type == 'list':
            return []
        elif param_type == 'dict':
            return {}

        return None

    def test_object_comprehensively(self, obj: Any, obj_name: str, methods_info: List[Dict] = None) -> Dict:
        """Test all methods of an object comprehensively"""

        print(f"\\n{'='*40}")
        print(f"TESTING {obj_name}")
        print(f"{'='*40}")

        results = {
            'object_name': obj_name,
            'object_type': type(obj).__name__,
            'total_methods': 0,
            'successful_methods': 0,
            'failed_methods': 0,
            'methods': {},
            'success_rate': 0.0,
            'creation_status': 'success'
        }

        # Get methods from the object
        try:
            method_names = [name for name in dir(obj) if not name.startswith('_')]
        except Exception as e:
            results['creation_status'] = f'failed: {e}'
            return results

        # Create method info lookup
        method_info_dict = {}
        if methods_info:
            for method in methods_info:
                if method['object_type'] == obj_name:
                    method_info_dict[method['method_name']] = method

        results['total_methods'] = len(method_names)

        # Test each method
        for method_name in sorted(method_names):
            method_info = method_info_dict.get(method_name)
            success, message, result = self.test_method(obj, obj_name, method_name, method_info)

            results['methods'][method_name] = {
                'success': success,
                'message': message,
                'result_type': type(result).__name__ if result is not None else None,
                'has_signature_info': method_info is not None
            }

            if success:
                results['successful_methods'] += 1
                print(f"✅ {method_name}: {message}")
            else:
                results['failed_methods'] += 1
                print(f"❌ {method_name}: {message}")

        # Calculate success rate
        if results['total_methods'] > 0:
            results['success_rate'] = (results['successful_methods'] / results['total_methods']) * 100

        print(f"\\n📊 {obj_name} Summary:")
        print(f"   Total methods: {results['total_methods']}")
        print(f"   Successful: {results['successful_methods']}")
        print(f"   Failed: {results['failed_methods']}")
        print(f"   Success rate: {results['success_rate']:.1f}%")

        return results

    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive test of the entire library"""

        print("🚀 COMPREHENSIVE GROGGY LIBRARY TESTING")
        print("=" * 80)
        print(f"Testing for release: 0.5.0")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Extract API meta-graph
        print("\\n📊 EXTRACTING API META-GRAPH...")
        try:
            meta_graph, summary = self.extractor.extract_complete_meta_graph()
            print(f"✅ API extraction successful")
            print(f"   Discovered {summary['meta_graph_stats']['types_discovered']} types")
            print(f"   Analyzed {summary['discovery_data']['discovery_stats']['total_methods']} methods")
        except Exception as e:
            print(f"❌ API extraction failed: {e}")
            return {'error': 'API extraction failed', 'details': str(e)}

        # Step 2: Create test objects
        objects = self.create_test_objects()

        if not objects:
            return {'error': 'No objects could be created for testing'}

        # Step 3: Test each object comprehensively
        print("\\n" + "=" * 80)
        print("COMPREHENSIVE OBJECT TESTING")
        print("=" * 80)

        all_results = {}
        methods_info = summary['discovery_data']['methods']

        for obj_name, obj in objects.items():
            try:
                result = self.test_object_comprehensively(obj, obj_name, methods_info)
                all_results[obj_name] = result

                # Track overall stats
                if result['success_rate'] > 80:
                    self.test_results['successes'].append(obj_name)
                else:
                    self.test_results['failures'].append(obj_name)

            except Exception as e:
                print(f"❌ Testing {obj_name} failed completely: {e}")
                all_results[obj_name] = {
                    'object_name': obj_name,
                    'error': str(e),
                    'success_rate': 0.0
                }
                self.test_results['failures'].append(obj_name)

        # Step 4: Generate comprehensive summary
        summary_stats = self.generate_summary(all_results, summary)

        # Step 5: Save results
        self.save_comprehensive_results(all_results, summary_stats, meta_graph, summary)

        return {
            'test_results': all_results,
            'summary': summary_stats,
            'meta_graph': meta_graph,
            'api_summary': summary
        }

    def generate_summary(self, test_results: Dict, api_summary: Dict) -> Dict:
        """Generate comprehensive testing summary"""

        print("\\n" + "=" * 80)
        print("COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)

        # Calculate overall statistics
        total_objects = len(test_results)
        total_methods = sum(result.get('total_methods', 0) for result in test_results.values())
        total_successful = sum(result.get('successful_methods', 0) for result in test_results.values())
        total_failed = sum(result.get('failed_methods', 0) for result in test_results.values())

        overall_success_rate = (total_successful / total_methods * 100) if total_methods > 0 else 0

        # Object success rates
        object_success_rates = []
        for obj_name, result in test_results.items():
            if 'success_rate' in result:
                object_success_rates.append((obj_name, result['success_rate']))

        object_success_rates.sort(key=lambda x: x[1], reverse=True)

        # Category analysis
        categories = {
            'excellent': [name for name, rate in object_success_rates if rate >= 90],
            'good': [name for name, rate in object_success_rates if 70 <= rate < 90],
            'needs_work': [name for name, rate in object_success_rates if 50 <= rate < 70],
            'critical': [name for name, rate in object_success_rates if rate < 50]
        }

        summary = {
            'overall_stats': {
                'total_objects_tested': total_objects,
                'total_methods_tested': total_methods,
                'total_successful_methods': total_successful,
                'total_failed_methods': total_failed,
                'overall_success_rate': overall_success_rate,
                'objects_created': len(self.created_objects),
                'objects_failed_creation': len(self.failed_objects)
            },
            'object_categories': categories,
            'top_performers': object_success_rates[:5],
            'needs_attention': object_success_rates[-5:],
            'release_readiness': {
                'excellent_objects': len(categories['excellent']),
                'good_objects': len(categories['good']),
                'problematic_objects': len(categories['needs_work']) + len(categories['critical']),
                'ready_for_release': overall_success_rate >= 75
            }
        }

        # Print detailed summary
        print(f"📈 OVERALL STATISTICS:")
        print(f"   Objects tested: {total_objects}")
        print(f"   Methods tested: {total_methods}")
        print(f"   Successful methods: {total_successful}")
        print(f"   Failed methods: {total_failed}")
        print(f"   Overall success rate: {overall_success_rate:.1f}%")

        print(f"\\n🏆 OBJECT CATEGORIES:")
        print(f"   Excellent (≥90%): {len(categories['excellent'])} objects")
        print(f"   Good (70-89%): {len(categories['good'])} objects")
        print(f"   Needs work (50-69%): {len(categories['needs_work'])} objects")
        print(f"   Critical (<50%): {len(categories['critical'])} objects")

        print(f"\\n🎯 RELEASE 0.5.0 READINESS:")
        readiness = "✅ READY" if summary['release_readiness']['ready_for_release'] else "⚠️  NEEDS WORK"
        print(f"   Status: {readiness}")
        print(f"   Recommendation: {'Proceed with release' if overall_success_rate >= 75 else 'Fix critical issues first'}")

        return summary

    def save_comprehensive_results(self, test_results: Dict, summary: Dict,
                                   meta_graph: Any, api_summary: Dict):
        """Save comprehensive test results"""

        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Save test results
        results_file = f'comprehensive_test_results_{timestamp}.json'
        with open(results_file, 'w') as handle:
            json.dump({
                'test_results': test_results,
                'summary': summary,
                'timestamp': timestamp,
                'api_stats': api_summary['meta_graph_stats']
            }, handle, indent=2, default=str)

        print(f"\n💾 RESULTS SAVED:")
        print(f"   Test results: {results_file}")

        # Generate markdown report
        report_file = f'comprehensive_test_report_{timestamp}.md'
        self.generate_markdown_report(test_results, summary, report_file)
        print(f"   Markdown report: {report_file}")

        # Generate CSV exports
        objects_csv, methods_csv = self.generate_csv_exports(test_results, summary, timestamp)
        print(f"   Objects CSV: {objects_csv}")
        print(f"   Methods CSV: {methods_csv}")

        return results_file, report_file

    def generate_markdown_report(self, test_results: Dict, summary: Dict, filename: str):
        """Generate a comprehensive markdown report."""

        with open(filename, 'w') as handle:
            handle.write("# Comprehensive Groggy Library Test Report\n")
            handle.write(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            handle.write("## Executive Summary\n\n")
            overall = summary['overall_stats']
            handle.write(f"- **Overall Success Rate**: {overall['overall_success_rate']:.1f}%\n")
            handle.write(f"- **Objects Tested**: {overall['total_objects_tested']}\n")
            handle.write(f"- **Methods Tested**: {overall['total_methods_tested']}\n")
            readiness_flag = '✅ Ready' if summary['release_readiness']['ready_for_release'] else '⚠️ Needs Work'
            handle.write(f"- **Release Readiness**: {readiness_flag}\n\n")

            handle.write("## Object Performance\n\n")
            handle.write("| Object | Success Rate | Status |\n")
            handle.write("|--------|--------------|--------|\n")

            for obj_name, result in sorted(
                test_results.items(),
                key=lambda item: item[1].get('success_rate', 0),
                reverse=True,
            ):
                rate = result.get('success_rate', 0)
                status = "🟢" if rate >= 90 else "🟡" if rate >= 70 else "🔴"
                handle.write(f"| {obj_name} | {rate:.1f}% | {status} |\n")

            handle.write("\n## Detailed Results\n\n")
            for obj_name, result in test_results.items():
                handle.write(f"### {obj_name} ({result.get('success_rate', 0):.1f}%)\n\n")

                if 'methods' not in result:
                    continue

                successful = [name for name, info in result['methods'].items() if info['success']]
                failed = [name for name, info in result['methods'].items() if not info['success']]

                if successful:
                    handle.write(
                        f"**✅ Working methods ({len(successful)})**: {', '.join(successful)}\n\n"
                    )
                if failed:
                    handle.write(
                        f"**❌ Failed methods ({len(failed)})**: {', '.join(failed)}\n\n"
                    )

    def generate_csv_exports(self, test_results: Dict, summary: Dict, timestamp: str) -> Tuple[str, str]:
        """Generate CSV exports for objects and methods"""

        # Objects CSV
        objects_csv_file = f'comprehensive_test_objects_{timestamp}.csv'
        with open(objects_csv_file, 'w', newline='') as csvfile:
            fieldnames = [
                'object_name', 'object_type', 'success_rate', 'total_methods',
                'successful_methods', 'failed_methods', 'status'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for obj_name, result in sorted(test_results.items(), key=lambda x: x[1].get('success_rate', 0), reverse=True):
                success_rate = result.get('success_rate', 0)
                status = "excellent" if success_rate >= 90 else "good" if success_rate >= 70 else "needs_work"

                writer.writerow({
                    'object_name': obj_name,
                    'object_type': result.get('object_type', 'unknown'),
                    'success_rate': f"{success_rate:.1f}",
                    'total_methods': result.get('total_methods', 0),
                    'successful_methods': result.get('successful_methods', 0),
                    'failed_methods': result.get('failed_methods', 0),
                    'status': status
                })

        # Methods CSV
        methods_csv_file = f'comprehensive_test_methods_{timestamp}.csv'
        with open(methods_csv_file, 'w', newline='') as csvfile:
            fieldnames = [
                'object_name', 'method_name', 'success', 'message', 'result_type',
                'has_signature_info', 'error_category'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for obj_name, result in test_results.items():
                if 'methods' in result:
                    for method_name, method_info in result['methods'].items():
                        # Categorize error types
                        error_category = "success"
                        if not method_info['success']:
                            message = method_info.get('message', '').lower()
                            if 'missing' in message and 'required' in message:
                                error_category = "missing_parameters"
                            elif "can't extract" in message or "convert" in message:
                                error_category = "type_conversion"
                            elif 'not yet implemented' in message or 'needs to be implemented' in message:
                                error_category = "not_implemented"
                            elif 'attribute' in message and 'does not exist' in message:
                                error_category = "missing_method"
                            else:
                                error_category = "other_error"

                        writer.writerow({
                            'object_name': obj_name,
                            'method_name': method_name,
                            'success': method_info['success'],
                            'message': method_info.get('message', ''),
                            'result_type': method_info.get('result_type', ''),
                            'has_signature_info': method_info.get('has_signature_info', False),
                            'error_category': error_category
                        })

        return objects_csv_file, methods_csv_file


def _resolve_result_paths(timestamp: Optional[str] = None) -> Tuple[Path, Path]:
    """Resolve nodes/edges CSV paths for a given comprehensive test run."""

    base = Path.cwd()

    if timestamp is not None:
        nodes_path = base / f"comprehensive_test_objects_{timestamp}.csv"
        edges_path = base / f"comprehensive_test_methods_{timestamp}.csv"
        if not nodes_path.exists() or not edges_path.exists():
            raise FileNotFoundError(
                f"Comprehensive test CSVs with timestamp '{timestamp}' not found."
            )
        return nodes_path, edges_path

    # Discover most recent run by modification time
    candidates = sorted(
        base.glob("comprehensive_test_objects_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "No comprehensive test CSVs found. Run ComprehensiveLibraryTester first."
        )

    latest_nodes = candidates[0]
    suffix = latest_nodes.stem.replace("comprehensive_test_objects_", "")
    edges_path = base / f"comprehensive_test_methods_{suffix}.csv"
    if not edges_path.exists():
        raise FileNotFoundError(
            f"Edge CSV missing for timestamp '{suffix}' (expected {edges_path.name})."
        )

    return latest_nodes, edges_path


def load_comprehensive_test_graph(
    timestamp: Optional[str] = None,
    *,
    to_graph: bool = True,
    rerun_if_missing: bool = False,
) -> "GraphTable":
    """Load the comprehensive test results as a Groggy Graph/GraphTable.

    Args:
        timestamp: Specific timestamp slug (e.g. "20251004_152053"). If ``None``
            the most recent CSV pair in the repository root is used.
        to_graph: Return the materialised ``Graph`` object instead of ``GraphTable``.
        rerun_if_missing: When True, automatically run the comprehensive test if
            no CSVs are available. This defers to ``ComprehensiveLibraryTester``.

    Returns:
        Graph or GraphTable containing one node per tested object and edges for
        each method invocation with the original CSV columns preserved as
        attributes (including error ``message`` and ``error_category``).
    """

    try:
        nodes_path, edges_path = _resolve_result_paths(timestamp)
    except FileNotFoundError:
        if not rerun_if_missing:
            raise
        tester = ComprehensiveLibraryTester()
        tester.run_comprehensive_test()
        nodes_path, edges_path = _resolve_result_paths(timestamp)

    import groggy as gr  # Local import to avoid module cost on CLI tools

    graph_table = gr.from_csv(
        nodes_filepath=str(nodes_path),
        edges_filepath=str(edges_path),
        node_id_column="object_name",
        source_id_column="object_name",
        target_id_column="result_type",
    )

    return graph_table.to_graph() if to_graph else graph_table


def iter_method_results(
    timestamp: Optional[str] = None,
    *,
    only_failed: bool = False,
) -> Iterator[Dict[str, str]]:
    """Yield method result rows from the comprehensive test edge CSV.

    Args:
        timestamp: Optional timestamp slug. When omitted the freshest CSV pair
            is used.
        only_failed: If True, yield only rows where ``success`` is ``False``.

    Yields:
        Dictionaries matching the CSV header with easy access to ``message`` and
        ``error_category`` fields for downstream diagnostics.
    """

    _, edges_path = _resolve_result_paths(timestamp)
    with edges_path.open() as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if only_failed and row.get("success", "").lower() == "true":
                continue
            yield row


def main():
    """Main execution"""
    tester = ComprehensiveLibraryTester()

    try:
        results = tester.run_comprehensive_test()

        if 'error' in results:
            print(f"\\n❌ Testing failed: {results['error']}")
            return False

        print("\\n✅ Comprehensive testing completed successfully!")
        return True

    except Exception as e:
        print(f"\\n❌ Critical error during testing: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
