#!/usr/bin/env python3
"""
Analyze test coverage by comparing comprehensive test results with module tests.

Answers: Which methods from the comprehensive test graph are actually being 
tested in the tests/modules/ directory?
"""

import re
import ast
from pathlib import Path
from typing import Set, Dict, List
from collections import defaultdict

# Add to path
import sys
sys.path.insert(0, 'python')

from comprehensive_library_testing import load_comprehensive_test_graph, iter_method_results


def extract_method_calls_from_test_file(filepath: Path) -> Set[tuple]:
    """Extract method calls from a test file.
    
    Returns:
        Set of (object_name, method_name) tuples found in the test file
    """
    methods_tested = set()
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Look for attribute access patterns like:
        # graph.add_node(...)
        # g.nodes.table()
        # subgraph_array.collect()
        # Also look for inherited test base class patterns
        
        class MethodCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.base_classes = set()
                self.tested_types = set()
                
            def visit_ClassDef(self, node):
                # Check if this class inherits from test base classes
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_name = base.id
                        if 'TestBase' in base_name or 'TestMixin' in base_name:
                            self.base_classes.add(base_name)
                            # Try to infer what types are being tested
                            if 'Array' in base_name:
                                self.tested_types.add('Array')
                            if 'Matrix' in base_name:
                                self.tested_types.add('Matrix')
                            if 'Table' in base_name:
                                self.tested_types.add('Table')
                            if 'Subgraph' in base_name:
                                self.tested_types.add('Subgraph')
                
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Check if it's a method call (func is an Attribute)
                if isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    
                    # Try to infer the object type
                    obj_name = self._infer_object_name(node.func.value)
                    if obj_name:
                        methods_tested.add((obj_name, method_name))
                
                self.generic_visit(node)
            
            def _infer_object_name(self, node):
                """Try to infer what type of object this is"""
                if isinstance(node, ast.Name):
                    # Direct variable name like 'graph', 'nodes', 'edges'
                    var_name = node.id
                    return self._map_var_to_type(var_name)
                elif isinstance(node, ast.Attribute):
                    # Chained attribute like graph.nodes
                    if isinstance(node.value, ast.Name):
                        base = node.value.id
                        attr = node.attr
                        if base in ('graph', 'g', 'empty_graph', 'simple_graph', 'attributed_graph'):
                            if attr == 'nodes':
                                return 'NodesAccessor'
                            elif attr == 'edges':
                                return 'EdgesAccessor'
                        return self._map_var_to_type(f"{base}.{attr}")
                return None
            
            def _map_var_to_type(self, var_name):
                """Map variable names to object types"""
                mappings = {
                    'graph': 'Graph',
                    'g': 'Graph',
                    'empty_graph': 'Graph',
                    'simple_graph': 'Graph',
                    'attributed_graph': 'Graph',
                    'nodes': 'NodesAccessor',
                    'edges': 'EdgesAccessor',
                    'node_table': 'NodesTable',
                    'edge_table': 'EdgesTable',
                    'edges_table': 'EdgesTable',
                    'nodes_table': 'NodesTable',
                    'graph_table': 'GraphTable',
                    'gt': 'GraphTable',
                    'table': 'GraphTable',
                    'subgraph_array': 'SubgraphArray',
                    'subgraphs': 'SubgraphArray',
                    'nodes_array': 'NodesArray',
                    'edges_array': 'EdgesArray',
                    'num_array': 'NumArray',
                    'array': 'BaseArray_from_builder',
                    'base_array': 'BaseArray_from_builder',
                    'table_array': 'TableArray',
                    'matrix': 'GraphMatrix',
                    'graph_matrix': 'GraphMatrix',
                    'laplacian': 'GraphMatrix',
                    'subgraph': 'Subgraph',
                }
                
                for pattern, obj_type in mappings.items():
                    if pattern in var_name.lower():
                        return obj_type
                
                return None
        
        visitor = MethodCallVisitor()
        visitor.visit(tree)
        
        # Also look for docstring mentions of what's being tested
        # This helps identify base classes that test multiple types
        if 'BaseArray' in content or 'base_array' in filepath.name:
            # Extract all method calls that might be array methods
            for line in content.split('\n'):
                if 'array.' in line.lower():
                    # Try to extract method name
                    import re
                    matches = re.findall(r'array\.(\w+)\(', line, re.IGNORECASE)
                    for method in matches:
                        methods_tested.add(('BaseArray_from_builder', method))
        
        if 'Matrix' in filepath.name or 'matrix' in content.lower():
            for line in content.split('\n'):
                if 'matrix.' in line.lower():
                    import re
                    matches = re.findall(r'matrix\.(\w+)\(', line, re.IGNORECASE)
                    for method in matches:
                        methods_tested.add(('GraphMatrix', method))
                        methods_tested.add(('GraphMatrix_from_builder', method))
        
        if 'table' in filepath.name.lower() or 'Table' in content:
            for line in content.split('\n'):
                if 'table.' in line.lower():
                    import re
                    matches = re.findall(r'table\.(\w+)\(', line, re.IGNORECASE)
                    for method in matches:
                        methods_tested.add(('BaseTable', method))
                        methods_tested.add(('BaseTable_from_builder', method))
        
    except Exception as e:
        print(f"Error parsing {filepath.name}: {e}")
    
    return methods_tested


def get_all_module_test_methods() -> Dict[str, Set[str]]:
    """Get all methods tested across all module test files.
    
    Returns:
        Dict mapping object_name -> set of method names tested
    """
    tests_dir = Path('tests/modules')
    all_methods = defaultdict(set)
    
    # Find all test_*.py files
    test_files = list(tests_dir.glob('test_*.py'))
    
    print(f"Scanning {len(test_files)} test files...")
    
    for test_file in test_files:
        print(f"  - {test_file.name}")
        methods = extract_method_calls_from_test_file(test_file)
        for obj_name, method_name in methods:
            all_methods[obj_name].add(method_name)
    
    return all_methods


def get_comprehensive_test_methods() -> Dict[str, Set[str]]:
    """Get all methods from the comprehensive test results.
    
    Returns:
        Dict mapping object_name -> set of method names
    """
    methods_by_object = defaultdict(set)
    
    try:
        # Iterate through comprehensive test results
        for row in iter_method_results():
            obj_name = row['object_name']
            method_name = row['method_name']
            methods_by_object[obj_name].add(method_name)
    except FileNotFoundError:
        print("\n⚠️  No comprehensive test results found!")
        print("Run: python comprehensive_library_testing.py")
        return {}
    
    return methods_by_object


def analyze_coverage():
    """Main analysis function"""
    print("=" * 70)
    print("TEST COVERAGE ANALYSIS")
    print("=" * 70)
    print()
    
    # Get methods from both sources
    print("1. Extracting methods from module test files...")
    module_methods = get_all_module_test_methods()
    
    print("\n2. Loading comprehensive test results...")
    comp_methods = get_comprehensive_test_methods()
    
    if not comp_methods:
        return
    
    print("\n" + "=" * 70)
    print("COVERAGE REPORT")
    print("=" * 70)
    
    # Calculate coverage for each object
    total_comp_methods = 0
    total_tested_methods = 0
    
    coverage_data = []
    
    for obj_name in sorted(comp_methods.keys()):
        comp_set = comp_methods[obj_name]
        module_set = module_methods.get(obj_name, set())
        
        tested = comp_set & module_set
        untested = comp_set - module_set
        
        coverage_pct = (len(tested) / len(comp_set) * 100) if comp_set else 0
        
        total_comp_methods += len(comp_set)
        total_tested_methods += len(tested)
        
        coverage_data.append({
            'object': obj_name,
            'total': len(comp_set),
            'tested': len(tested),
            'untested': len(untested),
            'coverage': coverage_pct,
            'tested_methods': tested,
            'untested_methods': untested
        })
    
    # Sort by coverage (lowest first to highlight gaps)
    coverage_data.sort(key=lambda x: x['coverage'])
    
    print(f"\n{'Object':<25} {'Total':<8} {'Tested':<8} {'Coverage':<10}")
    print("-" * 70)
    
    for data in coverage_data:
        coverage_bar = "█" * int(data['coverage'] / 10) + "░" * (10 - int(data['coverage'] / 10))
        print(f"{data['object']:<25} {data['total']:<8} {data['tested']:<8} {coverage_bar} {data['coverage']:>5.1f}%")
    
    # Overall statistics
    overall_coverage = (total_tested_methods / total_comp_methods * 100) if total_comp_methods else 0
    
    print("-" * 70)
    print(f"{'TOTAL':<25} {total_comp_methods:<8} {total_tested_methods:<8} {overall_coverage:>5.1f}%")
    print()
    
    # Detailed breakdown for objects with low coverage
    print("\n" + "=" * 70)
    print("DETAILED BREAKDOWN - Low Coverage Objects (<50%)")
    print("=" * 70)
    
    for data in coverage_data:
        if data['coverage'] < 50 and data['total'] > 0:
            print(f"\n{data['object']} ({data['coverage']:.1f}% coverage)")
            print(f"  Tested methods ({len(data['tested_methods'])}):")
            for method in sorted(data['tested_methods']):
                print(f"    ✅ {method}")
            print(f"  Untested methods ({len(data['untested_methods'])}):")
            for method in sorted(data['untested_methods']):
                print(f"    ❌ {method}")
    
    # High coverage objects
    print("\n" + "=" * 70)
    print("WELL-TESTED Objects (>80% coverage)")
    print("=" * 70)
    
    for data in coverage_data:
        if data['coverage'] >= 80:
            print(f"\n{data['object']} ({data['coverage']:.1f}% coverage)")
            print(f"  ✅ {len(data['tested_methods'])} methods tested")
            if data['untested_methods']:
                print(f"  Missing:")
                for method in sorted(data['untested_methods']):
                    print(f"    - {method}")
    
    return coverage_data


if __name__ == '__main__':
    analyze_coverage()
