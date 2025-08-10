#!/usr/bin/env python3
"""
Architecture Analysis Script for Groggy Graph Library

This script analyzes the Rust pseudocode to:
1. Extract function definitions and calls
2. Build a dependency graph
3. Identify missing implementations and loose ends
4. Validate architectural completeness

Usage: python analyze_architecture.py
"""

import os
import re
from collections import defaultdict, deque
from pathlib import Path
import json

class FunctionCall:
    def __init__(self, name, file_path, line_num, context=""):
        self.name = name
        self.file_path = file_path
        self.line_num = line_num
        self.context = context

class FunctionDef:
    def __init__(self, name, file_path, line_num, impl_status="implemented", visibility="private"):
        self.name = name
        self.file_path = file_path
        self.line_num = line_num
        self.impl_status = impl_status
        self.visibility = visibility

class ArchitectureAnalyzer:
    def __init__(self, src_dir="src"):
        self.src_dir = Path(src_dir)
        self.function_defs = {}  # name -> FunctionDef
        self.function_calls = defaultdict(list)  # name -> [FunctionCall]
        self.dependency_graph = defaultdict(set)  # caller -> {callees}
        self.reverse_deps = defaultdict(set)  # callee -> {callers}
        
        # Pattern matching for Rust function definitions and calls
        self.fn_def_pattern = re.compile(
            r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        )
        self.todo_pattern = re.compile(r'todo!\(|TODO:|FIXME:|HACK:|NOTE:')
        self.impl_pattern = re.compile(r'impl\s+(?:\w+\s+for\s+)?(\w+)')
        self.trait_def_pattern = re.compile(r'trait\s+(\w+)')
        
        # Common function call patterns in our codebase
        self.call_patterns = [
            re.compile(r'\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\('),  # method calls
            re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\('),     # direct calls
            re.compile(r'::([a-zA-Z_][a-zA-Z0-9_]*)\s*\('),   # static calls
        ]

    def analyze(self):
        """Run complete architecture analysis"""
        print("🔍 Analyzing Groggy architecture...")
        
        # Step 1: Find all Rust files
        rust_files = list(self.src_dir.rglob("*.rs"))
        print(f"📁 Found {len(rust_files)} Rust files")
        
        # Step 2: Parse each file
        for rust_file in rust_files:
            self.parse_file(rust_file)
        
        # Step 3: Build dependency graph
        self.build_dependency_graph()
        
        # Step 4: Generate analysis report
        return self.generate_report()

    def parse_file(self, file_path):
        """Parse a single Rust file for function definitions and calls"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            return

        current_impl = None
        current_trait = None
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Skip comments and empty lines
            if not line_stripped or line_stripped.startswith('//'):
                continue
            
            # Track current impl block
            impl_match = self.impl_pattern.search(line)
            if impl_match:
                current_impl = impl_match.group(1)
                continue
            
            # Track current trait
            trait_match = self.trait_def_pattern.search(line)
            if trait_match:
                current_trait = trait_match.group(1)
                continue
            
            # Find function definitions
            fn_def_match = self.fn_def_pattern.search(line)
            if fn_def_match:
                fn_name = fn_def_match.group(1)
                
                # Determine implementation status
                impl_status = "implemented"
                if "todo!" in line or any(self.todo_pattern.search(l) for l in lines[line_num:line_num+10]):
                    impl_status = "todo"
                
                # Determine visibility
                visibility = "public" if line.strip().startswith("pub") else "private"
                
                # Create qualified name if in impl block
                qualified_name = fn_name
                if current_impl:
                    qualified_name = f"{current_impl}::{fn_name}"
                elif current_trait:
                    qualified_name = f"{current_trait}::{fn_name}"
                
                self.function_defs[qualified_name] = FunctionDef(
                    qualified_name, str(file_path), line_num, impl_status, visibility
                )
                
                # Also store unqualified name for simple lookups
                if qualified_name != fn_name:
                    self.function_defs[fn_name] = self.function_defs[qualified_name]
            
            # Find function calls
            self.extract_function_calls(line, str(file_path), line_num)

    def extract_function_calls(self, line, file_path, line_num):
        """Extract function calls from a line of code"""
        for pattern in self.call_patterns:
            matches = pattern.finditer(line)
            for match in matches:
                fn_name = match.group(1)
                
                # Filter out common non-function calls
                if self.is_likely_function_call(fn_name, line):
                    self.function_calls[fn_name].append(
                        FunctionCall(fn_name, file_path, line_num, line.strip())
                    )

    def is_likely_function_call(self, name, context):
        """Heuristic to determine if a name is likely a function call"""
        # Skip common non-function patterns
        excluded = {
            'String', 'Vec', 'HashMap', 'HashSet', 'Option', 'Result', 'Box',
            'Arc', 'Rc', 'Cell', 'RefCell', 'Mutex', 'RwLock', 'u8', 'u16', 
            'u32', 'u64', 'i8', 'i16', 'i32', 'i64', 'f32', 'f64', 'bool',
            'usize', 'isize', 'Ok', 'Err', 'Some', 'None', 'Self', 'self',
            'true', 'false', 'const', 'static', 'let', 'mut', 'pub', 'fn',
            'struct', 'enum', 'trait', 'impl', 'use', 'mod', 'crate', 'super'
        }
        
        if name in excluded:
            return False
        
        # Skip macro calls (end with !)
        if '!' in context:
            return False
            
        # Skip type annotations
        if ':' in context and '->' not in context:
            return False
            
        return True

    def build_dependency_graph(self):
        """Build function call dependency graph"""
        print("🔗 Building dependency graph...")
        
        for caller_name, calls in self.function_calls.items():
            for call in calls:
                callee_name = call.name
                
                # Try to find the actual function definition
                if callee_name in self.function_defs:
                    self.dependency_graph[caller_name].add(callee_name)
                    self.reverse_deps[callee_name].add(caller_name)

    def find_missing_functions(self):
        """Find function calls that don't have corresponding definitions"""
        missing = set()
        
        for called_fn, calls in self.function_calls.items():
            if called_fn not in self.function_defs:
                # Check if it's a method on a known type or external crate
                if not self.is_external_or_builtin(called_fn):
                    missing.add(called_fn)
        
        return missing

    def is_external_or_builtin(self, fn_name):
        """Check if function is likely from external crate or built-in"""
        external_patterns = [
            'std', 'core', 'alloc', 'serde', 'tokio', 'futures',
            'log', 'debug', 'info', 'warn', 'error', 'trace',
            'println', 'print', 'format', 'panic', 'assert',
            'clone', 'iter', 'collect', 'map', 'filter', 'fold',
            'push', 'pop', 'len', 'is_empty', 'clear', 'insert',
            'get', 'contains', 'remove', 'extend', 'entry',
            'new', 'default', 'from', 'into', 'as_ref', 'as_mut'
        ]
        
        return any(pattern in fn_name.lower() for pattern in external_patterns)

    def find_todo_functions(self):
        """Find functions marked as TODO or unimplemented"""
        todos = []
        
        for fn_name, fn_def in self.function_defs.items():
            if fn_def.impl_status == "todo":
                todos.append(fn_def)
        
        return todos

    def find_orphaned_functions(self):
        """Find functions that are never called"""
        called_functions = set(self.function_calls.keys())
        defined_functions = set(self.function_defs.keys())
        
        # Functions that are defined but never called
        orphaned = defined_functions - called_functions
        
        # Filter out likely entry points and public APIs
        filtered_orphaned = []
        for fn_name in orphaned:
            fn_def = self.function_defs[fn_name]
            if fn_def.visibility == "private" and not self.is_likely_entry_point(fn_name):
                filtered_orphaned.append(fn_def)
        
        return filtered_orphaned

    def is_likely_entry_point(self, fn_name):
        """Check if function is likely an entry point (main, new, default, etc.)"""
        entry_points = ['main', 'new', 'default', 'from', 'into', 'test_']
        return any(fn_name.startswith(ep) for ep in entry_points)

    def analyze_connectivity(self):
        """Analyze connectivity patterns in the dependency graph"""
        # Find strongly connected components
        visited = set()
        components = []
        
        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.append(node)
            
            for neighbor in self.dependency_graph[node]:
                dfs(neighbor, component)
        
        for node in self.function_defs.keys():
            if node not in visited:
                component = []
                dfs(node, component)
                if len(component) > 1:
                    components.append(component)
        
        return components

    def generate_report(self):
        """Generate comprehensive architecture analysis report"""
        print("📊 Generating analysis report...")
        
        missing_functions = self.find_missing_functions()
        todo_functions = self.find_todo_functions()
        orphaned_functions = self.find_orphaned_functions()
        connectivity_components = self.analyze_connectivity()
        
        report = {
            'summary': {
                'total_functions_defined': len(self.function_defs),
                'total_function_calls': sum(len(calls) for calls in self.function_calls.values()),
                'missing_functions': len(missing_functions),
                'todo_functions': len(todo_functions),
                'orphaned_functions': len(orphaned_functions),
                'connectivity_components': len(connectivity_components)
            },
            'missing_functions': list(missing_functions),
            'todo_functions': [
                {
                    'name': fn.name,
                    'file': fn.file_path,
                    'line': fn.line_num,
                    'visibility': fn.visibility
                }
                for fn in todo_functions
            ],
            'orphaned_functions': [
                {
                    'name': fn.name,
                    'file': fn.file_path,
                    'line': fn.line_num,
                    'visibility': fn.visibility
                }
                for fn in orphaned_functions
            ],
            'function_definitions': {
                name: {
                    'file': fn_def.file_path,
                    'line': fn_def.line_num,
                    'status': fn_def.impl_status,
                    'visibility': fn_def.visibility
                }
                for name, fn_def in self.function_defs.items()
            },
            'dependency_graph': {
                caller: list(callees) 
                for caller, callees in self.dependency_graph.items()
            }
        }
        
        return report

    def print_report(self, report):
        """Print human-readable analysis report"""
        print("\n" + "="*80)
        print("🏗️  GROGGY ARCHITECTURE ANALYSIS REPORT")
        print("="*80)
        
        # Summary
        summary = report['summary']
        print(f"\n📈 SUMMARY:")
        print(f"  • Functions Defined: {summary['total_functions_defined']}")
        print(f"  • Function Calls: {summary['total_function_calls']}")
        print(f"  • Missing Functions: {summary['missing_functions']}")
        print(f"  • TODO Functions: {summary['todo_functions']}")
        print(f"  • Orphaned Functions: {summary['orphaned_functions']}")
        
        # Missing Functions (Loose Ends)
        if report['missing_functions']:
            print(f"\n❌ MISSING FUNCTIONS ({len(report['missing_functions'])}):")
            for fn_name in report['missing_functions'][:10]:  # Show first 10
                calls = self.function_calls[fn_name]
                print(f"  • {fn_name}")
                for call in calls[:3]:  # Show first 3 calls
                    relative_path = call.file_path.replace(str(self.src_dir), "src")
                    print(f"    └── called in {relative_path}:{call.line_num}")
            if len(report['missing_functions']) > 10:
                print(f"    ... and {len(report['missing_functions']) - 10} more")
        
        # TODO Functions
        if report['todo_functions']:
            print(f"\n⚠️  TODO FUNCTIONS ({len(report['todo_functions'])}):")
            for fn_info in report['todo_functions'][:10]:
                relative_path = fn_info['file'].replace(str(self.src_dir), "src")
                print(f"  • {fn_info['name']} ({fn_info['visibility']}) - {relative_path}:{fn_info['line']}")
            if len(report['todo_functions']) > 10:
                print(f"    ... and {len(report['todo_functions']) - 10} more")
        
        # Architecture Health Check
        print(f"\n🏥 ARCHITECTURE HEALTH:")
        completion_rate = ((summary['total_functions_defined'] - summary['todo_functions']) / 
                          max(summary['total_functions_defined'], 1)) * 100
        print(f"  • Implementation Completion: {completion_rate:.1f}%")
        
        missing_rate = (summary['missing_functions'] / 
                       max(summary['total_function_calls'], 1)) * 100
        print(f"  • Missing Function Rate: {missing_rate:.1f}%")
        
        if missing_rate < 5:
            print("  ✅ Low missing function rate - architecture is well-connected")
        elif missing_rate < 15:
            print("  ⚠️  Moderate missing function rate - some loose ends")
        else:
            print("  ❌ High missing function rate - significant architectural gaps")
        
        # Key Components Analysis
        print(f"\n🔧 KEY COMPONENTS:")
        component_files = {}
        for fn_name, fn_def in self.function_defs.items():
            file_key = Path(fn_def.file_path).stem
            if file_key not in component_files:
                component_files[file_key] = {'total': 0, 'todo': 0, 'public': 0}
            
            component_files[file_key]['total'] += 1
            if fn_def.impl_status == 'todo':
                component_files[file_key]['todo'] += 1
            if fn_def.visibility == 'public':
                component_files[file_key]['public'] += 1
        
        for component, stats in sorted(component_files.items()):
            if stats['total'] > 5:  # Only show significant components
                completion = ((stats['total'] - stats['todo']) / stats['total']) * 100
                print(f"  • {component}: {stats['total']} functions ({completion:.0f}% complete, {stats['public']} public)")

def main():
    """Main analysis function"""
    analyzer = ArchitectureAnalyzer()
    report = analyzer.analyze()
    analyzer.print_report(report)
    
    # Save detailed report to JSON
    output_file = "architecture_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {output_file}")
    
    # Generate DOT file for graph visualization
    dot_file = "function_dependencies.dot"
    with open(dot_file, 'w') as f:
        f.write("digraph function_dependencies {\n")
        f.write("  rankdir=TB;\n")
        f.write("  node [shape=box, style=filled];\n")
        
        # Color nodes by implementation status
        for fn_name, fn_def in analyzer.function_defs.items():
            color = "lightgreen" if fn_def.impl_status == "implemented" else "lightcoral"
            f.write(f'  "{fn_name}" [fillcolor="{color}"];\n')
        
        # Add edges
        for caller, callees in analyzer.dependency_graph.items():
            for callee in callees:
                f.write(f'  "{caller}" -> "{callee}";\n')
        
        f.write("}\n")
    
    print(f"📊 Dependency graph saved to: {dot_file}")
    print("   (Use 'dot -Tpng function_dependencies.dot -o dependencies.png' to visualize)")

if __name__ == "__main__":
    main()