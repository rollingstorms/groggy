#!/usr/bin/env python3
"""
Duplicate Function Finder for Rust Codebase

This script finds duplicate function definitions that may be outdated versions
of the same functionality.
"""

import os
import re
from collections import defaultdict
from pathlib import Path

class DuplicateFinder:
    def __init__(self, src_dir="src"):
        self.src_dir = Path(src_dir)
        self.function_locations = defaultdict(list)  # func_name -> [(file, line, signature)]
        
        # Pattern for function definitions including visibility and signatures
        self.fn_pattern = re.compile(
            r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\([^)]*\)(?:\s*->\s*[^{]+)?)\s*\{'
        )

    def find_all_functions(self):
        """Find all function definitions with their signatures"""
        rust_files = list(self.src_dir.rglob("*.rs"))
        
        for rust_file in rust_files:
            try:
                with open(rust_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Error reading {rust_file}: {e}")
                continue
            
            current_impl = None
            current_trait = None
            
            for line_num, line in enumerate(lines, 1):
                # Track impl blocks
                impl_match = re.search(r'impl\s+(?:\w+\s+for\s+)?(\w+)', line)
                if impl_match:
                    current_impl = impl_match.group(1)
                    continue
                
                # Track trait definitions
                trait_match = re.search(r'trait\s+(\w+)', line)
                if trait_match:
                    current_trait = trait_match.group(1)
                    continue
                
                # Reset on closing braces (simplified heuristic)
                if line.strip() == '}' and (current_impl or current_trait):
                    current_impl = None
                    current_trait = None
                    continue
                
                # Find function definitions
                fn_match = self.fn_pattern.search(line)
                if fn_match:
                    func_name = fn_match.group(1)
                    signature = fn_match.group(2).strip()
                    
                    # Create context-aware name
                    if current_impl:
                        context_name = f"{current_impl}::{func_name}"
                    elif current_trait:
                        context_name = f"{current_trait}::{func_name}"
                    else:
                        context_name = func_name
                    
                    # Get surrounding context for better comparison
                    start_line = max(0, line_num - 3)
                    end_line = min(len(lines), line_num + 10)
                    context_lines = ''.join(lines[start_line:end_line]).strip()
                    
                    self.function_locations[context_name].append({
                        'file': str(rust_file),
                        'line': line_num,
                        'signature': signature,
                        'context': context_lines,
                        'is_todo': 'todo!' in context_lines.lower(),
                        'visibility': 'public' if line.strip().startswith('pub') else 'private'
                    })

    def find_duplicates(self):
        """Find functions that appear multiple times"""
        duplicates = {}
        
        for func_name, locations in self.function_locations.items():
            if len(locations) > 1:
                # Group by similar signatures to find true duplicates
                signature_groups = defaultdict(list)
                
                for loc in locations:
                    # Normalize signature for comparison
                    normalized_sig = self.normalize_signature(loc['signature'])
                    signature_groups[normalized_sig].append(loc)
                
                # Only report if we have multiple locations with similar signatures
                for sig, locs in signature_groups.items():
                    if len(locs) > 1:
                        duplicates[f"{func_name} {sig}"] = locs
        
        return duplicates

    def normalize_signature(self, signature):
        """Normalize function signature for comparison"""
        # Remove whitespace variations
        normalized = re.sub(r'\s+', ' ', signature.strip())
        # Remove parameter names but keep types
        normalized = re.sub(r'(\w+):\s*', '', normalized)
        return normalized

    def find_similar_functions(self):
        """Find functions with very similar names that might be duplicates"""
        similar = defaultdict(list)
        func_names = list(self.function_locations.keys())
        
        for i, name1 in enumerate(func_names):
            base_name1 = name1.split('::')[-1]  # Remove context prefix
            for name2 in func_names[i+1:]:
                base_name2 = name2.split('::')[-1]
                
                # Check for similar names (edit distance, common prefixes, etc.)
                if self.are_similar_names(base_name1, base_name2):
                    similar[f"{base_name1} / {base_name2}"].extend([
                        self.function_locations[name1],
                        self.function_locations[name2]
                    ])
        
        return similar

    def are_similar_names(self, name1, name2):
        """Check if two function names are suspiciously similar"""
        if name1 == name2:
            return True
        
        # Check for common patterns of duplicates
        patterns = [
            # Version suffixes
            (name1.endswith('_v2') and name2 == name1[:-3]),
            (name2.endswith('_v2') and name1 == name2[:-3]),
            # Old/new prefixes  
            (name1.startswith('old_') and name2 == name1[4:]),
            (name2.startswith('old_') and name1 == name2[4:]),
            (name1.startswith('new_') and name2 == name1[4:]),
            (name2.startswith('new_') and name1 == name2[4:]),
            # Temporary suffixes
            (name1.endswith('_temp') and name2 == name1[:-5]),
            (name2.endswith('_temp') and name1 == name2[:-5]),
        ]
        
        return any(patterns)

    def generate_report(self):
        """Generate comprehensive duplicate analysis report"""
        print("🔍 Scanning for duplicate functions...")
        self.find_all_functions()
        
        duplicates = self.find_duplicates()
        similar = self.find_similar_functions()
        
        print(f"\n📊 DUPLICATE FUNCTION ANALYSIS")
        print(f"=" * 60)
        
        print(f"\nTotal functions found: {sum(len(locs) for locs in self.function_locations.values())}")
        print(f"Exact duplicates found: {len(duplicates)}")
        print(f"Similar name groups: {len(similar)}")
        
        # Report exact duplicates
        if duplicates:
            print(f"\n❌ EXACT DUPLICATES ({len(duplicates)}):")
            for func_sig, locations in duplicates.items():
                print(f"\n🔴 {func_sig}:")
                
                for i, loc in enumerate(locations):
                    rel_path = loc['file'].replace(str(self.src_dir), "src")
                    status = "🚧 TODO" if loc['is_todo'] else "✅ IMPL"
                    visibility = "🔓 pub" if loc['visibility'] == 'public' else "🔒 priv"
                    
                    print(f"  {i+1}. {rel_path}:{loc['line']} {visibility} {status}")
                    
                    # Show a snippet of the function
                    context_lines = loc['context'].split('\n')[:3]
                    for line in context_lines:
                        if line.strip() and not line.strip().startswith('//'):
                            print(f"     {line.strip()[:80]}")
                            break
        
        # Report similar functions
        if similar:
            print(f"\n⚠️  SIMILAR FUNCTION NAMES ({len(similar)}):")
            for name_pair, all_locations in similar.items():
                if len(all_locations) > 2:  # Only show if there are actual multiple locations
                    print(f"\n🟡 {name_pair}:")
                    for loc in all_locations[:4]:  # Show first 4 locations
                        rel_path = loc['file'].replace(str(self.src_dir), "src")
                        status = "🚧 TODO" if loc['is_todo'] else "✅ IMPL"
                        print(f"  • {rel_path}:{loc['line']} {status}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        priority_duplicates = []
        for func_sig, locations in duplicates.items():
            # Prioritize duplicates where one is TODO and one is implemented
            todo_count = sum(1 for loc in locations if loc['is_todo'])
            impl_count = len(locations) - todo_count
            
            if todo_count > 0 and impl_count > 0:
                priority_duplicates.append((func_sig, locations, 'mixed'))
            elif todo_count == len(locations):
                priority_duplicates.append((func_sig, locations, 'all_todo'))
            else:
                priority_duplicates.append((func_sig, locations, 'all_impl'))
        
        if priority_duplicates:
            print("\n🎯 HIGH PRIORITY FIXES:")
            for func_sig, locations, fix_type in priority_duplicates:
                if fix_type == 'mixed':
                    print(f"  🔥 {func_sig}: Remove TODO versions, keep implemented ones")
                elif fix_type == 'all_todo':
                    print(f"  📝 {func_sig}: Consolidate multiple TODO stubs")
                else:
                    print(f"  🤔 {func_sig}: Choose best implementation, remove others")
        
        return {
            'duplicates': duplicates,
            'similar': similar,
            'total_functions': sum(len(locs) for locs in self.function_locations.values())
        }

def main():
    finder = DuplicateFinder()
    report = finder.generate_report()
    
    # Save detailed analysis
    with open('duplicate_functions.txt', 'w') as f:
        f.write("DUPLICATE FUNCTION ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        for func_sig, locations in report['duplicates'].items():
            f.write(f"DUPLICATE: {func_sig}\n")
            for loc in locations:
                f.write(f"  - {loc['file']}:{loc['line']} ({loc['visibility']}, {'TODO' if loc['is_todo'] else 'IMPL'})\n")
            f.write("\n")
    
    print(f"\n💾 Detailed analysis saved to: duplicate_functions.txt")

if __name__ == "__main__":
    main()