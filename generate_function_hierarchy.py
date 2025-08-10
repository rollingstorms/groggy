#!/usr/bin/env python3
"""
Function Hierarchy Generator for Groggy Codebase

This script generates a comprehensive function hierarchy documentation
showing the clean API structure after duplicate cleanup.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json

class FunctionHierarchyGenerator:
    def __init__(self, src_dir="src"):
        self.src_dir = Path(src_dir)
        self.modules = {}  # module_name -> {functions: [], structs: [], traits: []}
        
    def analyze_codebase(self):
        """Analyze the entire codebase and build hierarchy"""
        rust_files = list(self.src_dir.rglob("*.rs"))
        
        for rust_file in rust_files:
            module_name = self.get_module_name(rust_file)
            self.modules[module_name] = self.analyze_file(rust_file)
            
        return self.modules
    
    def get_module_name(self, file_path):
        """Convert file path to module name"""
        relative = file_path.relative_to(self.src_dir)
        if relative.name == "mod.rs":
            return str(relative.parent).replace("/", "::")
        elif relative.name == "lib.rs":
            return "lib"
        else:
            return str(relative.with_suffix("")).replace("/", "::")
    
    def analyze_file(self, file_path):
        """Analyze a single Rust file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return {"functions": [], "structs": [], "traits": [], "impls": []}
        
        return {
            "functions": self.extract_functions(content),
            "structs": self.extract_structs(content),
            "traits": self.extract_traits(content),
            "impls": self.extract_impls(content)
        }
    
    def extract_functions(self, content):
        """Extract function definitions"""
        functions = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Match function definitions
            fn_match = re.search(r'^\s*(pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\([^)]*\)(?:\s*->\s*[^{]+)?)', line)
            if fn_match:
                visibility = "public" if fn_match.group(1) else "private"
                name = fn_match.group(2)
                signature = fn_match.group(3)
                
                # Check for TODO status
                is_todo = False
                for check_line in lines[i:min(i+10, len(lines))]:
                    if "todo!" in check_line.lower() or "todo:" in check_line.lower():
                        is_todo = True
                        break
                
                # Get doc comment if present
                doc_comment = ""
                if i > 0 and lines[i-1].strip().startswith("///"):
                    doc_comment = lines[i-1].strip()[3:].strip()
                
                functions.append({
                    "name": name,
                    "signature": signature,
                    "visibility": visibility,
                    "is_todo": is_todo,
                    "doc_comment": doc_comment,
                    "line": i + 1
                })
        
        return functions
    
    def extract_structs(self, content):
        """Extract struct definitions"""
        structs = []
        struct_pattern = re.compile(r'^\s*(pub\s+)?struct\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.MULTILINE)
        
        for match in struct_pattern.finditer(content):
            visibility = "public" if match.group(1) else "private"
            name = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            
            structs.append({
                "name": name,
                "visibility": visibility,
                "line": line_num
            })
        
        return structs
    
    def extract_traits(self, content):
        """Extract trait definitions"""
        traits = []
        trait_pattern = re.compile(r'^\s*(pub\s+)?trait\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.MULTILINE)
        
        for match in trait_pattern.finditer(content):
            visibility = "public" if match.group(1) else "private"
            name = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            
            traits.append({
                "name": name,
                "visibility": visibility,
                "line": line_num
            })
        
        return traits
    
    def extract_impls(self, content):
        """Extract impl blocks"""
        impls = []
        impl_pattern = re.compile(r'^\s*impl\s+(?:(\w+)\s+for\s+)?([a-zA-Z_][a-zA-Z0-9_]*)', re.MULTILINE)
        
        for match in impl_pattern.finditer(content):
            trait_name = match.group(1)
            struct_name = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            
            impls.append({
                "struct_name": struct_name,
                "trait_name": trait_name,
                "line": line_num
            })
        
        return impls

    def generate_markdown_report(self, modules):
        """Generate comprehensive markdown report"""
        md = ["# Groggy Function Hierarchy - Clean API Structure\n"]
        md.append("*Generated after duplicate function cleanup*\n")
        
        # Summary
        total_functions = sum(len(mod["functions"]) for mod in modules.values())
        total_structs = sum(len(mod["structs"]) for mod in modules.values())
        total_traits = sum(len(mod["traits"]) for mod in modules.values())
        todo_functions = sum(sum(1 for f in mod["functions"] if f["is_todo"]) for mod in modules.values())
        
        md.append("## 📊 Summary\n")
        md.append(f"- **Total Functions**: {total_functions}")
        md.append(f"- **Total Structs**: {total_structs}")
        md.append(f"- **Total Traits**: {total_traits}")
        md.append(f"- **TODO Functions**: {todo_functions}")
        md.append(f"- **Implementation Rate**: {((total_functions - todo_functions) / total_functions * 100):.1f}%\n")
        
        # Core API sections
        md.append("## 🎯 Core API (graph.rs)\n")
        if "api::graph" in modules:
            self.add_module_section(md, "api::graph", modules["api::graph"], show_details=True)
        
        md.append("## 🏗️ Architecture Components\n")
        
        # Key modules in order of importance
        key_modules = [
            ("core::pool", "Data Storage Layer"),
            ("core::space", "Active State Tracking"),  
            ("core::strategies", "Temporal Storage Strategies"),
            ("core::change_tracker", "Change Management"),
            ("core::history", "Version Control System"),
            ("core::state", "State Management"),
            ("core::delta", "Change Deltas"),
            ("core::query", "Query Engine"),
            ("core::ref_manager", "Branch/Tag Management"),
            ("config", "Configuration Management"),
            ("types", "Core Type System"),
            ("errors", "Error Handling")
        ]
        
        for module_name, description in key_modules:
            if module_name in modules:
                md.append(f"### {module_name} - {description}\n")
                self.add_module_section(md, module_name, modules[module_name])
        
        # Remaining modules
        remaining = set(modules.keys()) - {"api::graph"} - {m[0] for m in key_modules}
        if remaining:
            md.append("## 📚 Supporting Modules\n")
            for module_name in sorted(remaining):
                md.append(f"### {module_name}\n")
                self.add_module_section(md, module_name, modules[module_name])
        
        return "\n".join(md)
    
    def add_module_section(self, md, module_name, module_data, show_details=False):
        """Add a module section to the markdown"""
        functions = module_data["functions"]
        structs = module_data["structs"]
        traits = module_data["traits"]
        
        if not functions and not structs and not traits:
            md.append("*No public API*\n")
            return
        
        # Stats
        total_funcs = len(functions)
        todo_funcs = sum(1 for f in functions if f["is_todo"])
        pub_funcs = sum(1 for f in functions if f["visibility"] == "public")
        
        md.append(f"**Functions**: {total_funcs} total, {pub_funcs} public, {todo_funcs} TODO")
        md.append(f"**Structs**: {len(structs)}, **Traits**: {len(traits)}\n")
        
        # Show key structs
        if structs:
            pub_structs = [s for s in structs if s["visibility"] == "public"]
            if pub_structs:
                md.append("**Key Types:**")
                for struct in pub_structs[:3]:  # Show first 3
                    md.append(f"- `{struct['name']}`")
                if len(pub_structs) > 3:
                    md.append(f"- ... and {len(pub_structs) - 3} more")
                md.append("")
        
        # Show traits
        if traits:
            pub_traits = [t for t in traits if t["visibility"] == "public"]
            if pub_traits:
                md.append("**Traits:**")
                for trait in pub_traits:
                    md.append(f"- `{trait['name']}`")
                md.append("")
        
        # Show functions (detailed for main API, summary for others)
        if functions:
            pub_functions = [f for f in functions if f["visibility"] == "public"]
            
            if show_details and pub_functions:
                md.append("**Public API:**")
                
                # Group by category for Graph API
                if module_name == "api::graph":
                    categories = self.categorize_graph_functions(pub_functions)
                    for category, funcs in categories.items():
                        md.append(f"\n*{category}:*")
                        for func in funcs:
                            status = "🚧 TODO" if func["is_todo"] else "✅ IMPL"
                            md.append(f"- `{func['name']}{func['signature']}` {status}")
                else:
                    for func in pub_functions[:10]:  # Show first 10
                        status = "🚧 TODO" if func["is_todo"] else "✅ IMPL"
                        md.append(f"- `{func['name']}{func['signature']}` {status}")
                    
                    if len(pub_functions) > 10:
                        md.append(f"- ... and {len(pub_functions) - 10} more functions")
            
            elif pub_functions:
                md.append("**Key Functions:**")
                # Show just the names for non-detailed view
                key_funcs = [f for f in pub_functions if not f["is_todo"]][:5]
                for func in key_funcs:
                    md.append(f"- `{func['name']}()`")
                if len(pub_functions) > 5:
                    md.append(f"- ... {len(pub_functions) - 5} more functions")
        
        md.append("")
    
    def categorize_graph_functions(self, functions):
        """Categorize Graph API functions for better organization"""
        categories = defaultdict(list)
        
        for func in functions:
            name = func["name"]
            
            if name in ["new", "with_config", "load_from_path"]:
                categories["Construction"].append(func)
            elif name.startswith("add_"):
                categories["Entity Creation"].append(func)
            elif name.startswith("remove_"):
                categories["Entity Removal"].append(func)
            elif name.startswith("set_") and "attr" in name:
                categories["Attribute Setting"].append(func)
            elif name.startswith("get_") and "attr" in name:
                categories["Attribute Getting"].append(func)
            elif name in ["contains_node", "contains_edge", "node_ids", "edge_ids", "neighbors", "degree"]:
                categories["Topology Queries"].append(func)
            elif name in ["commit", "checkout", "create_branch", "merge_branch"]:
                categories["Version Control"].append(func)
            elif name.startswith("view_") or name.startswith("list_") or "history" in name:
                categories["History & Views"].append(func)
            elif name in ["statistics", "has_uncommitted_changes", "config"]:
                categories["Status & Config"].append(func)
            else:
                categories["Other Operations"].append(func)
        
        # Sort categories by importance
        order = ["Construction", "Entity Creation", "Entity Removal", "Attribute Setting", 
                "Attribute Getting", "Topology Queries", "Version Control", "History & Views", 
                "Status & Config", "Other Operations"]
        
        return {cat: categories[cat] for cat in order if categories[cat]}

def main():
    generator = FunctionHierarchyGenerator()
    modules = generator.analyze_codebase()
    
    # Generate markdown report
    report = generator.generate_markdown_report(modules)
    
    # Save report
    with open("FUNCTION_HIERARCHY_CLEANED.md", "w") as f:
        f.write(report)
    
    # Save raw data
    with open("function_hierarchy_data.json", "w") as f:
        json.dump(modules, f, indent=2)
    
    print("✅ Function hierarchy documentation generated:")
    print("   📄 FUNCTION_HIERARCHY_CLEANED.md - Readable documentation")
    print("   📊 function_hierarchy_data.json - Raw analysis data")
    
    # Quick stats
    total_functions = sum(len(mod["functions"]) for mod in modules.values())
    todo_functions = sum(sum(1 for f in mod["functions"] if f["is_todo"]) for mod in modules.values())
    
    print(f"\n📈 Quick Stats:")
    print(f"   • {len(modules)} modules analyzed")  
    print(f"   • {total_functions} total functions")
    print(f"   • {total_functions - todo_functions} implemented ({((total_functions - todo_functions) / total_functions * 100):.1f}%)")
    print(f"   • {todo_functions} TODO functions")

if __name__ == "__main__":
    main()