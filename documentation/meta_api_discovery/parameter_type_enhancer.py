#!/usr/bin/env python3
"""
Parameter Type Enhancer

Analyzes method documentation to extract parameter types and create enhanced signatures
with Python-style type hints.
"""

import json
import re
from typing import Dict, List, Optional, Tuple


class ParameterTypeInferencer:
    def __init__(self):
        # Common type patterns found in documentation
        self.type_patterns = {
            # Basic types (order matters - more specific first)
            r'\b(?:dictionary|dict|Dict)\b': 'dict',
            r'\b(?:list|List|array|Array)\b': 'list', 
            r'\b(?:string|str|String)\b': 'str',
            r'\b(?:integer|int|Integer)\b': 'int',
            r'\b(?:number|float|Float|numeric)\b': 'float',
            r'\b(?:boolean|bool|Boolean)\b': 'bool',
            r'\b(?:callable|function|Function)\b': 'callable',
            r'\b(?:path|Path|filepath)\b': 'str',
            
            # Groggy specific types
            r'\bBaseTable\b': 'BaseTable',
            r'\bNodesTable\b': 'NodesTable', 
            r'\bEdgesTable\b': 'EdgesTable',
            r'\bGraphTable\b': 'GraphTable',
            r'\bGraph\b': 'Graph',
            r'\bSubgraph\b': 'Subgraph',
            r'\bBaseArray\b': 'BaseArray',
            r'\bGraphArray\b': 'GraphArray',
            r'\bComponentsArray\b': 'ComponentsArray',
            r'\bMatrix\b': 'Matrix',
            r'\bnode[\s_]id[s]?\b': 'int',
            r'\bedge[\s_]id[s]?\b': 'int',
            
            # Collection patterns  
            r'list of (\w+)': r'List[\1]',
            r'array of (\w+)': r'List[\1]',
            r'sequence of (\w+)': r'List[\1]',
            r'mapping .* to (\w+)': r'Dict[str, \1]',
            r'dictionary mapping .* to (\w+)': r'Dict[str, \1]',
            
            # Optional patterns
            r'optional (\w+)': r'Optional[\1]',
            r'(\w+) \(optional\)': r'Optional[\1]',
            r'\(default[^)]*\)': '',  # Remove default value info
            
            # Common parameter name patterns
            r'\bcolumn[s]?\b': 'list',  # columns parameter usually list
            r'\bmask\b': 'list',       # mask usually boolean list
            r'\bvalue[s]?\b': 'Any',   # generic values
            r'\bdata\b': 'dict',       # data usually dict
            r'\bupdates\b': 'dict',    # updates usually dict
            r'\bspecs?\b': 'dict',     # specs usually dict
            r'\bparams?\b': 'dict',    # params usually dict
        }
    
    def extract_parameter_info_from_doc(self, doc: str, param_name: str) -> Optional[str]:
        """Extract parameter type from documentation string."""
        if not doc or not param_name:
            return None
            
        # Look for parameter documentation patterns
        patterns = [
            # * `param_name` - Description
            rf'\*\s*`{re.escape(param_name)}`\s*[-–]\s*(.+?)(?:\n|$)',
            # # Arguments section with param_name
            rf'#\s*Arguments.*\n.*`{re.escape(param_name)}`\s*[-–]\s*(.+?)(?:\n|$)',
            # param_name: Description
            rf'{re.escape(param_name)}\s*:\s*(.+?)(?:\n|$)',
            # param_name - Description  
            rf'{re.escape(param_name)}\s*[-–]\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, doc, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                param_description = match.group(1).strip()
                return self.infer_type_from_description(param_description)
        
        return None
    
    def infer_type_from_description(self, description: str) -> Optional[str]:
        """Infer parameter type from description text."""
        description_lower = description.lower()
        
        # Try each type pattern
        for pattern, type_name in self.type_patterns.items():
            if re.search(pattern, description_lower, re.IGNORECASE):
                return type_name
        
        # Special case: look for explicit type mentions in examples
        if 'example' in description_lower:
            # Look for patterns like {'key': 'value'} -> dict
            if re.search(r'\{[^}]*\}', description):
                return 'dict'
            elif re.search(r'\[[^\]]*\]', description):
                return 'list'
        
        return None
    
    def infer_type_from_param_name(self, param_name: str) -> Optional[str]:
        """Infer type from parameter name patterns."""
        param_lower = param_name.lower()
        
        # Common parameter name patterns
        name_patterns = {
            r'.*column[s]?.*': 'list',
            r'.*node[s]?.*': 'int',  
            r'.*edge[s]?.*': 'int',
            r'.*id[s]?$': 'int',
            r'.*path.*': 'str',
            r'.*name[s]?.*': 'str',
            r'.*data.*': 'dict',
            r'.*spec[s]?.*': 'dict',
            r'.*param[s]?.*': 'dict',
            r'.*config.*': 'dict',
            r'.*option[s]?.*': 'dict',
            r'.*mask.*': 'list',
            r'.*target[s]?.*': 'list',
            r'.*source[s]?.*': 'list',
        }
        
        for pattern, type_name in name_patterns.items():
            if re.match(pattern, param_lower):
                return type_name
        
        return None

    def enhance_method_signature(self, method_info: dict) -> dict:
        """Enhance method with parameter type information."""
        enhanced = method_info.copy()
        signature = method_info.get('signature', '')
        doc = method_info.get('doc', '')
        requires_params = method_info.get('requires_parameters', [])
        
        if not signature or not requires_params:
            return enhanced
        
        # Extract parameter types
        param_types = {}
        for param_name in requires_params:
            # Try documentation first
            param_type = self.extract_parameter_info_from_doc(doc, param_name)
            
            # Fall back to name-based inference
            if not param_type:
                param_type = self.infer_type_from_param_name(param_name)
            
            if param_type:
                param_types[param_name] = param_type
        
        # Build enhanced signature with type hints
        if param_types:
            enhanced_params = []
            # Parse existing signature to get parameter order
            sig_match = re.search(r'\(([^)]*)\)', signature)
            if sig_match:
                params_str = sig_match.group(1).strip()
                if params_str:
                    params = [p.strip() for p in params_str.split(',')]
                    for param in params:
                        # Remove default values for now
                        param_name = param.split('=')[0].strip()
                        if param_name in param_types:
                            enhanced_params.append(f"{param_name}: {param_types[param_name]}")
                        else:
                            enhanced_params.append(param)
            
            if enhanced_params:
                enhanced['enhanced_signature'] = f"({', '.join(enhanced_params)})"
                enhanced['parameter_types'] = param_types
        
        return enhanced
    
    def process_discovery_file(self, input_file: str, output_file: str):
        """Process entire discovery file and add parameter type hints."""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        enhanced_data = data.copy()
        total_methods = 0
        enhanced_methods = 0
        
        # Process objects and their methods
        if 'objects' in data:
            for obj_name, obj_data in data['objects'].items():
                if 'methods' in obj_data:
                    enhanced_methods_list = []
                    for method_info in obj_data['methods']:
                        total_methods += 1
                        enhanced_method = self.enhance_method_signature(method_info)
                        enhanced_methods_list.append(enhanced_method)
                        
                        if 'parameter_types' in enhanced_method:
                            enhanced_methods += 1
                            print(f"Enhanced {obj_name}.{method_info.get('name', 'unknown')}: {enhanced_method.get('enhanced_signature', '')}")
                    
                    enhanced_data['objects'][obj_name]['methods'] = enhanced_methods_list
        
        # Add enhancement metadata
        enhanced_data['enhancement_metadata'] = {
            'total_methods': total_methods,
            'enhanced_methods': enhanced_methods,
            'enhancement_rate': f"{enhanced_methods/total_methods*100:.1f}%" if total_methods > 0 else "0%"
        }
        
        # Save enhanced data
        with open(output_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        print(f"\n=== Parameter Type Enhancement Complete ===")
        print(f"Total methods: {total_methods}")
        print(f"Enhanced methods: {enhanced_methods}")
        print(f"Enhancement rate: {enhanced_methods/total_methods*100:.1f}%")
        print(f"Output saved to: {output_file}")
        
        return enhanced_data


def main():
    """Main execution function."""
    inferencer = ParameterTypeInferencer()
    
    input_file = "api_discovery_results.json"
    output_file = "api_discovery_results_enhanced_v2.json"
    
    enhanced_data = inferencer.process_discovery_file(input_file, output_file)
    
    return enhanced_data


if __name__ == "__main__":
    main()