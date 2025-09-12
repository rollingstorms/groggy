#!/usr/bin/env python3
"""
Groggy Display Template Generator

This script extracts the HTML structure and CSS themes from the Rust display engine
to create standalone HTML and CSS files for prototyping styles outside of Rust.

Usage:
    python template_generator.py

Generates:
    - template.html: Sample HTML structure with all data types
    - themes/[theme_name].css: Extracted CSS for each theme
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

class GroggyTemplateGenerator:
    def __init__(self, src_path: str = "../../src/core/display"):
        self.src_path = Path(src_path)
        self.themes_path = self.src_path / "themes"
        self.output_dir = Path("template_prototype")
        
    def generate_all(self):
        """Generate all template files"""
        print("🎨 Groggy Template Generator")
        print("=" * 40)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "themes").mkdir(exist_ok=True)
        
        # Extract and copy CSS themes
        self.extract_css_themes()
        
        # Generate sample HTML
        self.generate_sample_html()
        
        # Generate theme selector HTML
        self.generate_theme_selector()
        
        print(f"\n✅ Templates generated in: {self.output_dir}")
        print(f"   📁 {self.output_dir}/template.html")
        print(f"   📁 {self.output_dir}/theme_selector.html")
        print(f"   📁 {self.output_dir}/themes/")
        
    def extract_css_themes(self):
        """Extract CSS themes from Rust source"""
        print("\n📋 Extracting CSS themes...")
        
        theme_files = ["light.css", "dark.css", "minimal.css", "publication.css"]
        
        for theme_file in theme_files:
            src_file = self.themes_path / theme_file
            dest_file = self.output_dir / "themes" / theme_file
            
            if src_file.exists():
                css_content = src_file.read_text()
                dest_file.write_text(css_content)
                print(f"   ✓ {theme_file}")
            else:
                print(f"   ⚠️ {theme_file} not found")
                
    def generate_sample_html(self):
        """Generate sample HTML with realistic data"""
        print("\n🏗️ Generating sample HTML...")
        
        sample_data = self.create_sample_data()
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Groggy Display Template</title>
    <link rel="stylesheet" href="themes/light.css">
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        
        .template-controls {{
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .theme-selector {{
            margin-bottom: 10px;
        }}
        
        .theme-selector label {{
            font-weight: 500;
            margin-right: 10px;
        }}
        
        .theme-selector select {{
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        
        .sample-section {{
            margin-bottom: 30px;
        }}
        
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }}
        
        .section-description {{
            color: #666;
            margin-bottom: 15px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>🐸 Groggy Display Template Prototype</h1>
    
    <div class="template-controls">
        <div class="theme-selector">
            <label for="theme-select">Theme:</label>
            <select id="theme-select" onchange="changeTheme()">
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="minimal">Minimal</option>
                <option value="publication">Publication</option>
            </select>
        </div>
        <p>Use this template to prototype new styles and themes for Groggy tables.</p>
    </div>

{self.generate_sample_tables(sample_data)}

    <script>
        function changeTheme() {{
            const select = document.getElementById('theme-select');
            const theme = select.value;
            
            // Update CSS link
            const cssLink = document.querySelector('link[rel="stylesheet"]');
            cssLink.href = `themes/${{theme}}.css`;
            
            // Update all theme data attributes
            const containers = document.querySelectorAll('.groggy-display-container');
            containers.forEach(container => {{
                container.setAttribute('data-theme', theme);
            }});
            
            // Update table classes
            const tables = document.querySelectorAll('.groggy-table');
            tables.forEach(table => {{
                table.className = `groggy-table theme-${{theme}}`;
            }});
        }}

        // Interactive features from the original Rust code
        function launchInteractive() {{
            alert('Interactive streaming view coming in Phase 3!\\n\\nThis will launch a WebSocket-powered interface for browsing massive datasets with virtual scrolling.');
        }}

        // Keyboard navigation
        document.addEventListener('DOMContentLoaded', function() {{
            const tables = document.querySelectorAll('.groggy-table');
            tables.forEach(table => {{
                table.setAttribute('tabindex', '0');
                
                table.addEventListener('keydown', function(e) {{
                    if (e.key === 'Enter' || e.key === ' ') {{
                        const interactiveBtn = table.parentElement.querySelector('.interactive-btn');
                        if (interactiveBtn) {{
                            interactiveBtn.click();
                            e.preventDefault();
                        }}
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>"""
        
        output_file = self.output_dir / "template.html"
        output_file.write_text(html_content)
        print(f"   ✓ template.html")
        
    def generate_sample_tables(self, sample_data: Dict) -> str:
        """Generate HTML for sample tables"""
        sections = []
        
        for section_name, data in sample_data.items():
            section_html = f"""
    <div class="sample-section">
        <h2 class="section-title">{data['title']}</h2>
        <p class="section-description">{data['description']}</p>
        {self.generate_table_html(data['headers'], data['rows'], data['schema'], section_name)}
    </div>"""
            sections.append(section_html)
            
        return "\n".join(sections)
        
    def generate_table_html(self, headers: List[str], rows: List[List[str]], 
                           schema: List[Dict], section_name: str) -> str:
        """Generate HTML table based on the Rust HtmlRenderer structure"""
        
        # Generate header row
        header_cells = []
        for i, header in enumerate(headers):
            data_type = schema[i]['data_type'].lower()
            header_cells.append(f'''
            <th class="col-{data_type}" data-type="{schema[i]['data_type']}" title="Column: {header} ({schema[i]['data_type']})">
                {self.escape_html(header)}
            </th>''')
        
        header_row = f"<tr>{''.join(header_cells)}</tr>"
        
        # Generate data rows
        data_rows = []
        for row in rows:
            cells = []
            for i, cell in enumerate(row):
                data_type = schema[i]['data_type'].lower()
                formatted_value = self.format_cell_value(cell, schema[i]['data_type'])
                is_truncated = len(cell) > 30  # Simple truncation check
                
                cell_class = f"cell-{data_type}"
                if is_truncated:
                    cell_class += " cell-truncated"
                    
                title = f"Full value: {cell}" if is_truncated else f"{schema[i]['data_type']} value"
                
                cells.append(f'''
                <td class="{cell_class}" data-type="{schema[i]['data_type']}" title="{self.escape_html(title)}">
                    {self.escape_html(formatted_value)}
                </td>''')
            
            data_rows.append(f"<tr>{''.join(cells)}</tr>")
            
        tbody = "\n".join(data_rows)
        
        # Table info section
        table_info = f'''
        <div class="table-info">
            <span>Showing {len(rows)} of {len(rows) + 1000} rows (sample data)</span>
            <button class="interactive-btn" onclick="launchInteractive()" 
                    title="Launch interactive view for full dataset">
                View All ({len(rows) + 1000} rows) →
            </button>
        </div>'''
        
        return f'''
        <div class="groggy-display-container" data-theme="light">
            <table class="groggy-table theme-light">
                <thead>
                    {header_row}
                </thead>
                <tbody>
                    {tbody}
                </tbody>
            </table>
            {table_info}
        </div>'''
        
    def format_cell_value(self, value: str, data_type: str) -> str:
        """Format cell values based on data type (simplified version of Rust logic)"""
        if data_type == "Float":
            try:
                num = float(value)
                if abs(num) < 0.001 and num != 0.0:
                    return f"{num:.2e}"
                else:
                    return f"{num:.2f}"
            except:
                return value
                
        elif data_type == "Integer":
            try:
                num = int(value)
                if abs(num) > 9999:
                    return f"{num:,}"
                else:
                    return value
            except:
                return value
                
        elif data_type == "Boolean":
            if value.lower() in ["true", "1", "yes", "on"]:
                return "✓"
            elif value.lower() in ["false", "0", "no", "off"]:
                return "✗"
            else:
                return value
                
        # Truncate long values
        if len(value) > 30:
            return value[:29] + "…"
            
        return value
        
    def escape_html(self, text: str) -> str:
        """Escape HTML characters"""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#39;"))
        
    def create_sample_data(self) -> Dict:
        """Create sample data demonstrating all data types and features"""
        return {
            "basic_types": {
                "title": "Basic Data Types",
                "description": "Demonstrates styling for different data types: strings, numbers, booleans, etc.",
                "headers": ["name", "age", "score", "active", "salary", "join_date"],
                "schema": [
                    {"data_type": "String"},
                    {"data_type": "Integer"}, 
                    {"data_type": "Float"},
                    {"data_type": "Boolean"},
                    {"data_type": "Integer"},
                    {"data_type": "DateTime"}
                ],
                "rows": [
                    ["Alice Johnson", "25", "91.50", "true", "75000", "2023-01-15"],
                    ["Bob & Associates LLC", "30000", "87.00", "false", "125000", "2022-03-22"],
                    ["Carol Smith", "28", "95.75", "true", "82000", "2023-06-10"],
                    ["David <Test> User", "35", "88.25", "true", "95000", "2021-11-30"],
                    ["Eve 'Quotes' Wilson", "42", "92.00", "false", "110000", "2020-08-15"]
                ]
            },
            "large_numbers": {
                "title": "Large Numbers & Scientific Notation", 
                "description": "Tests number formatting with large values, scientific notation, and precision.",
                "headers": ["metric", "small_value", "large_value", "scientific", "percentage"],
                "schema": [
                    {"data_type": "String"},
                    {"data_type": "Float"},
                    {"data_type": "Integer"},
                    {"data_type": "Float"}, 
                    {"data_type": "Float"}
                ],
                "rows": [
                    ["Revenue", "0.00012", "1250000", "1.23e-8", "15.75"],
                    ["Users", "0.001", "50000000", "6.02e23", "99.99"],
                    ["Storage (TB)", "0.0001", "987654321", "3.14159265", "0.01"],
                    ["Throughput", "0.000001", "1000000000", "2.71828", "100.00"]
                ]
            },
            "edge_cases": {
                "title": "Edge Cases & Truncation",
                "description": "Tests truncation, special characters, and edge cases in data display.",
                "headers": ["type", "short", "very_long_content", "special_chars", "json_data"],
                "schema": [
                    {"data_type": "String"},
                    {"data_type": "String"},
                    {"data_type": "String"},
                    {"data_type": "String"},
                    {"data_type": "Json"}
                ],
                "rows": [
                    ["Normal", "Short", "This is a very long piece of content that should be truncated when displayed in the table to prevent very wide cells", "Normal text", '{"key": "value", "number": 42}'],
                    ["HTML", "Test", "Content with <script>alert('xss')</script> and other HTML", "&lt;tag&gt;", '{"html": "<div>content</div>"}'],
                    ["Unicode", "🚀", "Unicode content with émojis 🎉 and spëcial çharacters", "Spéciål ¢hars", '{"emoji": "🌟", "text": "unicode"}'],
                    ["Quotes", "'Test'", 'Content with "double quotes" and \'single quotes\'', '"Quoted"', '{"nested": {"quotes": "test"}}']
                ]
            }
        }
        
    def generate_theme_selector(self):
        """Generate a theme comparison page"""
        print("\n🎨 Generating theme selector...")
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Groggy Theme Selector</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        
        .theme-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .theme-preview {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .theme-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 10px;
            text-align: center;
        }}
        
        .preview-container {{
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
        }}
    </style>
</head>
<body>
    <h1>🎨 Groggy Theme Comparison</h1>
    <p>Compare all available themes side by side to help with styling decisions.</p>
    
    <div class="theme-grid">
        {self.generate_theme_previews()}
    </div>
</body>
</html>"""
        
        output_file = self.output_dir / "theme_selector.html"
        output_file.write_text(html_content)
        print(f"   ✓ theme_selector.html")
        
    def generate_theme_previews(self) -> str:
        """Generate preview for each theme"""
        themes = ["light", "dark", "minimal", "publication"]
        previews = []
        
        sample_data = self.create_sample_data()["basic_types"]
        
        for theme in themes:
            theme_title = theme.title()
            preview_html = f'''
        <div class="theme-preview">
            <div class="theme-title">{theme_title}</div>
            <div class="preview-container">
                <link rel="stylesheet" href="themes/{theme}.css">
                <div class="groggy-display-container" data-theme="{theme}">
                    <table class="groggy-table theme-{theme}">
                        <thead>
                            <tr>
                                <th class="col-string" data-type="String">Name</th>
                                <th class="col-integer" data-type="Integer">Age</th>
                                <th class="col-float" data-type="Float">Score</th>
                                <th class="col-boolean" data-type="Boolean">Active</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="cell-string" data-type="String">Alice</td>
                                <td class="cell-integer" data-type="Integer">25</td>
                                <td class="cell-float" data-type="Float">91.50</td>
                                <td class="cell-boolean" data-type="Boolean">✓</td>
                            </tr>
                            <tr>
                                <td class="cell-string" data-type="String">Bob</td>
                                <td class="cell-integer" data-type="Integer">30,000</td>
                                <td class="cell-float" data-type="Float">87.00</td>
                                <td class="cell-boolean" data-type="Boolean">✗</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>'''
            previews.append(preview_html)
            
        return "\n".join(previews)

def main():
    """Main function"""
    generator = GroggyTemplateGenerator()
    generator.generate_all()
    
    print("\n🎯 Next Steps:")
    print("   1. Open template.html in your browser")
    print("   2. Use browser dev tools to experiment with CSS")
    print("   3. Edit CSS files in themes/ directory")
    print("   4. Copy successful changes back to Rust source")
    print("   5. Use theme_selector.html to compare themes")

if __name__ == "__main__":
    main()
