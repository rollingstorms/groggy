# 🎨 Groggy Visualization Module Documentation

This directory contains documentation, templates, and prototyping tools for the Groggy visualization system.

## 📁 Contents

### Template Prototyping System
- **`template_generator.py`** - Extracts HTML/CSS from Rust display engine for prototyping
- **`dev_server.py`** - Local development server for testing templates
- **`sync_to_rust.py`** - Syncs modified CSS back to Rust source code
- **`template_prototype/`** - Generated HTML templates and CSS themes
- **`TEMPLATE_SYSTEM_SUMMARY.md`** - Detailed documentation of the template system

## 🚀 Quick Start

1. **Generate templates from Rust source**:
   ```bash
   cd documentation/viz_module
   python template_generator.py
   ```

2. **Start development server**:
   ```bash
   python dev_server.py
   ```
   This opens your browser to the template prototype.

3. **Prototype styles**:
   - Edit CSS files in `template_prototype/themes/`
   - Use the live CSS playground
   - Experiment with browser DevTools

4. **Sync changes back to Rust**:
   ```bash
   python sync_to_rust.py
   ```

## 🎯 Purpose

This system allows rapid prototyping of table styles and themes without requiring Rust compilation cycles. It extracts the exact HTML structure and CSS from the Rust display engine, maintaining perfect fidelity while enabling fast iteration.

## 🏗️ Architecture

The visualization system is built on:

- **Core Display Engine** (`src/core/display/`) - Rust implementation
- **HTML Renderer** (`html.rs`) - Semantic table generation  
- **Theme System** (`theme.rs`) - Unified styling across modes
- **CSS Themes** (`themes/*.css`) - Visual styling definitions

## 📖 Documentation

See `TEMPLATE_SYSTEM_SUMMARY.md` for comprehensive documentation of:
- Generated file structure
- Development workflows
- Architecture details
- Styling guidelines
- Sync procedures

## 🔗 Related

- `/src/core/display/` - Rust source code for display engine
- `/examples/` - Example usage of display system
- `/tests/` - Display system tests
