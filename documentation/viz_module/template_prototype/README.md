# 🎨 Groggy Display Template Prototype

This directory contains extracted HTML and CSS templates from the Groggy Rust display engine, allowing you to prototype and experiment with table styles outside of the Rust compilation cycle.

## 📁 Files

- **`template.html`** - Complete HTML template with sample data and theme switching
- **`theme_selector.html`** - Side-by-side comparison of all themes
- **`css_playground.html`** - Live CSS editor with real-time preview
- **`themes/`** - Directory containing all CSS theme files extracted from Rust

## 🚀 Quick Start

1. **Generate templates** (if not already done):
   ```bash
   python template_generator.py
   ```

2. **Start development server**:
   ```bash
   python dev_server.py
   ```
   This will open your browser to `http://localhost:8000/template.html`

3. **Choose your workflow**:
   - **Browse templates**: Open `template.html` for the main template
   - **Compare themes**: Open `theme_selector.html` to see all themes
   - **Live editing**: Open `css_playground.html` for real-time CSS editing

## 🎯 Workflow for Style Development

### Option 1: Direct CSS Editing
1. Edit CSS files in `themes/` directory directly
2. Refresh browser to see changes
3. Copy successful changes back to Rust source files

### Option 2: Live CSS Playground
1. Open `css_playground.html`
2. Select base theme to start from
3. Edit CSS in the left panel
4. See changes instantly in the right panel
5. Download modified CSS when satisfied

### Option 3: Browser DevTools
1. Open `template.html` in browser
2. Use browser's developer tools to experiment
3. Copy working CSS rules to theme files

## 🎨 Theme Architecture

Each theme consists of:

- **Container styles**: `.groggy-display-container[data-theme="name"]`
- **Table structure**: `.groggy-table.theme-name`
- **Data type styling**: `.cell-string`, `.cell-integer`, `.cell-float`, etc.
- **Interactive elements**: `.table-info`, `.interactive-btn`
- **Responsive design**: Media queries for mobile and print

## 📊 Sample Data Types

The templates include examples of:

- **Strings**: Regular text, HTML entities, Unicode
- **Numbers**: Integers, floats, large numbers with formatting
- **Booleans**: True/false values displayed as ✓/✗
- **DateTime**: Date and time values
- **JSON**: Structured data (truncated for display)
- **Edge cases**: Very long content, special characters

## 🎪 Key Features to Style

### Data Type Differentiation
```css
.cell-string { color: #212529; text-align: left; }
.cell-integer, .cell-float { color: #0066cc; text-align: right; }
.cell-boolean { color: #28a745; text-align: center; }
```

### Responsive Design
```css
@media (max-width: 768px) {
  .groggy-table { font-size: 12px; }
}
```

### Interactive Elements
```css
.interactive-btn {
  background-color: #007bff;
  color: white;
  /* hover and focus states */
}
```

### Accessibility
```css
.groggy-table:focus-within {
  outline: 2px solid #007bff;
}
```

## 🔄 Sync Back to Rust

When you've perfected your styles:

1. **Copy CSS**: Replace content in corresponding files in `src/core/display/themes/`
2. **Test in Rust**: Run Rust tests to ensure integration works
3. **Commit changes**: Add both template and Rust source changes

## 🛠️ Customization Tips

### Creating New Themes
1. Copy an existing theme CSS file
2. Modify colors, fonts, spacing
3. Test with various data types
4. Add to Rust theme system

### Color Palette Consistency
Use CSS custom properties for consistent theming:
```css
:root {
  --primary-bg: #ffffff;
  --text-color: #212529;
  --accent-color: #007bff;
}
```

### Typography Hierarchy
- **Headers**: Slightly larger, medium weight
- **Data**: Monospace for numbers, sans-serif for text
- **UI elements**: System fonts for buttons and controls

## 🧪 Testing Checklist

- [ ] All data types render correctly
- [ ] Responsive design works on mobile
- [ ] Print styles are appropriate  
- [ ] Accessibility (keyboard navigation, screen readers)
- [ ] Long content truncation
- [ ] Large number formatting
- [ ] Special character handling
- [ ] Theme switching works smoothly

## 📝 Notes

- CSS is extracted directly from Rust source using `include_str!()` macro
- Theme switching updates both `data-theme` attributes and table classes
- JavaScript includes keyboard navigation and interactive features
- Print styles automatically convert dark themes to light for better printing

## 🎉 Have Fun!

This setup lets you rapidly prototype table styles without Rust compilation. Experiment freely - you can always reset to the base themes!
