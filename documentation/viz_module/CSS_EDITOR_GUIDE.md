# 🎨 CSS Editor Guide

This guide explains how to use the enhanced streaming template with the direct CSS editing panel.

## 🚀 Quick Start

1. **Generate the template**:
   ```bash
   cd documentation/viz_module
   python streaming_template_generator.py
   ```

2. **Start the dev server**:
   ```bash
   python streaming_dev_server.py
   ```

3. **Open the template** in your browser (should open automatically)

4. **Click "🎨 Style Panel"** to open the editor

## 🎛️ Two Editing Modes

### **🎛️ Controls Tab**
- **GUI Controls** - Sliders, color pickers, dropdowns
- **Real-time Updates** - Changes apply instantly
- **Export Variables** - Generate CSS custom properties
- **Great for** - Quick tweaks, experimenting with colors

### **💻 CSS Tab**
- **Direct CSS Editing** - Write any CSS you want
- **Auto-apply** - Changes apply automatically as you type (1 second delay)
- **Preset Themes** - Dark, Minimal, Colorful, Neon
- **Export CSS** - Copy full CSS to clipboard
- **Great for** - Advanced styling, complex layouts, animations

## 🎨 CSS Editor Features

### **Live Editing**
- **Auto-apply**: CSS applies 1 second after you stop typing
- **Manual apply**: Click "⚡ Apply Now" for instant application
- **Error handling**: Shows CSS syntax errors with helpful messages

### **Preset Themes**
- **🌙 Dark**: Professional dark theme with blue accents
- **⚪ Minimal**: Clean, minimal styling with gray tones
- **🌈 Colorful**: Vibrant theme with gradients and bright colors
- **⚡ Neon**: Cyberpunk-style neon green with glowing effects

### **Quick Actions**
- **⚡ Apply Now**: Immediately apply current CSS
- **🔄 Reset**: Reset to default template CSS
- **📋 Copy CSS**: Copy current CSS to clipboard

## 💻 CSS Structure

The template uses CSS custom properties (variables) for easy theming:

```css
:root {
  /* Node styling */
  --node-default-color: #4dabf7;
  --node-selected-color: #ff6b6b;
  --node-hover-color: #339af0;

  /* Edge styling */
  --edge-default-color: #999;

  /* Table styling */
  --line: #eee;
  --row-hover: #f3f6ff;
  --bg: #fff;
  --fg: #1f2328;
}
```

## 🎯 Common CSS Targets

### **Graph Canvas**
```css
.graph-canvas {
  border: 2px solid #007bff;
  border-radius: 15px;
  background: linear-gradient(45deg, #f0f8ff, #e6f3ff);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
```

### **Table Styling**
```css
.groggy-table {
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.groggy-table th {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
```

### **Interactive Elements**
```css
.view-toggle-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  transition: all 0.3s ease;
}

.view-toggle-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
```

## 🔄 Workflow

### **Rapid Prototyping**
1. **Start with presets** - Load a preset close to your desired style
2. **Refine with controls** - Use GUI controls for quick color/size adjustments
3. **Custom CSS** - Switch to CSS tab for advanced styling
4. **Export** - Copy CSS and integrate into your Rust code

### **Advanced Styling**
1. **Experiment** - Try different themes and approaches
2. **Learn** - See how CSS changes affect the visualization
3. **Iterate** - Make incremental improvements
4. **Document** - Save successful CSS snippets

## 📋 Integration with Rust

### **Copy CSS Variables**
Use the "📋 Export Variables" button to generate CSS custom properties that you can integrate into your `graph_visualization.css` file.

### **Copy Full CSS**
Use the "📋 Copy CSS" button to copy the entire CSS, then:

1. **Extract key styles** from the CSS
2. **Update your external CSS files** in `src/viz/streaming/css/`
3. **Test in your application** with `cargo run`

## 🎨 Styling Tips

### **Performance**
- Use CSS transforms instead of changing layout properties
- Leverage GPU acceleration with `transform3d()` or `will-change`
- Minimize complex box-shadows on frequently updated elements

### **Consistency**
- Use CSS custom properties for color themes
- Maintain consistent spacing with CSS variables
- Follow existing naming conventions

### **Accessibility**
- Ensure sufficient color contrast
- Test with dark/light mode preferences
- Consider reduced motion preferences

## 🐛 Troubleshooting

### **CSS Not Applying**
- Check browser console for CSS syntax errors
- Try manual apply with "⚡ Apply Now"
- Reset and try again with "🔄 Reset"

### **Performance Issues**
- Complex CSS selectors can slow down rendering
- Avoid excessive use of `:hover` effects
- Test with realistic data sizes

### **Browser Compatibility**
- Modern CSS features may not work in older browsers
- Test CSS custom properties support
- Use fallback values for critical styles

## 🔗 Related Files

- **Template**: `streaming_prototype/streaming_template.html`
- **CSS Files**: `streaming_prototype/css/`
- **Generator**: `streaming_template_generator.py`
- **Dev Server**: `streaming_dev_server.py`
- **Rust Source**: `src/viz/streaming/css/`