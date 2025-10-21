# 🎨 Complete CSS Reference for Streaming Visualizer

This document lists ALL the CSS classes and styling options available in the streaming template system.

## 📁 CSS File Structure

### `css/sleek.css`
**Base table styling** - Core table appearance and responsive design
- 🎯 **Target**: Table display, data formatting, responsive behavior

### `css/graph_visualization.css`
**Everything else** - Graph canvas, controls, template UI, style panel
- 🎯 **Target**: Graph visualization, interactive controls, editor interface

## 🎨 All Available CSS Classes

### **📊 Table & Data Display**
```css
/* Container */
.groggy-display-container    /* Main table wrapper */
.groggy-table-container      /* Table container */
.groggy-table               /* Main table element */

/* Table structure */
.groggy-table th            /* Table headers */
.groggy-table td            /* Table cells */
.groggy-table tbody tr      /* Table rows */

/* Cell types */
.cell-num                   /* Numeric cells - right aligned */
.cell-bool                  /* Boolean cells - center aligned */
.cell-str                   /* String cells - left aligned */
.cell-time                  /* Time/date cells */
.cell-ellipsis              /* Truncated content with ... */
.cell-truncated             /* Truncated indicator */

/* Table states */
.is-selected               /* Selected row styling */
[data-col-priority="high"] /* High priority columns */
[data-col-priority="med"]  /* Medium priority columns */
[data-col-priority="low"]  /* Low priority columns (hidden on mobile) */

/* Table info */
.table-info                /* Info bar below table */
.interactive-btn           /* "View All" button */
```

### **🕸️ Graph Visualization**
```css
/* Canvas */
.graph-canvas              /* Main graph canvas element */
.graph-canvas:hover        /* Canvas hover state */
.graph-canvas:focus        /* Canvas focus state */
.graph-canvas.is-loading   /* Loading animation */
.graph-canvas.has-data     /* When graph has data */
.graph-canvas.is-dragging  /* During drag operations */
.graph-canvas.is-zoomed    /* When zoomed */
.graph-canvas.selection-mode /* Selection mode styling */

/* Controls */
.graph-controls            /* Graph control buttons container */
.graph-btn                 /* Individual control buttons */
.graph-btn:hover           /* Button hover state */

.layout-controls           /* Layout control container */
.layout-select             /* Layout dropdown */
```

### **🎛️ Interface Controls**
```css
/* Main navigation */
.table-container           /* Overall container */
.table-header              /* Header with title and controls */
.table-title               /* "Interactive Visualization" title */
.table-stats               /* Row/column count display */

/* View switching */
.view-controls             /* View toggle container */
.view-toggle               /* Toggle button group */
.view-toggle-btn           /* Individual toggle buttons */
.view-toggle-btn.active    /* Active toggle button */

/* Views */
.viz-container             /* Main visualization container */
.table-view               /* Table view container */
.graph-view               /* Graph view container */

/* Connection status */
.connection-status         /* Status indicator */
.status-connected          /* Connected state */
.status-disconnected       /* Disconnected state */
```

### **🎨 Template-Specific UI**
```css
/* Template controls */
.prototype-controls        /* Top control bar */
.control-group            /* Individual control groups */
.control-group label      /* Control labels */
.control-group select     /* Control dropdowns */
.control-group input      /* Control inputs */

/* Style panel */
.style-panel              /* Floating style editor */
.style-panel.active       /* When panel is open */

/* Panel structure */
.panel-header             /* Panel header area */
.panel-tabs               /* Tab navigation */
.panel-tab                /* Individual tabs */
.panel-tab.active         /* Active tab */
.panel-content            /* Tab content area */

/* Tab panes */
.tab-pane                 /* Tab content panels */
.tab-pane.active          /* Active tab panel */

/* Style controls */
.style-section            /* Control sections */
.style-section h4         /* Section headings */
.style-control            /* Individual controls */
.style-control label      /* Control labels */
.style-control input      /* Control inputs */
```

### **💻 CSS Editor**
```css
/* Editor */
.css-editor               /* Main CSS textarea */
.css-editor:focus         /* Editor focus state */

/* Actions */
.css-actions              /* Button container */
.css-btn                  /* Action buttons */
.css-btn:hover            /* Button hover */
.css-btn.secondary        /* Secondary buttons */

/* Presets */
.css-presets              /* Preset section */
.preset-grid              /* Preset button grid */
.preset-btn               /* Individual preset buttons */
.preset-btn:hover         /* Preset hover state */

/* Status */
.css-status               /* Status messages */
.css-status.error         /* Error messages */
```

## 🎨 CSS Custom Properties (Variables)

### **Core Variables** (from sleek.css)
```css
:root {
  --radius: 10px;              /* Border radius */
  --border: 2px;               /* Border width */
  --bg: #fff;                  /* Background color */
  --fg: #1f2328;               /* Foreground text color */
  --muted: #6a737d;            /* Muted text color */
  --line: #eee;                /* Border/line color */
  --accent: #3267e3;           /* Accent color */
  --hover: #f6f8fa;            /* Hover background */
  --row-hover: #f3f6ff;        /* Table row hover */
  --selected: #e8f0ff;         /* Selection background */
  --shadow: 0 1px 2px rgba(0,0,0,.05), 0 2px 8px rgba(0,0,0,.06);
  --cell-py: 5px;              /* Cell vertical padding */
  --cell-px: 12px;             /* Cell horizontal padding */
  --font: system-ui, -apple-system, sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}
```

### **Graph Variables** (from graph_visualization.css)
```css
:root {
  /* Node styling */
  --node-default-radius: 8px;
  --node-large-radius: 12px;
  --node-small-radius: 5px;
  --node-default-color: #4dabf7;
  --node-selected-color: #ff6b6b;
  --node-hover-color: #339af0;
  --node-border-color: #333;
  --node-border-width: 2px;
  --node-border-hover-width: 3px;
  --node-label-color: #333;
  --node-label-font: 10px Arial;
  --node-label-offset: 12px;

  /* Edge styling */
  --edge-default-color: #999;
  --edge-selected-color: #ff8cc8;
  --edge-hover-color: #666;
  --edge-default-width: 1px;
  --edge-thick-width: 2px;
  --edge-selected-width: 3px;
  --edge-opacity: 0.8;
  --edge-hover-opacity: 1.0;

  /* Graph interaction colors */
  --graph-highlight-color: #ffd43b;
  --graph-shadow-color: rgba(0,0,0,0.2);
  --graph-selection-box-color: rgba(50, 103, 227, 0.2);
  --graph-selection-box-border: #3267e3;

  /* Node type colors */
  --node-type-primary: #4dabf7;
  --node-type-secondary: #69db7c;
  --node-type-warning: #ffd43b;
  --node-type-danger: #ff6b6b;
  --node-type-info: #74c0fc;
  --node-type-success: #51cf66;

  /* Edge type colors */
  --edge-type-default: #999;
  --edge-type-strong: #495057;
  --edge-type-weak: #ced4da;
  --edge-type-directed: #fd7e14;
  --edge-type-bidirectional: #7950f2;
}
```

## 🎯 Quick Styling Examples

### **Change Table Theme**
```css
:root {
  --bg: #1a1a1a;              /* Dark background */
  --fg: #e2e8f0;              /* Light text */
  --line: #4a5568;            /* Dark borders */
  --row-hover: #2a4365;       /* Dark hover */
}
```

### **Customize Graph Nodes**
```css
:root {
  --node-default-color: #ff6b6b;    /* Red nodes */
  --node-default-radius: 12px;      /* Bigger nodes */
  --node-border-width: 3px;         /* Thicker borders */
}
```

### **Style the Canvas**
```css
.graph-canvas {
  border: 3px solid #ff6b6b;
  border-radius: 20px;
  background: linear-gradient(45deg, #fbb6ce, #fed7d7);
  box-shadow: 0 8px 16px rgba(255, 107, 107, 0.3);
}
```

### **Customize Control Panel**
```css
.style-panel {
  width: 500px;                     /* Wider panel */
  background: #f8f9fa;             /* Gray background */
  border: 2px solid #007bff;       /* Blue border */
}

.panel-tab.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

### **Theme the Editor**
```css
.css-editor {
  background: #2d3748;             /* Dark editor */
  color: #e2e8f0;                  /* Light text */
  border: 2px solid #4a5568;      /* Dark border */
  font-size: 14px;                /* Bigger font */
}

.css-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 8px;
  padding: 8px 16px;
}
```

## 🔧 Advanced Customization

### **Responsive Design**
```css
/* Mobile-first approach */
@media (max-width: 820px) {
  [data-col-priority="low"] { display: none; }
  .style-panel { width: 90vw; right: 5vw; }
}

@media (max-width: 560px) {
  [data-col-priority="med"] { display: none; }
  .prototype-controls { flex-direction: column; }
}
```

### **Animation & Transitions**
```css
/* Smooth animations */
.graph-canvas {
  transition: all 0.3s ease;
}

.view-toggle-btn {
  transition: all 0.2s ease;
  transform: translateY(0);
}

.view-toggle-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
```

### **Custom Node Types**
```css
/* Add support for new node types */
:root {
  --node-type-database: #38b2ac;
  --node-type-api: #ed8936;
  --node-type-user: #9f7aea;
}
```

## 🚀 Integration Workflow

1. **Edit CSS files** in `streaming_prototype/css/`
2. **Test changes** in the template browser
3. **Copy successful styles** to your Rust source: `src/viz/streaming/css/`
4. **Rebuild application** with `cargo run`

This gives you **complete control** over every visual aspect of the streaming visualizer!