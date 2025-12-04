"""
Elegant Widget Loader - Production-Ready JavaScript Loading

This module handles JavaScript loading exactly like Plotly and other production widgets:
1. Try CDN first (when available)
2. Fallback to bundled local files
3. Automatic registration with Jupyter
"""

import json
import os
from pathlib import Path

from IPython.display import HTML, Javascript, display


def get_widget_js_path():
    """Get path to bundled widget JavaScript file."""
    current_dir = Path(__file__).parent
    static_dir = current_dir.parent / "static"
    js_path = static_dir / "groggy-widget.js"
    return js_path


def get_widget_js_content():
    """Get widget JavaScript content for inline loading."""
    js_path = get_widget_js_path()
    if js_path.exists():
        with open(js_path, "r") as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Widget JavaScript not found at {js_path}")


def get_widget_js_url():
    """Get URL for widget JavaScript (CDN when available, local fallback)."""
    # Future CDN URL (when we set it up)
    cdn_url = None  # "https://cdn.groggy.dev/widgets/groggy-widget-0.1.0.js"

    if cdn_url:
        return cdn_url
    else:
        # For now, we'll inline the JavaScript since file serving is complex
        return None


def load_widget_js():
    """Load widget JavaScript in Jupyter environment (automatic registration)."""
    try:
        # Early-exit if the federated plugin is active or module is already resolvable
        # FIXED: Wrap in an IIFE to allow return statement
        js_code_with_guard = """
        // Early-exit if the federated plugin is active or module is already resolvable
        (function() {
          try {
            const labHasPlugin =
              !!(window.jupyterapp && window.jupyterapp._pluginMap &&
                 [...window.jupyterapp._pluginMap.keys()].some(k => k.includes('groggy-widgets:plugin')));

            const moduleIsDefined =
              (window.requirejs && window.requirejs.defined && window.requirejs.defined('groggy-widgets')) ||
              (window.require   && window.require.defined    && window.require.defined('groggy-widgets'));

            if (labHasPlugin || moduleIsDefined) {
              console.log('🧹 Skipping fallback loader: federated groggy-widgets is present.');
              // IMPORTANT: do not patch loadClass or define AMD here
              return; // Skip all fallback registration
            }
          } catch(e) {/* swallow */}
        
        // Continue with widget registration if we didn't return early
        """

        # Get the widget JavaScript content
        widget_js_content = get_widget_js_content()

        # Create direct widget manager registration that bypasses RequireJS
        js_code = (
            js_code_with_guard
            + """
        // Groggy Widget Registration - Direct Widget Manager approach  
        (function() {{
            console.log('🎨 Setting up Groggy widget registration...');
            
            var MODULE_NAME = 'groggy-widgets';
            var MODULE_VERSION = '^0.1.0';
            
            // Function to create our widget classes with proper base classes
            function createWidgetClasses(BaseModel, BaseView) {{
                // Create GroggyGraphModel
                class GroggyGraphModel extends BaseModel {{
                    defaults() {{
                        return Object.assign(super.defaults(), {{
                            _model_name: 'GroggyGraphModel',
                            _model_module: MODULE_NAME,
                            _model_module_version: MODULE_VERSION,
                            _view_name: 'GroggyGraphView',
                            _view_module: MODULE_NAME,
                            _view_module_version: MODULE_VERSION,
                            nodes: [],
                            edges: [],
                            layout_algorithm: 'force-directed',
                            theme: 'light',
                            width: 800,
                            height: 600,
                            title: 'Graph Visualization'
                        }});
                    }}
                }}
                
                // Create GroggyGraphView
                class GroggyGraphView extends BaseView {{
                    render() {{
                        this.el.className = 'groggy-widget-container';
                        this.el.style.cssText = `
                            border: 1px solid #ddd; border-radius: 4px; background: #fafafa;
                            min-height: 400px; width: 100%; display: flex;
                            align-items: center; justify-content: center; position: relative;
                        `;
                        
                        const nodeCount = (this.model.get('nodes') || []).length;
                        const edgeCount = (this.model.get('edges') || []).length;
                        const layout = this.model.get('layout_algorithm') || 'force-directed';
                        
                        this.el.innerHTML = `
                            <div style="text-align: center; color: #666; padding: 20px;">
                                <h3 style="margin: 0 0 15px 0; color: #333;">🎨 Groggy Graph Widget</h3>
                                <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 10px;">
                                    <div><strong>Nodes:</strong> ${{nodeCount}}</div>
                                    <div><strong>Edges:</strong> ${{edgeCount}}</div>
                                </div>
                                <div><strong>Layout:</strong> ${{layout}}</div>
                                <div style="margin-top: 15px; padding: 10px; background: #e8f4f8; border-radius: 3px; font-size: 0.9em;">
                                    ✅ Widget loaded successfully in Jupyter!
                                </div>
                            </div>
                        `;
                        
                        console.log('✅ Groggy widget rendered successfully');
                    }}
                }}
                
                return {{ GroggyGraphModel, GroggyGraphView }};
            }}
            
            // Direct widget manager registration approach
            function registerWithWidgetManager() {{
                console.log('🎯 Attempting direct widget manager registration...');
                
                // Look for Jupyter's widget manager in various locations
                var widgetManager = null;
                var baseWidgets = null;
                
                // Method 1: Try to find active widget manager
                try {{
                    if (window.Jupyter && window.Jupyter.notebook && window.Jupyter.notebook.kernel) {{
                        widgetManager = window.Jupyter.notebook.kernel.widget_manager;
                        console.log('📦 Found classic Jupyter widget manager');
                    }}
                }} catch(e) {{ /* ignore */ }}
                
                // Method 2: Try JupyterLab widget manager
                if (!widgetManager) {{
                    try {{
                        // Look for JupyterLab widget managers in the global scope
                        for (let prop in window) {{
                            if (prop.includes('widget') && window[prop] && window[prop]._managers) {{
                                widgetManager = window[prop];
                                console.log('📦 Found JupyterLab widget manager');
                                break;
                            }}
                        }}
                    }} catch(e) {{ /* ignore */ }}
                }}
                
                // Method 3: Get base module through require (check both contexts)
                try {{
                    if (window.requirejs && window.requirejs.defined && window.requirejs.defined('@jupyter-widgets/base')) {{
                        baseWidgets = window.requirejs('@jupyter-widgets/base');
                        console.log('📦 Got @jupyter-widgets/base through requirejs');
                    }} else if (window.require && window.require.defined && window.require.defined('@jupyter-widgets/base')) {{
                        baseWidgets = window.require('@jupyter-widgets/base');
                        console.log('📦 Got @jupyter-widgets/base through require');
                    }}
                }} catch(e) {{ /* ignore */ }}
                
                // Method 4: Try async require for base widgets
                if (!baseWidgets && window.require) {{
                    window.require(['@jupyter-widgets/base'], function(widgets) {{
                        console.log('📦 Async loaded @jupyter-widgets/base');
                        baseWidgets = widgets;
                        proceedWithRegistration();
                    }}, function(err) {{
                        console.warn('⚠️ Failed to load @jupyter-widgets/base:', err);
                        createFallbackRegistration();
                    }});
                    return; // Wait for async callback
                }}
                
                proceedWithRegistration();
                
                function proceedWithRegistration() {{
                    if (!baseWidgets) {{
                        console.warn('⚠️ No base widgets found, using fallback');
                        createFallbackRegistration();
                        return;
                    }}
                    
                    const BaseModel = baseWidgets.DOMWidgetModel || baseWidgets.WidgetModel;
                    const BaseView = baseWidgets.DOMWidgetView || baseWidgets.WidgetView;
                    
                    if (!BaseModel || !BaseView) {{
                        console.error('❌ Could not find base widget classes');
                        createFallbackRegistration();
                        return;
                    }}
                    
                    console.log('✅ Found base classes:', BaseModel.name, BaseView.name);
                    
                    // Create our widget classes
                    const {{ GroggyGraphModel, GroggyGraphView }} = createWidgetClasses(BaseModel, BaseView);
                    
                    // Set static properties
                    GroggyGraphModel.model_name = 'GroggyGraphModel';
                    GroggyGraphModel.model_module = MODULE_NAME;
                    GroggyGraphModel.model_module_version = MODULE_VERSION;
                    GroggyGraphModel.view_name = 'GroggyGraphView';
                    GroggyGraphModel.view_module = MODULE_NAME;
                    GroggyGraphModel.view_module_version = MODULE_VERSION;
                    
                    GroggyGraphView.view_name = 'GroggyGraphView';
                    GroggyGraphView.view_module = MODULE_NAME;
                    GroggyGraphView.view_module_version = MODULE_VERSION;
                    
                    // CRITICAL: Register directly with widget manager
                    if (widgetManager) {{
                        // Try to register our module directly in the widget manager
                        try {{
                            if (widgetManager._model_types) {{
                                widgetManager._model_types[MODULE_NAME + ':GroggyGraphModel'] = GroggyGraphModel;
                                console.log('✅ Registered model with widget manager _model_types');
                            }}
                            if (widgetManager._view_types) {{
                                widgetManager._view_types[MODULE_NAME + ':GroggyGraphView'] = GroggyGraphView;
                                console.log('✅ Registered view with widget manager _view_types');
                            }}
                            
                            // Also try the loadClass override approach
                            if (widgetManager.loadClass) {{
                                // Check if federated extension is handling this
                                const labHasPlugin =
                                    !!(window.jupyterapp && window.jupyterapp._pluginMap &&
                                       [...window.jupyterapp._pluginMap.keys()].some(k => k.includes('groggy-widgets:plugin')));
                                
                                if (!labHasPlugin) {{
                                    const originalLoadClass = widgetManager.loadClass;
                                    widgetManager.loadClass = function(className, moduleName, moduleVersion) {{
                                        console.log('🔍 loadClass called:', className, moduleName, moduleVersion);
                                        if (moduleName === MODULE_NAME) {{
                                            if (className === 'GroggyGraphModel') {{
                                                console.log('✅ Returning GroggyGraphModel');
                                                return Promise.resolve(GroggyGraphModel);
                                            }} else if (className === 'GroggyGraphView') {{
                                                console.log('✅ Returning GroggyGraphView');
                                                return Promise.resolve(GroggyGraphView);
                                            }}
                                        }}
                                        return originalLoadClass.call(this, className, moduleName, moduleVersion);
                                    }};
                                    console.log('✅ Patched widget manager loadClass');
                                }} else {{
                                    console.log('🧹 Skipping loadClass patch: federated plugin active');
                                }}
                            }}
                        }} catch(err) {{
                            console.warn('⚠️ Widget manager registration failed:', err);
                        }}
                    }}
                    
                    // BULLETPROOF: Register in all require contexts + force-load
                    (function registerAll() {{
                        const moduleVersion = MODULE_VERSION.replace(/^[^\\d]*/, ''); // "^0.1.0" -> "0.1.0"
                        const payload = {{
                            version: moduleVersion,
                            GroggyGraphModel,
                            GroggyGraphView
                        }};
                        
                        console.log('📦 Bulletproof registration with version:', moduleVersion);
                        
                        // Helper: define into a given require-like object
                        function defineInto(rjs, label) {{
                            try {{
                                if (rjs && typeof rjs.define === 'function') {{
                                    rjs.define(MODULE_NAME, ['@jupyter-widgets/base'], function(base) {{
                                        console.log('📦 AMD factory for groggy-widgets called via', label);
                                        return payload;
                                    }});
                                    console.log('✅ AMD module defined in', label, 'with version:', moduleVersion);
                                    
                                    // Force-load immediately so the widget manager can resolve it right now
                                    if (typeof rjs === 'function') {{
                                        rjs([MODULE_NAME], function(mod) {{
                                            console.log('🚚 Module loaded from', label, '->', mod && mod.version);
                                        }});
                                    }} else if (typeof rjs.require === 'function') {{
                                        rjs.require([MODULE_NAME], function(mod) {{
                                            console.log('🚚 Module loaded from', label, '->', mod && mod.version);
                                        }});
                                    }}
                                }}
                            }} catch (e) {{
                                console.warn('⚠️ defineInto failed for', label, e);
                            }}
                        }}
                        
                        // Prefer requirejs; also register into window.require if present
                        defineInto(window.requirejs || window.require, window.requirejs ? 'requirejs' : 'require');
                        if (window.requirejs && window.require) {{
                            // If both exist, register into the other context too
                            defineInto(window.require === (window.requirejs) ? null : window.require, 'require');
                        }}
                        
                        // Also try global define if available
                        if (typeof define !== 'undefined' && define.amd) {{
                            try {{
                                define(MODULE_NAME, ['@jupyter-widgets/base'], function(base) {{
                                    console.log('📦 Global AMD define for groggy-widgets called');
                                    return payload;
                                }});
                                console.log('✅ Global AMD module defined with version:', moduleVersion);
                            }} catch(e) {{
                                console.warn('⚠️ Global define failed:', e);
                            }}
                        }}
                        
                        // If the active widget manager has a registry, add there too
                        try {{
                            if (widgetManager && widgetManager.registry && widgetManager.registry.set) {{
                                widgetManager.registry.set(MODULE_NAME, moduleVersion, payload);
                                console.log('✅ Registered module via widgetManager.registry');
                            }}
                        }} catch (e) {{
                            console.warn('⚠️ widgetManager.registry.set failed:', e);
                        }}
                        
                        console.log('🚀 BULLETPROOF REGISTRATION COMPLETE!');
                    }})();
                    
                    // Register globally for debugging
                    window.GroggyGraphModel = GroggyGraphModel;
                    window.GroggyGraphView = GroggyGraphView;
                    window._groggyWidgetModule = {{ GroggyGraphModel, GroggyGraphView }};
                    
                    console.log('🚀 REGISTRATION COMPLETE!');
                    console.log('📋 Available classes:', typeof GroggyGraphModel, typeof GroggyGraphView);
                }}
            }}
            
            // Start the registration process
            registerWithWidgetManager();
            
            // Fallback registration for environments without proper require
            function createFallbackRegistration() {{
                console.log('📦 Setting up fallback widget registration...');
                
                // Create minimal base classes
                class MinimalWidgetModel {{
                    constructor(attributes, options) {{
                        this.attributes = attributes || {{}};
                        this.options = options || {{}};
                    }}
                    defaults() {{ return {{}}; }}
                    get(key) {{ return this.attributes[key]; }}
                    set(key, value) {{ this.attributes[key] = value; }}
                    save_changes() {{}}
                    send(content) {{}}
                }}
                
                class MinimalWidgetView {{
                    constructor(options) {{
                        this.options = options || {{}};
                        this.el = document.createElement('div');
                        this.model = options?.model;
                    }}
                    render() {{}}
                    remove() {{}}
                }}
                
                // Create our classes with minimal bases
                const {{ GroggyGraphModel, GroggyGraphView }} = createWidgetClasses(MinimalWidgetModel, MinimalWidgetView);
                
                // Set the static properties
                GroggyGraphModel.model_name = 'GroggyGraphModel';
                GroggyGraphModel.model_module = MODULE_NAME;
                GroggyGraphModel.model_module_version = MODULE_VERSION;
                GroggyGraphView.view_name = 'GroggyGraphView';
                GroggyGraphView.view_module = MODULE_NAME;
                GroggyGraphView.view_module_version = MODULE_VERSION;
                
                // BULLETPROOF FALLBACK: Register in all contexts
                (function fallbackRegisterAll() {{
                    const moduleVersion = MODULE_VERSION.replace(/^[^\\d]*/, ''); // "^0.1.0" -> "0.1.0"
                    const payload = {{
                        version: moduleVersion,
                        GroggyGraphModel, 
                        GroggyGraphView
                    }};
                    
                    console.log('📦 Bulletproof fallback registration with version:', moduleVersion);
                    
                    // Helper: define into a given require-like object
                    function defineInto(rjs, label) {{
                        try {{
                            if (rjs && typeof rjs.define === 'function') {{
                                rjs.define(MODULE_NAME, function() {{
                                    console.log('📦 Fallback AMD factory for groggy-widgets called via', label);
                                    return payload;
                                }});
                                console.log('✅ Fallback AMD module defined in', label, 'with version:', moduleVersion);
                                
                                // Force-load immediately
                                if (typeof rjs === 'function') {{
                                    rjs([MODULE_NAME], function(mod) {{
                                        console.log('🚚 Fallback module loaded from', label, '->', mod && mod.version);
                                    }});
                                }} else if (typeof rjs.require === 'function') {{
                                    rjs.require([MODULE_NAME], function(mod) {{
                                        console.log('🚚 Fallback module loaded from', label, '->', mod && mod.version);
                                    }});
                                }}
                            }}
                        }} catch (e) {{
                            console.warn('⚠️ Fallback defineInto failed for', label, e);
                        }}
                    }}
                    
                    // Register in all possible contexts
                    defineInto(window.requirejs || window.require, window.requirejs ? 'requirejs' : 'require');
                    if (window.requirejs && window.require && window.require !== window.requirejs) {{
                        defineInto(window.require, 'require');
                    }}
                    
                    // Also try global define
                    if (typeof define !== 'undefined' && define.amd) {{
                        try {{
                            define(MODULE_NAME, function() {{
                                console.log('📦 Fallback global AMD define for groggy-widgets');
                                return payload;
                            }});
                            console.log('✅ Fallback global AMD module defined with version:', moduleVersion);
                        }} catch(e) {{
                            console.warn('⚠️ Fallback global define failed:', e);
                        }}
                    }}
                }})();
                
                // Register globally for debugging
                window.GroggyGraphModel = GroggyGraphModel;
                window.GroggyGraphView = GroggyGraphView;
                window._groggyWidgetModule = {{ GroggyGraphModel, GroggyGraphView }};
                
                console.log('✅ Fallback registration complete with module definition');
            }}
            
        }})();
        })(); // Close the IIFE wrapper from early-exit guard
        """
        )

        # Display the JavaScript
        display(Javascript(js_code))

        return True

    except Exception as e:
        print(f"Warning: Failed to load widget JavaScript: {e}")
        return False


def ensure_widget_loaded():
    """Ensure widget is loaded before creating widget instances."""
    # This will be called automatically when widget is imported
    success = load_widget_js()
    if not success:
        print("⚠️  Widget JavaScript not loaded. Widget may not display correctly.")
    return success


# Auto-load when module is imported (like Plotly does)
_widget_loaded = False


def auto_load_widget():
    """Automatically load widget when in Jupyter environment."""
    global _widget_loaded

    if _widget_loaded:
        return True

    try:
        # Check if we're in a Jupyter environment
        from IPython import get_ipython

        ipython = get_ipython()

        if ipython is not None:
            # We're in IPython/Jupyter
            _widget_loaded = ensure_widget_loaded()
            return _widget_loaded
    except ImportError:
        # Not in Jupyter environment
        pass

    return False


# Call auto-load when module is imported
auto_load_widget()
