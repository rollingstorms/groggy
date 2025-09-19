/**
 * Elegant Jupyter Notebook Extension Entry Point
 * 
 * Minimal, beautiful registration of our widget for classic Jupyter notebooks.
 */

import { GroggyGraphModel, GroggyGraphView } from './widget';

// AMD module type declarations
declare const define: any;
declare const require: any;
declare const module: any;

/**
 * Elegant extension loading for Jupyter Notebook
 */
export function load_ipython_extension(): void {
    // Register our elegant widgets
    const widgets = require('@jupyter-widgets/base');
    
    widgets.registerWidget({
        model_name: 'GroggyGraphModel',
        model_module: 'groggy-widget',
        model_module_version: '0.1.0',
        view_name: 'GroggyGraphView', 
        view_module: 'groggy-widget',
        view_module_version: '0.1.0',
        model: GroggyGraphModel,
        view: GroggyGraphView
    });
    
    console.log('✨ Elegant Groggy widget extension loaded for Jupyter Notebook');
}

// Elegant AMD module definition
(function(root: any, factory: any) {
    if (typeof define === 'function' && define.amd) {
        // AMD
        define(['@jupyter-widgets/base'], factory);
    } else if (typeof module === 'object' && module.exports) {
        // CommonJS
        module.exports = factory(require('@jupyter-widgets/base'));
    } else {
        // Browser globals
        root.groggyWidget = factory(root.widgets);
    }
}(typeof self !== 'undefined' ? self : this, function(widgets: any) {
    
    return {
        load_ipython_extension,
        GroggyGraphModel,
        GroggyGraphView
    };
    
}));