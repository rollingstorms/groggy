/**
 * 🧠 Unified Jupyter Widget using GroggyVizCore
 *
 * Refactored to use the unified JavaScript core for consistency
 * across all visualization environments. This replaces the original
 * widget.ts with a version that leverages the unified architecture.
 */
import { DOMWidgetModel, DOMWidgetView } from '@jupyter-widgets/base';
export declare const MODULE_NAME = "groggy-widgets";
export declare const MODULE_VERSION = "0.1.0";
export declare class GroggyGraphModel extends DOMWidgetModel {
    static model_name: string;
    static model_module: string;
    static model_module_version: string;
    defaults(): {
        _model_name: string;
        _model_module: string;
        _model_module_version: string;
        _view_name: string;
        _view_module: string;
        _view_module_version: string;
        nodes: never[];
        edges: never[];
        title: string;
        width: number;
        height: number;
        layout_algorithm: string;
        theme: string;
        physics_enabled: boolean;
        force_strength: number;
        link_distance: number;
        link_strength: number;
        charge_strength: number;
        center_strength: number;
        enable_drag: boolean;
        enable_zoom: boolean;
        enable_pan: boolean;
        enable_selection: boolean;
        enable_hover: boolean;
        selected_nodes: never[];
        hovered_node: string;
        camera_position: {
            scale: number;
            translate: {
                x: number;
                y: number;
            };
        };
        is_dragging: boolean;
        node_color_scheme: string;
        background_color: string;
        enable_shadows: boolean;
        enable_animations: boolean;
    };
}
export declare class GroggyGraphView extends DOMWidgetView {
    static view_name: string;
    static view_module: string;
    static view_module_version: string;
    private vizCore;
    private container;
    render(): void;
    /**
     * 🧠 Initialize the unified visualization core
     */
    private initializeVizCore;
    /**
     * 🎧 Set up event handlers for core → Jupyter synchronization
     */
    private setupCoreEventHandlers;
    /**
     * 🔄 Set up model change listeners for Jupyter → core synchronization
     */
    private setupModelListeners;
    /**
     * 📊 Update graph data
     */
    private updateData;
    /**
     * 📐 Update canvas dimensions
     */
    private updateDimensions;
    /**
     * ⚛️ Update physics configuration
     */
    private updatePhysicsConfig;
    /**
     * 🎨 Update rendering configuration
     */
    private updateRenderingConfig;
    /**
     * 🖱️ Update interaction configuration
     */
    private updateInteractionConfig;
    /**
     * 📊 Update status display
     */
    private updateStatus;
    /**
     * 📨 Handle custom messages from Python
     */
    handleCustomMessage(content: any): void;
    /**
     * 🧹 Cleanup when widget is destroyed
     */
    remove(): void;
}
