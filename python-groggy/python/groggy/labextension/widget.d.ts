/**
 * Elegant Jupyter Widget Implementation
 *
 * Pure, minimal wrapper around our core visualization engine.
 * Philosophy: Thin interface, maximum elegance, zero duplication.
 */
import { DOMWidgetModel, DOMWidgetView, ISerializers } from '@jupyter-widgets/base';
export declare const MODULE_NAME = "groggy-widgets";
export declare const MODULE_VERSION = "0.1.0";
/**
 * Elegant Graph Model - Pure data synchronization
 */
export declare class GroggyGraphModel extends DOMWidgetModel {
    static model_name: string;
    static model_module: string;
    static model_module_version: string;
    static view_name: string;
    static view_module: string;
    static view_module_version: string;
    static serializers: ISerializers;
    defaults(): {
        _model_name: string;
        _model_module: string;
        _model_module_version: string;
        _view_name: string;
        _view_module: string;
        _view_module_version: string;
        nodes: never[];
        edges: never[];
        node_positions: {};
        layout_algorithm: string;
        theme: string;
        width: number;
        height: number;
        title: string;
        selected_nodes: never[];
        hovered_node: string;
        camera_position: {
            x: number;
            y: number;
            zoom: number;
        };
        is_dragging: boolean;
        enable_drag: boolean;
        enable_pan: boolean;
        enable_zoom: boolean;
        enable_animations: boolean;
        animation_duration: number;
        style_config: {};
    };
}
/**
 * Elegant Graph View - Pure visual interface
 */
export declare class GroggyGraphView extends DOMWidgetView {
    static view_name: string;
    static view_module: string;
    static view_module_version: string;
    private groggyView;
    /**
     * Elegant rendering - minimal setup, maximum power
     */
    render(): void;
    /**
     * Elegant custom message handling
     */
    private handleCustomMessage;
    /**
     * Elegant cleanup
     */
    remove(): void;
    /**
     * Elegant resize handling
     */
    onResize(): void;
}
