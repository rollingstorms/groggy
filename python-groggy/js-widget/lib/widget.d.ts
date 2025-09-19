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
    };
}
export declare class GroggyGraphView extends DOMWidgetView {
    static view_name: string;
    static view_module: string;
    static view_module_version: string;
    private nodePositions;
    private nodeVelocities;
    private isDragging;
    private draggedNode;
    private dragOffset;
    private animationId;
    private simulation;
    render(): void;
    private renderGraph;
    private addDragBehavior;
    private updateEdges;
    private getNodeColor;
    private onModelChange;
    /**
     * Start force-directed simulation for node positions
     */
    private startForceSimulation;
}
