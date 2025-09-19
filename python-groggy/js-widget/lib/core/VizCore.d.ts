export class GroggyVizCore {
    constructor(nodes?: any[], edges?: any[], config?: {});
    nodes: any[];
    edges: any[];
    config: {
        physics: any;
        rendering: any;
        interaction: any;
        width: number;
        height: number;
    };
    physics: PhysicsEngine;
    renderer: SVGRenderer;
    interactions: InteractionEngine;
    positions: Map<any, any>;
    velocities: Map<any, any>;
    isRunning: boolean;
    frameCount: number;
    selectedNodes: Set<any>;
    nodePositions: Map<any, any>;
    eventCallbacks: Map<any, any>;
    /**
     * 🎯 Single update method used by ALL visualization environments
     */
    update(frameData?: null): {
        nodes: any[];
        edges: any[];
        positions: any;
        velocities: any;
        frameCount: number;
        timestamp: number;
    };
    /**
     * Apply frame data from external source (WebSocket streaming)
     */
    applyFrameData(frameData: any): void;
    /**
     * Get current frame data for transmission
     */
    getFrameData(): {
        nodes: any[];
        edges: any[];
        positions: any;
        velocities: any;
        frameCount: number;
        timestamp: number;
    };
    /**
     * Attach to DOM element (for widgets and local display)
     */
    attachToDOM(element: any, callbacks?: {}): void;
    element: any;
    /**
     * Start physics simulation
     */
    start(): void;
    /**
     * Stop physics simulation
     */
    stop(): void;
    /**
     * Animation loop for local/widget environments
     */
    animationLoop(): void;
    /**
     * Event system
     */
    on(event: any, callback: any): void;
    emit(event: any, data: any): void;
    /**
     * Update graph data
     */
    setData(nodes: any, edges: any): void;
    /**
     * Get current configuration
     */
    getConfig(): {
        physics: any;
        rendering: any;
        interaction: any;
        width: number;
        height: number;
    };
    /**
     * Update configuration
     */
    updateConfig(newConfig: any): void;
    /**
     * Export current state for saving/serialization
     */
    exportState(): {
        nodes: any[];
        edges: any[];
        positions: any;
        velocities: any;
        selectedNodes: any[];
        config: {
            physics: any;
            rendering: any;
            interaction: any;
            width: number;
            height: number;
        };
        frameCount: number;
        timestamp: number;
    };
    /**
     * Clean up and destroy the visualization
     */
    destroy(): void;
}
export default GroggyVizCore;
import { PhysicsEngine } from "./PhysicsEngine.js";
import { SVGRenderer } from "./SVGRenderer.js";
import { InteractionEngine } from "./InteractionEngine.js";
