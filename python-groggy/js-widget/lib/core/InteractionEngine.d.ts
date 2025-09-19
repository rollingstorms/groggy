/**
 * 🖱️ Unified Interaction Engine
 *
 * Handles mouse, touch, and keyboard interactions consistently
 * across all visualization environments
 */
export class InteractionEngine {
    constructor(config?: {});
    config: {
        enableDrag: boolean;
        enableZoom: boolean;
        enablePan: boolean;
        enableMultiTouch: boolean;
        dragThreshold: number;
        zoomSensitivity: number;
        panSensitivity: number;
    };
    element: any;
    callbacks: Map<any, any>;
    state: {
        isDragging: boolean;
        isPanning: boolean;
        draggedNode: null;
        lastMousePos: {
            x: number;
            y: number;
        };
        dragStartPos: {
            x: number;
            y: number;
        };
        zoom: number;
        pan: {
            x: number;
            y: number;
        };
    };
    listeners: Map<any, any>;
    /**
     * Attach interaction engine to DOM element
     */
    attachToElement(element: any, callbacks?: {}): void;
    /**
     * Setup event listeners
     */
    setupEventListeners(): void;
    /**
     * Add event listener and track it
     */
    addListener(event: any, handler: any): void;
    /**
     * Handle mouse down events
     */
    handleMouseDown(event: any): void;
    /**
     * Handle mouse move events
     */
    handleMouseMove(event: any): void;
    /**
     * Handle mouse up events
     */
    handleMouseUp(event: any): void;
    /**
     * Handle wheel events for zooming
     */
    handleWheel(event: any): void;
    /**
     * Handle touch start events
     */
    handleTouchStart(event: any): void;
    /**
     * Handle touch move events
     */
    handleTouchMove(event: any): void;
    /**
     * Handle touch end events
     */
    handleTouchEnd(event: any): void;
    /**
     * Handle keyboard events
     */
    handleKeyDown(event: any): void;
    /**
     * Reset view to default state
     */
    resetView(): void;
    /**
     * Process frame (called from main update loop)
     */
    processFrame(): void;
    /**
     * Emit event to callbacks
     */
    emit(event: any, data: any): void;
    /**
     * Update configuration
     */
    updateConfig(newConfig: any): void;
    /**
     * Get current interaction state
     */
    getState(): {
        isDragging: boolean;
        isPanning: boolean;
        draggedNode: null;
        lastMousePos: {
            x: number;
            y: number;
        };
        dragStartPos: {
            x: number;
            y: number;
        };
        zoom: number;
        pan: {
            x: number;
            y: number;
        };
    };
    /**
     * Set interaction state
     */
    setState(newState: any): void;
    /**
     * Clean up event listeners
     */
    cleanup(): void;
}
export default InteractionEngine;
