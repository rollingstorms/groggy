/**
 * Elegant Core Bridge - Single Source of Truth
 *
 * This module provides the elegant abstraction layer that bridges
 * our existing GroggyGraphView with Jupyter's widget system.
 *
 * Philosophy: Extend, don't duplicate. One codebase, multiple interfaces.
 */
/**
 * Enhanced GroggyGraphView with Jupyter widget synchronization.
 *
 * This elegant wrapper extends our core visualization engine with
 * bidirectional communication capabilities for Jupyter widgets.
 */
export declare class JupyterGroggyView {
    private coreView;
    private model;
    private element;
    private syncCallbacks;
    constructor(element: HTMLElement, model: any);
    /**
     * Extract core configuration from Jupyter model traits.
     * Elegant transformation: Jupyter traits → Core config
     */
    private extractCoreConfig;
    /**
     * Elegant bidirectional synchronization setup.
     * Core events → Jupyter model updates
     * Jupyter model changes → Core engine updates
     */
    private setupElegantSync;
    /**
     * Sync core engine events to Jupyter model (Core → Jupyter)
     */
    private syncFromCore;
    /**
     * Sync Jupyter model changes to core engine (Jupyter → Core)
     */
    private syncFromJupyter;
    /**
     * Handle custom messages from Python (elegant command pattern)
     */
    handleCustomMessage(content: any): void;
    /**
     * Elegant position update with smooth synchronization
     */
    private updateCorePositions;
    /**
     * Elegant smooth node position animation
     */
    private animateNodePosition;
    /**
     * Update core engine size elegantly
     */
    private updateCoreSize;
    /**
     * Elegant initialization with data
     */
    initialize(): void;
    /**
     * Elegant cleanup
     */
    destroy(): void;
}
