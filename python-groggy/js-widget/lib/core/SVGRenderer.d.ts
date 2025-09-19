/**
 * 🎨 Unified SVG Renderer
 *
 * Efficient SVG rendering with Level-of-Detail (LOD) optimization
 * Consistent rendering pipeline across all visualization environments
 */
export class SVGRenderer {
    constructor(config?: {});
    config: {
        width: number;
        height: number;
        nodeRadius: number;
        edgeWidth: number;
        nodeColor: string;
        edgeColor: string;
        backgroundColor: string;
        enableLOD: boolean;
        lodThreshold: number;
        enableLabels: boolean;
        labelThreshold: number;
    };
    element: any;
    svg: SVGSVGElement | null;
    nodesGroup: SVGGElement | null;
    edgesGroup: SVGGElement | null;
    labelsGroup: SVGGElement | null;
    frameCount: number;
    lastRenderTime: number;
    renderTimes: any[];
    /**
     * Attach renderer to DOM element
     */
    attachToElement(element: any): void;
    /**
     * Create SVG structure
     */
    createSVGStructure(): void;
    /**
     * 🎯 Main render method - used by all environments
     */
    render(renderData: any): void;
    /**
     * Full quality rendering
     */
    renderFull(nodes: any, edges: any, positions: any, showLabels?: boolean): void;
    /**
     * Level-of-detail rendering for large graphs
     */
    renderLOD(nodes: any, edges: any, positions: any): void;
    /**
     * Render edges
     */
    renderEdges(edges: any, positions: any): void;
    /**
     * Render edges (LOD version)
     */
    renderEdgesLOD(edges: any, positions: any): void;
    /**
     * Render nodes
     */
    renderNodes(nodes: any, positions: any): void;
    /**
     * Render nodes (LOD version)
     */
    renderNodesLOD(nodes: any, positions: any): void;
    /**
     * Render labels
     */
    renderLabels(nodes: any, positions: any): void;
    /**
     * Record render time for performance monitoring
     */
    recordRenderTime(time: any): void;
    /**
     * Get performance statistics
     */
    getPerformanceStats(): {
        averageRenderTime: string;
        maxRenderTime: string;
        minRenderTime: string;
        frameCount: number;
        fps: string;
    } | null;
    /**
     * Update configuration
     */
    updateConfig(newConfig: any): void;
    /**
     * Clear renderer
     */
    clear(): void;
    /**
     * Export current SVG as string
     */
    exportSVG(): string | null;
}
export default SVGRenderer;
