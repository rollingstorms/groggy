/**
 * Elegant Core Bridge - Single Source of Truth
 * 
 * This module provides the elegant abstraction layer that bridges
 * our existing GroggyGraphView with Jupyter's widget system.
 * 
 * Philosophy: Extend, don't duplicate. One codebase, multiple interfaces.
 */

// Elegant import of our core visualization engine
// Using relative path for maximum elegance and portability
const GroggyGraphView = require('../../python/groggy/widgets/widget_view.js').GroggyGraphView;

/**
 * Enhanced GroggyGraphView with Jupyter widget synchronization.
 * 
 * This elegant wrapper extends our core visualization engine with
 * bidirectional communication capabilities for Jupyter widgets.
 */
export class JupyterGroggyView {
    private coreView: typeof GroggyGraphView;
    private model: any;
    private element: HTMLElement;
    private syncCallbacks: Map<string, Function[]> = new Map();
    
    constructor(element: HTMLElement, model: any) {
        this.element = element;
        this.model = model;
        
        // Initialize our elegant core engine
        this.coreView = new (GroggyGraphView as any)(element, this.extractCoreConfig());
        
        // Set up elegant bidirectional sync
        this.setupElegantSync();
    }
    
    /**
     * Extract core configuration from Jupyter model traits.
     * Elegant transformation: Jupyter traits → Core config
     */
    private extractCoreConfig() {
        return {
            width: this.model.get('width') || 800,
            height: this.model.get('height') || 600,
            layout: this.model.get('layout_algorithm') || 'force-directed',
            theme: this.model.get('theme') || 'light',
            enableDrag: this.model.get('enable_drag') !== false,
            enablePan: this.model.get('enable_pan') !== false,
            enableZoom: this.model.get('enable_zoom') !== false,
            animationDuration: this.model.get('animation_duration') || 300
        };
    }
    
    /**
     * Elegant bidirectional synchronization setup.
     * Core events → Jupyter model updates
     * Jupyter model changes → Core engine updates
     */
    private setupElegantSync() {
        // Elegant: Core → Jupyter synchronization
        this.syncFromCore();
        
        // Elegant: Jupyter → Core synchronization  
        this.syncFromJupyter();
    }
    
    /**
     * Sync core engine events to Jupyter model (Core → Jupyter)
     */
    private syncFromCore() {
        // Node interactions
        this.coreView.on('nodeClick', (data: any) => {
            this.model.send({
                type: 'node_click',
                node_id: data.nodeId,
                node_data: data.nodeData,
                position: data.position
            });
        });
        
        this.coreView.on('nodeDoubleClick', (data: any) => {
            this.model.send({
                type: 'node_double_click', 
                node_id: data.nodeId,
                node_data: data.nodeData
            });
        });
        
        this.coreView.on('nodeHover', (nodeId: string | null) => {
            this.model.set('hovered_node', nodeId || '');
            this.model.save_changes();
        });
        
        // Selection changes
        this.coreView.on('selectionChange', (selectedIds: string[]) => {
            this.model.set('selected_nodes', selectedIds);
            this.model.save_changes();
        });
        
        // Layout changes
        this.coreView.on('layoutChange', (data: any) => {
            this.model.set('layout_algorithm', data.to);
            this.model.save_changes();
        });
        
        // Camera changes (elegant throttling)
        let cameraThrottle: any = null;
        this.coreView.on('cameraChange', (camera: any) => {
            if (cameraThrottle) clearTimeout(cameraThrottle);
            cameraThrottle = setTimeout(() => {
                this.model.set('camera_position', camera);
                this.model.save_changes();
            }, 100); // Elegant 100ms throttling
        });
        
        // Drag state
        this.coreView.on('dragStart', () => {
            this.model.set('is_dragging', true);
            this.model.save_changes();
        });
        
        this.coreView.on('dragEnd', () => {
            this.model.set('is_dragging', false);
            this.model.save_changes();
        });
    }
    
    /**
     * Sync Jupyter model changes to core engine (Jupyter → Core)
     */
    private syncFromJupyter() {
        // Data updates
        this.model.on('change:nodes change:edges', () => {
            const nodes = this.model.get('nodes') || [];
            const edges = this.model.get('edges') || [];
            this.coreView.setData(nodes, edges);
        });
        
        // Layout changes from Python
        this.model.on('change:layout_algorithm', () => {
            const layout = this.model.get('layout_algorithm');
            this.coreView.setLayout(layout, true);
        });
        
        // Theme changes from Python
        this.model.on('change:theme', () => {
            const theme = this.model.get('theme');
            this.coreView.setTheme(theme);
        });
        
        // Selection changes from Python
        this.model.on('change:selected_nodes', () => {
            const selectedNodes = this.model.get('selected_nodes') || [];
            this.coreView.selectNodes(selectedNodes, true);
        });
        
        // Node positions from Python
        this.model.on('change:node_positions', () => {
            const positions = this.model.get('node_positions') || {};
            this.updateCorePositions(positions);
        });
        
        // Configuration changes
        this.model.on('change:width change:height', () => {
            this.updateCoreSize();
        });
    }
    
    /**
     * Handle custom messages from Python (elegant command pattern)
     */
    handleCustomMessage(content: any) {
        const { type } = content;
        
        switch (type) {
            case 'set_layout':
                this.coreView.setLayout(content.algorithm, content.animate !== false);
                break;
                
            case 'focus_node':
                this.coreView.focusOnNode(content.node_id, content.zoom || 2.0);
                break;
                
            case 'reset_camera':
                this.coreView.resetCamera();
                break;
                
            case 'update_positions':
                this.updateCorePositions(content.positions);
                break;
                
            default:
                console.warn(`Unknown command type: ${type}`);
        }
    }
    
    /**
     * Elegant position update with smooth synchronization
     */
    private updateCorePositions(positions: Record<string, {x: number, y: number}>) {
        // Update core engine positions elegantly
        Object.entries(positions).forEach(([nodeId, pos]) => {
            const nodeElement = this.coreView.findNodeElement?.(nodeId);
            if (nodeElement && pos) {
                // Elegant smooth position update
                this.animateNodePosition(nodeId, pos);
            }
        });
    }
    
    /**
     * Elegant smooth node position animation
     */
    private animateNodePosition(nodeId: string, targetPos: {x: number, y: number}) {
        // Use core engine's animation system for elegance
        if (this.coreView.animateNodeTo) {
            this.coreView.animateNodeTo(nodeId, targetPos.x, targetPos.y);
        }
    }
    
    /**
     * Update core engine size elegantly
     */
    private updateCoreSize() {
        const width = this.model.get('width');
        const height = this.model.get('height');
        
        if (this.coreView.resize) {
            this.coreView.resize(width, height);
        }
    }
    
    /**
     * Elegant initialization with data
     */
    initialize() {
        const nodes = this.model.get('nodes') || [];
        const edges = this.model.get('edges') || [];
        const positions = this.model.get('node_positions') || {};
        
        // Initialize core engine with data
        this.coreView.setData(nodes, edges);
        
        if (Object.keys(positions).length > 0) {
            this.updateCorePositions(positions);
        }
        
        console.log('🎨 Elegant Jupyter-Groggy bridge initialized');
    }
    
    /**
     * Elegant cleanup
     */
    destroy() {
        if (this.coreView.destroy) {
            this.coreView.destroy();
        }
    }
}