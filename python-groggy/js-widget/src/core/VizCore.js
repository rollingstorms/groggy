/**
 * 🎯 Unified Visualization Core - Single Source of Truth
 * 
 * This is the unified core engine that works across all visualization environments:
 * - Jupyter widgets
 * - WebSocket streaming clients  
 * - Static file exports
 * - Local browser displays
 */

import { PhysicsEngine } from './PhysicsEngine.js';
import { SVGRenderer } from './SVGRenderer.js';
import { InteractionEngine } from './InteractionEngine.js';

export class GroggyVizCore {
    constructor(nodes = [], edges = [], config = {}) {
        // Store core data
        this.nodes = nodes;
        this.edges = edges;
        this.config = {
            // Top-level properties for backward compatibility
            width: 800,
            height: 600,
            ...config,
            // Nested configuration
            physics: {
                enabled: true,
                charge: -300,
                distance: 50,
                damping: 0.9,
                iterations: 100,
                ...config.physics
            },
            rendering: {
                width: config.width || 800,
                height: config.height || 600,
                nodeRadius: 8,
                edgeWidth: 1,
                ...config.rendering
            },
            interaction: {
                enableDrag: true,
                enableZoom: true,
                enablePan: true,
                ...config.interaction
            }
        };
        
        // Initialize subsystems
        this.physics = new PhysicsEngine(this.config.physics);
        this.renderer = new SVGRenderer(this.config.rendering);
        this.interactions = new InteractionEngine(this.config.interaction);
        
        // State
        this.positions = new Map();
        this.velocities = new Map();
        this.isRunning = false;
        this.frameCount = 0;
        
        // Additional properties for widget compatibility
        this.selectedNodes = new Set();
        this.nodePositions = this.positions; // Alias for compatibility
        
        // Event system for bidirectional communication
        this.eventCallbacks = new Map();
        
        console.log('🎨 GroggyVizCore initialized:', {
            nodes: this.nodes.length,
            edges: this.edges.length,
            config: this.config
        });
    }
    
    /**
     * 🎯 Single update method used by ALL visualization environments
     */
    update(frameData = null) {
        if (frameData) {
            // External frame data (from WebSocket, etc.)
            this.applyFrameData(frameData);
        } else {
            // Internal physics simulation
            this.physics.updatePositions(
                this.nodes, 
                this.edges, 
                this.positions, 
                this.velocities,
                1/60 // 60 FPS
            );
        }
        
        // Render the current state
        this.renderer.render({
            nodes: this.nodes,
            edges: this.edges,
            positions: this.positions,
            frameCount: this.frameCount++
        });
        
        // Handle interactions
        this.interactions.processFrame();
        
        // Emit frame event
        this.emit('frame', {
            nodes: this.nodes,
            edges: this.edges,
            positions: this.positions,
            frameCount: this.frameCount
        });
        
        return this.getFrameData();
    }
    
    /**
     * Apply frame data from external source (WebSocket streaming)
     */
    applyFrameData(frameData) {
        if (frameData.positions) {
            for (const [nodeId, position] of Object.entries(frameData.positions)) {
                this.positions.set(nodeId, position);
            }
        }
        
        if (frameData.nodes) {
            this.nodes = frameData.nodes;
        }
        
        if (frameData.edges) {
            this.edges = frameData.edges;
        }
    }
    
    /**
     * Get current frame data for transmission
     */
    getFrameData() {
        return {
            nodes: this.nodes,
            edges: this.edges,
            positions: Object.fromEntries(this.positions),
            velocities: Object.fromEntries(this.velocities),
            frameCount: this.frameCount,
            timestamp: Date.now()
        };
    }
    
    /**
     * Attach to DOM element (for widgets and local display)
     */
    attachToDOM(element, callbacks = {}) {
        this.element = element;
        this.eventCallbacks = new Map(Object.entries(callbacks));
        
        // Initialize renderer with DOM element
        this.renderer.attachToElement(element);
        
        // Setup interactions
        this.interactions.attachToElement(element, {
            onNodeDrag: (nodeId, position) => {
                this.positions.set(nodeId, position);
                this.emit('node_drag', { nodeId, position });
            },
            onNodeClick: (nodeId) => {
                this.emit('node_click', { nodeId });
            },
            onEdgeClick: (edgeId) => {
                this.emit('edge_click', { edgeId });
            }
        });
        
        console.log('🔗 GroggyVizCore attached to DOM element:', element);
    }
    
    /**
     * Start physics simulation
     */
    start() {
        this.isRunning = true;
        this.physics.start();
        this.animationLoop();
        console.log('▶️ GroggyVizCore simulation started');
    }
    
    /**
     * Stop physics simulation
     */
    stop() {
        this.isRunning = false;
        this.physics.stop();
        console.log('⏹️ GroggyVizCore simulation stopped');
    }
    
    /**
     * Animation loop for local/widget environments
     */
    animationLoop() {
        if (!this.isRunning) return;
        
        this.update();
        requestAnimationFrame(() => this.animationLoop());
    }
    
    /**
     * Event system
     */
    on(event, callback) {
        if (!this.eventCallbacks.has(event)) {
            this.eventCallbacks.set(event, []);
        }
        this.eventCallbacks.get(event).push(callback);
    }
    
    emit(event, data) {
        if (this.eventCallbacks.has(event)) {
            this.eventCallbacks.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event callback for ${event}:`, error);
                }
            });
        }
    }
    
    /**
     * Update graph data
     */
    setData(nodes, edges) {
        this.nodes = nodes;
        this.edges = edges;
        
        // Reset physics state
        this.positions.clear();
        this.velocities.clear();
        this.frameCount = 0;
        
        // Initialize positions if not provided
        this.nodes.forEach((node, index) => {
            if (!this.positions.has(node.id)) {
                // Simple circular layout as default
                const angle = (index * 2 * Math.PI) / this.nodes.length;
                const radius = Math.min(this.config.rendering.width, this.config.rendering.height) / 4;
                this.positions.set(node.id, {
                    x: this.config.rendering.width / 2 + radius * Math.cos(angle),
                    y: this.config.rendering.height / 2 + radius * Math.sin(angle)
                });
            }
            
            if (!this.velocities.has(node.id)) {
                this.velocities.set(node.id, { x: 0, y: 0 });
            }
        });
        
        this.emit('data_updated', { nodes, edges });
        console.log('📊 GroggyVizCore data updated:', { nodes: nodes.length, edges: edges.length });
    }
    
    /**
     * Get current configuration
     */
    getConfig() {
        return { ...this.config };
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        this.config = {
            ...this.config,
            ...newConfig,
            physics: { ...this.config.physics, ...newConfig.physics },
            rendering: { ...this.config.rendering, ...newConfig.rendering },
            interaction: { ...this.config.interaction, ...newConfig.interaction }
        };
        
        // Update subsystems
        this.physics.updateConfig(this.config.physics);
        this.renderer.updateConfig(this.config.rendering);
        this.interactions.updateConfig(this.config.interaction);
        
        this.emit('config_updated', this.config);
        console.log('⚙️ GroggyVizCore config updated:', this.config);
    }
    
    /**
     * Export current state for saving/serialization
     */
    exportState() {
        return {
            nodes: this.nodes,
            edges: this.edges,
            positions: Object.fromEntries(this.positions),
            velocities: Object.fromEntries(this.velocities),
            selectedNodes: Array.from(this.selectedNodes),
            config: this.config,
            frameCount: this.frameCount,
            timestamp: Date.now()
        };
    }
    
    /**
     * Clean up and destroy the visualization
     */
    destroy() {
        this.stop();
        
        // Cleanup subsystems
        if (this.interactions && typeof this.interactions.cleanup === 'function') {
            this.interactions.cleanup();
        }
        
        if (this.renderer && typeof this.renderer.clear === 'function') {
            this.renderer.clear();
        }
        
        // Clear data
        this.positions.clear();
        this.velocities.clear();
        this.selectedNodes.clear();
        this.eventCallbacks.clear();
        
        // Clear references
        this.element = null;
        this.nodes = [];
        this.edges = [];
        
        console.log('🧹 GroggyVizCore destroyed');
    }
}

export default GroggyVizCore;