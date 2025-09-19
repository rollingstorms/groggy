/**
 * 🧠 GroggyVizCore - Unified JavaScript Core Engine
 * 
 * Single source of truth for all visualization logic across:
 * - Jupyter widgets
 * - WebSocket streaming client
 * - Static file exports
 * 
 * Architecture:
 * - PhysicsEngine: Force-directed layout simulation
 * - SVGRenderer: Efficient SVG rendering with optimizations
 * - InteractionEngine: Unified interaction handling
 * - Single update() method for all environments
 */

import { PhysicsEngine } from './PhysicsEngine.js';
import { SVGRenderer } from './SVGRenderer.js';
import { InteractionEngine } from './InteractionEngine.js';

export class GroggyVizCore {
    constructor(nodes = [], edges = [], config = {}) {
        // Core configuration with sensible defaults
        this.config = {
            // Canvas dimensions
            width: config.width || 800,
            height: config.height || 600,
            
            // Physics simulation
            physics: {
                enabled: config.physics?.enabled !== false,
                forceStrength: config.physics?.forceStrength || 30,
                linkDistance: config.physics?.linkDistance || 50,
                linkStrength: config.physics?.linkStrength || 0.1,
                chargeStrength: config.physics?.chargeStrength || -300,
                centerStrength: config.physics?.centerStrength || 0.1,
                velocityDecay: config.physics?.velocityDecay || 0.4,
                alpha: config.physics?.alpha || 1.0,
                alphaDecay: config.physics?.alphaDecay || 0.0228,
                alphaMin: config.physics?.alphaMin || 0.001,
                ...config.physics
            },
            
            // Rendering options
            rendering: {
                nodeRadius: config.rendering?.nodeRadius || 20,
                nodeStroke: config.rendering?.nodeStroke || '#333',
                nodeStrokeWidth: config.rendering?.nodeStrokeWidth || 2,
                edgeStroke: config.rendering?.edgeStroke || '#999',
                edgeStrokeWidth: config.rendering?.edgeStrokeWidth || 2,
                edgeOpacity: config.rendering?.edgeOpacity || 0.6,
                backgroundColor: config.rendering?.backgroundColor || '#ffffff',
                animationDuration: config.rendering?.animationDuration || 300,
                enableShadows: config.rendering?.enableShadows !== false,
                enableAnimations: config.rendering?.enableAnimations !== false,
                ...config.rendering
            },
            
            // Interaction settings
            interaction: {
                enableDrag: config.interaction?.enableDrag !== false,
                enableZoom: config.interaction?.enableZoom !== false,
                enablePan: config.interaction?.enablePan !== false,
                enableSelection: config.interaction?.enableSelection !== false,
                enableHover: config.interaction?.enableHover !== false,
                dragThreshold: config.interaction?.dragThreshold || 5,
                zoomExtent: config.interaction?.zoomExtent || [0.1, 10],
                ...config.interaction
            },
            
            ...config
        };
        
        // Initialize subsystems
        this.physics = new PhysicsEngine(this.config.physics);
        this.renderer = new SVGRenderer(this.config.rendering);
        this.interactions = new InteractionEngine(this.config.interaction);
        
        // Core state
        this.nodes = [];
        this.edges = [];
        this.nodePositions = new Map();
        this.nodeVelocities = new Map();
        this.selectedNodes = new Set();
        this.hoveredNode = null;
        
        // Animation and update state
        this.animationId = null;
        this.lastUpdateTime = 0;
        this.isSimulationRunning = false;
        this.isDragging = false;
        
        // Event system
        this.eventListeners = new Map();
        
        // Container element
        this.container = null;
        this.svgElement = null;
        
        // Set initial data if provided
        if (nodes.length > 0 || edges.length > 0) {
            this.setData(nodes, edges);
        }
        
        // Bind methods to maintain context
        this.update = this.update.bind(this);
        this.animate = this.animate.bind(this);
        
        console.log('🧠 GroggyVizCore initialized with unified architecture');
    }
    
    /**
     * 🔄 Single update method for all environments
     * Used by Jupyter widgets, WebSocket streaming, and file exports
     */
    update(frameData = null) {
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastUpdateTime;
        this.lastUpdateTime = currentTime;
        
        // Handle external frame data (from WebSocket or Python)
        if (frameData) {
            this.handleFrameData(frameData);
        }
        
        // Update physics simulation if enabled
        if (this.config.physics.enabled && this.isSimulationRunning) {
            this.physics.updatePositions(
                this.nodes,
                this.edges,
                this.nodePositions,
                this.nodeVelocities,
                deltaTime
            );
        }
        
        // Update rendering
        if (this.svgElement) {
            this.renderer.render({
                nodes: this.nodes,
                edges: this.edges,
                positions: this.nodePositions,
                selectedNodes: this.selectedNodes,
                hoveredNode: this.hoveredNode,
                container: this.svgElement,
                width: this.config.width,
                height: this.config.height
            });
        }
        
        // Process interaction events
        this.interactions.handleEvents({
            nodes: this.nodes,
            positions: this.nodePositions,
            selectedNodes: this.selectedNodes,
            hoveredNode: this.hoveredNode
        });
        
        // Emit update event
        this.emit('update', {
            timestamp: currentTime,
            deltaTime,
            nodes: this.nodes,
            edges: this.edges,
            positions: this.nodePositions
        });
        
        // Continue animation if simulation is running
        if (this.isSimulationRunning && this.physics.isActive()) {
            this.animationId = requestAnimationFrame(this.animate);
        } else {
            this.stopSimulation();
        }
        
        return {
            nodes: this.nodes,
            edges: this.edges,
            positions: this.nodePositions,
            metadata: {
                timestamp: currentTime,
                simulationRunning: this.isSimulationRunning,
                frameCount: this.physics.getFrameCount(),
                fps: this.calculateFPS(deltaTime)
            }
        };
    }
    
    /**
     * 🎯 Set graph data and initialize positions
     */
    setData(nodes, edges) {
        this.nodes = [...nodes];
        this.edges = [...edges];
        
        // Initialize node positions if not provided
        this.initializePositions();
        
        // Configure physics with new data
        this.physics.setData(this.nodes, this.edges);
        
        // Update renderer with new data
        this.renderer.setData(this.nodes, this.edges);
        
        // Start simulation if physics is enabled
        if (this.config.physics.enabled && this.nodes.length > 0) {
            this.startSimulation();
        }
        
        this.emit('dataChanged', { nodes: this.nodes, edges: this.edges });
    }
    
    /**
     * 🖥️ Attach to DOM element and initialize rendering
     */
    attachToDOM(element) {
        this.container = element;
        
        // Create SVG element
        this.svgElement = this.renderer.createSVG(
            this.config.width,
            this.config.height
        );
        
        // Clear container and append SVG
        element.innerHTML = '';
        element.appendChild(this.svgElement);
        
        // Set up interactions
        this.interactions.attachToElement(this.svgElement, {
            onNodeDrag: this.handleNodeDrag.bind(this),
            onNodeClick: this.handleNodeClick.bind(this),
            onNodeHover: this.handleNodeHover.bind(this),
            onPan: this.handlePan.bind(this),
            onZoom: this.handleZoom.bind(this),
            onSelection: this.handleSelection.bind(this)
        });
        
        // Initial render
        if (this.nodes.length > 0) {
            this.update();
        }
        
        this.emit('attached', { container: element });
    }
    
    /**
     * ⚡ Start physics simulation
     */
    startSimulation() {
        if (this.isSimulationRunning) return;
        
        this.isSimulationRunning = true;
        this.physics.start();
        this.lastUpdateTime = performance.now();
        this.animationId = requestAnimationFrame(this.animate);
        
        this.emit('simulationStarted');
    }
    
    /**
     * ⏹️ Stop physics simulation
     */
    stopSimulation() {
        if (!this.isSimulationRunning) return;
        
        this.isSimulationRunning = false;
        this.physics.stop();
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        this.emit('simulationStopped');
    }
    
    /**
     * 🎬 Animation frame handler
     */
    animate() {
        this.update();
    }
    
    /**
     * 📍 Initialize node positions if not provided
     */
    initializePositions() {
        const centerX = this.config.width / 2;
        const centerY = this.config.height / 2;
        
        this.nodes.forEach(node => {
            if (!node.position || (!node.position.x && !node.position.y)) {
                // Random initial positions in a circle around center
                const angle = Math.random() * 2 * Math.PI;
                const radius = Math.random() * Math.min(centerX, centerY) * 0.3;
                const x = centerX + radius * Math.cos(angle);
                const y = centerY + radius * Math.sin(angle);
                
                this.nodePositions.set(node.id, { x, y });
                this.nodeVelocities.set(node.id, { vx: 0, vy: 0 });
            } else {
                this.nodePositions.set(node.id, { 
                    x: node.position.x, 
                    y: node.position.y 
                });
                this.nodeVelocities.set(node.id, { vx: 0, vy: 0 });
            }
        });
    }
    
    /**
     * 📨 Handle external frame data (WebSocket, Python updates)
     */
    handleFrameData(frameData) {
        if (frameData.positions) {
            // Update positions from external source
            Object.entries(frameData.positions).forEach(([nodeId, pos]) => {
                if (this.nodePositions.has(nodeId)) {
                    this.nodePositions.set(nodeId, { x: pos.x, y: pos.y });
                }
            });
        }
        
        if (frameData.events) {
            // Handle external events
            frameData.events.forEach(event => {
                this.handleExternalEvent(event);
            });
        }
        
        if (frameData.config) {
            // Update configuration
            this.updateConfig(frameData.config);
        }
    }
    
    /**
     * 🖱️ Handle node drag events
     */
    handleNodeDrag(nodeId, position, isDragging) {
        if (isDragging) {
            this.isDragging = true;
            this.nodePositions.set(nodeId, position);
            this.nodeVelocities.set(nodeId, { vx: 0, vy: 0 }); // Stop physics for dragged node
            this.emit('nodeDrag', { nodeId, position, isDragging });
        } else {
            this.isDragging = false;
            this.emit('nodeDragEnd', { nodeId, position });
        }
    }
    
    /**
     * 👆 Handle node click events
     */
    handleNodeClick(nodeId, event) {
        if (this.config.interaction.enableSelection) {
            if (event.ctrlKey || event.metaKey) {
                // Multi-selection
                if (this.selectedNodes.has(nodeId)) {
                    this.selectedNodes.delete(nodeId);
                } else {
                    this.selectedNodes.add(nodeId);
                }
            } else {
                // Single selection
                this.selectedNodes.clear();
                this.selectedNodes.add(nodeId);
            }
        }
        
        this.emit('nodeClick', { nodeId, event, selectedNodes: [...this.selectedNodes] });
    }
    
    /**
     * 🔍 Handle node hover events
     */
    handleNodeHover(nodeId) {
        this.hoveredNode = nodeId;
        this.emit('nodeHover', { nodeId });
    }
    
    /**
     * 🔍 Handle pan events
     */
    handlePan(delta) {
        this.emit('pan', { delta });
    }
    
    /**
     * 🔍 Handle zoom events
     */
    handleZoom(scale, center) {
        this.emit('zoom', { scale, center });
    }
    
    /**
     * 📤 Handle selection events
     */
    handleSelection(selectedIds) {
        this.selectedNodes.clear();
        selectedIds.forEach(id => this.selectedNodes.add(id));
        this.emit('selectionChanged', { selectedNodes: [...this.selectedNodes] });
    }
    
    /**
     * ⚙️ Update configuration
     */
    updateConfig(newConfig) {
        // Deep merge configuration
        this.config = this.deepMerge(this.config, newConfig);
        
        // Update subsystems
        this.physics.updateConfig(this.config.physics);
        this.renderer.updateConfig(this.config.rendering);
        this.interactions.updateConfig(this.config.interaction);
        
        this.emit('configChanged', { config: this.config });
    }
    
    /**
     * 📊 Calculate FPS from delta time
     */
    calculateFPS(deltaTime) {
        return deltaTime > 0 ? Math.round(1000 / deltaTime) : 0;
    }
    
    /**
     * 🎧 Event system implementation
     */
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event).push(callback);
    }
    
    off(event, callback) {
        if (this.eventListeners.has(event)) {
            const listeners = this.eventListeners.get(event);
            const index = listeners.indexOf(callback);
            if (index > -1) {
                listeners.splice(index, 1);
            }
        }
    }
    
    emit(event, data) {
        if (this.eventListeners.has(event)) {
            this.eventListeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }
    
    /**
     * 🔧 Utility: Deep merge objects
     */
    deepMerge(target, source) {
        const result = { ...target };
        
        for (const key in source) {
            if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                result[key] = this.deepMerge(result[key] || {}, source[key]);
            } else {
                result[key] = source[key];
            }
        }
        
        return result;
    }
    
    /**
     * 🧹 Cleanup and destroy
     */
    destroy() {
        this.stopSimulation();
        
        if (this.container) {
            this.container.innerHTML = '';
        }
        
        this.physics.destroy();
        this.renderer.destroy();
        this.interactions.destroy();
        
        this.eventListeners.clear();
        this.nodePositions.clear();
        this.nodeVelocities.clear();
        this.selectedNodes.clear();
        
        this.emit('destroyed');
    }
    
    /**
     * 📤 Export current state for serialization
     */
    exportState() {
        return {
            nodes: this.nodes,
            edges: this.edges,
            positions: Object.fromEntries(this.nodePositions),
            selectedNodes: [...this.selectedNodes],
            hoveredNode: this.hoveredNode,
            config: this.config,
            metadata: {
                timestamp: performance.now(),
                simulationRunning: this.isSimulationRunning,
                frameCount: this.physics.getFrameCount()
            }
        };
    }
    
    /**
     * 📥 Import state from serialization
     */
    importState(state) {
        this.nodes = state.nodes || [];
        this.edges = state.edges || [];
        
        if (state.positions) {
            this.nodePositions.clear();
            Object.entries(state.positions).forEach(([nodeId, pos]) => {
                this.nodePositions.set(nodeId, pos);
            });
        }
        
        if (state.selectedNodes) {
            this.selectedNodes.clear();
            state.selectedNodes.forEach(id => this.selectedNodes.add(id));
        }
        
        this.hoveredNode = state.hoveredNode || null;
        
        if (state.config) {
            this.updateConfig(state.config);
        }
        
        this.emit('stateImported', { state });
    }
}