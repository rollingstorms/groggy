/**
 * 🖱️ Unified Interaction Engine
 * 
 * Handles mouse, touch, and keyboard interactions consistently
 * across all visualization environments
 */

export class InteractionEngine {
    constructor(config = {}) {
        this.config = {
            enableDrag: true,
            enableZoom: true,
            enablePan: true,
            enableMultiTouch: true,
            dragThreshold: 5,      // pixels before drag starts
            zoomSensitivity: 0.1,
            panSensitivity: 1.0,
            ...config
        };
        
        this.element = null;
        this.callbacks = new Map();
        this.state = {
            isDragging: false,
            isPanning: false,
            draggedNode: null,
            lastMousePos: { x: 0, y: 0 },
            dragStartPos: { x: 0, y: 0 },
            zoom: 1,
            pan: { x: 0, y: 0 }
        };
        
        // Event listeners
        this.listeners = new Map();
        
        console.log('🖱️ InteractionEngine initialized:', this.config);
    }
    
    /**
     * Attach interaction engine to DOM element
     */
    attachToElement(element, callbacks = {}) {
        this.element = element;
        this.callbacks = new Map(Object.entries(callbacks));
        
        this.setupEventListeners();
        
        console.log('🔗 InteractionEngine attached to element');
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        if (!this.element) return;
        
        // Mouse events
        this.addListener('mousedown', this.handleMouseDown.bind(this));
        this.addListener('mousemove', this.handleMouseMove.bind(this));
        this.addListener('mouseup', this.handleMouseUp.bind(this));
        this.addListener('wheel', this.handleWheel.bind(this));
        
        // Touch events for mobile
        if (this.config.enableMultiTouch) {
            this.addListener('touchstart', this.handleTouchStart.bind(this));
            this.addListener('touchmove', this.handleTouchMove.bind(this));
            this.addListener('touchend', this.handleTouchEnd.bind(this));
        }
        
        // Keyboard events
        this.addListener('keydown', this.handleKeyDown.bind(this));
        
        // Prevent context menu on right click
        this.addListener('contextmenu', (e) => e.preventDefault());
    }
    
    /**
     * Add event listener and track it
     */
    addListener(event, handler) {
        this.element.addEventListener(event, handler);
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(handler);
    }
    
    /**
     * Handle mouse down events
     */
    handleMouseDown(event) {
        event.preventDefault();
        
        const rect = this.element.getBoundingClientRect();
        const mousePos = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
        
        this.state.lastMousePos = mousePos;
        this.state.dragStartPos = { ...mousePos };
        
        // Check if clicking on a node
        const target = event.target;
        const nodeId = target.getAttribute('data-node-id');
        const edgeId = target.getAttribute('data-edge-id');
        
        if (nodeId && this.config.enableDrag) {
            this.state.draggedNode = nodeId;
            this.state.isDragging = false; // Will become true after threshold
            this.emit('node_mousedown', { nodeId, position: mousePos, event });
        } else if (edgeId) {
            this.emit('edge_click', { edgeId, position: mousePos, event });
        } else if (this.config.enablePan) {
            this.state.isPanning = true;
        }
    }
    
    /**
     * Handle mouse move events
     */
    handleMouseMove(event) {
        event.preventDefault();
        
        const rect = this.element.getBoundingClientRect();
        const mousePos = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
        
        const deltaX = mousePos.x - this.state.lastMousePos.x;
        const deltaY = mousePos.y - this.state.lastMousePos.y;
        
        // Node dragging
        if (this.state.draggedNode) {
            const distance = Math.sqrt(
                Math.pow(mousePos.x - this.state.dragStartPos.x, 2) +
                Math.pow(mousePos.y - this.state.dragStartPos.y, 2)
            );
            
            if (!this.state.isDragging && distance > this.config.dragThreshold) {
                this.state.isDragging = true;
                this.emit('drag_start', { nodeId: this.state.draggedNode, position: mousePos });
            }
            
            if (this.state.isDragging) {
                this.emit('node_drag', { 
                    nodeId: this.state.draggedNode, 
                    position: mousePos,
                    delta: { x: deltaX, y: deltaY }
                });
            }
        }
        // Canvas panning
        else if (this.state.isPanning && this.config.enablePan) {
            this.state.pan.x += deltaX * this.config.panSensitivity;
            this.state.pan.y += deltaY * this.config.panSensitivity;
            
            this.emit('pan', { 
                pan: { ...this.state.pan },
                delta: { x: deltaX, y: deltaY }
            });
        }
        
        this.state.lastMousePos = mousePos;
    }
    
    /**
     * Handle mouse up events
     */
    handleMouseUp(event) {
        event.preventDefault();
        
        const rect = this.element.getBoundingClientRect();
        const mousePos = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
        
        // End node dragging
        if (this.state.draggedNode) {
            if (this.state.isDragging) {
                this.emit('drag_end', { 
                    nodeId: this.state.draggedNode, 
                    position: mousePos 
                });
            } else {
                // Was a click, not a drag
                this.emit('node_click', { 
                    nodeId: this.state.draggedNode, 
                    position: mousePos,
                    event
                });
            }
            
            this.state.draggedNode = null;
            this.state.isDragging = false;
        }
        
        // End panning
        if (this.state.isPanning) {
            this.state.isPanning = false;
            this.emit('pan_end', { pan: { ...this.state.pan } });
        }
    }
    
    /**
     * Handle wheel events for zooming
     */
    handleWheel(event) {
        if (!this.config.enableZoom) return;
        
        event.preventDefault();
        
        const rect = this.element.getBoundingClientRect();
        const mousePos = {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
        
        const deltaZoom = -event.deltaY * this.config.zoomSensitivity * 0.01;
        const newZoom = Math.max(0.1, Math.min(5, this.state.zoom + deltaZoom));
        
        if (newZoom !== this.state.zoom) {
            this.state.zoom = newZoom;
            
            this.emit('zoom', { 
                zoom: this.state.zoom,
                center: mousePos,
                delta: deltaZoom
            });
        }
    }
    
    /**
     * Handle touch start events
     */
    handleTouchStart(event) {
        event.preventDefault();
        
        if (event.touches.length === 1) {
            // Single touch - treat as mouse down
            const touch = event.touches[0];
            const mouseEvent = {
                clientX: touch.clientX,
                clientY: touch.clientY,
                target: touch.target,
                preventDefault: () => {}
            };
            this.handleMouseDown(mouseEvent);
        }
        // Multi-touch gestures could be handled here
    }
    
    /**
     * Handle touch move events
     */
    handleTouchMove(event) {
        event.preventDefault();
        
        if (event.touches.length === 1) {
            // Single touch - treat as mouse move
            const touch = event.touches[0];
            const mouseEvent = {
                clientX: touch.clientX,
                clientY: touch.clientY,
                preventDefault: () => {}
            };
            this.handleMouseMove(mouseEvent);
        }
        // Multi-touch gestures could be handled here
    }
    
    /**
     * Handle touch end events
     */
    handleTouchEnd(event) {
        event.preventDefault();
        
        // Use last known position for touch end
        const mouseEvent = {
            clientX: this.state.lastMousePos.x,
            clientY: this.state.lastMousePos.y,
            preventDefault: () => {}
        };
        this.handleMouseUp(mouseEvent);
    }
    
    /**
     * Handle keyboard events
     */
    handleKeyDown(event) {
        // Space bar to reset view
        if (event.code === 'Space') {
            event.preventDefault();
            this.resetView();
        }
        
        // Arrow keys for panning
        if (this.config.enablePan) {
            const panStep = 20;
            let deltaX = 0, deltaY = 0;
            
            switch (event.code) {
                case 'ArrowLeft':
                    deltaX = -panStep;
                    break;
                case 'ArrowRight':
                    deltaX = panStep;
                    break;
                case 'ArrowUp':
                    deltaY = -panStep;
                    break;
                case 'ArrowDown':
                    deltaY = panStep;
                    break;
            }
            
            if (deltaX || deltaY) {
                event.preventDefault();
                this.state.pan.x += deltaX;
                this.state.pan.y += deltaY;
                
                this.emit('pan', { 
                    pan: { ...this.state.pan },
                    delta: { x: deltaX, y: deltaY }
                });
            }
        }
    }
    
    /**
     * Reset view to default state
     */
    resetView() {
        this.state.zoom = 1;
        this.state.pan = { x: 0, y: 0 };
        
        this.emit('view_reset', {
            zoom: this.state.zoom,
            pan: { ...this.state.pan }
        });
    }
    
    /**
     * Process frame (called from main update loop)
     */
    processFrame() {
        // Could be used for continuous interactions or animations
        // Currently not needed but available for future use
    }
    
    /**
     * Emit event to callbacks
     */
    emit(event, data) {
        const callback = this.callbacks.get(event);
        if (callback) {
            try {
                callback(data);
            } catch (error) {
                console.error(`Error in interaction callback for ${event}:`, error);
            }
        }
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        console.log('⚙️ InteractionEngine config updated:', this.config);
    }
    
    /**
     * Get current interaction state
     */
    getState() {
        return { ...this.state };
    }
    
    /**
     * Set interaction state
     */
    setState(newState) {
        this.state = { ...this.state, ...newState };
    }
    
    /**
     * Clean up event listeners
     */
    cleanup() {
        if (!this.element) return;
        
        for (const [event, handlers] of this.listeners) {
            handlers.forEach(handler => {
                this.element.removeEventListener(event, handler);
            });
        }
        
        this.listeners.clear();
        console.log('🧹 InteractionEngine cleaned up');
    }
}

export default InteractionEngine;