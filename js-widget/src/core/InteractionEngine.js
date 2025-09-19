/**
 * 🖱️ InteractionEngine - Unified Interaction Handling
 * 
 * Unified interaction system for all visualization environments.
 * Provides consistent interaction behavior across Jupyter widgets,
 * WebSocket streaming, and static exports.
 * 
 * Features:
 * - Multi-touch and mouse support
 * - Drag and drop with physics integration
 * - Zoom and pan with smooth animations
 * - Selection (single and multi-select)
 * - Hover and focus states
 * - Keyboard shortcuts
 * - Accessibility support
 * - Touch gestures for mobile
 */

export class InteractionEngine {
    constructor(config = {}) {
        this.config = {
            // Drag settings
            enableDrag: config.enableDrag !== false,
            dragThreshold: config.dragThreshold || 5,
            dragDamping: config.dragDamping || 0.8,
            
            // Zoom settings
            enableZoom: config.enableZoom !== false,
            zoomExtent: config.zoomExtent || [0.1, 10],
            zoomSensitivity: config.zoomSensitivity || 0.001,
            
            // Pan settings
            enablePan: config.enablePan !== false,
            panSensitivity: config.panSensitivity || 1.0,
            
            // Selection settings
            enableSelection: config.enableSelection !== false,
            multiSelectKey: config.multiSelectKey || 'ctrlKey', // or 'metaKey' for Mac
            
            // Hover settings
            enableHover: config.enableHover !== false,
            hoverDelay: config.hoverDelay || 100,
            
            // Touch settings
            enableTouch: config.enableTouch !== false,
            touchSensitivity: config.touchSensitivity || 1.0,
            
            // Keyboard settings
            enableKeyboard: config.enableKeyboard !== false,
            
            ...config
        };
        
        // Interaction state
        this.isDragging = false;
        this.isPanning = false;
        this.isZooming = false;
        this.draggedNode = null;
        this.lastTouchDistance = 0;
        this.currentScale = 1;
        this.currentTranslate = { x: 0, y: 0 };
        
        // Event tracking
        this.mousePosition = { x: 0, y: 0 };
        this.dragStart = { x: 0, y: 0 };
        this.panStart = { x: 0, y: 0 };
        this.lastClickTime = 0;
        this.clickTimeout = null;
        
        // Touch tracking
        this.touches = new Map();
        this.lastTouchCenter = { x: 0, y: 0 };
        
        // Hover state
        this.hoveredElement = null;
        this.hoverTimeout = null;
        
        // Event callbacks
        this.callbacks = {
            onNodeDrag: null,
            onNodeClick: null,
            onNodeDoubleClick: null,
            onNodeHover: null,
            onPan: null,
            onZoom: null,
            onSelection: null,
            onKeyboard: null
        };
        
        // Bound event handlers
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);
        this.handleWheel = this.handleWheel.bind(this);
        this.handleTouchStart = this.handleTouchStart.bind(this);
        this.handleTouchMove = this.handleTouchMove.bind(this);
        this.handleTouchEnd = this.handleTouchEnd.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);
        this.handleKeyUp = this.handleKeyUp.bind(this);
        
        // Element reference
        this.element = null;
        
        console.log('🖱️ InteractionEngine initialized with unified interaction handling');
    }
    
    /**
     * 🔗 Attach to DOM element and set up event listeners
     */
    attachToElement(element, callbacks = {}) {
        this.element = element;
        this.callbacks = { ...this.callbacks, ...callbacks };
        
        // Mouse events
        element.addEventListener('mousedown', this.handleMouseDown);
        document.addEventListener('mousemove', this.handleMouseMove);
        document.addEventListener('mouseup', this.handleMouseUp);
        
        // Wheel events for zoom
        if (this.config.enableZoom) {
            element.addEventListener('wheel', this.handleWheel, { passive: false });
        }
        
        // Touch events
        if (this.config.enableTouch) {
            element.addEventListener('touchstart', this.handleTouchStart, { passive: false });
            element.addEventListener('touchmove', this.handleTouchMove, { passive: false });
            element.addEventListener('touchend', this.handleTouchEnd, { passive: false });
        }
        
        // Keyboard events
        if (this.config.enableKeyboard) {
            element.setAttribute('tabindex', '0'); // Make element focusable
            element.addEventListener('keydown', this.handleKeyDown);
            element.addEventListener('keyup', this.handleKeyUp);
        }
        
        // Context menu (disable for better UX)
        element.addEventListener('contextmenu', (e) => e.preventDefault());
        
        console.log('🖱️ InteractionEngine attached to element');
    }
    
    /**
     * 🖱️ Handle mouse down events
     */
    handleMouseDown(event) {
        event.preventDefault();
        
        const rect = this.element.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        this.mousePosition = { x, y };
        this.dragStart = { x, y };
        this.panStart = { x: event.clientX, y: event.clientY };
        
        // Check if clicking on a node
        const target = event.target;
        if (target.classList.contains('groggy-node')) {
            this.handleNodeMouseDown(target, event);
        } else {
            this.handleCanvasMouseDown(event);
        }
    }
    
    /**
     * 🖱️ Handle mouse move events
     */
    handleMouseMove(event) {
        const rect = this.element.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        this.mousePosition = { x, y };
        
        if (this.isDragging && this.draggedNode) {
            this.handleNodeDrag(event);
        } else if (this.isPanning) {
            this.handlePanMove(event);
        } else {
            this.handleMouseHover(event.target);
        }
    }
    
    /**
     * 🖱️ Handle mouse up events
     */
    handleMouseUp(event) {
        if (this.isDragging && this.draggedNode) {
            this.handleNodeDragEnd();
        } else if (this.isPanning) {
            this.handlePanEnd();
        }
        
        this.isDragging = false;
        this.isPanning = false;
        this.draggedNode = null;
    }
    
    /**
     * 🎯 Handle node mouse down
     */
    handleNodeMouseDown(nodeElement, event) {
        const nodeId = this.getNodeIdFromElement(nodeElement);
        
        if (this.config.enableDrag) {
            const dragDistance = Math.sqrt(
                Math.pow(event.clientX - this.dragStart.x, 2) +
                Math.pow(event.clientY - this.dragStart.y, 2)
            );
            
            if (dragDistance < this.config.dragThreshold) {
                // Handle click
                this.handleNodeClick(nodeId, event);
            } else {
                // Start drag
                this.isDragging = true;
                this.draggedNode = nodeId;
                nodeElement.style.cursor = 'grabbing';
                
                if (this.callbacks.onNodeDrag) {
                    this.callbacks.onNodeDrag(nodeId, this.getTransformedPosition(this.mousePosition), true);
                }
            }
        } else {
            this.handleNodeClick(nodeId, event);
        }
    }
    
    /**
     * 🎯 Handle node click
     */
    handleNodeClick(nodeId, event) {
        const currentTime = Date.now();
        const timeSinceLastClick = currentTime - this.lastClickTime;
        
        // Handle double click
        if (timeSinceLastClick < 300) {
            clearTimeout(this.clickTimeout);
            this.handleNodeDoubleClick(nodeId, event);
        } else {
            // Delay single click to check for double click
            this.clickTimeout = setTimeout(() => {
                if (this.callbacks.onNodeClick) {
                    this.callbacks.onNodeClick(nodeId, event);
                }
            }, 300);
        }
        
        this.lastClickTime = currentTime;
    }
    
    /**
     * 🎯 Handle node double click
     */
    handleNodeDoubleClick(nodeId, event) {
        if (this.callbacks.onNodeDoubleClick) {
            this.callbacks.onNodeDoubleClick(nodeId, event);
        }
    }
    
    /**
     * 🖱️ Handle node drag
     */
    handleNodeDrag(event) {
        if (!this.draggedNode || !this.callbacks.onNodeDrag) return;
        
        const position = this.getTransformedPosition(this.mousePosition);
        
        this.callbacks.onNodeDrag(this.draggedNode, position, true);
    }
    
    /**
     * 🖱️ Handle node drag end
     */
    handleNodeDragEnd() {
        if (!this.draggedNode) return;
        
        // Reset cursor
        const nodeElement = document.getElementById(`node-${this.draggedNode}`);
        if (nodeElement) {
            nodeElement.style.cursor = 'grab';
        }
        
        if (this.callbacks.onNodeDrag) {
            this.callbacks.onNodeDrag(this.draggedNode, this.getTransformedPosition(this.mousePosition), false);
        }
        
        this.draggedNode = null;
    }
    
    /**
     * 🖱️ Handle canvas mouse down (for panning)
     */
    handleCanvasMouseDown(event) {
        if (this.config.enablePan) {
            this.isPanning = true;
            this.element.style.cursor = 'grabbing';
        }
    }
    
    /**
     * 🖱️ Handle pan move
     */
    handlePanMove(event) {
        if (!this.isPanning || !this.callbacks.onPan) return;
        
        const deltaX = (event.clientX - this.panStart.x) * this.config.panSensitivity;
        const deltaY = (event.clientY - this.panStart.y) * this.config.panSensitivity;
        
        this.currentTranslate.x += deltaX;
        this.currentTranslate.y += deltaY;
        
        this.callbacks.onPan({ x: deltaX, y: deltaY });
        
        this.panStart = { x: event.clientX, y: event.clientY };
    }
    
    /**
     * 🖱️ Handle pan end
     */
    handlePanEnd() {
        this.isPanning = false;
        this.element.style.cursor = 'grab';
    }
    
    /**
     * 🔍 Handle mouse hover
     */
    handleMouseHover(target) {
        if (!this.config.enableHover) return;
        
        // Clear existing hover timeout
        if (this.hoverTimeout) {
            clearTimeout(this.hoverTimeout);
        }
        
        this.hoverTimeout = setTimeout(() => {
            if (target.classList.contains('groggy-node')) {
                const nodeId = this.getNodeIdFromElement(target);
                if (nodeId !== this.hoveredElement) {
                    this.hoveredElement = nodeId;
                    if (this.callbacks.onNodeHover) {
                        this.callbacks.onNodeHover(nodeId);
                    }
                }
            } else if (this.hoveredElement) {
                this.hoveredElement = null;
                if (this.callbacks.onNodeHover) {
                    this.callbacks.onNodeHover(null);
                }
            }
        }, this.config.hoverDelay);
    }
    
    /**
     * 🎡 Handle wheel events for zoom
     */
    handleWheel(event) {
        if (!this.config.enableZoom) return;
        
        event.preventDefault();
        
        const rect = this.element.getBoundingClientRect();
        const centerX = event.clientX - rect.left;
        const centerY = event.clientY - rect.top;
        
        const delta = -event.deltaY * this.config.zoomSensitivity;
        const scaleFactor = Math.exp(delta);
        const newScale = Math.max(
            this.config.zoomExtent[0],
            Math.min(this.config.zoomExtent[1], this.currentScale * scaleFactor)
        );
        
        if (newScale !== this.currentScale) {
            // Zoom towards mouse position
            const scaleRatio = newScale / this.currentScale;
            this.currentTranslate.x = centerX - (centerX - this.currentTranslate.x) * scaleRatio;
            this.currentTranslate.y = centerY - (centerY - this.currentTranslate.y) * scaleRatio;
            
            this.currentScale = newScale;
            
            if (this.callbacks.onZoom) {
                this.callbacks.onZoom(this.currentScale, { x: centerX, y: centerY });
            }
        }
    }
    
    /**
     * 📱 Handle touch start events
     */
    handleTouchStart(event) {
        event.preventDefault();
        
        const touches = Array.from(event.touches);
        
        if (touches.length === 1) {
            // Single touch - treat as mouse down
            const touch = touches[0];
            const rect = this.element.getBoundingClientRect();
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;
            
            this.mousePosition = { x, y };
            this.dragStart = { x, y };
            
            const target = document.elementFromPoint(touch.clientX, touch.clientY);
            if (target && target.classList.contains('groggy-node')) {
                this.handleNodeMouseDown(target, { 
                    clientX: touch.clientX, 
                    clientY: touch.clientY,
                    preventDefault: () => {},
                    target
                });
            } else {
                this.handleCanvasMouseDown({ 
                    clientX: touch.clientX, 
                    clientY: touch.clientY 
                });
            }
        } else if (touches.length === 2) {
            // Two finger gesture - zoom/pan
            this.handleTwoFingerStart(touches);
        }
        
        // Store touch positions
        touches.forEach(touch => {
            this.touches.set(touch.identifier, {
                x: touch.clientX,
                y: touch.clientY,
                startX: touch.clientX,
                startY: touch.clientY
            });
        });
    }
    
    /**
     * 📱 Handle touch move events
     */
    handleTouchMove(event) {
        event.preventDefault();
        
        const touches = Array.from(event.touches);
        
        if (touches.length === 1) {
            // Single touch - treat as mouse move
            const touch = touches[0];
            const rect = this.element.getBoundingClientRect();
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;
            
            this.mousePosition = { x, y };
            
            if (this.isDragging && this.draggedNode) {
                this.handleNodeDrag({ clientX: touch.clientX, clientY: touch.clientY });
            } else if (this.isPanning) {
                this.handlePanMove({ clientX: touch.clientX, clientY: touch.clientY });
            }
        } else if (touches.length === 2) {
            this.handleTwoFingerMove(touches);
        }
        
        // Update stored touch positions
        touches.forEach(touch => {
            if (this.touches.has(touch.identifier)) {
                const stored = this.touches.get(touch.identifier);
                stored.x = touch.clientX;
                stored.y = touch.clientY;
            }
        });
    }
    
    /**
     * 📱 Handle touch end events
     */
    handleTouchEnd(event) {
        event.preventDefault();
        
        // Remove ended touches
        const remainingTouches = Array.from(event.touches);
        const remainingIds = new Set(remainingTouches.map(t => t.identifier));
        
        for (const [id] of this.touches) {
            if (!remainingIds.has(id)) {
                this.touches.delete(id);
            }
        }
        
        // Handle drag/pan end
        if (remainingTouches.length === 0) {
            this.handleMouseUp({});
        }
    }
    
    /**
     * ✌️ Handle two finger start
     */
    handleTwoFingerStart(touches) {
        const touch1 = touches[0];
        const touch2 = touches[1];
        
        const centerX = (touch1.clientX + touch2.clientX) / 2;
        const centerY = (touch1.clientY + touch2.clientY) / 2;
        
        this.lastTouchCenter = { x: centerX, y: centerY };
        this.lastTouchDistance = Math.sqrt(
            Math.pow(touch2.clientX - touch1.clientX, 2) +
            Math.pow(touch2.clientY - touch1.clientY, 2)
        );
        
        this.isZooming = true;
    }
    
    /**
     * ✌️ Handle two finger move (pinch zoom)
     */
    handleTwoFingerMove(touches) {
        if (!this.isZooming || touches.length !== 2) return;
        
        const touch1 = touches[0];
        const touch2 = touches[1];
        
        const centerX = (touch1.clientX + touch2.clientX) / 2;
        const centerY = (touch1.clientY + touch2.clientY) / 2;
        const distance = Math.sqrt(
            Math.pow(touch2.clientX - touch1.clientX, 2) +
            Math.pow(touch2.clientY - touch1.clientY, 2)
        );
        
        if (this.lastTouchDistance > 0) {
            // Handle zoom
            const scaleFactor = distance / this.lastTouchDistance;
            const newScale = Math.max(
                this.config.zoomExtent[0],
                Math.min(this.config.zoomExtent[1], this.currentScale * scaleFactor)
            );
            
            if (newScale !== this.currentScale) {
                const rect = this.element.getBoundingClientRect();
                const localCenterX = centerX - rect.left;
                const localCenterY = centerY - rect.top;
                
                const scaleRatio = newScale / this.currentScale;
                this.currentTranslate.x = localCenterX - (localCenterX - this.currentTranslate.x) * scaleRatio;
                this.currentTranslate.y = localCenterY - (localCenterY - this.currentTranslate.y) * scaleRatio;
                
                this.currentScale = newScale;
                
                if (this.callbacks.onZoom) {
                    this.callbacks.onZoom(this.currentScale, { x: localCenterX, y: localCenterY });
                }
            }
            
            // Handle pan
            const deltaX = centerX - this.lastTouchCenter.x;
            const deltaY = centerY - this.lastTouchCenter.y;
            
            if (Math.abs(deltaX) > 2 || Math.abs(deltaY) > 2) {
                this.currentTranslate.x += deltaX * this.config.touchSensitivity;
                this.currentTranslate.y += deltaY * this.config.touchSensitivity;
                
                if (this.callbacks.onPan) {
                    this.callbacks.onPan({ x: deltaX, y: deltaY });
                }
            }
        }
        
        this.lastTouchCenter = { x: centerX, y: centerY };
        this.lastTouchDistance = distance;
    }
    
    /**
     * ⌨️ Handle keyboard events
     */
    handleKeyDown(event) {
        if (!this.config.enableKeyboard || !this.callbacks.onKeyboard) return;
        
        this.callbacks.onKeyboard({
            type: 'keydown',
            key: event.key,
            code: event.code,
            ctrlKey: event.ctrlKey,
            metaKey: event.metaKey,
            shiftKey: event.shiftKey,
            altKey: event.altKey
        });
        
        // Handle built-in shortcuts
        if (event.key === 'Escape') {
            // Clear selection
            if (this.callbacks.onSelection) {
                this.callbacks.onSelection([]);
            }
        }
    }
    
    /**
     * ⌨️ Handle key up events
     */
    handleKeyUp(event) {
        if (!this.config.enableKeyboard || !this.callbacks.onKeyboard) return;
        
        this.callbacks.onKeyboard({
            type: 'keyup',
            key: event.key,
            code: event.code,
            ctrlKey: event.ctrlKey,
            metaKey: event.metaKey,
            shiftKey: event.shiftKey,
            altKey: event.altKey
        });
    }
    
    /**
     * 🔧 Process events (called by main update loop)
     */
    handleEvents(eventData) {
        // This method can be used for processing events that come from external sources
        // like WebSocket messages or Python callbacks
        const { events } = eventData;
        
        if (events) {
            events.forEach(event => {
                switch (event.type) {
                    case 'select':
                        if (this.callbacks.onSelection) {
                            this.callbacks.onSelection(event.nodeIds);
                        }
                        break;
                    case 'hover':
                        if (this.callbacks.onNodeHover) {
                            this.callbacks.onNodeHover(event.nodeId);
                        }
                        break;
                    case 'zoom':
                        this.currentScale = event.scale;
                        if (this.callbacks.onZoom) {
                            this.callbacks.onZoom(event.scale, event.center);
                        }
                        break;
                    case 'pan':
                        this.currentTranslate = event.translate;
                        if (this.callbacks.onPan) {
                            this.callbacks.onPan(event.delta);
                        }
                        break;
                }
            });
        }
    }
    
    /**
     * 🎯 Get node ID from DOM element
     */
    getNodeIdFromElement(element) {
        const id = element.getAttribute('id');
        return id ? id.replace('node-', '') : null;
    }
    
    /**
     * 📐 Transform screen position to graph coordinates
     */
    getTransformedPosition(screenPos) {
        return {
            x: (screenPos.x - this.currentTranslate.x) / this.currentScale,
            y: (screenPos.y - this.currentTranslate.y) / this.currentScale
        };
    }
    
    /**
     * ⚙️ Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
    }
    
    /**
     * 📊 Get current interaction state
     */
    getState() {
        return {
            isDragging: this.isDragging,
            isPanning: this.isPanning,
            isZooming: this.isZooming,
            draggedNode: this.draggedNode,
            hoveredElement: this.hoveredElement,
            currentScale: this.currentScale,
            currentTranslate: { ...this.currentTranslate },
            mousePosition: { ...this.mousePosition }
        };
    }
    
    /**
     * 🧹 Cleanup and destroy
     */
    destroy() {
        if (this.element) {
            this.element.removeEventListener('mousedown', this.handleMouseDown);
            this.element.removeEventListener('wheel', this.handleWheel);
            this.element.removeEventListener('touchstart', this.handleTouchStart);
            this.element.removeEventListener('touchmove', this.handleTouchMove);
            this.element.removeEventListener('touchend', this.handleTouchEnd);
            this.element.removeEventListener('keydown', this.handleKeyDown);
            this.element.removeEventListener('keyup', this.handleKeyUp);
        }
        
        document.removeEventListener('mousemove', this.handleMouseMove);
        document.removeEventListener('mouseup', this.handleMouseUp);
        
        if (this.clickTimeout) {
            clearTimeout(this.clickTimeout);
        }
        
        if (this.hoverTimeout) {
            clearTimeout(this.hoverTimeout);
        }
        
        this.touches.clear();
        
        console.log('🖱️ InteractionEngine destroyed');
    }
}