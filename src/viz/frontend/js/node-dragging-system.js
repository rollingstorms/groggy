/**
 * Advanced Node Dragging and Repositioning System
 * Provides smooth, responsive node dragging with physics simulation,
 * collision detection, and multi-node dragging capabilities.
 */

class NodeDraggingSystem {
    constructor(graphRenderer, config = {}) {
        this.graphRenderer = graphRenderer;
        this.canvas = graphRenderer.canvas;
        this.ctx = this.canvas.getContext('2d');
        
        // Configuration with intelligent defaults
        this.config = {
            enableDragging: true,
            enableMultiDrag: true,
            enablePhysicsSimulation: true,
            enableCollisionDetection: false,
            enableSnapToGrid: false,
            enableMagneticSnapping: true,
            
            // Drag behavior
            dragThreshold: 5,           // Minimum distance to start drag
            smoothingFactor: 0.85,      // Velocity smoothing (0-1)
            dampingFactor: 0.95,        // Velocity damping (0-1)
            
            // Physics simulation
            springConstant: 0.01,       // Spring force strength
            repulsionForce: 50,         // Node repulsion strength
            attractionForce: 0.1,       // Node attraction strength
            maxVelocity: 10,            // Maximum node velocity
            
            // Grid and snapping
            gridSize: 20,               // Grid cell size for snapping
            magneticRadius: 30,         // Magnetic snapping radius
            magneticStrength: 0.3,      // Magnetic attraction strength
            
            // Visual feedback
            showDragGuides: true,       // Show alignment guides during drag
            showVelocityVectors: false, // Debug: show velocity vectors
            highlightConnections: true, // Highlight connections of dragged nodes
            
            ...config
        };
        
        // Drag state management
        this.dragState = {
            isDragging: false,
            draggedNodes: new Set(),
            dragOffset: new Map(),      // Node ID -> {x, y} offset
            dragStartPos: null,
            lastMousePos: null,
            dragDistance: 0,
            
            // Multi-drag state
            isMultiDrag: false,
            selectionBox: null,
            
            // Physics state
            nodeVelocities: new Map(),  // Node ID -> {vx, vy}
            forces: new Map(),          // Node ID -> {fx, fy}
            
            // Animation state
            animationFrame: null,
            lastUpdateTime: 0
        };
        
        // Interaction state
        this.interactionState = {
            hoveredNode: null,
            cursor: 'default',
            guides: {
                horizontal: [],
                vertical: [],
                visible: false
            }
        };
        
        // Performance optimization
        this.performance = {
            lastFrameTime: 0,
            frameCount: 0,
            avgFrameTime: 16.67,
            adaptiveQuality: true,
            qualityLevel: 1.0
        };
        
        this.initializeEventListeners();
        this.initializePhysicsEngine();
    }
    
    /**
     * Initialize all event listeners for dragging interactions
     */
    initializeEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.handleMouseLeave.bind(this));
        
        // Touch events for mobile support
        this.canvas.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
        this.canvas.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        this.canvas.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });
        
        // Keyboard events for modifier keys
        document.addEventListener('keydown', this.handleKeyDown.bind(this));
        document.addEventListener('keyup', this.handleKeyUp.bind(this));
        
        // Context menu prevention during drag
        this.canvas.addEventListener('contextmenu', (e) => {
            if (this.dragState.isDragging) {
                e.preventDefault();
            }
        });
    }
    
    /**
     * Initialize physics simulation engine
     */
    initializePhysicsEngine() {
        this.physics = {
            enabled: this.config.enablePhysicsSimulation,
            gravity: { x: 0, y: 0 },
            friction: 0.1,
            
            // Force calculation methods
            calculateSpringForces: this.calculateSpringForces.bind(this),
            calculateRepulsionForces: this.calculateRepulsionForces.bind(this),
            calculateAttractionForces: this.calculateAttractionForces.bind(this),
            applyForces: this.applyForces.bind(this),
            
            // Collision detection
            detectCollisions: this.detectCollisions.bind(this),
            resolveCollisions: this.resolveCollisions.bind(this)
        };
    }
    
    /**
     * Handle mouse down events - start of potential drag operation
     */
    handleMouseDown(event) {
        if (!this.config.enableDragging) return;
        
        const mousePos = this.getMousePosition(event);
        const hitNode = this.getNodeAtPosition(mousePos.x, mousePos.y);
        
        if (hitNode) {
            this.startDrag(hitNode, mousePos, event);
        } else if (this.config.enableMultiDrag && event.shiftKey) {
            this.startMultiSelection(mousePos);
        }
    }
    
    /**
     * Handle mouse move events - during drag operation
     */
    handleMouseMove(event) {
        const mousePos = this.getMousePosition(event);
        this.interactionState.lastMousePos = mousePos;
        
        if (this.dragState.isDragging) {
            this.updateDrag(mousePos, event);
        } else if (this.dragState.isMultiDrag) {
            this.updateMultiSelection(mousePos);
        } else {
            this.updateHover(mousePos);
        }
        
        this.updateCursor();
    }
    
    /**
     * Handle mouse up events - end of drag operation
     */
    handleMouseUp(event) {
        if (this.dragState.isDragging) {
            this.endDrag(event);
        } else if (this.dragState.isMultiDrag) {
            this.endMultiSelection(event);
        }
    }
    
    /**
     * Handle mouse leave events - cancel operations
     */
    handleMouseLeave(event) {
        if (this.dragState.isDragging) {
            this.endDrag(event, true); // Force end
        }
        this.clearHover();
    }
    
    /**
     * Start drag operation for a specific node
     */
    startDrag(node, mousePos, event) {
        this.dragState.isDragging = true;
        this.dragState.dragStartPos = { ...mousePos };
        this.dragState.lastMousePos = { ...mousePos };
        this.dragState.dragDistance = 0;
        
        // Determine which nodes to drag
        const nodesToDrag = this.determineNodesToDrag(node, event);
        
        // Setup drag state for each node
        for (const dragNode of nodesToDrag) {
            this.dragState.draggedNodes.add(dragNode.id);
            this.dragState.dragOffset.set(dragNode.id, {
                x: mousePos.x - dragNode.x,
                y: mousePos.y - dragNode.y
            });
            
            // Initialize physics state
            if (!this.dragState.nodeVelocities.has(dragNode.id)) {
                this.dragState.nodeVelocities.set(dragNode.id, { vx: 0, vy: 0 });
            }
            if (!this.dragState.forces.has(dragNode.id)) {
                this.dragState.forces.set(dragNode.id, { fx: 0, fy: 0 });
            }
        }
        
        // Visual feedback
        this.canvas.style.cursor = 'grabbing';
        if (this.config.highlightConnections) {
            this.highlightNodeConnections(nodesToDrag);
        }
        
        // Start physics simulation if enabled
        if (this.config.enablePhysicsSimulation) {
            this.startPhysicsSimulation();
        }
        
        // Emit drag start event
        this.emitEvent('dragStart', {
            nodes: Array.from(nodesToDrag),
            startPosition: mousePos,
            modifiers: this.getModifierKeys(event)
        });
    }
    
    /**
     * Update drag operation
     */
    updateDrag(mousePos, event) {
        if (!this.dragState.isDragging) return;
        
        const deltaX = mousePos.x - this.dragState.lastMousePos.x;
        const deltaY = mousePos.y - this.dragState.lastMousePos.y;
        
        this.dragState.dragDistance += Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        
        // Update positions for all dragged nodes
        for (const nodeId of this.dragState.draggedNodes) {
            const node = this.graphRenderer.getNode(nodeId);
            if (!node) continue;
            
            const offset = this.dragState.dragOffset.get(nodeId);
            let newX = mousePos.x - offset.x;
            let newY = mousePos.y - offset.y;
            
            // Apply constraints and snapping
            const constrained = this.applyConstraints(newX, newY, node);
            newX = constrained.x;
            newY = constrained.y;
            
            // Update node position
            const oldPos = { x: node.x, y: node.y };
            node.x = newX;
            node.y = newY;
            
            // Update velocity for physics simulation
            const velocity = this.dragState.nodeVelocities.get(nodeId);
            if (velocity) {
                velocity.vx = (newX - oldPos.x) * this.config.smoothingFactor + velocity.vx * (1 - this.config.smoothingFactor);
                velocity.vy = (newY - oldPos.y) * this.config.smoothingFactor + velocity.vy * (1 - this.config.smoothingFactor);
            }
        }
        
        // Update alignment guides
        if (this.config.showDragGuides) {
            this.updateAlignmentGuides();
        }
        
        // Handle physics simulation
        if (this.config.enablePhysicsSimulation) {
            this.updatePhysicsSimulation();
        }
        
        this.dragState.lastMousePos = { ...mousePos };
        
        // Request redraw
        this.graphRenderer.requestRedraw();
        
        // Emit drag update event
        this.emitEvent('dragUpdate', {
            nodes: Array.from(this.dragState.draggedNodes).map(id => this.graphRenderer.getNode(id)),
            delta: { x: deltaX, y: deltaY },
            totalDistance: this.dragState.dragDistance,
            modifiers: this.getModifierKeys(event)
        });
    }
    
    /**
     * End drag operation
     */
    endDrag(event, forced = false) {
        if (!this.dragState.isDragging) return;
        
        const draggedNodes = Array.from(this.dragState.draggedNodes).map(id => this.graphRenderer.getNode(id));
        
        // Apply final physics if enabled
        if (this.config.enablePhysicsSimulation && !forced) {
            this.applyFinalPhysics();
        }
        
        // Clean up drag state
        this.dragState.isDragging = false;
        this.dragState.draggedNodes.clear();
        this.dragState.dragOffset.clear();
        this.dragState.dragStartPos = null;
        this.dragState.lastMousePos = null;
        this.dragState.dragDistance = 0;
        
        // Clean up visual feedback
        this.canvas.style.cursor = 'default';
        this.clearAlignmentGuides();
        this.clearConnectionHighlights();
        
        // Stop physics simulation
        if (this.dragState.animationFrame) {
            cancelAnimationFrame(this.dragState.animationFrame);
            this.dragState.animationFrame = null;
        }
        
        // Emit drag end event
        this.emitEvent('dragEnd', {
            nodes: draggedNodes,
            finalPositions: draggedNodes.map(node => ({ id: node.id, x: node.x, y: node.y })),
            totalDistance: this.dragState.dragDistance,
            forced: forced,
            modifiers: this.getModifierKeys(event)
        });
        
        // Request final redraw
        this.graphRenderer.requestRedraw();
    }
    
    /**
     * Determine which nodes should be dragged based on selection and modifiers
     */
    determineNodesToDrag(primaryNode, event) {
        const nodes = [primaryNode];
        
        // If the primary node is part of a selection, drag all selected nodes
        if (this.graphRenderer.selectionManager && this.graphRenderer.selectionManager.isNodeSelected(primaryNode.id)) {
            const selectedNodes = this.graphRenderer.selectionManager.getSelectedNodes();
            nodes.push(...selectedNodes.filter(node => node.id !== primaryNode.id));
        }
        
        // If Ctrl/Cmd is held, add connected nodes
        if (event.ctrlKey || event.metaKey) {
            const connectedNodes = this.getConnectedNodes(primaryNode);
            nodes.push(...connectedNodes.filter(node => !nodes.includes(node)));
        }
        
        return nodes;
    }
    
    /**
     * Apply position constraints and snapping
     */
    applyConstraints(x, y, node) {
        let constrainedX = x;
        let constrainedY = y;
        
        // Canvas boundary constraints
        const nodeRadius = node.radius || 10;
        constrainedX = Math.max(nodeRadius, Math.min(this.canvas.width - nodeRadius, constrainedX));
        constrainedY = Math.max(nodeRadius, Math.min(this.canvas.height - nodeRadius, constrainedY));
        
        // Grid snapping
        if (this.config.enableSnapToGrid) {
            const snapped = this.snapToGrid(constrainedX, constrainedY);
            constrainedX = snapped.x;
            constrainedY = snapped.y;
        }
        
        // Magnetic snapping to other nodes
        if (this.config.enableMagneticSnapping) {
            const magneticSnap = this.calculateMagneticSnapping(constrainedX, constrainedY, node);
            constrainedX = magneticSnap.x;
            constrainedY = magneticSnap.y;
        }
        
        // Collision detection
        if (this.config.enableCollisionDetection) {
            const collisionFree = this.resolveCollisions(constrainedX, constrainedY, node);
            constrainedX = collisionFree.x;
            constrainedY = collisionFree.y;
        }
        
        return { x: constrainedX, y: constrainedY };
    }
    
    /**
     * Snap position to grid
     */
    snapToGrid(x, y) {
        const gridSize = this.config.gridSize;
        return {
            x: Math.round(x / gridSize) * gridSize,
            y: Math.round(y / gridSize) * gridSize
        };
    }
    
    /**
     * Calculate magnetic snapping to nearby nodes
     */
    calculateMagneticSnapping(x, y, draggedNode) {
        let snapX = x;
        let snapY = y;
        let minDistance = this.config.magneticRadius;
        
        const allNodes = this.graphRenderer.getNodes();
        
        for (const node of allNodes) {
            if (node.id === draggedNode.id || this.dragState.draggedNodes.has(node.id)) {
                continue;
            }
            
            const dx = Math.abs(x - node.x);
            const dy = Math.abs(y - node.y);
            
            // Horizontal alignment
            if (dy < this.config.magneticRadius && dx < minDistance) {
                snapX = node.x;
                minDistance = dx;
            }
            
            // Vertical alignment
            if (dx < this.config.magneticRadius && dy < minDistance) {
                snapY = node.y;
                minDistance = dy;
            }
        }
        
        // Apply magnetic strength
        const strength = this.config.magneticStrength;
        return {
            x: x + (snapX - x) * strength,
            y: y + (snapY - y) * strength
        };
    }
    
    /**
     * Update alignment guides during drag
     */
    updateAlignmentGuides() {
        const guides = this.interactionState.guides;
        guides.horizontal = [];
        guides.vertical = [];
        
        if (this.dragState.draggedNodes.size === 0) return;
        
        const draggedNode = this.graphRenderer.getNode(Array.from(this.dragState.draggedNodes)[0]);
        if (!draggedNode) return;
        
        const allNodes = this.graphRenderer.getNodes();
        const tolerance = 10;
        
        for (const node of allNodes) {
            if (this.dragState.draggedNodes.has(node.id)) continue;
            
            // Horizontal alignment
            if (Math.abs(draggedNode.y - node.y) < tolerance) {
                guides.horizontal.push({
                    y: node.y,
                    x1: Math.min(draggedNode.x, node.x) - 50,
                    x2: Math.max(draggedNode.x, node.x) + 50
                });
            }
            
            // Vertical alignment
            if (Math.abs(draggedNode.x - node.x) < tolerance) {
                guides.vertical.push({
                    x: node.x,
                    y1: Math.min(draggedNode.y, node.y) - 50,
                    y2: Math.max(draggedNode.y, node.y) + 50
                });
            }
        }
        
        guides.visible = guides.horizontal.length > 0 || guides.vertical.length > 0;
    }
    
    /**
     * Clear alignment guides
     */
    clearAlignmentGuides() {
        this.interactionState.guides.horizontal = [];
        this.interactionState.guides.vertical = [];
        this.interactionState.guides.visible = false;
    }
    
    /**
     * Physics simulation methods
     */
    startPhysicsSimulation() {
        if (this.dragState.animationFrame) return;
        
        this.dragState.lastUpdateTime = performance.now();
        this.updatePhysicsLoop();
    }
    
    updatePhysicsLoop() {
        const currentTime = performance.now();
        const deltaTime = Math.min(currentTime - this.dragState.lastUpdateTime, 16.67);
        
        this.updatePhysicsSimulation(deltaTime / 1000);
        
        this.dragState.lastUpdateTime = currentTime;
        this.dragState.animationFrame = requestAnimationFrame(() => this.updatePhysicsLoop());
    }
    
    updatePhysicsSimulation(deltaTime = 0.016) {
        if (!this.config.enablePhysicsSimulation) return;
        
        // Calculate forces
        this.physics.calculateSpringForces();
        this.physics.calculateRepulsionForces();
        
        // Apply forces to velocities and positions
        this.physics.applyForces(deltaTime);
        
        // Handle collisions
        if (this.config.enableCollisionDetection) {
            this.physics.detectCollisions();
        }
    }
    
    calculateSpringForces() {
        const edges = this.graphRenderer.getEdges();
        const springConstant = this.config.springConstant;
        
        for (const edge of edges) {
            const sourceNode = this.graphRenderer.getNode(edge.source);
            const targetNode = this.graphRenderer.getNode(edge.target);
            
            if (!sourceNode || !targetNode) continue;
            
            const dx = targetNode.x - sourceNode.x;
            const dy = targetNode.y - sourceNode.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance === 0) continue;
            
            const idealLength = edge.length || 100;
            const force = springConstant * (distance - idealLength);
            
            const fx = (dx / distance) * force;
            const fy = (dy / distance) * force;
            
            // Apply forces
            const sourceForce = this.dragState.forces.get(sourceNode.id) || { fx: 0, fy: 0 };
            const targetForce = this.dragState.forces.get(targetNode.id) || { fx: 0, fy: 0 };
            
            sourceForce.fx += fx;
            sourceForce.fy += fy;
            targetForce.fx -= fx;
            targetForce.fy -= fy;
            
            this.dragState.forces.set(sourceNode.id, sourceForce);
            this.dragState.forces.set(targetNode.id, targetForce);
        }
    }
    
    calculateRepulsionForces() {
        const nodes = this.graphRenderer.getNodes();
        const repulsionForce = this.config.repulsionForce;
        
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const nodeA = nodes[i];
                const nodeB = nodes[j];
                
                const dx = nodeB.x - nodeA.x;
                const dy = nodeB.y - nodeA.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance === 0 || distance > 200) continue;
                
                const force = repulsionForce / (distance * distance);
                const fx = (dx / distance) * force;
                const fy = (dy / distance) * force;
                
                const forceA = this.dragState.forces.get(nodeA.id) || { fx: 0, fy: 0 };
                const forceB = this.dragState.forces.get(nodeB.id) || { fx: 0, fy: 0 };
                
                forceA.fx -= fx;
                forceA.fy -= fy;
                forceB.fx += fx;
                forceB.fy += fy;
                
                this.dragState.forces.set(nodeA.id, forceA);
                this.dragState.forces.set(nodeB.id, forceB);
            }
        }
    }
    
    applyForces(deltaTime) {
        const dampingFactor = this.config.dampingFactor;
        const maxVelocity = this.config.maxVelocity;
        
        for (const [nodeId, force] of this.dragState.forces) {
            // Skip nodes that are being dragged
            if (this.dragState.draggedNodes.has(nodeId)) continue;
            
            const node = this.graphRenderer.getNode(nodeId);
            const velocity = this.dragState.nodeVelocities.get(nodeId);
            
            if (!node || !velocity) continue;
            
            // Update velocity
            velocity.vx = (velocity.vx + force.fx * deltaTime) * dampingFactor;
            velocity.vy = (velocity.vy + force.fy * deltaTime) * dampingFactor;
            
            // Limit velocity
            const speed = Math.sqrt(velocity.vx * velocity.vx + velocity.vy * velocity.vy);
            if (speed > maxVelocity) {
                velocity.vx = (velocity.vx / speed) * maxVelocity;
                velocity.vy = (velocity.vy / speed) * maxVelocity;
            }
            
            // Update position
            node.x += velocity.vx * deltaTime * 60;
            node.y += velocity.vy * deltaTime * 60;
            
            // Reset forces
            force.fx = 0;
            force.fy = 0;
        }
    }
    
    applyFinalPhysics() {
        // Apply a few physics steps with reduced forces
        const originalSpring = this.config.springConstant;
        const originalRepulsion = this.config.repulsionForce;
        
        this.config.springConstant *= 0.1;
        this.config.repulsionForce *= 0.1;
        
        for (let i = 0; i < 10; i++) {
            this.updatePhysicsSimulation(0.016);
        }
        
        this.config.springConstant = originalSpring;
        this.config.repulsionForce = originalRepulsion;
    }
    
    /**
     * Collision detection and resolution
     */
    detectCollisions() {
        const nodes = this.graphRenderer.getNodes();
        const collisions = [];
        
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const nodeA = nodes[i];
                const nodeB = nodes[j];
                
                const dx = nodeB.x - nodeA.x;
                const dy = nodeB.y - nodeA.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                const minDistance = (nodeA.radius || 10) + (nodeB.radius || 10) + 5;
                
                if (distance < minDistance) {
                    collisions.push({
                        nodeA: nodeA,
                        nodeB: nodeB,
                        overlap: minDistance - distance,
                        normal: { x: dx / distance, y: dy / distance }
                    });
                }
            }
        }
        
        return collisions;
    }
    
    resolveCollisions(x, y, node) {
        const nodes = this.graphRenderer.getNodes();
        let resolvedX = x;
        let resolvedY = y;
        
        for (const otherNode of nodes) {
            if (otherNode.id === node.id || this.dragState.draggedNodes.has(otherNode.id)) {
                continue;
            }
            
            const dx = resolvedX - otherNode.x;
            const dy = resolvedY - otherNode.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const minDistance = (node.radius || 10) + (otherNode.radius || 10) + 5;
            
            if (distance < minDistance && distance > 0) {
                const pushDistance = minDistance - distance;
                const pushX = (dx / distance) * pushDistance;
                const pushY = (dy / distance) * pushDistance;
                
                resolvedX += pushX;
                resolvedY += pushY;
            }
        }
        
        return { x: resolvedX, y: resolvedY };
    }
    
    /**
     * Touch event handlers for mobile support
     */
    handleTouchStart(event) {
        event.preventDefault();
        const touch = event.touches[0];
        const mouseEvent = new MouseEvent('mousedown', {
            clientX: touch.clientX,
            clientY: touch.clientY,
            button: 0
        });
        this.handleMouseDown(mouseEvent);
    }
    
    handleTouchMove(event) {
        event.preventDefault();
        const touch = event.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        this.handleMouseMove(mouseEvent);
    }
    
    handleTouchEnd(event) {
        event.preventDefault();
        const mouseEvent = new MouseEvent('mouseup', {
            button: 0
        });
        this.handleMouseUp(mouseEvent);
    }
    
    /**
     * Keyboard event handlers
     */
    handleKeyDown(event) {
        // ESC to cancel drag
        if (event.key === 'Escape' && this.dragState.isDragging) {
            this.cancelDrag();
        }
        
        // Toggle physics simulation
        if (event.key === 'p' && event.ctrlKey) {
            this.config.enablePhysicsSimulation = !this.config.enablePhysicsSimulation;
            event.preventDefault();
        }
        
        // Toggle snap to grid
        if (event.key === 'g' && event.ctrlKey) {
            this.config.enableSnapToGrid = !this.config.enableSnapToGrid;
            event.preventDefault();
        }
    }
    
    handleKeyUp(event) {
        // Update drag behavior based on modifier keys
        if (this.dragState.isDragging) {
            this.updateDragBehavior(event);
        }
    }
    
    /**
     * Utility methods
     */
    getMousePosition(event) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
    }
    
    getNodeAtPosition(x, y) {
        const nodes = this.graphRenderer.getNodes();
        
        for (const node of nodes) {
            const distance = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);
            if (distance <= (node.radius || 10)) {
                return node;
            }
        }
        
        return null;
    }
    
    getConnectedNodes(node) {
        const edges = this.graphRenderer.getEdges();
        const connectedNodes = [];
        
        for (const edge of edges) {
            if (edge.source === node.id) {
                const target = this.graphRenderer.getNode(edge.target);
                if (target) connectedNodes.push(target);
            } else if (edge.target === node.id) {
                const source = this.graphRenderer.getNode(edge.source);
                if (source) connectedNodes.push(source);
            }
        }
        
        return connectedNodes;
    }
    
    getModifierKeys(event) {
        return {
            shift: event.shiftKey,
            ctrl: event.ctrlKey,
            alt: event.altKey,
            meta: event.metaKey
        };
    }
    
    updateHover(mousePos) {
        const hoveredNode = this.getNodeAtPosition(mousePos.x, mousePos.y);
        
        if (hoveredNode !== this.interactionState.hoveredNode) {
            this.interactionState.hoveredNode = hoveredNode;
            this.graphRenderer.requestRedraw();
        }
    }
    
    clearHover() {
        if (this.interactionState.hoveredNode) {
            this.interactionState.hoveredNode = null;
            this.graphRenderer.requestRedraw();
        }
    }
    
    updateCursor() {
        let cursor = 'default';
        
        if (this.dragState.isDragging) {
            cursor = 'grabbing';
        } else if (this.interactionState.hoveredNode) {
            cursor = 'grab';
        }
        
        if (this.canvas.style.cursor !== cursor) {
            this.canvas.style.cursor = cursor;
        }
    }
    
    highlightNodeConnections(nodes) {
        // Implementation depends on graph renderer
        if (this.graphRenderer.highlightConnections) {
            this.graphRenderer.highlightConnections(nodes);
        }
    }
    
    clearConnectionHighlights() {
        if (this.graphRenderer.clearHighlights) {
            this.graphRenderer.clearHighlights();
        }
    }
    
    cancelDrag() {
        if (this.dragState.isDragging) {
            // Restore original positions
            for (const nodeId of this.dragState.draggedNodes) {
                const node = this.graphRenderer.getNode(nodeId);
                // Implementation depends on having position history
            }
            
            this.endDrag(null, true);
        }
    }
    
    /**
     * Event system
     */
    emitEvent(eventType, data) {
        const event = new CustomEvent(`nodeDrag${eventType.charAt(0).toUpperCase() + eventType.slice(1)}`, {
            detail: data
        });
        this.canvas.dispatchEvent(event);
    }
    
    /**
     * Public API methods
     */
    
    /**
     * Enable or disable dragging
     */
    setDraggingEnabled(enabled) {
        this.config.enableDragging = enabled;
        if (!enabled && this.dragState.isDragging) {
            this.endDrag(null, true);
        }
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        Object.assign(this.config, newConfig);
        
        // Update physics engine if needed
        if ('enablePhysicsSimulation' in newConfig) {
            this.physics.enabled = newConfig.enablePhysicsSimulation;
        }
    }
    
    /**
     * Get current drag state
     */
    getDragState() {
        return {
            isDragging: this.dragState.isDragging,
            draggedNodes: Array.from(this.dragState.draggedNodes),
            dragDistance: this.dragState.dragDistance
        };
    }
    
    /**
     * Performance monitoring
     */
    updatePerformanceMetrics() {
        const currentTime = performance.now();
        const frameTime = currentTime - this.performance.lastFrameTime;
        
        this.performance.frameCount++;
        this.performance.avgFrameTime = (this.performance.avgFrameTime * 0.9) + (frameTime * 0.1);
        
        // Adaptive quality adjustment
        if (this.performance.adaptiveQuality) {
            if (this.performance.avgFrameTime > 33) { // < 30 FPS
                this.performance.qualityLevel = Math.max(0.5, this.performance.qualityLevel - 0.1);
            } else if (this.performance.avgFrameTime < 16) { // > 60 FPS
                this.performance.qualityLevel = Math.min(1.0, this.performance.qualityLevel + 0.05);
            }
        }
        
        this.performance.lastFrameTime = currentTime;
    }
    
    /**
     * Render drag-related visual elements
     */
    render(ctx) {
        // Render alignment guides
        if (this.config.showDragGuides && this.interactionState.guides.visible) {
            this.renderAlignmentGuides(ctx);
        }
        
        // Render velocity vectors for debugging
        if (this.config.showVelocityVectors) {
            this.renderVelocityVectors(ctx);
        }
    }
    
    renderAlignmentGuides(ctx) {
        ctx.save();
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        
        // Horizontal guides
        for (const guide of this.interactionState.guides.horizontal) {
            ctx.beginPath();
            ctx.moveTo(guide.x1, guide.y);
            ctx.lineTo(guide.x2, guide.y);
            ctx.stroke();
        }
        
        // Vertical guides
        for (const guide of this.interactionState.guides.vertical) {
            ctx.beginPath();
            ctx.moveTo(guide.x, guide.y1);
            ctx.lineTo(guide.x, guide.y2);
            ctx.stroke();
        }
        
        ctx.restore();
    }
    
    renderVelocityVectors(ctx) {
        ctx.save();
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 2;
        
        for (const [nodeId, velocity] of this.dragState.nodeVelocities) {
            const node = this.graphRenderer.getNode(nodeId);
            if (!node) continue;
            
            const scale = 10;
            const endX = node.x + velocity.vx * scale;
            const endY = node.y + velocity.vy * scale;
            
            ctx.beginPath();
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(endX, endY);
            ctx.stroke();
            
            // Arrow head
            const angle = Math.atan2(velocity.vy, velocity.vx);
            const headLength = 5;
            
            ctx.beginPath();
            ctx.moveTo(endX, endY);
            ctx.lineTo(
                endX - headLength * Math.cos(angle - Math.PI / 6),
                endY - headLength * Math.sin(angle - Math.PI / 6)
            );
            ctx.moveTo(endX, endY);
            ctx.lineTo(
                endX - headLength * Math.cos(angle + Math.PI / 6),
                endY - headLength * Math.sin(angle + Math.PI / 6)
            );
            ctx.stroke();
        }
        
        ctx.restore();
    }
    
    /**
     * Cleanup and destroy
     */
    destroy() {
        // Remove event listeners
        this.canvas.removeEventListener('mousedown', this.handleMouseDown);
        this.canvas.removeEventListener('mousemove', this.handleMouseMove);
        this.canvas.removeEventListener('mouseup', this.handleMouseUp);
        this.canvas.removeEventListener('mouseleave', this.handleMouseLeave);
        this.canvas.removeEventListener('touchstart', this.handleTouchStart);
        this.canvas.removeEventListener('touchmove', this.handleTouchMove);
        this.canvas.removeEventListener('touchend', this.handleTouchEnd);
        
        document.removeEventListener('keydown', this.handleKeyDown);
        document.removeEventListener('keyup', this.handleKeyUp);
        
        // Cancel animation frame
        if (this.dragState.animationFrame) {
            cancelAnimationFrame(this.dragState.animationFrame);
        }
        
        // Clear state
        this.dragState.draggedNodes.clear();
        this.dragState.dragOffset.clear();
        this.dragState.nodeVelocities.clear();
        this.dragState.forces.clear();
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NodeDraggingSystem;
}