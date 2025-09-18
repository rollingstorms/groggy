/**
 * Advanced Selection Tools System
 * Provides multiple selection methods including lasso, polygon, rectangle,
 * and magic wand selection with intelligent intersection detection.
 */

class AdvancedSelectionTools {
    constructor(graphRenderer, config = {}) {
        this.graphRenderer = graphRenderer;
        this.canvas = graphRenderer.canvas;
        this.ctx = this.canvas.getContext('2d');
        
        // Configuration with intelligent defaults
        this.config = {
            enableAdvancedSelection: true,
            defaultSelectionTool: 'rectangle',
            
            // Selection tools configuration
            tools: {
                rectangle: { enabled: true, showPreview: true },
                lasso: { enabled: true, smoothing: true, simplification: true },
                polygon: { enabled: true, snapToNodes: true, minPoints: 3 },
                circle: { enabled: true, showRadius: true },
                magicWand: { enabled: true, tolerance: 10, contiguous: false }
            },
            
            // Visual appearance
            selectionColor: '#007bff',
            selectionOpacity: 0.3,
            borderColor: '#0056b3',
            borderWidth: 2,
            borderStyle: 'dashed',
            
            // Animation settings
            enableAnimations: true,
            animationDuration: 300,
            pulseSelection: true,
            
            // Interaction settings
            multiSelectModifier: 'shift',
            toggleSelectModifier: 'ctrl',
            minSelectionSize: 10,
            maxPolygonPoints: 50,
            
            // Performance
            updateThreshold: 16, // ~60fps
            simplificationTolerance: 2,
            
            ...config
        };
        
        // Selection state
        this.selectionState = {
            isActive: false,
            currentTool: this.config.defaultSelectionTool,
            isDrawing: false,
            startPoint: null,
            currentPath: [],
            previewPath: [],
            selectedElements: new Set(),
            
            // Tool-specific state
            polygonPoints: [],
            isPolygonClosed: false,
            circleCenter: null,
            circleRadius: 0,
            
            // Performance tracking
            lastUpdate: 0,
            pendingUpdate: false
        };
        
        // Drawing state for visual feedback
        this.drawingState = {
            overlayCanvas: null,
            overlayCtx: null,
            animationFrame: null,
            showPreview: true
        };
        
        // Tool implementations
        this.tools = {
            rectangle: new RectangleSelectionTool(this),
            lasso: new LassoSelectionTool(this),
            polygon: new PolygonSelectionTool(this),
            circle: new CircleSelectionTool(this),
            magicWand: new MagicWandSelectionTool(this)
        };
        
        this.initializeSelectionSystem();
    }
    
    /**
     * Initialize the selection system
     */
    initializeSelectionSystem() {
        this.createOverlayCanvas();
        this.bindEventListeners();
        this.setupKeyboardShortcuts();
    }
    
    /**
     * Create overlay canvas for selection visualization
     */
    createOverlayCanvas() {
        this.drawingState.overlayCanvas = document.createElement('canvas');
        this.drawingState.overlayCtx = this.drawingState.overlayCanvas.getContext('2d');
        
        // Match main canvas size and position
        const updateOverlaySize = () => {
            this.drawingState.overlayCanvas.width = this.canvas.width;
            this.drawingState.overlayCanvas.height = this.canvas.height;
            this.drawingState.overlayCanvas.style.position = 'absolute';
            this.drawingState.overlayCanvas.style.top = this.canvas.offsetTop + 'px';
            this.drawingState.overlayCanvas.style.left = this.canvas.offsetLeft + 'px';
            this.drawingState.overlayCanvas.style.pointerEvents = 'none';
            this.drawingState.overlayCanvas.style.zIndex = '1000';
        };
        
        updateOverlaySize();
        
        // Insert overlay after main canvas
        this.canvas.parentNode.insertBefore(this.drawingState.overlayCanvas, this.canvas.nextSibling);
        
        // Resize observer
        if (window.ResizeObserver) {
            new ResizeObserver(updateOverlaySize).observe(this.canvas);
        } else {
            window.addEventListener('resize', updateOverlaySize);
        }
    }
    
    /**
     * Bind event listeners
     */
    bindEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('dblclick', this.handleDoubleClick.bind(this));
        
        // Touch events for mobile support
        this.canvas.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
        this.canvas.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        this.canvas.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });
        
        // Keyboard events
        document.addEventListener('keydown', this.handleKeyDown.bind(this));
        document.addEventListener('keyup', this.handleKeyUp.bind(this));
        
        // Context menu prevention during selection
        this.canvas.addEventListener('contextmenu', (e) => {
            if (this.selectionState.isDrawing) {
                e.preventDefault();
            }
        });
    }
    
    /**
     * Setup keyboard shortcuts for selection tools
     */
    setupKeyboardShortcuts() {
        this.shortcuts = {
            'KeyR': () => this.setSelectionTool('rectangle'),
            'KeyL': () => this.setSelectionTool('lasso'),
            'KeyP': () => this.setSelectionTool('polygon'),
            'KeyC': () => this.setSelectionTool('circle'),
            'KeyM': () => this.setSelectionTool('magicWand'),
            'Escape': () => this.cancelSelection(),
            'Enter': () => this.completeSelection(),
            'Delete': () => this.deleteSelected(),
            'KeyA': (e) => { if (e.ctrlKey) this.selectAll(); },
            'KeyD': (e) => { if (e.ctrlKey) this.deselectAll(); }
        };
    }
    
    /**
     * Event handlers
     */
    handleMouseDown(event) {
        if (!this.config.enableAdvancedSelection) return;
        
        const mousePos = this.getMousePosition(event);
        const tool = this.tools[this.selectionState.currentTool];
        
        if (tool && tool.handleMouseDown) {
            tool.handleMouseDown(mousePos, event);
        }
    }
    
    handleMouseMove(event) {
        if (!this.config.enableAdvancedSelection) return;
        
        const mousePos = this.getMousePosition(event);
        const tool = this.tools[this.selectionState.currentTool];
        
        if (tool && tool.handleMouseMove) {
            tool.handleMouseMove(mousePos, event);
        }
        
        this.throttledUpdate();
    }
    
    handleMouseUp(event) {
        if (!this.config.enableAdvancedSelection) return;
        
        const mousePos = this.getMousePosition(event);
        const tool = this.tools[this.selectionState.currentTool];
        
        if (tool && tool.handleMouseUp) {
            tool.handleMouseUp(mousePos, event);
        }
    }
    
    handleDoubleClick(event) {
        const tool = this.tools[this.selectionState.currentTool];
        
        if (tool && tool.handleDoubleClick) {
            const mousePos = this.getMousePosition(event);
            tool.handleDoubleClick(mousePos, event);
        }
    }
    
    handleKeyDown(event) {
        const shortcut = this.shortcuts[event.code];
        if (shortcut) {
            shortcut(event);
            event.preventDefault();
        }
    }
    
    handleKeyUp(event) {
        // Handle modifier key releases
        if (this.selectionState.isDrawing) {
            this.updateSelectionBehavior(event);
        }
    }
    
    /**
     * Touch event handlers
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
     * Selection tool management
     */
    setSelectionTool(toolName) {
        if (!this.tools[toolName] || !this.config.tools[toolName]?.enabled) {
            console.warn(`Selection tool '${toolName}' is not available or disabled`);
            return;
        }
        
        // Cancel current selection if changing tools
        if (this.selectionState.isDrawing) {
            this.cancelSelection();
        }
        
        this.selectionState.currentTool = toolName;
        this.clearOverlay();
        
        // Emit tool change event
        this.emitEvent('selectionToolChanged', { tool: toolName });
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
    
    throttledUpdate() {
        const now = performance.now();
        if (now - this.selectionState.lastUpdate < this.config.updateThreshold) {
            if (!this.selectionState.pendingUpdate) {
                this.selectionState.pendingUpdate = true;
                requestAnimationFrame(() => {
                    this.updateSelection();
                    this.selectionState.pendingUpdate = false;
                    this.selectionState.lastUpdate = performance.now();
                });
            }
            return;
        }
        
        this.updateSelection();
        this.selectionState.lastUpdate = now;
    }
    
    updateSelection() {
        if (!this.selectionState.isDrawing) return;
        
        const tool = this.tools[this.selectionState.currentTool];
        if (tool && tool.updateSelection) {
            tool.updateSelection();
        }
        
        this.renderSelection();
    }
    
    renderSelection() {
        this.clearOverlay();
        
        const tool = this.tools[this.selectionState.currentTool];
        if (tool && tool.render) {
            tool.render(this.drawingState.overlayCtx);
        }
    }
    
    clearOverlay() {
        if (this.drawingState.overlayCtx) {
            this.drawingState.overlayCtx.clearRect(0, 0, this.drawingState.overlayCanvas.width, this.drawingState.overlayCanvas.height);
        }
    }
    
    /**
     * Selection completion methods
     */
    completeSelection() {
        if (!this.selectionState.isDrawing) return;
        
        const tool = this.tools[this.selectionState.currentTool];
        if (tool && tool.completeSelection) {
            const selectedElements = tool.completeSelection();
            this.applySelection(selectedElements);
        }
        
        this.finishSelection();
    }
    
    cancelSelection() {
        if (!this.selectionState.isDrawing) return;
        
        const tool = this.tools[this.selectionState.currentTool];
        if (tool && tool.cancelSelection) {
            tool.cancelSelection();
        }
        
        this.finishSelection();
    }
    
    finishSelection() {
        this.selectionState.isDrawing = false;
        this.selectionState.isActive = false;
        this.selectionState.currentPath = [];
        this.selectionState.previewPath = [];
        this.selectionState.polygonPoints = [];
        this.selectionState.isPolygonClosed = false;
        
        this.clearOverlay();
        
        // Emit selection end event
        this.emitEvent('selectionEnd', {
            tool: this.selectionState.currentTool,
            selectedCount: this.selectionState.selectedElements.size
        });
    }
    
    applySelection(elements, mode = 'replace') {
        const modifierKeys = this.getModifierKeys();
        
        // Determine selection mode
        if (modifierKeys.shift || mode === 'add') {
            // Add to selection
            elements.forEach(el => this.selectionState.selectedElements.add(el));
        } else if (modifierKeys.ctrl || mode === 'toggle') {
            // Toggle selection
            elements.forEach(el => {
                if (this.selectionState.selectedElements.has(el)) {
                    this.selectionState.selectedElements.delete(el);
                } else {
                    this.selectionState.selectedElements.add(el);
                }
            });
        } else {
            // Replace selection
            this.selectionState.selectedElements.clear();
            elements.forEach(el => this.selectionState.selectedElements.add(el));
        }
        
        // Update graph renderer selection
        this.updateRendererSelection();
        
        // Emit selection changed event
        this.emitEvent('selectionChanged', {
            selectedElements: Array.from(this.selectionState.selectedElements),
            mode: mode
        });
    }
    
    updateRendererSelection() {
        if (this.graphRenderer.selectionManager) {
            const nodes = [];
            const edges = [];
            
            this.selectionState.selectedElements.forEach(el => {
                if (el.type === 'node') {
                    nodes.push(el.id);
                } else if (el.type === 'edge') {
                    edges.push(el.id);
                }
            });
            
            this.graphRenderer.selectionManager.setSelection(nodes, edges);
        }
    }
    
    /**
     * Public API methods
     */
    selectAll() {
        const allNodes = this.graphRenderer.getNodes().map(node => ({ type: 'node', ...node }));
        const allEdges = this.graphRenderer.getEdges().map(edge => ({ type: 'edge', ...edge }));
        
        this.applySelection([...allNodes, ...allEdges], 'replace');
    }
    
    deselectAll() {
        this.selectionState.selectedElements.clear();
        this.updateRendererSelection();
        this.emitEvent('selectionChanged', { selectedElements: [], mode: 'clear' });
    }
    
    deleteSelected() {
        const selectedElements = Array.from(this.selectionState.selectedElements);
        if (selectedElements.length === 0) return;
        
        this.emitEvent('deleteSelected', { elements: selectedElements });
        this.deselectAll();
    }
    
    getSelectedElements() {
        return Array.from(this.selectionState.selectedElements);
    }
    
    getSelectionBounds() {
        const elements = this.getSelectedElements();
        if (elements.length === 0) return null;
        
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;
        
        elements.forEach(el => {
            if (el.type === 'node') {
                const radius = el.radius || 10;
                minX = Math.min(minX, el.x - radius);
                minY = Math.min(minY, el.y - radius);
                maxX = Math.max(maxX, el.x + radius);
                maxY = Math.max(maxY, el.y + radius);
            }
        });
        
        return {
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY
        };
    }
    
    getModifierKeys() {
        return {
            shift: false, // Will be updated by actual event handlers
            ctrl: false,
            alt: false,
            meta: false
        };
    }
    
    /**
     * Event system
     */
    emitEvent(eventType, data) {
        const event = new CustomEvent(`advancedSelection${eventType.charAt(0).toUpperCase() + eventType.slice(1)}`, {
            detail: data
        });
        this.canvas.dispatchEvent(event);
    }
    
    /**
     * Configuration and state
     */
    updateConfig(newConfig) {
        Object.assign(this.config, newConfig);
        
        // Update tool configurations
        Object.keys(this.tools).forEach(toolName => {
            if (this.tools[toolName].updateConfig) {
                this.tools[toolName].updateConfig(this.config);
            }
        });
    }
    
    getCurrentTool() {
        return this.selectionState.currentTool;
    }
    
    isSelecting() {
        return this.selectionState.isDrawing;
    }
    
    /**
     * Cleanup and destroy
     */
    destroy() {
        // Cancel any ongoing animation
        if (this.drawingState.animationFrame) {
            cancelAnimationFrame(this.drawingState.animationFrame);
        }
        
        // Remove event listeners
        this.canvas.removeEventListener('mousedown', this.handleMouseDown);
        this.canvas.removeEventListener('mousemove', this.handleMouseMove);
        this.canvas.removeEventListener('mouseup', this.handleMouseUp);
        this.canvas.removeEventListener('dblclick', this.handleDoubleClick);
        this.canvas.removeEventListener('touchstart', this.handleTouchStart);
        this.canvas.removeEventListener('touchmove', this.handleTouchMove);
        this.canvas.removeEventListener('touchend', this.handleTouchEnd);
        
        document.removeEventListener('keydown', this.handleKeyDown);
        document.removeEventListener('keyup', this.handleKeyUp);
        
        // Remove overlay canvas
        if (this.drawingState.overlayCanvas && this.drawingState.overlayCanvas.parentNode) {
            this.drawingState.overlayCanvas.parentNode.removeChild(this.drawingState.overlayCanvas);
        }
        
        // Destroy tools
        Object.values(this.tools).forEach(tool => {
            if (tool.destroy) {
                tool.destroy();
            }
        });
        
        // Clear state
        this.selectionState.selectedElements.clear();
    }
}

/**
 * Rectangle Selection Tool
 */
class RectangleSelectionTool {
    constructor(selectionSystem) {
        this.selectionSystem = selectionSystem;
        this.config = selectionSystem.config;
        this.startPoint = null;
        this.currentPoint = null;
    }
    
    handleMouseDown(mousePos, event) {
        this.startPoint = { ...mousePos };
        this.currentPoint = { ...mousePos };
        this.selectionSystem.selectionState.isDrawing = true;
        this.selectionSystem.selectionState.isActive = true;
        
        this.selectionSystem.emitEvent('selectionStart', {
            tool: 'rectangle',
            startPoint: this.startPoint
        });
    }
    
    handleMouseMove(mousePos, event) {
        if (!this.selectionSystem.selectionState.isDrawing) return;
        
        this.currentPoint = { ...mousePos };
    }
    
    handleMouseUp(mousePos, event) {
        if (!this.selectionSystem.selectionState.isDrawing) return;
        
        this.currentPoint = { ...mousePos };
        this.selectionSystem.completeSelection();
    }
    
    updateSelection() {
        // Real-time selection update
        const elements = this.getElementsInRectangle();
        this.selectionSystem.selectionState.previewSelection = elements;
    }
    
    completeSelection() {
        return this.getElementsInRectangle();
    }
    
    getElementsInRectangle() {
        if (!this.startPoint || !this.currentPoint) return [];
        
        const rect = this.getRectangle();
        const elements = [];
        
        // Check nodes
        const nodes = this.selectionSystem.graphRenderer.getNodes();
        nodes.forEach(node => {
            if (this.isNodeInRectangle(node, rect)) {
                elements.push({ type: 'node', ...node });
            }
        });
        
        // Check edges
        const edges = this.selectionSystem.graphRenderer.getEdges();
        edges.forEach(edge => {
            if (this.isEdgeInRectangle(edge, rect)) {
                elements.push({ type: 'edge', ...edge });
            }
        });
        
        return elements;
    }
    
    getRectangle() {
        const x1 = Math.min(this.startPoint.x, this.currentPoint.x);
        const y1 = Math.min(this.startPoint.y, this.currentPoint.y);
        const x2 = Math.max(this.startPoint.x, this.currentPoint.x);
        const y2 = Math.max(this.startPoint.y, this.currentPoint.y);
        
        return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
    }
    
    isNodeInRectangle(node, rect) {
        const nodeRadius = node.radius || 10;
        return (node.x + nodeRadius >= rect.x &&
                node.x - nodeRadius <= rect.x + rect.width &&
                node.y + nodeRadius >= rect.y &&
                node.y - nodeRadius <= rect.y + rect.height);
    }
    
    isEdgeInRectangle(edge, rect) {
        const sourceNode = this.selectionSystem.graphRenderer.getNode(edge.source);
        const targetNode = this.selectionSystem.graphRenderer.getNode(edge.target);
        
        if (!sourceNode || !targetNode) return false;
        
        // Check if edge intersects with rectangle
        return this.lineIntersectsRectangle(
            sourceNode.x, sourceNode.y,
            targetNode.x, targetNode.y,
            rect
        );
    }
    
    lineIntersectsRectangle(x1, y1, x2, y2, rect) {
        // Check if either endpoint is inside rectangle
        if ((x1 >= rect.x && x1 <= rect.x + rect.width && y1 >= rect.y && y1 <= rect.y + rect.height) ||
            (x2 >= rect.x && x2 <= rect.x + rect.width && y2 >= rect.y && y2 <= rect.y + rect.height)) {
            return true;
        }
        
        // Check line-rectangle intersection using Liang-Barsky algorithm
        const dx = x2 - x1;
        const dy = y2 - y1;
        
        const p = [-dx, dx, -dy, dy];
        const q = [x1 - rect.x, rect.x + rect.width - x1, y1 - rect.y, rect.y + rect.height - y1];
        
        let u1 = 0, u2 = 1;
        
        for (let i = 0; i < 4; i++) {
            if (p[i] === 0) {
                if (q[i] < 0) return false;
            } else {
                const t = q[i] / p[i];
                if (p[i] < 0) {
                    u1 = Math.max(u1, t);
                } else {
                    u2 = Math.min(u2, t);
                }
            }
        }
        
        return u1 <= u2;
    }
    
    render(ctx) {
        if (!this.startPoint || !this.currentPoint) return;
        
        const rect = this.getRectangle();
        
        ctx.save();
        ctx.strokeStyle = this.config.borderColor;
        ctx.lineWidth = this.config.borderWidth;
        ctx.setLineDash([5, 5]);
        ctx.fillStyle = this.config.selectionColor + Math.floor(this.config.selectionOpacity * 255).toString(16).padStart(2, '0');
        
        ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
        ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
        
        ctx.restore();
    }
    
    cancelSelection() {
        this.startPoint = null;
        this.currentPoint = null;
    }
}

/**
 * Lasso Selection Tool
 */
class LassoSelectionTool {
    constructor(selectionSystem) {
        this.selectionSystem = selectionSystem;
        this.config = selectionSystem.config;
        this.path = [];
        this.simplifiedPath = [];
    }
    
    handleMouseDown(mousePos, event) {
        this.path = [{ ...mousePos }];
        this.simplifiedPath = [{ ...mousePos }];
        this.selectionSystem.selectionState.isDrawing = true;
        this.selectionSystem.selectionState.isActive = true;
        
        this.selectionSystem.emitEvent('selectionStart', {
            tool: 'lasso',
            startPoint: mousePos
        });
    }
    
    handleMouseMove(mousePos, event) {
        if (!this.selectionSystem.selectionState.isDrawing) return;
        
        this.path.push({ ...mousePos });
        
        // Simplify path for performance
        if (this.config.tools.lasso.simplification) {
            this.simplifiedPath = this.simplifyPath(this.path, this.config.simplificationTolerance);
        } else {
            this.simplifiedPath = [...this.path];
        }
    }
    
    handleMouseUp(mousePos, event) {
        if (!this.selectionSystem.selectionState.isDrawing) return;
        
        this.path.push({ ...mousePos });
        this.selectionSystem.completeSelection();
    }
    
    completeSelection() {
        if (this.simplifiedPath.length < 3) return [];
        
        return this.getElementsInLasso();
    }
    
    getElementsInLasso() {
        const elements = [];
        
        // Check nodes
        const nodes = this.selectionSystem.graphRenderer.getNodes();
        nodes.forEach(node => {
            if (this.isPointInPolygon(node.x, node.y, this.simplifiedPath)) {
                elements.push({ type: 'node', ...node });
            }
        });
        
        // Check edges
        const edges = this.selectionSystem.graphRenderer.getEdges();
        edges.forEach(edge => {
            if (this.isEdgeInLasso(edge)) {
                elements.push({ type: 'edge', ...edge });
            }
        });
        
        return elements;
    }
    
    isEdgeInLasso(edge) {
        const sourceNode = this.selectionSystem.graphRenderer.getNode(edge.source);
        const targetNode = this.selectionSystem.graphRenderer.getNode(edge.target);
        
        if (!sourceNode || !targetNode) return false;
        
        // Check if edge intersects with lasso polygon
        return this.lineIntersectsPolygon(
            sourceNode.x, sourceNode.y,
            targetNode.x, targetNode.y,
            this.simplifiedPath
        );
    }
    
    isPointInPolygon(x, y, polygon) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            if (((polygon[i].y > y) !== (polygon[j].y > y)) &&
                (x < (polygon[j].x - polygon[i].x) * (y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x)) {
                inside = !inside;
            }
        }
        return inside;
    }
    
    lineIntersectsPolygon(x1, y1, x2, y2, polygon) {
        // Check if either endpoint is inside polygon
        if (this.isPointInPolygon(x1, y1, polygon) || this.isPointInPolygon(x2, y2, polygon)) {
            return true;
        }
        
        // Check if line intersects any edge of polygon
        for (let i = 0; i < polygon.length; i++) {
            const j = (i + 1) % polygon.length;
            if (this.linesIntersect(x1, y1, x2, y2, polygon[i].x, polygon[i].y, polygon[j].x, polygon[j].y)) {
                return true;
            }
        }
        
        return false;
    }
    
    linesIntersect(x1, y1, x2, y2, x3, y3, x4, y4) {
        const denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
        if (denom === 0) return false;
        
        const ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;
        const ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom;
        
        return ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1;
    }
    
    simplifyPath(path, tolerance) {
        if (path.length <= 2) return path;
        
        // Douglas-Peucker algorithm
        function douglasPeucker(points, tolerance) {
            if (points.length <= 2) return points;
            
            let maxDistance = 0;
            let maxIndex = 0;
            
            for (let i = 1; i < points.length - 1; i++) {
                const distance = perpendicularDistance(points[i], points[0], points[points.length - 1]);
                if (distance > maxDistance) {
                    maxDistance = distance;
                    maxIndex = i;
                }
            }
            
            if (maxDistance > tolerance) {
                const left = douglasPeucker(points.slice(0, maxIndex + 1), tolerance);
                const right = douglasPeucker(points.slice(maxIndex), tolerance);
                return [...left.slice(0, -1), ...right];
            } else {
                return [points[0], points[points.length - 1]];
            }
        }
        
        function perpendicularDistance(point, lineStart, lineEnd) {
            const dx = lineEnd.x - lineStart.x;
            const dy = lineEnd.y - lineStart.y;
            
            if (dx === 0 && dy === 0) {
                return Math.sqrt((point.x - lineStart.x) ** 2 + (point.y - lineStart.y) ** 2);
            }
            
            const t = ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / (dx * dx + dy * dy);
            const clampedT = Math.max(0, Math.min(1, t));
            
            const closestX = lineStart.x + clampedT * dx;
            const closestY = lineStart.y + clampedT * dy;
            
            return Math.sqrt((point.x - closestX) ** 2 + (point.y - closestY) ** 2);
        }
        
        return douglasPeucker(path, tolerance);
    }
    
    render(ctx) {
        if (this.simplifiedPath.length < 2) return;
        
        ctx.save();
        ctx.strokeStyle = this.config.borderColor;
        ctx.lineWidth = this.config.borderWidth;
        ctx.setLineDash([3, 3]);
        ctx.fillStyle = this.config.selectionColor + Math.floor(this.config.selectionOpacity * 255).toString(16).padStart(2, '0');
        
        ctx.beginPath();
        ctx.moveTo(this.simplifiedPath[0].x, this.simplifiedPath[0].y);
        
        for (let i = 1; i < this.simplifiedPath.length; i++) {
            ctx.lineTo(this.simplifiedPath[i].x, this.simplifiedPath[i].y);
        }
        
        if (this.selectionSystem.selectionState.isDrawing) {
            ctx.stroke();
        } else {
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        }
        
        ctx.restore();
    }
    
    cancelSelection() {
        this.path = [];
        this.simplifiedPath = [];
    }
}

/**
 * Polygon Selection Tool
 */
class PolygonSelectionTool {
    constructor(selectionSystem) {
        this.selectionSystem = selectionSystem;
        this.config = selectionSystem.config;
        this.points = [];
        this.isComplete = false;
        this.previewPoint = null;
    }
    
    handleMouseDown(mousePos, event) {
        if (this.isComplete) {
            this.reset();
        }
        
        if (this.points.length === 0) {
            this.selectionSystem.selectionState.isDrawing = true;
            this.selectionSystem.selectionState.isActive = true;
            
            this.selectionSystem.emitEvent('selectionStart', {
                tool: 'polygon',
                startPoint: mousePos
            });
        }
        
        // Add point to polygon
        this.addPoint(mousePos);
    }
    
    handleMouseMove(mousePos, event) {
        if (!this.selectionSystem.selectionState.isDrawing) return;
        
        this.previewPoint = { ...mousePos };
    }
    
    handleDoubleClick(mousePos, event) {
        if (this.points.length >= this.config.tools.polygon.minPoints) {
            this.completePolygon();
        }
    }
    
    addPoint(mousePos) {
        if (this.points.length >= this.config.maxPolygonPoints) return;
        
        // Snap to nodes if enabled
        if (this.config.tools.polygon.snapToNodes) {
            const nearbyNode = this.findNearbyNode(mousePos, 20);
            if (nearbyNode) {
                mousePos = { x: nearbyNode.x, y: nearbyNode.y };
            }
        }
        
        this.points.push({ ...mousePos });
        
        // Check for polygon closure
        if (this.points.length > 2 && this.isNearFirstPoint(mousePos, 15)) {
            this.completePolygon();
        }
    }
    
    completePolygon() {
        if (this.points.length < this.config.tools.polygon.minPoints) return;
        
        this.isComplete = true;
        this.selectionSystem.completeSelection();
    }
    
    isNearFirstPoint(mousePos, threshold) {
        if (this.points.length === 0) return false;
        
        const dx = mousePos.x - this.points[0].x;
        const dy = mousePos.y - this.points[0].y;
        return Math.sqrt(dx * dx + dy * dy) < threshold;
    }
    
    findNearbyNode(mousePos, threshold) {
        const nodes = this.selectionSystem.graphRenderer.getNodes();
        
        for (const node of nodes) {
            const dx = mousePos.x - node.x;
            const dy = mousePos.y - node.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < threshold) {
                return node;
            }
        }
        
        return null;
    }
    
    completeSelection() {
        if (this.points.length < this.config.tools.polygon.minPoints) return [];
        
        return this.getElementsInPolygon();
    }
    
    getElementsInPolygon() {
        const elements = [];
        
        // Check nodes
        const nodes = this.selectionSystem.graphRenderer.getNodes();
        nodes.forEach(node => {
            if (this.isPointInPolygon(node.x, node.y, this.points)) {
                elements.push({ type: 'node', ...node });
            }
        });
        
        // Check edges
        const edges = this.selectionSystem.graphRenderer.getEdges();
        edges.forEach(edge => {
            if (this.isEdgeInPolygon(edge)) {
                elements.push({ type: 'edge', ...edge });
            }
        });
        
        return elements;
    }
    
    isPointInPolygon(x, y, polygon) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            if (((polygon[i].y > y) !== (polygon[j].y > y)) &&
                (x < (polygon[j].x - polygon[i].x) * (y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x)) {
                inside = !inside;
            }
        }
        return inside;
    }
    
    isEdgeInPolygon(edge) {
        const sourceNode = this.selectionSystem.graphRenderer.getNode(edge.source);
        const targetNode = this.selectionSystem.graphRenderer.getNode(edge.target);
        
        if (!sourceNode || !targetNode) return false;
        
        return this.isPointInPolygon(sourceNode.x, sourceNode.y, this.points) ||
               this.isPointInPolygon(targetNode.x, targetNode.y, this.points);
    }
    
    render(ctx) {
        if (this.points.length === 0) return;
        
        ctx.save();
        ctx.strokeStyle = this.config.borderColor;
        ctx.lineWidth = this.config.borderWidth;
        ctx.setLineDash([5, 5]);
        
        // Draw polygon
        ctx.beginPath();
        ctx.moveTo(this.points[0].x, this.points[0].y);
        
        for (let i = 1; i < this.points.length; i++) {
            ctx.lineTo(this.points[i].x, this.points[i].y);
        }
        
        // Draw preview line to current mouse position
        if (this.previewPoint && !this.isComplete) {
            ctx.lineTo(this.previewPoint.x, this.previewPoint.y);
            
            // Line back to start if near completion
            if (this.points.length > 2 && this.isNearFirstPoint(this.previewPoint, 15)) {
                ctx.lineTo(this.points[0].x, this.points[0].y);
            }
        }
        
        if (this.isComplete) {
            ctx.closePath();
            ctx.fillStyle = this.config.selectionColor + Math.floor(this.config.selectionOpacity * 255).toString(16).padStart(2, '0');
            ctx.fill();
        }
        
        ctx.stroke();
        
        // Draw points
        ctx.setLineDash([]);
        ctx.fillStyle = this.config.borderColor;
        
        this.points.forEach((point, index) => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
            ctx.fill();
            
            // Number the points
            ctx.fillStyle = '#ffffff';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText((index + 1).toString(), point.x, point.y);
            ctx.fillStyle = this.config.borderColor;
        });
        
        ctx.restore();
    }
    
    reset() {
        this.points = [];
        this.isComplete = false;
        this.previewPoint = null;
    }
    
    cancelSelection() {
        this.reset();
    }
}

/**
 * Circle Selection Tool
 */
class CircleSelectionTool {
    constructor(selectionSystem) {
        this.selectionSystem = selectionSystem;
        this.config = selectionSystem.config;
        this.center = null;
        this.radius = 0;
    }
    
    handleMouseDown(mousePos, event) {
        this.center = { ...mousePos };
        this.radius = 0;
        this.selectionSystem.selectionState.isDrawing = true;
        this.selectionSystem.selectionState.isActive = true;
        
        this.selectionSystem.emitEvent('selectionStart', {
            tool: 'circle',
            center: this.center
        });
    }
    
    handleMouseMove(mousePos, event) {
        if (!this.selectionSystem.selectionState.isDrawing) return;
        
        const dx = mousePos.x - this.center.x;
        const dy = mousePos.y - this.center.y;
        this.radius = Math.sqrt(dx * dx + dy * dy);
    }
    
    handleMouseUp(mousePos, event) {
        if (!this.selectionSystem.selectionState.isDrawing) return;
        
        const dx = mousePos.x - this.center.x;
        const dy = mousePos.y - this.center.y;
        this.radius = Math.sqrt(dx * dx + dy * dy);
        
        this.selectionSystem.completeSelection();
    }
    
    completeSelection() {
        if (this.radius < this.config.minSelectionSize) return [];
        
        return this.getElementsInCircle();
    }
    
    getElementsInCircle() {
        const elements = [];
        
        // Check nodes
        const nodes = this.selectionSystem.graphRenderer.getNodes();
        nodes.forEach(node => {
            const dx = node.x - this.center.x;
            const dy = node.y - this.center.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance <= this.radius) {
                elements.push({ type: 'node', ...node });
            }
        });
        
        // Check edges
        const edges = this.selectionSystem.graphRenderer.getEdges();
        edges.forEach(edge => {
            if (this.isEdgeInCircle(edge)) {
                elements.push({ type: 'edge', ...edge });
            }
        });
        
        return elements;
    }
    
    isEdgeInCircle(edge) {
        const sourceNode = this.selectionSystem.graphRenderer.getNode(edge.source);
        const targetNode = this.selectionSystem.graphRenderer.getNode(edge.target);
        
        if (!sourceNode || !targetNode) return false;
        
        // Check if line segment intersects with circle
        return this.lineIntersectsCircle(
            sourceNode.x, sourceNode.y,
            targetNode.x, targetNode.y,
            this.center.x, this.center.y,
            this.radius
        );
    }
    
    lineIntersectsCircle(x1, y1, x2, y2, cx, cy, radius) {
        // Calculate closest point on line segment to circle center
        const dx = x2 - x1;
        const dy = y2 - y1;
        const length = Math.sqrt(dx * dx + dy * dy);
        
        if (length === 0) {
            // Line is a point
            const dist = Math.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2);
            return dist <= radius;
        }
        
        const t = Math.max(0, Math.min(1, ((cx - x1) * dx + (cy - y1) * dy) / (length * length)));
        const closestX = x1 + t * dx;
        const closestY = y1 + t * dy;
        
        const distance = Math.sqrt((closestX - cx) ** 2 + (closestY - cy) ** 2);
        return distance <= radius;
    }
    
    render(ctx) {
        if (!this.center || this.radius <= 0) return;
        
        ctx.save();
        ctx.strokeStyle = this.config.borderColor;
        ctx.lineWidth = this.config.borderWidth;
        ctx.setLineDash([5, 5]);
        ctx.fillStyle = this.config.selectionColor + Math.floor(this.config.selectionOpacity * 255).toString(16).padStart(2, '0');
        
        ctx.beginPath();
        ctx.arc(this.center.x, this.center.y, this.radius, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        
        // Draw radius line if enabled
        if (this.config.tools.circle.showRadius) {
            ctx.setLineDash([]);
            ctx.beginPath();
            ctx.moveTo(this.center.x, this.center.y);
            ctx.lineTo(this.center.x + this.radius, this.center.y);
            ctx.stroke();
            
            // Radius text
            ctx.fillStyle = this.config.borderColor;
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(
                `r: ${Math.round(this.radius)}px`,
                this.center.x + this.radius / 2,
                this.center.y - 5
            );
        }
        
        ctx.restore();
    }
    
    cancelSelection() {
        this.center = null;
        this.radius = 0;
    }
}

/**
 * Magic Wand Selection Tool
 */
class MagicWandSelectionTool {
    constructor(selectionSystem) {
        this.selectionSystem = selectionSystem;
        this.config = selectionSystem.config;
    }
    
    handleMouseDown(mousePos, event) {
        // Magic wand selects on click
        const clickedElement = this.getElementAtPosition(mousePos);
        if (clickedElement) {
            this.performMagicSelection(clickedElement);
        }
    }
    
    handleMouseMove(mousePos, event) {
        // No continuous interaction for magic wand
    }
    
    handleMouseUp(mousePos, event) {
        // Selection already completed on mouse down
    }
    
    performMagicSelection(seedElement) {
        this.selectionSystem.selectionState.isActive = true;
        
        let selectedElements = [];
        
        if (seedElement.type === 'node') {
            selectedElements = this.selectSimilarNodes(seedElement);
        } else if (seedElement.type === 'edge') {
            selectedElements = this.selectSimilarEdges(seedElement);
        }
        
        this.selectionSystem.applySelection(selectedElements);
        
        this.selectionSystem.emitEvent('magicWandSelection', {
            seedElement: seedElement,
            selectedCount: selectedElements.length
        });
    }
    
    selectSimilarNodes(seedNode) {
        const tolerance = this.config.tools.magicWand.tolerance;
        const nodes = this.selectionSystem.graphRenderer.getNodes();
        const selected = [];
        
        nodes.forEach(node => {
            if (this.nodesAreSimilar(seedNode, node, tolerance)) {
                selected.push({ type: 'node', ...node });
            }
        });
        
        return selected;
    }
    
    selectSimilarEdges(seedEdge) {
        const tolerance = this.config.tools.magicWand.tolerance;
        const edges = this.selectionSystem.graphRenderer.getEdges();
        const selected = [];
        
        edges.forEach(edge => {
            if (this.edgesAreSimilar(seedEdge, edge, tolerance)) {
                selected.push({ type: 'edge', ...edge });
            }
        });
        
        return selected;
    }
    
    nodesAreSimilar(node1, node2, tolerance) {
        // Compare based on visual properties
        const colorSimilar = this.colorSimilarity(node1.color, node2.color) > (100 - tolerance) / 100;
        const sizeSimilar = Math.abs((node1.radius || 10) - (node2.radius || 10)) <= tolerance / 10;
        const typeSimilar = (node1.type || 'default') === (node2.type || 'default');
        
        return colorSimilar && sizeSimilar && typeSimilar;
    }
    
    edgesAreSimilar(edge1, edge2, tolerance) {
        // Compare based on visual properties
        const colorSimilar = this.colorSimilarity(edge1.color, edge2.color) > (100 - tolerance) / 100;
        const widthSimilar = Math.abs((edge1.width || 1) - (edge2.width || 1)) <= tolerance / 20;
        const typeSimilar = (edge1.type || 'default') === (edge2.type || 'default');
        
        return colorSimilar && widthSimilar && typeSimilar;
    }
    
    colorSimilarity(color1, color2) {
        // Simple color similarity (can be enhanced with LAB color space)
        if (!color1 || !color2) return color1 === color2 ? 1 : 0;
        
        const rgb1 = this.hexToRgb(color1);
        const rgb2 = this.hexToRgb(color2);
        
        if (!rgb1 || !rgb2) return color1 === color2 ? 1 : 0;
        
        const deltaR = rgb1.r - rgb2.r;
        const deltaG = rgb1.g - rgb2.g;
        const deltaB = rgb1.b - rgb2.b;
        
        const distance = Math.sqrt(deltaR * deltaR + deltaG * deltaG + deltaB * deltaB);
        const maxDistance = Math.sqrt(255 * 255 * 3);
        
        return 1 - (distance / maxDistance);
    }
    
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }
    
    getElementAtPosition(mousePos) {
        // Check for nodes first
        const node = this.selectionSystem.graphRenderer.getNodeAtPosition(mousePos.x, mousePos.y);
        if (node) {
            return { type: 'node', ...node };
        }
        
        // Check for edges
        const edge = this.selectionSystem.graphRenderer.getEdgeAtPosition(mousePos.x, mousePos.y);
        if (edge) {
            return { type: 'edge', ...edge };
        }
        
        return null;
    }
    
    render(ctx) {
        // Magic wand doesn't have continuous visual feedback
    }
    
    cancelSelection() {
        // Nothing to cancel for magic wand
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedSelectionTools;
}