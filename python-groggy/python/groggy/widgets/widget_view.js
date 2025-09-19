/**
 * GroggyGraphView - JavaScript Implementation for Widget
 * 
 * This is a simplified JavaScript view that can be embedded in HTML
 * for testing the widget functionality. In a full Jupyter widget,
 * this would be built as a proper npm package with webpack.
 */

class GroggyGraphView {
    constructor(container, config = {}) {
        this.container = container;
        this.config = {
            width: 800,
            height: 600,
            layout: 'force-directed',
            theme: 'light',
            enableDrag: true,
            enablePan: true,
            enableZoom: true,
            animationDuration: 300,
            ...config
        };
        
        // State
        this.nodes = [];
        this.edges = [];
        this.positions = [];
        this.selectedNodes = [];
        this.hoveredNode = null;
        this.camera = { x: 0, y: 0, zoom: 1.0 };
        
        // Interaction state
        this.isDragging = false;
        this.dragNode = null;
        this.isPanning = false;
        this.lastMouse = { x: 0, y: 0 };
        
        // Callbacks for Python communication
        this.callbacks = {
            nodeClick: [],
            nodeHover: [],
            nodeDoubleClick: [],
            layoutChange: [],
            cameraChange: [],
            selectionChange: []
        };
        
        this.init();
    }
    
    init() {
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.config.width;
        this.canvas.height = this.config.height;
        this.canvas.style.border = '1px solid #ddd';
        this.canvas.style.cursor = 'default';
        this.canvas.style.display = 'block';
        
        this.ctx = this.canvas.getContext('2d');
        this.container.appendChild(this.canvas);
        
        // Create controls
        this.createControls();
        
        // Set up event listeners
        this.setupEventListeners();
        
        console.log('GroggyGraphView initialized');
    }
    
    createControls() {
        const controlsDiv = document.createElement('div');
        controlsDiv.style.cssText = `
            margin: 10px 0;
            text-align: center;
        `;
        
        // Layout button
        const layoutBtn = document.createElement('button');
        layoutBtn.textContent = 'Toggle Layout';
        layoutBtn.style.cssText = `
            background: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 4px;
            border-radius: 4px;
            cursor: pointer;
        `;
        layoutBtn.onclick = () => this.toggleLayout();
        
        // Reset button
        const resetBtn = document.createElement('button');
        resetBtn.textContent = 'Reset View';
        resetBtn.style.cssText = layoutBtn.style.cssText.replace('#007bff', '#28a745');
        resetBtn.onclick = () => this.resetCamera();
        
        // Theme button
        const themeBtn = document.createElement('button');
        themeBtn.textContent = 'Toggle Theme';
        themeBtn.style.cssText = layoutBtn.style.cssText.replace('#007bff', '#6c757d');
        themeBtn.onclick = () => this.toggleTheme();
        
        controlsDiv.appendChild(layoutBtn);
        controlsDiv.appendChild(resetBtn);
        controlsDiv.appendChild(themeBtn);
        
        this.container.insertBefore(controlsDiv, this.canvas);
        
        // Info panel
        this.infoPanel = document.createElement('div');
        this.infoPanel.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            display: none;
        `;
        this.container.style.position = 'relative';
        this.container.appendChild(this.infoPanel);
    }
    
    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('click', this.onClick.bind(this));
        this.canvas.addEventListener('dblclick', this.onDoubleClick.bind(this));
        this.canvas.addEventListener('wheel', this.onWheel.bind(this));
        this.canvas.addEventListener('mouseleave', this.onMouseLeave.bind(this));
    }
    
    // Public API
    setData(nodes, edges) {
        this.nodes = nodes || [];
        this.edges = edges || [];
        this.calculateLayout();
        this.render();
        console.log(`Data updated: ${this.nodes.length} nodes, ${this.edges.length} edges`);
    }
    
    setLayout(algorithm, animate = true) {
        const oldLayout = this.config.layout;
        this.config.layout = algorithm;
        
        if (animate) {
            this.animateLayoutChange();
        } else {
            this.calculateLayout();
            this.render();
        }
        
        this.trigger('layoutChange', { from: oldLayout, to: algorithm });
        console.log(`Layout changed to: ${algorithm}`);
    }
    
    setTheme(theme) {
        this.config.theme = theme;
        this.render();
        console.log(`Theme changed to: ${theme}`);
    }
    
    selectNodes(nodeIds, clearExisting = true) {
        if (clearExisting) {
            this.selectedNodes = [...nodeIds];
        } else {
            this.selectedNodes = [...new Set([...this.selectedNodes, ...nodeIds])];
        }
        this.render();
        this.trigger('selectionChange', this.selectedNodes);
    }
    
    focusOnNode(nodeId, zoomLevel = 2.0) {
        const node = this.nodes.find(n => n.id === nodeId);
        const pos = this.positions.find(p => p.id === nodeId);
        
        if (node && pos) {
            // Center camera on node
            this.camera.x = this.config.width / 2 - pos.x * this.camera.zoom;
            this.camera.y = this.config.height / 2 - pos.y * this.camera.zoom;
            this.camera.zoom = zoomLevel;
            
            this.render();
            this.trigger('cameraChange', this.camera);
        }
    }
    
    resetCamera() {
        this.camera = { x: 0, y: 0, zoom: 1.0 };
        this.calculateLayout();
        this.render();
        this.trigger('cameraChange', this.camera);
    }
    
    // Event handling
    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - this.camera.x) / this.camera.zoom;
        const y = (e.clientY - rect.top - this.camera.y) / this.camera.zoom;
        
        const clickedNode = this.findNodeAt(x, y);
        
        if (clickedNode && this.config.enableDrag) {
            this.isDragging = true;
            this.dragNode = clickedNode;
            this.canvas.style.cursor = 'grabbing';
        } else if (this.config.enablePan) {
            this.isPanning = true;
            this.lastMouse = { x: e.clientX, y: e.clientY };
            this.canvas.style.cursor = 'grabbing';
        }
    }
    
    onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - this.camera.x) / this.camera.zoom;
        const y = (e.clientY - rect.top - this.camera.y) / this.camera.zoom;
        
        if (this.isDragging && this.dragNode) {
            // Update node position
            const pos = this.positions.find(p => p.id === this.dragNode.id);
            if (pos) {
                pos.x = x;
                pos.y = y;
                this.render();
            }
        } else if (this.isPanning) {
            // Update camera position
            this.camera.x += e.clientX - this.lastMouse.x;
            this.camera.y += e.clientY - this.lastMouse.y;
            this.lastMouse = { x: e.clientX, y: e.clientY };
            this.render();
        } else {
            // Hover detection
            const hoveredNode = this.findNodeAt(x, y);
            if (hoveredNode !== this.hoveredNode) {
                this.hoveredNode = hoveredNode;
                this.canvas.style.cursor = hoveredNode ? 'pointer' : 'default';
                
                if (hoveredNode) {
                    this.infoPanel.textContent = `${hoveredNode.label || hoveredNode.id} (${hoveredNode.id})`;
                    this.infoPanel.style.display = 'block';
                } else {
                    this.infoPanel.style.display = 'none';
                }
                
                this.render();
                this.trigger('nodeHover', hoveredNode ? hoveredNode.id : null);
            }
        }
    }
    
    onMouseUp() {
        this.isDragging = false;
        this.isPanning = false;
        this.dragNode = null;
        this.canvas.style.cursor = 'default';
    }
    
    onClick(e) {
        if (!this.isDragging) {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left - this.camera.x) / this.camera.zoom;
            const y = (e.clientY - rect.top - this.camera.y) / this.camera.zoom;
            
            const clickedNode = this.findNodeAt(x, y);
            if (clickedNode) {
                // Handle selection
                if (e.ctrlKey || e.metaKey) {
                    // Add to selection
                    if (this.selectedNodes.includes(clickedNode.id)) {
                        this.selectedNodes = this.selectedNodes.filter(id => id !== clickedNode.id);
                    } else {
                        this.selectedNodes.push(clickedNode.id);
                    }
                } else {
                    // Replace selection
                    this.selectedNodes = [clickedNode.id];
                }
                
                this.render();
                this.trigger('nodeClick', { nodeId: clickedNode.id, nodeData: clickedNode });
                this.trigger('selectionChange', this.selectedNodes);
            } else {
                // Clear selection
                if (!e.ctrlKey && !e.metaKey) {
                    this.selectedNodes = [];
                    this.render();
                    this.trigger('selectionChange', this.selectedNodes);
                }
            }
        }
    }
    
    onDoubleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - this.camera.x) / this.camera.zoom;
        const y = (e.clientY - rect.top - this.camera.y) / this.camera.zoom;
        
        const clickedNode = this.findNodeAt(x, y);
        if (clickedNode) {
            this.trigger('nodeDoubleClick', { nodeId: clickedNode.id, nodeData: clickedNode });
            this.focusOnNode(clickedNode.id, 2.0);
        }
    }
    
    onWheel(e) {
        if (this.config.enableZoom) {
            e.preventDefault();
            const rect = this.canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const newZoom = Math.max(0.1, Math.min(5.0, this.camera.zoom * zoomFactor));
            
            // Zoom toward mouse position
            this.camera.x = mouseX - (mouseX - this.camera.x) * (newZoom / this.camera.zoom);
            this.camera.y = mouseY - (mouseY - this.camera.y) * (newZoom / this.camera.zoom);
            this.camera.zoom = newZoom;
            
            this.render();
            this.trigger('cameraChange', this.camera);
        }
    }
    
    onMouseLeave() {
        this.hoveredNode = null;
        this.infoPanel.style.display = 'none';
        this.render();
    }
    
    // Utility methods
    findNodeAt(x, y) {
        return this.nodes.find(node => {
            const pos = this.positions.find(p => p.id === node.id);
            if (pos) {
                const radius = node.size || 8;
                const dx = x - pos.x;
                const dy = y - pos.y;
                return (dx * dx + dy * dy) <= (radius * radius);
            }
            return false;
        });
    }
    
    toggleLayout() {
        const layouts = ['circular', 'grid', 'force-directed'];
        const currentIndex = layouts.indexOf(this.config.layout);
        const nextLayout = layouts[(currentIndex + 1) % layouts.length];
        this.setLayout(nextLayout, true);
    }
    
    toggleTheme() {
        const themes = ['light', 'dark'];
        const currentIndex = themes.indexOf(this.config.theme);
        const nextTheme = themes[(currentIndex + 1) % themes.length];
        this.setTheme(nextTheme);
    }
    
    // Layout algorithms
    calculateLayout() {
        const { width, height, layout } = this.config;
        const padding = 50;
        this.positions = [];
        
        if (layout === 'circular' && this.nodes.length > 0) {
            const centerX = width / 2;
            const centerY = height / 2;
            const radius = Math.min(width, height) / 2 - padding;
            
            this.nodes.forEach((node, i) => {
                const angle = (i * 2 * Math.PI) / this.nodes.length;
                this.positions.push({
                    id: node.id,
                    x: centerX + radius * Math.cos(angle),
                    y: centerY + radius * Math.sin(angle)
                });
            });
        } else if (layout === 'force-directed') {
            // Initialize random positions
            this.positions = this.nodes.map(node => ({
                id: node.id,
                x: padding + Math.random() * (width - 2 * padding),
                y: padding + Math.random() * (height - 2 * padding)
            }));
            
            // Force simulation
            for (let iter = 0; iter < 50; iter++) {
                // Repulsion
                for (let i = 0; i < this.positions.length; i++) {
                    for (let j = i + 1; j < this.positions.length; j++) {
                        const dx = this.positions[j].x - this.positions[i].x;
                        const dy = this.positions[j].y - this.positions[i].y;
                        const distance = Math.sqrt(dx * dx + dy * dy) || 1;
                        const force = 500 / (distance * distance);
                        
                        this.positions[i].x -= force * dx / distance;
                        this.positions[i].y -= force * dy / distance;
                        this.positions[j].x += force * dx / distance;
                        this.positions[j].y += force * dy / distance;
                    }
                }
                
                // Attraction for edges
                this.edges.forEach(edge => {
                    const src = this.positions.find(p => p.id === edge.source);
                    const dst = this.positions.find(p => p.id === edge.target);
                    if (src && dst) {
                        const dx = dst.x - src.x;
                        const dy = dst.y - src.y;
                        const distance = Math.sqrt(dx * dx + dy * dy) || 1;
                        const force = distance * 0.01;
                        
                        src.x += force * dx / distance;
                        src.y += force * dy / distance;
                        dst.x -= force * dx / distance;
                        dst.y -= force * dy / distance;
                    }
                });
                
                // Keep in bounds
                this.positions.forEach(pos => {
                    pos.x = Math.max(padding, Math.min(width - padding, pos.x));
                    pos.y = Math.max(padding, Math.min(height - padding, pos.y));
                });
            }
        } else {
            // Grid layout
            const cols = Math.ceil(Math.sqrt(this.nodes.length));
            const cellW = (width - 2 * padding) / cols;
            const cellH = (height - 2 * padding) / Math.ceil(this.nodes.length / cols);
            
            this.nodes.forEach((node, i) => {
                this.positions.push({
                    id: node.id,
                    x: padding + (i % cols) * cellW + cellW / 2,
                    y: padding + Math.floor(i / cols) * cellH + cellH / 2
                });
            });
        }
    }
    
    animateLayoutChange() {
        const oldPositions = [...this.positions];
        this.calculateLayout();
        const newPositions = [...this.positions];
        
        // Simple animation (could be enhanced with proper easing)
        const startTime = Date.now();
        const duration = this.config.animationDuration;
        
        const animate = () => {
            const progress = Math.min(1, (Date.now() - startTime) / duration);
            
            this.positions = oldPositions.map((oldPos, i) => {
                const newPos = newPositions[i];
                return {
                    id: oldPos.id,
                    x: oldPos.x + (newPos.x - oldPos.x) * progress,
                    y: oldPos.y + (newPos.y - oldPos.y) * progress
                };
            });
            
            this.render();
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    // Rendering
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.save();
        this.ctx.translate(this.camera.x, this.camera.y);
        this.ctx.scale(this.camera.zoom, this.camera.zoom);
        
        this.drawEdges();
        this.drawNodes();
        
        this.ctx.restore();
        
        // Draw UI elements (not affected by camera)
        this.drawStats();
    }
    
    drawEdges() {
        const { theme } = this.config;
        
        this.edges.forEach(edge => {
            const src = this.positions.find(p => p.id === edge.source);
            const dst = this.positions.find(p => p.id === edge.target);
            if (src && dst) {
                this.ctx.strokeStyle = theme === 'dark' ? '#666' : '#999';
                this.ctx.lineWidth = 1;
                this.ctx.beginPath();
                this.ctx.moveTo(src.x, src.y);
                this.ctx.lineTo(dst.x, dst.y);
                this.ctx.stroke();
            }
        });
    }
    
    drawNodes() {
        const { theme } = this.config;
        
        this.nodes.forEach(node => {
            const pos = this.positions.find(p => p.id === node.id);
            if (pos) {
                const isSelected = this.selectedNodes.includes(node.id);
                const isHovered = this.hoveredNode && this.hoveredNode.id === node.id;
                const radius = (node.size || 8) * (isHovered ? 1.3 : 1);
                
                // Shadow for hover
                if (isHovered) {
                    this.ctx.shadowColor = 'rgba(0,0,0,0.3)';
                    this.ctx.shadowBlur = 10;
                    this.ctx.shadowOffsetX = 2;
                    this.ctx.shadowOffsetY = 2;
                }
                
                // Node circle
                this.ctx.fillStyle = node.color || (theme === 'dark' ? '#4CAF50' : '#007bff');
                this.ctx.beginPath();
                this.ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);
                this.ctx.fill();
                
                this.ctx.shadowColor = 'transparent';
                this.ctx.shadowBlur = 0;
                
                // Border
                if (isSelected || isHovered) {
                    this.ctx.strokeStyle = isSelected ? '#ff0000' : '#ffa500';
                    this.ctx.lineWidth = 3;
                    this.ctx.stroke();
                }
                
                // Label
                this.ctx.fillStyle = theme === 'dark' ? '#fff' : '#000';
                this.ctx.font = '12px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText(node.label || node.id, pos.x, pos.y - radius - 5);
            }
        });
    }
    
    drawStats() {
        const stats = `${this.nodes.length} nodes | ${this.edges.length} edges | ${this.config.layout} | zoom: ${this.camera.zoom.toFixed(1)}x`;
        this.ctx.fillStyle = this.config.theme === 'dark' ? '#fff' : '#666';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(stats, 10, this.canvas.height - 10);
    }
    
    // Event system
    on(event, callback) {
        if (!this.callbacks[event]) {
            this.callbacks[event] = [];
        }
        this.callbacks[event].push(callback);
    }
    
    trigger(event, data) {
        if (this.callbacks[event]) {
            this.callbacks[event].forEach(callback => {
                try {
                    callback(data);
                } catch (e) {
                    console.error(`Error in ${event} callback:`, e);
                }
            });
        }
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GroggyGraphView;
}