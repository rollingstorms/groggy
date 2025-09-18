/**
 * Advanced Context Menu System
 * Provides intelligent right-click context menus with dynamic options
 * based on the clicked element (node, edge, canvas) and current state.
 */

class ContextMenuSystem {
    constructor(graphRenderer, config = {}) {
        this.graphRenderer = graphRenderer;
        this.canvas = graphRenderer.canvas;
        
        // Configuration with intelligent defaults
        this.config = {
            enableContextMenus: true,
            showIcons: true,
            showKeyboardShortcuts: true,
            enableSubmenuExpansion: true,
            enableDynamicOptions: true,
            
            // Menu behavior
            hideOnScroll: true,
            hideOnResize: true,
            hideOnCanvasClick: true,
            preventDefaultContextMenu: true,
            
            // Animation settings
            enableAnimations: true,
            animationDuration: 200,
            animationEasing: 'cubic-bezier(0.25, 0.8, 0.25, 1)',
            
            // Menu appearance
            maxWidth: 250,
            maxHeight: 400,
            itemHeight: 32,
            iconSize: 16,
            borderRadius: 8,
            
            // Custom actions
            customNodeActions: [],
            customEdgeActions: [],
            customCanvasActions: [],
            
            ...config
        };
        
        // Menu state
        this.menuState = {
            isVisible: false,
            currentMenu: null,
            clickedElement: null,
            clickedPosition: null,
            activeSubmenu: null,
            hoveredItem: null
        };
        
        // Menu container
        this.menuContainer = null;
        this.submenuContainer = null;
        
        // Event bindings
        this.boundHandlers = {};
        
        this.initializeContextMenuSystem();
    }
    
    /**
     * Initialize the context menu system
     */
    initializeContextMenuSystem() {
        this.createMenuContainer();
        this.bindEventListeners();
        this.setupMenuTemplates();
    }
    
    /**
     * Create the main menu container
     */
    createMenuContainer() {
        this.menuContainer = document.createElement('div');
        this.menuContainer.className = 'groggy-context-menu';
        this.menuContainer.style.cssText = `
            position: fixed;
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: ${this.config.borderRadius}px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            padding: 4px 0;
            min-width: 150px;
            max-width: ${this.config.maxWidth}px;
            max-height: ${this.config.maxHeight}px;
            overflow-y: auto;
            z-index: 10000;
            display: none;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            line-height: 1.4;
            user-select: none;
        `;
        
        // Add dark theme styles
        this.addDarkThemeStyles();
        
        document.body.appendChild(this.menuContainer);
    }
    
    /**
     * Add dark theme CSS support
     */
    addDarkThemeStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .groggy-context-menu {
                color: #333;
            }
            
            .groggy-context-menu-item {
                display: flex;
                align-items: center;
                padding: 6px 12px;
                cursor: pointer;
                transition: background-color 0.15s ease;
                position: relative;
            }
            
            .groggy-context-menu-item:hover {
                background-color: #f5f5f5;
            }
            
            .groggy-context-menu-item.disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .groggy-context-menu-item.disabled:hover {
                background-color: transparent;
            }
            
            .groggy-context-menu-icon {
                width: ${this.config.iconSize}px;
                height: ${this.config.iconSize}px;
                margin-right: 8px;
                flex-shrink: 0;
            }
            
            .groggy-context-menu-text {
                flex: 1;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            
            .groggy-context-menu-shortcut {
                color: #888;
                font-size: 12px;
                margin-left: 8px;
            }
            
            .groggy-context-menu-arrow {
                margin-left: 8px;
                font-size: 10px;
                color: #888;
            }
            
            .groggy-context-menu-separator {
                height: 1px;
                background-color: #e0e0e0;
                margin: 4px 0;
            }
            
            .groggy-context-menu-header {
                padding: 8px 12px 4px;
                font-weight: 600;
                color: #666;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            /* Dark theme */
            @media (prefers-color-scheme: dark) {
                .groggy-context-menu {
                    background: #2d2d2d;
                    border-color: #555;
                    color: #e0e0e0;
                }
                
                .groggy-context-menu-item:hover {
                    background-color: #404040;
                }
                
                .groggy-context-menu-separator {
                    background-color: #555;
                }
                
                .groggy-context-menu-shortcut,
                .groggy-context-menu-arrow {
                    color: #aaa;
                }
                
                .groggy-context-menu-header {
                    color: #ccc;
                }
            }
            
            /* Animation classes */
            .groggy-context-menu.fade-in {
                animation: contextMenuFadeIn ${this.config.animationDuration}ms ${this.config.animationEasing};
            }
            
            .groggy-context-menu.fade-out {
                animation: contextMenuFadeOut ${this.config.animationDuration}ms ${this.config.animationEasing};
            }
            
            @keyframes contextMenuFadeIn {
                from {
                    opacity: 0;
                    transform: scale(0.95) translateY(-5px);
                }
                to {
                    opacity: 1;
                    transform: scale(1) translateY(0);
                }
            }
            
            @keyframes contextMenuFadeOut {
                from {
                    opacity: 1;
                    transform: scale(1) translateY(0);
                }
                to {
                    opacity: 0;
                    transform: scale(0.95) translateY(-5px);
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    /**
     * Bind event listeners
     */
    bindEventListeners() {
        // Canvas right-click
        this.boundHandlers.contextmenu = this.handleContextMenu.bind(this);
        this.canvas.addEventListener('contextmenu', this.boundHandlers.contextmenu);
        
        // Global click to hide menu
        this.boundHandlers.click = this.handleGlobalClick.bind(this);
        document.addEventListener('click', this.boundHandlers.click);
        
        // Keyboard events
        this.boundHandlers.keydown = this.handleKeyDown.bind(this);
        document.addEventListener('keydown', this.boundHandlers.keydown);
        
        // Window events
        if (this.config.hideOnScroll) {
            this.boundHandlers.scroll = this.hideMenu.bind(this);
            window.addEventListener('scroll', this.boundHandlers.scroll, true);
        }
        
        if (this.config.hideOnResize) {
            this.boundHandlers.resize = this.hideMenu.bind(this);
            window.addEventListener('resize', this.boundHandlers.resize);
        }
    }
    
    /**
     * Setup predefined menu templates
     */
    setupMenuTemplates() {
        this.menuTemplates = {
            node: this.createNodeMenuTemplate(),
            edge: this.createEdgeMenuTemplate(),
            canvas: this.createCanvasMenuTemplate(),
            multipleNodes: this.createMultipleNodesMenuTemplate(),
            multipleEdges: this.createMultipleEdgesMenuTemplate(),
            mixed: this.createMixedSelectionMenuTemplate()
        };
    }
    
    /**
     * Create node context menu template
     */
    createNodeMenuTemplate() {
        return [
            {
                type: 'header',
                text: 'Node Actions'
            },
            {
                id: 'select-node',
                text: 'Select Node',
                icon: '✓',
                shortcut: 'Click',
                action: (node) => this.selectNode(node),
                condition: (node) => !this.isNodeSelected(node)
            },
            {
                id: 'deselect-node',
                text: 'Deselect Node',
                icon: '✗',
                action: (node) => this.deselectNode(node),
                condition: (node) => this.isNodeSelected(node)
            },
            {
                id: 'center-on-node',
                text: 'Center View',
                icon: '🎯',
                shortcut: 'Double-click',
                action: (node) => this.centerOnNode(node)
            },
            {
                id: 'highlight-connections',
                text: 'Highlight Connections',
                icon: '🔗',
                action: (node) => this.highlightNodeConnections(node)
            },
            {
                type: 'separator'
            },
            {
                id: 'edit-node',
                text: 'Edit Properties',
                icon: '✏️',
                shortcut: 'F2',
                action: (node) => this.editNode(node)
            },
            {
                id: 'duplicate-node',
                text: 'Duplicate Node',
                icon: '📋',
                shortcut: 'Ctrl+D',
                action: (node) => this.duplicateNode(node)
            },
            {
                type: 'separator'
            },
            {
                id: 'expand-neighbors',
                text: 'Expand Neighbors',
                icon: '🔍',
                submenu: [
                    {
                        id: 'expand-1-hop',
                        text: '1-hop Neighbors',
                        action: (node) => this.expandNeighbors(node, 1)
                    },
                    {
                        id: 'expand-2-hop',
                        text: '2-hop Neighbors',
                        action: (node) => this.expandNeighbors(node, 2)
                    },
                    {
                        id: 'expand-all',
                        text: 'All Connected',
                        action: (node) => this.expandNeighbors(node, -1)
                    }
                ]
            },
            {
                id: 'hide-node',
                text: 'Hide Node',
                icon: '👁️',
                action: (node) => this.hideNode(node)
            },
            {
                id: 'delete-node',
                text: 'Delete Node',
                icon: '🗑️',
                shortcut: 'Delete',
                action: (node) => this.deleteNode(node),
                className: 'destructive'
            }
        ];
    }
    
    /**
     * Create edge context menu template
     */
    createEdgeMenuTemplate() {
        return [
            {
                type: 'header',
                text: 'Edge Actions'
            },
            {
                id: 'select-edge',
                text: 'Select Edge',
                icon: '✓',
                action: (edge) => this.selectEdge(edge),
                condition: (edge) => !this.isEdgeSelected(edge)
            },
            {
                id: 'deselect-edge',
                text: 'Deselect Edge',
                icon: '✗',
                action: (edge) => this.deselectEdge(edge),
                condition: (edge) => this.isEdgeSelected(edge)
            },
            {
                id: 'highlight-path',
                text: 'Highlight Path',
                icon: '🛤️',
                action: (edge) => this.highlightEdgePath(edge)
            },
            {
                type: 'separator'
            },
            {
                id: 'edit-edge',
                text: 'Edit Properties',
                icon: '✏️',
                action: (edge) => this.editEdge(edge)
            },
            {
                id: 'reverse-edge',
                text: 'Reverse Direction',
                icon: '🔄',
                action: (edge) => this.reverseEdge(edge)
            },
            {
                type: 'separator'
            },
            {
                id: 'select-endpoints',
                text: 'Select Endpoints',
                icon: '⚫',
                action: (edge) => this.selectEdgeEndpoints(edge)
            },
            {
                id: 'hide-edge',
                text: 'Hide Edge',
                icon: '👁️',
                action: (edge) => this.hideEdge(edge)
            },
            {
                id: 'delete-edge',
                text: 'Delete Edge',
                icon: '🗑️',
                shortcut: 'Delete',
                action: (edge) => this.deleteEdge(edge),
                className: 'destructive'
            }
        ];
    }
    
    /**
     * Create canvas context menu template
     */
    createCanvasMenuTemplate() {
        return [
            {
                type: 'header',
                text: 'Canvas Actions'
            },
            {
                id: 'add-node',
                text: 'Add Node',
                icon: '➕',
                shortcut: 'A',
                action: (position) => this.addNode(position)
            },
            {
                id: 'paste-nodes',
                text: 'Paste',
                icon: '📋',
                shortcut: 'Ctrl+V',
                action: (position) => this.pasteNodes(position),
                condition: () => this.hasClipboardData()
            },
            {
                type: 'separator'
            },
            {
                id: 'select-all',
                text: 'Select All',
                icon: '🔲',
                shortcut: 'Ctrl+A',
                action: () => this.selectAll()
            },
            {
                id: 'clear-selection',
                text: 'Clear Selection',
                icon: '🔳',
                shortcut: 'Escape',
                action: () => this.clearSelection(),
                condition: () => this.hasSelection()
            },
            {
                type: 'separator'
            },
            {
                id: 'fit-to-view',
                text: 'Fit to View',
                icon: '📐',
                shortcut: 'F',
                action: () => this.fitToView()
            },
            {
                id: 'center-view',
                text: 'Center View',
                icon: '🎯',
                shortcut: 'Home',
                action: () => this.centerView()
            },
            {
                type: 'separator'
            },
            {
                id: 'layout-submenu',
                text: 'Apply Layout',
                icon: '🔧',
                submenu: [
                    {
                        id: 'force-directed',
                        text: 'Force-Directed',
                        action: () => this.applyLayout('force-directed')
                    },
                    {
                        id: 'circular',
                        text: 'Circular',
                        action: () => this.applyLayout('circular')
                    },
                    {
                        id: 'hierarchical',
                        text: 'Hierarchical',
                        action: () => this.applyLayout('hierarchical')
                    },
                    {
                        id: 'grid',
                        text: 'Grid',
                        action: () => this.applyLayout('grid')
                    }
                ]
            },
            {
                id: 'export-submenu',
                text: 'Export',
                icon: '💾',
                submenu: [
                    {
                        id: 'export-svg',
                        text: 'Export as SVG',
                        action: () => this.exportGraph('svg')
                    },
                    {
                        id: 'export-png',
                        text: 'Export as PNG',
                        action: () => this.exportGraph('png')
                    },
                    {
                        id: 'export-pdf',
                        text: 'Export as PDF',
                        action: () => this.exportGraph('pdf')
                    }
                ]
            }
        ];
    }
    
    /**
     * Create multiple nodes context menu template
     */
    createMultipleNodesMenuTemplate() {
        return [
            {
                type: 'header',
                text: 'Multiple Nodes'
            },
            {
                id: 'group-nodes',
                text: 'Group Nodes',
                icon: '📦',
                action: (nodes) => this.groupNodes(nodes)
            },
            {
                id: 'align-nodes',
                text: 'Align Nodes',
                icon: '📏',
                submenu: [
                    {
                        id: 'align-left',
                        text: 'Align Left',
                        action: (nodes) => this.alignNodes(nodes, 'left')
                    },
                    {
                        id: 'align-center',
                        text: 'Align Center',
                        action: (nodes) => this.alignNodes(nodes, 'center')
                    },
                    {
                        id: 'align-right',
                        text: 'Align Right',
                        action: (nodes) => this.alignNodes(nodes, 'right')
                    },
                    {
                        type: 'separator'
                    },
                    {
                        id: 'align-top',
                        text: 'Align Top',
                        action: (nodes) => this.alignNodes(nodes, 'top')
                    },
                    {
                        id: 'align-middle',
                        text: 'Align Middle',
                        action: (nodes) => this.alignNodes(nodes, 'middle')
                    },
                    {
                        id: 'align-bottom',
                        text: 'Align Bottom',
                        action: (nodes) => this.alignNodes(nodes, 'bottom')
                    }
                ]
            },
            {
                id: 'distribute-nodes',
                text: 'Distribute Nodes',
                icon: '↔️',
                submenu: [
                    {
                        id: 'distribute-horizontal',
                        text: 'Horizontally',
                        action: (nodes) => this.distributeNodes(nodes, 'horizontal')
                    },
                    {
                        id: 'distribute-vertical',
                        text: 'Vertically',
                        action: (nodes) => this.distributeNodes(nodes, 'vertical')
                    }
                ]
            },
            {
                type: 'separator'
            },
            {
                id: 'connect-nodes',
                text: 'Connect All',
                icon: '🔗',
                action: (nodes) => this.connectNodes(nodes)
            },
            {
                id: 'copy-nodes',
                text: 'Copy',
                icon: '📋',
                shortcut: 'Ctrl+C',
                action: (nodes) => this.copyNodes(nodes)
            },
            {
                id: 'delete-nodes',
                text: 'Delete All',
                icon: '🗑️',
                shortcut: 'Delete',
                action: (nodes) => this.deleteNodes(nodes),
                className: 'destructive'
            }
        ];
    }
    
    /**
     * Handle context menu events
     */
    handleContextMenu(event) {
        if (!this.config.enableContextMenus) return;
        
        if (this.config.preventDefaultContextMenu) {
            event.preventDefault();
        }
        
        const position = this.getMousePosition(event);
        const clickedElement = this.getElementAtPosition(position.x, position.y);
        
        this.showContextMenu(clickedElement, position, event);
    }
    
    /**
     * Show context menu based on clicked element
     */
    showContextMenu(element, position, event) {
        this.hideMenu(); // Hide any existing menu
        
        const menuType = this.determineMenuType(element);
        const menuItems = this.buildMenuItems(menuType, element);
        
        if (menuItems.length === 0) return;
        
        this.menuState.isVisible = true;
        this.menuState.clickedElement = element;
        this.menuState.clickedPosition = position;
        
        this.renderMenu(menuItems, position);
        
        // Emit event
        this.emitEvent('contextMenuShow', {
            element: element,
            position: position,
            menuType: menuType
        });
    }
    
    /**
     * Determine menu type based on clicked element
     */
    determineMenuType(element) {
        if (!element) {
            return 'canvas';
        }
        
        if (element.type === 'node') {
            const selectedNodes = this.getSelectedNodes();
            if (selectedNodes.length > 1 && selectedNodes.includes(element)) {
                return 'multipleNodes';
            }
            return 'node';
        }
        
        if (element.type === 'edge') {
            const selectedEdges = this.getSelectedEdges();
            if (selectedEdges.length > 1 && selectedEdges.includes(element)) {
                return 'multipleEdges';
            }
            return 'edge';
        }
        
        // Check for mixed selection
        const selection = this.getSelection();
        if (selection.nodes.length > 0 && selection.edges.length > 0) {
            return 'mixed';
        }
        
        return 'canvas';
    }
    
    /**
     * Build menu items based on type and context
     */
    buildMenuItems(menuType, element) {
        let template = this.menuTemplates[menuType] || this.menuTemplates.canvas;
        
        // Add custom actions
        const customActions = this.getCustomActions(menuType);
        if (customActions.length > 0) {
            template = [...template, { type: 'separator' }, ...customActions];
        }
        
        // Filter items based on conditions
        return this.filterMenuItems(template, element);
    }
    
    /**
     * Filter menu items based on conditions
     */
    filterMenuItems(items, element) {
        return items.filter(item => {
            if (item.condition) {
                return item.condition(element);
            }
            return true;
        }).map(item => {
            if (item.submenu) {
                return {
                    ...item,
                    submenu: this.filterMenuItems(item.submenu, element)
                };
            }
            return item;
        });
    }
    
    /**
     * Render the context menu
     */
    renderMenu(items, position) {
        this.menuContainer.innerHTML = '';
        
        items.forEach(item => {
            const menuItem = this.createMenuItem(item);
            this.menuContainer.appendChild(menuItem);
        });
        
        // Position the menu
        this.positionMenu(position);
        
        // Show with animation
        this.menuContainer.style.display = 'block';
        if (this.config.enableAnimations) {
            this.menuContainer.classList.add('fade-in');
        }
    }
    
    /**
     * Create a single menu item element
     */
    createMenuItem(item) {
        if (item.type === 'separator') {
            const separator = document.createElement('div');
            separator.className = 'groggy-context-menu-separator';
            return separator;
        }
        
        if (item.type === 'header') {
            const header = document.createElement('div');
            header.className = 'groggy-context-menu-header';
            header.textContent = item.text;
            return header;
        }
        
        const menuItem = document.createElement('div');
        menuItem.className = 'groggy-context-menu-item';
        
        if (item.className) {
            menuItem.classList.add(item.className);
        }
        
        if (item.disabled) {
            menuItem.classList.add('disabled');
        }
        
        // Icon
        if (this.config.showIcons && item.icon) {
            const icon = document.createElement('span');
            icon.className = 'groggy-context-menu-icon';
            icon.textContent = item.icon;
            menuItem.appendChild(icon);
        }
        
        // Text
        const text = document.createElement('span');
        text.className = 'groggy-context-menu-text';
        text.textContent = item.text;
        menuItem.appendChild(text);
        
        // Keyboard shortcut
        if (this.config.showKeyboardShortcuts && item.shortcut) {
            const shortcut = document.createElement('span');
            shortcut.className = 'groggy-context-menu-shortcut';
            shortcut.textContent = item.shortcut;
            menuItem.appendChild(shortcut);
        }
        
        // Submenu arrow
        if (item.submenu && item.submenu.length > 0) {
            const arrow = document.createElement('span');
            arrow.className = 'groggy-context-menu-arrow';
            arrow.textContent = '▶';
            menuItem.appendChild(arrow);
        }
        
        // Event handlers
        if (!item.disabled) {
            menuItem.addEventListener('click', (e) => {
                e.stopPropagation();
                this.handleMenuItemClick(item);
            });
            
            if (item.submenu) {
                menuItem.addEventListener('mouseenter', () => {
                    this.showSubmenu(item, menuItem);
                });
                
                menuItem.addEventListener('mouseleave', () => {
                    this.scheduleHideSubmenu();
                });
            }
        }
        
        return menuItem;
    }
    
    /**
     * Position the menu relative to click position
     */
    positionMenu(position) {
        const rect = this.menuContainer.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        let x = position.x;
        let y = position.y;
        
        // Adjust for viewport boundaries
        if (x + rect.width > viewportWidth) {
            x = viewportWidth - rect.width - 10;
        }
        
        if (y + rect.height > viewportHeight) {
            y = viewportHeight - rect.height - 10;
        }
        
        // Ensure minimum distance from edges
        x = Math.max(10, x);
        y = Math.max(10, y);
        
        this.menuContainer.style.left = x + 'px';
        this.menuContainer.style.top = y + 'px';
    }
    
    /**
     * Handle menu item clicks
     */
    handleMenuItemClick(item) {
        if (item.disabled) return;
        
        if (item.submenu && item.submenu.length > 0) {
            // Submenu items are handled by hover
            return;
        }
        
        if (item.action) {
            try {
                item.action(this.menuState.clickedElement);
            } catch (error) {
                console.error('Error executing menu action:', error);
            }
        }
        
        this.hideMenu();
        
        // Emit event
        this.emitEvent('contextMenuAction', {
            item: item,
            element: this.menuState.clickedElement
        });
    }
    
    /**
     * Hide the context menu
     */
    hideMenu() {
        if (!this.menuState.isVisible) return;
        
        if (this.config.enableAnimations) {
            this.menuContainer.classList.remove('fade-in');
            this.menuContainer.classList.add('fade-out');
            
            setTimeout(() => {
                this.menuContainer.style.display = 'none';
                this.menuContainer.classList.remove('fade-out');
            }, this.config.animationDuration);
        } else {
            this.menuContainer.style.display = 'none';
        }
        
        this.hideSubmenu();
        
        this.menuState.isVisible = false;
        this.menuState.currentMenu = null;
        this.menuState.clickedElement = null;
        this.menuState.clickedPosition = null;
        
        // Emit event
        this.emitEvent('contextMenuHide', {});
    }
    
    /**
     * Global click handler to hide menu
     */
    handleGlobalClick(event) {
        if (this.menuState.isVisible && !this.menuContainer.contains(event.target)) {
            this.hideMenu();
        }
    }
    
    /**
     * Keyboard event handler
     */
    handleKeyDown(event) {
        if (this.menuState.isVisible) {
            if (event.key === 'Escape') {
                this.hideMenu();
                event.preventDefault();
            }
        }
    }
    
    /**
     * Utility methods for menu actions
     */
    
    getMousePosition(event) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: event.clientX,
            y: event.clientY,
            canvasX: event.clientX - rect.left,
            canvasY: event.clientY - rect.top
        };
    }
    
    getElementAtPosition(x, y) {
        // Convert to canvas coordinates
        const rect = this.canvas.getBoundingClientRect();
        const canvasX = x - rect.left;
        const canvasY = y - rect.top;
        
        // Check for nodes first
        const node = this.graphRenderer.getNodeAtPosition(canvasX, canvasY);
        if (node) {
            return { type: 'node', ...node };
        }
        
        // Check for edges
        const edge = this.graphRenderer.getEdgeAtPosition(canvasX, canvasY);
        if (edge) {
            return { type: 'edge', ...edge };
        }
        
        return null;
    }
    
    getSelectedNodes() {
        return this.graphRenderer.selectionManager?.getSelectedNodes() || [];
    }
    
    getSelectedEdges() {
        return this.graphRenderer.selectionManager?.getSelectedEdges() || [];
    }
    
    getSelection() {
        return {
            nodes: this.getSelectedNodes(),
            edges: this.getSelectedEdges()
        };
    }
    
    isNodeSelected(node) {
        return this.graphRenderer.selectionManager?.isNodeSelected(node.id) || false;
    }
    
    isEdgeSelected(edge) {
        return this.graphRenderer.selectionManager?.isEdgeSelected(edge.id) || false;
    }
    
    hasSelection() {
        const selection = this.getSelection();
        return selection.nodes.length > 0 || selection.edges.length > 0;
    }
    
    hasClipboardData() {
        // Implementation depends on clipboard manager
        return this.graphRenderer.clipboardManager?.hasData() || false;
    }
    
    getCustomActions(menuType) {
        const customKey = `custom${menuType.charAt(0).toUpperCase() + menuType.slice(1)}Actions`;
        return this.config[customKey] || [];
    }
    
    /**
     * Menu action implementations
     * These can be overridden or extended
     */
    
    selectNode(node) {
        this.graphRenderer.selectionManager?.selectNode(node.id);
    }
    
    deselectNode(node) {
        this.graphRenderer.selectionManager?.deselectNode(node.id);
    }
    
    centerOnNode(node) {
        this.graphRenderer.centerOnPosition(node.x, node.y);
    }
    
    highlightNodeConnections(node) {
        this.graphRenderer.highlightConnections([node]);
    }
    
    editNode(node) {
        this.emitEvent('nodeEdit', { node });
    }
    
    duplicateNode(node) {
        this.emitEvent('nodeDuplicate', { node });
    }
    
    expandNeighbors(node, hops) {
        this.emitEvent('expandNeighbors', { node, hops });
    }
    
    hideNode(node) {
        this.graphRenderer.hideNode(node.id);
    }
    
    deleteNode(node) {
        this.emitEvent('nodeDelete', { node });
    }
    
    selectEdge(edge) {
        this.graphRenderer.selectionManager?.selectEdge(edge.id);
    }
    
    deselectEdge(edge) {
        this.graphRenderer.selectionManager?.deselectEdge(edge.id);
    }
    
    editEdge(edge) {
        this.emitEvent('edgeEdit', { edge });
    }
    
    reverseEdge(edge) {
        this.emitEvent('edgeReverse', { edge });
    }
    
    deleteEdge(edge) {
        this.emitEvent('edgeDelete', { edge });
    }
    
    addNode(position) {
        this.emitEvent('nodeAdd', { position: position });
    }
    
    selectAll() {
        this.graphRenderer.selectionManager?.selectAll();
    }
    
    clearSelection() {
        this.graphRenderer.selectionManager?.clearSelection();
    }
    
    fitToView() {
        this.graphRenderer.fitToView();
    }
    
    centerView() {
        this.graphRenderer.centerView();
    }
    
    applyLayout(layoutType) {
        this.emitEvent('layoutApply', { layoutType });
    }
    
    exportGraph(format) {
        this.emitEvent('graphExport', { format });
    }
    
    alignNodes(nodes, alignment) {
        this.emitEvent('nodesAlign', { nodes, alignment });
    }
    
    distributeNodes(nodes, direction) {
        this.emitEvent('nodesDistribute', { nodes, direction });
    }
    
    groupNodes(nodes) {
        this.emitEvent('nodesGroup', { nodes });
    }
    
    connectNodes(nodes) {
        this.emitEvent('nodesConnect', { nodes });
    }
    
    copyNodes(nodes) {
        this.graphRenderer.clipboardManager?.copy(nodes);
    }
    
    deleteNodes(nodes) {
        this.emitEvent('nodesDelete', { nodes });
    }
    
    /**
     * Event system
     */
    emitEvent(eventType, data) {
        const event = new CustomEvent(`contextMenu${eventType.charAt(0).toUpperCase() + eventType.slice(1)}`, {
            detail: data
        });
        this.canvas.dispatchEvent(event);
    }
    
    /**
     * Public API methods
     */
    
    /**
     * Add custom menu items
     */
    addCustomAction(menuType, action) {
        const customKey = `custom${menuType.charAt(0).toUpperCase() + menuType.slice(1)}Actions`;
        if (!this.config[customKey]) {
            this.config[customKey] = [];
        }
        this.config[customKey].push(action);
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        Object.assign(this.config, newConfig);
    }
    
    /**
     * Enable or disable context menus
     */
    setEnabled(enabled) {
        this.config.enableContextMenus = enabled;
        if (!enabled) {
            this.hideMenu();
        }
    }
    
    /**
     * Check if menu is currently visible
     */
    isVisible() {
        return this.menuState.isVisible;
    }
    
    /**
     * Cleanup and destroy
     */
    destroy() {
        // Remove event listeners
        this.canvas.removeEventListener('contextmenu', this.boundHandlers.contextmenu);
        document.removeEventListener('click', this.boundHandlers.click);
        document.removeEventListener('keydown', this.boundHandlers.keydown);
        
        if (this.boundHandlers.scroll) {
            window.removeEventListener('scroll', this.boundHandlers.scroll, true);
        }
        
        if (this.boundHandlers.resize) {
            window.removeEventListener('resize', this.boundHandlers.resize);
        }
        
        // Remove menu container
        if (this.menuContainer && this.menuContainer.parentNode) {
            this.menuContainer.parentNode.removeChild(this.menuContainer);
        }
        
        // Clear state
        this.menuState.isVisible = false;
        this.menuContainer = null;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ContextMenuSystem;
}