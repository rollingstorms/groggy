/**
 * 🎛️ Layout Configuration Panel
 * Part of Groggy Phase 11: Layout Algorithms Enhancement
 * 
 * Provides interactive configuration interface for layout algorithms with:
 * - Real-time parameter adjustment
 * - Algorithm selection and comparison
 * - Animation control and preview
 * - Parameter validation and constraints
 * - Export/import configurations
 * - Performance monitoring integration
 */

class LayoutConfigPanel {
    constructor(container, layoutSystem, config = {}) {
        this.container = container;
        this.layoutSystem = layoutSystem; // Reference to layout engine/registry
        this.config = {
            // Panel appearance
            position: 'right', // 'left', 'right', 'bottom', 'floating'
            width: 350,
            height: 'auto',
            collapsible: true,
            draggable: true,
            
            // Functionality
            enablePreview: true,
            enableComparison: true,
            enableExport: true,
            enablePresets: true,
            
            // Real-time updates
            updateDelay: 300, // ms
            enableLivePreview: true,
            
            // UI preferences
            theme: 'dark',
            compactMode: false,
            showAdvanced: false,
            
            ...config
        };
        
        // State
        this.isVisible = false;
        this.isCollapsed = false;
        this.currentLayout = null;
        this.currentParameters = new Map();
        this.updateTimeout = null;
        this.previewTimeout = null;
        
        // Available layouts and their configurations
        this.availableLayouts = new Map();
        this.layoutPresets = new Map();
        
        // Event handlers
        this.eventHandlers = new Map();
        
        // UI elements
        this.panelElement = null;
        this.headerElement = null;
        this.contentElement = null;
        this.controlsContainer = null;
        
        this.initializePanel();
        this.loadAvailableLayouts();
        this.setupEventListeners();
        
        console.log('🎛️ LayoutConfigPanel initialized with config:', this.config);
    }
    
    /**
     * Initialize the panel UI structure
     */
    initializePanel() {
        // Create main panel element
        this.panelElement = document.createElement('div');
        this.panelElement.className = `layout-config-panel layout-config-panel--${this.config.position} layout-config-panel--${this.config.theme}`;
        this.panelElement.style.width = `${this.config.width}px`;
        
        if (this.config.position === 'floating') {
            this.panelElement.style.position = 'fixed';
            this.panelElement.style.top = '20px';
            this.panelElement.style.right = '20px';
            this.panelElement.style.zIndex = '10000';
        }
        
        // Create header
        this.headerElement = document.createElement('div');
        this.headerElement.className = 'layout-config-panel__header';
        this.headerElement.innerHTML = `
            <div class="layout-config-panel__title">
                <span class="layout-config-panel__icon">🎛️</span>
                Layout Configuration
            </div>
            <div class="layout-config-panel__header-controls">
                <button class="layout-config-panel__preset-btn" title="Load Preset">📋</button>
                <button class="layout-config-panel__comparison-btn" title="Compare Layouts">📊</button>
                <button class="layout-config-panel__export-btn" title="Export Config">💾</button>
                <button class="layout-config-panel__collapse-btn" title="Collapse Panel">🔽</button>
                <button class="layout-config-panel__close-btn" title="Close Panel">✕</button>
            </div>
        `;
        
        // Create content area
        this.contentElement = document.createElement('div');
        this.contentElement.className = 'layout-config-panel__content';
        
        // Create main sections
        this.createLayoutSelector();
        this.createParameterControls();
        this.createAnimationControls();
        this.createPreviewControls();
        this.createComparisonTools();
        
        // Assemble panel
        this.panelElement.appendChild(this.headerElement);
        this.panelElement.appendChild(this.contentElement);
        
        // Add to container
        this.container.appendChild(this.panelElement);
        
        // Add styles
        this.addPanelStyles();
        
        // Make draggable if configured
        if (this.config.draggable && this.config.position === 'floating') {
            this.makeDraggable();
        }
    }
    
    /**
     * Create layout algorithm selector
     */
    createLayoutSelector() {
        const selectorSection = document.createElement('div');
        selectorSection.className = 'layout-config-section';
        selectorSection.innerHTML = `
            <h3 class="layout-config-section__title">Algorithm</h3>
            <div class="layout-config-section__content">
                <select class="layout-selector" id="layout-algorithm-select">
                    <option value="">Select Layout Algorithm</option>
                </select>
                <div class="layout-info">
                    <div class="layout-description" id="layout-description"></div>
                    <div class="layout-metadata" id="layout-metadata"></div>
                </div>
            </div>
        `;
        
        this.contentElement.appendChild(selectorSection);
        this.layoutSelector = selectorSection.querySelector('.layout-selector');
    }
    
    /**
     * Create parameter controls section
     */
    createParameterControls() {
        this.controlsContainer = document.createElement('div');
        this.controlsContainer.className = 'layout-config-section';
        this.controlsContainer.innerHTML = `
            <h3 class="layout-config-section__title">
                Parameters
                <button class="advanced-toggle" id="advanced-toggle">Advanced ⚙️</button>
            </h3>
            <div class="layout-config-section__content">
                <div class="parameter-controls" id="parameter-controls"></div>
                <div class="parameter-actions">
                    <button class="reset-params-btn" id="reset-params">Reset to Defaults</button>
                    <button class="randomize-params-btn" id="randomize-params">Randomize</button>
                </div>
            </div>
        `;
        
        this.contentElement.appendChild(this.controlsContainer);
        this.parameterControlsContainer = this.controlsContainer.querySelector('#parameter-controls');
    }
    
    /**
     * Create animation controls
     */
    createAnimationControls() {
        const animationSection = document.createElement('div');
        animationSection.className = 'layout-config-section';
        animationSection.innerHTML = `
            <h3 class="layout-config-section__title">Animation</h3>
            <div class="layout-config-section__content">
                <div class="animation-control">
                    <label for="animation-duration">Duration (ms)</label>
                    <input type="range" id="animation-duration" min="100" max="5000" value="1500" step="100">
                    <span class="value-display" id="duration-value">1500</span>
                </div>
                <div class="animation-control">
                    <label for="animation-easing">Easing</label>
                    <select id="animation-easing">
                        <option value="linear">Linear</option>
                        <option value="easeInOutCubic" selected>Ease In-Out Cubic</option>
                        <option value="easeInOutQuad">Ease In-Out Quad</option>
                        <option value="easeOutBounce">Bounce</option>
                        <option value="easeInOutElastic">Elastic</option>
                        <option value="spring">Spring</option>
                    </select>
                </div>
                <div class="animation-control">
                    <label for="animation-stagger">Stagger</label>
                    <input type="range" id="animation-stagger" min="0" max="1" value="0" step="0.05">
                    <span class="value-display" id="stagger-value">0</span>
                </div>
                <div class="animation-presets">
                    <button class="preset-btn" data-preset="smooth">Smooth</button>
                    <button class="preset-btn" data-preset="quick">Quick</button>
                    <button class="preset-btn" data-preset="dramatic">Dramatic</button>
                    <button class="preset-btn" data-preset="bouncy">Bouncy</button>
                </div>
            </div>
        `;
        
        this.contentElement.appendChild(animationSection);
    }
    
    /**
     * Create preview controls
     */
    createPreviewControls() {
        const previewSection = document.createElement('div');
        previewSection.className = 'layout-config-section';
        previewSection.innerHTML = `
            <h3 class="layout-config-section__title">Preview & Actions</h3>
            <div class="layout-config-section__content">
                <div class="preview-controls">
                    <label class="checkbox-control">
                        <input type="checkbox" id="live-preview" ${this.config.enableLivePreview ? 'checked' : ''}>
                        <span>Live Preview</span>
                    </label>
                    <label class="checkbox-control">
                        <input type="checkbox" id="show-ghost-nodes" checked>
                        <span>Ghost Nodes</span>
                    </label>
                </div>
                <div class="action-buttons">
                    <button class="action-btn preview-btn" id="preview-layout">👁️ Preview</button>
                    <button class="action-btn apply-btn" id="apply-layout">✅ Apply</button>
                    <button class="action-btn compute-btn" id="compute-layout">⚡ Compute</button>
                </div>
                <div class="performance-info" id="performance-info">
                    <div class="perf-metric">
                        <label>Compute Time:</label>
                        <span id="compute-time">--</span>
                    </div>
                    <div class="perf-metric">
                        <label>Quality Score:</label>
                        <span id="quality-score">--</span>
                    </div>
                </div>
            </div>
        `;
        
        this.contentElement.appendChild(previewSection);
    }
    
    /**
     * Create comparison tools
     */
    createComparisonTools() {
        if (!this.config.enableComparison) return;
        
        const comparisonSection = document.createElement('div');
        comparisonSection.className = 'layout-config-section layout-config-section--comparison';
        comparisonSection.innerHTML = `
            <h3 class="layout-config-section__title">Layout Comparison</h3>
            <div class="layout-config-section__content">
                <div class="comparison-layouts">
                    <select id="compare-layout-1">
                        <option value="">Layout A</option>
                    </select>
                    <span class="vs-label">vs</span>
                    <select id="compare-layout-2">
                        <option value="">Layout B</option>
                    </select>
                </div>
                <div class="comparison-metrics" id="comparison-metrics">
                    <!-- Metrics will be populated dynamically -->
                </div>
                <button class="action-btn compare-btn" id="start-comparison">📊 Compare</button>
            </div>
        `;
        
        this.contentElement.appendChild(comparisonSection);
    }
    
    /**
     * Load available layouts from the layout registry
     */
    loadAvailableLayouts() {
        if (!this.layoutSystem || !this.layoutSystem.listLayouts) {
            console.warn('Layout system not available or incompatible');
            return;
        }
        
        try {
            const layouts = this.layoutSystem.listLayouts();
            
            layouts.forEach(layoutName => {
                const pluginInfo = this.layoutSystem.getPluginInfo(layoutName);
                if (pluginInfo) {
                    this.availableLayouts.set(layoutName, pluginInfo);
                    
                    // Add to selector
                    const option = document.createElement('option');
                    option.value = layoutName;
                    option.textContent = `${pluginInfo.name} - ${pluginInfo.description}`;
                    this.layoutSelector.appendChild(option);
                }
            });
            
            console.log(`🎛️ Loaded ${this.availableLayouts.size} layout algorithms`);
            
        } catch (error) {
            console.error('Failed to load available layouts:', error);
        }
    }
    
    /**
     * Create parameter control for a specific parameter
     */
    createParameterControl(paramName, paramSpec) {
        const controlDiv = document.createElement('div');
        controlDiv.className = `parameter-control parameter-control--${paramSpec.parameter_type.toLowerCase()}`;
        
        const isAdvanced = paramSpec.tags && paramSpec.tags.includes('advanced');
        if (isAdvanced) {
            controlDiv.classList.add('parameter-control--advanced');
        }
        
        let controlHtml = `
            <label class="parameter-label" for="param-${paramName}">
                ${paramSpec.name}
                <span class="parameter-description">${paramSpec.description}</span>
            </label>
        `;
        
        switch (paramSpec.parameter_type) {
            case 'Float':
                const constraints = paramSpec.constraints || {};
                const min = constraints.min_value || 0;
                const max = constraints.max_value || 100;
                const step = (max - min) / 100;
                
                controlHtml += `
                    <div class="range-control">
                        <input 
                            type="range" 
                            id="param-${paramName}" 
                            min="${min}" 
                            max="${max}" 
                            step="${step}"
                            value="${paramSpec.default_value}"
                            data-param="${paramName}"
                        >
                        <input 
                            type="number" 
                            class="number-input"
                            min="${min}" 
                            max="${max}" 
                            step="${step}"
                            value="${paramSpec.default_value}"
                            data-param="${paramName}"
                        >
                    </div>
                `;
                break;
                
            case 'Integer':
                const intConstraints = paramSpec.constraints || {};
                const intMin = intConstraints.min_value || 0;
                const intMax = intConstraints.max_value || 1000;
                
                controlHtml += `
                    <div class="range-control">
                        <input 
                            type="range" 
                            id="param-${paramName}" 
                            min="${intMin}" 
                            max="${intMax}" 
                            step="1"
                            value="${paramSpec.default_value}"
                            data-param="${paramName}"
                        >
                        <input 
                            type="number" 
                            class="number-input"
                            min="${intMin}" 
                            max="${intMax}" 
                            step="1"
                            value="${paramSpec.default_value}"
                            data-param="${paramName}"
                        >
                    </div>
                `;
                break;
                
            case 'Boolean':
                controlHtml += `
                    <label class="checkbox-control">
                        <input 
                            type="checkbox" 
                            id="param-${paramName}" 
                            ${paramSpec.default_value ? 'checked' : ''}
                            data-param="${paramName}"
                        >
                        <span>Enable</span>
                    </label>
                `;
                break;
                
            case 'String':
                controlHtml += `
                    <input 
                        type="text" 
                        id="param-${paramName}" 
                        value="${paramSpec.default_value}"
                        data-param="${paramName}"
                        class="text-input"
                    >
                `;
                break;
                
            case 'Choice':
                controlHtml += '<select id="param-' + paramName + '" data-param="' + paramName + '" class="select-input">';
                if (paramSpec.parameter_type.choices) {
                    paramSpec.parameter_type.choices.forEach(choice => {
                        const selected = choice === paramSpec.default_value ? 'selected' : '';
                        controlHtml += `<option value="${choice}" ${selected}>${choice}</option>`;
                    });
                }
                controlHtml += '</select>';
                break;
        }
        
        controlDiv.innerHTML = controlHtml;
        return controlDiv;
    }
    
    /**
     * Update parameter controls when layout changes
     */
    updateParameterControls(layoutName) {
        const pluginInfo = this.availableLayouts.get(layoutName);
        if (!pluginInfo) return;
        
        // Clear existing controls
        this.parameterControlsContainer.innerHTML = '';
        this.currentParameters.clear();
        
        // Create controls for each parameter
        Object.entries(pluginInfo.parameters).forEach(([paramName, paramSpec]) => {
            const control = this.createParameterControl(paramName, paramSpec);
            this.parameterControlsContainer.appendChild(control);
            
            // Store default value
            this.currentParameters.set(paramName, paramSpec.default_value);
        });
        
        // Update layout info
        this.updateLayoutInfo(pluginInfo);
        
        // Setup parameter change listeners
        this.setupParameterListeners();
    }
    
    /**
     * Update layout information display
     */
    updateLayoutInfo(pluginInfo) {
        const descriptionEl = document.getElementById('layout-description');
        const metadataEl = document.getElementById('layout-metadata');
        
        if (descriptionEl) {
            descriptionEl.textContent = pluginInfo.description;
        }
        
        if (metadataEl) {
            metadataEl.innerHTML = `
                <div class="metadata-item">
                    <strong>Author:</strong> ${pluginInfo.author || 'Unknown'}
                </div>
                <div class="metadata-item">
                    <strong>Version:</strong> ${pluginInfo.version || 'N/A'}
                </div>
                <div class="metadata-item">
                    <strong>Tags:</strong> ${pluginInfo.tags ? pluginInfo.tags.join(', ') : 'None'}
                </div>
            `;
        }
    }
    
    /**
     * Setup event listeners for parameter controls
     */
    setupParameterListeners() {
        const parameterControls = this.parameterControlsContainer.querySelectorAll('[data-param]');
        
        parameterControls.forEach(control => {
            const paramName = control.getAttribute('data-param');
            
            control.addEventListener('input', (event) => {
                this.handleParameterChange(paramName, event.target);
            });
            
            // Special handling for number inputs with range sync
            if (control.type === 'number') {
                const rangeInput = this.parameterControlsContainer.querySelector(`input[type="range"][data-param="${paramName}"]`);
                if (rangeInput) {
                    control.addEventListener('input', () => {
                        rangeInput.value = control.value;
                    });
                }
            }
            
            if (control.type === 'range') {
                const numberInput = this.parameterControlsContainer.querySelector(`input[type="number"][data-param="${paramName}"]`);
                if (numberInput) {
                    control.addEventListener('input', () => {
                        numberInput.value = control.value;
                    });
                }
            }
        });
    }
    
    /**
     * Handle parameter value changes
     */
    handleParameterChange(paramName, inputElement) {
        let value;
        
        switch (inputElement.type) {
            case 'checkbox':
                value = inputElement.checked;
                break;
            case 'number':
            case 'range':
                value = parseFloat(inputElement.value);
                break;
            default:
                value = inputElement.value;
        }
        
        this.currentParameters.set(paramName, value);
        
        // Trigger live preview if enabled
        if (this.config.enableLivePreview && document.getElementById('live-preview').checked) {
            this.schedulePreview();
        }
        
        // Emit parameter change event
        this.emit('parameterChange', { paramName, value, allParameters: this.getAllParameters() });
    }
    
    /**
     * Schedule a preview update with debouncing
     */
    schedulePreview() {
        if (this.previewTimeout) {
            clearTimeout(this.previewTimeout);
        }
        
        this.previewTimeout = setTimeout(() => {
            this.previewCurrentLayout();
        }, this.config.updateDelay);
    }
    
    /**
     * Preview current layout configuration
     */
    previewCurrentLayout() {
        if (!this.currentLayout) return;
        
        const parameters = this.getAllParameters();
        
        try {
            // Get layout with current parameters
            const layout = this.layoutSystem.createLayoutWithParams(this.currentLayout, parameters);
            
            // Show preview through animator if available
            if (window.GroggyLayoutAnimator) {
                // This would compute and preview the layout
                console.log('🎛️ Previewing layout:', this.currentLayout, parameters);
                
                // Emit preview event
                this.emit('layoutPreview', { layout: this.currentLayout, parameters });
            }
            
        } catch (error) {
            console.error('Failed to preview layout:', error);
            this.showError('Preview failed: ' + error.message);
        }
    }
    
    /**
     * Apply current layout configuration
     */
    applyCurrentLayout() {
        if (!this.currentLayout) {
            this.showError('No layout selected');
            return;
        }
        
        const parameters = this.getAllParameters();
        const startTime = performance.now();
        
        try {
            // Apply layout with animation
            const animationConfig = this.getAnimationConfig();
            
            // Emit apply event
            this.emit('layoutApply', { 
                layout: this.currentLayout, 
                parameters, 
                animation: animationConfig 
            });
            
            // Update performance metrics
            const computeTime = performance.now() - startTime;
            this.updatePerformanceMetrics({ computeTime });
            
            console.log('🎛️ Applied layout:', this.currentLayout, parameters);
            
        } catch (error) {
            console.error('Failed to apply layout:', error);
            this.showError('Apply failed: ' + error.message);
        }
    }
    
    /**
     * Get all current parameter values
     */
    getAllParameters() {
        const parameters = {};
        this.currentParameters.forEach((value, key) => {
            parameters[key] = value;
        });
        return parameters;
    }
    
    /**
     * Get current animation configuration
     */
    getAnimationConfig() {
        return {
            duration: parseInt(document.getElementById('animation-duration').value),
            easing: document.getElementById('animation-easing').value,
            stagger: parseFloat(document.getElementById('animation-stagger').value),
            enablePreview: document.getElementById('show-ghost-nodes').checked
        };
    }
    
    /**
     * Setup main event listeners
     */
    setupEventListeners() {
        // Layout selector
        this.layoutSelector.addEventListener('change', (event) => {
            this.currentLayout = event.target.value;
            if (this.currentLayout) {
                this.updateParameterControls(this.currentLayout);
            }
        });
        
        // Header controls
        this.headerElement.addEventListener('click', (event) => {
            const target = event.target;
            
            if (target.classList.contains('layout-config-panel__collapse-btn')) {
                this.toggleCollapse();
            } else if (target.classList.contains('layout-config-panel__close-btn')) {
                this.hide();
            } else if (target.classList.contains('layout-config-panel__export-btn')) {
                this.exportConfiguration();
            } else if (target.classList.contains('layout-config-panel__preset-btn')) {
                this.showPresets();
            } else if (target.classList.contains('layout-config-panel__comparison-btn')) {
                this.toggleComparison();
            }
        });
        
        // Action buttons
        const applyBtn = document.getElementById('apply-layout');
        const previewBtn = document.getElementById('preview-layout');
        const computeBtn = document.getElementById('compute-layout');
        
        if (applyBtn) applyBtn.addEventListener('click', () => this.applyCurrentLayout());
        if (previewBtn) previewBtn.addEventListener('click', () => this.previewCurrentLayout());
        if (computeBtn) computeBtn.addEventListener('click', () => this.computeLayout());
        
        // Animation controls
        const durationSlider = document.getElementById('animation-duration');
        const staggerSlider = document.getElementById('animation-stagger');
        
        if (durationSlider) {
            durationSlider.addEventListener('input', (event) => {
                document.getElementById('duration-value').textContent = event.target.value;
            });
        }
        
        if (staggerSlider) {
            staggerSlider.addEventListener('input', (event) => {
                document.getElementById('stagger-value').textContent = event.target.value;
            });
        }
        
        // Animation presets
        this.contentElement.addEventListener('click', (event) => {
            if (event.target.classList.contains('preset-btn')) {
                this.applyAnimationPreset(event.target.getAttribute('data-preset'));
            }
        });
        
        // Parameter actions
        const resetBtn = document.getElementById('reset-params');
        const randomizeBtn = document.getElementById('randomize-params');
        
        if (resetBtn) resetBtn.addEventListener('click', () => this.resetParameters());
        if (randomizeBtn) randomizeBtn.addEventListener('click', () => this.randomizeParameters());
        
        // Advanced toggle
        const advancedToggle = document.getElementById('advanced-toggle');
        if (advancedToggle) {
            advancedToggle.addEventListener('click', () => this.toggleAdvancedMode());
        }
    }
    
    /**
     * Apply animation preset
     */
    applyAnimationPreset(presetName) {
        // Get preset from LayoutTransitionPresets if available
        if (window.LayoutTransitionPresets && window.LayoutTransitionPresets[presetName.toUpperCase()]) {
            const preset = window.LayoutTransitionPresets[presetName.toUpperCase()];
            
            document.getElementById('animation-duration').value = preset.duration;
            document.getElementById('duration-value').textContent = preset.duration;
            document.getElementById('animation-easing').value = preset.easing;
            document.getElementById('animation-stagger').value = preset.stagger || 0;
            document.getElementById('stagger-value').textContent = preset.stagger || 0;
            
            console.log(`🎛️ Applied animation preset: ${presetName}`);
        }
    }
    
    /**
     * Update performance metrics display
     */
    updatePerformanceMetrics(metrics) {
        const computeTimeEl = document.getElementById('compute-time');
        const qualityScoreEl = document.getElementById('quality-score');
        
        if (computeTimeEl && metrics.computeTime) {
            computeTimeEl.textContent = `${metrics.computeTime.toFixed(1)}ms`;
        }
        
        if (qualityScoreEl && metrics.qualityScore) {
            qualityScoreEl.textContent = metrics.qualityScore.toFixed(2);
        }
    }
    
    /**
     * Show/hide panel
     */
    show() {
        this.isVisible = true;
        this.panelElement.style.display = 'block';
        this.panelElement.classList.add('layout-config-panel--visible');
        
        this.emit('panelShow');
    }
    
    hide() {
        this.isVisible = false;
        this.panelElement.classList.remove('layout-config-panel--visible');
        
        setTimeout(() => {
            if (!this.isVisible) {
                this.panelElement.style.display = 'none';
            }
        }, 300);
        
        this.emit('panelHide');
    }
    
    toggle() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }
    
    /**
     * Toggle collapsed state
     */
    toggleCollapse() {
        this.isCollapsed = !this.isCollapsed;
        
        if (this.isCollapsed) {
            this.contentElement.style.display = 'none';
            this.panelElement.classList.add('layout-config-panel--collapsed');
            this.headerElement.querySelector('.layout-config-panel__collapse-btn').textContent = '🔼';
        } else {
            this.contentElement.style.display = 'block';
            this.panelElement.classList.remove('layout-config-panel--collapsed');
            this.headerElement.querySelector('.layout-config-panel__collapse-btn').textContent = '🔽';
        }
    }
    
    /**
     * Export current configuration
     */
    exportConfiguration() {
        const config = {
            layout: this.currentLayout,
            parameters: this.getAllParameters(),
            animation: this.getAnimationConfig(),
            timestamp: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `groggy-layout-config-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
        
        console.log('🎛️ Configuration exported');
    }
    
    /**
     * Add CSS styles for the panel
     */
    addPanelStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .layout-config-panel {
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 8px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
                overflow: hidden;
            }
            
            .layout-config-panel--floating {
                position: fixed;
                z-index: 10000;
            }
            
            .layout-config-panel__header {
                background: #333;
                padding: 12px 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #444;
                cursor: move;
            }
            
            .layout-config-panel__title {
                color: #fff;
                font-weight: bold;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .layout-config-panel__header-controls {
                display: flex;
                gap: 4px;
            }
            
            .layout-config-panel__header-controls button {
                background: #555;
                border: none;
                color: #fff;
                padding: 4px 8px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            }
            
            .layout-config-panel__header-controls button:hover {
                background: #666;
            }
            
            .layout-config-panel__content {
                max-height: 70vh;
                overflow-y: auto;
                color: #ddd;
            }
            
            .layout-config-section {
                border-bottom: 1px solid #333;
                padding: 16px;
            }
            
            .layout-config-section:last-child {
                border-bottom: none;
            }
            
            .layout-config-section__title {
                margin: 0 0 12px 0;
                color: #fff;
                font-size: 14px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .layout-selector,
            .select-input {
                width: 100%;
                padding: 8px;
                background: #444;
                border: 1px solid #555;
                color: #fff;
                border-radius: 4px;
                margin-bottom: 8px;
            }
            
            .layout-info {
                margin-top: 12px;
                padding: 12px;
                background: #333;
                border-radius: 4px;
                font-size: 11px;
            }
            
            .parameter-control {
                margin-bottom: 16px;
                padding: 12px;
                background: #333;
                border-radius: 4px;
            }
            
            .parameter-control--advanced {
                display: none;
            }
            
            .layout-config-panel--advanced .parameter-control--advanced {
                display: block;
            }
            
            .parameter-label {
                display: block;
                margin-bottom: 8px;
                color: #fff;
                font-weight: bold;
            }
            
            .parameter-description {
                display: block;
                font-weight: normal;
                color: #aaa;
                font-size: 10px;
                margin-top: 4px;
            }
            
            .range-control {
                display: flex;
                gap: 8px;
                align-items: center;
            }
            
            .range-control input[type="range"] {
                flex: 1;
            }
            
            .range-control input[type="number"] {
                width: 80px;
                padding: 4px;
                background: #444;
                border: 1px solid #555;
                color: #fff;
                border-radius: 4px;
            }
            
            .checkbox-control {
                display: flex;
                align-items: center;
                gap: 8px;
                cursor: pointer;
            }
            
            .action-buttons {
                display: flex;
                gap: 8px;
                margin-top: 16px;
                flex-wrap: wrap;
            }
            
            .action-btn {
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                flex: 1;
                min-width: 80px;
            }
            
            .preview-btn {
                background: #4a90e2;
                color: white;
            }
            
            .apply-btn {
                background: #5cb85c;
                color: white;
            }
            
            .compute-btn {
                background: #f0ad4e;
                color: white;
            }
            
            .action-btn:hover {
                opacity: 0.8;
            }
            
            .performance-info {
                margin-top: 12px;
                padding: 8px;
                background: #333;
                border-radius: 4px;
                display: flex;
                justify-content: space-between;
                font-size: 11px;
            }
            
            .animation-presets {
                display: flex;
                gap: 4px;
                margin-top: 8px;
                flex-wrap: wrap;
            }
            
            .preset-btn {
                padding: 4px 8px;
                border: 1px solid #555;
                background: #444;
                color: #fff;
                border-radius: 4px;
                cursor: pointer;
                font-size: 10px;
            }
            
            .preset-btn:hover {
                background: #555;
            }
            
            .advanced-toggle {
                background: #666;
                border: none;
                color: #fff;
                padding: 2px 6px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 10px;
            }
            
            .comparison-layouts {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 12px;
            }
            
            .vs-label {
                color: #fff;
                font-weight: bold;
            }
            
            .error-message {
                background: #d32f2f;
                color: white;
                padding: 8px;
                border-radius: 4px;
                margin: 8px 0;
                font-size: 11px;
            }
        `;
        
        document.head.appendChild(style);
    }
    
    /**
     * Make panel draggable
     */
    makeDraggable() {
        let isDragging = false;
        let dragOffset = { x: 0, y: 0 };
        
        this.headerElement.addEventListener('mousedown', (event) => {
            if (event.target.tagName === 'BUTTON') return;
            
            isDragging = true;
            const rect = this.panelElement.getBoundingClientRect();
            dragOffset.x = event.clientX - rect.left;
            dragOffset.y = event.clientY - rect.top;
            
            this.panelElement.style.transition = 'none';
        });
        
        document.addEventListener('mousemove', (event) => {
            if (!isDragging) return;
            
            const x = event.clientX - dragOffset.x;
            const y = event.clientY - dragOffset.y;
            
            this.panelElement.style.left = `${x}px`;
            this.panelElement.style.top = `${y}px`;
            this.panelElement.style.right = 'auto';
        });
        
        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                this.panelElement.style.transition = 'all 0.3s ease';
            }
        });
    }
    
    /**
     * Show error message
     */
    showError(message) {
        // Remove existing error messages
        const existingErrors = this.contentElement.querySelectorAll('.error-message');
        existingErrors.forEach(error => error.remove());
        
        // Create new error message
        const errorEl = document.createElement('div');
        errorEl.className = 'error-message';
        errorEl.textContent = message;
        
        // Insert at top of content
        this.contentElement.insertBefore(errorEl, this.contentElement.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            errorEl.remove();
        }, 5000);
    }
    
    /**
     * Event system
     */
    on(eventType, handler) {
        if (!this.eventHandlers.has(eventType)) {
            this.eventHandlers.set(eventType, []);
        }
        this.eventHandlers.get(eventType).push(handler);
    }
    
    emit(eventType, data) {
        if (this.eventHandlers.has(eventType)) {
            this.eventHandlers.get(eventType).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in ${eventType} handler:`, error);
                }
            });
        }
    }
    
    /**
     * Cleanup and destroy
     */
    destroy() {
        if (this.updateTimeout) clearTimeout(this.updateTimeout);
        if (this.previewTimeout) clearTimeout(this.previewTimeout);
        
        this.panelElement.remove();
        this.eventHandlers.clear();
        
        console.log('🎛️ LayoutConfigPanel destroyed');
    }
}

// Global instance
window.GroggyLayoutConfigPanel = null;

/**
 * Initialize layout configuration panel
 */
function initializeLayoutConfigPanel(container, layoutSystem, config = {}) {
    if (window.GroggyLayoutConfigPanel) {
        console.warn('Layout config panel already initialized');
        return window.GroggyLayoutConfigPanel;
    }
    
    window.GroggyLayoutConfigPanel = new LayoutConfigPanel(container, layoutSystem, config);
    
    // Add keyboard shortcut to toggle panel (Ctrl+Shift+L)
    document.addEventListener('keydown', (event) => {
        if (event.ctrlKey && event.shiftKey && event.key === 'L') {
            event.preventDefault();
            window.GroggyLayoutConfigPanel.toggle();
        }
    });
    
    console.log('🎛️ Layout configuration panel initialized (Ctrl+Shift+L to toggle)');
    
    return window.GroggyLayoutConfigPanel;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        LayoutConfigPanel,
        initializeLayoutConfigPanel
    };
}