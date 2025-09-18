/**
 * 🎛️ Export Configuration Panel
 * Part of Groggy Phase 12: Static Export System
 * 
 * Interactive configuration interface for graph exports with:
 * - Format-specific options (SVG, PNG, PDF)
 * - Quality and DPI settings
 * - Style and appearance controls
 * - Batch export configuration
 * - Preview and validation
 * - Export templates and presets
 */

class ExportConfigPanel {
    constructor(container, exportSystem, config = {}) {
        this.container = container;
        this.exportSystem = exportSystem;
        this.config = {
            // Panel appearance
            position: 'right',
            width: 380,
            height: 'auto',
            collapsible: true,
            
            // Export presets
            enablePresets: true,
            enableTemplates: true,
            
            // Preview options
            enablePreview: true,
            previewSize: 200,
            
            // Validation
            enableValidation: true,
            showWarnings: true,
            
            ...config
        };
        
        // State
        this.isVisible = false;
        this.currentFormat = 'svg';
        this.currentConfig = {};
        this.previewElement = null;
        
        // Export presets
        this.exportPresets = new Map();
        this.customTemplates = new Map();
        
        // Validation rules
        this.validationRules = new Map();
        
        this.initializePanel();
        this.setupExportPresets();
        this.setupValidationRules();
        this.setupEventListeners();
        
        console.log('🎛️ ExportConfigPanel initialized');
    }
    
    /**
     * Initialize the panel UI
     */
    initializePanel() {
        this.panelElement = document.createElement('div');
        this.panelElement.className = 'export-config-panel';
        this.panelElement.style.width = `${this.config.width}px`;
        
        this.panelElement.innerHTML = `
            <div class="export-config-panel__header">
                <div class="export-config-panel__title">
                    <span class="export-config-panel__icon">📸</span>
                    Export Configuration
                </div>
                <div class="export-config-panel__controls">
                    <button class="preset-btn" title="Load Preset">📋</button>
                    <button class="template-btn" title="Save Template">💾</button>
                    <button class="help-btn" title="Help">❓</button>
                    <button class="close-btn" title="Close">✕</button>
                </div>
            </div>
            
            <div class="export-config-panel__content">
                <!-- Format Selection -->
                <div class="config-section format-section">
                    <h3>Export Format</h3>
                    <div class="format-selector">
                        <label class="format-option">
                            <input type="radio" name="format" value="svg" checked>
                            <span class="format-label">
                                <strong>SVG</strong>
                                <small>Vector graphics, scalable</small>
                            </span>
                        </label>
                        <label class="format-option">
                            <input type="radio" name="format" value="png">
                            <span class="format-label">
                                <strong>PNG</strong>
                                <small>Raster image, high quality</small>
                            </span>
                        </label>
                        <label class="format-option">
                            <input type="radio" name="format" value="pdf">
                            <span class="format-label">
                                <strong>PDF</strong>
                                <small>Publication ready</small>
                            </span>
                        </label>
                    </div>
                </div>
                
                <!-- General Options -->
                <div class="config-section general-section">
                    <h3>General Settings</h3>
                    <div class="config-controls">
                        <div class="control-group">
                            <label for="filename">Filename</label>
                            <input type="text" id="filename" placeholder="groggy-graph" value="groggy-graph">
                        </div>
                        <div class="control-group">
                            <label for="title">Title</label>
                            <input type="text" id="title" placeholder="Graph Visualization">
                        </div>
                        <div class="control-group">
                            <label for="description">Description</label>
                            <textarea id="description" placeholder="Description of the graph..." rows="2"></textarea>
                        </div>
                        <div class="control-group checkbox-group">
                            <label>
                                <input type="checkbox" id="includeMetadata" checked>
                                Include metadata
                            </label>
                        </div>
                    </div>
                </div>
                
                <!-- Format-specific options will be inserted here -->
                <div class="config-section format-options">
                    <h3 id="format-options-title">SVG Options</h3>
                    <div id="format-options-content">
                        <!-- Content will be populated based on selected format -->
                    </div>
                </div>
                
                <!-- Style Options -->
                <div class="config-section style-section">
                    <h3>Style & Appearance</h3>
                    <div class="config-controls">
                        <div class="control-group">
                            <label for="backgroundColor">Background Color</label>
                            <div class="color-input-group">
                                <input type="color" id="backgroundColor" value="#ffffff">
                                <input type="text" id="backgroundColorText" value="#ffffff">
                                <button type="button" id="transparentBg">Transparent</button>
                            </div>
                        </div>
                        <div class="control-group checkbox-group">
                            <label>
                                <input type="checkbox" id="includeLabels" checked>
                                Include node labels
                            </label>
                        </div>
                        <div class="control-group">
                            <label for="fontFamily">Font Family</label>
                            <select id="fontFamily">
                                <option value="Arial, sans-serif">Arial</option>
                                <option value="Helvetica, sans-serif">Helvetica</option>
                                <option value="Times, serif">Times</option>
                                <option value="Georgia, serif">Georgia</option>
                                <option value="Monaco, monospace">Monaco</option>
                                <option value="Inter, sans-serif">Inter</option>
                            </select>
                        </div>
                        <div class="control-group">
                            <label for="fontSize">Font Size</label>
                            <div class="range-input-group">
                                <input type="range" id="fontSize" min="8" max="24" value="12">
                                <span class="range-value">12px</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Preview Section -->
                <div class="config-section preview-section" id="preview-section">
                    <h3>Preview</h3>
                    <div class="preview-container">
                        <div class="preview-placeholder" id="preview-placeholder">
                            <span>Preview will appear here</span>
                        </div>
                        <div class="preview-info" id="preview-info">
                            <div class="info-item">
                                <label>Estimated Size:</label>
                                <span id="estimated-size">--</span>
                            </div>
                            <div class="info-item">
                                <label>Dimensions:</label>
                                <span id="dimensions">--</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Export Actions -->
                <div class="config-section actions-section">
                    <div class="action-buttons">
                        <button class="action-btn preview-btn" id="preview-export">
                            👁️ Preview
                        </button>
                        <button class="action-btn export-btn" id="start-export">
                            📸 Export
                        </button>
                        <button class="action-btn batch-btn" id="batch-export">
                            📦 Batch Export
                        </button>
                    </div>
                    <div class="export-progress" id="export-progress" style="display: none;">
                        <div class="progress-bar">
                            <div class="progress-fill" id="progress-fill"></div>
                        </div>
                        <div class="progress-text" id="progress-text">Exporting...</div>
                    </div>
                </div>
                
                <!-- Validation Messages -->
                <div class="config-section validation-section" id="validation-section" style="display: none;">
                    <h3>⚠️ Validation Warnings</h3>
                    <div class="validation-messages" id="validation-messages">
                        <!-- Validation messages will be populated here -->
                    </div>
                </div>
            </div>
        `;
        
        this.container.appendChild(this.panelElement);
        this.addPanelStyles();
        
        // Get references to key elements
        this.formatOptionsContent = this.panelElement.querySelector('#format-options-content');
        this.formatOptionsTitle = this.panelElement.querySelector('#format-options-title');
        this.previewPlaceholder = this.panelElement.querySelector('#preview-placeholder');
        this.validationSection = this.panelElement.querySelector('#validation-section');
        this.validationMessages = this.panelElement.querySelector('#validation-messages');
        
        // Initialize with SVG options
        this.updateFormatOptions('svg');
    }
    
    /**
     * Update format-specific options
     */
    updateFormatOptions(format) {
        this.currentFormat = format;
        this.formatOptionsTitle.textContent = `${format.toUpperCase()} Options`;
        
        let optionsHTML = '';
        
        switch (format) {
            case 'svg':
                optionsHTML = this.createSVGOptions();
                break;
            case 'png':
                optionsHTML = this.createPNGOptions();
                break;
            case 'pdf':
                optionsHTML = this.createPDFOptions();
                break;
        }
        
        this.formatOptionsContent.innerHTML = optionsHTML;
        this.setupFormatOptionListeners();
    }
    
    /**
     * Create SVG-specific options
     */
    createSVGOptions() {
        return `
            <div class="config-controls">
                <div class="control-group checkbox-group">
                    <label>
                        <input type="checkbox" id="embedStyles" checked>
                        Embed CSS styles
                    </label>
                </div>
                <div class="control-group checkbox-group">
                    <label>
                        <input type="checkbox" id="embedFonts" checked>
                        Embed fonts
                    </label>
                </div>
                <div class="control-group checkbox-group">
                    <label>
                        <input type="checkbox" id="optimizeSize" checked>
                        Optimize for file size
                    </label>
                </div>
                <div class="control-group checkbox-group">
                    <label>
                        <input type="checkbox" id="preserveAspectRatio" checked>
                        Preserve aspect ratio
                    </label>
                </div>
                <div class="control-group">
                    <label for="precision">Numeric precision</label>
                    <div class="range-input-group">
                        <input type="range" id="precision" min="1" max="6" value="2">
                        <span class="range-value">2 decimals</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Create PNG-specific options
     */
    createPNGOptions() {
        return `
            <div class="config-controls">
                <div class="control-group">
                    <label for="dpi">DPI (Resolution)</label>
                    <div class="dpi-presets">
                        <button type="button" class="dpi-preset" data-dpi="96">Web (96)</button>
                        <button type="button" class="dpi-preset" data-dpi="150">Print (150)</button>
                        <button type="button" class="dpi-preset" data-dpi="300">High (300)</button>
                        <button type="button" class="dpi-preset" data-dpi="600">Ultra (600)</button>
                    </div>
                    <div class="range-input-group">
                        <input type="range" id="dpi" min="72" max="600" value="300" step="6">
                        <span class="range-value">300 DPI</span>
                    </div>
                </div>
                <div class="control-group">
                    <label for="compression">Quality</label>
                    <div class="range-input-group">
                        <input type="range" id="compression" min="0.1" max="1.0" value="0.9" step="0.1">
                        <span class="range-value">90%</span>
                    </div>
                </div>
                <div class="control-group">
                    <label for="maxWidth">Max Width (px)</label>
                    <input type="number" id="maxWidth" min="100" max="10000" value="2048" step="1">
                </div>
                <div class="control-group">
                    <label for="maxHeight">Max Height (px)</label>
                    <input type="number" id="maxHeight" min="100" max="10000" value="2048" step="1">
                </div>
                <div class="control-group checkbox-group">
                    <label>
                        <input type="checkbox" id="antialiasing" checked>
                        Anti-aliasing
                    </label>
                </div>
            </div>
        `;
    }
    
    /**
     * Create PDF-specific options
     */
    createPDFOptions() {
        return `
            <div class="config-controls">
                <div class="control-group">
                    <label for="pageFormat">Page Format</label>
                    <select id="pageFormat">
                        <option value="A4">A4 (210 × 297 mm)</option>
                        <option value="A3">A3 (297 × 420 mm)</option>
                        <option value="Letter">Letter (216 × 279 mm)</option>
                        <option value="Legal">Legal (216 × 356 mm)</option>
                        <option value="Tabloid">Tabloid (279 × 432 mm)</option>
                        <option value="Custom">Custom Size</option>
                    </select>
                </div>
                <div class="control-group" id="custom-size-group" style="display: none;">
                    <label>Custom Size (mm)</label>
                    <div class="size-inputs">
                        <input type="number" id="customWidth" placeholder="Width" min="50" max="1000" value="210">
                        <span>×</span>
                        <input type="number" id="customHeight" placeholder="Height" min="50" max="1000" value="297">
                    </div>
                </div>
                <div class="control-group">
                    <label for="orientation">Orientation</label>
                    <div class="radio-group">
                        <label>
                            <input type="radio" name="orientation" value="portrait" checked>
                            Portrait
                        </label>
                        <label>
                            <input type="radio" name="orientation" value="landscape">
                            Landscape
                        </label>
                    </div>
                </div>
                <div class="control-group">
                    <label for="margins">Margins (mm)</label>
                    <input type="number" id="margins" min="0" max="50" value="20" step="1">
                </div>
                <div class="control-group checkbox-group">
                    <label>
                        <input type="checkbox" id="fitToPage" checked>
                        Fit to page
                    </label>
                </div>
                <div class="control-group checkbox-group">
                    <label>
                        <input type="checkbox" id="centerOnPage" checked>
                        Center on page
                    </label>
                </div>
                <div class="control-group">
                    <label for="imageDPI">Image DPI</label>
                    <div class="range-input-group">
                        <input type="range" id="imageDPI" min="72" max="300" value="150">
                        <span class="range-value">150 DPI</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Setup event listeners for format-specific options
     */
    setupFormatOptionListeners() {
        // DPI presets for PNG
        if (this.currentFormat === 'png') {
            const dpiPresets = this.formatOptionsContent.querySelectorAll('.dpi-preset');
            dpiPresets.forEach(preset => {
                preset.addEventListener('click', (e) => {
                    const dpi = parseInt(e.target.getAttribute('data-dpi'));
                    document.getElementById('dpi').value = dpi;
                    document.querySelector('#dpi + .range-value').textContent = `${dpi} DPI`;
                    this.updatePreview();
                });
            });
            
            // Range value updates
            const dpiRange = document.getElementById('dpi');
            if (dpiRange) {
                dpiRange.addEventListener('input', (e) => {
                    document.querySelector('#dpi + .range-value').textContent = `${e.target.value} DPI`;
                    this.updatePreview();
                });
            }
            
            const compressionRange = document.getElementById('compression');
            if (compressionRange) {
                compressionRange.addEventListener('input', (e) => {
                    const percent = Math.round(e.target.value * 100);
                    document.querySelector('#compression + .range-value').textContent = `${percent}%`;
                    this.updatePreview();
                });
            }
        }
        
        // PDF page format handling
        if (this.currentFormat === 'pdf') {
            const pageFormat = document.getElementById('pageFormat');
            const customSizeGroup = document.getElementById('custom-size-group');
            
            if (pageFormat) {
                pageFormat.addEventListener('change', (e) => {
                    customSizeGroup.style.display = e.target.value === 'Custom' ? 'block' : 'none';
                    this.updatePreview();
                });
            }
            
            const imageDPIRange = document.getElementById('imageDPI');
            if (imageDPIRange) {
                imageDPIRange.addEventListener('input', (e) => {
                    document.querySelector('#imageDPI + .range-value').textContent = `${e.target.value} DPI`;
                    this.updatePreview();
                });
            }
        }
        
        // SVG precision handling
        if (this.currentFormat === 'svg') {
            const precisionRange = document.getElementById('precision');
            if (precisionRange) {
                precisionRange.addEventListener('input', (e) => {
                    document.querySelector('#precision + .range-value').textContent = `${e.target.value} decimals`;
                    this.updatePreview();
                });
            }
        }
        
        // Add change listeners to all inputs for live preview
        const inputs = this.formatOptionsContent.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('change', () => this.updatePreview());
        });
    }
    
    /**
     * Setup main event listeners
     */
    setupEventListeners() {
        // Format selection
        const formatInputs = this.panelElement.querySelectorAll('input[name="format"]');
        formatInputs.forEach(input => {
            input.addEventListener('change', (e) => {
                this.updateFormatOptions(e.target.value);
                this.updatePreview();
            });
        });
        
        // General options
        const generalInputs = this.panelElement.querySelectorAll('.general-section input, .general-section textarea');
        generalInputs.forEach(input => {
            input.addEventListener('input', () => this.updatePreview());
        });
        
        // Style options
        const styleInputs = this.panelElement.querySelectorAll('.style-section input, .style-section select');
        styleInputs.forEach(input => {
            input.addEventListener('input', () => this.updatePreview());
        });
        
        // Color input synchronization
        const colorInput = document.getElementById('backgroundColor');
        const colorTextInput = document.getElementById('backgroundColorText');
        const transparentBtn = document.getElementById('transparentBg');
        
        colorInput.addEventListener('input', (e) => {
            colorTextInput.value = e.target.value;
            this.updatePreview();
        });
        
        colorTextInput.addEventListener('input', (e) => {
            if (/^#[0-9A-F]{6}$/i.test(e.target.value)) {
                colorInput.value = e.target.value;
                this.updatePreview();
            }
        });
        
        transparentBtn.addEventListener('click', () => {
            colorInput.value = '#ffffff';
            colorTextInput.value = 'transparent';
            this.updatePreview();
        });
        
        // Font size range
        const fontSizeRange = document.getElementById('fontSize');
        fontSizeRange.addEventListener('input', (e) => {
            document.querySelector('#fontSize + .range-value').textContent = `${e.target.value}px`;
            this.updatePreview();
        });
        
        // Action buttons
        document.getElementById('preview-export').addEventListener('click', () => this.previewExport());
        document.getElementById('start-export').addEventListener('click', () => this.startExport());
        document.getElementById('batch-export').addEventListener('click', () => this.startBatchExport());
        
        // Header controls
        this.panelElement.querySelector('.preset-btn').addEventListener('click', () => this.showPresets());
        this.panelElement.querySelector('.template-btn').addEventListener('click', () => this.saveTemplate());
        this.panelElement.querySelector('.help-btn').addEventListener('click', () => this.showHelp());
        this.panelElement.querySelector('.close-btn').addEventListener('click', () => this.hide());
    }
    
    /**
     * Update preview based on current configuration
     */
    updatePreview() {
        if (!this.config.enablePreview) return;
        
        const config = this.getCurrentConfig();
        
        // Update estimated size and dimensions
        this.updatePreviewInfo(config);
        
        // Validate configuration
        if (this.config.enableValidation) {
            this.validateConfiguration(config);
        }
        
        // Generate preview
        this.generatePreview(config);
    }
    
    /**
     * Get current configuration from form
     */
    getCurrentConfig() {
        const config = {
            format: this.currentFormat,
            filename: document.getElementById('filename').value || 'groggy-graph',
            title: document.getElementById('title').value,
            description: document.getElementById('description').value,
            includeMetadata: document.getElementById('includeMetadata').checked,
            backgroundColor: document.getElementById('backgroundColorText').value,
            includeLabels: document.getElementById('includeLabels').checked,
            fontFamily: document.getElementById('fontFamily').value,
            fontSize: parseInt(document.getElementById('fontSize').value)
        };
        
        // Add format-specific options
        switch (this.currentFormat) {
            case 'svg':
                config.embedStyles = document.getElementById('embedStyles').checked;
                config.embedFonts = document.getElementById('embedFonts').checked;
                config.optimizeSize = document.getElementById('optimizeSize').checked;
                config.preserveAspectRatio = document.getElementById('preserveAspectRatio').checked;
                config.precision = parseInt(document.getElementById('precision').value);
                break;
                
            case 'png':
                config.dpi = parseInt(document.getElementById('dpi').value);
                config.compression = parseFloat(document.getElementById('compression').value);
                config.maxWidth = parseInt(document.getElementById('maxWidth').value);
                config.maxHeight = parseInt(document.getElementById('maxHeight').value);
                config.antialiasing = document.getElementById('antialiasing').checked;
                break;
                
            case 'pdf':
                config.pageFormat = document.getElementById('pageFormat').value;
                config.orientation = document.querySelector('input[name="orientation"]:checked').value;
                config.margins = parseInt(document.getElementById('margins').value);
                config.fitToPage = document.getElementById('fitToPage').checked;
                config.centerOnPage = document.getElementById('centerOnPage').checked;
                config.imageDPI = parseInt(document.getElementById('imageDPI').value);
                
                if (config.pageFormat === 'Custom') {
                    config.customWidth = parseInt(document.getElementById('customWidth').value);
                    config.customHeight = parseInt(document.getElementById('customHeight').value);
                }
                break;
        }
        
        return config;
    }
    
    /**
     * Update preview information
     */
    updatePreviewInfo(config) {
        const estimatedSizeEl = document.getElementById('estimated-size');
        const dimensionsEl = document.getElementById('dimensions');
        
        // Calculate estimated dimensions and file size
        const bounds = this.exportSystem.graphRenderer.getBounds();
        let width = bounds.width;
        let height = bounds.height;
        let estimatedSize = 'Unknown';
        
        switch (config.format) {
            case 'svg':
                estimatedSize = this.estimateSVGSize(width, height, config);
                break;
            case 'png':
                const scale = config.dpi / 96;
                width = Math.round(width * scale);
                height = Math.round(height * scale);
                estimatedSize = this.estimatePNGSize(width, height, config);
                break;
            case 'pdf':
                estimatedSize = this.estimatePDFSize(config);
                break;
        }
        
        estimatedSizeEl.textContent = estimatedSize;
        dimensionsEl.textContent = `${width} × ${height}`;
    }
    
    /**
     * Estimate SVG file size
     */
    estimateSVGSize(width, height, config) {
        const nodeCount = this.exportSystem.graphRenderer.getNodes().length;
        const edgeCount = this.exportSystem.graphRenderer.getEdges().length;
        
        // Rough estimation based on element count and options
        let baseSize = (nodeCount * 150) + (edgeCount * 100); // bytes per element
        
        if (config.embedStyles) baseSize += 2000;
        if (config.embedFonts) baseSize += 10000;
        if (config.includeMetadata) baseSize += 1000;
        
        return this.formatFileSize(baseSize);
    }
    
    /**
     * Estimate PNG file size
     */
    estimatePNGSize(width, height, config) {
        // PNG size depends on image complexity and compression
        const pixels = width * height;
        const bytesPerPixel = 3; // RGB
        const compressionRatio = config.compression * 0.3; // Rough estimate
        
        const estimatedBytes = pixels * bytesPerPixel * compressionRatio;
        return this.formatFileSize(estimatedBytes);
    }
    
    /**
     * Estimate PDF file size
     */
    estimatePDFSize(config) {
        // PDF includes compressed image + metadata
        const imageSize = 200000; // Rough estimate for graph image
        const overhead = 50000; // PDF structure overhead
        
        return this.formatFileSize(imageSize + overhead);
    }
    
    /**
     * Format file size for display
     */
    formatFileSize(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }
    
    /**
     * Validate configuration
     */
    validateConfiguration(config) {
        const warnings = [];
        const errors = [];
        
        // Get validation rules for current format
        const rules = this.validationRules.get(config.format) || [];
        
        rules.forEach(rule => {
            const result = rule.validate(config);
            if (!result.valid) {
                if (result.severity === 'error') {
                    errors.push(result.message);
                } else {
                    warnings.push(result.message);
                }
            }
        });
        
        this.displayValidationMessages(warnings, errors);
    }
    
    /**
     * Display validation messages
     */
    displayValidationMessages(warnings, errors) {
        const hasMessages = warnings.length > 0 || errors.length > 0;
        
        if (hasMessages) {
            this.validationSection.style.display = 'block';
            
            let messagesHTML = '';
            
            errors.forEach(error => {
                messagesHTML += `<div class="validation-message error">❌ ${error}</div>`;
            });
            
            warnings.forEach(warning => {
                messagesHTML += `<div class="validation-message warning">⚠️ ${warning}</div>`;
            });
            
            this.validationMessages.innerHTML = messagesHTML;
        } else {
            this.validationSection.style.display = 'none';
        }
    }
    
    /**
     * Generate preview
     */
    generatePreview(config) {
        // This would generate a small preview of the export
        this.previewPlaceholder.innerHTML = `
            <div class="preview-loading">
                Generating preview...
            </div>
        `;
        
        // Simulate preview generation
        setTimeout(() => {
            this.previewPlaceholder.innerHTML = `
                <div class="preview-content">
                    <div class="preview-format-badge">${config.format.toUpperCase()}</div>
                    <div class="preview-graph">
                        <!-- Simplified graph preview would go here -->
                        <svg width="150" height="100" viewBox="0 0 150 100">
                            <circle cx="30" cy="30" r="8" fill="#4CAF50" stroke="#2E7D32" stroke-width="2"/>
                            <circle cx="120" cy="30" r="8" fill="#4CAF50" stroke="#2E7D32" stroke-width="2"/>
                            <circle cx="75" cy="70" r="8" fill="#4CAF50" stroke="#2E7D32" stroke-width="2"/>
                            <line x1="30" y1="30" x2="120" y2="30" stroke="#757575" stroke-width="1"/>
                            <line x1="30" y1="30" x2="75" y2="70" stroke="#757575" stroke-width="1"/>
                            <line x1="120" y1="30" x2="75" y2="70" stroke="#757575" stroke-width="1"/>
                        </svg>
                    </div>
                </div>
            `;
        }, 500);
    }
    
    /**
     * Preview export
     */
    async previewExport() {
        const config = this.getCurrentConfig();
        config.autoDownload = false;
        
        try {
            const result = await this.exportSystem.exportGraph(config.format, config);
            console.log('📸 Preview generated:', result);
            
            // Show preview in a modal or new window
            this.showExportPreview(result);
            
        } catch (error) {
            console.error('Preview failed:', error);
            this.showError(`Preview failed: ${error.message}`);
        }
    }
    
    /**
     * Start export
     */
    async startExport() {
        const config = this.getCurrentConfig();
        
        this.showProgress(true);
        
        try {
            const result = await this.exportSystem.exportGraph(config.format, config);
            console.log('📸 Export completed:', result);
            
            this.showProgress(false);
            this.showSuccess(`Export completed: ${result.filename}`);
            
        } catch (error) {
            console.error('Export failed:', error);
            this.showProgress(false);
            this.showError(`Export failed: ${error.message}`);
        }
    }
    
    /**
     * Start batch export
     */
    async startBatchExport() {
        // This would open a batch export dialog
        console.log('📸 Starting batch export...');
        
        // For now, export in all formats
        const config = this.getCurrentConfig();
        const formats = ['svg', 'png', 'pdf'];
        
        this.showProgress(true, 'Batch exporting...');
        
        for (let i = 0; i < formats.length; i++) {
            const format = formats[i];
            const formatConfig = { ...config, format };
            
            try {
                await this.exportSystem.exportGraph(format, formatConfig);
                this.updateProgress(((i + 1) / formats.length) * 100, `Exported ${format.toUpperCase()}`);
            } catch (error) {
                console.error(`Batch export failed for ${format}:`, error);
            }
        }
        
        this.showProgress(false);
        this.showSuccess('Batch export completed');
    }
    
    /**
     * Setup export presets
     */
    setupExportPresets() {
        this.exportPresets.set('web-svg', {
            name: 'Web SVG',
            description: 'Optimized SVG for web use',
            config: {
                format: 'svg',
                embedStyles: true,
                embedFonts: false,
                optimizeSize: true,
                backgroundColor: 'transparent'
            }
        });
        
        this.exportPresets.set('print-png', {
            name: 'Print PNG',
            description: 'High-quality PNG for printing',
            config: {
                format: 'png',
                dpi: 300,
                compression: 1.0,
                backgroundColor: '#ffffff'
            }
        });
        
        this.exportPresets.set('publication-pdf', {
            name: 'Publication PDF',
            description: 'PDF ready for academic publication',
            config: {
                format: 'pdf',
                pageFormat: 'A4',
                orientation: 'landscape',
                margins: 20,
                imageDPI: 300
            }
        });
    }
    
    /**
     * Setup validation rules
     */
    setupValidationRules() {
        // SVG validation rules
        this.validationRules.set('svg', [
            {
                validate: (config) => ({
                    valid: config.precision >= 1 && config.precision <= 6,
                    severity: 'warning',
                    message: 'Precision should be between 1-6 decimals'
                })
            }
        ]);
        
        // PNG validation rules
        this.validationRules.set('png', [
            {
                validate: (config) => ({
                    valid: config.dpi <= 600,
                    severity: 'warning',
                    message: 'Very high DPI may result in large file sizes'
                })
            },
            {
                validate: (config) => ({
                    valid: config.maxWidth * config.maxHeight < 100000000,
                    severity: 'error',
                    message: 'Image dimensions too large (>100MP)'
                })
            }
        ]);
        
        // PDF validation rules
        this.validationRules.set('pdf', [
            {
                validate: (config) => ({
                    valid: config.margins >= 0 && config.margins <= 50,
                    severity: 'warning',
                    message: 'Margins should be between 0-50mm'
                })
            }
        ]);
    }
    
    /**
     * Show/hide progress
     */
    showProgress(show, text = 'Exporting...') {
        const progressElement = document.getElementById('export-progress');
        const progressText = document.getElementById('progress-text');
        
        if (show) {
            progressElement.style.display = 'block';
            progressText.textContent = text;
            this.updateProgress(0);
        } else {
            progressElement.style.display = 'none';
        }
    }
    
    /**
     * Update progress
     */
    updateProgress(percent, text) {
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        progressFill.style.width = `${percent}%`;
        if (text) progressText.textContent = text;
    }
    
    /**
     * Show success message
     */
    showSuccess(message) {
        // This would show a success notification
        console.log('✅ Success:', message);
    }
    
    /**
     * Show error message
     */
    showError(message) {
        // This would show an error notification
        console.error('❌ Error:', message);
    }
    
    /**
     * Show/hide panel
     */
    show() {
        this.isVisible = true;
        this.panelElement.style.display = 'block';
        this.updatePreview();
    }
    
    hide() {
        this.isVisible = false;
        this.panelElement.style.display = 'none';
    }
    
    /**
     * Add CSS styles
     */
    addPanelStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .export-config-panel {
                background: #2a2a2a;
                border: 1px solid #444;
                border-radius: 8px;
                font-family: 'Inter', sans-serif;
                font-size: 13px;
                color: #ddd;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                overflow: hidden;
            }
            
            .export-config-panel__header {
                background: #333;
                padding: 12px 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #444;
            }
            
            .export-config-panel__title {
                color: #fff;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .export-config-panel__controls {
                display: flex;
                gap: 4px;
            }
            
            .export-config-panel__controls button {
                background: #555;
                border: none;
                color: #fff;
                padding: 6px 8px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            }
            
            .export-config-panel__controls button:hover {
                background: #666;
            }
            
            .export-config-panel__content {
                max-height: 80vh;
                overflow-y: auto;
                padding: 0;
            }
            
            .config-section {
                padding: 16px;
                border-bottom: 1px solid #333;
            }
            
            .config-section:last-child {
                border-bottom: none;
            }
            
            .config-section h3 {
                margin: 0 0 12px 0;
                color: #fff;
                font-size: 14px;
                font-weight: 600;
            }
            
            .format-selector {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .format-option {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 12px;
                background: #333;
                border-radius: 6px;
                cursor: pointer;
                transition: background 0.2s;
            }
            
            .format-option:hover {
                background: #3a3a3a;
            }
            
            .format-option input[type="radio"] {
                margin: 0;
            }
            
            .format-label strong {
                color: #fff;
                display: block;
            }
            
            .format-label small {
                color: #aaa;
                font-size: 11px;
            }
            
            .config-controls {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            
            .control-group {
                display: flex;
                flex-direction: column;
                gap: 4px;
            }
            
            .control-group label {
                color: #fff;
                font-weight: 500;
                font-size: 12px;
            }
            
            .control-group input,
            .control-group select,
            .control-group textarea {
                padding: 8px;
                background: #444;
                border: 1px solid #555;
                color: #fff;
                border-radius: 4px;
                font-size: 12px;
            }
            
            .control-group input:focus,
            .control-group select:focus,
            .control-group textarea:focus {
                outline: none;
                border-color: #4CAF50;
            }
            
            .checkbox-group {
                flex-direction: row;
                align-items: center;
                gap: 8px;
            }
            
            .checkbox-group label {
                display: flex;
                align-items: center;
                gap: 6px;
                cursor: pointer;
            }
            
            .radio-group {
                display: flex;
                gap: 16px;
            }
            
            .radio-group label {
                display: flex;
                align-items: center;
                gap: 6px;
                cursor: pointer;
            }
            
            .range-input-group {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .range-input-group input[type="range"] {
                flex: 1;
            }
            
            .range-value {
                min-width: 60px;
                text-align: right;
                font-size: 11px;
                color: #aaa;
            }
            
            .color-input-group {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .color-input-group input[type="color"] {
                width: 40px;
                height: 32px;
                padding: 0;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            
            .color-input-group input[type="text"] {
                flex: 1;
            }
            
            .color-input-group button {
                padding: 6px 12px;
                background: #555;
                border: none;
                color: #fff;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
            }
            
            .dpi-presets {
                display: flex;
                gap: 4px;
                margin-bottom: 8px;
            }
            
            .dpi-preset {
                padding: 4px 8px;
                background: #444;
                border: 1px solid #555;
                color: #fff;
                border-radius: 4px;
                cursor: pointer;
                font-size: 10px;
            }
            
            .dpi-preset:hover {
                background: #555;
            }
            
            .size-inputs {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .size-inputs input {
                width: 80px;
            }
            
            .preview-container {
                background: #333;
                border-radius: 6px;
                padding: 16px;
            }
            
            .preview-placeholder {
                background: #2a2a2a;
                border: 2px dashed #555;
                border-radius: 4px;
                height: 120px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 12px;
                color: #aaa;
            }
            
            .preview-content {
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100%;
            }
            
            .preview-format-badge {
                position: absolute;
                top: 8px;
                right: 8px;
                background: #4CAF50;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            
            .preview-info {
                display: flex;
                justify-content: space-between;
                font-size: 11px;
            }
            
            .info-item {
                display: flex;
                flex-direction: column;
                gap: 2px;
            }
            
            .info-item label {
                color: #aaa;
                font-weight: normal;
            }
            
            .info-item span {
                color: #fff;
                font-weight: 500;
            }
            
            .action-buttons {
                display: flex;
                gap: 8px;
                margin-bottom: 12px;
            }
            
            .action-btn {
                flex: 1;
                padding: 10px 16px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 500;
                font-size: 12px;
                transition: all 0.2s;
            }
            
            .preview-btn {
                background: #2196F3;
                color: white;
            }
            
            .export-btn {
                background: #4CAF50;
                color: white;
            }
            
            .batch-btn {
                background: #FF9800;
                color: white;
            }
            
            .action-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }
            
            .export-progress {
                margin-top: 12px;
            }
            
            .progress-bar {
                background: #444;
                border-radius: 4px;
                height: 8px;
                overflow: hidden;
                margin-bottom: 8px;
            }
            
            .progress-fill {
                background: #4CAF50;
                height: 100%;
                width: 0%;
                transition: width 0.3s ease;
            }
            
            .progress-text {
                font-size: 11px;
                color: #aaa;
                text-align: center;
            }
            
            .validation-messages {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .validation-message {
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            
            .validation-message.warning {
                background: rgba(255, 152, 0, 0.1);
                border: 1px solid rgba(255, 152, 0, 0.3);
                color: #FFB74D;
            }
            
            .validation-message.error {
                background: rgba(244, 67, 54, 0.1);
                border: 1px solid rgba(244, 67, 54, 0.3);
                color: #EF5350;
            }
        `;
        
        document.head.appendChild(style);
    }
}

// Global instance
window.GroggyExportConfigPanel = null;

/**
 * Initialize export configuration panel
 */
function initializeExportConfigPanel(container, exportSystem, config = {}) {
    if (window.GroggyExportConfigPanel) {
        console.warn('Export config panel already initialized');
        return window.GroggyExportConfigPanel;
    }
    
    window.GroggyExportConfigPanel = new ExportConfigPanel(container, exportSystem, config);
    
    console.log('🎛️ Export configuration panel initialized');
    
    return window.GroggyExportConfigPanel;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ExportConfigPanel,
        initializeExportConfigPanel
    };
}