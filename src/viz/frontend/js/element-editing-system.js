/**
 * Element Editing System
 * Provides comprehensive editing capabilities for nodes and edges including
 * inline editing, property panels, batch editing, and validation.
 */

class ElementEditingSystem {
    constructor(graphRenderer, config = {}) {
        this.graphRenderer = graphRenderer;
        this.canvas = graphRenderer.canvas;
        
        // Configuration with intelligent defaults
        this.config = {
            enableInlineEditing: true,
            enablePropertyPanel: true,
            enableBatchEditing: true,
            enableValidation: true,
            
            // Editing modes
            defaultEditMode: 'inline', // 'inline', 'panel', 'modal'
            autoSave: true,
            autoSaveDelay: 1000,
            
            // Inline editing
            inlineEditing: {
                enabled: true,
                editOnDoubleClick: true,
                editOnF2: true,
                editableProperties: ['label', 'name', 'title'],
                fontSize: '14px',
                fontFamily: 'Arial, sans-serif',
                padding: 4,
                minWidth: 100,
                maxWidth: 300
            },
            
            // Property panel
            propertyPanel: {
                position: 'right', // 'left', 'right', 'bottom', 'floating'
                width: 300,
                height: 400,
                resizable: true,
                collapsible: true,
                autoHide: false
            },
            
            // Validation
            validation: {
                required: ['id'],
                patterns: {
                    email: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
                    url: /^https?:\/\/.+/,
                    number: /^-?\d*\.?\d+$/
                },
                customValidators: new Map()
            },
            
            // Styling
            theme: {
                backgroundColor: '#ffffff',
                borderColor: '#cccccc',
                focusColor: '#007bff',
                errorColor: '#dc3545',
                successColor: '#28a745',
                textColor: '#333333',
                labelColor: '#666666'
            },
            
            ...config
        };
        
        // Editing state
        this.editingState = {
            isEditing: false,
            editingElement: null,
            editingProperty: null,
            originalValue: null,
            currentEditor: null,
            hasChanges: false,
            
            // Batch editing
            batchEditMode: false,
            selectedElements: new Set(),
            
            // Validation
            validationErrors: new Map(),
            validationWarnings: new Map()
        };
        
        // UI components
        this.components = {
            inlineEditor: null,
            propertyPanel: null,
            batchEditor: null,
            validationTooltip: null
        };
        
        // Auto-save timer
        this.autoSaveTimer = null;
        
        this.initializeEditingSystem();
    }
    
    /**
     * Initialize the editing system
     */
    initializeEditingSystem() {
        this.createInlineEditor();
        this.createPropertyPanel();
        this.createBatchEditor();
        this.createValidationTooltip();
        this.bindEventListeners();
        this.setupEditingShortcuts();
    }
    
    /**
     * Create inline editor component
     */
    createInlineEditor() {
        this.components.inlineEditor = document.createElement('input');
        this.components.inlineEditor.className = 'groggy-inline-editor';
        this.components.inlineEditor.style.cssText = `
            position: absolute;
            z-index: 10001;
            border: 2px solid ${this.config.theme.focusColor};
            border-radius: 4px;
            padding: ${this.config.inlineEditing.padding}px;
            font-size: ${this.config.inlineEditing.fontSize};
            font-family: ${this.config.inlineEditing.fontFamily};
            background: ${this.config.theme.backgroundColor};
            color: ${this.config.theme.textColor};
            outline: none;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            display: none;
            min-width: ${this.config.inlineEditing.minWidth}px;
            max-width: ${this.config.inlineEditing.maxWidth}px;
        `;
        
        document.body.appendChild(this.components.inlineEditor);
        
        // Inline editor event handlers
        this.components.inlineEditor.addEventListener('blur', () => this.commitInlineEdit());
        this.components.inlineEditor.addEventListener('keydown', (e) => this.handleInlineEditorKeydown(e));
        this.components.inlineEditor.addEventListener('input', () => this.handleInlineEditorInput());
    }
    
    /**
     * Create property panel
     */
    createPropertyPanel() {
        this.components.propertyPanel = document.createElement('div');
        this.components.propertyPanel.className = 'groggy-property-panel';
        this.components.propertyPanel.style.cssText = `
            position: fixed;
            top: 0;
            right: -${this.config.propertyPanel.width}px;
            width: ${this.config.propertyPanel.width}px;
            height: 100vh;
            background: ${this.config.theme.backgroundColor};
            border-left: 1px solid ${this.config.theme.borderColor};
            box-shadow: -2px 0 8px rgba(0, 0, 0, 0.1);
            z-index: 9999;
            transition: right 0.3s ease;
            overflow-y: auto;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
        `;
        
        this.components.propertyPanel.innerHTML = `
            <div class="property-panel-header">
                <h3>Properties</h3>
                <button class="close-btn">&times;</button>
            </div>
            <div class="property-panel-content">
                <div class="property-form"></div>
            </div>
        `;
        
        this.addPropertyPanelStyles();
        document.body.appendChild(this.components.propertyPanel);
        
        // Property panel event handlers
        const closeBtn = this.components.propertyPanel.querySelector('.close-btn');
        closeBtn.addEventListener('click', () => this.hidePropertyPanel());
    }
    
    /**
     * Add CSS styles for property panel
     */
    addPropertyPanelStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .groggy-property-panel .property-panel-header {
                padding: 16px;
                border-bottom: 1px solid ${this.config.theme.borderColor};
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: #f8f9fa;
            }
            
            .groggy-property-panel h3 {
                margin: 0;
                color: ${this.config.theme.textColor};
                font-size: 16px;
                font-weight: 600;
            }
            
            .groggy-property-panel .close-btn {
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                color: ${this.config.theme.labelColor};
                padding: 0;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .groggy-property-panel .close-btn:hover {
                color: ${this.config.theme.textColor};
            }
            
            .groggy-property-panel .property-panel-content {
                padding: 16px;
            }
            
            .property-field {
                margin-bottom: 16px;
            }
            
            .property-field label {
                display: block;
                margin-bottom: 4px;
                font-weight: 500;
                color: ${this.config.theme.labelColor};
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .property-field input,
            .property-field textarea,
            .property-field select {
                width: 100%;
                padding: 8px 12px;
                border: 1px solid ${this.config.theme.borderColor};
                border-radius: 4px;
                font-size: 14px;
                color: ${this.config.theme.textColor};
                background: ${this.config.theme.backgroundColor};
                box-sizing: border-box;
            }
            
            .property-field input:focus,
            .property-field textarea:focus,
            .property-field select:focus {
                outline: none;
                border-color: ${this.config.theme.focusColor};
                box-shadow: 0 0 0 2px ${this.config.theme.focusColor}33;
            }
            
            .property-field .error {
                border-color: ${this.config.theme.errorColor};
            }
            
            .property-field .error-message {
                color: ${this.config.theme.errorColor};
                font-size: 12px;
                margin-top: 4px;
            }
            
            .property-field .warning-message {
                color: #ffc107;
                font-size: 12px;
                margin-top: 4px;
            }
            
            .property-field-group {
                border: 1px solid ${this.config.theme.borderColor};
                border-radius: 4px;
                padding: 12px;
                margin-bottom: 16px;
            }
            
            .property-field-group-title {
                font-weight: 600;
                margin-bottom: 12px;
                color: ${this.config.theme.textColor};
            }
            
            .color-picker-container {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .color-picker {
                width: 40px;
                height: 32px;
                border: 1px solid ${this.config.theme.borderColor};
                border-radius: 4px;
                cursor: pointer;
            }
            
            .property-actions {
                margin-top: 20px;
                padding-top: 16px;
                border-top: 1px solid ${this.config.theme.borderColor};
                display: flex;
                gap: 8px;
            }
            
            .property-actions button {
                flex: 1;
                padding: 8px 16px;
                border: 1px solid ${this.config.theme.borderColor};
                border-radius: 4px;
                background: ${this.config.theme.backgroundColor};
                color: ${this.config.theme.textColor};
                cursor: pointer;
                font-size: 14px;
            }
            
            .property-actions button.primary {
                background: ${this.config.theme.focusColor};
                border-color: ${this.config.theme.focusColor};
                color: white;
            }
            
            .property-actions button:hover {
                opacity: 0.8;
            }
        `;
        document.head.appendChild(style);
    }
    
    /**
     * Create batch editor
     */
    createBatchEditor() {
        this.components.batchEditor = document.createElement('div');
        this.components.batchEditor.className = 'groggy-batch-editor';
        this.components.batchEditor.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: ${this.config.theme.backgroundColor};
            border: 1px solid ${this.config.theme.borderColor};
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            padding: 16px;
            min-width: 300px;
            z-index: 10000;
            display: none;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        `;
        
        document.body.appendChild(this.components.batchEditor);
    }
    
    /**
     * Create validation tooltip
     */
    createValidationTooltip() {
        this.components.validationTooltip = document.createElement('div');
        this.components.validationTooltip.className = 'groggy-validation-tooltip';
        this.components.validationTooltip.style.cssText = `
            position: absolute;
            background: ${this.config.theme.errorColor};
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 10002;
            display: none;
            max-width: 200px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        `;
        
        document.body.appendChild(this.components.validationTooltip);
    }
    
    /**
     * Bind event listeners
     */
    bindEventListeners() {
        // Double-click for inline editing
        if (this.config.inlineEditing.editOnDoubleClick) {
            this.canvas.addEventListener('dblclick', this.handleCanvasDoubleClick.bind(this));
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeydown.bind(this));
        
        // Context menu integration
        this.canvas.addEventListener('contextMenuNodeEdit', (e) => {
            this.editElement(e.detail.node, 'panel');
        });
        
        this.canvas.addEventListener('contextMenuEdgeEdit', (e) => {
            this.editElement(e.detail.edge, 'panel');
        });
        
        // Selection changes
        this.canvas.addEventListener('selectionChanged', (e) => {
            this.handleSelectionChange(e.detail);
        });
    }
    
    /**
     * Setup editing keyboard shortcuts
     */
    setupEditingShortcuts() {
        this.shortcuts = {
            'F2': () => this.startInlineEdit(),
            'Enter': () => this.commitCurrentEdit(),
            'Escape': () => this.cancelCurrentEdit(),
            'Delete': () => this.deleteSelected(),
            'Backspace': () => this.deleteSelected()
        };
    }
    
    /**
     * Event handlers
     */
    handleCanvasDoubleClick(event) {
        const mousePos = this.getMousePosition(event);
        const element = this.getElementAtPosition(mousePos.x, mousePos.y);
        
        if (element) {
            this.editElement(element, 'inline');
        }
    }
    
    handleKeydown(event) {
        const shortcut = this.shortcuts[event.key];
        if (shortcut && !this.isEditingActive()) {
            shortcut();
            event.preventDefault();
        }
    }
    
    handleInlineEditorKeydown(event) {
        switch (event.key) {
            case 'Enter':
                this.commitInlineEdit();
                event.preventDefault();
                break;
            case 'Escape':
                this.cancelInlineEdit();
                event.preventDefault();
                break;
            case 'Tab':
                this.commitInlineEdit();
                this.editNextProperty();
                event.preventDefault();
                break;
        }
    }
    
    handleInlineEditorInput() {
        this.editingState.hasChanges = true;
        
        if (this.config.enableValidation) {
            this.validateCurrentEdit();
        }
        
        if (this.config.autoSave) {
            this.scheduleAutoSave();
        }
    }
    
    handleSelectionChange(detail) {
        const selectedElements = detail.selectedElements || [];
        
        if (selectedElements.length === 1) {
            // Single selection - show in property panel
            if (this.config.enablePropertyPanel && !this.config.propertyPanel.autoHide) {
                this.showPropertyPanel(selectedElements[0]);
            }
        } else if (selectedElements.length > 1) {
            // Multiple selection - enable batch editing
            if (this.config.enableBatchEditing) {
                this.enableBatchEditing(selectedElements);
            }
        } else {
            // No selection - hide panels
            this.hidePropertyPanel();
            this.disableBatchEditing();
        }
    }
    
    /**
     * Core editing methods
     */
    editElement(element, mode = null) {
        if (this.editingState.isEditing) {
            this.commitCurrentEdit();
        }
        
        const editMode = mode || this.config.defaultEditMode;
        
        switch (editMode) {
            case 'inline':
                this.startInlineEdit(element);
                break;
            case 'panel':
                this.showPropertyPanel(element);
                break;
            case 'modal':
                this.showEditModal(element);
                break;
        }
        
        this.emitEvent('editStart', { element, mode: editMode });
    }
    
    startInlineEdit(element = null, property = null) {
        if (!this.config.inlineEditing.enabled) return;
        
        // Use selected element if none specified
        if (!element) {
            const selected = this.getSelectedElements();
            if (selected.length !== 1) return;
            element = selected[0];
        }
        
        // Determine editable property
        const editableProperty = property || this.getEditableProperty(element);
        if (!editableProperty) return;
        
        // Setup editing state
        this.editingState.isEditing = true;
        this.editingState.editingElement = element;
        this.editingState.editingProperty = editableProperty;
        this.editingState.originalValue = element[editableProperty];
        this.editingState.currentEditor = 'inline';
        
        // Position and show inline editor
        this.positionInlineEditor(element);
        this.components.inlineEditor.value = element[editableProperty] || '';
        this.components.inlineEditor.style.display = 'block';
        this.components.inlineEditor.focus();
        this.components.inlineEditor.select();
        
        this.emitEvent('inlineEditStart', {
            element,
            property: editableProperty,
            value: element[editableProperty]
        });
    }
    
    commitInlineEdit() {
        if (!this.editingState.isEditing || this.editingState.currentEditor !== 'inline') return;
        
        const newValue = this.components.inlineEditor.value;
        const element = this.editingState.editingElement;
        const property = this.editingState.editingProperty;
        
        // Validate the new value
        const validation = this.validateValue(newValue, property, element);
        if (!validation.isValid) {
            this.showValidationError(validation.errors[0]);
            return false;
        }
        
        // Apply the change
        this.applyElementChange(element, property, newValue);
        
        this.finishInlineEdit();
        return true;
    }
    
    cancelInlineEdit() {
        if (!this.editingState.isEditing || this.editingState.currentEditor !== 'inline') return;
        
        this.finishInlineEdit();
        
        this.emitEvent('inlineEditCancel', {
            element: this.editingState.editingElement,
            property: this.editingState.editingProperty
        });
    }
    
    finishInlineEdit() {
        this.components.inlineEditor.style.display = 'none';
        this.hideValidationError();
        
        this.editingState.isEditing = false;
        this.editingState.editingElement = null;
        this.editingState.editingProperty = null;
        this.editingState.originalValue = null;
        this.editingState.currentEditor = null;
        this.editingState.hasChanges = false;
        
        this.clearAutoSave();
    }
    
    /**
     * Property panel methods
     */
    showPropertyPanel(element) {
        this.components.propertyPanel.style.right = '0px';
        this.populatePropertyPanel(element);
        
        this.emitEvent('propertyPanelShow', { element });
    }
    
    hidePropertyPanel() {
        this.components.propertyPanel.style.right = `-${this.config.propertyPanel.width}px`;
        
        this.emitEvent('propertyPanelHide', {});
    }
    
    populatePropertyPanel(element) {
        const form = this.components.propertyPanel.querySelector('.property-form');
        form.innerHTML = '';
        
        // Get element schema
        const schema = this.getElementSchema(element);
        
        // Create form fields
        schema.properties.forEach(prop => {
            const field = this.createPropertyField(prop, element[prop.name]);
            form.appendChild(field);
        });
        
        // Add action buttons
        const actions = this.createPropertyActions();
        form.appendChild(actions);
    }
    
    createPropertyField(property, value) {
        const field = document.createElement('div');
        field.className = 'property-field';
        
        const label = document.createElement('label');
        label.textContent = property.label || property.name;
        field.appendChild(label);
        
        let input;
        
        switch (property.type) {
            case 'text':
            case 'string':
                input = document.createElement('input');
                input.type = 'text';
                input.value = value || '';
                break;
                
            case 'number':
                input = document.createElement('input');
                input.type = 'number';
                input.value = value || '';
                if (property.min !== undefined) input.min = property.min;
                if (property.max !== undefined) input.max = property.max;
                if (property.step !== undefined) input.step = property.step;
                break;
                
            case 'textarea':
                input = document.createElement('textarea');
                input.value = value || '';
                input.rows = property.rows || 3;
                break;
                
            case 'select':
                input = document.createElement('select');
                property.options.forEach(option => {
                    const optionEl = document.createElement('option');
                    optionEl.value = option.value;
                    optionEl.textContent = option.label;
                    optionEl.selected = option.value === value;
                    input.appendChild(optionEl);
                });
                break;
                
            case 'color':
                const colorContainer = document.createElement('div');
                colorContainer.className = 'color-picker-container';
                
                input = document.createElement('input');
                input.type = 'color';
                input.value = value || '#000000';
                input.className = 'color-picker';
                
                const colorText = document.createElement('input');
                colorText.type = 'text';
                colorText.value = value || '#000000';
                colorText.style.flex = '1';
                
                input.addEventListener('change', () => {
                    colorText.value = input.value;
                });
                
                colorText.addEventListener('change', () => {
                    if (this.isValidColor(colorText.value)) {
                        input.value = colorText.value;
                    }
                });
                
                colorContainer.appendChild(input);
                colorContainer.appendChild(colorText);
                field.appendChild(colorContainer);
                
                input.dataset.propertyName = property.name;
                colorText.dataset.propertyName = property.name;
                return field;
                
            case 'boolean':
                input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = !!value;
                break;
                
            default:
                input = document.createElement('input');
                input.type = 'text';
                input.value = value || '';
        }
        
        input.dataset.propertyName = property.name;
        input.addEventListener('change', () => this.handlePropertyChange(property, input));
        input.addEventListener('input', () => this.handlePropertyInput(property, input));
        
        field.appendChild(input);
        
        // Add validation messages container
        const messagesContainer = document.createElement('div');
        messagesContainer.className = 'property-messages';
        field.appendChild(messagesContainer);
        
        return field;
    }
    
    createPropertyActions() {
        const actions = document.createElement('div');
        actions.className = 'property-actions';
        
        const saveBtn = document.createElement('button');
        saveBtn.textContent = 'Save Changes';
        saveBtn.className = 'primary';
        saveBtn.addEventListener('click', () => this.savePropertyChanges());
        
        const resetBtn = document.createElement('button');
        resetBtn.textContent = 'Reset';
        resetBtn.addEventListener('click', () => this.resetPropertyChanges());
        
        actions.appendChild(saveBtn);
        actions.appendChild(resetBtn);
        
        return actions;
    }
    
    handlePropertyChange(property, input) {
        const element = this.getCurrentEditingElement();
        if (!element) return;
        
        let value = this.getInputValue(input);
        
        // Validate the value
        const validation = this.validateValue(value, property.name, element);
        this.updateFieldValidation(input, validation);
        
        if (validation.isValid) {
            // Apply change immediately if auto-save is enabled
            if (this.config.autoSave) {
                this.applyElementChange(element, property.name, value);
                this.scheduleAutoSave();
            } else {
                // Store pending change
                this.storePendingChange(element, property.name, value);
            }
        }
    }
    
    handlePropertyInput(property, input) {
        // Real-time validation feedback
        if (this.config.enableValidation) {
            const element = this.getCurrentEditingElement();
            if (element) {
                const value = this.getInputValue(input);
                const validation = this.validateValue(value, property.name, element);
                this.updateFieldValidation(input, validation);
            }
        }
    }
    
    /**
     * Batch editing methods
     */
    enableBatchEditing(elements) {
        this.editingState.batchEditMode = true;
        this.editingState.selectedElements = new Set(elements);
        
        this.showBatchEditor(elements);
    }
    
    disableBatchEditing() {
        this.editingState.batchEditMode = false;
        this.editingState.selectedElements.clear();
        
        this.hideBatchEditor();
    }
    
    showBatchEditor(elements) {
        const editor = this.components.batchEditor;
        
        editor.innerHTML = `
            <div class="batch-editor-header">
                <h4>Batch Edit (${elements.length} items)</h4>
                <button class="close-batch-btn">&times;</button>
            </div>
            <div class="batch-editor-content">
                ${this.createBatchFields(elements)}
            </div>
        `;
        
        editor.style.display = 'block';
        
        // Event handlers
        const closeBtn = editor.querySelector('.close-batch-btn');
        closeBtn.addEventListener('click', () => this.disableBatchEditing());
    }
    
    hideBatchEditor() {
        this.components.batchEditor.style.display = 'none';
    }
    
    createBatchFields(elements) {
        // Find common properties
        const commonProperties = this.findCommonProperties(elements);
        
        return commonProperties.map(prop => {
            return `
                <div class="batch-field">
                    <label>
                        <input type="checkbox" data-property="${prop.name}">
                        ${prop.label}
                    </label>
                    <input type="${prop.type}" data-property="${prop.name}" placeholder="New value">
                </div>
            `;
        }).join('');
    }
    
    /**
     * Validation methods
     */
    validateValue(value, propertyName, element) {
        const validation = {
            isValid: true,
            errors: [],
            warnings: []
        };
        
        // Required validation
        if (this.config.validation.required.includes(propertyName) && (!value || value.trim() === '')) {
            validation.isValid = false;
            validation.errors.push(`${propertyName} is required`);
        }
        
        // Pattern validation
        const pattern = this.config.validation.patterns[propertyName];
        if (pattern && value && !pattern.test(value)) {
            validation.isValid = false;
            validation.errors.push(`${propertyName} format is invalid`);
        }
        
        // Custom validation
        const customValidator = this.config.validation.customValidators.get(propertyName);
        if (customValidator) {
            const customResult = customValidator(value, element);
            if (!customResult.isValid) {
                validation.isValid = false;
                validation.errors.push(...customResult.errors);
            }
            if (customResult.warnings) {
                validation.warnings.push(...customResult.warnings);
            }
        }
        
        return validation;
    }
    
    updateFieldValidation(input, validation) {
        const field = input.closest('.property-field');
        const messagesContainer = field.querySelector('.property-messages');
        
        // Clear previous messages
        messagesContainer.innerHTML = '';
        
        // Update input styling
        input.classList.remove('error', 'warning');
        
        if (!validation.isValid) {
            input.classList.add('error');
            validation.errors.forEach(error => {
                const errorMsg = document.createElement('div');
                errorMsg.className = 'error-message';
                errorMsg.textContent = error;
                messagesContainer.appendChild(errorMsg);
            });
        } else if (validation.warnings.length > 0) {
            input.classList.add('warning');
            validation.warnings.forEach(warning => {
                const warningMsg = document.createElement('div');
                warningMsg.className = 'warning-message';
                warningMsg.textContent = warning;
                messagesContainer.appendChild(warningMsg);
            });
        }
    }
    
    showValidationError(message) {
        const tooltip = this.components.validationTooltip;
        const editor = this.components.inlineEditor;
        
        tooltip.textContent = message;
        tooltip.style.display = 'block';
        
        // Position tooltip above inline editor
        const rect = editor.getBoundingClientRect();
        tooltip.style.left = rect.left + 'px';
        tooltip.style.top = (rect.top - tooltip.offsetHeight - 5) + 'px';
        
        // Hide after delay
        setTimeout(() => this.hideValidationError(), 3000);
    }
    
    hideValidationError() {
        this.components.validationTooltip.style.display = 'none';
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
    
    getElementAtPosition(x, y) {
        // Check for nodes first
        const node = this.graphRenderer.getNodeAtPosition(x, y);
        if (node) {
            return { type: 'node', ...node };
        }
        
        // Check for edges
        const edge = this.graphRenderer.getEdgeAtPosition(x, y);
        if (edge) {
            return { type: 'edge', ...edge };
        }
        
        return null;
    }
    
    getSelectedElements() {
        return this.graphRenderer.selectionManager?.getSelectedElements() || [];
    }
    
    getCurrentEditingElement() {
        return this.editingState.editingElement;
    }
    
    getEditableProperty(element) {
        const editableProps = this.config.inlineEditing.editableProperties;
        
        for (const prop of editableProps) {
            if (element.hasOwnProperty(prop)) {
                return prop;
            }
        }
        
        return editableProps[0]; // Default to first editable property
    }
    
    getElementSchema(element) {
        // Define schemas for different element types
        const nodeSchema = {
            properties: [
                { name: 'id', label: 'ID', type: 'text', readonly: true },
                { name: 'label', label: 'Label', type: 'text' },
                { name: 'color', label: 'Color', type: 'color' },
                { name: 'radius', label: 'Size', type: 'number', min: 5, max: 50 },
                { name: 'type', label: 'Type', type: 'select', options: [
                    { value: 'default', label: 'Default' },
                    { value: 'important', label: 'Important' },
                    { value: 'secondary', label: 'Secondary' }
                ]},
                { name: 'description', label: 'Description', type: 'textarea' },
                { name: 'visible', label: 'Visible', type: 'boolean' }
            ]
        };
        
        const edgeSchema = {
            properties: [
                { name: 'id', label: 'ID', type: 'text', readonly: true },
                { name: 'label', label: 'Label', type: 'text' },
                { name: 'color', label: 'Color', type: 'color' },
                { name: 'width', label: 'Width', type: 'number', min: 1, max: 10 },
                { name: 'type', label: 'Type', type: 'select', options: [
                    { value: 'solid', label: 'Solid' },
                    { value: 'dashed', label: 'Dashed' },
                    { value: 'dotted', label: 'Dotted' }
                ]},
                { name: 'weight', label: 'Weight', type: 'number' },
                { name: 'visible', label: 'Visible', type: 'boolean' }
            ]
        };
        
        return element.type === 'node' ? nodeSchema : edgeSchema;
    }
    
    findCommonProperties(elements) {
        if (elements.length === 0) return [];
        
        // Get properties from first element
        const firstElement = elements[0];
        const schema = this.getElementSchema(firstElement);
        
        // Filter to properties that exist in all elements
        return schema.properties.filter(prop => {
            return elements.every(el => el.hasOwnProperty(prop.name));
        });
    }
    
    positionInlineEditor(element) {
        const rect = this.canvas.getBoundingClientRect();
        const editor = this.components.inlineEditor;
        
        // Calculate position based on element type
        let x, y;
        
        if (element.type === 'node') {
            x = rect.left + element.x - (editor.offsetWidth / 2);
            y = rect.top + element.y - (element.radius || 10) - editor.offsetHeight - 5;
        } else if (element.type === 'edge') {
            // Position at edge midpoint
            const sourceNode = this.graphRenderer.getNode(element.source);
            const targetNode = this.graphRenderer.getNode(element.target);
            
            if (sourceNode && targetNode) {
                const midX = (sourceNode.x + targetNode.x) / 2;
                const midY = (sourceNode.y + targetNode.y) / 2;
                
                x = rect.left + midX - (editor.offsetWidth / 2);
                y = rect.top + midY - editor.offsetHeight - 5;
            }
        }
        
        // Ensure editor stays within viewport
        x = Math.max(10, Math.min(x, window.innerWidth - editor.offsetWidth - 10));
        y = Math.max(10, Math.min(y, window.innerHeight - editor.offsetHeight - 10));
        
        editor.style.left = x + 'px';
        editor.style.top = y + 'px';
    }
    
    getInputValue(input) {
        switch (input.type) {
            case 'checkbox':
                return input.checked;
            case 'number':
                return input.value ? parseFloat(input.value) : null;
            default:
                return input.value;
        }
    }
    
    isValidColor(color) {
        const style = new Option().style;
        style.color = color;
        return style.color !== '';
    }
    
    isEditingActive() {
        return this.editingState.isEditing;
    }
    
    /**
     * Change management
     */
    applyElementChange(element, property, newValue) {
        const oldValue = element[property];
        element[property] = newValue;
        
        // Update renderer
        this.graphRenderer.requestRedraw();
        
        // Emit change event
        this.emitEvent('elementChanged', {
            element,
            property,
            oldValue,
            newValue
        });
    }
    
    storePendingChange(element, property, value) {
        // Implementation for storing changes before commit
        if (!this.editingState.pendingChanges) {
            this.editingState.pendingChanges = new Map();
        }
        
        const key = `${element.id}_${property}`;
        this.editingState.pendingChanges.set(key, { element, property, value });
    }
    
    commitCurrentEdit() {
        if (this.editingState.currentEditor === 'inline') {
            return this.commitInlineEdit();
        }
        return false;
    }
    
    cancelCurrentEdit() {
        if (this.editingState.currentEditor === 'inline') {
            this.cancelInlineEdit();
        }
    }
    
    savePropertyChanges() {
        // Save all pending changes from property panel
        this.emitEvent('propertiesChanged', {
            element: this.getCurrentEditingElement()
        });
    }
    
    resetPropertyChanges() {
        // Reset property panel to original values
        const element = this.getCurrentEditingElement();
        if (element) {
            this.populatePropertyPanel(element);
        }
    }
    
    deleteSelected() {
        const selected = this.getSelectedElements();
        if (selected.length > 0) {
            this.emitEvent('elementsDelete', { elements: selected });
        }
    }
    
    editNextProperty() {
        // Move to next editable property
        const element = this.editingState.editingElement;
        if (!element) return;
        
        const editableProps = this.config.inlineEditing.editableProperties;
        const currentIndex = editableProps.indexOf(this.editingState.editingProperty);
        const nextIndex = (currentIndex + 1) % editableProps.length;
        const nextProperty = editableProps[nextIndex];
        
        if (element.hasOwnProperty(nextProperty)) {
            this.startInlineEdit(element, nextProperty);
        }
    }
    
    /**
     * Auto-save functionality
     */
    scheduleAutoSave() {
        this.clearAutoSave();
        
        this.autoSaveTimer = setTimeout(() => {
            this.performAutoSave();
        }, this.config.autoSaveDelay);
    }
    
    clearAutoSave() {
        if (this.autoSaveTimer) {
            clearTimeout(this.autoSaveTimer);
            this.autoSaveTimer = null;
        }
    }
    
    performAutoSave() {
        this.emitEvent('autoSave', {
            element: this.editingState.editingElement
        });
    }
    
    /**
     * Public API methods
     */
    
    /**
     * Add custom validator
     */
    addValidator(propertyName, validator) {
        this.config.validation.customValidators.set(propertyName, validator);
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        Object.assign(this.config, newConfig);
    }
    
    /**
     * Enable/disable editing
     */
    setEditingEnabled(enabled) {
        this.config.enableInlineEditing = enabled;
        this.config.enablePropertyPanel = enabled;
        
        if (!enabled && this.editingState.isEditing) {
            this.cancelCurrentEdit();
        }
    }
    
    /**
     * Check if element can be edited
     */
    canEditElement(element) {
        return element && (element.type === 'node' || element.type === 'edge');
    }
    
    /**
     * Get current editing state
     */
    getEditingState() {
        return {
            isEditing: this.editingState.isEditing,
            element: this.editingState.editingElement,
            property: this.editingState.editingProperty,
            hasChanges: this.editingState.hasChanges
        };
    }
    
    /**
     * Event system
     */
    emitEvent(eventType, data) {
        const event = new CustomEvent(`elementEdit${eventType.charAt(0).toUpperCase() + eventType.slice(1)}`, {
            detail: data
        });
        this.canvas.dispatchEvent(event);
    }
    
    /**
     * Cleanup and destroy
     */
    destroy() {
        // Clear auto-save timer
        this.clearAutoSave();
        
        // Remove event listeners
        this.canvas.removeEventListener('dblclick', this.handleCanvasDoubleClick);
        document.removeEventListener('keydown', this.handleKeydown);
        
        // Remove UI components
        Object.values(this.components).forEach(component => {
            if (component && component.parentNode) {
                component.parentNode.removeChild(component);
            }
        });
        
        // Clear state
        this.editingState.selectedElements.clear();
        this.editingState.validationErrors.clear();
        this.editingState.validationWarnings.clear();
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ElementEditingSystem;
}