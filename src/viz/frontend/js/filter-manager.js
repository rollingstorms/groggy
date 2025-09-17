/**
 * Phase 8: Filtering & Search - FilterManager Implementation
 * 
 * Comprehensive attribute-based filtering system as outlined in the
 * GROGGY_0_5_0_VISUALIZATION_ROADMAP.md section 2.2
 * 
 * Features:
 * - Real-time attribute-based filtering
 * - Multiple filter operators (equals, contains, greater_than, etc.)
 * - Filter combination logic (AND/OR)
 * - Dynamic filter UI generation
 * - Filter history and saved filters
 * - Performance optimization for large graphs
 */

class FilterManager {
    constructor(graphRenderer, websocketClient) {
        this.graphRenderer = graphRenderer;
        this.websocketClient = websocketClient;
        
        // Active filters storage
        this.activeFilters = new Map();
        this.filterHistory = [];
        this.savedFilters = new Map();
        
        // Filter UI elements
        this.filtersContainer = document.getElementById('filters-container');
        this.clearFiltersButton = document.getElementById('clear-filters');
        
        // Performance optimization
        this.filterCache = new Map();
        this.lastFilterHash = null;
        
        // Filter operators and their implementations
        this.operators = {
            'equals': (nodeValue, filterValue) => nodeValue === filterValue,
            'not_equals': (nodeValue, filterValue) => nodeValue !== filterValue,
            'contains': (nodeValue, filterValue) => {
                return String(nodeValue).toLowerCase().includes(String(filterValue).toLowerCase());
            },
            'not_contains': (nodeValue, filterValue) => {
                return !String(nodeValue).toLowerCase().includes(String(filterValue).toLowerCase());
            },
            'starts_with': (nodeValue, filterValue) => {
                return String(nodeValue).toLowerCase().startsWith(String(filterValue).toLowerCase());
            },
            'ends_with': (nodeValue, filterValue) => {
                return String(nodeValue).toLowerCase().endsWith(String(filterValue).toLowerCase());
            },
            'greater_than': (nodeValue, filterValue) => {
                const numNodeValue = Number(nodeValue);
                const numFilterValue = Number(filterValue);
                return !isNaN(numNodeValue) && !isNaN(numFilterValue) && numNodeValue > numFilterValue;
            },
            'less_than': (nodeValue, filterValue) => {
                const numNodeValue = Number(nodeValue);
                const numFilterValue = Number(filterValue);
                return !isNaN(numNodeValue) && !isNaN(numFilterValue) && numNodeValue < numFilterValue;
            },
            'greater_equal': (nodeValue, filterValue) => {
                const numNodeValue = Number(nodeValue);
                const numFilterValue = Number(filterValue);
                return !isNaN(numNodeValue) && !isNaN(numFilterValue) && numNodeValue >= numFilterValue;
            },
            'less_equal': (nodeValue, filterValue) => {
                const numNodeValue = Number(nodeValue);
                const numFilterValue = Number(filterValue);
                return !isNaN(numNodeValue) && !isNaN(numFilterValue) && numNodeValue <= numFilterValue;
            },
            'between': (nodeValue, filterValue) => {
                const numNodeValue = Number(nodeValue);
                const [min, max] = filterValue;
                return !isNaN(numNodeValue) && numNodeValue >= min && numNodeValue <= max;
            },
            'in': (nodeValue, filterValue) => {
                return Array.isArray(filterValue) && filterValue.includes(nodeValue);
            },
            'regex': (nodeValue, filterValue) => {
                try {
                    const regex = new RegExp(filterValue, 'i');
                    return regex.test(String(nodeValue));
                } catch (e) {
                    console.warn('Invalid regex pattern:', filterValue);
                    return false;
                }
            },
            'is_null': (nodeValue, filterValue) => {
                return nodeValue === null || nodeValue === undefined || nodeValue === '';
                }
            },
            'exists': (nodeValue, filterValue) => {
                return nodeValue !== undefined && nodeValue !== null;
            },
            'empty': (nodeValue, filterValue) => {
                return nodeValue === undefined || nodeValue === null || nodeValue === '';
            }
        };
        
        this.init();
    }
    
    init() {
        console.log('🔍 Initializing FilterManager for Phase 8');
        
        // Set up event listeners
        this.clearFiltersButton.addEventListener('click', () => this.clearAllFilters());
        
        // Initialize with available attributes
        this.setupInitialFilters();
    }
    
    /**
     * Set up initial filter UI based on available node attributes
     */
    async setupInitialFilters() {
        try {
            // Get graph metadata to understand available attributes
            const nodes = this.graphRenderer.getNodes();
            const availableAttributes = this.extractAvailableAttributes(nodes);
            
            console.log('📋 Available attributes for filtering:', availableAttributes);
            
            // Create initial filter UI
            this.renderFilterControls(availableAttributes);
            
        } catch (error) {
            console.error('❌ Error setting up filters:', error);
        }
    }
    
    /**
     * Extract all unique attributes from nodes for filter UI
     */
    extractAvailableAttributes(nodes) {
        const attributes = new Map();
        
        for (const node of nodes) {
            if (node.attributes) {
                for (const [key, value] of Object.entries(node.attributes)) {
                    if (!attributes.has(key)) {
                        attributes.set(key, {
                            name: key,
                            type: this.inferAttributeType(value),
                            uniqueValues: new Set(),
                            examples: []
                        });
                    }
                    
                    const attr = attributes.get(key);
                    attr.uniqueValues.add(value);
                    
                    // Keep first 5 examples for UI
                    if (attr.examples.length < 5 && !attr.examples.includes(value)) {
                        attr.examples.push(value);
                    }
                }
            }
        }
        
        // Convert to array and add metadata
        return Array.from(attributes.values()).map(attr => ({
            ...attr,
            uniqueValues: Array.from(attr.uniqueValues).slice(0, 20), // Limit for performance
            valueCount: attr.uniqueValues.size
        }));
    }
    
    /**
     * Infer the type of an attribute value
     */
    inferAttributeType(value) {
        if (typeof value === 'number') return 'number';
        if (typeof value === 'boolean') return 'boolean';
        if (value instanceof Date) return 'date';
        if (Array.isArray(value)) return 'array';
        if (typeof value === 'object') return 'object';
        
        // Check if string looks like a number
        if (typeof value === 'string') {
            if (!isNaN(Number(value)) && value.trim() !== '') return 'numeric_string';
            if (value.match(/^\d{4}-\d{2}-\d{2}/)) return 'date_string';
        }
        
        return 'string';
    }
    
    /**
     * Render the dynamic filter controls UI
     */
    renderFilterControls(availableAttributes) {
        this.filtersContainer.innerHTML = '';
        
        // Add filter button
        const addFilterButton = document.createElement('button');
        addFilterButton.textContent = '+ Add Filter';
        addFilterButton.className = 'control-button primary';
        addFilterButton.addEventListener('click', () => this.showAddFilterDialog(availableAttributes));
        
        this.filtersContainer.appendChild(addFilterButton);
        
        // Filter combination selector
        const combinationDiv = document.createElement('div');
        combinationDiv.className = 'filter-combination';
        combinationDiv.innerHTML = `
            <label>Combine filters:</label>
            <select id="filter-combination-select" class="control-select">
                <option value="AND">All conditions (AND)</option>
                <option value="OR">Any condition (OR)</option>
            </select>
        `;
        this.filtersContainer.appendChild(combinationDiv);
        
        // Active filters container
        const activeFiltersDiv = document.createElement('div');
        activeFiltersDiv.id = 'active-filters';
        activeFiltersDiv.className = 'active-filters';
        this.filtersContainer.appendChild(activeFiltersDiv);
        
        // Filter presets
        this.renderFilterPresets();
        
        // Update filter combination when changed
        const combinationSelect = document.getElementById('filter-combination-select');
        combinationSelect.addEventListener('change', () => this.applyFilters());
    }
    
    /**
     * Show dialog to add a new filter
     */
    showAddFilterDialog(availableAttributes) {
        // Create modal dialog for adding filters
        const dialog = document.createElement('div');
        dialog.className = 'modal filter-dialog';
        dialog.style.display = 'flex';
        
        dialog.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Add Filter</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="form-group">
                        <label>Attribute:</label>
                        <select id="filter-attribute-select" class="control-select">
                            ${availableAttributes.map(attr => 
                                `<option value="${attr.name}" data-type="${attr.type}">
                                    ${attr.name} (${attr.valueCount} values)
                                </option>`
                            ).join('')}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Operator:</label>
                        <select id="filter-operator-select" class="control-select">
                            <option value="equals">Equals</option>
                            <option value="not_equals">Not Equals</option>
                            <option value="contains">Contains</option>
                            <option value="not_contains">Does Not Contain</option>
                            <option value="starts_with">Starts With</option>
                            <option value="ends_with">Ends With</option>
                            <option value="greater_than">Greater Than</option>
                            <option value="less_than">Less Than</option>
                            <option value="greater_equal">Greater or Equal</option>
                            <option value="less_equal">Less or Equal</option>
                            <option value="exists">Has Value</option>
                            <option value="empty">Is Empty</option>
                            <option value="regex">Regex Match</option>
                        </select>
                    </div>
                    
                    <div class="form-group" id="filter-value-group">
                        <label>Value:</label>
                        <input type="text" id="filter-value-input" class="control-input" placeholder="Enter filter value">
                        <div id="filter-suggestions" class="filter-suggestions"></div>
                    </div>
                    
                    <div class="form-actions">
                        <button id="add-filter-confirm" class="control-button primary">Add Filter</button>
                        <button id="add-filter-cancel" class="control-button secondary">Cancel</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(dialog);
        
        // Set up dialog functionality
        this.setupAddFilterDialog(dialog, availableAttributes);
    }
    
    /**
     * Set up the add filter dialog functionality
     */
    setupAddFilterDialog(dialog, availableAttributes) {
        const attributeSelect = dialog.querySelector('#filter-attribute-select');
        const operatorSelect = dialog.querySelector('#filter-operator-select');
        const valueInput = dialog.querySelector('#filter-value-input');
        const suggestionsDiv = dialog.querySelector('#filter-suggestions');
        const confirmButton = dialog.querySelector('#add-filter-confirm');
        const cancelButton = dialog.querySelector('#add-filter-cancel');
        const closeButton = dialog.querySelector('.modal-close');
        
        // Update operator options based on attribute type
        const updateOperatorOptions = () => {
            const selectedAttr = availableAttributes.find(a => a.name === attributeSelect.value);
            const operators = this.getOperatorsForType(selectedAttr.type);
            
            operatorSelect.innerHTML = operators.map(op => 
                `<option value="${op.value}">${op.label}</option>`
            ).join('');
            
            this.updateValueInputForOperator(operatorSelect.value, selectedAttr, valueInput, suggestionsDiv);
        };
        
        // Update value input based on operator
        const updateValueInput = () => {
            const selectedAttr = availableAttributes.find(a => a.name === attributeSelect.value);
            this.updateValueInputForOperator(operatorSelect.value, selectedAttr, valueInput, suggestionsDiv);
        };
        
        attributeSelect.addEventListener('change', updateOperatorOptions);
        operatorSelect.addEventListener('change', updateValueInput);
        
        // Add suggestions on input
        valueInput.addEventListener('input', () => {
            const selectedAttr = availableAttributes.find(a => a.name === attributeSelect.value);
            this.showFilterSuggestions(valueInput.value, selectedAttr, suggestionsDiv);
        });
        
        // Confirm button
        confirmButton.addEventListener('click', () => {
            const filterId = Date.now().toString();
            const filter = {
                id: filterId,
                attribute: attributeSelect.value,
                operator: operatorSelect.value,
                value: this.parseFilterValue(valueInput.value, operatorSelect.value),
                label: this.createFilterLabel(attributeSelect.value, operatorSelect.value, valueInput.value)
            };
            
            this.addFilter(filter);
            document.body.removeChild(dialog);
        });
        
        // Cancel/close buttons
        const closeDialog = () => document.body.removeChild(dialog);
        cancelButton.addEventListener('click', closeDialog);
        closeButton.addEventListener('click', closeDialog);
        
        // Initialize with first attribute
        updateOperatorOptions();
    }
    
    /**
     * Get appropriate operators for attribute type
     */
    getOperatorsForType(type) {
        const baseOperators = [
            { value: 'equals', label: 'Equals' },
            { value: 'not_equals', label: 'Not Equals' },
            { value: 'exists', label: 'Has Value' },
            { value: 'empty', label: 'Is Empty' }
        ];
        
        const stringOperators = [
            { value: 'contains', label: 'Contains' },
            { value: 'not_contains', label: 'Does Not Contain' },
            { value: 'starts_with', label: 'Starts With' },
            { value: 'ends_with', label: 'Ends With' },
            { value: 'regex', label: 'Regex Match' }
        ];
        
        const numericOperators = [
            { value: 'greater_than', label: 'Greater Than' },
            { value: 'less_than', label: 'Less Than' },
            { value: 'greater_equal', label: 'Greater or Equal' },
            { value: 'less_equal', label: 'Less or Equal' }
        ];
        
        switch (type) {
            case 'number':
            case 'numeric_string':
                return [...baseOperators, ...numericOperators];
            case 'string':
                return [...baseOperators, ...stringOperators];
            case 'boolean':
                return baseOperators;
            default:
                return [...baseOperators, ...stringOperators];
        }
    }
    
    /**
     * Update value input based on selected operator
     */
    updateValueInputForOperator(operator, attribute, valueInput, suggestionsDiv) {
        valueInput.style.display = ['exists', 'empty'].includes(operator) ? 'none' : 'block';
        
        if (operator === 'in') {
            valueInput.placeholder = 'Enter comma-separated values';
        } else if (operator === 'between') {
            valueInput.placeholder = 'Enter min,max values';
        } else if (operator === 'regex') {
            valueInput.placeholder = 'Enter regular expression';
        } else {
            valueInput.placeholder = `Enter ${attribute.name} value`;
        }
        
        // Show example values as suggestions
        if (attribute.examples.length > 0) {
            this.showFilterSuggestions('', attribute, suggestionsDiv);
        }
    }
    
    /**
     * Show filter value suggestions
     */
    showFilterSuggestions(inputValue, attribute, suggestionsDiv) {
        if (!attribute.examples.length) return;
        
        const filteredExamples = attribute.examples.filter(example => 
            String(example).toLowerCase().includes(inputValue.toLowerCase())
        );
        
        suggestionsDiv.innerHTML = filteredExamples.slice(0, 5).map(example => 
            `<div class="filter-suggestion" data-value="${example}">${example}</div>`
        ).join('');
        
        // Add click handlers for suggestions
        suggestionsDiv.querySelectorAll('.filter-suggestion').forEach(suggestion => {
            suggestion.addEventListener('click', () => {
                const valueInput = suggestionsDiv.parentElement.querySelector('#filter-value-input');
                valueInput.value = suggestion.dataset.value;
                suggestionsDiv.innerHTML = '';
            });
        });
    }
    
    /**
     * Parse filter value based on operator
     */
    parseFilterValue(value, operator) {
        switch (operator) {
            case 'in':
                return value.split(',').map(v => v.trim());
            case 'between':
                const parts = value.split(',').map(v => parseFloat(v.trim()));
                return parts.length === 2 ? parts : [0, 0];
            case 'exists':
            case 'empty':
                return null;
            default:
                return value;
        }
    }
    
    /**
     * Create human-readable filter label
     */
    createFilterLabel(attribute, operator, value) {
        const operatorLabels = {
            'equals': '=',
            'not_equals': '≠',
            'contains': '∋',
            'not_contains': '∌',
            'starts_with': 'starts with',
            'ends_with': 'ends with',
            'greater_than': '>',
            'less_than': '<',
            'greater_equal': '≥',
            'less_equal': '≤',
            'exists': 'exists',
            'empty': 'is empty',
            'regex': '~'
        };
        
        const opLabel = operatorLabels[operator] || operator;
        
        if (['exists', 'empty'].includes(operator)) {
            return `${attribute} ${opLabel}`;
        }
        
        return `${attribute} ${opLabel} ${value}`;
    }
    
    /**
     * Add a new filter and apply it
     */
    addFilter(filter) {
        console.log('➕ Adding filter:', filter);
        
        this.activeFilters.set(filter.id, filter);
        this.renderActiveFilters();
        this.applyFilters();
        
        // Add to history
        this.addToFilterHistory('add_filter', filter);
    }
    
    /**
     * Remove a filter
     */
    removeFilter(filterId) {
        console.log('➖ Removing filter:', filterId);
        
        const filter = this.activeFilters.get(filterId);
        this.activeFilters.delete(filterId);
        this.renderActiveFilters();
        this.applyFilters();
        
        // Add to history
        this.addToFilterHistory('remove_filter', filter);
    }
    
    /**
     * Clear all active filters
     */
    clearAllFilters() {
        console.log('🧹 Clearing all filters');
        
        const clearedFilters = Array.from(this.activeFilters.values());
        this.activeFilters.clear();
        this.renderActiveFilters();
        this.applyFilters();
        
        // Add to history
        this.addToFilterHistory('clear_all', { filters: clearedFilters });
    }
    
    /**
     * Render active filters in the UI
     */
    renderActiveFilters() {
        const activeFiltersDiv = document.getElementById('active-filters');
        
        if (this.activeFilters.size === 0) {
            activeFiltersDiv.innerHTML = '<p class="no-filters">No active filters</p>';
            return;
        }
        
        activeFiltersDiv.innerHTML = Array.from(this.activeFilters.values()).map(filter => `
            <div class="active-filter" data-filter-id="${filter.id}">
                <span class="filter-label">${filter.label}</span>
                <button class="filter-remove" data-filter-id="${filter.id}" title="Remove filter">&times;</button>
            </div>
        `).join('');
        
        // Add remove button event listeners
        activeFiltersDiv.querySelectorAll('.filter-remove').forEach(button => {
            button.addEventListener('click', () => {
                this.removeFilter(button.dataset.filterId);
            });
        });
    }
    
    /**
     * Apply all active filters to the graph
     */
    async applyFilters() {
        console.log('🔄 Applying filters...');
        
        const nodes = this.graphRenderer.getNodes();
        const combinationType = document.getElementById('filter-combination-select')?.value || 'AND';
        
        // Performance optimization: check if filters changed
        const filterHash = this.getFilterHash();
        if (filterHash === this.lastFilterHash) {
            console.log('⚡ Filters unchanged, using cache');
            return;
        }
        
        // Apply filters
        const filteredNodeIds = this.filterNodes(nodes, combinationType);
        
        // Update graph renderer
        this.graphRenderer.setFilteredNodes(filteredNodeIds);
        
        // Update UI
        this.updateFilterStats(filteredNodeIds, nodes);
        
        // Cache results
        this.lastFilterHash = filterHash;
        this.filterCache.set(filterHash, filteredNodeIds);
        
        console.log(`✅ Applied ${this.activeFilters.size} filters: ${filteredNodeIds.length}/${nodes.length} nodes visible`);
    }
    
    /**
     * Filter nodes based on active filters and combination type
     */
    filterNodes(nodes, combinationType) {
        if (this.activeFilters.size === 0) {
            return nodes.map(node => node.id);
        }
        
        const filters = Array.from(this.activeFilters.values());
        
        return nodes.filter(node => {
            const results = filters.map(filter => this.evaluateFilter(node, filter));
            
            return combinationType === 'AND' ? 
                results.every(result => result) : 
                results.some(result => result);
        }).map(node => node.id);
    }
    
    /**
     * Evaluate a single filter against a node
     */
    evaluateFilter(node, filter) {
        const nodeValue = node.attributes[filter.attribute];
        const operator = this.operators[filter.operator];
        
        if (!operator) {
            console.warn(`Unknown filter operator: ${filter.operator}`);
            return true;
        }
        
        return operator(nodeValue, filter.value);
    }
    
    /**
     * Generate hash for current filter state (for caching)
     */
    getFilterHash() {
        const filterData = Array.from(this.activeFilters.values()).map(f => 
            `${f.attribute}:${f.operator}:${JSON.stringify(f.value)}`
        );
        const combinationType = document.getElementById('filter-combination-select')?.value || 'AND';
        
        return `${combinationType}|${filterData.sort().join('|')}`;
    }
    
    /**
     * Update filter statistics in the UI
     */
    updateFilterStats(filteredNodeIds, allNodes) {
        const visibleCount = filteredNodeIds.length;
        const totalCount = allNodes.length;
        const hiddenCount = totalCount - visibleCount;
        
        // Update status message
        const statusMessage = document.getElementById('status-message');
        if (statusMessage) {
            if (this.activeFilters.size > 0) {
                statusMessage.textContent = `Showing ${visibleCount} of ${totalCount} nodes (${hiddenCount} hidden)`;
            } else {
                statusMessage.textContent = 'Ready';
            }
        }
    }
    
    /**
     * Render filter presets for common filtering scenarios
     */
    renderFilterPresets() {
        const presetsDiv = document.createElement('div');
        presetsDiv.className = 'filter-presets';
        presetsDiv.innerHTML = `
            <h4>Quick Filters:</h4>
            <div class="preset-buttons">
                <button class="preset-button" data-preset="high-degree">High Degree Nodes</button>
                <button class="preset-button" data-preset="isolated">Isolated Nodes</button>
                <button class="preset-button" data-preset="central">Central Nodes</button>
            </div>
        `;
        
        this.filtersContainer.appendChild(presetsDiv);
        
        // Add preset button handlers
        presetsDiv.querySelectorAll('.preset-button').forEach(button => {
            button.addEventListener('click', () => {
                this.applyPresetFilter(button.dataset.preset);
            });
        });
    }
    
    /**
     * Apply preset filters
     */
    applyPresetFilter(presetType) {
        console.log('🎯 Applying preset filter:', presetType);
        
        // Clear existing filters first
        this.clearAllFilters();
        
        // Apply preset based on type
        switch (presetType) {
            case 'high-degree':
                // Add filter for nodes with degree > average
                this.addPresetFilter('degree', 'greater_than', 'auto_calculate');
                break;
            case 'isolated':
                // Add filter for nodes with degree <= 1
                this.addPresetFilter('degree', 'less_equal', 1);
                break;
            case 'central':
                // Add filter for nodes with high centrality
                this.addPresetFilter('betweenness_centrality', 'greater_than', 0.1);
                break;
        }
    }
    
    /**
     * Add a preset filter
     */
    addPresetFilter(attribute, operator, value) {
        const filterId = `preset_${Date.now()}`;
        const filter = {
            id: filterId,
            attribute,
            operator,
            value,
            label: this.createFilterLabel(attribute, operator, value),
            isPreset: true
        };
        
        this.addFilter(filter);
    }
    
    /**
     * Add action to filter history
     */
    addToFilterHistory(action, data) {
        const historyEntry = {
            timestamp: Date.now(),
            action,
            data,
            filterState: this.getFilterSnapshot()
        };
        
        this.filterHistory.push(historyEntry);
        
        // Limit history size
        if (this.filterHistory.length > 50) {
            this.filterHistory.shift();
        }
    }
    
    /**
     * Get snapshot of current filter state
     */
    getFilterSnapshot() {
        return {
            filters: Array.from(this.activeFilters.entries()),
            combination: document.getElementById('filter-combination-select')?.value || 'AND'
        };
    }
    
    /**
     * Save current filter configuration
     */
    saveFilterConfiguration(name) {
        const config = {
            name,
            timestamp: Date.now(),
            filters: Array.from(this.activeFilters.values()),
            combination: document.getElementById('filter-combination-select')?.value || 'AND'
        };
        
        this.savedFilters.set(name, config);
        console.log('💾 Saved filter configuration:', name);
        
        // Update UI if saved filters panel exists
        this.updateSavedFiltersUI();
    }
    
    /**
     * Load saved filter configuration
     */
    loadFilterConfiguration(name) {
        const config = this.savedFilters.get(name);
        if (!config) {
            console.warn('⚠️  Saved filter configuration not found:', name);
            return;
        }
        
        console.log('📂 Loading filter configuration:', name);
        
        // Clear current filters
        this.activeFilters.clear();
        
        // Load saved filters
        config.filters.forEach(filter => {
            this.activeFilters.set(filter.id, filter);
        });
        
        // Set combination type
        const combinationSelect = document.getElementById('filter-combination-select');
        if (combinationSelect) {
            combinationSelect.value = config.combination;
        }
        
        // Update UI and apply
        this.renderActiveFilters();
        this.applyFilters();
    }
    
    /**
     * Update saved filters UI
     */
    updateSavedFiltersUI() {
        // This would update a saved filters panel if implemented
        // For now, just log the available saved filters
        console.log('💾 Available saved filters:', Array.from(this.savedFilters.keys()));
    }
    
    /**
     * Get current filter state for external access
     */
    getFilterState() {
        return {
            activeFilters: Array.from(this.activeFilters.values()),
            combination: document.getElementById('filter-combination-select')?.value || 'AND',
            history: this.filterHistory.slice(-10), // Last 10 history entries
            savedFilters: Array.from(this.savedFilters.keys())
        };
    }
    
    /**
     * Export filter configuration as JSON
     */
    exportFilters() {
        const exportData = {
            version: '1.0',
            timestamp: Date.now(),
            activeFilters: Array.from(this.activeFilters.values()),
            savedFilters: Array.from(this.savedFilters.values())
        };
        
        return JSON.stringify(exportData, null, 2);
    }
    
    /**
     * Import filter configuration from JSON
     */
    importFilters(jsonData) {
        try {
            const importData = JSON.parse(jsonData);
            
            // Import saved filters
            if (importData.savedFilters) {
                importData.savedFilters.forEach(config => {
                    this.savedFilters.set(config.name, config);
                });
            }
            
            console.log('📥 Imported filter configurations');
            this.updateSavedFiltersUI();
            
        } catch (error) {
            console.error('❌ Error importing filters:', error);
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FilterManager;
}