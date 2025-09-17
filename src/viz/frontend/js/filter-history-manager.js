/**
 * Phase 8: Filtering & Search - FilterHistoryManager Implementation
 * 
 * Manages filter and search history, saved configurations, and user sessions.
 * Provides undo/redo functionality and filter session persistence.
 * 
 * This corresponds to task 8.5: Create filter history and saved filters
 * 
 * Features:
 * - Filter action history with undo/redo
 * - Saved filter configurations
 * - Session persistence across browser refreshes
 * - Filter sharing and import/export
 * - Usage analytics and recommendations
 */

class FilterHistoryManager {
    constructor(filterManager, searchManager) {
        this.filterManager = filterManager;
        this.searchManager = searchManager;
        
        // History tracking
        this.actionHistory = [];
        this.currentHistoryIndex = -1;
        this.maxHistorySize = 50;
        
        // Saved configurations
        this.savedConfigurations = new Map();
        this.configurationPresets = new Map();
        
        // Session management
        this.sessionId = this.generateSessionId();
        this.sessionData = {
            startTime: Date.now(),
            actions: [],
            searches: [],
            filters: []
        };
        
        // UI elements
        this.historyPanel = null;
        this.savedFiltersPanel = null;
        
        // Analytics
        this.usageAnalytics = {
            totalActions: 0,
            mostUsedFilters: new Map(),
            searchPatterns: new Map(),
            sessionDuration: 0
        };
        
        this.init();
    }
    
    init() {
        console.log('📚 Initializing FilterHistoryManager for Phase 8');
        
        this.loadSavedConfigurations();
        this.loadSessionData();
        this.setupEventListeners();
        this.createHistoryUI();
        this.loadDefaultPresets();
    }
    
    /**
     * Set up event listeners for tracking filter and search actions
     */
    setupEventListeners() {
        // Track filter manager actions
        document.addEventListener('filterAdded', (event) => {
            this.recordAction('filter_added', event.detail);
        });
        
        document.addEventListener('filterRemoved', (event) => {
            this.recordAction('filter_removed', event.detail);
        });
        
        document.addEventListener('filtersCleared', (event) => {
            this.recordAction('filters_cleared', event.detail);
        });
        
        // Track search manager actions
        document.addEventListener('searchPerformed', (event) => {
            this.recordAction('search_performed', event.detail);
        });
        
        // Keyboard shortcuts for history navigation
        document.addEventListener('keydown', (event) => {
            if (event.ctrlKey || event.metaKey) {
                this.handleHistoryShortcuts(event);
            }
        });
        
        // Auto-save session data periodically
        setInterval(() => this.saveSessionData(), 30000); // Every 30 seconds
        
        // Save session data before page unload
        window.addEventListener('beforeunload', () => {
            this.saveSessionData();
        });
        
        console.log('✅ History event listeners set up');
    }
    
    /**
     * Handle keyboard shortcuts for history operations
     */
    handleHistoryShortcuts(event) {
        switch (event.key) {
            case 'z': // Ctrl+Z - Undo
                if (!event.shiftKey) {
                    event.preventDefault();
                    this.undo();
                }
                break;
            case 'y': // Ctrl+Y - Redo
                event.preventDefault();
                this.redo();
                break;
            case 'Z': // Ctrl+Shift+Z - Redo (alternative)
                if (event.shiftKey) {
                    event.preventDefault();
                    this.redo();
                }
                break;
        }
    }
    
    /**
     * Record a filter/search action in history
     */
    recordAction(actionType, actionData) {
        const action = {
            id: Date.now().toString(),
            type: actionType,
            data: actionData,
            timestamp: Date.now(),
            sessionId: this.sessionId,
            filterState: this.captureFilterState(),
            searchState: this.captureSearchState()
        };
        
        console.log(`📝 Recording action: ${actionType}`, action);
        
        // Remove any actions after current position (when undoing then doing new action)
        this.actionHistory = this.actionHistory.slice(0, this.currentHistoryIndex + 1);
        
        // Add new action
        this.actionHistory.push(action);
        this.currentHistoryIndex = this.actionHistory.length - 1;
        
        // Limit history size
        if (this.actionHistory.length > this.maxHistorySize) {
            this.actionHistory.shift();
            this.currentHistoryIndex--;
        }
        
        // Update session data
        this.sessionData.actions.push(action);
        
        // Update analytics
        this.updateAnalytics(actionType, actionData);
        
        // Update UI
        this.updateHistoryUI();
        
        // Auto-save if significant action
        if (['filter_added', 'search_performed', 'filters_cleared'].includes(actionType)) {
            this.saveSessionData();
        }
    }
    
    /**
     * Undo the last action
     */
    undo() {
        if (this.currentHistoryIndex < 0) {
            console.log('⚠️  No actions to undo');
            return false;
        }
        
        const currentAction = this.actionHistory[this.currentHistoryIndex];
        console.log(`↶ Undoing action: ${currentAction.type}`);
        
        // Move to previous state
        this.currentHistoryIndex--;
        
        // Restore previous state
        if (this.currentHistoryIndex >= 0) {
            const previousAction = this.actionHistory[this.currentHistoryIndex];
            this.restoreState(previousAction.filterState, previousAction.searchState);
        } else {
            // Restore to initial empty state
            this.restoreInitialState();
        }
        
        // Update UI
        this.updateHistoryUI();
        
        // Record undo action
        this.recordUndoRedoAction('undo', currentAction);
        
        return true;
    }
    
    /**
     * Redo the last undone action
     */
    redo() {
        if (this.currentHistoryIndex >= this.actionHistory.length - 1) {
            console.log('⚠️  No actions to redo');
            return false;
        }
        
        // Move to next state
        this.currentHistoryIndex++;
        const targetAction = this.actionHistory[this.currentHistoryIndex];
        
        console.log(`↷ Redoing action: ${targetAction.type}`);
        
        // Restore target state
        this.restoreState(targetAction.filterState, targetAction.searchState);
        
        // Update UI
        this.updateHistoryUI();
        
        // Record redo action
        this.recordUndoRedoAction('redo', targetAction);
        
        return true;
    }
    
    /**
     * Capture current filter state
     */
    captureFilterState() {
        return {
            activeFilters: this.filterManager.getFilterState().activeFilters,
            combination: document.getElementById('filter-combination-select')?.value || 'AND'
        };
    }
    
    /**
     * Capture current search state
     */
    captureSearchState() {
        return {
            query: this.searchManager.currentQuery || '',
            searchType: this.searchManager.currentSearchType || 'global',
            results: this.searchManager.currentResults || []
        };
    }
    
    /**
     * Restore filter and search state
     */
    restoreState(filterState, searchState) {
        // Restore filter state
        if (filterState) {
            this.filterManager.restoreFilterState(filterState);
        }
        
        // Restore search state
        if (searchState && searchState.query) {
            this.searchManager.restoreSearchState(searchState);
        }
    }
    
    /**
     * Restore to initial empty state
     */
    restoreInitialState() {
        this.filterManager.clearAllFilters();
        this.searchManager.clearSearch();
    }
    
    /**
     * Save current configuration with a name
     */
    saveConfiguration(name, description = '') {
        const configuration = {
            id: Date.now().toString(),
            name,
            description,
            timestamp: Date.now(),
            filterState: this.captureFilterState(),
            searchState: this.captureSearchState(),
            metadata: {
                nodeCount: this.getVisibleNodeCount(),
                sessionId: this.sessionId
            }
        };
        
        this.savedConfigurations.set(name, configuration);
        
        console.log('💾 Saved configuration:', name);
        
        // Update UI
        this.updateSavedConfigurationsUI();
        
        // Persist to storage
        this.saveSavedConfigurations();
        
        return configuration;
    }
    
    /**
     * Load a saved configuration
     */
    loadConfiguration(name) {
        const configuration = this.savedConfigurations.get(name);
        if (!configuration) {
            console.warn('⚠️  Configuration not found:', name);
            return false;
        }
        
        console.log('📂 Loading configuration:', name);
        
        // Record this as an action for undo capability
        this.recordAction('configuration_loaded', { name, configuration });
        
        // Restore the configuration state
        this.restoreState(configuration.filterState, configuration.searchState);
        
        return true;
    }
    
    /**
     * Delete a saved configuration
     */
    deleteConfiguration(name) {
        if (this.savedConfigurations.has(name)) {
            this.savedConfigurations.delete(name);
            console.log('🗑️ Deleted configuration:', name);
            
            // Update UI
            this.updateSavedConfigurationsUI();
            
            // Persist changes
            this.saveSavedConfigurations();
            
            return true;
        }
        return false;
    }
    
    /**
     * Create history UI panel
     */
    createHistoryUI() {
        // Create history panel button
        const historyButton = document.createElement('button');
        historyButton.id = 'show-history-panel';
        historyButton.className = 'control-button secondary';
        historyButton.textContent = '📚 History';
        historyButton.title = 'Show filter and search history';
        
        // Add to sidebar or appropriate location
        const filtersContainer = document.getElementById('filters-container');
        if (filtersContainer) {
            filtersContainer.appendChild(historyButton);
        }
        
        historyButton.addEventListener('click', () => {
            this.showHistoryPanel();
        });
        
        // Create saved configurations button
        const savedButton = document.createElement('button');
        savedButton.id = 'show-saved-panel';
        savedButton.className = 'control-button secondary';
        savedButton.textContent = '💾 Saved';
        savedButton.title = 'Show saved filter configurations';
        
        if (filtersContainer) {
            filtersContainer.appendChild(savedButton);
        }
        
        savedButton.addEventListener('click', () => {
            this.showSavedConfigurationsPanel();
        });
        
        console.log('✅ History UI components created');
    }
    
    /**
     * Show history panel
     */
    showHistoryPanel() {
        if (this.historyPanel) {
            this.historyPanel.style.display = 'block';
            this.updateHistoryUI();
            return;
        }
        
        this.historyPanel = document.createElement('div');
        this.historyPanel.className = 'modal history-panel';
        this.historyPanel.style.display = 'flex';
        
        this.historyPanel.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Filter & Search History</h3>
                    <div class="history-controls">
                        <button id="undo-btn" class="control-button small" title="Undo (Ctrl+Z)">↶ Undo</button>
                        <button id="redo-btn" class="control-button small" title="Redo (Ctrl+Y)">↷ Redo</button>
                    </div>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="history-tabs">
                        <button class="history-tab active" data-tab="actions">Actions</button>
                        <button class="history-tab" data-tab="analytics">Analytics</button>
                        <button class="history-tab" data-tab="session">Session</button>
                    </div>
                    
                    <div id="history-actions-tab" class="history-tab-content">
                        <div id="history-list" class="history-list">
                            <!-- History items will be populated here -->
                        </div>
                    </div>
                    
                    <div id="history-analytics-tab" class="history-tab-content" style="display: none;">
                        <div id="history-analytics" class="history-analytics">
                            <!-- Analytics will be populated here -->
                        </div>
                    </div>
                    
                    <div id="history-session-tab" class="history-tab-content" style="display: none;">
                        <div id="session-info" class="session-info">
                            <!-- Session info will be populated here -->
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button id="clear-history-btn" class="control-button secondary">Clear History</button>
                    <button id="export-history-btn" class="control-button secondary">Export</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(this.historyPanel);
        
        // Set up panel functionality
        this.setupHistoryPanel();
        this.updateHistoryUI();
    }
    
    /**
     * Set up history panel event listeners
     */
    setupHistoryPanel() {
        // Close button
        this.historyPanel.querySelector('.modal-close').addEventListener('click', () => {
            this.historyPanel.style.display = 'none';
        });
        
        // Tab switching
        this.historyPanel.querySelectorAll('.history-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                this.switchHistoryTab(tab.dataset.tab);
            });
        });
        
        // Undo/Redo buttons
        this.historyPanel.querySelector('#undo-btn').addEventListener('click', () => {
            this.undo();
        });
        
        this.historyPanel.querySelector('#redo-btn').addEventListener('click', () => {
            this.redo();
        });
        
        // Clear history
        this.historyPanel.querySelector('#clear-history-btn').addEventListener('click', () => {
            this.clearHistory();
        });
        
        // Export history
        this.historyPanel.querySelector('#export-history-btn').addEventListener('click', () => {
            this.exportHistory();
        });
    }
    
    /**
     * Update history UI with current state
     */
    updateHistoryUI() {
        if (!this.historyPanel) return;
        
        const historyList = this.historyPanel.querySelector('#history-list');
        if (!historyList) return;
        
        // Render history items
        historyList.innerHTML = this.actionHistory.map((action, index) => {
            const isCurrentAction = index === this.currentHistoryIndex;
            const isFutureAction = index > this.currentHistoryIndex;
            
            return `
                <div class="history-item ${isCurrentAction ? 'current' : ''} ${isFutureAction ? 'future' : ''}">
                    <div class="history-item-header">
                        <span class="history-action-type">${this.formatActionType(action.type)}</span>
                        <span class="history-timestamp">${this.formatTimestamp(action.timestamp)}</span>
                    </div>
                    <div class="history-item-details">
                        ${this.formatActionDetails(action)}
                    </div>
                    <div class="history-item-actions">
                        <button class="history-restore-btn" data-index="${index}">Restore</button>
                    </div>
                </div>
            `;
        }).join('');
        
        // Add restore button listeners
        historyList.querySelectorAll('.history-restore-btn').forEach(button => {
            button.addEventListener('click', () => {
                this.restoreToHistoryIndex(parseInt(button.dataset.index));
            });
        });
        
        // Update undo/redo button states
        const undoBtn = this.historyPanel.querySelector('#undo-btn');
        const redoBtn = this.historyPanel.querySelector('#redo-btn');
        
        undoBtn.disabled = this.currentHistoryIndex < 0;
        redoBtn.disabled = this.currentHistoryIndex >= this.actionHistory.length - 1;
        
        // Update analytics tab
        this.updateAnalyticsUI();
        
        // Update session tab
        this.updateSessionUI();
    }
    
    /**
     * Show saved configurations panel
     */
    showSavedConfigurationsPanel() {
        if (this.savedFiltersPanel) {
            this.savedFiltersPanel.style.display = 'block';
            this.updateSavedConfigurationsUI();
            return;
        }
        
        this.savedFiltersPanel = document.createElement('div');
        this.savedFiltersPanel.className = 'modal saved-filters-panel';
        this.savedFiltersPanel.style.display = 'flex';
        
        this.savedFiltersPanel.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Saved Filter Configurations</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="saved-configs-actions">
                        <button id="save-current-config" class="control-button primary">💾 Save Current</button>
                        <button id="import-configs" class="control-button secondary">📥 Import</button>
                    </div>
                    <div id="saved-configs-list" class="saved-configs-list">
                        <!-- Saved configurations will be populated here -->
                    </div>
                </div>
                <div class="modal-footer">
                    <button id="export-configs" class="control-button secondary">Export All</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(this.savedFiltersPanel);
        
        // Set up panel functionality
        this.setupSavedConfigurationsPanel();
        this.updateSavedConfigurationsUI();
    }
    
    /**
     * Set up saved configurations panel
     */
    setupSavedConfigurationsPanel() {
        // Close button
        this.savedFiltersPanel.querySelector('.modal-close').addEventListener('click', () => {
            this.savedFiltersPanel.style.display = 'none';
        });
        
        // Save current configuration
        this.savedFiltersPanel.querySelector('#save-current-config').addEventListener('click', () => {
            this.promptSaveCurrentConfiguration();
        });
        
        // Import configurations
        this.savedFiltersPanel.querySelector('#import-configs').addEventListener('click', () => {
            this.promptImportConfigurations();
        });
        
        // Export configurations
        this.savedFiltersPanel.querySelector('#export-configs').addEventListener('click', () => {
            this.exportSavedConfigurations();
        });
    }
    
    /**
     * Update saved configurations UI
     */
    updateSavedConfigurationsUI() {
        if (!this.savedFiltersPanel) return;
        
        const configsList = this.savedFiltersPanel.querySelector('#saved-configs-list');
        if (!configsList) return;
        
        if (this.savedConfigurations.size === 0) {
            configsList.innerHTML = '<p class="no-configs">No saved configurations</p>';
            return;
        }
        
        configsList.innerHTML = Array.from(this.savedConfigurations.values()).map(config => `
            <div class="saved-config-item">
                <div class="config-header">
                    <h4 class="config-name">${config.name}</h4>
                    <span class="config-timestamp">${this.formatTimestamp(config.timestamp)}</span>
                </div>
                <div class="config-description">${config.description}</div>
                <div class="config-details">
                    <span class="config-stat">${config.filterState.activeFilters.length} filters</span>
                    ${config.searchState.query ? `<span class="config-stat">Search: "${config.searchState.query}"</span>` : ''}
                </div>
                <div class="config-actions">
                    <button class="config-load-btn" data-name="${config.name}">Load</button>
                    <button class="config-delete-btn" data-name="${config.name}">Delete</button>
                </div>
            </div>
        `).join('');
        
        // Add event listeners
        configsList.querySelectorAll('.config-load-btn').forEach(button => {
            button.addEventListener('click', () => {
                this.loadConfiguration(button.dataset.name);
                this.savedFiltersPanel.style.display = 'none';
            });
        });
        
        configsList.querySelectorAll('.config-delete-btn').forEach(button => {
            button.addEventListener('click', () => {
                if (confirm(`Delete configuration "${button.dataset.name}"?`)) {
                    this.deleteConfiguration(button.dataset.name);
                }
            });
        });
    }
    
    /**
     * Load default filter presets
     */
    loadDefaultPresets() {
        const presets = [
            {
                name: 'High Degree Nodes',
                description: 'Nodes with above-average degree',
                filterState: {
                    activeFilters: [{
                        attribute: 'degree',
                        operator: 'greater_than',
                        value: 3,
                        label: 'degree > 3'
                    }],
                    combination: 'AND'
                },
                searchState: { query: '', searchType: 'global', results: [] }
            },
            {
                name: 'Engineering Team',
                description: 'All engineering department members',
                filterState: {
                    activeFilters: [{
                        attribute: 'department',
                        operator: 'equals',
                        value: 'Engineering',
                        label: 'department = Engineering'
                    }],
                    combination: 'AND'
                },
                searchState: { query: '', searchType: 'global', results: [] }
            }
        ];
        
        presets.forEach(preset => {
            this.configurationPresets.set(preset.name, preset);
        });
        
        console.log('✅ Default presets loaded:', presets.length);
    }
    
    // Helper methods
    
    formatActionType(actionType) {
        const labels = {
            'filter_added': '🔍 Filter Added',
            'filter_removed': '❌ Filter Removed',
            'filters_cleared': '🧹 Filters Cleared',
            'search_performed': '🔍 Search',
            'configuration_loaded': '📂 Config Loaded',
            'undo': '↶ Undo',
            'redo': '↷ Redo'
        };
        
        return labels[actionType] || actionType;
    }
    
    formatActionDetails(action) {
        switch (action.type) {
            case 'filter_added':
                return `Added: ${action.data.label || 'New filter'}`;
            case 'filter_removed':
                return `Removed: ${action.data.label || 'Filter'}`;
            case 'search_performed':
                return `Query: "${action.data.query}" (${action.data.results?.length || 0} results)`;
            case 'configuration_loaded':
                return `Loaded: ${action.data.name}`;
            default:
                return JSON.stringify(action.data).substring(0, 100);
        }
    }
    
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
        
        return date.toLocaleDateString();
    }
    
    generateSessionId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }
    
    /**
     * Save history data to localStorage
     */
    saveToStorage() {
        try {
            const historyData = {
                actionHistory: this.actionHistory,
                savedConfigurations: this.savedConfigurations,
                sessionId: this.sessionId,
                timestamp: Date.now()
            };
            
            localStorage.setItem('groggy_filter_history', JSON.stringify(historyData));
            console.log('✅ History saved to localStorage');
            
        } catch (error) {
            console.error('❌ Failed to save history to localStorage:', error);
        }
    }
    
    /**
     * Load history data from localStorage
     */
    loadFromStorage() {
        try {
            const savedData = localStorage.getItem('groggy_filter_history');
            if (!savedData) {
                console.log('ℹ️  No saved history found in localStorage');
                return false;
            }
            
            const historyData = JSON.parse(savedData);
            
            // Validate and restore data
            if (historyData.actionHistory && Array.isArray(historyData.actionHistory)) {
                this.actionHistory = historyData.actionHistory;
                this.currentHistoryIndex = this.actionHistory.length - 1;
            }
            
            if (historyData.savedConfigurations) {
                this.savedConfigurations = new Map(Object.entries(historyData.savedConfigurations));
            }
            
            console.log(`✅ Loaded ${this.actionHistory.length} actions from localStorage`);
            return true;
            
        } catch (error) {
            console.error('❌ Failed to load history from localStorage:', error);
            return false;
        }
    }
    
    getVisibleNodeCount() {
        // This would get the actual count from the graph renderer
        return 0; // Placeholder
    }
    
    // Storage methods
    
    saveSessionData() {
        try {
            localStorage.setItem('groggy_session_data', JSON.stringify(this.sessionData));
            console.log('💾 Session data saved');
        } catch (error) {
            console.warn('⚠️  Could not save session data:', error);
        }
    }
    
    loadSessionData() {
        try {
            const saved = localStorage.getItem('groggy_session_data');
            if (saved) {
                const data = JSON.parse(saved);
                // Only restore if from recent session (within 24 hours)
                if (Date.now() - data.startTime < 24 * 60 * 60 * 1000) {
                    this.sessionData = data;
                    console.log('📂 Session data loaded');
                }
            }
        } catch (error) {
            console.warn('⚠️  Could not load session data:', error);
        }
    }
    
    saveSavedConfigurations() {
        try {
            const configs = Array.from(this.savedConfigurations.entries());
            localStorage.setItem('groggy_saved_configurations', JSON.stringify(configs));
        } catch (error) {
            console.warn('⚠️  Could not save configurations:', error);
        }
    }
    
    loadSavedConfigurations() {
        try {
            const saved = localStorage.getItem('groggy_saved_configurations');
            if (saved) {
                const configs = JSON.parse(saved);
                this.savedConfigurations = new Map(configs);
                console.log('📂 Loaded saved configurations:', configs.length);
            }
        } catch (error) {
            console.warn('⚠️  Could not load saved configurations:', error);
            this.savedConfigurations = new Map();
        }
    }
    
    // Analytics and metrics
    
    updateAnalytics(actionType, actionData) {
        this.usageAnalytics.totalActions++;
        
        // Track filter usage
        if (actionType === 'filter_added' && actionData.attribute) {
            const count = this.usageAnalytics.mostUsedFilters.get(actionData.attribute) || 0;
            this.usageAnalytics.mostUsedFilters.set(actionData.attribute, count + 1);
        }
        
        // Track search patterns
        if (actionType === 'search_performed' && actionData.query) {
            const count = this.usageAnalytics.searchPatterns.get(actionData.query) || 0;
            this.usageAnalytics.searchPatterns.set(actionData.query, count + 1);
        }
        
        // Update session duration
        this.usageAnalytics.sessionDuration = Date.now() - this.sessionData.startTime;
    }
    
    updateAnalyticsUI() {
        const analyticsDiv = this.historyPanel?.querySelector('#history-analytics');
        if (!analyticsDiv) return;
        
        analyticsDiv.innerHTML = `
            <div class="analytics-summary">
                <h4>Usage Analytics</h4>
                <div class="analytics-stats">
                    <div class="stat-item">
                        <span class="stat-label">Total Actions:</span>
                        <span class="stat-value">${this.usageAnalytics.totalActions}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Session Duration:</span>
                        <span class="stat-value">${Math.round(this.usageAnalytics.sessionDuration / 60000)}m</span>
                    </div>
                </div>
            </div>
            
            <div class="most-used-filters">
                <h4>Most Used Filters</h4>
                ${Array.from(this.usageAnalytics.mostUsedFilters.entries())
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([filter, count]) => `
                        <div class="usage-item">
                            <span class="usage-filter">${filter}</span>
                            <span class="usage-count">${count}</span>
                        </div>
                    `).join('')}
            </div>
            
            <div class="search-patterns">
                <h4>Common Searches</h4>
                ${Array.from(this.usageAnalytics.searchPatterns.entries())
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([query, count]) => `
                        <div class="usage-item">
                            <span class="usage-query">"${query}"</span>
                            <span class="usage-count">${count}</span>
                        </div>
                    `).join('')}
            </div>
        `;
    }
    
    updateSessionUI() {
        const sessionDiv = this.historyPanel?.querySelector('#session-info');
        if (!sessionDiv) return;
        
        sessionDiv.innerHTML = `
            <div class="session-details">
                <h4>Current Session</h4>
                <div class="session-stats">
                    <div class="stat-item">
                        <span class="stat-label">Session ID:</span>
                        <span class="stat-value">${this.sessionId}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Start Time:</span>
                        <span class="stat-value">${new Date(this.sessionData.startTime).toLocaleString()}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Actions:</span>
                        <span class="stat-value">${this.sessionData.actions.length}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Additional utility methods for the class
    
    switchHistoryTab(tabName) {
        // Hide all tab contents
        this.historyPanel.querySelectorAll('.history-tab-content').forEach(content => {
            content.style.display = 'none';
        });
        
        // Remove active class from all tabs
        this.historyPanel.querySelectorAll('.history-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Show selected tab content
        const selectedContent = this.historyPanel.querySelector(`#history-${tabName}-tab`);
        if (selectedContent) {
            selectedContent.style.display = 'block';
        }
        
        // Add active class to selected tab
        const selectedTab = this.historyPanel.querySelector(`[data-tab="${tabName}"]`);
        if (selectedTab) {
            selectedTab.classList.add('active');
        }
    }
    
    restoreToHistoryIndex(index) {
        if (index < 0 || index >= this.actionHistory.length) return;
        
        this.currentHistoryIndex = index;
        const targetAction = this.actionHistory[index];
        
        this.restoreState(targetAction.filterState, targetAction.searchState);
        this.updateHistoryUI();
    }
    
    recordUndoRedoAction(actionType, originalAction) {
        // Don't record undo/redo in main history to avoid infinite loops
        console.log(`📝 ${actionType} performed on: ${originalAction.type}`);
    }
    
    clearHistory() {
        if (confirm('Clear all history? This cannot be undone.')) {
            this.actionHistory = [];
            this.currentHistoryIndex = -1;
            this.updateHistoryUI();
            console.log('🧹 History cleared');
        }
    }
    
    exportHistory() {
        const exportData = {
            version: '1.0',
            timestamp: Date.now(),
            sessionId: this.sessionId,
            history: this.actionHistory,
            analytics: {
                totalActions: this.usageAnalytics.totalActions,
                sessionDuration: this.usageAnalytics.sessionDuration,
                mostUsedFilters: Array.from(this.usageAnalytics.mostUsedFilters.entries()),
                searchPatterns: Array.from(this.usageAnalytics.searchPatterns.entries())
            }
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `groggy_history_${this.sessionId}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    promptSaveCurrentConfiguration() {
        const name = prompt('Enter a name for this configuration:');
        if (name && name.trim()) {
            const description = prompt('Enter a description (optional):') || '';
            this.saveConfiguration(name.trim(), description);
        }
    }
    
    promptImportConfigurations() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        
        input.onchange = (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const data = JSON.parse(e.target.result);
                        this.importConfigurations(data);
                    } catch (error) {
                        alert('Error importing configurations: ' + error.message);
                    }
                };
                reader.readAsText(file);
            }
        };
        
        input.click();
    }
    
    importConfigurations(data) {
        // Import configurations from exported data
        if (data.savedConfigurations) {
            data.savedConfigurations.forEach(config => {
                this.savedConfigurations.set(config.name, config);
            });
        }
        
        this.updateSavedConfigurationsUI();
        this.saveSavedConfigurations();
        
        console.log('📥 Configurations imported successfully');
    }
    
    exportSavedConfigurations() {
        const exportData = {
            version: '1.0',
            timestamp: Date.now(),
            savedConfigurations: Array.from(this.savedConfigurations.values())
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `groggy_saved_configurations_${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    /**
     * Get comprehensive history and configuration data for analytics
     */
    getHistoryData() {
        return {
            sessionId: this.sessionId,
            history: {
                actions: this.actionHistory,
                currentIndex: this.currentHistoryIndex,
                totalActions: this.usageAnalytics.totalActions
            },
            savedConfigurations: Array.from(this.savedConfigurations.values()),
            analytics: this.usageAnalytics,
            session: this.sessionData
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FilterHistoryManager;
}