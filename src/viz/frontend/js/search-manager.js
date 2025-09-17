/**
 * Phase 8: Filtering & Search - SearchManager Implementation
 * 
 * Real-time search functionality connecting to the backend search system
 * implemented in Phase 7. Provides instant search with highlighting,
 * search suggestions, and search history.
 * 
 * Features:
 * - Real-time search as you type
 * - Search highlighting and visual feedback
 * - Search suggestions and autocomplete
 * - Search history and saved searches
 * - Multiple search types (nodes, edges, attributes, global)
 * - Advanced search with filters
 * - Search result navigation
 */

class SearchManager {
    constructor(graphRenderer, websocketClient, filterManager) {
        this.graphRenderer = graphRenderer;
        this.websocketClient = websocketClient;
        this.filterManager = filterManager;
        
        // Search UI elements
        this.searchInput = document.getElementById('node-search');
        this.searchResults = null; // Will be created dynamically
        this.searchHighlights = new Map();
        
        // Search state
        this.currentQuery = '';
        this.currentResults = [];
        this.selectedResultIndex = -1;
        this.searchHistory = [];
        this.searchSuggestions = [];
        
        // Search configuration
        this.searchTypes = {
            'global': 'Global Search',
            'node': 'Nodes Only',
            'edge': 'Edges Only',
            'attribute': 'Attributes Only'
        };
        this.currentSearchType = 'global';
        
        // Performance optimization
        this.searchDebounceTimer = null;
        this.searchCache = new Map();
        this.minQueryLength = 2;
        
        // Search result tracking
        this.lastSearchTime = 0;
        this.searchMetrics = {
            totalSearches: 0,
            averageResultCount: 0,
            averageResponseTime: 0
        };
        
        this.init();
    }
    
    init() {
        console.log('🔍 Initializing SearchManager for Phase 8');
        
        this.setupSearchUI();
        this.setupEventListeners();
        this.setupWebSocketHandlers();
        this.loadSearchHistory();
    }
    
    /**
     * Set up the search UI components
     */
    setupSearchUI() {
        // Enhance search input with advanced features
        const searchContainer = this.searchInput.parentElement;
        searchContainer.classList.add('search-container');
        
        // Add search type selector
        const searchTypeSelect = document.createElement('select');
        searchTypeSelect.id = 'search-type-select';
        searchTypeSelect.className = 'search-type-select';
        
        Object.entries(this.searchTypes).forEach(([value, label]) => {
            const option = document.createElement('option');
            option.value = value;
            option.textContent = label;
            searchTypeSelect.appendChild(option);
        });
        
        searchContainer.insertBefore(searchTypeSelect, this.searchInput);
        
        // Add search results container
        this.searchResults = document.createElement('div');
        this.searchResults.id = 'search-results';
        this.searchResults.className = 'search-results';
        this.searchResults.style.display = 'none';
        searchContainer.appendChild(this.searchResults);
        
        // Add search suggestions container
        const suggestionsContainer = document.createElement('div');
        suggestionsContainer.id = 'search-suggestions';
        suggestionsContainer.className = 'search-suggestions';
        suggestionsContainer.style.display = 'none';
        searchContainer.appendChild(suggestionsContainer);
        
        // Add search status indicator
        const searchStatus = document.createElement('div');
        searchStatus.id = 'search-status';
        searchStatus.className = 'search-status';
        searchContainer.appendChild(searchStatus);
        
        // Add advanced search toggle
        const advancedToggle = document.createElement('button');
        advancedToggle.id = 'advanced-search-toggle';
        advancedToggle.className = 'control-button secondary small';
        advancedToggle.textContent = 'Advanced';
        advancedToggle.title = 'Toggle advanced search options';
        searchContainer.appendChild(advancedToggle);
        
        console.log('✅ Search UI components created');
    }
    
    /**
     * Set up event listeners for search functionality
     */
    setupEventListeners() {
        // Real-time search as user types
        this.searchInput.addEventListener('input', (event) => {
            this.handleSearchInput(event.target.value);
        });
        
        // Search input focus and blur
        this.searchInput.addEventListener('focus', () => {
            this.showSearchSuggestions();
        });
        
        this.searchInput.addEventListener('blur', () => {
            // Delay hiding to allow clicking suggestions
            setTimeout(() => this.hideSearchSuggestions(), 150);
        });
        
        // Keyboard navigation
        this.searchInput.addEventListener('keydown', (event) => {
            this.handleSearchKeydown(event);
        });
        
        // Search type change
        const searchTypeSelect = document.getElementById('search-type-select');
        searchTypeSelect.addEventListener('change', (event) => {
            this.currentSearchType = event.target.value;
            if (this.currentQuery) {
                this.performSearch(this.currentQuery);
            }
        });
        
        // Advanced search toggle
        const advancedToggle = document.getElementById('advanced-search-toggle');
        advancedToggle.addEventListener('click', () => {
            this.toggleAdvancedSearch();
        });
        
        // Clear search shortcut (Escape key)
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && this.currentQuery) {
                this.clearSearch();
            }
        });
        
        console.log('✅ Search event listeners set up');
    }
    
    /**
     * Set up WebSocket message handlers for search responses
     */
    setupWebSocketHandlers() {
        this.websocketClient.addMessageHandler('SearchResponse', (data) => {
            this.handleSearchResponse(data);
        });
        
        // Generic message handler that checks message.type
        this.websocketClient.addEventListener('message', (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'SearchResponse') {
                this.handleSearchResponse(message.data);
            }
        });
        
        console.log('✅ Search WebSocket handlers set up');
    }
    
    /**
     * Handle search input with debouncing
     */
    handleSearchInput(query) {
        // Clear previous debounce timer
        if (this.searchDebounceTimer) {
            clearTimeout(this.searchDebounceTimer);
        }
        
        // Update current query
        this.currentQuery = query.trim();
        
        // Show loading state immediately
        this.updateSearchStatus('typing');
        
        // Debounce search to avoid too many requests
        this.searchDebounceTimer = setTimeout(() => {
            if (this.currentQuery.length >= this.minQueryLength) {
                this.performSearch(this.currentQuery);
            } else if (this.currentQuery.length === 0) {
                this.clearSearch();
            } else {
                this.showSearchSuggestions();
            }
        }, 300); // 300ms delay
    }
    
    /**
     * Handle keyboard navigation in search
     */
    handleSearchKeydown(event) {
        switch (event.key) {
            case 'ArrowDown':
                event.preventDefault();
                this.navigateResults('down');
                break;
            case 'ArrowUp':
                event.preventDefault();
                this.navigateResults('up');
                break;
            case 'Enter':
                event.preventDefault();
                this.selectCurrentResult();
                break;
            case 'Escape':
                this.clearSearch();
                break;
        }
    }
    
    /**
     * Perform search via WebSocket
     */
    async performSearch(query) {
        if (!query || query.length < this.minQueryLength) return;
        
        console.log(`🔍 Searching for: "${query}" (type: ${this.currentSearchType})`);
        
        // Check cache first
        const cacheKey = `${this.currentSearchType}:${query}`;
        if (this.searchCache.has(cacheKey)) {
            console.log('⚡ Using cached search results');
            this.handleSearchResponse(this.searchCache.get(cacheKey));
            return;
        }
        
        // Update search status
        this.updateSearchStatus('searching');
        
        // Record search start time
        this.lastSearchTime = performance.now();
        
        // Create search filters based on current filter state
        const filters = this.createSearchFilters();
        
        // Send search request via WebSocket
        const searchRequest = {
            type: 'SearchRequest',
            query: query,
            search_type: this.capitalizeSearchType(this.currentSearchType),
            filters: filters
        };
        
        try {
            this.websocketClient.send(searchRequest);
            this.searchMetrics.totalSearches++;
        } catch (error) {
            console.error('❌ Error sending search request:', error);
            this.updateSearchStatus('error');
        }
    }
    
    /**
     * Handle search response from WebSocket
     */
    handleSearchResponse(data) {
        const responseTime = performance.now() - this.lastSearchTime;
        console.log(`✅ Search results received: ${data.total_matches} matches in ${data.query_time_ms}ms`);
        
        // Update metrics
        this.searchMetrics.averageResultCount = 
            (this.searchMetrics.averageResultCount + data.total_matches) / 2;
        this.searchMetrics.averageResponseTime = 
            (this.searchMetrics.averageResponseTime + responseTime) / 2;
        
        // Store results
        this.currentResults = data.results || [];
        this.selectedResultIndex = -1;
        
        // Cache results
        const cacheKey = `${this.currentSearchType}:${this.currentQuery}`;
        this.searchCache.set(cacheKey, data);
        
        // Update UI
        this.displaySearchResults(data);
        this.updateSearchHighlights();
        this.updateSearchStatus('completed', data.total_matches);
        
        // Add to search history
        this.addToSearchHistory(this.currentQuery, data.total_matches);
    }
    
    /**
     * Display search results in the UI
     */
    displaySearchResults(data) {
        if (data.total_matches === 0) {
            this.searchResults.innerHTML = `
                <div class="search-no-results">
                    <p>No results found for "${this.currentQuery}"</p>
                    <div class="search-suggestions">
                        <p>Try:</p>
                        <ul>
                            <li>Checking your spelling</li>
                            <li>Using different keywords</li>
                            <li>Changing the search type</li>
                            <li>Clearing active filters</li>
                        </ul>
                    </div>
                </div>
            `;
            this.searchResults.style.display = 'block';
            return;
        }
        
        // Group results by type
        const groupedResults = this.groupResultsByType(data.results);
        
        // Render results
        this.searchResults.innerHTML = `
            <div class="search-results-header">
                <span class="results-count">${data.total_matches} results</span>
                <span class="search-time">${data.query_time_ms}ms</span>
                <button class="clear-search-btn" title="Clear search">&times;</button>
            </div>
            <div class="search-results-content">
                ${Object.entries(groupedResults).map(([type, results]) => 
                    this.renderResultGroup(type, results)
                ).join('')}
            </div>
        `;
        
        this.searchResults.style.display = 'block';
        
        // Add event listeners to results
        this.setupSearchResultListeners();
    }
    
    /**
     * Group search results by type
     */
    groupResultsByType(results) {
        const grouped = {};
        
        results.forEach(result => {
            const type = result.result_type || 'Unknown';
            if (!grouped[type]) {
                grouped[type] = [];
            }
            grouped[type].push(result);
        });
        
        return grouped;
    }
    
    /**
     * Render a group of results by type
     */
    renderResultGroup(type, results) {
        const typeLabels = {
            'Node': '🔵 Nodes',
            'Edge': '🔗 Edges',
            'Attribute': '🏷️ Attributes'
        };
        
        const label = typeLabels[type] || type;
        
        return `
            <div class="result-group">
                <h4 class="result-group-header">${label} (${results.length})</h4>
                <div class="result-group-items">
                    ${results.map((result, index) => this.renderSearchResult(result, index)).join('')}
                </div>
            </div>
        `;
    }
    
    /**
     * Render individual search result
     */
    renderSearchResult(result, index) {
        const matchedFields = result.matched_fields || [];
        const relevanceWidth = Math.max(20, result.relevance_score * 100);
        
        return `
            <div class="search-result" data-result-index="${index}" data-element-id="${result.id}">
                <div class="result-header">
                    <span class="result-title">${this.highlightText(result.title, this.currentQuery)}</span>
                    <span class="relevance-bar" style="width: ${relevanceWidth}%"></span>
                </div>
                ${result.subtitle ? `<div class="result-subtitle">${result.subtitle}</div>` : ''}
                ${matchedFields.length > 0 ? `
                    <div class="result-matches">
                        ${matchedFields.slice(0, 3).map(field => `
                            <div class="result-match">
                                <strong>${field.field_name}:</strong> 
                                ${this.highlightText(field.context, this.currentQuery)}
                            </div>
                        `).join('')}
                        ${matchedFields.length > 3 ? `<div class="result-more">+${matchedFields.length - 3} more</div>` : ''}
                    </div>
                ` : ''}
                <div class="result-actions">
                    <button class="result-action" data-action="focus" data-element-id="${result.id}">Focus</button>
                    <button class="result-action" data-action="select" data-element-id="${result.id}">Select</button>
                    <button class="result-action" data-action="details" data-element-id="${result.id}">Details</button>
                </div>
            </div>
        `;
    }
    
    /**
     * Set up event listeners for search result interactions
     */
    setupSearchResultListeners() {
        // Clear search button
        const clearButton = this.searchResults.querySelector('.clear-search-btn');
        if (clearButton) {
            clearButton.addEventListener('click', () => this.clearSearch());
        }
        
        // Result item clicks
        this.searchResults.querySelectorAll('.search-result').forEach((resultElement, index) => {
            resultElement.addEventListener('click', () => {
                this.selectResult(index);
            });
            
            resultElement.addEventListener('mouseenter', () => {
                this.highlightResult(index);
            });
        });
        
        // Result action buttons
        this.searchResults.querySelectorAll('.result-action').forEach(button => {
            button.addEventListener('click', (event) => {
                event.stopPropagation();
                const action = button.dataset.action;
                const elementId = button.dataset.elementId;
                this.performResultAction(action, elementId);
            });
        });
    }
    
    /**
     * Perform action on search result
     */
    performResultAction(action, elementId) {
        console.log(`🎯 Performing action "${action}" on element: ${elementId}`);
        
        switch (action) {
            case 'focus':
                this.graphRenderer.focusOnElement(elementId);
                break;
            case 'select':
                this.graphRenderer.selectElement(elementId);
                break;
            case 'details':
                this.graphRenderer.showElementDetails(elementId);
                break;
        }
    }
    
    /**
     * Navigate search results with keyboard
     */
    navigateResults(direction) {
        if (this.currentResults.length === 0) return;
        
        const previousIndex = this.selectedResultIndex;
        
        if (direction === 'down') {
            this.selectedResultIndex = Math.min(
                this.selectedResultIndex + 1,
                this.currentResults.length - 1
            );
        } else if (direction === 'up') {
            this.selectedResultIndex = Math.max(this.selectedResultIndex - 1, -1);
        }
        
        // Update visual selection
        this.updateResultSelection(previousIndex, this.selectedResultIndex);
    }
    
    /**
     * Update visual selection of search results
     */
    updateResultSelection(previousIndex, currentIndex) {
        // Remove previous selection
        if (previousIndex >= 0) {
            const prevElement = this.searchResults.querySelector(`[data-result-index="${previousIndex}"]`);
            if (prevElement) {
                prevElement.classList.remove('selected');
            }
        }
        
        // Add current selection
        if (currentIndex >= 0) {
            const currentElement = this.searchResults.querySelector(`[data-result-index="${currentIndex}"]`);
            if (currentElement) {
                currentElement.classList.add('selected');
                currentElement.scrollIntoView({ block: 'nearest' });
            }
        }
    }
    
    /**
     * Select the currently highlighted result
     */
    selectCurrentResult() {
        if (this.selectedResultIndex >= 0 && this.currentResults[this.selectedResultIndex]) {
            this.selectResult(this.selectedResultIndex);
        }
    }
    
    /**
     * Select a specific search result
     */
    selectResult(index) {
        const result = this.currentResults[index];
        if (!result) return;
        
        console.log('✅ Selected search result:', result.title);
        
        // Focus on the element in the graph
        this.graphRenderer.focusOnElement(result.id);
        
        // Update search input to show selected result
        this.searchInput.value = result.title;
        this.currentQuery = result.title;
        
        // Hide search results
        this.hideSearchResults();
    }
    
    /**
     * Update search highlights on the graph
     */
    updateSearchHighlights() {
        // Clear existing highlights
        this.clearSearchHighlights();
        
        // Add new highlights
        this.currentResults.forEach(result => {
            this.addSearchHighlight(result.id, result.result_type);
        });
    }
    
    /**
     * Add search highlight to an element
     */
    addSearchHighlight(elementId, elementType) {
        const highlightData = {
            elementId,
            elementType: elementType.toLowerCase(),
            highlightType: 'search',
            timestamp: Date.now()
        };
        
        this.searchHighlights.set(elementId, highlightData);
        this.graphRenderer.addHighlight(highlightData);
    }
    
    /**
     * Clear all search highlights
     */
    clearSearchHighlights() {
        this.searchHighlights.forEach(highlight => {
            this.graphRenderer.removeHighlight(highlight.elementId);
        });
        this.searchHighlights.clear();
    }
    
    /**
     * Highlight text with search query
     */
    highlightText(text, query) {
        if (!query || !text) return text;
        
        const regex = new RegExp(`(${this.escapeRegex(query)})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }
    
    /**
     * Escape regex special characters
     */
    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
    }
    
    /**
     * Update search status indicator
     */
    updateSearchStatus(status, resultCount = 0) {
        const statusElement = document.getElementById('search-status');
        if (!statusElement) return;
        
        const statusMessages = {
            'typing': '⌨️ Typing...',
            'searching': '🔍 Searching...',
            'completed': `✅ ${resultCount} results`,
            'error': '❌ Search error',
            'cleared': ''
        };
        
        statusElement.textContent = statusMessages[status] || '';
        statusElement.className = `search-status ${status}`;
    }
    
    /**
     * Show search suggestions
     */
    showSearchSuggestions() {
        const suggestionsContainer = document.getElementById('search-suggestions');
        if (!suggestionsContainer) return;
        
        // Get suggestions from history and available attributes
        const suggestions = this.generateSearchSuggestions();
        
        if (suggestions.length === 0) {
            suggestionsContainer.style.display = 'none';
            return;
        }
        
        suggestionsContainer.innerHTML = suggestions.map(suggestion => `
            <div class="search-suggestion" data-suggestion="${suggestion.text}">
                <span class="suggestion-text">${suggestion.text}</span>
                <span class="suggestion-type">${suggestion.type}</span>
            </div>
        `).join('');
        
        // Add click handlers
        suggestionsContainer.querySelectorAll('.search-suggestion').forEach(element => {
            element.addEventListener('click', () => {
                this.searchInput.value = element.dataset.suggestion;
                this.handleSearchInput(element.dataset.suggestion);
                this.hideSearchSuggestions();
            });
        });
        
        suggestionsContainer.style.display = 'block';
    }
    
    /**
     * Hide search suggestions
     */
    hideSearchSuggestions() {
        const suggestionsContainer = document.getElementById('search-suggestions');
        if (suggestionsContainer) {
            suggestionsContainer.style.display = 'none';
        }
    }
    
    /**
     * Generate search suggestions
     */
    generateSearchSuggestions() {
        const suggestions = [];
        
        // Recent searches
        this.searchHistory.slice(-5).forEach(entry => {
            suggestions.push({
                text: entry.query,
                type: '🕐 Recent',
                relevance: entry.resultCount > 0 ? 1 : 0.5
            });
        });
        
        // Common search patterns
        const commonPatterns = [
            { text: 'high degree', type: '🎯 Pattern' },
            { text: 'central nodes', type: '🎯 Pattern' },
            { text: 'isolated', type: '🎯 Pattern' }
        ];
        
        suggestions.push(...commonPatterns);
        
        return suggestions.sort((a, b) => (b.relevance || 0) - (a.relevance || 0));
    }
    
    /**
     * Clear search and reset UI
     */
    clearSearch() {
        console.log('🧹 Clearing search');
        
        this.searchInput.value = '';
        this.currentQuery = '';
        this.currentResults = [];
        this.selectedResultIndex = -1;
        
        this.hideSearchResults();
        this.hideSearchSuggestions();
        this.clearSearchHighlights();
        this.updateSearchStatus('cleared');
        
        // Clear search debounce timer
        if (this.searchDebounceTimer) {
            clearTimeout(this.searchDebounceTimer);
        }
    }
    
    /**
     * Hide search results
     */
    hideSearchResults() {
        if (this.searchResults) {
            this.searchResults.style.display = 'none';
        }
    }
    
    /**
     * Toggle advanced search options
     */
    toggleAdvancedSearch() {
        console.log('🔧 Toggling advanced search');
        
        // Create or toggle advanced search panel
        let advancedPanel = document.getElementById('advanced-search-panel');
        
        if (advancedPanel) {
            // Toggle visibility
            advancedPanel.style.display = 
                advancedPanel.style.display === 'none' ? 'block' : 'none';
        } else {
            // Create advanced search panel
            this.createAdvancedSearchPanel();
        }
    }
    
    /**
     * Create advanced search panel
     */
    createAdvancedSearchPanel() {
        const panel = document.createElement('div');
        panel.id = 'advanced-search-panel';
        panel.className = 'advanced-search-panel';
        
        panel.innerHTML = `
            <h4>Advanced Search</h4>
            <div class="advanced-search-options">
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="search-case-sensitive"> Case sensitive
                    </label>
                </div>
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="search-whole-words"> Whole words only
                    </label>
                </div>
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="search-regex"> Use regular expressions
                    </label>
                </div>
                <div class="form-group">
                    <label>Maximum results:</label>
                    <input type="number" id="search-max-results" value="50" min="10" max="500">
                </div>
            </div>
        `;
        
        this.searchInput.parentElement.appendChild(panel);
    }
    
    /**
     * Create search filters from current filter state
     */
    createSearchFilters() {
        // Get current filter state from FilterManager
        const filterState = this.filterManager.getFilterState();
        
        return filterState.activeFilters.map(filter => ({
            field: filter.attribute,
            operator: filter.operator,
            value: filter.value
        }));
    }
    
    /**
     * Capitalize search type for WebSocket message
     */
    capitalizeSearchType(searchType) {
        const mapping = {
            'global': 'Global',
            'node': 'Node',
            'edge': 'Edge',
            'attribute': 'Attribute'
        };
        
        return mapping[searchType] || 'Global';
    }
    
    /**
     * Add search to history
     */
    addToSearchHistory(query, resultCount) {
        const historyEntry = {
            query,
            resultCount,
            searchType: this.currentSearchType,
            timestamp: Date.now()
        };
        
        // Remove duplicate
        this.searchHistory = this.searchHistory.filter(entry => entry.query !== query);
        
        // Add to beginning
        this.searchHistory.unshift(historyEntry);
        
        // Limit history size
        if (this.searchHistory.length > 20) {
            this.searchHistory.pop();
        }
        
        // Save to localStorage
        this.saveSearchHistory();
    }
    
    /**
     * Save search history to localStorage
     */
    saveSearchHistory() {
        try {
            localStorage.setItem('groggy_search_history', JSON.stringify(this.searchHistory));
        } catch (error) {
            console.warn('⚠️  Could not save search history:', error);
        }
    }
    
    /**
     * Load search history from localStorage
     */
    loadSearchHistory() {
        try {
            const saved = localStorage.getItem('groggy_search_history');
            if (saved) {
                this.searchHistory = JSON.parse(saved);
                console.log('📂 Loaded search history:', this.searchHistory.length, 'entries');
            }
        } catch (error) {
            console.warn('⚠️  Could not load search history:', error);
            this.searchHistory = [];
        }
    }
    
    /**
     * Get search metrics for analytics
     */
    getSearchMetrics() {
        return {
            ...this.searchMetrics,
            historyLength: this.searchHistory.length,
            cacheSize: this.searchCache.size,
            currentQuery: this.currentQuery,
            resultCount: this.currentResults.length
        };
    }
    
    /**
     * Export search configuration and history
     */
    exportSearchData() {
        return {
            version: '1.0',
            timestamp: Date.now(),
            history: this.searchHistory,
            metrics: this.searchMetrics,
            configuration: {
                searchType: this.currentSearchType,
                minQueryLength: this.minQueryLength
            }
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SearchManager;
}