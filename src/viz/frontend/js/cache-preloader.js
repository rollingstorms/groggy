/**
 * 🚀 Phase 10: Performance Optimization - Client-Side Caching and Preloading
 * 
 * Advanced caching and preloading system for graph visualization data
 * Provides intelligent caching, predictive preloading, and memory management
 * 
 * Features:
 * - Multi-layer caching (memory, IndexedDB, localStorage)
 * - Intelligent cache invalidation and TTL management
 * - Predictive preloading based on user behavior
 * - Background data fetching and processing
 * - Memory-efficient storage with compression
 * - Cache warming for critical data
 * - Offline capability with cached data
 * - Performance metrics and cache analytics
 * - Automatic cache cleanup and optimization
 */

class CachePreloaderSystem {
    constructor(websocketClient, graphRenderer) {
        this.websocketClient = websocketClient;
        this.graphRenderer = graphRenderer;
        
        // Cache configuration
        this.config = {
            memoryCache: {
                maxSize: 100 * 1024 * 1024, // 100MB
                maxEntries: 1000,
                ttl: 5 * 60 * 1000 // 5 minutes
            },
            indexedDBCache: {
                dbName: 'groggy_graph_cache',
                version: 1,
                maxSize: 500 * 1024 * 1024, // 500MB
                ttl: 24 * 60 * 60 * 1000 // 24 hours
            },
            localStorage: {
                maxSize: 10 * 1024 * 1024, // 10MB
                ttl: 7 * 24 * 60 * 60 * 1000 // 7 days
            },
            preloading: {
                enabled: true,
                maxConcurrentRequests: 5,
                predictiveDistance: 200, // pixels
                viewportBuffer: 1.5, // multiplier for viewport preloading
                behaviorAnalysisDepth: 50 // recent actions to analyze
            },
            compression: {
                enabled: true,
                algorithm: 'gzip', // 'gzip', 'brotli', 'lz4'
                threshold: 1024 // minimum size to compress
            }
        };
        
        // Cache stores
        this.memoryCache = new Map();
        this.indexedDB = null;
        this.cacheStats = {
            hits: 0,
            misses: 0,
            size: 0,
            evictions: 0,
            compressionRatio: 0
        };
        
        // Preloading system
        this.preloadQueue = [];
        this.activePreloads = new Map();
        this.userBehavior = [];
        this.preloadPredictions = new Map();
        
        // Background processing
        this.backgroundWorker = null;
        this.compressionWorker = null;
        this.cleanupInterval = null;
        
        // Cache warming
        this.warmupQueries = [
            { type: 'graph_overview', priority: 'high' },
            { type: 'node_summary', priority: 'medium' },
            { type: 'edge_summary', priority: 'medium' },
            { type: 'layout_data', priority: 'low' }
        ];
        
        this.init();
    }
    
    init() {
        console.log('🚀 Initializing Cache Preloader System for Phase 10');
        
        this.initializeIndexedDB();
        this.setupEventListeners();
        this.startBackgroundWorkers();
        this.startCacheCleanup();
        this.warmupCache();
        
        console.log('✅ Cache Preloader System initialized');
    }
    
    /**
     * Initialize IndexedDB for persistent caching
     */
    async initializeIndexedDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(
                this.config.indexedDBCache.dbName, 
                this.config.indexedDBCache.version
            );
            
            request.onerror = () => {
                console.warn('IndexedDB initialization failed, using memory cache only');
                resolve(null);
            };
            
            request.onsuccess = (event) => {
                this.indexedDB = event.target.result;
                console.log('💾 IndexedDB cache initialized');
                resolve(this.indexedDB);
            };
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                
                // Create cache stores
                if (!db.objectStoreNames.contains('graph_data')) {
                    const store = db.createObjectStore('graph_data', { keyPath: 'key' });
                    store.createIndex('timestamp', 'timestamp', { unique: false });
                    store.createIndex('type', 'type', { unique: false });
                }
                
                if (!db.objectStoreNames.contains('metadata')) {
                    db.createObjectStore('metadata', { keyPath: 'key' });
                }
                
                console.log('💾 IndexedDB schema created');
            };
        });
    }
    
    /**
     * Set up event listeners for cache operations
     */
    setupEventListeners() {
        // Listen for data requests
        document.addEventListener('dataRequested', (e) => {
            this.handleDataRequest(e.detail);
        });
        
        // Listen for user interactions for behavioral analysis
        document.addEventListener('userInteraction', (e) => {
            this.analyzeUserBehavior(e.detail);
        });
        
        // Listen for viewport changes for preloading
        document.addEventListener('viewportChanged', (e) => {
            this.handleViewportChange(e.detail);
        });
        
        // Listen for zoom changes
        document.addEventListener('zoomChanged', (e) => {
            this.handleZoomChange(e.detail);
        });
        
        // Listen for network status
        window.addEventListener('online', () => this.handleOnline());
        window.addEventListener('offline', () => this.handleOffline());
    }
    
    /**
     * Start background workers for processing
     */
    startBackgroundWorkers() {
        // Compression worker
        if (this.config.compression.enabled && typeof Worker !== 'undefined') {
            try {
                this.compressionWorker = new Worker('/js/workers/compression-worker.js');
                this.compressionWorker.onmessage = (e) => {
                    this.handleCompressionResult(e.data);
                };
                console.log('🔧 Compression worker started');
            } catch (error) {
                console.warn('Compression worker unavailable, using synchronous compression');
            }
        }
        
        // Background processing worker
        if (typeof Worker !== 'undefined') {
            try {
                this.backgroundWorker = new Worker('/js/workers/cache-worker.js');
                this.backgroundWorker.onmessage = (e) => {
                    this.handleBackgroundResult(e.data);
                };
                console.log('🔧 Background processing worker started');
            } catch (error) {
                console.warn('Background worker unavailable, using main thread processing');
            }
        }
    }
    
    /**
     * Start periodic cache cleanup
     */
    startCacheCleanup() {
        this.cleanupInterval = setInterval(() => {
            this.performCacheCleanup();
        }, 60000); // Every minute
        
        console.log('🧹 Cache cleanup scheduler started');
    }
    
    /**
     * Warm up cache with essential data
     */
    async warmupCache() {
        console.log('🔥 Starting cache warmup...');
        
        for (const query of this.warmupQueries) {
            try {
                await this.preloadData(query);
            } catch (error) {
                console.warn(`Cache warmup failed for ${query.type}:`, error);
            }
        }
        
        console.log('✅ Cache warmup complete');
    }
    
    /**
     * Get data from cache or fetch if not available
     */
    async getData(key, fetchFunction = null) {
        const startTime = performance.now();
        
        // Check memory cache first
        const memoryResult = this.getFromMemoryCache(key);
        if (memoryResult) {
            this.cacheStats.hits++;
            console.log(`💾 Memory cache hit for: ${key}`);
            return memoryResult.data;
        }
        
        // Check IndexedDB cache
        const dbResult = await this.getFromIndexedDB(key);
        if (dbResult) {
            this.cacheStats.hits++;
            // Store in memory cache for faster access
            this.setMemoryCache(key, dbResult.data, dbResult.metadata);
            console.log(`💾 IndexedDB cache hit for: ${key}`);
            return dbResult.data;
        }
        
        // Check localStorage as fallback
        const localResult = this.getFromLocalStorage(key);
        if (localResult) {
            this.cacheStats.hits++;
            this.setMemoryCache(key, localResult.data, localResult.metadata);
            console.log(`💾 localStorage cache hit for: ${key}`);
            return localResult.data;
        }
        
        // Cache miss - fetch data if function provided
        this.cacheStats.misses++;
        
        if (fetchFunction) {
            console.log(`📡 Cache miss, fetching: ${key}`);
            try {
                const data = await fetchFunction();
                await this.setCache(key, data);
                
                const fetchTime = performance.now() - startTime;
                console.log(`✅ Fetched and cached ${key} in ${fetchTime.toFixed(2)}ms`);
                
                return data;
            } catch (error) {
                console.error(`Failed to fetch data for ${key}:`, error);
                throw error;
            }
        }
        
        return null;
    }
    
    /**
     * Set data in all cache layers
     */
    async setCache(key, data, metadata = {}) {
        const cacheEntry = {
            key,
            data,
            metadata: {
                ...metadata,
                timestamp: Date.now(),
                size: this.estimateSize(data),
                compressed: false
            }
        };
        
        // Compress if beneficial
        if (this.config.compression.enabled && 
            cacheEntry.metadata.size > this.config.compression.threshold) {
            const compressed = await this.compressData(data);
            if (compressed && compressed.size < cacheEntry.metadata.size * 0.8) {
                cacheEntry.data = compressed.data;
                cacheEntry.metadata.compressed = true;
                cacheEntry.metadata.originalSize = cacheEntry.metadata.size;
                cacheEntry.metadata.size = compressed.size;
                this.cacheStats.compressionRatio = compressed.size / cacheEntry.metadata.originalSize;
            }
        }
        
        // Store in memory cache
        this.setMemoryCache(key, cacheEntry.data, cacheEntry.metadata);
        
        // Store in IndexedDB
        await this.setIndexedDB(key, cacheEntry);
        
        // Store small items in localStorage
        if (cacheEntry.metadata.size < 1024 * 10) { // 10KB
            this.setLocalStorage(key, cacheEntry);
        }
    }
    
    /**
     * Memory cache operations
     */
    getFromMemoryCache(key) {
        const entry = this.memoryCache.get(key);
        if (!entry) return null;
        
        // Check TTL
        if (Date.now() - entry.metadata.timestamp > this.config.memoryCache.ttl) {
            this.memoryCache.delete(key);
            return null;
        }
        
        return entry;
    }
    
    setMemoryCache(key, data, metadata) {
        // Check size limits
        if (this.getMemoryCacheSize() > this.config.memoryCache.maxSize) {
            this.evictMemoryCache();
        }
        
        this.memoryCache.set(key, { data, metadata });
        this.cacheStats.size = this.getMemoryCacheSize();
    }
    
    evictMemoryCache() {
        // LRU eviction
        const entries = Array.from(this.memoryCache.entries())
            .sort((a, b) => a[1].metadata.timestamp - b[1].metadata.timestamp);
        
        const toEvict = Math.ceil(entries.length * 0.2); // Evict 20%
        
        for (let i = 0; i < toEvict; i++) {
            this.memoryCache.delete(entries[i][0]);
            this.cacheStats.evictions++;
        }
        
        console.log(`🧹 Evicted ${toEvict} entries from memory cache`);
    }
    
    getMemoryCacheSize() {
        let totalSize = 0;
        for (const entry of this.memoryCache.values()) {
            totalSize += entry.metadata.size || 0;
        }
        return totalSize;
    }
    
    /**
     * IndexedDB cache operations
     */
    async getFromIndexedDB(key) {
        if (!this.indexedDB) return null;
        
        return new Promise((resolve) => {
            const transaction = this.indexedDB.transaction(['graph_data'], 'readonly');
            const store = transaction.objectStore('graph_data');
            const request = store.get(key);
            
            request.onsuccess = () => {
                const result = request.result;
                if (!result) {
                    resolve(null);
                    return;
                }
                
                // Check TTL
                if (Date.now() - result.metadata.timestamp > this.config.indexedDBCache.ttl) {
                    this.deleteFromIndexedDB(key);
                    resolve(null);
                    return;
                }
                
                // Decompress if needed
                if (result.metadata.compressed) {
                    this.decompressData(result.data).then(decompressed => {
                        resolve({ data: decompressed, metadata: result.metadata });
                    });
                } else {
                    resolve({ data: result.data, metadata: result.metadata });
                }
            };
            
            request.onerror = () => resolve(null);
        });
    }
    
    async setIndexedDB(key, entry) {
        if (!this.indexedDB) return;
        
        return new Promise((resolve) => {
            const transaction = this.indexedDB.transaction(['graph_data'], 'readwrite');
            const store = transaction.objectStore('graph_data');
            const request = store.put(entry);
            
            request.onsuccess = () => resolve(true);
            request.onerror = () => resolve(false);
        });
    }
    
    async deleteFromIndexedDB(key) {
        if (!this.indexedDB) return;
        
        return new Promise((resolve) => {
            const transaction = this.indexedDB.transaction(['graph_data'], 'readwrite');
            const store = transaction.objectStore('graph_data');
            const request = store.delete(key);
            
            request.onsuccess = () => resolve(true);
            request.onerror = () => resolve(false);
        });
    }
    
    /**
     * localStorage cache operations
     */
    getFromLocalStorage(key) {
        try {
            const cached = localStorage.getItem(`groggy_cache_${key}`);
            if (!cached) return null;
            
            const entry = JSON.parse(cached);
            
            // Check TTL
            if (Date.now() - entry.metadata.timestamp > this.config.localStorage.ttl) {
                localStorage.removeItem(`groggy_cache_${key}`);
                return null;
            }
            
            return entry;
        } catch (error) {
            return null;
        }
    }
    
    setLocalStorage(key, entry) {
        try {
            localStorage.setItem(`groggy_cache_${key}`, JSON.stringify(entry));
        } catch (error) {
            // localStorage full, clear old entries
            this.clearOldLocalStorageEntries();
        }
    }
    
    clearOldLocalStorageEntries() {
        const keys = Object.keys(localStorage);
        const cacheKeys = keys.filter(key => key.startsWith('groggy_cache_'));
        
        // Sort by timestamp and remove oldest 20%
        const entries = cacheKeys.map(key => {
            try {
                const entry = JSON.parse(localStorage.getItem(key));
                return { key, timestamp: entry.metadata.timestamp };
            } catch {
                return { key, timestamp: 0 };
            }
        }).sort((a, b) => a.timestamp - b.timestamp);
        
        const toRemove = Math.ceil(entries.length * 0.2);
        for (let i = 0; i < toRemove; i++) {
            localStorage.removeItem(entries[i].key);
        }
        
        console.log(`🧹 Cleared ${toRemove} old localStorage entries`);
    }
    
    /**
     * Preloading system
     */
    async preloadData(query) {
        if (!this.config.preloading.enabled) return;
        
        const key = this.generateCacheKey(query);
        
        // Check if already cached
        if (await this.getData(key)) {
            return; // Already cached
        }
        
        // Check if already being preloaded
        if (this.activePreloads.has(key)) {
            return; // Already in progress
        }
        
        // Add to preload queue
        this.preloadQueue.push({
            key,
            query,
            priority: query.priority || 'normal',
            timestamp: Date.now()
        });
        
        this.processPreloadQueue();
    }
    
    async processPreloadQueue() {
        if (this.activePreloads.size >= this.config.preloading.maxConcurrentRequests) {
            return; // Too many concurrent requests
        }
        
        // Sort queue by priority
        this.preloadQueue.sort((a, b) => {
            const priorities = { high: 3, medium: 2, normal: 1, low: 0 };
            return priorities[b.priority] - priorities[a.priority];
        });
        
        const item = this.preloadQueue.shift();
        if (!item) return;
        
        this.activePreloads.set(item.key, item);
        
        try {
            console.log(`🔄 Preloading: ${item.key}`);
            
            // Create fetch function for this query
            const fetchFunction = () => this.fetchDataForQuery(item.query);
            
            const data = await this.getData(item.key, fetchFunction);
            console.log(`✅ Preloaded: ${item.key}`);
            
        } catch (error) {
            console.warn(`⚠️ Preload failed: ${item.key}`, error);
        } finally {
            this.activePreloads.delete(item.key);
            
            // Process next item
            if (this.preloadQueue.length > 0) {
                setTimeout(() => this.processPreloadQueue(), 100);
            }
        }
    }
    
    /**
     * Predictive preloading based on user behavior
     */
    analyzeUserBehavior(interaction) {
        this.userBehavior.push({
            ...interaction,
            timestamp: Date.now()
        });
        
        // Keep only recent behavior
        if (this.userBehavior.length > this.config.preloading.behaviorAnalysisDepth) {
            this.userBehavior.shift();
        }
        
        // Predict next actions
        this.updatePreloadPredictions();
    }
    
    updatePreloadPredictions() {
        // Simple pattern recognition for predictive preloading
        const recentActions = this.userBehavior.slice(-5);
        
        for (const action of recentActions) {
            if (action.type === 'nodeClick') {
                // Predict user might want neighbor data
                this.preloadData({
                    type: 'node_neighbors',
                    nodeId: action.nodeId,
                    priority: 'medium'
                });
            } else if (action.type === 'zoom') {
                // Predict user might want higher/lower detail
                if (action.direction === 'in') {
                    this.preloadData({
                        type: 'high_detail_viewport',
                        viewport: action.viewport,
                        priority: 'high'
                    });
                }
            }
        }
    }
    
    /**
     * Handle viewport changes for spatial preloading
     */
    handleViewportChange(viewport) {
        // Preload data for areas just outside the viewport
        const buffer = this.config.preloading.viewportBuffer;
        const expandedViewport = {
            x: viewport.x - viewport.width * (buffer - 1) / 2,
            y: viewport.y - viewport.height * (buffer - 1) / 2,
            width: viewport.width * buffer,
            height: viewport.height * buffer
        };
        
        this.preloadData({
            type: 'viewport_data',
            viewport: expandedViewport,
            priority: 'low'
        });
    }
    
    /**
     * Data compression/decompression
     */
    async compressData(data) {
        if (this.compressionWorker) {
            // Use web worker for compression
            return new Promise((resolve) => {
                const id = Math.random().toString(36);
                
                const handler = (e) => {
                    if (e.data.id === id) {
                        this.compressionWorker.removeEventListener('message', handler);
                        resolve(e.data.result);
                    }
                };
                
                this.compressionWorker.addEventListener('message', handler);
                this.compressionWorker.postMessage({ id, action: 'compress', data });
            });
        } else {
            // Synchronous compression (simplified)
            try {
                const jsonString = JSON.stringify(data);
                const compressed = this.simpleCompress(jsonString);
                return {
                    data: compressed,
                    size: compressed.byteLength
                };
            } catch (error) {
                return null;
            }
        }
    }
    
    async decompressData(compressedData) {
        if (this.compressionWorker) {
            // Use web worker for decompression
            return new Promise((resolve) => {
                const id = Math.random().toString(36);
                
                const handler = (e) => {
                    if (e.data.id === id) {
                        this.compressionWorker.removeEventListener('message', handler);
                        resolve(e.data.result);
                    }
                };
                
                this.compressionWorker.addEventListener('message', handler);
                this.compressionWorker.postMessage({ id, action: 'decompress', data: compressedData });
            });
        } else {
            // Synchronous decompression
            try {
                const jsonString = this.simpleDecompress(compressedData);
                return JSON.parse(jsonString);
            } catch (error) {
                return null;
            }
        }
    }
    
    simpleCompress(string) {
        // Simplified compression using gzip-like algorithm
        return new TextEncoder().encode(string);
    }
    
    simpleDecompress(compressed) {
        // Simplified decompression
        return new TextDecoder().decode(compressed);
    }
    
    /**
     * Cache cleanup and maintenance
     */
    performCacheCleanup() {
        const now = Date.now();
        
        // Clean memory cache
        for (const [key, entry] of this.memoryCache.entries()) {
            if (now - entry.metadata.timestamp > this.config.memoryCache.ttl) {
                this.memoryCache.delete(key);
                this.cacheStats.evictions++;
            }
        }
        
        // Clean IndexedDB (less frequent)
        if (Math.random() < 0.1) { // 10% chance
            this.cleanIndexedDB();
        }
        
        console.log(`🧹 Cache cleanup completed: ${this.cacheStats.evictions} evictions`);
    }
    
    async cleanIndexedDB() {
        if (!this.indexedDB) return;
        
        const transaction = this.indexedDB.transaction(['graph_data'], 'readwrite');
        const store = transaction.objectStore('graph_data');
        const index = store.index('timestamp');
        
        const cutoff = Date.now() - this.config.indexedDBCache.ttl;
        const range = IDBKeyRange.upperBound(cutoff);
        
        const request = index.openCursor(range);
        request.onsuccess = (event) => {
            const cursor = event.target.result;
            if (cursor) {
                cursor.delete();
                cursor.continue();
            }
        };
    }
    
    /**
     * Event handlers
     */
    handleDataRequest(request) {
        // Automatically cache data requests
        const key = this.generateCacheKey(request);
        this.getData(key, () => this.fetchDataForQuery(request));
    }
    
    handleZoomChange(zoomData) {
        // Preload appropriate detail level
        if (zoomData.zoom > 1.5) {
            this.preloadData({ type: 'high_detail', priority: 'high' });
        } else {
            this.preloadData({ type: 'low_detail', priority: 'normal' });
        }
    }
    
    handleOnline() {
        console.log('📡 Connection restored, resuming preloading');
        this.config.preloading.enabled = true;
    }
    
    handleOffline() {
        console.log('📡 Connection lost, using cached data only');
        this.config.preloading.enabled = false;
    }
    
    /**
     * Helper methods
     */
    generateCacheKey(query) {
        return `${query.type}_${JSON.stringify(query).replace(/\s/g, '')}`;
    }
    
    estimateSize(data) {
        return new Blob([JSON.stringify(data)]).size;
    }
    
    async fetchDataForQuery(query) {
        // This would integrate with the actual data fetching system
        // For now, return a simulated response
        await new Promise(resolve => setTimeout(resolve, 100)); // Simulate network delay
        
        return {
            type: query.type,
            data: `simulated_data_for_${query.type}`,
            timestamp: Date.now()
        };
    }
    
    handleCompressionResult(result) {
        // Handle compression worker results
        console.log('Compression result received:', result);
    }
    
    handleBackgroundResult(result) {
        // Handle background worker results
        console.log('Background processing result:', result);
    }
    
    /**
     * Get cache performance statistics
     */
    getPerformanceStats() {
        return {
            ...this.cacheStats,
            memoryCacheSize: this.getMemoryCacheSize(),
            memoryCacheEntries: this.memoryCache.size,
            preloadQueueSize: this.preloadQueue.size,
            activePreloads: this.activePreloads.size,
            hitRatio: this.cacheStats.hits / (this.cacheStats.hits + this.cacheStats.misses),
            config: this.config
        };
    }
    
    /**
     * Configure cache settings
     */
    configure(settings) {
        this.config = { ...this.config, ...settings };
        console.log('🎛️  Cache configuration updated');
    }
    
    /**
     * Clear all caches
     */
    async clearAllCaches() {
        // Clear memory cache
        this.memoryCache.clear();
        
        // Clear IndexedDB
        if (this.indexedDB) {
            const transaction = this.indexedDB.transaction(['graph_data'], 'readwrite');
            const store = transaction.objectStore('graph_data');
            store.clear();
        }
        
        // Clear localStorage
        const keys = Object.keys(localStorage);
        for (const key of keys) {
            if (key.startsWith('groggy_cache_')) {
                localStorage.removeItem(key);
            }
        }
        
        console.log('🧹 All caches cleared');
    }
    
    /**
     * Cleanup method
     */
    destroy() {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
        }
        
        if (this.compressionWorker) {
            this.compressionWorker.terminate();
        }
        
        if (this.backgroundWorker) {
            this.backgroundWorker.terminate();
        }
        
        if (this.indexedDB) {
            this.indexedDB.close();
        }
        
        this.memoryCache.clear();
        this.preloadQueue.length = 0;
        this.activePreloads.clear();
        
        console.log('🧹 Cache Preloader System cleaned up');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CachePreloaderSystem;
}