/**
 * 🎯 Performance Monitoring and FPS Tracking System
 * Part of Groggy Phase 10: Performance Optimization
 * 
 * Comprehensive performance monitoring system that tracks:
 * - Real-time FPS and frame timing
 * - Memory usage patterns and garbage collection
 * - Rendering performance metrics
 * - Network latency and throughput
 * - User interaction responsiveness
 * - Browser resource utilization
 * - Performance warnings and optimization suggestions
 */

class PerformanceMonitor {
    constructor(config = {}) {
        this.config = {
            // FPS Tracking
            targetFPS: 60,
            fpsUpdateInterval: 100, // ms
            fpsHistorySize: 300, // ~5 seconds at 60 FPS
            
            // Memory Monitoring
            memoryCheckInterval: 1000, // ms
            memoryHistorySize: 300, // 5 minutes
            gcThreshold: 0.8, // Trigger warnings at 80% memory usage
            
            // Performance Thresholds
            frameTimeWarning: 16.67, // ms (60 FPS threshold)
            frameTimeCritical: 33.33, // ms (30 FPS threshold)
            renderTimeWarning: 10, // ms
            networkLatencyWarning: 200, // ms
            
            // Monitoring Options
            enableDetailedMetrics: true,
            enableNetworkMonitoring: true,
            enableMemoryProfiling: true,
            enableUserInteractionTracking: true,
            enableAutoOptimization: true,
            
            // Reporting
            reportInterval: 5000, // ms
            maxReportHistory: 100,
            
            ...config
        };
        
        // Core metrics storage
        this.metrics = {
            fps: {
                current: 0,
                average: 0,
                min: Infinity,
                max: 0,
                history: [],
                lastUpdate: 0
            },
            frameTime: {
                current: 0,
                average: 0,
                history: [],
                worstFrames: []
            },
            memory: {
                used: 0,
                total: 0,
                percentage: 0,
                history: [],
                gcEvents: []
            },
            rendering: {
                nodesRendered: 0,
                edgesRendered: 0,
                renderTime: 0,
                culledNodes: 0,
                lodLevel: 'MEDIUM',
                history: []
            },
            network: {
                latency: 0,
                throughput: 0,
                messagesPerSecond: 0,
                bytesPerSecond: 0,
                history: []
            },
            interactions: {
                responseTime: 0,
                eventQueue: 0,
                droppedEvents: 0,
                history: []
            }
        };
        
        // Performance state
        this.isMonitoring = false;
        this.perfObserver = null;
        this.frameCounter = 0;
        this.lastFrameTime = 0;
        this.animationFrameId = null;
        
        // Warning system
        this.warnings = {
            active: new Set(),
            history: [],
            callbacks: new Map()
        };
        
        // Auto-optimization
        this.optimizations = {
            lastApplied: new Map(),
            cooldown: 30000, // 30 seconds
            suggestions: []
        };
        
        // Initialize monitoring components
        this.initializePerformanceObserver();
        this.initializeMemoryMonitoring();
        this.initializeNetworkMonitoring();
        this.initializeUserInteractionTracking();
        
        console.log('🎯 PerformanceMonitor initialized with comprehensive tracking');
    }
    
    /**
     * Initialize Performance Observer API for detailed metrics
     */
    initializePerformanceObserver() {
        if (!('PerformanceObserver' in window)) {
            console.warn('PerformanceObserver not supported, using fallback timing');
            return;
        }
        
        try {
            this.perfObserver = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                
                entries.forEach(entry => {
                    switch (entry.entryType) {
                        case 'measure':
                            this.processMeasureEntry(entry);
                            break;
                        case 'navigation':
                            this.processNavigationEntry(entry);
                            break;
                        case 'resource':
                            this.processResourceEntry(entry);
                            break;
                        case 'paint':
                            this.processPaintEntry(entry);
                            break;
                    }
                });
            });
            
            // Observe different types of performance entries
            ['measure', 'navigation', 'resource', 'paint'].forEach(type => {
                try {
                    this.perfObserver.observe({ entryTypes: [type] });
                } catch (e) {
                    console.warn(`Cannot observe ${type} entries:`, e);
                }
            });
            
        } catch (error) {
            console.warn('Failed to initialize PerformanceObserver:', error);
        }
    }
    
    /**
     * Initialize memory usage monitoring
     */
    initializeMemoryMonitoring() {
        if (!('memory' in performance)) {
            console.warn('Performance.memory not available');
            return;
        }
        
        setInterval(() => {
            if (this.isMonitoring) {
                this.updateMemoryMetrics();
            }
        }, this.config.memoryCheckInterval);
    }
    
    /**
     * Initialize network performance monitoring
     */
    initializeNetworkMonitoring() {
        if (!this.config.enableNetworkMonitoring) return;
        
        // Track WebSocket metrics
        this.networkMetrics = {
            messagesSent: 0,
            messagesReceived: 0,
            bytesSent: 0,
            bytesReceived: 0,
            connectionTime: 0,
            lastLatencyCheck: 0
        };
        
        // Override WebSocket send to track metrics
        this.originalWebSocketSend = WebSocket.prototype.send;
        WebSocket.prototype.send = (data) => {
            this.trackNetworkSend(data);
            return this.originalWebSocketSend.call(this, data);
        };
    }
    
    /**
     * Initialize user interaction responsiveness tracking
     */
    initializeUserInteractionTracking() {
        if (!this.config.enableUserInteractionTracking) return;
        
        const interactionEvents = ['click', 'mousedown', 'keydown', 'touchstart'];
        
        interactionEvents.forEach(eventType => {
            document.addEventListener(eventType, (event) => {
                this.trackUserInteraction(event);
            }, { passive: true, capture: true });
        });
        
        // Track event queue depth
        setInterval(() => {
            this.updateEventQueueMetrics();
        }, 100);
    }
    
    /**
     * Start performance monitoring
     */
    start() {
        if (this.isMonitoring) return;
        
        this.isMonitoring = true;
        this.lastFrameTime = performance.now();
        this.frameCounter = 0;
        
        // Start FPS tracking
        this.startFPSTracking();
        
        // Start periodic reporting
        this.reportInterval = setInterval(() => {
            this.generatePerformanceReport();
        }, this.config.reportInterval);
        
        console.log('🎯 Performance monitoring started');
    }
    
    /**
     * Stop performance monitoring
     */
    stop() {
        if (!this.isMonitoring) return;
        
        this.isMonitoring = false;
        
        // Stop FPS tracking
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        
        // Stop reporting
        if (this.reportInterval) {
            clearInterval(this.reportInterval);
            this.reportInterval = null;
        }
        
        console.log('🎯 Performance monitoring stopped');
    }
    
    /**
     * Start FPS tracking using requestAnimationFrame
     */
    startFPSTracking() {
        const trackFrame = (timestamp) => {
            if (!this.isMonitoring) return;
            
            // Calculate frame time
            const frameTime = timestamp - this.lastFrameTime;
            this.lastFrameTime = timestamp;
            
            // Update FPS metrics
            if (frameTime > 0) {
                const currentFPS = 1000 / frameTime;
                this.updateFPSMetrics(currentFPS, frameTime);
            }
            
            this.frameCounter++;
            this.animationFrameId = requestAnimationFrame(trackFrame);
        };
        
        this.animationFrameId = requestAnimationFrame(trackFrame);
    }
    
    /**
     * Update FPS metrics
     */
    updateFPSMetrics(currentFPS, frameTime) {
        const fps = this.metrics.fps;
        const ft = this.metrics.frameTime;
        
        // Update current values
        fps.current = currentFPS;
        ft.current = frameTime;
        
        // Update history
        fps.history.push({ value: currentFPS, timestamp: performance.now() });
        ft.history.push({ value: frameTime, timestamp: performance.now() });
        
        // Trim history
        if (fps.history.length > this.config.fpsHistorySize) {
            fps.history.shift();
        }
        if (ft.history.length > this.config.fpsHistorySize) {
            ft.history.shift();
        }
        
        // Update aggregates
        fps.min = Math.min(fps.min, currentFPS);
        fps.max = Math.max(fps.max, currentFPS);
        
        // Calculate averages
        const recentHistory = fps.history.slice(-60); // Last 60 frames
        fps.average = recentHistory.reduce((sum, entry) => sum + entry.value, 0) / recentHistory.length;
        
        const recentFrameTime = ft.history.slice(-60);
        ft.average = recentFrameTime.reduce((sum, entry) => sum + entry.value, 0) / recentFrameTime.length;
        
        // Track worst performing frames
        if (frameTime > this.config.frameTimeWarning) {
            ft.worstFrames.push({
                frameTime,
                timestamp: performance.now(),
                fps: currentFPS
            });
            
            // Keep only worst 20 frames
            ft.worstFrames.sort((a, b) => b.frameTime - a.frameTime);
            ft.worstFrames = ft.worstFrames.slice(0, 20);
        }
        
        // Check for performance warnings
        this.checkPerformanceWarnings();
    }
    
    /**
     * Update memory usage metrics
     */
    updateMemoryMetrics() {
        if (!('memory' in performance)) return;
        
        const memory = performance.memory;
        const memoryMetrics = this.metrics.memory;
        
        // Update current values
        memoryMetrics.used = memory.usedJSHeapSize;
        memoryMetrics.total = memory.totalJSHeapSize;
        memoryMetrics.percentage = (memoryMetrics.used / memoryMetrics.total) * 100;
        
        // Update history
        memoryMetrics.history.push({
            used: memoryMetrics.used,
            total: memoryMetrics.total,
            percentage: memoryMetrics.percentage,
            timestamp: performance.now()
        });
        
        // Trim history
        if (memoryMetrics.history.length > this.config.memoryHistorySize) {
            memoryMetrics.history.shift();
        }
        
        // Detect potential garbage collection events
        if (memoryMetrics.history.length > 1) {
            const previous = memoryMetrics.history[memoryMetrics.history.length - 2];
            const memoryDrop = previous.used - memoryMetrics.used;
            
            if (memoryDrop > 1024 * 1024) { // 1MB drop indicates possible GC
                memoryMetrics.gcEvents.push({
                    timestamp: performance.now(),
                    memoryFreed: memoryDrop,
                    beforeGC: previous.used,
                    afterGC: memoryMetrics.used
                });
                
                // Keep only recent GC events
                memoryMetrics.gcEvents = memoryMetrics.gcEvents.slice(-50);
            }
        }
    }
    
    /**
     * Track rendering performance
     */
    trackRenderingPerformance(renderData) {
        const rendering = this.metrics.rendering;
        
        // Update current values
        rendering.nodesRendered = renderData.nodesRendered || 0;
        rendering.edgesRendered = renderData.edgesRendered || 0;
        rendering.renderTime = renderData.renderTime || 0;
        rendering.culledNodes = renderData.culledNodes || 0;
        rendering.lodLevel = renderData.lodLevel || 'MEDIUM';
        
        // Update history
        rendering.history.push({
            ...rendering,
            timestamp: performance.now()
        });
        
        // Trim history
        if (rendering.history.length > 200) {
            rendering.history.shift();
        }
    }
    
    /**
     * Track network performance
     */
    trackNetworkSend(data) {
        if (!this.config.enableNetworkMonitoring) return;
        
        this.networkMetrics.messagesSent++;
        this.networkMetrics.bytesSent += data.length || 0;
        
        // Update metrics
        const network = this.metrics.network;
        const now = performance.now();
        
        // Calculate messages per second
        const recentMessages = this.networkMetrics.messagesSent;
        const timeWindow = 1000; // 1 second
        network.messagesPerSecond = recentMessages; // Simplified calculation
        
        // Calculate bytes per second
        network.bytesPerSecond = this.networkMetrics.bytesSent;
        
        // Update history
        network.history.push({
            messagesSent: this.networkMetrics.messagesSent,
            bytesSent: this.networkMetrics.bytesSent,
            messagesPerSecond: network.messagesPerSecond,
            bytesPerSecond: network.bytesPerSecond,
            timestamp: now
        });
        
        // Trim history
        if (network.history.length > 300) {
            network.history.shift();
        }
    }
    
    /**
     * Track user interaction responsiveness
     */
    trackUserInteraction(event) {
        const interactionStart = performance.now();
        
        // Use setTimeout to measure response time
        setTimeout(() => {
            const responseTime = performance.now() - interactionStart;
            
            const interactions = this.metrics.interactions;
            interactions.responseTime = responseTime;
            
            // Update history
            interactions.history.push({
                eventType: event.type,
                responseTime,
                timestamp: interactionStart
            });
            
            // Trim history
            if (interactions.history.length > 100) {
                interactions.history.shift();
            }
            
            // Check for slow interactions
            if (responseTime > 100) { // 100ms is noticeable delay
                this.addWarning('slow_interaction', {
                    eventType: event.type,
                    responseTime,
                    timestamp: interactionStart
                });
            }
        }, 0);
    }
    
    /**
     * Update event queue metrics
     */
    updateEventQueueMetrics() {
        // This is a simplified approximation
        // In a real implementation, you'd track actual event queue depth
        const interactions = this.metrics.interactions;
        interactions.eventQueue = 0; // Placeholder
    }
    
    /**
     * Check for performance warnings
     */
    checkPerformanceWarnings() {
        const fps = this.metrics.fps;
        const frameTime = this.metrics.frameTime;
        const memory = this.metrics.memory;
        
        // FPS warnings
        if (fps.current < 30) {
            this.addWarning('critical_fps', { fps: fps.current });
        } else if (fps.current < 45) {
            this.addWarning('low_fps', { fps: fps.current });
        } else {
            this.removeWarning('critical_fps');
            this.removeWarning('low_fps');
        }
        
        // Frame time warnings
        if (frameTime.current > this.config.frameTimeCritical) {
            this.addWarning('critical_frame_time', { frameTime: frameTime.current });
        } else if (frameTime.current > this.config.frameTimeWarning) {
            this.addWarning('high_frame_time', { frameTime: frameTime.current });
        } else {
            this.removeWarning('critical_frame_time');
            this.removeWarning('high_frame_time');
        }
        
        // Memory warnings
        if (memory.percentage > 90) {
            this.addWarning('critical_memory', { percentage: memory.percentage });
        } else if (memory.percentage > this.config.gcThreshold * 100) {
            this.addWarning('high_memory', { percentage: memory.percentage });
        } else {
            this.removeWarning('critical_memory');
            this.removeWarning('high_memory');
        }
        
        // Auto-optimization
        if (this.config.enableAutoOptimization) {
            this.checkAutoOptimization();
        }
    }
    
    /**
     * Add performance warning
     */
    addWarning(type, data) {
        if (this.warnings.active.has(type)) return;
        
        this.warnings.active.add(type);
        
        const warning = {
            type,
            data,
            timestamp: performance.now(),
            id: Math.random().toString(36).substr(2, 9)
        };
        
        this.warnings.history.push(warning);
        
        // Trim history
        if (this.warnings.history.length > 100) {
            this.warnings.history.shift();
        }
        
        // Trigger callbacks
        if (this.warnings.callbacks.has(type)) {
            this.warnings.callbacks.get(type).forEach(callback => {
                try {
                    callback(warning);
                } catch (error) {
                    console.error('Warning callback error:', error);
                }
            });
        }
        
        console.warn(`🎯 Performance warning: ${type}`, data);
    }
    
    /**
     * Remove performance warning
     */
    removeWarning(type) {
        this.warnings.active.delete(type);
    }
    
    /**
     * Check for auto-optimization opportunities
     */
    checkAutoOptimization() {
        const now = performance.now();
        const fps = this.metrics.fps;
        const memory = this.metrics.memory;
        
        // FPS-based optimizations
        if (fps.current < 30) {
            this.suggestOptimization('reduce_quality', {
                reason: 'Critical FPS drop',
                currentFPS: fps.current,
                cooldown: this.optimizations.cooldown
            });
        }
        
        // Memory-based optimizations
        if (memory.percentage > 85) {
            this.suggestOptimization('cleanup_cache', {
                reason: 'High memory usage',
                memoryPercentage: memory.percentage,
                cooldown: this.optimizations.cooldown
            });
        }
    }
    
    /**
     * Suggest optimization
     */
    suggestOptimization(type, data) {
        const now = performance.now();
        const lastApplied = this.optimizations.lastApplied.get(type) || 0;
        
        // Check cooldown
        if (now - lastApplied < this.optimizations.cooldown) {
            return;
        }
        
        const suggestion = {
            type,
            data,
            timestamp: now,
            id: Math.random().toString(36).substr(2, 9)
        };
        
        this.optimizations.suggestions.push(suggestion);
        
        // Trim suggestions
        if (this.optimizations.suggestions.length > 20) {
            this.optimizations.suggestions.shift();
        }
        
        console.log(`🎯 Performance optimization suggested: ${type}`, data);
    }
    
    /**
     * Apply optimization
     */
    applyOptimization(type) {
        this.optimizations.lastApplied.set(type, performance.now());
        
        // Remove applied suggestion
        this.optimizations.suggestions = this.optimizations.suggestions.filter(
            suggestion => suggestion.type !== type
        );
        
        console.log(`🎯 Applied optimization: ${type}`);
    }
    
    /**
     * Generate comprehensive performance report
     */
    generatePerformanceReport() {
        const report = {
            timestamp: performance.now(),
            summary: {
                fps: {
                    current: Math.round(this.metrics.fps.current * 100) / 100,
                    average: Math.round(this.metrics.fps.average * 100) / 100,
                    min: Math.round(this.metrics.fps.min * 100) / 100,
                    max: Math.round(this.metrics.fps.max * 100) / 100
                },
                frameTime: {
                    current: Math.round(this.metrics.frameTime.current * 100) / 100,
                    average: Math.round(this.metrics.frameTime.average * 100) / 100
                },
                memory: {
                    used: Math.round(this.metrics.memory.used / 1024 / 1024 * 100) / 100, // MB
                    percentage: Math.round(this.metrics.memory.percentage * 100) / 100
                },
                rendering: {
                    nodesRendered: this.metrics.rendering.nodesRendered,
                    renderTime: Math.round(this.metrics.rendering.renderTime * 100) / 100,
                    lodLevel: this.metrics.rendering.lodLevel
                }
            },
            warnings: Array.from(this.warnings.active),
            suggestions: this.optimizations.suggestions.length,
            health: this.calculateHealthScore()
        };
        
        // Dispatch performance report event
        window.dispatchEvent(new CustomEvent('performancereport', {
            detail: report
        }));
        
        return report;
    }
    
    /**
     * Calculate overall performance health score (0-100)
     */
    calculateHealthScore() {
        let score = 100;
        
        // FPS impact
        const fps = this.metrics.fps.current;
        if (fps < 30) score -= 40;
        else if (fps < 45) score -= 20;
        else if (fps < 55) score -= 10;
        
        // Memory impact
        const memoryPercentage = this.metrics.memory.percentage;
        if (memoryPercentage > 90) score -= 30;
        else if (memoryPercentage > 80) score -= 15;
        else if (memoryPercentage > 70) score -= 5;
        
        // Frame time impact
        const frameTime = this.metrics.frameTime.current;
        if (frameTime > 33.33) score -= 20; // Below 30 FPS
        else if (frameTime > 22.22) score -= 10; // Below 45 FPS
        
        // Active warnings impact
        score -= this.warnings.active.size * 5;
        
        return Math.max(0, Math.min(100, score));
    }
    
    /**
     * Process performance measure entries
     */
    processMeasureEntry(entry) {
        if (entry.name.startsWith('groggy-')) {
            // Track custom groggy performance measures
            const measureType = entry.name.replace('groggy-', '');
            
            if (!this.customMeasures) {
                this.customMeasures = {};
            }
            
            if (!this.customMeasures[measureType]) {
                this.customMeasures[measureType] = [];
            }
            
            this.customMeasures[measureType].push({
                duration: entry.duration,
                timestamp: entry.startTime
            });
            
            // Keep only recent measures
            if (this.customMeasures[measureType].length > 100) {
                this.customMeasures[measureType].shift();
            }
        }
    }
    
    /**
     * Process navigation timing entries
     */
    processNavigationEntry(entry) {
        this.navigationTiming = {
            dnsLookup: entry.domainLookupEnd - entry.domainLookupStart,
            tcpConnect: entry.connectEnd - entry.connectStart,
            request: entry.responseStart - entry.requestStart,
            response: entry.responseEnd - entry.responseStart,
            domReady: entry.domContentLoadedEventStart - entry.navigationStart,
            loadComplete: entry.loadEventStart - entry.navigationStart
        };
    }
    
    /**
     * Process resource timing entries
     */
    processResourceEntry(entry) {
        if (entry.name.includes('groggy')) {
            // Track groggy-specific resource loading
            if (!this.resourceTiming) {
                this.resourceTiming = [];
            }
            
            this.resourceTiming.push({
                name: entry.name,
                duration: entry.duration,
                size: entry.transferSize,
                timestamp: entry.startTime
            });
            
            // Keep only recent resources
            if (this.resourceTiming.length > 50) {
                this.resourceTiming.shift();
            }
        }
    }
    
    /**
     * Process paint timing entries
     */
    processPaintEntry(entry) {
        if (!this.paintTiming) {
            this.paintTiming = {};
        }
        
        this.paintTiming[entry.name] = entry.startTime;
    }
    
    /**
     * Register warning callback
     */
    onWarning(type, callback) {
        if (!this.warnings.callbacks.has(type)) {
            this.warnings.callbacks.set(type, []);
        }
        this.warnings.callbacks.get(type).push(callback);
    }
    
    /**
     * Get current performance snapshot
     */
    getSnapshot() {
        return {
            timestamp: performance.now(),
            metrics: JSON.parse(JSON.stringify(this.metrics)),
            warnings: Array.from(this.warnings.active),
            suggestions: [...this.optimizations.suggestions],
            health: this.calculateHealthScore(),
            customMeasures: this.customMeasures || {},
            navigationTiming: this.navigationTiming || {},
            resourceTiming: this.resourceTiming || [],
            paintTiming: this.paintTiming || {}
        };
    }
    
    /**
     * Export performance data
     */
    exportData(format = 'json') {
        const data = this.getSnapshot();
        
        switch (format) {
            case 'json':
                return JSON.stringify(data, null, 2);
            case 'csv':
                return this.convertToCSV(data);
            default:
                throw new Error(`Unsupported export format: ${format}`);
        }
    }
    
    /**
     * Convert performance data to CSV
     */
    convertToCSV(data) {
        const rows = [];
        
        // Headers
        rows.push([
            'Timestamp',
            'FPS_Current',
            'FPS_Average',
            'FrameTime_Current',
            'FrameTime_Average',
            'Memory_Used_MB',
            'Memory_Percentage',
            'Nodes_Rendered',
            'Render_Time',
            'LOD_Level',
            'Health_Score',
            'Active_Warnings'
        ]);
        
        // Data row
        rows.push([
            data.timestamp,
            data.metrics.fps.current,
            data.metrics.fps.average,
            data.metrics.frameTime.current,
            data.metrics.frameTime.average,
            Math.round(data.metrics.memory.used / 1024 / 1024 * 100) / 100,
            data.metrics.memory.percentage,
            data.metrics.rendering.nodesRendered,
            data.metrics.rendering.renderTime,
            data.metrics.rendering.lodLevel,
            data.health,
            data.warnings.join(';')
        ]);
        
        return rows.map(row => row.join(',')).join('\n');
    }
    
    /**
     * Cleanup and destroy
     */
    destroy() {
        this.stop();
        
        if (this.perfObserver) {
            this.perfObserver.disconnect();
            this.perfObserver = null;
        }
        
        // Restore original WebSocket.send if we overrode it
        if (this.originalWebSocketSend) {
            WebSocket.prototype.send = this.originalWebSocketSend;
        }
        
        console.log('🎯 PerformanceMonitor destroyed');
    }
}

/**
 * Performance Dashboard Widget
 * Real-time display of performance metrics
 */
class PerformanceDashboard {
    constructor(container, monitor) {
        this.container = container;
        this.monitor = monitor;
        this.isVisible = false;
        this.updateInterval = null;
        
        this.createDashboard();
        this.setupEventListeners();
        
        console.log('🎯 Performance Dashboard initialized');
    }
    
    /**
     * Create dashboard UI
     */
    createDashboard() {
        this.container.innerHTML = `
            <div class="performance-dashboard" style="display: none;">
                <div class="dashboard-header">
                    <h3>🎯 Performance Monitor</h3>
                    <div class="dashboard-controls">
                        <button class="toggle-details">Details</button>
                        <button class="export-data">Export</button>
                        <button class="close-dashboard">×</button>
                    </div>
                </div>
                
                <div class="dashboard-content">
                    <div class="metrics-grid">
                        <div class="metric-card fps-card">
                            <div class="metric-label">FPS</div>
                            <div class="metric-value fps-value">0</div>
                            <div class="metric-chart fps-chart"></div>
                        </div>
                        
                        <div class="metric-card memory-card">
                            <div class="metric-label">Memory</div>
                            <div class="metric-value memory-value">0 MB</div>
                            <div class="metric-chart memory-chart"></div>
                        </div>
                        
                        <div class="metric-card render-card">
                            <div class="metric-label">Render Time</div>
                            <div class="metric-value render-value">0 ms</div>
                            <div class="metric-chart render-chart"></div>
                        </div>
                        
                        <div class="metric-card health-card">
                            <div class="metric-label">Health Score</div>
                            <div class="metric-value health-value">100</div>
                            <div class="metric-indicator health-indicator"></div>
                        </div>
                    </div>
                    
                    <div class="warnings-section">
                        <h4>Active Warnings</h4>
                        <div class="warnings-list"></div>
                    </div>
                    
                    <div class="details-section" style="display: none;">
                        <h4>Detailed Metrics</h4>
                        <div class="details-content"></div>
                    </div>
                </div>
            </div>
        `;
        
        this.addDashboardStyles();
    }
    
    /**
     * Add dashboard CSS styles
     */
    addDashboardStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .performance-dashboard {
                position: fixed;
                top: 20px;
                right: 20px;
                width: 400px;
                background: rgba(0, 0, 0, 0.9);
                color: white;
                border-radius: 8px;
                padding: 16px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 12px;
                z-index: 10000;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            }
            
            .dashboard-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 16px;
                border-bottom: 1px solid #333;
                padding-bottom: 8px;
            }
            
            .dashboard-header h3 {
                margin: 0;
                font-size: 14px;
            }
            
            .dashboard-controls button {
                background: #333;
                color: white;
                border: none;
                padding: 4px 8px;
                margin-left: 4px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 10px;
            }
            
            .dashboard-controls button:hover {
                background: #555;
            }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                margin-bottom: 16px;
            }
            
            .metric-card {
                background: #1a1a1a;
                padding: 12px;
                border-radius: 6px;
                border: 1px solid #333;
            }
            
            .metric-label {
                font-size: 10px;
                color: #aaa;
                margin-bottom: 4px;
            }
            
            .metric-value {
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 8px;
            }
            
            .metric-chart {
                height: 30px;
                background: #111;
                border-radius: 3px;
                position: relative;
                overflow: hidden;
            }
            
            .fps-card .metric-value { color: #4CAF50; }
            .memory-card .metric-value { color: #FF9800; }
            .render-card .metric-value { color: #2196F3; }
            .health-card .metric-value { color: #8BC34A; }
            
            .health-indicator {
                width: 100%;
                height: 8px;
                background: #333;
                border-radius: 4px;
                overflow: hidden;
            }
            
            .health-indicator::after {
                content: '';
                display: block;
                height: 100%;
                background: linear-gradient(90deg, #f44336, #ff9800, #4caf50);
                transition: width 0.3s ease;
            }
            
            .warnings-section {
                margin-bottom: 16px;
            }
            
            .warnings-section h4 {
                margin: 0 0 8px 0;
                font-size: 12px;
                color: #ff5722;
            }
            
            .warnings-list {
                background: #1a1a1a;
                padding: 8px;
                border-radius: 4px;
                min-height: 20px;
            }
            
            .warning-item {
                background: #ff5722;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                display: inline-block;
                margin: 2px;
                font-size: 10px;
            }
            
            .details-section {
                border-top: 1px solid #333;
                padding-top: 12px;
            }
            
            .details-section h4 {
                margin: 0 0 8px 0;
                font-size: 12px;
            }
            
            .details-content {
                background: #1a1a1a;
                padding: 8px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 10px;
                max-height: 200px;
                overflow-y: auto;
            }
        `;
        
        document.head.appendChild(style);
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        const dashboard = this.container.querySelector('.performance-dashboard');
        
        // Toggle details
        dashboard.querySelector('.toggle-details').addEventListener('click', () => {
            const details = dashboard.querySelector('.details-section');
            const isVisible = details.style.display !== 'none';
            details.style.display = isVisible ? 'none' : 'block';
        });
        
        // Export data
        dashboard.querySelector('.export-data').addEventListener('click', () => {
            this.exportPerformanceData();
        });
        
        // Close dashboard
        dashboard.querySelector('.close-dashboard').addEventListener('click', () => {
            this.hide();
        });
    }
    
    /**
     * Show dashboard
     */
    show() {
        const dashboard = this.container.querySelector('.performance-dashboard');
        dashboard.style.display = 'block';
        this.isVisible = true;
        
        // Start updating
        this.updateInterval = setInterval(() => {
            this.updateDashboard();
        }, 100);
    }
    
    /**
     * Hide dashboard
     */
    hide() {
        const dashboard = this.container.querySelector('.performance-dashboard');
        dashboard.style.display = 'none';
        this.isVisible = false;
        
        // Stop updating
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
    
    /**
     * Toggle dashboard visibility
     */
    toggle() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }
    
    /**
     * Update dashboard with current metrics
     */
    updateDashboard() {
        if (!this.isVisible) return;
        
        const metrics = this.monitor.metrics;
        const dashboard = this.container.querySelector('.performance-dashboard');
        
        // Update FPS
        const fpsValue = dashboard.querySelector('.fps-value');
        fpsValue.textContent = Math.round(metrics.fps.current);
        
        // Update Memory
        const memoryValue = dashboard.querySelector('.memory-value');
        const memoryMB = Math.round(metrics.memory.used / 1024 / 1024);
        memoryValue.textContent = `${memoryMB} MB (${Math.round(metrics.memory.percentage)}%)`;
        
        // Update Render Time
        const renderValue = dashboard.querySelector('.render-value');
        renderValue.textContent = `${Math.round(metrics.rendering.renderTime * 100) / 100} ms`;
        
        // Update Health Score
        const healthValue = dashboard.querySelector('.health-value');
        const healthScore = this.monitor.calculateHealthScore();
        healthValue.textContent = Math.round(healthScore);
        
        // Update health indicator
        const healthIndicator = dashboard.querySelector('.health-indicator');
        healthIndicator.style.setProperty('--health-width', `${healthScore}%`);
        
        // Update warnings
        this.updateWarnings(dashboard);
        
        // Update details if visible
        const detailsSection = dashboard.querySelector('.details-section');
        if (detailsSection.style.display !== 'none') {
            this.updateDetails(dashboard);
        }
    }
    
    /**
     * Update warnings display
     */
    updateWarnings(dashboard) {
        const warningsList = dashboard.querySelector('.warnings-list');
        const activeWarnings = Array.from(this.monitor.warnings.active);
        
        if (activeWarnings.length === 0) {
            warningsList.innerHTML = '<span style="color: #4CAF50;">No warnings</span>';
        } else {
            warningsList.innerHTML = activeWarnings
                .map(warning => `<span class="warning-item">${warning}</span>`)
                .join('');
        }
    }
    
    /**
     * Update detailed metrics
     */
    updateDetails(dashboard) {
        const detailsContent = dashboard.querySelector('.details-content');
        const snapshot = this.monitor.getSnapshot();
        
        detailsContent.innerHTML = `
            <strong>FPS Details:</strong><br>
            Current: ${Math.round(snapshot.metrics.fps.current * 100) / 100}<br>
            Average: ${Math.round(snapshot.metrics.fps.average * 100) / 100}<br>
            Min: ${Math.round(snapshot.metrics.fps.min * 100) / 100}<br>
            Max: ${Math.round(snapshot.metrics.fps.max * 100) / 100}<br><br>
            
            <strong>Memory Details:</strong><br>
            Used: ${Math.round(snapshot.metrics.memory.used / 1024 / 1024)} MB<br>
            Total: ${Math.round(snapshot.metrics.memory.total / 1024 / 1024)} MB<br>
            Percentage: ${Math.round(snapshot.metrics.memory.percentage)}%<br>
            GC Events: ${snapshot.metrics.memory.gcEvents.length}<br><br>
            
            <strong>Rendering Details:</strong><br>
            Nodes: ${snapshot.metrics.rendering.nodesRendered}<br>
            Edges: ${snapshot.metrics.rendering.edgesRendered}<br>
            Culled: ${snapshot.metrics.rendering.culledNodes}<br>
            LOD Level: ${snapshot.metrics.rendering.lodLevel}<br><br>
            
            <strong>Optimization Suggestions:</strong><br>
            ${snapshot.suggestions.length > 0 ? 
                snapshot.suggestions.map(s => s.type).join(', ') : 
                'None'}<br>
        `;
    }
    
    /**
     * Export performance data
     */
    exportPerformanceData() {
        try {
            const data = this.monitor.exportData('json');
            const blob = new Blob([data], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `groggy-performance-${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            URL.revokeObjectURL(url);
            
            console.log('🎯 Performance data exported');
        } catch (error) {
            console.error('Failed to export performance data:', error);
        }
    }
}

// Global performance monitoring instance
window.GroggyPerformanceMonitor = null;
window.GroggyPerformanceDashboard = null;

/**
 * Initialize global performance monitoring
 */
function initializePerformanceMonitoring(config = {}) {
    if (window.GroggyPerformanceMonitor) {
        console.warn('Performance monitoring already initialized');
        return window.GroggyPerformanceMonitor;
    }
    
    // Create performance monitor
    window.GroggyPerformanceMonitor = new PerformanceMonitor(config);
    
    // Create dashboard container
    const dashboardContainer = document.createElement('div');
    dashboardContainer.id = 'groggy-performance-dashboard';
    document.body.appendChild(dashboardContainer);
    
    // Create dashboard
    window.GroggyPerformanceDashboard = new PerformanceDashboard(
        dashboardContainer, 
        window.GroggyPerformanceMonitor
    );
    
    // Add keyboard shortcut to toggle dashboard (Ctrl+Shift+P)
    document.addEventListener('keydown', (event) => {
        if (event.ctrlKey && event.shiftKey && event.key === 'P') {
            event.preventDefault();
            window.GroggyPerformanceDashboard.toggle();
        }
    });
    
    // Start monitoring
    window.GroggyPerformanceMonitor.start();
    
    console.log('🎯 Global performance monitoring initialized (Ctrl+Shift+P to toggle dashboard)');
    
    return window.GroggyPerformanceMonitor;
}

// Auto-initialize if in browser environment
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            initializePerformanceMonitoring();
        });
    } else {
        initializePerformanceMonitoring();
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        PerformanceMonitor,
        PerformanceDashboard,
        initializePerformanceMonitoring
    };
}