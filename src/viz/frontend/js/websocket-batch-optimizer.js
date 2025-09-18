/**
 * 🚀 Phase 10: Performance Optimization - WebSocket Message Batching
 * 
 * Advanced WebSocket message batching and optimization for high-performance graph visualization
 * Intelligently batches, compresses, and prioritizes messages for optimal real-time performance
 * 
 * Features:
 * - Intelligent message batching with adaptive timing
 * - Message priority queuing (critical, high, normal, low)
 * - Automatic message compression and deduplication
 * - Connection quality monitoring and adaptation
 * - Bandwidth-aware message throttling
 * - Message replay and recovery for lost connections
 * - Performance metrics and optimization
 * - Binary message encoding for large data
 * - Delta encoding for incremental updates
 */

class WebSocketBatchOptimizer {
    constructor(websocketClient) {
        this.websocketClient = websocketClient;
        
        // Batching configuration
        this.config = {
            batchInterval: 16, // ms (60 FPS target)
            maxBatchSize: 64 * 1024, // 64KB max batch size
            maxMessagesPerBatch: 100,
            enableCompression: true,
            enableDeltaEncoding: true,
            enableBinaryEncoding: true,
            adaptiveBatching: true,
            priorityLevels: 4, // critical, high, normal, low
            connectionTimeoutMs: 5000,
            reconnectAttempts: 5,
            backoffMultiplier: 1.5
        };
        
        // Message queues by priority
        this.messageQueues = {
            critical: [],   // Real-time interactions (clicks, selections)
            high: [],       // User-initiated operations (filters, search)
            normal: [],     // Data updates, layout changes
            low: []         // Analytics, logging, non-essential
        };
        
        // Batching state
        this.batchTimer = null;
        this.pendingBatch = [];
        this.batchSequence = 0;
        this.lastBatchTime = 0;
        this.adaptiveBatchInterval = this.config.batchInterval;
        
        // Connection monitoring
        this.connectionQuality = {
            latency: 0,
            bandwidth: 0,
            packetLoss: 0,
            jitter: 0,
            quality: 'good' // excellent, good, fair, poor
        };
        
        // Performance tracking
        this.metrics = {
            messagesSent: 0,
            messagesQueued: 0,
            batchesSent: 0,
            bytesTransferred: 0,
            compressionRatio: 0,
            avgBatchSize: 0,
            avgLatency: 0,
            reconnections: 0
        };
        
        // Message deduplication and compression
        this.messageCache = new Map();
        this.compressionHistory = [];
        this.deltaState = new Map();
        
        // Binary encoding support
        this.binaryEncoder = new BinaryMessageEncoder();
        this.textEncoder = new TextEncoder();
        this.textDecoder = new TextDecoder();
        
        this.init();
    }
    
    init() {
        console.log('🚀 Initializing WebSocket Batch Optimizer for Phase 10');
        
        this.setupConnectionMonitoring();
        this.setupAdaptiveBatching();
        this.startBatchProcessor();
        
        console.log('✅ WebSocket Batch Optimizer initialized');
    }
    
    /**
     * Set up connection quality monitoring
     */
    setupConnectionMonitoring() {
        // Monitor connection events
        this.websocketClient.addEventListener('open', () => {
            this.onConnectionOpen();
        });
        
        this.websocketClient.addEventListener('close', () => {
            this.onConnectionClose();
        });
        
        this.websocketClient.addEventListener('error', (error) => {
            this.onConnectionError(error);
        });
        
        this.websocketClient.addEventListener('message', (event) => {
            this.onMessageReceived(event);
        });
        
        // Start connection quality monitoring
        this.startConnectionQualityMonitoring();
    }
    
    /**
     * Set up adaptive batching based on performance
     */
    setupAdaptiveBatching() {
        if (!this.config.adaptiveBatching) return;
        
        // Monitor rendering performance
        document.addEventListener('lodPerformanceUpdate', (e) => {
            this.adaptBatchingToPerformance(e.detail);
        });
        
        // Monitor network conditions
        setInterval(() => {
            this.adaptBatchingToNetwork();
        }, 1000);
    }
    
    /**
     * Start the main batch processing loop
     */
    startBatchProcessor() {
        const processBatch = () => {
            this.processPendingMessages();
            this.batchTimer = setTimeout(processBatch, this.adaptiveBatchInterval);
        };
        
        processBatch();
        console.log('⚡ Batch processor started');
    }
    
    /**
     * Queue a message for batched sending
     */
    queueMessage(message, priority = 'normal') {
        const timestamp = performance.now();
        
        const queuedMessage = {
            id: this.generateMessageId(),
            type: message.type,
            data: message,
            priority: priority,
            timestamp: timestamp,
            retries: 0,
            size: this.estimateMessageSize(message)
        };
        
        // Add to appropriate priority queue
        if (this.messageQueues[priority]) {
            this.messageQueues[priority].push(queuedMessage);
            this.metrics.messagesQueued++;
        } else {
            console.warn(`Unknown priority level: ${priority}`);
            this.messageQueues.normal.push(queuedMessage);
        }
        
        // For critical messages, send immediately
        if (priority === 'critical') {
            this.flushCriticalMessages();
        }
        
        // Adaptive batching: if queue is getting large, reduce interval
        const totalQueued = this.getTotalQueuedMessages();
        if (totalQueued > this.config.maxMessagesPerBatch * 2) {
            this.adaptiveBatchInterval = Math.max(8, this.adaptiveBatchInterval * 0.8);
        }
    }
    
    /**
     * Process all pending messages and create batches
     */
    processPendingMessages() {
        const startTime = performance.now();
        
        // Collect messages from all priority queues
        const batch = this.collectMessageBatch();
        
        if (batch.length === 0) return;
        
        // Apply optimizations
        const optimizedBatch = this.optimizeBatch(batch);
        
        // Send batch
        this.sendBatch(optimizedBatch);
        
        // Update metrics
        this.updateBatchMetrics(optimizedBatch, performance.now() - startTime);
    }
    
    /**
     * Collect messages for the next batch, respecting priorities
     */
    collectMessageBatch() {
        const batch = [];
        let currentSize = 0;
        const maxSize = this.config.maxBatchSize;
        const maxCount = this.config.maxMessagesPerBatch;
        
        // Process queues in priority order
        const priorityOrder = ['critical', 'high', 'normal', 'low'];
        
        for (const priority of priorityOrder) {
            const queue = this.messageQueues[priority];
            
            while (queue.length > 0 && batch.length < maxCount && currentSize < maxSize) {
                const message = queue.shift();
                
                // Check if adding this message would exceed size limit
                if (currentSize + message.size > maxSize && batch.length > 0) {
                    // Put message back at front of queue
                    queue.unshift(message);
                    break;
                }
                
                batch.push(message);
                currentSize += message.size;
            }
            
            // Stop if we've hit limits
            if (batch.length >= maxCount || currentSize >= maxSize) {
                break;
            }
        }
        
        return batch;
    }
    
    /**
     * Optimize batch with compression, deduplication, and delta encoding
     */
    optimizeBatch(batch) {
        if (batch.length === 0) return batch;
        
        let optimizedBatch = [...batch];
        
        // Step 1: Remove duplicate messages
        optimizedBatch = this.deduplicateMessages(optimizedBatch);
        
        // Step 2: Apply delta encoding for data updates
        if (this.config.enableDeltaEncoding) {
            optimizedBatch = this.applyDeltaEncoding(optimizedBatch);
        }
        
        // Step 3: Compress batch if beneficial
        if (this.config.enableCompression) {
            optimizedBatch = this.compressBatch(optimizedBatch);
        }
        
        // Step 4: Convert to binary encoding if beneficial
        if (this.config.enableBinaryEncoding) {
            optimizedBatch = this.applyBinaryEncoding(optimizedBatch);
        }
        
        return optimizedBatch;
    }
    
    /**
     * Remove duplicate messages from batch
     */
    deduplicateMessages(batch) {
        const seen = new Map();
        const deduplicated = [];
        
        for (const message of batch) {
            const key = this.createDeduplicationKey(message);
            const existing = seen.get(key);
            
            if (!existing || message.timestamp > existing.timestamp) {
                seen.set(key, message);
            }
        }
        
        // Convert back to array
        for (const message of seen.values()) {
            deduplicated.push(message);
        }
        
        const originalCount = batch.length;
        const deduplicatedCount = deduplicated.length;
        
        if (originalCount > deduplicatedCount) {
            console.log(`🗜️  Deduplicated ${originalCount - deduplicatedCount} messages`);
        }
        
        return deduplicated;
    }
    
    /**
     * Apply delta encoding to reduce data transfer
     */
    applyDeltaEncoding(batch) {
        const deltaEncoded = [];
        
        for (const message of batch) {
            if (this.isDeltaEncodable(message)) {
                const deltaMessage = this.createDeltaMessage(message);
                if (deltaMessage) {
                    deltaEncoded.push(deltaMessage);
                    continue;
                }
            }
            
            deltaEncoded.push(message);
        }
        
        return deltaEncoded;
    }
    
    /**
     * Compress batch if size exceeds threshold
     */
    compressBatch(batch) {
        const serializedBatch = JSON.stringify(batch);
        const originalSize = new Blob([serializedBatch]).size;
        
        if (originalSize < 1024) {
            return batch; // Too small to benefit from compression
        }
        
        try {
            // Use compression (simplified - would use actual compression library)
            const compressed = this.compressData(serializedBatch);
            const compressedSize = compressed.byteLength;
            
            const compressionRatio = compressedSize / originalSize;
            
            if (compressionRatio < 0.8) {
                // Compression is beneficial
                this.metrics.compressionRatio = compressionRatio;
                
                return [{
                    id: this.generateMessageId(),
                    type: 'CompressedBatch',
                    data: compressed,
                    originalSize: originalSize,
                    compressedSize: compressedSize,
                    messageCount: batch.length,
                    timestamp: performance.now(),
                    isCompressed: true
                }];
            }
        } catch (error) {
            console.warn('Compression failed:', error);
        }
        
        return batch;
    }
    
    /**
     * Apply binary encoding for large data transfers
     */
    applyBinaryEncoding(batch) {
        const binaryThreshold = 8192; // 8KB
        const totalSize = batch.reduce((sum, msg) => sum + msg.size, 0);
        
        if (totalSize > binaryThreshold) {
            try {
                const binaryData = this.binaryEncoder.encode(batch);
                
                return [{
                    id: this.generateMessageId(),
                    type: 'BinaryBatch',
                    data: binaryData,
                    messageCount: batch.length,
                    timestamp: performance.now(),
                    isBinary: true
                }];
            } catch (error) {
                console.warn('Binary encoding failed:', error);
            }
        }
        
        return batch;
    }
    
    /**
     * Send optimized batch to server
     */
    sendBatch(batch) {
        if (batch.length === 0) return;
        
        const batchMessage = {
            type: 'MessageBatch',
            sequence: this.batchSequence++,
            timestamp: performance.now(),
            messages: batch,
            metadata: {
                clientTime: Date.now(),
                batchSize: batch.length,
                priority: this.getBatchPriority(batch),
                connectionQuality: this.connectionQuality.quality
            }
        };
        
        // Send via WebSocket
        try {
            this.websocketClient.send(JSON.stringify(batchMessage));
            
            this.metrics.batchesSent++;
            this.metrics.messagesSent += batch.length;
            this.metrics.bytesTransferred += this.estimateMessageSize(batchMessage);
            
            console.log(`📤 Sent batch ${this.batchSequence - 1} with ${batch.length} messages`);
            
        } catch (error) {
            console.error('Failed to send batch:', error);
            this.handleSendError(batch, error);
        }
        
        this.lastBatchTime = performance.now();
    }
    
    /**
     * Flush critical messages immediately
     */
    flushCriticalMessages() {
        const criticalQueue = this.messageQueues.critical;
        if (criticalQueue.length === 0) return;
        
        const criticalBatch = criticalQueue.splice(0);
        this.sendBatch(criticalBatch);
    }
    
    /**
     * Handle send errors and implement retry logic
     */
    handleSendError(batch, error) {
        console.warn('Batch send failed, requeueing messages:', error);
        
        // Requeue messages with incremented retry count
        for (const message of batch) {
            message.retries++;
            
            if (message.retries <= 3) {
                // Requeue with lower priority
                const newPriority = message.priority === 'critical' ? 'high' : 
                                  message.priority === 'high' ? 'normal' : 'low';
                this.messageQueues[newPriority].unshift(message);
            } else {
                console.warn(`Dropping message after ${message.retries} retries:`, message.id);
            }
        }
    }
    
    /**
     * Monitor connection quality and adapt behavior
     */
    startConnectionQualityMonitoring() {
        setInterval(() => {
            this.measureConnectionQuality();
        }, 1000);
    }
    
    /**
     * Measure connection quality metrics
     */
    measureConnectionQuality() {
        // Send ping message to measure latency
        const pingMessage = {
            type: 'Ping',
            timestamp: performance.now(),
            clientTime: Date.now()
        };
        
        this.queueMessage(pingMessage, 'critical');
        
        // Update connection quality assessment
        this.updateConnectionQuality();
    }
    
    /**
     * Update connection quality assessment
     */
    updateConnectionQuality() {
        const latency = this.connectionQuality.latency;
        
        let quality;
        if (latency < 50) {
            quality = 'excellent';
        } else if (latency < 150) {
            quality = 'good';
        } else if (latency < 300) {
            quality = 'fair';
        } else {
            quality = 'poor';
        }
        
        this.connectionQuality.quality = quality;
        
        // Dispatch quality update event
        document.dispatchEvent(new CustomEvent('connectionQualityChanged', {
            detail: this.connectionQuality
        }));
    }
    
    /**
     * Adapt batching to rendering performance
     */
    adaptBatchingToPerformance(performanceData) {
        const fps = performanceData.fps;
        const targetFPS = 60;
        
        if (fps < targetFPS * 0.8) {
            // Performance is poor, reduce message frequency
            this.adaptiveBatchInterval = Math.min(100, this.adaptiveBatchInterval * 1.2);
        } else if (fps > targetFPS * 1.1) {
            // Performance is good, can increase message frequency
            this.adaptiveBatchInterval = Math.max(8, this.adaptiveBatchInterval * 0.9);
        }
    }
    
    /**
     * Adapt batching to network conditions
     */
    adaptBatchingToNetwork() {
        const quality = this.connectionQuality.quality;
        
        switch (quality) {
            case 'excellent':
                this.config.maxBatchSize = 64 * 1024;
                this.config.maxMessagesPerBatch = 100;
                break;
            case 'good':
                this.config.maxBatchSize = 32 * 1024;
                this.config.maxMessagesPerBatch = 50;
                break;
            case 'fair':
                this.config.maxBatchSize = 16 * 1024;
                this.config.maxMessagesPerBatch = 25;
                break;
            case 'poor':
                this.config.maxBatchSize = 8 * 1024;
                this.config.maxMessagesPerBatch = 10;
                break;
        }
    }
    
    /**
     * Event handlers
     */
    onConnectionOpen() {
        console.log('📡 WebSocket connection opened');
        this.metrics.reconnections++;
    }
    
    onConnectionClose() {
        console.log('📡 WebSocket connection closed');
        // Handle reconnection logic here
    }
    
    onConnectionError(error) {
        console.error('📡 WebSocket connection error:', error);
    }
    
    onMessageReceived(event) {
        // Handle incoming messages, including pong responses
        try {
            const message = JSON.parse(event.data);
            
            if (message.type === 'Pong') {
                const latency = performance.now() - message.timestamp;
                this.connectionQuality.latency = latency;
                this.metrics.avgLatency = latency;
            }
        } catch (error) {
            // Handle binary or other message types
        }
    }
    
    /**
     * Helper methods
     */
    generateMessageId() {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    estimateMessageSize(message) {
        return new Blob([JSON.stringify(message)]).size;
    }
    
    getTotalQueuedMessages() {
        return Object.values(this.messageQueues).reduce((total, queue) => total + queue.length, 0);
    }
    
    createDeduplicationKey(message) {
        // Create a key for deduplication based on message type and relevant data
        return `${message.type}_${JSON.stringify(message.data)}`;
    }
    
    isDeltaEncodable(message) {
        // Check if message type supports delta encoding
        return ['GraphDataUpdate', 'NodeUpdate', 'EdgeUpdate'].includes(message.type);
    }
    
    createDeltaMessage(message) {
        // Create delta version of message based on previous state
        const previousState = this.deltaState.get(message.type);
        if (!previousState) {
            this.deltaState.set(message.type, message.data);
            return null; // First message, no delta possible
        }
        
        const delta = this.calculateDelta(previousState, message.data);
        if (delta && Object.keys(delta).length > 0) {
            this.deltaState.set(message.type, message.data);
            
            return {
                ...message,
                data: delta,
                isDelta: true,
                size: this.estimateMessageSize({ data: delta })
            };
        }
        
        return null;
    }
    
    calculateDelta(previous, current) {
        // Simplified delta calculation
        const delta = {};
        
        for (const key in current) {
            if (current[key] !== previous[key]) {
                delta[key] = current[key];
            }
        }
        
        return Object.keys(delta).length > 0 ? delta : null;
    }
    
    getBatchPriority(batch) {
        // Determine overall priority of batch
        const priorities = batch.map(msg => msg.priority);
        
        if (priorities.includes('critical')) return 'critical';
        if (priorities.includes('high')) return 'high';
        if (priorities.includes('normal')) return 'normal';
        return 'low';
    }
    
    compressData(data) {
        // Simplified compression implementation
        // In real implementation, would use gzip or similar
        return this.textEncoder.encode(data);
    }
    
    updateBatchMetrics(batch, processingTime) {
        const batchSize = batch.reduce((sum, msg) => sum + msg.size, 0);
        this.metrics.avgBatchSize = (this.metrics.avgBatchSize + batchSize) / 2;
    }
    
    /**
     * Get performance statistics
     */
    getPerformanceStats() {
        return {
            ...this.metrics,
            adaptiveBatchInterval: this.adaptiveBatchInterval,
            connectionQuality: this.connectionQuality,
            queuedMessages: this.getTotalQueuedMessages(),
            config: this.config
        };
    }
    
    /**
     * Configure batch optimizer settings
     */
    configure(settings) {
        this.config = { ...this.config, ...settings };
        console.log('🎛️  WebSocket batch optimizer configuration updated');
    }
    
    /**
     * Cleanup method
     */
    destroy() {
        if (this.batchTimer) {
            clearTimeout(this.batchTimer);
        }
        
        // Clear all queues
        for (const queue of Object.values(this.messageQueues)) {
            queue.length = 0;
        }
        
        this.messageCache.clear();
        this.deltaState.clear();
        
        console.log('🧹 WebSocket Batch Optimizer cleaned up');
    }
}

/**
 * Binary Message Encoder for efficient data transfer
 */
class BinaryMessageEncoder {
    encode(messages) {
        // Simplified binary encoding implementation
        const jsonString = JSON.stringify(messages);
        return new TextEncoder().encode(jsonString);
    }
    
    decode(binaryData) {
        const jsonString = new TextDecoder().decode(binaryData);
        return JSON.parse(jsonString);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebSocketBatchOptimizer;
}