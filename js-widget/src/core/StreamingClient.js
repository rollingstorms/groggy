/**
 * 🌐 StreamingClient - Unified WebSocket Client using GroggyVizCore
 * 
 * WebSocket client that uses the same unified core as Jupyter widgets
 * for consistent visualization behavior across all environments.
 * 
 * Features:
 * - Real-time data streaming via WebSocket
 * - Automatic reconnection with exponential backoff
 * - Frame-based updates with efficient rendering
 * - Bidirectional communication (client events → server)
 * - Same physics simulation and interaction as Jupyter widgets
 * - Performance monitoring and adaptive quality
 */

import { GroggyVizCore } from './VizCore.js';

export class StreamingClient {
    constructor(config = {}) {
        this.config = {
            // WebSocket connection
            url: config.url || 'ws://localhost:8080',
            reconnectAttempts: config.reconnectAttempts || 10,
            reconnectDelay: config.reconnectDelay || 1000,
            maxReconnectDelay: config.maxReconnectDelay || 30000,
            
            // Performance settings
            maxFPS: config.maxFPS || 60,
            adaptiveQuality: config.adaptiveQuality !== false,
            qualityThresholds: config.qualityThresholds || {
                high: 50,    // FPS > 50 = high quality
                medium: 30,  // FPS > 30 = medium quality
                low: 15      // FPS > 15 = low quality
            },
            
            // Visualization config
            container: config.container || null,
            width: config.width || 800,
            height: config.height || 600,
            
            // Core visualization settings
            physics: {
                enabled: config.physics?.enabled !== false,
                forceStrength: config.physics?.forceStrength || 30,
                linkDistance: config.physics?.linkDistance || 50,
                linkStrength: config.physics?.linkStrength || 0.1,
                chargeStrength: config.physics?.chargeStrength || -300,
                centerStrength: config.physics?.centerStrength || 0.1,
                ...config.physics
            },
            
            rendering: {
                backgroundColor: config.rendering?.backgroundColor || '#ffffff',
                nodeColorScheme: config.rendering?.nodeColorScheme || 'default',
                enableShadows: config.rendering?.enableShadows !== false,
                enableAnimations: config.rendering?.enableAnimations !== false,
                enableLOD: config.rendering?.enableLOD !== false,
                ...config.rendering
            },
            
            interaction: {
                enableDrag: config.interaction?.enableDrag !== false,
                enableZoom: config.interaction?.enableZoom !== false,
                enablePan: config.interaction?.enablePan !== false,
                enableSelection: config.interaction?.enableSelection !== false,
                enableHover: config.interaction?.enableHover !== false,
                ...config.interaction
            },
            
            ...config
        };
        
        // WebSocket connection state
        this.websocket = null;
        this.isConnected = false;
        this.reconnectAttempt = 0;
        this.reconnectTimer = null;
        
        // Unified visualization core (same as Jupyter widgets!)
        this.vizCore = null;
        
        // Streaming state
        this.isStreaming = false;
        this.frameQueue = [];
        this.lastFrameTime = 0;
        this.frameRateMonitor = null;
        
        // Performance monitoring
        this.performanceMetrics = {
            framesReceived: 0,
            framesDropped: 0,
            averageLatency: 0,
            currentFPS: 0,
            currentQuality: 'high'
        };
        
        // Event handlers
        this.eventHandlers = {
            onConnect: null,
            onDisconnect: null,
            onError: null,
            onDataReceived: null,
            onFrameProcessed: null
        };
        
        // Bound methods
        this.handleWebSocketMessage = this.handleWebSocketMessage.bind(this);
        this.handleWebSocketOpen = this.handleWebSocketOpen.bind(this);
        this.handleWebSocketClose = this.handleWebSocketClose.bind(this);
        this.handleWebSocketError = this.handleWebSocketError.bind(this);
        this.processFrameQueue = this.processFrameQueue.bind(this);
        
        console.log('🌐 StreamingClient initialized with unified core architecture');
    }
    
    /**
     * 🔌 Connect to WebSocket server
     */
    connect() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            console.log('🌐 Already connected to WebSocket server');
            return Promise.resolve();
        }
        
        return new Promise((resolve, reject) => {
            try {
                console.log(`🌐 Connecting to ${this.config.url}...`);
                
                this.websocket = new WebSocket(this.config.url);
                
                // Set up event handlers
                this.websocket.onopen = (event) => {
                    this.handleWebSocketOpen(event);
                    resolve(event);
                };
                
                this.websocket.onclose = this.handleWebSocketClose;
                this.websocket.onerror = (error) => {
                    this.handleWebSocketError(error);
                    reject(error);
                };
                
                this.websocket.onmessage = this.handleWebSocketMessage;
                
            } catch (error) {
                console.error('🌐 Failed to create WebSocket connection:', error);
                reject(error);
            }
        });
    }
    
    /**
     * 🔌 Disconnect from WebSocket server
     */
    disconnect() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        this.isConnected = false;
        this.isStreaming = false;
        
        console.log('🌐 Disconnected from WebSocket server');
    }
    
    /**
     * 🎯 Initialize unified visualization core (same as Jupyter!)
     */
    initializeVizCore() {
        if (this.vizCore) {
            this.vizCore.destroy();
        }
        
        // Create unified core with same architecture as Jupyter widgets
        this.vizCore = new GroggyVizCore([], [], {
            width: this.config.width,
            height: this.config.height,
            physics: this.config.physics,
            rendering: this.config.rendering,
            interaction: this.config.interaction
        });
        
        // Set up event handlers for bidirectional communication
        this.setupVizCoreEventHandlers();
        
        // Attach to DOM if container is provided
        if (this.config.container) {
            this.vizCore.attachToDOM(this.config.container);
        }
        
        console.log('🌐 Unified VizCore initialized for streaming client');
    }
    
    /**
     * 🎧 Set up event handlers for client → server communication
     */
    setupVizCoreEventHandlers() {
        if (!this.vizCore) return;
        
        // Send node interactions to server
        this.vizCore.on('nodeClick', (data) => {
            this.sendMessage({
                type: 'interaction',
                interaction: 'node_click',
                nodeId: data.nodeId,
                timestamp: Date.now()
            });
        });
        
        this.vizCore.on('nodeHover', (data) => {
            this.sendMessage({
                type: 'interaction',
                interaction: 'node_hover',
                nodeId: data.nodeId,
                timestamp: Date.now()
            });
        });
        
        this.vizCore.on('selectionChanged', (data) => {
            this.sendMessage({
                type: 'interaction',
                interaction: 'selection_changed',
                selectedNodes: data.selectedNodes,
                timestamp: Date.now()
            });
        });
        
        // Send drag events to server
        this.vizCore.on('nodeDrag', (data) => {
            if (data.isDragging) {
                this.sendMessage({
                    type: 'interaction',
                    interaction: 'node_drag',
                    nodeId: data.nodeId,
                    position: data.position,
                    timestamp: Date.now()
                });
            }
        });
        
        // Send camera changes to server
        this.vizCore.on('zoom', (data) => {
            this.sendMessage({
                type: 'interaction',
                interaction: 'zoom',
                scale: data.scale,
                center: data.center,
                timestamp: Date.now()
            });
        });
        
        this.vizCore.on('pan', (data) => {
            this.sendMessage({
                type: 'interaction',
                interaction: 'pan',
                delta: data.delta,
                timestamp: Date.now()
            });
        });
    }
    
    /**
     * 📨 Handle incoming WebSocket messages
     */
    handleWebSocketMessage(event) {
        try {
            const message = JSON.parse(event.data);
            const receiveTime = Date.now();
            
            this.performanceMetrics.framesReceived++;
            
            // Calculate latency if timestamp is provided
            if (message.timestamp) {
                const latency = receiveTime - message.timestamp;
                this.updateLatencyMetrics(latency);
            }
            
            switch (message.type) {
                case 'frame_update':
                    this.handleFrameUpdate(message.data, receiveTime);
                    break;
                    
                case 'data_update':
                    this.handleDataUpdate(message.data);
                    break;
                    
                case 'config_update':
                    this.handleConfigUpdate(message.data);
                    break;
                    
                case 'interaction_response':
                    this.handleInteractionResponse(message.data);
                    break;
                    
                case 'status':
                    this.handleStatusMessage(message.data);
                    break;
                    
                default:
                    console.warn(`🌐 Unknown message type: ${message.type}`);
            }
            
            if (this.eventHandlers.onDataReceived) {
                this.eventHandlers.onDataReceived(message);
            }
            
        } catch (error) {
            console.error('🌐 Failed to parse WebSocket message:', error);
        }
    }
    
    /**
     * 🖼️ Handle frame updates from server
     */
    handleFrameUpdate(frameData, receiveTime) {
        // Add frame to queue with timing info
        this.frameQueue.push({
            data: frameData,
            receiveTime: receiveTime,
            processed: false
        });
        
        // Process frames immediately if not already processing
        if (this.frameQueue.length === 1) {
            this.processFrameQueue();
        }
    }
    
    /**
     * 🔄 Process queued frames with adaptive quality
     */
    processFrameQueue() {
        if (this.frameQueue.length === 0 || !this.vizCore) return;
        
        const currentTime = Date.now();
        const targetFrameTime = 1000 / this.config.maxFPS;
        
        // Check if enough time has passed since last frame
        if (currentTime - this.lastFrameTime < targetFrameTime) {
            requestAnimationFrame(this.processFrameQueue);
            return;
        }
        
        // Get next frame from queue
        const frame = this.frameQueue.shift();
        if (!frame) return;
        
        // Apply adaptive quality based on performance
        this.applyAdaptiveQuality();
        
        // Update unified core with frame data (same as Jupyter!)
        this.vizCore.update(frame.data);
        
        this.lastFrameTime = currentTime;
        frame.processed = true;
        
        // Update performance metrics
        this.updatePerformanceMetrics(currentTime);
        
        if (this.eventHandlers.onFrameProcessed) {
            this.eventHandlers.onFrameProcessed(frame);
        }
        
        // Continue processing queue
        if (this.frameQueue.length > 0) {
            requestAnimationFrame(this.processFrameQueue);
        }
    }
    
    /**
     * 📊 Handle data updates (new nodes/edges)
     */
    handleDataUpdate(data) {
        if (!this.vizCore) return;
        
        const { nodes, edges } = data;
        
        // Use unified core's setData method (same as Jupyter!)
        this.vizCore.setData(nodes || [], edges || []);
        
        console.log(`🌐 Updated data: ${nodes?.length || 0} nodes, ${edges?.length || 0} edges`);
    }
    
    /**
     * ⚙️ Handle configuration updates
     */
    handleConfigUpdate(configData) {
        if (!this.vizCore) return;
        
        // Update unified core configuration (same as Jupyter!)
        this.vizCore.updateConfig(configData);
        
        console.log('🌐 Updated configuration from server');
    }
    
    /**
     * 🖱️ Handle interaction responses from server
     */
    handleInteractionResponse(responseData) {
        // Server might send back interaction results or state changes
        if (responseData.type === 'selection_update') {
            if (this.vizCore) {
                this.vizCore.selectedNodes.clear();
                responseData.selectedNodes?.forEach(nodeId => {
                    this.vizCore.selectedNodes.add(nodeId);
                });
            }
        }
    }
    
    /**
     * 📊 Handle status messages
     */
    handleStatusMessage(statusData) {
        console.log('🌐 Server status:', statusData);
        
        if (statusData.type === 'performance') {
            // Server sending performance hints
            this.adjustQualityBasedOnServerLoad(statusData.serverLoad);
        }
    }
    
    /**
     * 📈 Apply adaptive quality based on performance
     */
    applyAdaptiveQuality() {
        if (!this.config.adaptiveQuality || !this.vizCore) return;
        
        const fps = this.performanceMetrics.currentFPS;
        const thresholds = this.config.qualityThresholds;
        
        let newQuality = 'low';
        if (fps > thresholds.high) {
            newQuality = 'high';
        } else if (fps > thresholds.medium) {
            newQuality = 'medium';
        } else if (fps > thresholds.low) {
            newQuality = 'low';
        }
        
        if (newQuality !== this.performanceMetrics.currentQuality) {
            this.performanceMetrics.currentQuality = newQuality;
            
            // Adjust rendering quality
            const qualityConfig = {
                high: {
                    enableShadows: true,
                    enableAnimations: true,
                    enableLOD: false,
                    lodThreshold: 1000
                },
                medium: {
                    enableShadows: true,
                    enableAnimations: false,
                    enableLOD: true,
                    lodThreshold: 500
                },
                low: {
                    enableShadows: false,
                    enableAnimations: false,
                    enableLOD: true,
                    lodThreshold: 200
                }
            };
            
            this.vizCore.updateConfig({
                rendering: qualityConfig[newQuality]
            });
            
            console.log(`🌐 Adaptive quality: ${newQuality} (FPS: ${fps})`);
        }
    }
    
    /**
     * 📊 Update performance metrics
     */
    updatePerformanceMetrics(currentTime) {
        // Calculate FPS
        if (this.lastFrameTime > 0) {
            const frameTime = currentTime - this.lastFrameTime;
            this.performanceMetrics.currentFPS = 1000 / frameTime;
        }
        
        // Check for dropped frames
        if (this.frameQueue.length > 5) {
            this.performanceMetrics.framesDropped++;
            this.frameQueue.splice(0, this.frameQueue.length - 2); // Keep only latest 2 frames
        }
    }
    
    /**
     * 📊 Update latency metrics
     */
    updateLatencyMetrics(latency) {
        this.performanceMetrics.averageLatency = 
            (this.performanceMetrics.averageLatency * 0.9) + (latency * 0.1);
    }
    
    /**
     * 📨 Send message to server
     */
    sendMessage(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            try {
                this.websocket.send(JSON.stringify(message));
            } catch (error) {
                console.error('🌐 Failed to send WebSocket message:', error);
            }
        }
    }
    
    /**
     * 🔌 Handle WebSocket connection open
     */
    handleWebSocketOpen(event) {
        console.log('🌐 Connected to WebSocket server');
        
        this.isConnected = true;
        this.reconnectAttempt = 0;
        
        // Initialize visualization core
        this.initializeVizCore();
        
        // Start streaming
        this.isStreaming = true;
        
        // Send initial handshake
        this.sendMessage({
            type: 'handshake',
            clientType: 'groggy-streaming-client',
            version: '1.0.0',
            capabilities: {
                supportsInteraction: true,
                supportsAdaptiveQuality: true,
                maxFPS: this.config.maxFPS
            }
        });
        
        if (this.eventHandlers.onConnect) {
            this.eventHandlers.onConnect(event);
        }
    }
    
    /**
     * 🔌 Handle WebSocket connection close
     */
    handleWebSocketClose(event) {
        console.log('🌐 WebSocket connection closed');
        
        this.isConnected = false;
        this.isStreaming = false;
        
        if (this.eventHandlers.onDisconnect) {
            this.eventHandlers.onDisconnect(event);
        }
        
        // Attempt reconnection if not manually closed
        if (event.code !== 1000 && this.reconnectAttempt < this.config.reconnectAttempts) {
            this.attemptReconnection();
        }
    }
    
    /**
     * 🔌 Handle WebSocket errors
     */
    handleWebSocketError(error) {
        console.error('🌐 WebSocket error:', error);
        
        if (this.eventHandlers.onError) {
            this.eventHandlers.onError(error);
        }
    }
    
    /**
     * 🔄 Attempt reconnection with exponential backoff
     */
    attemptReconnection() {
        this.reconnectAttempt++;
        
        const delay = Math.min(
            this.config.reconnectDelay * Math.pow(2, this.reconnectAttempt - 1),
            this.config.maxReconnectDelay
        );
        
        console.log(`🌐 Attempting reconnection ${this.reconnectAttempt}/${this.config.reconnectAttempts} in ${delay}ms`);
        
        this.reconnectTimer = setTimeout(() => {
            this.connect().catch(error => {
                console.error('🌐 Reconnection failed:', error);
            });
        }, delay);
    }
    
    /**
     * 📊 Get performance metrics
     */
    getPerformanceMetrics() {
        return {
            ...this.performanceMetrics,
            isConnected: this.isConnected,
            isStreaming: this.isStreaming,
            queueLength: this.frameQueue.length,
            reconnectAttempt: this.reconnectAttempt
        };
    }
    
    /**
     * 🎧 Set event handlers
     */
    setEventHandlers(handlers) {
        this.eventHandlers = { ...this.eventHandlers, ...handlers };
    }
    
    /**
     * 🧹 Cleanup and destroy
     */
    destroy() {
        this.disconnect();
        
        if (this.vizCore) {
            this.vizCore.destroy();
            this.vizCore = null;
        }
        
        if (this.frameRateMonitor) {
            clearInterval(this.frameRateMonitor);
        }
        
        this.frameQueue = [];
        
        console.log('🌐 StreamingClient destroyed');
    }
}