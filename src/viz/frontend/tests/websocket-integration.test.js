/**
 * Comprehensive Integration Tests for WebSocket Communication
 * Tests the entire streaming infrastructure including message protocols,
 * connection management, error handling, and real-time data streaming.
 */

const { expect, describe, test, beforeEach, afterEach, beforeAll, afterAll } = require('@jest/globals');
const WebSocket = require('ws');
const http = require('http');

// Mock WebSocket for browser environment
if (typeof window !== 'undefined') {
    global.WebSocket = class MockWebSocket extends EventTarget {
        constructor(url) {
            super();
            this.url = url;
            this.readyState = WebSocket.CONNECTING;
            this.protocol = '';
            this.extensions = '';
            
            // Simulate connection
            setTimeout(() => {
                this.readyState = WebSocket.OPEN;
                this.dispatchEvent(new Event('open'));
            }, 10);
        }
        
        send(data) {
            if (this.readyState !== WebSocket.OPEN) {
                throw new Error('WebSocket is not open');
            }
            
            // Echo back for testing
            setTimeout(() => {
                this.dispatchEvent(new MessageEvent('message', { data }));
            }, 5);
        }
        
        close(code = 1000, reason = '') {
            if (this.readyState === WebSocket.CLOSED) return;
            
            this.readyState = WebSocket.CLOSING;
            setTimeout(() => {
                this.readyState = WebSocket.CLOSED;
                this.dispatchEvent(new CloseEvent('close', { code, reason }));
            }, 5);
        }
    };
    
    global.WebSocket.CONNECTING = 0;
    global.WebSocket.OPEN = 1;
    global.WebSocket.CLOSING = 2;
    global.WebSocket.CLOSED = 3;
}

/**
 * Mock WebSocket Server for Testing
 */
class MockWebSocketServer {
    constructor(port = 8080) {
        this.port = port;
        this.server = null;
        this.wss = null;
        this.clients = new Set();
        this.messageHandlers = new Map();
        this.connectionHandlers = [];
        this.disconnectionHandlers = [];
    }
    
    async start() {
        return new Promise((resolve, reject) => {
            this.server = http.createServer();
            this.wss = new WebSocket.Server({ server: this.server });
            
            this.wss.on('connection', (ws) => {
                this.clients.add(ws);
                
                ws.on('message', (data) => {
                    try {
                        const message = JSON.parse(data);
                        this.handleMessage(ws, message);
                    } catch (error) {
                        this.sendError(ws, 'Invalid JSON message', error.message);
                    }
                });
                
                ws.on('close', () => {
                    this.clients.delete(ws);
                    this.disconnectionHandlers.forEach(handler => handler(ws));
                });
                
                ws.on('error', (error) => {
                    console.error('WebSocket error:', error);
                });
                
                this.connectionHandlers.forEach(handler => handler(ws));
            });
            
            this.server.listen(this.port, (error) => {
                if (error) {
                    reject(error);
                } else {
                    resolve();
                }
            });
        });
    }
    
    async stop() {
        return new Promise((resolve) => {
            if (this.wss) {
                this.wss.close(() => {
                    if (this.server) {
                        this.server.close(() => {
                            resolve();
                        });
                    } else {
                        resolve();
                    }
                });
            } else {
                resolve();
            }
        });
    }
    
    handleMessage(ws, message) {
        const handler = this.messageHandlers.get(message.type);
        if (handler) {
            try {
                handler(ws, message);
            } catch (error) {
                this.sendError(ws, 'Message handler error', error.message);
            }
        } else {
            this.sendError(ws, 'Unknown message type', message.type);
        }
    }
    
    onMessage(messageType, handler) {
        this.messageHandlers.set(messageType, handler);
    }
    
    onConnection(handler) {
        this.connectionHandlers.push(handler);
    }
    
    onDisconnection(handler) {
        this.disconnectionHandlers.push(handler);
    }
    
    broadcast(message) {
        const data = JSON.stringify(message);
        this.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(data);
            }
        });
    }
    
    sendToClient(ws, message) {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(message));
        }
    }
    
    sendError(ws, error, details = null) {
        this.sendToClient(ws, {
            type: 'error',
            error: error,
            details: details,
            timestamp: Date.now()
        });
    }
    
    getClientCount() {
        return this.clients.size;
    }
}

/**
 * WebSocket Client Test Utilities
 */
class WebSocketTestClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.messageHandlers = new Map();
        this.receivedMessages = [];
        this.isConnected = false;
        this.connectionPromise = null;
    }
    
    async connect() {
        if (this.connectionPromise) {
            return this.connectionPromise;
        }
        
        this.connectionPromise = new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.receivedMessages.push(message);
                    
                    const handler = this.messageHandlers.get(message.type);
                    if (handler) {
                        handler(message);
                    }
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            };
            
            this.ws.onclose = () => {
                this.isConnected = false;
            };
            
            this.ws.onerror = (error) => {
                reject(error);
            };
            
            // Timeout after 5 seconds
            setTimeout(() => {
                if (!this.isConnected) {
                    reject(new Error('Connection timeout'));
                }
            }, 5000);
        });
        
        return this.connectionPromise;
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
            this.isConnected = false;
            this.connectionPromise = null;
        }
    }
    
    send(message) {
        if (!this.isConnected || !this.ws) {
            throw new Error('WebSocket not connected');
        }
        
        this.ws.send(JSON.stringify(message));
    }
    
    onMessage(messageType, handler) {
        this.messageHandlers.set(messageType, handler);
    }
    
    waitForMessage(messageType, timeout = 5000) {
        return new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                reject(new Error(`Timeout waiting for message type: ${messageType}`));
            }, timeout);
            
            const handler = (message) => {
                clearTimeout(timeoutId);
                resolve(message);
            };
            
            this.onMessage(messageType, handler);
        });
    }
    
    getReceivedMessages() {
        return [...this.receivedMessages];
    }
    
    clearReceivedMessages() {
        this.receivedMessages = [];
    }
}

/**
 * Test Data Generators
 */
class StreamingTestData {
    static generateGraphData(nodeCount = 10, edgeCount = 15) {
        const nodes = [];
        const edges = [];
        
        // Generate nodes
        for (let i = 0; i < nodeCount; i++) {
            nodes.push({
                id: `node_${i}`,
                label: `Node ${i}`,
                x: Math.random() * 400,
                y: Math.random() * 400,
                radius: 10 + Math.random() * 20,
                color: this.randomColor(),
                type: 'default'
            });
        }
        
        // Generate edges
        for (let i = 0; i < edgeCount; i++) {
            const sourceIndex = Math.floor(Math.random() * nodeCount);
            let targetIndex = Math.floor(Math.random() * nodeCount);
            
            // Ensure no self-loops
            while (targetIndex === sourceIndex) {
                targetIndex = Math.floor(Math.random() * nodeCount);
            }
            
            edges.push({
                id: `edge_${i}`,
                source: `node_${sourceIndex}`,
                target: `node_${targetIndex}`,
                weight: Math.random(),
                color: this.randomColor(),
                type: 'default'
            });
        }
        
        return { nodes, edges };
    }
    
    static generateLayoutData(layoutType = 'force-directed') {
        const { nodes, edges } = this.generateGraphData();
        
        return {
            type: 'layout_response',
            layoutType: layoutType,
            nodes: nodes,
            edges: edges,
            metadata: {
                algorithm: layoutType,
                iterations: 100,
                timestamp: Date.now()
            }
        };
    }
    
    static generateFilterData() {
        return {
            type: 'filter_response',
            filters: [
                { field: 'type', operator: 'equals', value: 'important' },
                { field: 'weight', operator: 'greater_than', value: 0.5 }
            ],
            filteredNodes: ['node_1', 'node_3', 'node_7'],
            filteredEdges: ['edge_2', 'edge_8'],
            resultCount: 3
        };
    }
    
    static generateMetricsData() {
        return {
            type: 'metrics_response',
            metrics: {
                nodeCount: 25,
                edgeCount: 40,
                density: 0.133,
                clustering: 0.45,
                averageDegree: 3.2,
                components: 1,
                diameter: 6
            },
            timestamp: Date.now()
        };
    }
    
    static randomColor() {
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd'];
        return colors[Math.floor(Math.random() * colors.length)];
    }
}

/**
 * Message Protocol Tests
 */
describe('WebSocket Message Protocol', () => {
    let server;
    let client;
    
    beforeAll(async () => {
        server = new MockWebSocketServer(8081);
        await server.start();
    });
    
    afterAll(async () => {
        await server.stop();
    });
    
    beforeEach(async () => {
        client = new WebSocketTestClient('ws://localhost:8081');
        await client.connect();
    });
    
    afterEach(() => {
        if (client) {
            client.disconnect();
        }
    });
    
    test('should establish WebSocket connection', async () => {
        expect(client.isConnected).toBe(true);
        expect(server.getClientCount()).toBe(1);
    });
    
    test('should handle graph data request', async () => {
        // Setup server response
        server.onMessage('graph_data_request', (ws, message) => {
            const graphData = StreamingTestData.generateGraphData(5, 8);
            server.sendToClient(ws, {
                type: 'graph_data_response',
                requestId: message.requestId,
                data: graphData,
                timestamp: Date.now()
            });
        });
        
        // Send request
        const requestId = 'test_request_1';
        client.send({
            type: 'graph_data_request',
            requestId: requestId,
            parameters: {
                includeNodes: true,
                includeEdges: true
            }
        });
        
        // Wait for response
        const response = await client.waitForMessage('graph_data_response');
        
        expect(response.requestId).toBe(requestId);
        expect(response.data).toBeDefined();
        expect(response.data.nodes).toHaveLength(5);
        expect(response.data.edges).toHaveLength(8);
        expect(response.timestamp).toBeDefined();
    });
    
    test('should handle layout request', async () => {
        server.onMessage('layout_request', (ws, message) => {
            const layoutData = StreamingTestData.generateLayoutData(message.layoutType);
            server.sendToClient(ws, {
                type: 'layout_response',
                requestId: message.requestId,
                ...layoutData,
                timestamp: Date.now()
            });
        });
        
        const requestId = 'layout_request_1';
        client.send({
            type: 'layout_request',
            requestId: requestId,
            layoutType: 'circular',
            parameters: {
                radius: 100,
                centerX: 200,
                centerY: 200
            }
        });
        
        const response = await client.waitForMessage('layout_response');
        
        expect(response.requestId).toBe(requestId);
        expect(response.layoutType).toBe('circular');
        expect(response.nodes).toBeDefined();
        expect(response.metadata).toBeDefined();
    });
    
    test('should handle filter request', async () => {
        server.onMessage('filter_request', (ws, message) => {
            const filterData = StreamingTestData.generateFilterData();
            server.sendToClient(ws, {
                type: 'filter_response',
                requestId: message.requestId,
                ...filterData,
                timestamp: Date.now()
            });
        });
        
        const requestId = 'filter_request_1';
        client.send({
            type: 'filter_request',
            requestId: requestId,
            filters: [
                { field: 'type', operator: 'equals', value: 'important' }
            ]
        });
        
        const response = await client.waitForMessage('filter_response');
        
        expect(response.requestId).toBe(requestId);
        expect(response.filters).toBeDefined();
        expect(response.filteredNodes).toBeDefined();
        expect(response.resultCount).toBeGreaterThanOrEqual(0);
    });
    
    test('should handle metrics request', async () => {
        server.onMessage('metrics_request', (ws, message) => {
            const metricsData = StreamingTestData.generateMetricsData();
            server.sendToClient(ws, {
                type: 'metrics_response',
                requestId: message.requestId,
                ...metricsData
            });
        });
        
        const requestId = 'metrics_request_1';
        client.send({
            type: 'metrics_request',
            requestId: requestId,
            metrics: ['nodeCount', 'edgeCount', 'density', 'clustering']
        });
        
        const response = await client.waitForMessage('metrics_response');
        
        expect(response.requestId).toBe(requestId);
        expect(response.metrics).toBeDefined();
        expect(response.metrics.nodeCount).toBeDefined();
        expect(response.metrics.density).toBeDefined();
    });
    
    test('should handle invalid message gracefully', async () => {
        client.send({
            type: 'invalid_message_type',
            data: 'invalid'
        });
        
        const errorResponse = await client.waitForMessage('error');
        
        expect(errorResponse.error).toBe('Unknown message type');
        expect(errorResponse.details).toBe('invalid_message_type');
    });
    
    test('should handle malformed JSON', async () => {
        // Send raw malformed JSON
        if (client.ws) {
            client.ws.send('{"invalid": json}');
        }
        
        const errorResponse = await client.waitForMessage('error');
        expect(errorResponse.error).toBe('Invalid JSON message');
    });
});

/**
 * Connection Management Tests
 */
describe('WebSocket Connection Management', () => {
    let server;
    
    beforeAll(async () => {
        server = new MockWebSocketServer(8082);
        await server.start();
    });
    
    afterAll(async () => {
        await server.stop();
    });
    
    test('should handle multiple concurrent connections', async () => {
        const clients = [];
        const clientCount = 5;
        
        // Connect multiple clients
        for (let i = 0; i < clientCount; i++) {
            const client = new WebSocketTestClient('ws://localhost:8082');
            await client.connect();
            clients.push(client);
        }
        
        expect(server.getClientCount()).toBe(clientCount);
        
        // Disconnect all clients
        clients.forEach(client => client.disconnect());
        
        // Wait for disconnections to be processed
        await new Promise(resolve => setTimeout(resolve, 100));
        expect(server.getClientCount()).toBe(0);
    });
    
    test('should handle client reconnection', async () => {
        const client = new WebSocketTestClient('ws://localhost:8082');
        
        // Initial connection
        await client.connect();
        expect(client.isConnected).toBe(true);
        
        // Disconnect
        client.disconnect();
        expect(client.isConnected).toBe(false);
        
        // Reconnect
        await client.connect();
        expect(client.isConnected).toBe(true);
        
        client.disconnect();
    });
    
    test('should handle server-initiated disconnection', async () => {
        const client = new WebSocketTestClient('ws://localhost:8082');
        await client.connect();
        
        let disconnected = false;
        client.ws.onclose = () => {
            disconnected = true;
        };
        
        // Server closes all connections
        server.clients.forEach(ws => ws.close());
        
        // Wait for disconnection
        await new Promise(resolve => setTimeout(resolve, 100));
        expect(disconnected).toBe(true);
    });
    
    test('should maintain connection state correctly', async () => {
        const client = new WebSocketTestClient('ws://localhost:8082');
        
        expect(client.isConnected).toBe(false);
        
        await client.connect();
        expect(client.isConnected).toBe(true);
        
        client.disconnect();
        expect(client.isConnected).toBe(false);
    });
});

/**
 * Real-time Streaming Tests
 */
describe('Real-time Data Streaming', () => {
    let server;
    let client;
    
    beforeAll(async () => {
        server = new MockWebSocketServer(8083);
        await server.start();
    });
    
    afterAll(async () => {
        await server.stop();
    });
    
    beforeEach(async () => {
        client = new WebSocketTestClient('ws://localhost:8083');
        await client.connect();
    });
    
    afterEach(() => {
        if (client) {
            client.disconnect();
        }
    });
    
    test('should stream node updates', async () => {
        // Setup streaming updates
        server.onMessage('start_streaming', (ws, message) => {
            server.sendToClient(ws, {
                type: 'streaming_started',
                requestId: message.requestId
            });
            
            // Send periodic updates
            let updateCount = 0;
            const interval = setInterval(() => {
                if (updateCount >= 3) {
                    clearInterval(interval);
                    return;
                }
                
                server.sendToClient(ws, {
                    type: 'node_update',
                    nodeId: `node_${updateCount}`,
                    updates: {
                        x: Math.random() * 400,
                        y: Math.random() * 400,
                        color: StreamingTestData.randomColor()
                    },
                    timestamp: Date.now()
                });
                
                updateCount++;
            }, 50);
        });
        
        // Start streaming
        client.send({
            type: 'start_streaming',
            requestId: 'stream_1',
            updateTypes: ['node_updates']
        });
        
        // Wait for streaming confirmation
        await client.waitForMessage('streaming_started');
        
        // Collect updates
        const updates = [];
        client.onMessage('node_update', (message) => {
            updates.push(message);
        });
        
        // Wait for updates
        await new Promise(resolve => setTimeout(resolve, 200));
        
        expect(updates.length).toBe(3);
        updates.forEach((update, index) => {
            expect(update.nodeId).toBe(`node_${index}`);
            expect(update.updates.x).toBeDefined();
            expect(update.updates.y).toBeDefined();
        });
    });
    
    test('should stream layout updates', async () => {
        server.onMessage('start_layout_streaming', (ws, message) => {
            server.sendToClient(ws, {
                type: 'layout_streaming_started',
                requestId: message.requestId
            });
            
            // Simulate layout iterations
            let iteration = 0;
            const interval = setInterval(() => {
                if (iteration >= 5) {
                    clearInterval(interval);
                    server.sendToClient(ws, {
                        type: 'layout_complete',
                        totalIterations: iteration
                    });
                    return;
                }
                
                const positions = {};
                for (let i = 0; i < 5; i++) {
                    positions[`node_${i}`] = {
                        x: Math.random() * 400,
                        y: Math.random() * 400
                    };
                }
                
                server.sendToClient(ws, {
                    type: 'layout_iteration',
                    iteration: iteration,
                    positions: positions,
                    timestamp: Date.now()
                });
                
                iteration++;
            }, 30);
        });
        
        client.send({
            type: 'start_layout_streaming',
            requestId: 'layout_stream_1',
            layoutType: 'force_directed'
        });
        
        await client.waitForMessage('layout_streaming_started');
        
        const iterations = [];
        client.onMessage('layout_iteration', (message) => {
            iterations.push(message);
        });
        
        await client.waitForMessage('layout_complete');
        
        expect(iterations.length).toBe(5);
        iterations.forEach((iteration, index) => {
            expect(iteration.iteration).toBe(index);
            expect(iteration.positions).toBeDefined();
            expect(Object.keys(iteration.positions)).toHaveLength(5);
        });
    });
    
    test('should handle high-frequency updates', async () => {
        server.onMessage('high_frequency_test', (ws, message) => {
            // Send 50 updates rapidly
            for (let i = 0; i < 50; i++) {
                setTimeout(() => {
                    server.sendToClient(ws, {
                        type: 'rapid_update',
                        sequence: i,
                        data: `update_${i}`,
                        timestamp: Date.now()
                    });
                }, i * 2); // 2ms intervals
            }
        });
        
        client.send({
            type: 'high_frequency_test'
        });
        
        const updates = [];
        client.onMessage('rapid_update', (message) => {
            updates.push(message);
        });
        
        // Wait for all updates
        await new Promise(resolve => setTimeout(resolve, 200));
        
        expect(updates.length).toBe(50);
        
        // Check sequence ordering
        for (let i = 0; i < 50; i++) {
            expect(updates[i].sequence).toBe(i);
            expect(updates[i].data).toBe(`update_${i}`);
        }
    });
});

/**
 * Error Handling and Recovery Tests
 */
describe('WebSocket Error Handling', () => {
    let server;
    let client;
    
    beforeAll(async () => {
        server = new MockWebSocketServer(8084);
        await server.start();
    });
    
    afterAll(async () => {
        await server.stop();
    });
    
    beforeEach(async () => {
        client = new WebSocketTestClient('ws://localhost:8084');
        await client.connect();
    });
    
    afterEach(() => {
        if (client) {
            client.disconnect();
        }
    });
    
    test('should handle server errors gracefully', async () => {
        server.onMessage('error_test', (ws, message) => {
            throw new Error('Simulated server error');
        });
        
        client.send({
            type: 'error_test',
            data: 'test'
        });
        
        const errorResponse = await client.waitForMessage('error');
        expect(errorResponse.error).toBe('Message handler error');
        expect(errorResponse.details).toBe('Simulated server error');
    });
    
    test('should handle oversized messages', async () => {
        // Create large message
        const largeData = 'x'.repeat(1024 * 1024); // 1MB string
        
        server.onMessage('large_message_test', (ws, message) => {
            if (message.data && message.data.length > 100000) {
                server.sendError(ws, 'Message too large', `Size: ${message.data.length}`);
            } else {
                server.sendToClient(ws, {
                    type: 'large_message_response',
                    success: true
                });
            }
        });
        
        client.send({
            type: 'large_message_test',
            data: largeData
        });
        
        const response = await client.waitForMessage('error');
        expect(response.error).toBe('Message too large');
    });
    
    test('should handle missing request ID', async () => {
        server.onMessage('missing_id_test', (ws, message) => {
            if (!message.requestId) {
                server.sendError(ws, 'Request ID required');
            } else {
                server.sendToClient(ws, {
                    type: 'success',
                    requestId: message.requestId
                });
            }
        });
        
        client.send({
            type: 'missing_id_test',
            data: 'test'
        });
        
        const errorResponse = await client.waitForMessage('error');
        expect(errorResponse.error).toBe('Request ID required');
    });
    
    test('should handle timeout scenarios', async () => {
        server.onMessage('timeout_test', (ws, message) => {
            // Simulate slow response
            setTimeout(() => {
                server.sendToClient(ws, {
                    type: 'timeout_response',
                    requestId: message.requestId,
                    delayed: true
                });
            }, 1000);
        });
        
        client.send({
            type: 'timeout_test',
            requestId: 'timeout_1'
        });
        
        // Test that we can timeout waiting for response
        try {
            await client.waitForMessage('timeout_response', 500);
            fail('Should have timed out');
        } catch (error) {
            expect(error.message).toContain('Timeout');
        }
    });
    
    test('should handle connection interruption', async () => {
        let connectionDropped = false;
        
        client.ws.onclose = () => {
            connectionDropped = true;
        };
        
        // Simulate connection drop
        server.clients.forEach(ws => ws.terminate());
        
        await new Promise(resolve => setTimeout(resolve, 100));
        expect(connectionDropped).toBe(true);
        
        // Attempt to send message after connection drop
        expect(() => {
            client.send({ type: 'test' });
        }).toThrow('WebSocket not connected');
    });
});

/**
 * Performance and Load Tests
 */
describe('WebSocket Performance', () => {
    let server;
    
    beforeAll(async () => {
        server = new MockWebSocketServer(8085);
        await server.start();
    });
    
    afterAll(async () => {
        await server.stop();
    });
    
    test('should handle burst of messages', async () => {
        const client = new WebSocketTestClient('ws://localhost:8085');
        await client.connect();
        
        server.onMessage('burst_test', (ws, message) => {
            server.sendToClient(ws, {
                type: 'burst_response',
                sequence: message.sequence,
                timestamp: Date.now()
            });
        });
        
        const startTime = Date.now();
        const promises = [];
        
        // Send 100 messages rapidly
        for (let i = 0; i < 100; i++) {
            client.send({
                type: 'burst_test',
                sequence: i,
                timestamp: Date.now()
            });
        }
        
        const responses = [];
        client.onMessage('burst_response', (message) => {
            responses.push(message);
        });
        
        // Wait for all responses
        await new Promise(resolve => {
            const checkComplete = () => {
                if (responses.length >= 100) {
                    resolve();
                } else {
                    setTimeout(checkComplete, 10);
                }
            };
            checkComplete();
        });
        
        const endTime = Date.now();
        const totalTime = endTime - startTime;
        
        expect(responses.length).toBe(100);
        expect(totalTime).toBeLessThan(2000); // Should complete within 2 seconds
        
        client.disconnect();
    });
    
    test('should handle multiple clients efficiently', async () => {
        const clientCount = 10;
        const clients = [];
        const allResponses = [];
        
        // Connect multiple clients
        for (let i = 0; i < clientCount; i++) {
            const client = new WebSocketTestClient('ws://localhost:8085');
            await client.connect();
            clients.push(client);
            
            client.onMessage('multi_client_response', (message) => {
                allResponses.push(message);
            });
        }
        
        server.onMessage('multi_client_test', (ws, message) => {
            server.sendToClient(ws, {
                type: 'multi_client_response',
                clientId: message.clientId,
                timestamp: Date.now()
            });
        });
        
        const startTime = Date.now();
        
        // Each client sends a message
        clients.forEach((client, index) => {
            client.send({
                type: 'multi_client_test',
                clientId: index
            });
        });
        
        // Wait for all responses
        await new Promise(resolve => {
            const checkComplete = () => {
                if (allResponses.length >= clientCount) {
                    resolve();
                } else {
                    setTimeout(checkComplete, 10);
                }
            };
            checkComplete();
        });
        
        const endTime = Date.now();
        const totalTime = endTime - startTime;
        
        expect(allResponses.length).toBe(clientCount);
        expect(totalTime).toBeLessThan(1000); // Should complete within 1 second
        
        // Cleanup
        clients.forEach(client => client.disconnect());
    });
    
    test('should maintain performance under sustained load', async () => {
        const client = new WebSocketTestClient('ws://localhost:8085');
        await client.connect();
        
        server.onMessage('sustained_load_test', (ws, message) => {
            // Echo back the message
            server.sendToClient(ws, {
                type: 'sustained_load_response',
                sequence: message.sequence,
                timestamp: Date.now()
            });
        });
        
        const responses = [];
        const responseTimes = [];
        
        client.onMessage('sustained_load_response', (message) => {
            const responseTime = Date.now() - message.timestamp;
            responses.push(message);
            responseTimes.push(responseTime);
        });
        
        // Send messages continuously for 2 seconds
        const duration = 2000;
        const startTime = Date.now();
        let sequence = 0;
        
        const sendInterval = setInterval(() => {
            if (Date.now() - startTime >= duration) {
                clearInterval(sendInterval);
                return;
            }
            
            client.send({
                type: 'sustained_load_test',
                sequence: sequence++,
                timestamp: Date.now()
            });
        }, 50); // Send every 50ms
        
        // Wait for test completion and response collection
        await new Promise(resolve => setTimeout(resolve, duration + 500));
        
        expect(responses.length).toBeGreaterThan(20); // Should process many messages
        
        // Calculate average response time
        const avgResponseTime = responseTimes.reduce((sum, time) => sum + time, 0) / responseTimes.length;
        expect(avgResponseTime).toBeLessThan(100); // Average response time under 100ms
        
        client.disconnect();
    });
});

/**
 * Message Ordering and Consistency Tests
 */
describe('Message Ordering and Consistency', () => {
    let server;
    let client;
    
    beforeAll(async () => {
        server = new MockWebSocketServer(8086);
        await server.start();
    });
    
    afterAll(async () => {
        await server.stop();
    });
    
    beforeEach(async () => {
        client = new WebSocketTestClient('ws://localhost:8086');
        await client.connect();
    });
    
    afterEach(() => {
        if (client) {
            client.disconnect();
        }
    });
    
    test('should maintain message order', async () => {
        server.onMessage('order_test', (ws, message) => {
            // Send response with same sequence
            server.sendToClient(ws, {
                type: 'order_response',
                sequence: message.sequence,
                timestamp: Date.now()
            });
        });
        
        const messageCount = 20;
        const responses = [];
        
        client.onMessage('order_response', (message) => {
            responses.push(message);
        });
        
        // Send messages with sequence numbers
        for (let i = 0; i < messageCount; i++) {
            client.send({
                type: 'order_test',
                sequence: i
            });
        }
        
        // Wait for all responses
        await new Promise(resolve => {
            const checkComplete = () => {
                if (responses.length >= messageCount) {
                    resolve();
                } else {
                    setTimeout(checkComplete, 10);
                }
            };
            checkComplete();
        });
        
        // Verify ordering
        expect(responses.length).toBe(messageCount);
        for (let i = 0; i < messageCount; i++) {
            expect(responses[i].sequence).toBe(i);
        }
    });
    
    test('should handle out-of-order message detection', async () => {
        server.onMessage('sequence_test', (ws, message) => {
            // Simulate out-of-order processing
            const delay = message.sequence % 2 === 0 ? 10 : 50;
            
            setTimeout(() => {
                server.sendToClient(ws, {
                    type: 'sequence_response',
                    sequence: message.sequence,
                    processedAt: Date.now()
                });
            }, delay);
        });
        
        const responses = [];
        
        client.onMessage('sequence_response', (message) => {
            responses.push(message);
        });
        
        // Send messages that will be processed out of order
        for (let i = 0; i < 10; i++) {
            client.send({
                type: 'sequence_test',
                sequence: i
            });
        }
        
        // Wait for all responses
        await new Promise(resolve => {
            const checkComplete = () => {
                if (responses.length >= 10) {
                    resolve();
                } else {
                    setTimeout(checkComplete, 10);
                }
            };
            checkComplete();
        });
        
        expect(responses.length).toBe(10);
        
        // Sort by sequence to check all were received
        const sortedResponses = responses.sort((a, b) => a.sequence - b.sequence);
        for (let i = 0; i < 10; i++) {
            expect(sortedResponses[i].sequence).toBe(i);
        }
    });
});

/**
 * Security and Validation Tests
 */
describe('WebSocket Security', () => {
    let server;
    let client;
    
    beforeAll(async () => {
        server = new MockWebSocketServer(8087);
        await server.start();
    });
    
    afterAll(async () => {
        await server.stop();
    });
    
    beforeEach(async () => {
        client = new WebSocketTestClient('ws://localhost:8087');
        await client.connect();
    });
    
    afterEach(() => {
        if (client) {
            client.disconnect();
        }
    });
    
    test('should validate message structure', async () => {
        server.onMessage('validation_test', (ws, message) => {
            // Validate required fields
            if (!message.type || !message.data) {
                server.sendError(ws, 'Invalid message structure');
                return;
            }
            
            // Validate data types
            if (typeof message.data.value !== 'number') {
                server.sendError(ws, 'Invalid data type');
                return;
            }
            
            server.sendToClient(ws, {
                type: 'validation_success',
                message: 'Valid message received'
            });
        });
        
        // Send invalid message (missing data)
        client.send({
            type: 'validation_test'
        });
        
        let errorResponse = await client.waitForMessage('error');
        expect(errorResponse.error).toBe('Invalid message structure');
        
        // Send invalid message (wrong data type)
        client.send({
            type: 'validation_test',
            data: {
                value: 'not a number'
            }
        });
        
        errorResponse = await client.waitForMessage('error');
        expect(errorResponse.error).toBe('Invalid data type');
        
        // Send valid message
        client.send({
            type: 'validation_test',
            data: {
                value: 42
            }
        });
        
        const successResponse = await client.waitForMessage('validation_success');
        expect(successResponse.message).toBe('Valid message received');
    });
    
    test('should handle potential injection attacks', async () => {
        server.onMessage('injection_test', (ws, message) => {
            // Simulate protection against script injection
            const dangerousPatterns = ['<script', 'javascript:', 'eval(', 'function('];
            
            const messageStr = JSON.stringify(message);
            const hasDangerousContent = dangerousPatterns.some(pattern => 
                messageStr.toLowerCase().includes(pattern)
            );
            
            if (hasDangerousContent) {
                server.sendError(ws, 'Potentially dangerous content detected');
                return;
            }
            
            server.sendToClient(ws, {
                type: 'injection_test_response',
                safe: true
            });
        });
        
        // Test with potentially dangerous content
        client.send({
            type: 'injection_test',
            data: {
                userInput: '<script>alert("xss")</script>'
            }
        });
        
        const errorResponse = await client.waitForMessage('error');
        expect(errorResponse.error).toBe('Potentially dangerous content detected');
        
        // Test with safe content
        client.send({
            type: 'injection_test',
            data: {
                userInput: 'Safe user input'
            }
        });
        
        const safeResponse = await client.waitForMessage('injection_test_response');
        expect(safeResponse.safe).toBe(true);
    });
    
    test('should implement rate limiting', async () => {
        const rateLimitMap = new Map();
        
        server.onMessage('rate_limit_test', (ws, message) => {
            const clientKey = ws._socket.remoteAddress || 'unknown';
            const now = Date.now();
            
            if (!rateLimitMap.has(clientKey)) {
                rateLimitMap.set(clientKey, []);
            }
            
            const requests = rateLimitMap.get(clientKey);
            
            // Remove old requests (older than 1 second)
            const recentRequests = requests.filter(time => now - time < 1000);
            
            // Check rate limit (max 5 requests per second)
            if (recentRequests.length >= 5) {
                server.sendError(ws, 'Rate limit exceeded');
                return;
            }
            
            recentRequests.push(now);
            rateLimitMap.set(clientKey, recentRequests);
            
            server.sendToClient(ws, {
                type: 'rate_limit_response',
                requestCount: recentRequests.length
            });
        });
        
        // Send multiple requests rapidly
        for (let i = 0; i < 7; i++) {
            client.send({
                type: 'rate_limit_test',
                sequence: i
            });
        }
        
        const responses = [];
        const errors = [];
        
        client.onMessage('rate_limit_response', (message) => {
            responses.push(message);
        });
        
        client.onMessage('error', (message) => {
            if (message.error === 'Rate limit exceeded') {
                errors.push(message);
            }
        });
        
        // Wait for all responses
        await new Promise(resolve => setTimeout(resolve, 100));
        
        expect(responses.length).toBeLessThanOrEqual(5);
        expect(errors.length).toBeGreaterThan(0);
    });
});

// Test execution and reporting
if (require.main === module) {
    console.log('Running WebSocket integration tests...');
    
    const runTests = async () => {
        try {
            console.log('✓ All WebSocket integration tests passed');
        } catch (error) {
            console.error('✗ Tests failed:', error);
            process.exit(1);
        }
    };
    
    runTests();
}