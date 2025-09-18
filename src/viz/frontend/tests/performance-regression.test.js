/**
 * Performance Regression Tests for Groggy Visualization System
 * 
 * These tests ensure that performance optimizations are maintained and
 * catch regressions in critical rendering and interaction paths.
 */

describe('Performance Regression Tests', () => {
    let renderer;
    let performanceBaselines;
    let testGraphs;
    
    beforeAll(async () => {
        // Initialize renderer with performance monitoring
        renderer = new GraphRenderer({
            canvas: document.createElement('canvas'),
            enablePerformanceMonitoring: true,
            performanceTracing: true
        });
        
        // Performance baselines (in milliseconds)
        performanceBaselines = {
            smallGraphRender: 16,      // 60fps target
            mediumGraphRender: 33,     // 30fps target  
            largeGraphRender: 100,     // 10fps acceptable
            nodeSelection: 5,          // Interactive response
            nodeInteraction: 2,        // Mouse hover response
            layoutUpdate: 50,          // Layout algorithm step
            canvasResize: 10,          // Window resize response
            exportGeneration: 500,     // Static export time
            streamingUpdate: 16       // Real-time streaming frame
        };
        
        // Create test graphs of various sizes
        testGraphs = {
            small: TestGraphGenerator.createGraph({
                nodeCount: 50,
                edgeCount: 100,
                distribution: 'random'
            }),
            medium: TestGraphGenerator.createGraph({
                nodeCount: 500,
                edgeCount: 1000,
                distribution: 'scale-free'
            }),
            large: TestGraphGenerator.createGraph({
                nodeCount: 2000,
                edgeCount: 5000,
                distribution: 'hierarchical'
            }),
            dense: TestGraphGenerator.createGraph({
                nodeCount: 1000,
                edgeCount: 10000,
                distribution: 'dense'
            })
        };
    });
    
    afterEach(() => {
        // Clean up any performance monitoring
        renderer.clearPerformanceData();
    });
    
    // =========================================================================
    // RENDERING PERFORMANCE TESTS
    // =========================================================================
    
    describe('Rendering Performance', () => {
        test('small graph renders within 60fps target', async () => {
            const startTime = performance.now();
            
            await renderer.loadGraph(testGraphs.small);
            await renderer.render();
            
            const renderTime = performance.now() - startTime;
            
            expect(renderTime).toBeLessThan(performanceBaselines.smallGraphRender);
            
            // Verify frame rate consistency
            const frameMetrics = await measureFrameRate(renderer, 1000); // 1 second
            expect(frameMetrics.averageFPS).toBeGreaterThan(55); // Allow 10% tolerance
        });
        
        test('medium graph renders within 30fps target', async () => {
            const startTime = performance.now();
            
            await renderer.loadGraph(testGraphs.medium);
            await renderer.render();
            
            const renderTime = performance.now() - startTime;
            
            expect(renderTime).toBeLessThan(performanceBaselines.mediumGraphRender);
            
            // Check memory usage doesn't grow excessively
            const memoryBefore = performance.memory?.usedJSHeapSize || 0;
            await renderer.render(); // Second render
            const memoryAfter = performance.memory?.usedJSHeapSize || 0;
            
            const memoryGrowth = memoryAfter - memoryBefore;
            expect(memoryGrowth).toBeLessThan(1024 * 1024); // Less than 1MB growth
        });
        
        test('large graph renders within acceptable limits', async () => {
            const startTime = performance.now();
            
            await renderer.loadGraph(testGraphs.large);
            await renderer.render();
            
            const renderTime = performance.now() - startTime;
            
            expect(renderTime).toBeLessThan(performanceBaselines.largeGraphRender);
            
            // Verify level-of-detail kicks in for large graphs
            const renderingStats = renderer.getLastRenderStats();
            expect(renderingStats.lodEnabled).toBe(true);
            expect(renderingStats.culledNodes).toBeGreaterThan(0);
        });
        
        test('dense graph uses appropriate optimizations', async () => {
            const startTime = performance.now();
            
            await renderer.loadGraph(testGraphs.dense);
            await renderer.render();
            
            const renderTime = performance.now() - startTime;
            
            // Dense graphs should use edge bundling and clustering
            const renderingStats = renderer.getLastRenderStats();
            expect(renderingStats.edgeBundlingEnabled).toBe(true);
            expect(renderingStats.nodeClusteringEnabled).toBe(true);
            
            // Render time should still be reasonable
            expect(renderTime).toBeLessThan(performanceBaselines.largeGraphRender * 1.5);
        });
    });
    
    // =========================================================================
    // INTERACTION PERFORMANCE TESTS
    // =========================================================================
    
    describe('Interaction Performance', () => {
        test('node selection responds quickly', async () => {
            await renderer.loadGraph(testGraphs.medium);
            await renderer.render();
            
            const startTime = performance.now();
            
            // Simulate clicking on a node
            const testNode = testGraphs.medium.nodes[0];
            const clickEvent = new MouseEvent('click', {
                clientX: testNode.x,
                clientY: testNode.y
            });
            
            renderer.handleNodeClick(clickEvent);
            
            const responseTime = performance.now() - startTime;
            expect(responseTime).toBeLessThan(performanceBaselines.nodeSelection);
        });
        
        test('node hover interaction is responsive', async () => {
            await renderer.loadGraph(testGraphs.medium);
            await renderer.render();
            
            const hoverTimes = [];
            
            // Test multiple hover interactions
            for (let i = 0; i < 10; i++) {
                const startTime = performance.now();
                
                const testNode = testGraphs.medium.nodes[i];
                const hoverEvent = new MouseEvent('mouseover', {
                    clientX: testNode.x,
                    clientY: testNode.y
                });
                
                renderer.handleNodeHover(hoverEvent);
                
                hoverTimes.push(performance.now() - startTime);
            }
            
            const averageHoverTime = hoverTimes.reduce((a, b) => a + b) / hoverTimes.length;
            expect(averageHoverTime).toBeLessThan(performanceBaselines.nodeInteraction);
        });
        
        test('dragging performance remains smooth', async () => {
            await renderer.loadGraph(testGraphs.medium);
            await renderer.render();
            
            const draggingSystem = renderer.getNodeDraggingSystem();
            const testNode = testGraphs.medium.nodes[0];
            
            // Start drag
            const startTime = performance.now();
            draggingSystem.startDrag(testNode.id, { x: testNode.x, y: testNode.y });
            
            // Simulate 20 drag move events
            const dragTimes = [];
            for (let i = 0; i < 20; i++) {
                const moveStart = performance.now();
                
                draggingSystem.updateDrag({
                    x: testNode.x + i * 5,
                    y: testNode.y + i * 5
                });
                
                dragTimes.push(performance.now() - moveStart);
            }
            
            draggingSystem.endDrag();
            
            const averageDragTime = dragTimes.reduce((a, b) => a + b) / dragTimes.length;
            expect(averageDragTime).toBeLessThan(performanceBaselines.nodeInteraction);
        });
        
        test('zoom and pan operations are smooth', async () => {
            await renderer.loadGraph(testGraphs.medium);
            await renderer.render();
            
            const zoomTimes = [];
            
            // Test zoom operations
            for (let zoom = 0.5; zoom <= 2.0; zoom += 0.1) {
                const startTime = performance.now();
                
                renderer.setZoom(zoom);
                await renderer.render();
                
                zoomTimes.push(performance.now() - startTime);
            }
            
            const averageZoomTime = zoomTimes.reduce((a, b) => a + b) / zoomTimes.length;
            expect(averageZoomTime).toBeLessThan(performanceBaselines.nodeInteraction * 2);
        });
    });
    
    // =========================================================================
    // LAYOUT ALGORITHM PERFORMANCE TESTS
    // =========================================================================
    
    describe('Layout Algorithm Performance', () => {
        test('force-directed layout converges efficiently', async () => {
            const layout = new ForceDirectedLayout({
                iterations: 100,
                convergenceThreshold: 0.01
            });
            
            const startTime = performance.now();
            
            const result = layout.applyLayout(testGraphs.medium);
            
            const layoutTime = performance.now() - startTime;
            
            expect(layoutTime).toBeLessThan(performanceBaselines.layoutUpdate * 10); // 10 iterations worth
            expect(result.converged).toBe(true);
            expect(result.finalIteration).toBeLessThan(100); // Should converge early
        });
        
        test('hierarchical layout handles large graphs', async () => {
            const layout = new HierarchicalLayout({
                direction: 'top-down',
                layerSpacing: 100,
                nodeSpacing: 50
            });
            
            const startTime = performance.now();
            
            layout.applyLayout(testGraphs.large);
            
            const layoutTime = performance.now() - startTime;
            
            expect(layoutTime).toBeLessThan(performanceBaselines.layoutUpdate * 5);
        });
        
        test('circular layout scales linearly', async () => {
            const layout = new CircularLayout();
            
            // Test with different graph sizes
            const graphSizes = [100, 200, 500, 1000];
            const layoutTimes = [];
            
            for (const size of graphSizes) {
                const testGraph = TestGraphGenerator.createGraph({
                    nodeCount: size,
                    edgeCount: size * 2
                });
                
                const startTime = performance.now();
                layout.applyLayout(testGraph);
                const layoutTime = performance.now() - startTime;
                
                layoutTimes.push({ size, time: layoutTime });
            }
            
            // Verify linear scaling (not exponential)
            for (let i = 1; i < layoutTimes.length; i++) {
                const prev = layoutTimes[i - 1];
                const curr = layoutTimes[i];
                
                const sizeRatio = curr.size / prev.size;
                const timeRatio = curr.time / prev.time;
                
                // Time ratio should be close to size ratio (linear scaling)
                expect(timeRatio).toBeLessThan(sizeRatio * 1.5); // Allow 50% overhead
            }
        });
        
        test('grid layout is consistently fast', async () => {
            const layout = new GridLayout({
                columns: 'auto',
                cellSize: 50
            });
            
            const layoutTimes = [];
            
            // Run layout multiple times to check consistency
            for (let i = 0; i < 10; i++) {
                const startTime = performance.now();
                layout.applyLayout(testGraphs.medium);
                layoutTimes.push(performance.now() - startTime);
            }
            
            const averageTime = layoutTimes.reduce((a, b) => a + b) / layoutTimes.length;
            const variance = layoutTimes.reduce((sum, time) => sum + Math.pow(time - averageTime, 2), 0) / layoutTimes.length;
            
            expect(averageTime).toBeLessThan(performanceBaselines.layoutUpdate);
            expect(variance).toBeLessThan(10); // Low variance means consistent performance
        });
    });
    
    // =========================================================================
    // EXPORT PERFORMANCE TESTS
    // =========================================================================
    
    describe('Export Performance', () => {
        test('SVG export completes within time limit', async () => {
            await renderer.loadGraph(testGraphs.medium);
            await renderer.render();
            
            const exportSystem = renderer.getExportSystem();
            const startTime = performance.now();
            
            const svgResult = await exportSystem.exportSVG({
                embedFonts: true,
                embedStyles: true,
                optimizeSize: true
            });
            
            const exportTime = performance.now() - startTime;
            
            expect(exportTime).toBeLessThan(performanceBaselines.exportGeneration);
            expect(svgResult.size).toBeGreaterThan(0);
        });
        
        test('PNG export with high DPI performs well', async () => {
            await renderer.loadGraph(testGraphs.medium);
            await renderer.render();
            
            const exportSystem = renderer.getExportSystem();
            const startTime = performance.now();
            
            const pngResult = await exportSystem.exportPNG({
                dpi: 300,
                width: 1920,
                height: 1080
            });
            
            const exportTime = performance.now() - startTime;
            
            expect(exportTime).toBeLessThan(performanceBaselines.exportGeneration * 2); // High DPI takes longer
            expect(pngResult.size).toBeGreaterThan(0);
        });
        
        test('batch export handles multiple formats efficiently', async () => {
            await renderer.loadGraph(testGraphs.small); // Use smaller graph for batch test
            await renderer.render();
            
            const exportSystem = renderer.getExportSystem();
            const startTime = performance.now();
            
            const batchResult = await exportSystem.batchExport([
                { format: 'svg', filename: 'test.svg' },
                { format: 'png', filename: 'test.png', dpi: 150 },
                { format: 'pdf', filename: 'test.pdf' }
            ]);
            
            const totalExportTime = performance.now() - startTime;
            
            // Batch export should be more efficient than individual exports
            expect(totalExportTime).toBeLessThan(performanceBaselines.exportGeneration * 2);
            expect(batchResult.results).toHaveLength(3);
            expect(batchResult.results.every(r => r.success)).toBe(true);
        });
    });
    
    // =========================================================================
    // STREAMING PERFORMANCE TESTS
    // =========================================================================
    
    describe('Streaming Performance', () => {
        test('real-time updates maintain frame rate', async () => {
            await renderer.loadGraph(testGraphs.small);
            await renderer.render();
            
            const streamingSystem = renderer.getStreamingSystem();
            const updateTimes = [];
            
            // Simulate 60 updates (1 second at 60fps)
            for (let i = 0; i < 60; i++) {
                const startTime = performance.now();
                
                // Add a new node and edge
                const newNode = {
                    id: `stream_node_${i}`,
                    x: Math.random() * 800,
                    y: Math.random() * 600,
                    attributes: { value: i }
                };
                
                streamingSystem.addNode(newNode);
                
                if (testGraphs.small.nodes.length > 0) {
                    const randomExisting = testGraphs.small.nodes[Math.floor(Math.random() * testGraphs.small.nodes.length)];
                    streamingSystem.addEdge({
                        id: `stream_edge_${i}`,
                        source: randomExisting.id,
                        target: newNode.id
                    });
                }
                
                await renderer.render();
                
                updateTimes.push(performance.now() - startTime);
            }
            
            const averageUpdateTime = updateTimes.reduce((a, b) => a + b) / updateTimes.length;
            expect(averageUpdateTime).toBeLessThan(performanceBaselines.streamingUpdate);
            
            // Check that frame rate remained stable
            const maxUpdateTime = Math.max(...updateTimes);
            expect(maxUpdateTime).toBeLessThan(performanceBaselines.streamingUpdate * 2);
        });
        
        test('bulk streaming updates are optimized', async () => {
            await renderer.loadGraph(testGraphs.small);
            await renderer.render();
            
            const streamingSystem = renderer.getStreamingSystem();
            const startTime = performance.now();
            
            // Send 100 updates as a bulk operation
            const bulkUpdates = [];
            for (let i = 0; i < 100; i++) {
                bulkUpdates.push({
                    type: 'add_node',
                    data: {
                        id: `bulk_node_${i}`,
                        x: Math.random() * 800,
                        y: Math.random() * 600
                    }
                });
            }
            
            await streamingSystem.applyBulkUpdates(bulkUpdates);
            await renderer.render();
            
            const bulkUpdateTime = performance.now() - startTime;
            
            // Bulk updates should be much faster than individual updates
            expect(bulkUpdateTime).toBeLessThan(performanceBaselines.streamingUpdate * 10);
        });
        
        test('streaming memory usage remains stable', async () => {
            await renderer.loadGraph(testGraphs.small);
            await renderer.render();
            
            const streamingSystem = renderer.getStreamingSystem();
            const initialMemory = performance.memory?.usedJSHeapSize || 0;
            
            // Add many streaming updates
            for (let i = 0; i < 1000; i++) {
                streamingSystem.addNode({
                    id: `memory_test_node_${i}`,
                    x: Math.random() * 800,
                    y: Math.random() * 600
                });
                
                // Trigger garbage collection every 100 updates
                if (i % 100 === 0) {
                    await new Promise(resolve => setTimeout(resolve, 1));
                }
            }
            
            await renderer.render();
            
            const finalMemory = performance.memory?.usedJSHeapSize || 0;
            const memoryGrowth = finalMemory - initialMemory;
            
            // Memory growth should be reasonable (less than 10MB for 1000 nodes)
            expect(memoryGrowth).toBeLessThan(10 * 1024 * 1024);
        });
    });
    
    // =========================================================================
    // UTILITY FUNCTIONS
    // =========================================================================
    
    async function measureFrameRate(renderer, durationMs) {
        const startTime = performance.now();
        let frameCount = 0;
        const frameTimes = [];
        
        while (performance.now() - startTime < durationMs) {
            const frameStart = performance.now();
            await renderer.render();
            const frameTime = performance.now() - frameStart;
            
            frameTimes.push(frameTime);
            frameCount++;
        }
        
        const totalTime = performance.now() - startTime;
        const averageFPS = (frameCount / totalTime) * 1000;
        const averageFrameTime = frameTimes.reduce((a, b) => a + b) / frameTimes.length;
        
        return {
            averageFPS,
            averageFrameTime,
            frameCount,
            frameTimes
        };
    }
    
    // =========================================================================
    // REGRESSION DETECTION
    // =========================================================================
    
    describe('Performance Regression Detection', () => {
        test('performance metrics stay within expected ranges', () => {
            // This test can be configured to fail CI builds if performance regresses
            const performanceReport = renderer.getPerformanceReport();
            
            // Check key metrics against baselines
            Object.keys(performanceBaselines).forEach(metric => {
                if (performanceReport[metric]) {
                    const baseline = performanceBaselines[metric];
                    const actual = performanceReport[metric];
                    const tolerance = baseline * 0.2; // 20% tolerance
                    
                    expect(actual).toBeLessThan(baseline + tolerance);
                }
            });
        });
        
        test('memory usage stays within bounds', () => {
            const memoryReport = renderer.getMemoryReport();
            
            // Check for memory leaks
            expect(memoryReport.leakDetected).toBe(false);
            
            // Check peak memory usage
            expect(memoryReport.peakMemoryMB).toBeLessThan(500); // 500MB limit
            
            // Check memory growth rate
            expect(memoryReport.growthRateMBPerSecond).toBeLessThan(10);
        });
    });
});

/**
 * Performance Test Configuration
 */
class PerformanceTestConfig {
    static getBaselines() {
        return {
            // Rendering performance (ms)
            smallGraphRender: 16,
            mediumGraphRender: 33,
            largeGraphRender: 100,
            
            // Interaction performance (ms)
            nodeSelection: 5,
            nodeInteraction: 2,
            
            // Layout performance (ms)
            layoutUpdate: 50,
            
            // Export performance (ms)
            exportGeneration: 500,
            
            // Streaming performance (ms)
            streamingUpdate: 16,
            
            // Memory limits (MB)
            maxMemoryUsage: 500,
            maxMemoryGrowthRate: 10
        };
    }
    
    static getTestGraphs() {
        return {
            small: { nodes: 50, edges: 100 },
            medium: { nodes: 500, edges: 1000 },
            large: { nodes: 2000, edges: 5000 },
            dense: { nodes: 1000, edges: 10000 }
        };
    }
}

/**
 * Performance Monitoring Utilities
 */
class PerformanceMonitor {
    constructor() {
        this.metrics = new Map();
        this.startTimes = new Map();
    }
    
    startTimer(name) {
        this.startTimes.set(name, performance.now());
    }
    
    endTimer(name) {
        const startTime = this.startTimes.get(name);
        if (startTime !== undefined) {
            const duration = performance.now() - startTime;
            
            if (!this.metrics.has(name)) {
                this.metrics.set(name, []);
            }
            
            this.metrics.get(name).push(duration);
            this.startTimes.delete(name);
            
            return duration;
        }
        return 0;
    }
    
    getAverageTime(name) {
        const times = this.metrics.get(name) || [];
        return times.length > 0 ? times.reduce((a, b) => a + b) / times.length : 0;
    }
    
    getReport() {
        const report = {};
        
        for (const [name, times] of this.metrics) {
            report[name] = {
                average: times.reduce((a, b) => a + b) / times.length,
                min: Math.min(...times),
                max: Math.max(...times),
                count: times.length
            };
        }
        
        return report;
    }
}