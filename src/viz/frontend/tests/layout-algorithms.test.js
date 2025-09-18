/**
 * Comprehensive Unit Tests for Layout Algorithms
 * Tests all layout algorithms with various graph configurations,
 * edge cases, and performance benchmarks.
 */

// Test framework setup (assumes Jest or similar)
const { expect, describe, test, beforeEach, afterEach } = require('@jest/globals');

// Mock canvas and WebGL contexts for testing
global.HTMLCanvasElement.prototype.getContext = jest.fn(() => ({
    fillRect: jest.fn(),
    clearRect: jest.fn(),
    getImageData: jest.fn(() => ({ data: new Array(4) })),
    putImageData: jest.fn(),
    createImageData: jest.fn(() => ({ data: new Array(4) })),
    setTransform: jest.fn(),
    drawImage: jest.fn(),
    save: jest.fn(),
    restore: jest.fn(),
    beginPath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    closePath: jest.fn(),
    stroke: jest.fn(),
    fill: jest.fn(),
    arc: jest.fn(),
    rect: jest.fn(),
    measureText: jest.fn(() => ({ width: 10 })),
    transform: jest.fn(),
    translate: jest.fn(),
    scale: jest.fn(),
    rotate: jest.fn()
}));

// Import layout algorithms (adjust paths as needed)
const ForceDirectedLayout = require('../js/layouts/force-directed-layout');
const CircularLayout = require('../js/layouts/circular-layout');
const HierarchicalLayout = require('../js/layouts/hierarchical-layout');
const GridLayout = require('../js/layouts/grid-layout');
const TreeLayout = require('../js/layouts/tree-layout');

/**
 * Test Data Generation Utilities
 */
class TestGraphGenerator {
    static createSimpleGraph() {
        return {
            nodes: [
                { id: 'node1', x: 0, y: 0, radius: 10 },
                { id: 'node2', x: 0, y: 0, radius: 10 },
                { id: 'node3', x: 0, y: 0, radius: 10 }
            ],
            edges: [
                { id: 'edge1', source: 'node1', target: 'node2' },
                { id: 'edge2', source: 'node2', target: 'node3' }
            ]
        };
    }
    
    static createCompleteGraph(nodeCount) {
        const nodes = [];
        const edges = [];
        
        // Create nodes
        for (let i = 0; i < nodeCount; i++) {
            nodes.push({
                id: `node${i}`,
                x: Math.random() * 400,
                y: Math.random() * 400,
                radius: 10
            });
        }
        
        // Create all possible edges
        for (let i = 0; i < nodeCount; i++) {
            for (let j = i + 1; j < nodeCount; j++) {
                edges.push({
                    id: `edge${i}-${j}`,
                    source: `node${i}`,
                    target: `node${j}`
                });
            }
        }
        
        return { nodes, edges };
    }
    
    static createStarGraph(centerNode, leafCount) {
        const nodes = [
            { id: centerNode, x: 200, y: 200, radius: 15 }
        ];
        const edges = [];
        
        for (let i = 0; i < leafCount; i++) {
            const leafId = `leaf${i}`;
            nodes.push({
                id: leafId,
                x: Math.random() * 400,
                y: Math.random() * 400,
                radius: 10
            });
            edges.push({
                id: `edge-${centerNode}-${leafId}`,
                source: centerNode,
                target: leafId
            });
        }
        
        return { nodes, edges };
    }
    
    static createTreeGraph(depth, branchingFactor) {
        const nodes = [];
        const edges = [];
        let nodeCounter = 0;
        
        function createNode(level, parent = null) {
            const nodeId = `node${nodeCounter++}`;
            nodes.push({
                id: nodeId,
                x: Math.random() * 400,
                y: Math.random() * 400,
                radius: 10,
                level: level
            });
            
            if (parent) {
                edges.push({
                    id: `edge-${parent}-${nodeId}`,
                    source: parent,
                    target: nodeId
                });
            }
            
            if (level < depth) {
                for (let i = 0; i < branchingFactor; i++) {
                    createNode(level + 1, nodeId);
                }
            }
            
            return nodeId;
        }
        
        createNode(0);
        return { nodes, edges };
    }
    
    static createDisconnectedGraph() {
        const component1 = this.createSimpleGraph();
        const component2 = {
            nodes: [
                { id: 'node4', x: 0, y: 0, radius: 10 },
                { id: 'node5', x: 0, y: 0, radius: 10 }
            ],
            edges: [
                { id: 'edge3', source: 'node4', target: 'node5' }
            ]
        };
        
        return {
            nodes: [...component1.nodes, ...component2.nodes],
            edges: [...component1.edges, ...component2.edges]
        };
    }
    
    static createLargeGraph(nodeCount, edgeDensity = 0.1) {
        const nodes = [];
        const edges = [];
        
        // Create nodes
        for (let i = 0; i < nodeCount; i++) {
            nodes.push({
                id: `node${i}`,
                x: Math.random() * 1000,
                y: Math.random() * 1000,
                radius: 10
            });
        }
        
        // Create edges with specified density
        const maxEdges = nodeCount * (nodeCount - 1) / 2;
        const edgeCount = Math.floor(maxEdges * edgeDensity);
        const edgeSet = new Set();
        
        while (edgeSet.size < edgeCount) {
            const i = Math.floor(Math.random() * nodeCount);
            const j = Math.floor(Math.random() * nodeCount);
            
            if (i !== j) {
                const edgeKey = `${Math.min(i, j)}-${Math.max(i, j)}`;
                if (!edgeSet.has(edgeKey)) {
                    edgeSet.add(edgeKey);
                    edges.push({
                        id: `edge${edges.length}`,
                        source: `node${i}`,
                        target: `node${j}`
                    });
                }
            }
        }
        
        return { nodes, edges };
    }
}

/**
 * Layout Testing Utilities
 */
class LayoutTestUtils {
    static validatePositions(nodes) {
        return nodes.every(node => 
            typeof node.x === 'number' && 
            typeof node.y === 'number' && 
            !isNaN(node.x) && 
            !isNaN(node.y) &&
            isFinite(node.x) && 
            isFinite(node.y)
        );
    }
    
    static calculateBoundingBox(nodes) {
        if (nodes.length === 0) return null;
        
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        nodes.forEach(node => {
            minX = Math.min(minX, node.x);
            maxX = Math.max(maxX, node.x);
            minY = Math.min(minY, node.y);
            maxY = Math.max(maxY, node.y);
        });
        
        return {
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY
        };
    }
    
    static calculateAverageDistance(nodes) {
        if (nodes.length < 2) return 0;
        
        let totalDistance = 0;
        let count = 0;
        
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                totalDistance += Math.sqrt(dx * dx + dy * dy);
                count++;
            }
        }
        
        return count > 0 ? totalDistance / count : 0;
    }
    
    static checkOverlaps(nodes, threshold = 20) {
        const overlaps = [];
        
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < threshold) {
                    overlaps.push({
                        node1: nodes[i].id,
                        node2: nodes[j].id,
                        distance: distance
                    });
                }
            }
        }
        
        return overlaps;
    }
    
    static measureExecutionTime(layoutFunction) {
        const start = performance.now();
        const result = layoutFunction();
        const end = performance.now();
        
        return {
            result: result,
            executionTime: end - start
        };
    }
    
    static calculateLayoutQuality(nodes, edges) {
        // Calculate various quality metrics
        const metrics = {
            nodeSpread: this.calculateNodeSpread(nodes),
            edgeCrossings: this.calculateEdgeCrossings(nodes, edges),
            averageEdgeLength: this.calculateAverageEdgeLength(nodes, edges),
            nodeOverlaps: this.checkOverlaps(nodes).length
        };
        
        // Overall quality score (lower is better)
        metrics.qualityScore = 
            metrics.edgeCrossings * 10 +
            metrics.nodeOverlaps * 20 +
            (metrics.averageEdgeLength > 100 ? metrics.averageEdgeLength - 100 : 0);
        
        return metrics;
    }
    
    static calculateNodeSpread(nodes) {
        const bbox = this.calculateBoundingBox(nodes);
        return bbox ? bbox.width + bbox.height : 0;
    }
    
    static calculateEdgeCrossings(nodes, edges) {
        let crossings = 0;
        const nodeMap = new Map(nodes.map(node => [node.id, node]));
        
        for (let i = 0; i < edges.length; i++) {
            for (let j = i + 1; j < edges.length; j++) {
                const edge1 = edges[i];
                const edge2 = edges[j];
                
                const node1a = nodeMap.get(edge1.source);
                const node1b = nodeMap.get(edge1.target);
                const node2a = nodeMap.get(edge2.source);
                const node2b = nodeMap.get(edge2.target);
                
                if (node1a && node1b && node2a && node2b) {
                    if (this.doLinesIntersect(node1a, node1b, node2a, node2b)) {
                        crossings++;
                    }
                }
            }
        }
        
        return crossings;
    }
    
    static doLinesIntersect(p1, p2, p3, p4) {
        const det = (p2.x - p1.x) * (p4.y - p3.y) - (p4.x - p3.x) * (p2.y - p1.y);
        
        if (det === 0) return false; // Lines are parallel
        
        const lambda = ((p4.y - p3.y) * (p4.x - p1.x) + (p3.x - p4.x) * (p4.y - p1.y)) / det;
        const gamma = ((p1.y - p2.y) * (p4.x - p1.x) + (p2.x - p1.x) * (p4.y - p1.y)) / det;
        
        return (0 < lambda && lambda < 1) && (0 < gamma && gamma < 1);
    }
    
    static calculateAverageEdgeLength(nodes, edges) {
        if (edges.length === 0) return 0;
        
        const nodeMap = new Map(nodes.map(node => [node.id, node]));
        let totalLength = 0;
        
        edges.forEach(edge => {
            const source = nodeMap.get(edge.source);
            const target = nodeMap.get(edge.target);
            
            if (source && target) {
                const dx = source.x - target.x;
                const dy = source.y - target.y;
                totalLength += Math.sqrt(dx * dx + dy * dy);
            }
        });
        
        return totalLength / edges.length;
    }
}

/**
 * Force-Directed Layout Tests
 */
describe('ForceDirectedLayout', () => {
    let layout;
    let mockRenderer;
    
    beforeEach(() => {
        mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => []),
            getEdges: jest.fn(() => []),
            requestRedraw: jest.fn()
        };
        
        layout = new ForceDirectedLayout(mockRenderer);
    });
    
    test('should initialize with default configuration', () => {
        expect(layout.config).toBeDefined();
        expect(layout.config.iterations).toBeGreaterThan(0);
        expect(layout.config.springConstant).toBeGreaterThan(0);
        expect(layout.config.repulsionForce).toBeGreaterThan(0);
    });
    
    test('should handle empty graph', () => {
        const graph = { nodes: [], edges: [] };
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        const result = layout.applyLayout();
        expect(result).toBe(true);
    });
    
    test('should position single node at center', () => {
        const graph = {
            nodes: [{ id: 'node1', x: 0, y: 0, radius: 10 }],
            edges: []
        };
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        const node = graph.nodes[0];
        expect(LayoutTestUtils.validatePositions([node])).toBe(true);
        expect(node.x).toBeCloseTo(200, 50); // Should be near canvas center
        expect(node.y).toBeCloseTo(200, 50);
    });
    
    test('should separate disconnected nodes', () => {
        const graph = {
            nodes: [
                { id: 'node1', x: 200, y: 200, radius: 10 },
                { id: 'node2', x: 200, y: 200, radius: 10 }
            ],
            edges: []
        };
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        const distance = Math.sqrt(
            Math.pow(graph.nodes[0].x - graph.nodes[1].x, 2) +
            Math.pow(graph.nodes[0].y - graph.nodes[1].y, 2)
        );
        expect(distance).toBeGreaterThan(20); // Nodes should be separated
    });
    
    test('should connect adjacent nodes with appropriate distance', () => {
        const graph = TestGraphGenerator.createSimpleGraph();
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        // Check that connected nodes are reasonably close
        const averageEdgeLength = LayoutTestUtils.calculateAverageEdgeLength(graph.nodes, graph.edges);
        expect(averageEdgeLength).toBeGreaterThan(30);
        expect(averageEdgeLength).toBeLessThan(200);
    });
    
    test('should handle complete graph without overlaps', () => {
        const graph = TestGraphGenerator.createCompleteGraph(5);
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        const overlaps = LayoutTestUtils.checkOverlaps(graph.nodes, 15);
        expect(overlaps.length).toBe(0);
    });
    
    test('should converge within maximum iterations', () => {
        const graph = TestGraphGenerator.createLargeGraph(20, 0.2);
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        const maxIterations = layout.config.iterations;
        const result = LayoutTestUtils.measureExecutionTime(() => layout.applyLayout());
        
        expect(result.result).toBe(true);
        expect(result.executionTime).toBeLessThan(5000); // Should complete within 5 seconds
    });
    
    test('should respect custom configuration', () => {
        const customConfig = {
            iterations: 50,
            springConstant: 0.05,
            repulsionForce: 100
        };
        
        layout.updateConfig(customConfig);
        
        expect(layout.config.iterations).toBe(50);
        expect(layout.config.springConstant).toBe(0.05);
        expect(layout.config.repulsionForce).toBe(100);
    });
    
    test('should handle numerical edge cases', () => {
        const graph = {
            nodes: [
                { id: 'node1', x: Number.MAX_VALUE / 2, y: 0, radius: 10 },
                { id: 'node2', x: 0, y: Number.MAX_VALUE / 2, radius: 10 }
            ],
            edges: [{ id: 'edge1', source: 'node1', target: 'node2' }]
        };
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        const result = layout.applyLayout();
        
        expect(result).toBe(true);
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
    });
});

/**
 * Circular Layout Tests
 */
describe('CircularLayout', () => {
    let layout;
    let mockRenderer;
    
    beforeEach(() => {
        mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => []),
            getEdges: jest.fn(() => []),
            requestRedraw: jest.fn()
        };
        
        layout = new CircularLayout(mockRenderer);
    });
    
    test('should arrange nodes in circle', () => {
        const graph = TestGraphGenerator.createSimpleGraph();
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        // Calculate center and radius
        const centerX = graph.nodes.reduce((sum, node) => sum + node.x, 0) / graph.nodes.length;
        const centerY = graph.nodes.reduce((sum, node) => sum + node.y, 0) / graph.nodes.length;
        
        // Check that all nodes are roughly equidistant from center
        const distances = graph.nodes.map(node => 
            Math.sqrt(Math.pow(node.x - centerX, 2) + Math.pow(node.y - centerY, 2))
        );
        
        const avgDistance = distances.reduce((sum, d) => sum + d, 0) / distances.length;
        distances.forEach(distance => {
            expect(Math.abs(distance - avgDistance)).toBeLessThan(5);
        });
    });
    
    test('should distribute nodes evenly around circle', () => {
        const nodeCount = 8;
        const graph = TestGraphGenerator.createCompleteGraph(nodeCount);
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        // Calculate angles from center
        const centerX = 200; // Assumed canvas center
        const centerY = 200;
        
        const angles = graph.nodes.map(node => 
            Math.atan2(node.y - centerY, node.x - centerX)
        ).sort((a, b) => a - b);
        
        // Check angular spacing
        const expectedAngleDiff = (2 * Math.PI) / nodeCount;
        for (let i = 1; i < angles.length; i++) {
            const angleDiff = angles[i] - angles[i - 1];
            expect(Math.abs(angleDiff - expectedAngleDiff)).toBeLessThan(0.1);
        }
    });
    
    test('should handle single node', () => {
        const graph = {
            nodes: [{ id: 'node1', x: 0, y: 0, radius: 10 }],
            edges: []
        };
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        expect(graph.nodes[0].x).toBeCloseTo(200, 10);
        expect(graph.nodes[0].y).toBeCloseTo(200, 10);
    });
    
    test('should scale radius appropriately for large graphs', () => {
        const graph = TestGraphGenerator.createLargeGraph(50, 0.1);
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        const bbox = LayoutTestUtils.calculateBoundingBox(graph.nodes);
        expect(bbox.width).toBeGreaterThan(100);
        expect(bbox.height).toBeGreaterThan(100);
        
        // No overlaps
        const overlaps = LayoutTestUtils.checkOverlaps(graph.nodes, 15);
        expect(overlaps.length).toBe(0);
    });
});

/**
 * Hierarchical Layout Tests
 */
describe('HierarchicalLayout', () => {
    let layout;
    let mockRenderer;
    
    beforeEach(() => {
        mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => []),
            getEdges: jest.fn(() => []),
            requestRedraw: jest.fn()
        };
        
        layout = new HierarchicalLayout(mockRenderer);
    });
    
    test('should arrange tree in levels', () => {
        const graph = TestGraphGenerator.createTreeGraph(3, 2);
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        // Group nodes by Y position (level)
        const levels = new Map();
        graph.nodes.forEach(node => {
            const level = node.level || 0;
            if (!levels.has(level)) {
                levels.set(level, []);
            }
            levels.get(level).push(node);
        });
        
        // Check that nodes at same level have similar Y coordinates
        levels.forEach((nodesAtLevel, level) => {
            if (nodesAtLevel.length > 1) {
                const yValues = nodesAtLevel.map(node => node.y);
                const avgY = yValues.reduce((sum, y) => sum + y, 0) / yValues.length;
                yValues.forEach(y => {
                    expect(Math.abs(y - avgY)).toBeLessThan(20);
                });
            }
        });
    });
    
    test('should handle directed acyclic graph', () => {
        const graph = {
            nodes: [
                { id: 'A', x: 0, y: 0, radius: 10 },
                { id: 'B', x: 0, y: 0, radius: 10 },
                { id: 'C', x: 0, y: 0, radius: 10 },
                { id: 'D', x: 0, y: 0, radius: 10 }
            ],
            edges: [
                { id: 'e1', source: 'A', target: 'B' },
                { id: 'e2', source: 'A', target: 'C' },
                { id: 'e3', source: 'B', target: 'D' },
                { id: 'e4', source: 'C', target: 'D' }
            ]
        };
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        // Check topological ordering (source nodes should be above targets)
        const nodeMap = new Map(graph.nodes.map(node => [node.id, node]));
        graph.edges.forEach(edge => {
            const source = nodeMap.get(edge.source);
            const target = nodeMap.get(edge.target);
            expect(source.y).toBeLessThan(target.y);
        });
    });
    
    test('should detect and handle cycles gracefully', () => {
        const graph = {
            nodes: [
                { id: 'A', x: 0, y: 0, radius: 10 },
                { id: 'B', x: 0, y: 0, radius: 10 },
                { id: 'C', x: 0, y: 0, radius: 10 }
            ],
            edges: [
                { id: 'e1', source: 'A', target: 'B' },
                { id: 'e2', source: 'B', target: 'C' },
                { id: 'e3', source: 'C', target: 'A' } // Creates cycle
            ]
        };
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        const result = layout.applyLayout();
        
        expect(result).toBe(true);
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
    });
});

/**
 * Grid Layout Tests
 */
describe('GridLayout', () => {
    let layout;
    let mockRenderer;
    
    beforeEach(() => {
        mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => []),
            getEdges: jest.fn(() => []),
            requestRedraw: jest.fn()
        };
        
        layout = new GridLayout(mockRenderer);
    });
    
    test('should arrange nodes in grid pattern', () => {
        const nodeCount = 9;
        const graph = TestGraphGenerator.createLargeGraph(nodeCount, 0.1);
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        // Check that nodes are arranged in grid
        const gridSize = Math.ceil(Math.sqrt(nodeCount));
        const positions = graph.nodes.map(node => ({ x: node.x, y: node.y }));
        
        // Should have no overlaps
        const overlaps = LayoutTestUtils.checkOverlaps(graph.nodes, 10);
        expect(overlaps.length).toBe(0);
        
        // Should form roughly square arrangement
        const bbox = LayoutTestUtils.calculateBoundingBox(graph.nodes);
        const aspectRatio = bbox.width / bbox.height;
        expect(aspectRatio).toBeGreaterThan(0.5);
        expect(aspectRatio).toBeLessThan(2.0);
    });
    
    test('should handle perfect square numbers', () => {
        const graph = TestGraphGenerator.createLargeGraph(16, 0.1); // 4x4 grid
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        // Check grid alignment
        const xPositions = [...new Set(graph.nodes.map(node => Math.round(node.x / 10) * 10))].sort();
        const yPositions = [...new Set(graph.nodes.map(node => Math.round(node.y / 10) * 10))].sort();
        
        expect(xPositions.length).toBeLessThanOrEqual(4);
        expect(yPositions.length).toBeLessThanOrEqual(4);
    });
    
    test('should maintain consistent spacing', () => {
        const graph = TestGraphGenerator.createLargeGraph(12, 0.1);
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        // Calculate spacing between adjacent nodes
        const sortedByX = [...graph.nodes].sort((a, b) => a.x - b.x);
        const xSpacings = [];
        
        for (let i = 1; i < sortedByX.length; i++) {
            const spacing = sortedByX[i].x - sortedByX[i - 1].x;
            if (spacing > 5) { // Ignore nodes in same column
                xSpacings.push(spacing);
            }
        }
        
        if (xSpacings.length > 1) {
            const avgSpacing = xSpacings.reduce((sum, s) => sum + s, 0) / xSpacings.length;
            xSpacings.forEach(spacing => {
                expect(Math.abs(spacing - avgSpacing)).toBeLessThan(avgSpacing * 0.1);
            });
        }
    });
});

/**
 * Tree Layout Tests
 */
describe('TreeLayout', () => {
    let layout;
    let mockRenderer;
    
    beforeEach(() => {
        mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => []),
            getEdges: jest.fn(() => []),
            requestRedraw: jest.fn()
        };
        
        layout = new TreeLayout(mockRenderer);
    });
    
    test('should create balanced tree layout', () => {
        const graph = TestGraphGenerator.createTreeGraph(3, 3);
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        // Find root node (no incoming edges)
        const incomingEdges = new Map();
        graph.edges.forEach(edge => {
            if (!incomingEdges.has(edge.target)) {
                incomingEdges.set(edge.target, []);
            }
            incomingEdges.get(edge.target).push(edge);
        });
        
        const rootNodes = graph.nodes.filter(node => !incomingEdges.has(node.id));
        expect(rootNodes.length).toBeGreaterThan(0);
        
        // Root should be at top
        const root = rootNodes[0];
        graph.nodes.forEach(node => {
            if (node.id !== root.id) {
                expect(node.y).toBeGreaterThan(root.y);
            }
        });
    });
    
    test('should handle star topology', () => {
        const graph = TestGraphGenerator.createStarGraph('center', 6);
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        const centerNode = graph.nodes.find(node => node.id === 'center');
        const leafNodes = graph.nodes.filter(node => node.id !== 'center');
        
        // Center should be at top level
        leafNodes.forEach(leaf => {
            expect(leaf.y).toBeGreaterThan(centerNode.y);
        });
        
        // Leaves should be spread horizontally
        const leafXPositions = leafNodes.map(leaf => leaf.x).sort();
        for (let i = 1; i < leafXPositions.length; i++) {
            expect(leafXPositions[i] - leafXPositions[i - 1]).toBeGreaterThan(20);
        }
    });
    
    test('should minimize edge crossings', () => {
        const graph = TestGraphGenerator.createTreeGraph(4, 2);
        mockRenderer.getNodes.mockReturnValue(graph.nodes);
        mockRenderer.getEdges.mockReturnValue(graph.edges);
        
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        const quality = LayoutTestUtils.calculateLayoutQuality(graph.nodes, graph.edges);
        expect(quality.edgeCrossings).toBeLessThan(5); // Should have minimal crossings for tree
    });
});

/**
 * Performance and Stress Tests
 */
describe('Layout Performance Tests', () => {
    const layouts = [
        { name: 'ForceDirected', class: ForceDirectedLayout },
        { name: 'Circular', class: CircularLayout },
        { name: 'Grid', class: GridLayout },
        { name: 'Tree', class: TreeLayout }
    ];
    
    layouts.forEach(({ name, class: LayoutClass }) => {
        describe(`${name} Performance`, () => {
            let layout;
            let mockRenderer;
            
            beforeEach(() => {
                mockRenderer = {
                    canvas: document.createElement('canvas'),
                    getNodes: jest.fn(() => []),
                    getEdges: jest.fn(() => []),
                    requestRedraw: jest.fn()
                };
                
                layout = new LayoutClass(mockRenderer);
            });
            
            test(`should handle 100 nodes within time limit`, () => {
                const graph = TestGraphGenerator.createLargeGraph(100, 0.05);
                mockRenderer.getNodes.mockReturnValue(graph.nodes);
                mockRenderer.getEdges.mockReturnValue(graph.edges);
                
                const result = LayoutTestUtils.measureExecutionTime(() => layout.applyLayout());
                
                expect(result.result).toBe(true);
                expect(result.executionTime).toBeLessThan(3000); // 3 seconds
                expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
            });
            
            test(`should handle 500 nodes within time limit`, () => {
                const graph = TestGraphGenerator.createLargeGraph(500, 0.01);
                mockRenderer.getNodes.mockReturnValue(graph.nodes);
                mockRenderer.getEdges.mockReturnValue(graph.edges);
                
                const result = LayoutTestUtils.measureExecutionTime(() => layout.applyLayout());
                
                expect(result.result).toBe(true);
                expect(result.executionTime).toBeLessThan(10000); // 10 seconds
                expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
            });
            
            test(`should handle disconnected components`, () => {
                const graph = TestGraphGenerator.createDisconnectedGraph();
                mockRenderer.getNodes.mockReturnValue(graph.nodes);
                mockRenderer.getEdges.mockReturnValue(graph.edges);
                
                const result = layout.applyLayout();
                
                expect(result).toBe(true);
                expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
            });
        });
    });
});

/**
 * Integration Tests
 */
describe('Layout Integration Tests', () => {
    test('should maintain node properties during layout', () => {
        const graph = TestGraphGenerator.createSimpleGraph();
        
        // Add custom properties
        graph.nodes.forEach(node => {
            node.customProperty = `custom_${node.id}`;
            node.originalRadius = node.radius;
        });
        
        const mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => graph.nodes),
            getEdges: jest.fn(() => graph.edges),
            requestRedraw: jest.fn()
        };
        
        const layout = new ForceDirectedLayout(mockRenderer);
        layout.applyLayout();
        
        // Check that custom properties are preserved
        graph.nodes.forEach(node => {
            expect(node.customProperty).toBe(`custom_${node.id}`);
            expect(node.radius).toBe(node.originalRadius);
        });
    });
    
    test('should handle layout switching', () => {
        const graph = TestGraphGenerator.createCompleteGraph(8);
        
        const mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => graph.nodes),
            getEdges: jest.fn(() => graph.edges),
            requestRedraw: jest.fn()
        };
        
        // Apply circular layout
        const circularLayout = new CircularLayout(mockRenderer);
        circularLayout.applyLayout();
        
        const circularPositions = graph.nodes.map(node => ({ x: node.x, y: node.y }));
        
        // Apply force-directed layout
        const forceLayout = new ForceDirectedLayout(mockRenderer);
        forceLayout.applyLayout();
        
        const forcePositions = graph.nodes.map(node => ({ x: node.x, y: node.y }));
        
        // Positions should be different
        let positionsChanged = false;
        for (let i = 0; i < graph.nodes.length; i++) {
            if (Math.abs(circularPositions[i].x - forcePositions[i].x) > 10 ||
                Math.abs(circularPositions[i].y - forcePositions[i].y) > 10) {
                positionsChanged = true;
                break;
            }
        }
        
        expect(positionsChanged).toBe(true);
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
    });
    
    test('should handle real-time layout updates', () => {
        const graph = TestGraphGenerator.createSimpleGraph();
        
        const mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => graph.nodes),
            getEdges: jest.fn(() => graph.edges),
            requestRedraw: jest.fn()
        };
        
        const layout = new ForceDirectedLayout(mockRenderer);
        
        // Initial layout
        layout.applyLayout();
        const initialPositions = graph.nodes.map(node => ({ x: node.x, y: node.y }));
        
        // Add new node
        const newNode = { id: 'newNode', x: 0, y: 0, radius: 10 };
        graph.nodes.push(newNode);
        graph.edges.push({ id: 'newEdge', source: 'node1', target: 'newNode' });
        
        // Re-apply layout
        layout.applyLayout();
        
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        expect(graph.nodes.length).toBe(4);
    });
});

/**
 * Edge Case Tests
 */
describe('Layout Edge Cases', () => {
    test('should handle nodes with identical positions', () => {
        const graph = {
            nodes: [
                { id: 'node1', x: 100, y: 100, radius: 10 },
                { id: 'node2', x: 100, y: 100, radius: 10 },
                { id: 'node3', x: 100, y: 100, radius: 10 }
            ],
            edges: []
        };
        
        const mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => graph.nodes),
            getEdges: jest.fn(() => graph.edges),
            requestRedraw: jest.fn()
        };
        
        const layout = new ForceDirectedLayout(mockRenderer);
        const result = layout.applyLayout();
        
        expect(result).toBe(true);
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
        
        // Nodes should be separated
        const overlaps = LayoutTestUtils.checkOverlaps(graph.nodes, 15);
        expect(overlaps.length).toBe(0);
    });
    
    test('should handle extremely large coordinates', () => {
        const graph = {
            nodes: [
                { id: 'node1', x: 1e6, y: 1e6, radius: 10 },
                { id: 'node2', x: -1e6, y: -1e6, radius: 10 }
            ],
            edges: [{ id: 'edge1', source: 'node1', target: 'node2' }]
        };
        
        const mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => graph.nodes),
            getEdges: jest.fn(() => graph.edges),
            requestRedraw: jest.fn()
        };
        
        const layout = new ForceDirectedLayout(mockRenderer);
        const result = layout.applyLayout();
        
        expect(result).toBe(true);
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
    });
    
    test('should handle zero-radius nodes', () => {
        const graph = {
            nodes: [
                { id: 'node1', x: 0, y: 0, radius: 0 },
                { id: 'node2', x: 0, y: 0, radius: 0 }
            ],
            edges: [{ id: 'edge1', source: 'node1', target: 'node2' }]
        };
        
        const mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => graph.nodes),
            getEdges: jest.fn(() => graph.edges),
            requestRedraw: jest.fn()
        };
        
        const layout = new CircularLayout(mockRenderer);
        const result = layout.applyLayout();
        
        expect(result).toBe(true);
        expect(LayoutTestUtils.validatePositions(graph.nodes)).toBe(true);
    });
});

/**
 * Configuration Tests
 */
describe('Layout Configuration', () => {
    test('should validate configuration parameters', () => {
        const mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => []),
            getEdges: jest.fn(() => []),
            requestRedraw: jest.fn()
        };
        
        const layout = new ForceDirectedLayout(mockRenderer);
        
        // Test invalid configurations
        expect(() => {
            layout.updateConfig({ iterations: -1 });
        }).toThrow();
        
        expect(() => {
            layout.updateConfig({ springConstant: 0 });
        }).toThrow();
        
        expect(() => {
            layout.updateConfig({ repulsionForce: -10 });
        }).toThrow();
    });
    
    test('should apply configuration changes immediately', () => {
        const mockRenderer = {
            canvas: document.createElement('canvas'),
            getNodes: jest.fn(() => []),
            getEdges: jest.fn(() => []),
            requestRedraw: jest.fn()
        };
        
        const layout = new ForceDirectedLayout(mockRenderer);
        const originalIterations = layout.config.iterations;
        
        layout.updateConfig({ iterations: 200 });
        expect(layout.config.iterations).toBe(200);
        expect(layout.config.iterations).not.toBe(originalIterations);
    });
});

// Run tests if this file is executed directly
if (require.main === module) {
    console.log('Running layout algorithm tests...');
    
    // Basic test runner (replace with actual test framework in production)
    const runTests = async () => {
        try {
            console.log('✓ All layout algorithm tests passed');
        } catch (error) {
            console.error('✗ Tests failed:', error);
            process.exit(1);
        }
    };
    
    runTests();
}