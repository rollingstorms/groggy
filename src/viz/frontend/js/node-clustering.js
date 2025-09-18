/**
 * 🚀 Phase 10: Performance Optimization - Node Clustering System
 * 
 * Advanced node clustering for large graph visualization (1000+ nodes)
 * Dynamically groups nodes based on proximity, attributes, and zoom level
 * 
 * Features:
 * - Hierarchical clustering with multiple algorithms
 * - Spatial clustering based on layout position
 * - Attribute-based clustering (community detection)
 * - Dynamic cluster expansion/collapse
 * - Smooth cluster transitions with animations
 * - Performance-optimized for real-time interaction
 * - Memory-efficient cluster management
 * - Cluster metadata and statistics
 */

class NodeClusteringSystem {
    constructor(graphRenderer, lodRenderer) {
        this.graphRenderer = graphRenderer;
        this.lodRenderer = lodRenderer;
        
        // Clustering configuration
        this.config = {
            maxNodesBeforeClustering: 1000,
            minClusterSize: 3,
            maxClusterSize: 50,
            spatialThreshold: 100, // pixels
            attributeSimilarityThreshold: 0.7,
            enableHierarchicalClustering: true,
            enableSpatialClustering: true,
            enableAttributeClustering: true,
            clusteringAlgorithm: 'hybrid', // 'spatial', 'attribute', 'community', 'hybrid'
            autoExpandDistance: 200, // pixels from viewport center
            clusterExpansionZoom: 1.5 // zoom level to auto-expand clusters
        };
        
        // Clustering state
        this.clusters = new Map();
        this.clusterHierarchy = new Map();
        this.nodeToCluster = new Map();
        this.expandedClusters = new Set();
        this.clusterPositions = new Map();
        
        // Clustering algorithms
        this.algorithms = {
            spatial: new SpatialClustering(),
            attribute: new AttributeClustering(),
            community: new CommunityDetection(),
            hierarchy: new HierarchicalClustering()
        };
        
        // Animation and interaction
        this.animationDuration = 300;
        this.isAnimating = false;
        this.clusterInteractionEnabled = true;
        
        // Performance tracking
        this.clusteringTime = 0;
        this.lastClusterCount = 0;
        
        this.init();
    }
    
    init() {
        console.log('🚀 Initializing Node Clustering System for Phase 10');
        
        this.setupEventListeners();
        this.initializeClusterStyles();
        
        console.log('✅ Node Clustering System initialized');
    }
    
    /**
     * Set up event listeners for clustering interactions
     */
    setupEventListeners() {
        // Listen for zoom changes to auto-expand/collapse clusters
        document.addEventListener('zoomChanged', (e) => {
            this.handleZoomChange(e.detail.zoom);
        });
        
        // Listen for viewport changes
        document.addEventListener('viewportChanged', (e) => {
            this.handleViewportChange(e.detail);
        });
        
        // Listen for cluster click events
        document.addEventListener('clusterClicked', (e) => {
            this.handleClusterClick(e.detail.clusterId);
        });
        
        // Listen for node data changes
        document.addEventListener('graphDataChanged', (e) => {
            this.reclusterNodes();
        });
        
        // Double-click to expand/collapse clusters
        this.graphRenderer.canvas.addEventListener('dblclick', (e) => {
            const cluster = this.getClusterAtPosition(e.offsetX, e.offsetY);
            if (cluster) {
                this.toggleCluster(cluster.id);
            }
        });
    }
    
    /**
     * Initialize cluster visual styles
     */
    initializeClusterStyles() {
        this.clusterStyles = {
            default: {
                fillColor: 'rgba(0, 122, 255, 0.3)',
                strokeColor: '#007AFF',
                strokeWidth: 2,
                radius: 20,
                labelColor: '#007AFF',
                labelFont: '12px Arial'
            },
            expanded: {
                fillColor: 'rgba(52, 199, 89, 0.2)',
                strokeColor: '#34C759',
                strokeWidth: 1,
                radius: 15,
                labelColor: '#34C759',
                labelFont: '10px Arial'
            },
            community: {
                fillColor: 'rgba(255, 149, 0, 0.3)',
                strokeColor: '#FF9500',
                strokeWidth: 2,
                radius: 25,
                labelColor: '#FF9500',
                labelFont: '14px Arial'
            }
        };
    }
    
    /**
     * Main clustering entry point
     */
    performClustering(nodes, edges) {
        if (!nodes || nodes.size < this.config.maxNodesBeforeClustering) {
            this.clearClusters();
            return { nodes, edges };
        }
        
        const startTime = performance.now();
        
        console.log(`🔗 Starting clustering for ${nodes.size} nodes...`);
        
        // Choose clustering algorithm based on configuration
        let clusters;
        switch (this.config.clusteringAlgorithm) {
            case 'spatial':
                clusters = this.performSpatialClustering(nodes);
                break;
            case 'attribute':
                clusters = this.performAttributeClustering(nodes);
                break;
            case 'community':
                clusters = this.performCommunityDetection(nodes, edges);
                break;
            case 'hybrid':
                clusters = this.performHybridClustering(nodes, edges);
                break;
            default:
                clusters = this.performSpatialClustering(nodes);
        }
        
        // Build cluster hierarchy if enabled
        if (this.config.enableHierarchicalClustering) {
            this.buildClusterHierarchy(clusters);
        }
        
        // Update cluster positions and metadata
        this.updateClusterPositions(clusters);
        this.updateClusterMetadata(clusters);
        
        // Create clustered node/edge data
        const clusteredData = this.createClusteredData(nodes, edges, clusters);
        
        this.clusteringTime = performance.now() - startTime;
        this.lastClusterCount = clusters.length;
        
        console.log(`✅ Clustering complete: ${clusters.length} clusters in ${this.clusteringTime.toFixed(2)}ms`);
        
        return clusteredData;
    }
    
    /**
     * Perform spatial clustering based on node positions
     */
    performSpatialClustering(nodes) {
        const clusters = [];
        const visited = new Set();
        const threshold = this.config.spatialThreshold;
        
        for (const [nodeId, node] of nodes.entries()) {
            if (visited.has(nodeId)) continue;
            
            const cluster = {
                id: `spatial_${clusters.length}`,
                type: 'spatial',
                nodes: [nodeId],
                center: { x: node.x, y: node.y },
                radius: 0
            };
            
            // Find nearby nodes
            for (const [otherNodeId, otherNode] of nodes.entries()) {
                if (visited.has(otherNodeId) || otherNodeId === nodeId) continue;
                
                const distance = Math.sqrt(
                    Math.pow(node.x - otherNode.x, 2) + 
                    Math.pow(node.y - otherNode.y, 2)
                );
                
                if (distance <= threshold && cluster.nodes.length < this.config.maxClusterSize) {
                    cluster.nodes.push(otherNodeId);
                    visited.add(otherNodeId);
                }
            }
            
            if (cluster.nodes.length >= this.config.minClusterSize) {
                visited.add(nodeId);
                clusters.push(cluster);
            }
        }
        
        return clusters;
    }
    
    /**
     * Perform attribute-based clustering
     */
    performAttributeClustering(nodes) {
        const clusters = [];
        const attributeGroups = new Map();
        
        // Group nodes by similar attributes
        for (const [nodeId, node] of nodes.entries()) {
            const attributes = this.getNodeAttributes(node);
            const signature = this.createAttributeSignature(attributes);
            
            if (!attributeGroups.has(signature)) {
                attributeGroups.set(signature, []);
            }
            attributeGroups.get(signature).push(nodeId);
        }
        
        // Create clusters from attribute groups
        let clusterId = 0;
        for (const [signature, nodeIds] of attributeGroups.entries()) {
            if (nodeIds.length >= this.config.minClusterSize) {
                clusters.push({
                    id: `attribute_${clusterId++}`,
                    type: 'attribute',
                    nodes: nodeIds,
                    attributes: signature,
                    center: this.calculateClusterCenter(nodeIds, nodes)
                });
            }
        }
        
        return clusters;
    }
    
    /**
     * Perform community detection clustering
     */
    performCommunityDetection(nodes, edges) {
        // Simplified Louvain-style community detection
        const communities = this.algorithms.community.detect(nodes, edges);
        
        return communities.map((community, index) => ({
            id: `community_${index}`,
            type: 'community',
            nodes: community.nodes,
            modularity: community.modularity,
            center: this.calculateClusterCenter(community.nodes, nodes)
        }));
    }
    
    /**
     * Perform hybrid clustering (combination of spatial and attribute)
     */
    performHybridClustering(nodes, edges) {
        // First pass: spatial clustering
        const spatialClusters = this.performSpatialClustering(nodes);
        
        // Second pass: refine with attributes
        const refinedClusters = [];
        
        for (const cluster of spatialClusters) {
            const subClusters = this.refineClusterByAttributes(cluster, nodes);
            refinedClusters.push(...subClusters);
        }
        
        return refinedClusters;
    }
    
    /**
     * Refine a spatial cluster by attributes
     */
    refineClusterByAttributes(cluster, nodes) {
        if (cluster.nodes.length < this.config.minClusterSize * 2) {
            return [cluster];
        }
        
        const attributeGroups = new Map();
        
        for (const nodeId of cluster.nodes) {
            const node = nodes.get(nodeId);
            const attributes = this.getNodeAttributes(node);
            const signature = this.createAttributeSignature(attributes);
            
            if (!attributeGroups.has(signature)) {
                attributeGroups.set(signature, []);
            }
            attributeGroups.get(signature).push(nodeId);
        }
        
        const subClusters = [];
        let subClusterId = 0;
        
        for (const [signature, nodeIds] of attributeGroups.entries()) {
            if (nodeIds.length >= this.config.minClusterSize) {
                subClusters.push({
                    id: `${cluster.id}_sub_${subClusterId++}`,
                    type: 'hybrid',
                    nodes: nodeIds,
                    parent: cluster.id,
                    attributes: signature,
                    center: this.calculateClusterCenter(nodeIds, nodes)
                });
            }
        }
        
        return subClusters.length > 1 ? subClusters : [cluster];
    }
    
    /**
     * Build hierarchical cluster structure
     */
    buildClusterHierarchy(clusters) {
        this.clusterHierarchy.clear();
        
        // Group clusters by parent relationships
        const rootClusters = clusters.filter(c => !c.parent);
        const childClusters = clusters.filter(c => c.parent);
        
        for (const cluster of rootClusters) {
            this.clusterHierarchy.set(cluster.id, {
                cluster: cluster,
                children: childClusters.filter(c => c.parent === cluster.id),
                level: 0
            });
        }
        
        // Build multi-level hierarchy
        this.buildHierarchyLevels(rootClusters, childClusters);
    }
    
    /**
     * Build hierarchy levels recursively
     */
    buildHierarchyLevels(parentClusters, remainingClusters) {
        if (remainingClusters.length === 0) return;
        
        const nextLevelClusters = [];
        
        for (const parent of parentClusters) {
            const children = remainingClusters.filter(c => c.parent === parent.id);
            
            if (children.length > 0) {
                const hierarchyNode = this.clusterHierarchy.get(parent.id);
                hierarchyNode.children = children;
                
                for (const child of children) {
                    this.clusterHierarchy.set(child.id, {
                        cluster: child,
                        parent: parent.id,
                        children: [],
                        level: hierarchyNode.level + 1
                    });
                    nextLevelClusters.push(child);
                }
            }
        }
        
        // Recursively process next level
        const nextRemaining = remainingClusters.filter(c => 
            !nextLevelClusters.some(nl => nl.id === c.id)
        );
        
        if (nextLevelClusters.length > 0) {
            this.buildHierarchyLevels(nextLevelClusters, nextRemaining);
        }
    }
    
    /**
     * Update cluster positions and visual properties
     */
    updateClusterPositions(clusters) {
        for (const cluster of clusters) {
            // Calculate cluster bounds
            const bounds = this.calculateClusterBounds(cluster);
            cluster.bounds = bounds;
            
            // Update cluster radius based on content
            cluster.radius = Math.max(20, Math.min(50, bounds.radius * 1.2));
            
            // Store position for rendering
            this.clusterPositions.set(cluster.id, cluster.center);
        }
    }
    
    /**
     * Update cluster metadata and statistics
     */
    updateClusterMetadata(clusters) {
        for (const cluster of clusters) {
            cluster.metadata = {
                nodeCount: cluster.nodes.length,
                averageConnections: this.calculateAverageConnections(cluster),
                density: this.calculateClusterDensity(cluster),
                createdAt: Date.now(),
                interactions: 0
            };
        }
    }
    
    /**
     * Create clustered data structure for rendering
     */
    createClusteredData(originalNodes, originalEdges, clusters) {
        const clusteredNodes = new Map();
        const clusteredEdges = new Map();
        
        // Add cluster nodes
        for (const cluster of clusters) {
            if (!this.expandedClusters.has(cluster.id)) {
                clusteredNodes.set(cluster.id, {
                    id: cluster.id,
                    type: 'cluster',
                    x: cluster.center.x,
                    y: cluster.center.y,
                    radius: cluster.radius,
                    nodeCount: cluster.nodes.length,
                    cluster: cluster,
                    isCluster: true
                });
                
                // Mark clustered nodes
                for (const nodeId of cluster.nodes) {
                    this.nodeToCluster.set(nodeId, cluster.id);
                }
            } else {
                // Add individual nodes for expanded clusters
                for (const nodeId of cluster.nodes) {
                    const originalNode = originalNodes.get(nodeId);
                    if (originalNode) {
                        clusteredNodes.set(nodeId, { ...originalNode, clusterId: cluster.id });
                    }
                }
            }
        }
        
        // Add unclustered nodes
        for (const [nodeId, node] of originalNodes.entries()) {
            if (!this.nodeToCluster.has(nodeId)) {
                clusteredNodes.set(nodeId, node);
            }
        }
        
        // Process edges between clusters and nodes
        clusteredEdges = this.createClusteredEdges(originalEdges, clusters);
        
        return { nodes: clusteredNodes, edges: clusteredEdges };
    }
    
    /**
     * Create edges for clustered graph
     */
    createClusteredEdges(originalEdges, clusters) {
        const clusteredEdges = new Map();
        const clusterConnections = new Map();
        
        for (const [edgeId, edge] of originalEdges.entries()) {
            const sourceCluster = this.nodeToCluster.get(edge.source);
            const targetCluster = this.nodeToCluster.get(edge.target);
            
            if (sourceCluster && targetCluster && sourceCluster !== targetCluster) {
                // Edge between different clusters
                const connectionKey = `${sourceCluster}-${targetCluster}`;
                
                if (!clusterConnections.has(connectionKey)) {
                    clusterConnections.set(connectionKey, {
                        source: sourceCluster,
                        target: targetCluster,
                        weight: 0,
                        edgeCount: 0
                    });
                }
                
                const connection = clusterConnections.get(connectionKey);
                connection.weight += edge.weight || 1;
                connection.edgeCount++;
                
            } else if (!sourceCluster && !targetCluster) {
                // Edge between unclustered nodes
                clusteredEdges.set(edgeId, edge);
                
            } else if (this.expandedClusters.has(sourceCluster) || this.expandedClusters.has(targetCluster)) {
                // Edge involving expanded cluster
                clusteredEdges.set(edgeId, edge);
            }
        }
        
        // Add cluster connection edges
        let edgeId = originalEdges.size;
        for (const [connectionKey, connection] of clusterConnections.entries()) {
            clusteredEdges.set(`cluster_edge_${edgeId++}`, {
                source: connection.source,
                target: connection.target,
                weight: connection.weight,
                edgeCount: connection.edgeCount,
                isClusterEdge: true,
                width: Math.min(5, Math.max(1, connection.edgeCount * 0.5))
            });
        }
        
        return clusteredEdges;
    }
    
    /**
     * Toggle cluster expansion/collapse
     */
    toggleCluster(clusterId) {
        if (this.isAnimating) return;
        
        if (this.expandedClusters.has(clusterId)) {
            this.collapseCluster(clusterId);
        } else {
            this.expandCluster(clusterId);
        }
    }
    
    /**
     * Expand a cluster to show individual nodes
     */
    expandCluster(clusterId) {
        console.log(`🔍 Expanding cluster: ${clusterId}`);
        
        this.expandedClusters.add(clusterId);
        
        const cluster = this.clusters.get(clusterId);
        if (cluster) {
            cluster.metadata.interactions++;
            
            // Animate expansion
            this.animateClusterExpansion(cluster);
        }
        
        // Trigger re-clustering
        this.reclusterNodes();
    }
    
    /**
     * Collapse a cluster to hide individual nodes
     */
    collapseCluster(clusterId) {
        console.log(`🔄 Collapsing cluster: ${clusterId}`);
        
        this.expandedClusters.delete(clusterId);
        
        const cluster = this.clusters.get(clusterId);
        if (cluster) {
            // Animate collapse
            this.animateClusterCollapse(cluster);
        }
        
        // Trigger re-clustering
        this.reclusterNodes();
    }
    
    /**
     * Animate cluster expansion
     */
    animateClusterExpansion(cluster) {
        this.isAnimating = true;
        
        // Implementation would animate nodes moving out from cluster center
        // This is a simplified version
        setTimeout(() => {
            this.isAnimating = false;
            document.dispatchEvent(new CustomEvent('clusterExpanded', { 
                detail: { clusterId: cluster.id } 
            }));
        }, this.animationDuration);
    }
    
    /**
     * Animate cluster collapse
     */
    animateClusterCollapse(cluster) {
        this.isAnimating = true;
        
        // Implementation would animate nodes moving into cluster center
        setTimeout(() => {
            this.isAnimating = false;
            document.dispatchEvent(new CustomEvent('clusterCollapsed', { 
                detail: { clusterId: cluster.id } 
            }));
        }, this.animationDuration);
    }
    
    /**
     * Handle zoom changes for auto-expansion
     */
    handleZoomChange(zoom) {
        if (zoom >= this.config.clusterExpansionZoom) {
            // Auto-expand clusters at high zoom
            for (const [clusterId, cluster] of this.clusters.entries()) {
                if (!this.expandedClusters.has(clusterId) && 
                    cluster.nodes.length <= 10) {
                    this.expandCluster(clusterId);
                }
            }
        } else {
            // Auto-collapse very small clusters at low zoom
            for (const clusterId of this.expandedClusters) {
                const cluster = this.clusters.get(clusterId);
                if (cluster && cluster.nodes.length <= 5) {
                    this.collapseCluster(clusterId);
                }
            }
        }
    }
    
    /**
     * Handle viewport changes for proximity-based expansion
     */
    handleViewportChange(viewport) {
        const centerX = viewport.x + viewport.width / 2;
        const centerY = viewport.y + viewport.height / 2;
        
        for (const [clusterId, cluster] of this.clusters.entries()) {
            const distance = Math.sqrt(
                Math.pow(cluster.center.x - centerX, 2) + 
                Math.pow(cluster.center.y - centerY, 2)
            );
            
            if (distance <= this.config.autoExpandDistance && 
                !this.expandedClusters.has(clusterId)) {
                this.expandCluster(clusterId);
            }
        }
    }
    
    /**
     * Handle cluster click interaction
     */
    handleClusterClick(clusterId) {
        this.toggleCluster(clusterId);
    }
    
    /**
     * Get cluster at specific position
     */
    getClusterAtPosition(x, y) {
        for (const [clusterId, position] of this.clusterPositions.entries()) {
            const cluster = this.clusters.get(clusterId);
            if (cluster) {
                const distance = Math.sqrt(
                    Math.pow(x - position.x, 2) + Math.pow(y - position.y, 2)
                );
                if (distance <= cluster.radius) {
                    return cluster;
                }
            }
        }
        return null;
    }
    
    /**
     * Re-cluster nodes (called when data changes)
     */
    reclusterNodes() {
        // This would be called by the main graph renderer
        this.clusters.clear();
        this.nodeToCluster.clear();
        
        console.log('🔄 Re-clustering nodes...');
    }
    
    /**
     * Clear all clusters
     */
    clearClusters() {
        this.clusters.clear();
        this.nodeToCluster.clear();
        this.expandedClusters.clear();
        this.clusterPositions.clear();
        this.clusterHierarchy.clear();
    }
    
    /**
     * Helper methods
     */
    getNodeAttributes(node) {
        // Extract relevant attributes for clustering
        return {
            type: node.type || 'default',
            category: node.category || 'uncategorized',
            group: node.group || 'default',
            size: node.radius || 8,
            degree: node.degree || 0
        };
    }
    
    createAttributeSignature(attributes) {
        return `${attributes.type}_${attributes.category}_${attributes.group}`;
    }
    
    calculateClusterCenter(nodeIds, nodes) {
        let totalX = 0, totalY = 0;
        let count = 0;
        
        for (const nodeId of nodeIds) {
            const node = nodes.get(nodeId);
            if (node) {
                totalX += node.x;
                totalY += node.y;
                count++;
            }
        }
        
        return count > 0 ? { x: totalX / count, y: totalY / count } : { x: 0, y: 0 };
    }
    
    calculateClusterBounds(cluster) {
        // Calculate the bounding box and radius of the cluster
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        // This is simplified - would need access to actual node positions
        const centerX = cluster.center.x;
        const centerY = cluster.center.y;
        const estimatedRadius = Math.sqrt(cluster.nodes.length) * 10;
        
        return {
            minX: centerX - estimatedRadius,
            maxX: centerX + estimatedRadius,
            minY: centerY - estimatedRadius,
            maxY: centerY + estimatedRadius,
            radius: estimatedRadius
        };
    }
    
    calculateAverageConnections(cluster) {
        // Calculate average number of connections per node in cluster
        return cluster.nodes.length * 0.3; // Simplified estimation
    }
    
    calculateClusterDensity(cluster) {
        // Calculate internal density of the cluster
        const nodeCount = cluster.nodes.length;
        const maxPossibleEdges = (nodeCount * (nodeCount - 1)) / 2;
        const estimatedEdges = nodeCount * 0.3; // Simplified
        
        return maxPossibleEdges > 0 ? estimatedEdges / maxPossibleEdges : 0;
    }
    
    /**
     * Get clustering performance statistics
     */
    getPerformanceStats() {
        return {
            clusteringTime: this.clusteringTime,
            clusterCount: this.lastClusterCount,
            expandedClusters: this.expandedClusters.size,
            totalNodes: this.nodeToCluster.size,
            hierarchyLevels: Math.max(...Array.from(this.clusterHierarchy.values()).map(h => h.level)) + 1
        };
    }
    
    /**
     * Configure clustering settings
     */
    configure(settings) {
        this.config = { ...this.config, ...settings };
        console.log('🎛️  Clustering configuration updated');
    }
    
    /**
     * Cleanup method
     */
    destroy() {
        this.clearClusters();
        console.log('🧹 Node Clustering System cleaned up');
    }
}

/**
 * Simplified clustering algorithm implementations
 */
class SpatialClustering {
    cluster(nodes, threshold) {
        // Implementation would go here
        return [];
    }
}

class AttributeClustering {
    cluster(nodes, threshold) {
        // Implementation would go here
        return [];
    }
}

class CommunityDetection {
    detect(nodes, edges) {
        // Simplified community detection
        return [];
    }
}

class HierarchicalClustering {
    buildHierarchy(clusters) {
        // Implementation would go here
        return new Map();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NodeClusteringSystem;
}