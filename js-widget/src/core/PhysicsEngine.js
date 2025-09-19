/**
 * ⚛️ PhysicsEngine - Force-Directed Layout Simulation
 * 
 * Unified physics simulation for all visualization environments.
 * Provides consistent force-directed layout across Jupyter widgets,
 * WebSocket streaming, and static exports.
 * 
 * Features:
 * - Verlet integration for stable simulation
 * - Multiple force types (charge, link, center, collision)
 * - Adaptive time stepping
 * - Performance optimizations for large graphs
 * - Barnes-Hut approximation for n-body forces
 */

export class PhysicsEngine {
    constructor(config = {}) {
        this.config = {
            // Force strengths
            forceStrength: config.forceStrength || 30,
            linkDistance: config.linkDistance || 50,
            linkStrength: config.linkStrength || 0.1,
            chargeStrength: config.chargeStrength || -300,
            centerStrength: config.centerStrength || 0.1,
            collisionRadius: config.collisionRadius || 25,
            
            // Simulation parameters
            velocityDecay: config.velocityDecay || 0.4,
            alpha: config.alpha || 1.0,
            alphaDecay: config.alphaDecay || 0.0228,
            alphaMin: config.alphaMin || 0.001,
            alphaTarget: config.alphaTarget || 0,
            
            // Performance optimizations
            useBarnesHut: config.useBarnesHut !== false,
            barnesHutTheta: config.barnesHutTheta || 0.9,
            maxNodes: config.maxNodes || 5000,
            
            // Time stepping
            timeStep: config.timeStep || 0.016, // 60 FPS
            maxTimeStep: config.maxTimeStep || 0.033, // 30 FPS minimum
            
            ...config
        };
        
        // Simulation state
        this.alpha = this.config.alpha;
        this.isActive = false;
        this.frameCount = 0;
        
        // Data
        this.nodes = [];
        this.edges = [];
        this.forces = new Map();
        
        // Performance tracking
        this.lastFrameTime = 0;
        this.averageFrameTime = 0;
        this.frameTimeHistory = [];
        
        // Initialize force functions
        this.initializeForces();
        
        console.log('⚛️ PhysicsEngine initialized with force-directed simulation');
    }
    
    /**
     * 🎯 Set simulation data
     */
    setData(nodes, edges) {
        this.nodes = nodes;
        this.edges = edges;
        
        // Reset alpha to restart simulation with new data
        this.alpha = this.config.alpha;
        
        console.log(`⚛️ Physics simulation set with ${nodes.length} nodes, ${edges.length} edges`);
    }
    
    /**
     * ▶️ Start simulation
     */
    start() {
        this.isActive = true;
        this.alpha = Math.max(this.alpha, this.config.alphaTarget);
        console.log('⚛️ Physics simulation started');
    }
    
    /**
     * ⏹️ Stop simulation
     */
    stop() {
        this.isActive = false;
        console.log('⚛️ Physics simulation stopped');
    }
    
    /**
     * 🔄 Update node positions using Verlet integration
     */
    updatePositions(nodes, edges, nodePositions, nodeVelocities, deltaTime) {
        if (!this.isActive || this.alpha < this.config.alphaMin) {
            this.stop();
            return;
        }
        
        const startTime = performance.now();
        
        // Adaptive time stepping
        const timeStep = Math.min(deltaTime / 1000, this.config.maxTimeStep);
        
        // Clear forces
        const forces = new Map();
        nodes.forEach(node => {
            forces.set(node.id, { x: 0, y: 0 });
        });
        
        // Apply all force types
        this.applyChargeForces(nodes, nodePositions, forces);
        this.applyLinkForces(edges, nodePositions, forces);
        this.applyCenterForces(nodes, nodePositions, forces);
        this.applyCollisionForces(nodes, nodePositions, forces);
        
        // Update positions using Verlet integration
        nodes.forEach(node => {
            const pos = nodePositions.get(node.id);
            const vel = nodeVelocities.get(node.id);
            const force = forces.get(node.id);
            
            if (!pos || !vel || !force) return;
            
            // Apply forces to velocity
            vel.vx += force.x * timeStep;
            vel.vy += force.y * timeStep;
            
            // Apply velocity decay
            vel.vx *= this.config.velocityDecay;
            vel.vy *= this.config.velocityDecay;
            
            // Update position
            pos.x += vel.vx * timeStep;
            pos.y += vel.vy * timeStep;
        });
        
        // Update alpha (cooling)
        this.alpha += (this.config.alphaTarget - this.alpha) * this.config.alphaDecay;
        this.frameCount++;
        
        // Performance tracking
        const frameTime = performance.now() - startTime;
        this.updatePerformanceMetrics(frameTime);
    }
    
    /**
     * ⚡ Apply repulsive charge forces between nodes
     */
    applyChargeForces(nodes, nodePositions, forces) {
        if (this.config.useBarnesHut && nodes.length > 50) {
            this.applyChargeForcesBH(nodes, nodePositions, forces);
        } else {
            this.applyChargeForcesNaive(nodes, nodePositions, forces);
        }
    }
    
    /**
     * ⚡ Naive O(n²) charge force calculation
     */
    applyChargeForcesNaive(nodes, nodePositions, forces) {
        const strength = this.config.chargeStrength * this.alpha;
        
        for (let i = 0; i < nodes.length; i++) {
            const nodeA = nodes[i];
            const posA = nodePositions.get(nodeA.id);
            const forceA = forces.get(nodeA.id);
            
            if (!posA || !forceA) continue;
            
            for (let j = i + 1; j < nodes.length; j++) {
                const nodeB = nodes[j];
                const posB = nodePositions.get(nodeB.id);
                const forceB = forces.get(nodeB.id);
                
                if (!posB || !forceB) continue;
                
                const dx = posB.x - posA.x;
                const dy = posB.y - posA.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 1) continue; // Avoid division by zero
                
                const force = strength / (distance * distance);
                const fx = (dx / distance) * force;
                const fy = (dy / distance) * force;
                
                forceA.x -= fx;
                forceA.y -= fy;
                forceB.x += fx;
                forceB.y += fy;
            }
        }
    }
    
    /**
     * ⚡ Barnes-Hut O(n log n) approximation for charge forces
     */
    applyChargeForcesBH(nodes, nodePositions, forces) {
        // Build quadtree
        const quadtree = this.buildQuadtree(nodes, nodePositions);
        
        // Apply forces using quadtree
        const strength = this.config.chargeStrength * this.alpha;
        const theta = this.config.barnesHutTheta;
        
        nodes.forEach(node => {
            const pos = nodePositions.get(node.id);
            const force = forces.get(node.id);
            
            if (!pos || !force) return;
            
            const appliedForce = this.calculateBHForce(quadtree, pos, theta, strength);
            force.x += appliedForce.x;
            force.y += appliedForce.y;
        });
    }
    
    /**
     * 🔗 Apply attractive link forces
     */
    applyLinkForces(edges, nodePositions, forces) {
        const strength = this.config.linkStrength * this.alpha;
        const distance = this.config.linkDistance;
        
        edges.forEach(edge => {
            const sourcePos = nodePositions.get(edge.source);
            const targetPos = nodePositions.get(edge.target);
            const sourceForce = forces.get(edge.source);
            const targetForce = forces.get(edge.target);
            
            if (!sourcePos || !targetPos || !sourceForce || !targetForce) return;
            
            const dx = targetPos.x - sourcePos.x;
            const dy = targetPos.y - sourcePos.y;
            const currentDistance = Math.sqrt(dx * dx + dy * dy);
            
            if (currentDistance < 1) return; // Avoid division by zero
            
            const force = (currentDistance - distance) * strength;
            const fx = (dx / currentDistance) * force;
            const fy = (dy / currentDistance) * force;
            
            sourceForce.x += fx;
            sourceForce.y += fy;
            targetForce.x -= fx;
            targetForce.y -= fy;
        });
    }
    
    /**
     * 🎯 Apply centering forces to keep graph centered
     */
    applyCenterForces(nodes, nodePositions, forces) {
        const strength = this.config.centerStrength * this.alpha;
        
        // Calculate center of mass
        let centerX = 0, centerY = 0;
        nodes.forEach(node => {
            const pos = nodePositions.get(node.id);
            if (pos) {
                centerX += pos.x;
                centerY += pos.y;
            }
        });
        
        if (nodes.length > 0) {
            centerX /= nodes.length;
            centerY /= nodes.length;
            
            // Apply centering force
            nodes.forEach(node => {
                const pos = nodePositions.get(node.id);
                const force = forces.get(node.id);
                
                if (pos && force) {
                    force.x -= centerX * strength;
                    force.y -= centerY * strength;
                }
            });
        }
    }
    
    /**
     * 💥 Apply collision forces to prevent node overlap
     */
    applyCollisionForces(nodes, nodePositions, forces) {
        const radius = this.config.collisionRadius;
        const strength = this.alpha;
        
        for (let i = 0; i < nodes.length; i++) {
            const nodeA = nodes[i];
            const posA = nodePositions.get(nodeA.id);
            const forceA = forces.get(nodeA.id);
            
            if (!posA || !forceA) continue;
            
            for (let j = i + 1; j < nodes.length; j++) {
                const nodeB = nodes[j];
                const posB = nodePositions.get(nodeB.id);
                const forceB = forces.get(nodeB.id);
                
                if (!posB || !forceB) continue;
                
                const dx = posB.x - posA.x;
                const dy = posB.y - posA.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                const minDistance = radius * 2;
                
                if (distance < minDistance && distance > 0) {
                    const overlap = minDistance - distance;
                    const force = overlap * strength * 0.5;
                    const fx = (dx / distance) * force;
                    const fy = (dy / distance) * force;
                    
                    forceA.x -= fx;
                    forceA.y -= fy;
                    forceB.x += fx;
                    forceB.y += fy;
                }
            }
        }
    }
    
    /**
     * 🌳 Build quadtree for Barnes-Hut approximation
     */
    buildQuadtree(nodes, nodePositions) {
        // Find bounds
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;
        
        nodes.forEach(node => {
            const pos = nodePositions.get(node.id);
            if (pos) {
                minX = Math.min(minX, pos.x);
                minY = Math.min(minY, pos.y);
                maxX = Math.max(maxX, pos.x);
                maxY = Math.max(maxY, pos.y);
            }
        });
        
        // Create root quadtree node
        const size = Math.max(maxX - minX, maxY - minY) || 1;
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        
        const root = {
            x: centerX - size / 2,
            y: centerY - size / 2,
            size: size,
            mass: 0,
            centerX: 0,
            centerY: 0,
            children: null,
            nodes: []
        };
        
        // Insert nodes
        nodes.forEach(node => {
            const pos = nodePositions.get(node.id);
            if (pos) {
                this.insertNodeIntoQuadtree(root, { id: node.id, x: pos.x, y: pos.y });
            }
        });
        
        return root;
    }
    
    /**
     * 📍 Insert node into quadtree
     */
    insertNodeIntoQuadtree(quad, node) {
        if (!this.isInQuadtree(quad, node.x, node.y)) return;
        
        quad.mass += 1;
        quad.centerX = (quad.centerX * (quad.mass - 1) + node.x) / quad.mass;
        quad.centerY = (quad.centerY * (quad.mass - 1) + node.y) / quad.mass;
        
        if (!quad.children) {
            if (quad.nodes.length === 0) {
                quad.nodes.push(node);
                return;
            }
            
            // Subdivide
            this.subdivideQuadtree(quad);
            
            // Move existing node to child
            const existingNode = quad.nodes[0];
            quad.nodes = [];
            this.insertNodeIntoQuadtree(quad, existingNode);
        }
        
        // Insert into appropriate child
        const half = quad.size / 2;
        const childIndex = (node.x < quad.x + half ? 0 : 1) + (node.y < quad.y + half ? 0 : 2);
        this.insertNodeIntoQuadtree(quad.children[childIndex], node);
    }
    
    /**
     * 🔪 Subdivide quadtree node
     */
    subdivideQuadtree(quad) {
        const half = quad.size / 2;
        quad.children = [
            { x: quad.x, y: quad.y, size: half, mass: 0, centerX: 0, centerY: 0, children: null, nodes: [] },
            { x: quad.x + half, y: quad.y, size: half, mass: 0, centerX: 0, centerY: 0, children: null, nodes: [] },
            { x: quad.x, y: quad.y + half, size: half, mass: 0, centerX: 0, centerY: 0, children: null, nodes: [] },
            { x: quad.x + half, y: quad.y + half, size: half, mass: 0, centerX: 0, centerY: 0, children: null, nodes: [] }
        ];
    }
    
    /**
     * 📏 Check if point is in quadtree bounds
     */
    isInQuadtree(quad, x, y) {
        return x >= quad.x && x < quad.x + quad.size && 
               y >= quad.y && y < quad.y + quad.size;
    }
    
    /**
     * ⚡ Calculate Barnes-Hut force for a point
     */
    calculateBHForce(quad, pos, theta, strength) {
        const force = { x: 0, y: 0 };
        
        if (quad.mass === 0) return force;
        
        const dx = quad.centerX - pos.x;
        const dy = quad.centerY - pos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 1) return force;
        
        // If quad is far enough or is a leaf, apply force
        if (!quad.children || quad.size / distance < theta) {
            const forceStrength = strength * quad.mass / (distance * distance);
            force.x = (dx / distance) * forceStrength;
            force.y = (dy / distance) * forceStrength;
        } else {
            // Recurse into children
            quad.children.forEach(child => {
                const childForce = this.calculateBHForce(child, pos, theta, strength);
                force.x += childForce.x;
                force.y += childForce.y;
            });
        }
        
        return force;
    }
    
    /**
     * 📊 Update performance metrics
     */
    updatePerformanceMetrics(frameTime) {
        this.lastFrameTime = frameTime;
        this.frameTimeHistory.push(frameTime);
        
        // Keep only last 60 frames for average
        if (this.frameTimeHistory.length > 60) {
            this.frameTimeHistory.shift();
        }
        
        this.averageFrameTime = this.frameTimeHistory.reduce((a, b) => a + b, 0) / this.frameTimeHistory.length;
    }
    
    /**
     * 🔧 Initialize force functions map
     */
    initializeForces() {
        this.forces.set('charge', this.applyChargeForces.bind(this));
        this.forces.set('link', this.applyLinkForces.bind(this));
        this.forces.set('center', this.applyCenterForces.bind(this));
        this.forces.set('collision', this.applyCollisionForces.bind(this));
    }
    
    /**
     * ⚙️ Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        
        // Reset alpha if significant changes
        if (newConfig.alpha !== undefined) {
            this.alpha = newConfig.alpha;
        }
    }
    
    /**
     * 📈 Get performance metrics
     */
    getPerformanceMetrics() {
        return {
            frameCount: this.frameCount,
            lastFrameTime: this.lastFrameTime,
            averageFrameTime: this.averageFrameTime,
            fps: this.averageFrameTime > 0 ? 1000 / this.averageFrameTime : 0,
            alpha: this.alpha,
            isActive: this.isActive
        };
    }
    
    /**
     * 🎯 Check if simulation is active
     */
    isSimulationActive() {
        return this.isActive && this.alpha >= this.config.alphaMin;
    }
    
    /**
     * 📊 Get frame count
     */
    getFrameCount() {
        return this.frameCount;
    }
    
    /**
     * 🧹 Cleanup
     */
    destroy() {
        this.stop();
        this.forces.clear();
        this.frameTimeHistory = [];
        console.log('⚛️ PhysicsEngine destroyed');
    }
}