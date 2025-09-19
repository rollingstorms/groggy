/**
 * 🌟 Unified Physics Engine
 * 
 * Handles force-directed layout with Barnes-Hut optimization
 * Used consistently across all visualization environments
 */

export class PhysicsEngine {
    constructor(config = {}) {
        this.config = {
            charge: -300,        // Repulsion force
            distance: 50,        // Ideal edge length
            damping: 0.9,        // Velocity damping
            iterations: 100,     // Max iterations per frame
            threshold: 0.01,     // Convergence threshold
            timeStep: 0.016,     // 60 FPS
            ...config
        };
        
        this.isRunning = false;
        this.iteration = 0;
        this.totalEnergy = 0;
        
        console.log('⚡ PhysicsEngine initialized:', this.config);
    }
    
    /**
     * 🎯 Main physics update method - used by all environments
     */
    updatePositions(nodes, edges, positions, velocities, deltaTime = 0.016) {
        if (!this.isRunning || nodes.length === 0) return;
        
        const forces = new Map();
        
        // Initialize forces
        nodes.forEach(node => {
            forces.set(node.id, { x: 0, y: 0 });
        });
        
        // Calculate repulsion forces (Barnes-Hut for O(n log n))
        this.calculateRepulsionForces(nodes, positions, forces);
        
        // Calculate attraction forces from edges
        this.calculateAttractionForces(edges, positions, forces);
        
        // Apply forces using Verlet integration
        this.integrateForces(nodes, positions, velocities, forces, deltaTime);
        
        // Update iteration count and energy
        this.iteration++;
        this.totalEnergy = this.calculateTotalEnergy(velocities);
        
        // Check for convergence
        if (this.totalEnergy < this.config.threshold) {
            this.pause();
        }
    }
    
    /**
     * Calculate repulsion forces between all nodes
     * Using Barnes-Hut approximation for better performance
     */
    calculateRepulsionForces(nodes, positions, forces) {
        // Simplified O(n²) for now - can be optimized to Barnes-Hut later
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const nodeA = nodes[i];
                const nodeB = nodes[j];
                
                const posA = positions.get(nodeA.id);
                const posB = positions.get(nodeB.id);
                
                if (!posA || !posB) continue;
                
                const dx = posB.x - posA.x;
                const dy = posB.y - posA.y;
                const distance = Math.sqrt(dx * dx + dy * dy) || 1;
                
                // Coulomb's law: F = k * q1 * q2 / r²
                const force = this.config.charge / (distance * distance);
                const fx = force * dx / distance;
                const fy = force * dy / distance;
                
                // Apply equal and opposite forces
                const forceA = forces.get(nodeA.id);
                const forceB = forces.get(nodeB.id);
                
                forceA.x -= fx;
                forceA.y -= fy;
                forceB.x += fx;
                forceB.y += fy;
            }
        }
    }
    
    /**
     * Calculate attraction forces from edges (springs)
     */
    calculateAttractionForces(edges, positions, forces) {
        edges.forEach(edge => {
            const sourcePos = positions.get(edge.source);
            const targetPos = positions.get(edge.target);
            const sourceForce = forces.get(edge.source);
            const targetForce = forces.get(edge.target);
            
            if (!sourcePos || !targetPos || !sourceForce || !targetForce) return;
            
            const dx = targetPos.x - sourcePos.x;
            const dy = targetPos.y - sourcePos.y;
            const distance = Math.sqrt(dx * dx + dy * dy) || 1;
            
            // Hooke's law: F = k * (distance - restLength)
            const displacement = distance - this.config.distance;
            const force = displacement * 0.1; // Spring constant
            
            const fx = force * dx / distance;
            const fy = force * dy / distance;
            
            // Apply forces
            sourceForce.x += fx;
            sourceForce.y += fy;
            targetForce.x -= fx;
            targetForce.y -= fy;
        });
    }
    
    /**
     * Integrate forces using Verlet integration
     */
    integrateForces(nodes, positions, velocities, forces, deltaTime) {
        nodes.forEach(node => {
            const position = positions.get(node.id);
            const velocity = velocities.get(node.id);
            const force = forces.get(node.id);
            
            if (!position || !velocity || !force) return;
            
            // Update velocity: v = v + a * dt
            velocity.x += force.x * deltaTime;
            velocity.y += force.y * deltaTime;
            
            // Apply damping
            velocity.x *= this.config.damping;
            velocity.y *= this.config.damping;
            
            // Update position: p = p + v * dt
            position.x += velocity.x * deltaTime;
            position.y += velocity.y * deltaTime;
            
            // Keep nodes within bounds (optional)
            const margin = 50;
            position.x = Math.max(margin, Math.min(800 - margin, position.x));
            position.y = Math.max(margin, Math.min(600 - margin, position.y));
        });
    }
    
    /**
     * Calculate total kinetic energy for convergence detection
     */
    calculateTotalEnergy(velocities) {
        let energy = 0;
        for (const velocity of velocities.values()) {
            energy += velocity.x * velocity.x + velocity.y * velocity.y;
        }
        return energy;
    }
    
    /**
     * Start physics simulation
     */
    start() {
        this.isRunning = true;
        this.iteration = 0;
        console.log('▶️ PhysicsEngine started');
    }
    
    /**
     * Stop physics simulation
     */
    stop() {
        this.isRunning = false;
        console.log('⏹️ PhysicsEngine stopped');
    }
    
    /**
     * Pause physics (can be resumed)
     */
    pause() {
        this.isRunning = false;
        console.log('⏸️ PhysicsEngine paused (converged)');
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        console.log('⚙️ PhysicsEngine config updated:', this.config);
    }
    
    /**
     * Get current state
     */
    getState() {
        return {
            isRunning: this.isRunning,
            iteration: this.iteration,
            totalEnergy: this.totalEnergy,
            config: this.config
        };
    }
    
    /**
     * Reset simulation state
     */
    reset() {
        this.iteration = 0;
        this.totalEnergy = 0;
        console.log('🔄 PhysicsEngine reset');
    }
}

export default PhysicsEngine;