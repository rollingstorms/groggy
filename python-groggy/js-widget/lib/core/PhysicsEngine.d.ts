/**
 * 🌟 Unified Physics Engine
 *
 * Handles force-directed layout with Barnes-Hut optimization
 * Used consistently across all visualization environments
 */
export class PhysicsEngine {
    constructor(config?: {});
    config: {
        charge: number;
        distance: number;
        damping: number;
        iterations: number;
        threshold: number;
        timeStep: number;
    };
    isRunning: boolean;
    iteration: number;
    totalEnergy: number;
    /**
     * 🎯 Main physics update method - used by all environments
     */
    updatePositions(nodes: any, edges: any, positions: any, velocities: any, deltaTime?: number): void;
    /**
     * Calculate repulsion forces between all nodes
     * Using Barnes-Hut approximation for better performance
     */
    calculateRepulsionForces(nodes: any, positions: any, forces: any): void;
    /**
     * Calculate attraction forces from edges (springs)
     */
    calculateAttractionForces(edges: any, positions: any, forces: any): void;
    /**
     * Integrate forces using Verlet integration
     */
    integrateForces(nodes: any, positions: any, velocities: any, forces: any, deltaTime: any): void;
    /**
     * Calculate total kinetic energy for convergence detection
     */
    calculateTotalEnergy(velocities: any): number;
    /**
     * Start physics simulation
     */
    start(): void;
    /**
     * Stop physics simulation
     */
    stop(): void;
    /**
     * Pause physics (can be resumed)
     */
    pause(): void;
    /**
     * Update configuration
     */
    updateConfig(newConfig: any): void;
    /**
     * Get current state
     */
    getState(): {
        isRunning: boolean;
        iteration: number;
        totalEnergy: number;
        config: {
            charge: number;
            distance: number;
            damping: number;
            iterations: number;
            threshold: number;
            timeStep: number;
        };
    };
    /**
     * Reset simulation state
     */
    reset(): void;
}
export default PhysicsEngine;
