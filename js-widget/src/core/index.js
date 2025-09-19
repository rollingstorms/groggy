/**
 * 🧠 Unified JavaScript Core - Exports
 * 
 * Single source of truth for all visualization logic.
 * This unified core is used by:
 * - Jupyter widgets
 * - WebSocket streaming clients
 * - Static file exports
 * - Future visualization environments
 * 
 * Benefits of this unified architecture:
 * ✅ Single physics implementation
 * ✅ Single rendering pipeline
 * ✅ Single interaction system
 * ✅ Consistent behavior across all environments
 * ✅ Easier maintenance and feature development
 * ✅ Guaranteed feature parity
 */

// Core engine
export { GroggyVizCore } from './VizCore.js';

// Subsystems
export { PhysicsEngine } from './PhysicsEngine.js';
export { SVGRenderer } from './SVGRenderer.js';
export { InteractionEngine } from './InteractionEngine.js';

// Client implementations
export { StreamingClient } from './StreamingClient.js';

// Utility functions
export function createStandardConfig(overrides = {}) {
    return {
        width: 800,
        height: 600,
        
        physics: {
            enabled: true,
            forceStrength: 30,
            linkDistance: 50,
            linkStrength: 0.1,
            chargeStrength: -300,
            centerStrength: 0.1,
            velocityDecay: 0.4,
            alpha: 1.0,
            alphaDecay: 0.0228,
            alphaMin: 0.001,
            ...overrides.physics
        },
        
        rendering: {
            backgroundColor: '#ffffff',
            nodeColorScheme: 'default',
            enableShadows: true,
            enableAnimations: true,
            enableLOD: true,
            lodThreshold: 1000,
            ...overrides.rendering
        },
        
        interaction: {
            enableDrag: true,
            enableZoom: true,
            enablePan: true,
            enableSelection: true,
            enableHover: true,
            dragThreshold: 5,
            zoomExtent: [0.1, 10],
            ...overrides.interaction
        },
        
        ...overrides
    };
}

export function createDarkThemeConfig(overrides = {}) {
    const darkConfig = createStandardConfig(overrides);
    
    darkConfig.rendering = {
        ...darkConfig.rendering,
        backgroundColor: '#1a1a1a',
        nodeColorScheme: 'dark',
        edgeStroke: '#666',
        ...overrides.rendering
    };
    
    return darkConfig;
}

export function createHighPerformanceConfig(overrides = {}) {
    const perfConfig = createStandardConfig(overrides);
    
    perfConfig.rendering = {
        ...perfConfig.rendering,
        enableShadows: false,
        enableAnimations: false,
        enableLOD: true,
        lodThreshold: 500,
        ...overrides.rendering
    };
    
    perfConfig.physics = {
        ...perfConfig.physics,
        useBarnesHut: true,
        barnesHutTheta: 0.9,
        ...overrides.physics
    };
    
    return perfConfig;
}

// Version information
export const VERSION = '1.0.0';
export const UNIFIED_ARCHITECTURE_VERSION = 'Phase3-Complete';

console.log(`🧠 Groggy Unified JavaScript Core v${VERSION} loaded`);
console.log(`🏗️ Architecture: ${UNIFIED_ARCHITECTURE_VERSION}`);
console.log('✅ Single source of truth for all visualization environments');