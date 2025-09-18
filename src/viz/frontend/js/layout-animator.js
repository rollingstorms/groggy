/**
 * 🎬 Layout Animation System
 * Part of Groggy Phase 11: Layout Algorithms Enhancement
 * 
 * Provides smooth transitions between different layout algorithms with:
 * - Interpolated position transitions
 * - Easing functions and timing control
 * - Animation states and events
 * - Performance optimization for large graphs
 * - Concurrent animation management
 * - Layout preview and comparison modes
 */

class LayoutAnimator {
    constructor(graphRenderer, config = {}) {
        this.graphRenderer = graphRenderer;
        this.config = {
            // Animation timing
            defaultDuration: 1500, // ms
            defaultEasing: 'easeInOutCubic',
            
            // Performance thresholds
            maxNodesForFullAnimation: 1000,
            simplifiedAnimationThreshold: 2000,
            
            // Animation quality
            frameRate: 60,
            interpolationSteps: 60,
            
            // Visual effects
            enableGhostNodes: true,
            enableTrails: false,
            enableMorphing: true,
            
            // Layout comparison
            enablePreview: true,
            previewOpacity: 0.3,
            
            // Performance optimization
            useRequestAnimationFrame: true,
            enableGpuAcceleration: true,
            batchUpdates: true,
            
            ...config
        };
        
        // Animation state
        this.isAnimating = false;
        this.currentAnimation = null;
        this.animationQueue = [];
        this.frameId = null;
        this.startTime = 0;
        
        // Layout data
        this.sourceLayout = new Map(); // nodeId -> {x, y}
        this.targetLayout = new Map();
        this.currentPositions = new Map();
        this.animationProgress = 0;
        
        // Performance tracking
        this.performanceMetrics = {
            frameCount: 0,
            droppedFrames: 0,
            averageFrameTime: 0,
            lastFrameTime: 0
        };
        
        // Event handlers
        this.eventHandlers = new Map();
        
        // Animation effects
        this.ghostNodes = new Map();
        this.trails = new Map();
        
        console.log('🎬 LayoutAnimator initialized with config:', this.config);
    }
    
    /**
     * Animate transition from current layout to new layout
     */
    animateToLayout(newLayout, options = {}) {
        const animationOptions = {
            duration: options.duration || this.config.defaultDuration,
            easing: options.easing || this.config.defaultEasing,
            stagger: options.stagger || 0, // Delay between node animations
            onUpdate: options.onUpdate || null,
            onComplete: options.onComplete || null,
            onCancel: options.onCancel || null,
            priority: options.priority || 'normal',
            allowConcurrent: options.allowConcurrent || false,
            ...options
        };
        
        // Validate layout data
        if (!newLayout || typeof newLayout !== 'object') {
            console.error('Invalid layout data provided to animateToLayout');
            return Promise.reject(new Error('Invalid layout data'));
        }
        
        // Handle concurrent animations
        if (this.isAnimating && !animationOptions.allowConcurrent) {
            if (animationOptions.priority === 'high') {
                this.cancelCurrentAnimation();
            } else {
                return this.queueAnimation(newLayout, animationOptions);
            }
        }
        
        return this.startLayoutAnimation(newLayout, animationOptions);
    }
    
    /**
     * Start the main layout animation
     */
    startLayoutAnimation(newLayout, options) {
        return new Promise((resolve, reject) => {
            try {
                // Prepare animation data
                this.prepareAnimationData(newLayout, options);
                
                // Setup animation state
                this.isAnimating = true;
                this.currentAnimation = {
                    options,
                    resolve,
                    reject,
                    startTime: performance.now(),
                    id: Math.random().toString(36).substr(2, 9)
                };
                
                // Emit start event
                this.emit('animationStart', {
                    animationId: this.currentAnimation.id,
                    sourceLayout: this.sourceLayout,
                    targetLayout: this.targetLayout,
                    options
                });
                
                // Start animation loop
                this.startAnimationLoop();
                
                console.log(`🎬 Started layout animation (${this.currentAnimation.id}):`, options);
                
            } catch (error) {
                console.error('Failed to start layout animation:', error);
                reject(error);
            }
        });
    }
    
    /**
     * Prepare animation data from current and target layouts
     */
    prepareAnimationData(newLayout, options) {
        // Get current node positions
        this.sourceLayout.clear();
        this.targetLayout.clear();
        this.currentPositions.clear();
        
        const currentNodes = this.graphRenderer.getNodes();
        
        // Collect source positions
        currentNodes.forEach(node => {
            const position = this.graphRenderer.getNodePosition(node.id);
            this.sourceLayout.set(node.id, { x: position.x, y: position.y });
            this.currentPositions.set(node.id, { x: position.x, y: position.y });
        });
        
        // Collect target positions
        for (const [nodeId, position] of Object.entries(newLayout)) {
            this.targetLayout.set(nodeId, { x: position.x, y: position.y });
            
            // Initialize current position if node is new
            if (!this.currentPositions.has(nodeId)) {
                this.currentPositions.set(nodeId, { x: position.x, y: position.y });
            }
        }
        
        // Handle nodes that exist in source but not target (fade out)
        this.sourceLayout.forEach((position, nodeId) => {
            if (!this.targetLayout.has(nodeId)) {
                // Keep current position for fade-out animation
                this.targetLayout.set(nodeId, { x: position.x, y: position.y });
            }
        });
        
        // Optimize for large graphs
        this.optimizeForPerformance(options);
        
        // Setup ghost nodes if enabled
        if (this.config.enableGhostNodes && options.showPreview) {
            this.setupGhostNodes();
        }
        
        // Reset animation progress
        this.animationProgress = 0;
        this.performanceMetrics.frameCount = 0;
        this.performanceMetrics.droppedFrames = 0;
    }
    
    /**
     * Optimize animation based on graph size and performance
     */
    optimizeForPerformance(options) {
        const nodeCount = this.sourceLayout.size;
        
        if (nodeCount > this.config.simplifiedAnimationThreshold) {
            // Very large graphs - simplified animation
            options.duration = Math.min(options.duration, 800);
            options.easing = 'linear';
            this.config.enableTrails = false;
            this.config.enableMorphing = false;
            
            console.log(`🎬 Using simplified animation for ${nodeCount} nodes`);
            
        } else if (nodeCount > this.config.maxNodesForFullAnimation) {
            // Medium graphs - reduced effects
            options.duration = Math.min(options.duration, 1200);
            this.config.enableTrails = false;
            
            console.log(`🎬 Using optimized animation for ${nodeCount} nodes`);
        }
        
        // Adjust frame rate based on performance
        const targetFrameTime = 1000 / this.config.frameRate;
        if (this.performanceMetrics.averageFrameTime > targetFrameTime * 1.5) {
            this.config.frameRate = Math.max(30, this.config.frameRate * 0.8);
            console.log(`🎬 Reduced frame rate to ${this.config.frameRate} due to performance`);
        }
    }
    
    /**
     * Setup ghost nodes for layout preview
     */
    setupGhostNodes() {
        this.ghostNodes.clear();
        
        this.targetLayout.forEach((position, nodeId) => {
            const sourcePos = this.sourceLayout.get(nodeId);
            if (sourcePos) {
                const ghostNode = this.createGhostNode(nodeId, position);
                this.ghostNodes.set(nodeId, ghostNode);
            }
        });
    }
    
    /**
     * Create a ghost node for preview
     */
    createGhostNode(nodeId, position) {
        const originalNode = this.graphRenderer.getNode(nodeId);
        if (!originalNode) return null;
        
        const ghostNode = originalNode.cloneNode(true);
        ghostNode.style.opacity = this.config.previewOpacity;
        ghostNode.style.pointerEvents = 'none';
        ghostNode.style.filter = 'brightness(0.7)';
        ghostNode.classList.add('ghost-node');
        
        // Position at target location
        this.graphRenderer.setNodePosition(ghostNode, position);
        
        // Add to DOM
        originalNode.parentNode.appendChild(ghostNode);
        
        return ghostNode;
    }
    
    /**
     * Start the animation loop
     */
    startAnimationLoop() {
        const animate = (timestamp) => {
            if (!this.isAnimating || !this.currentAnimation) {
                return;
            }
            
            const frameStartTime = performance.now();
            const elapsed = timestamp - this.currentAnimation.startTime;
            const duration = this.currentAnimation.options.duration;
            
            // Calculate progress
            const rawProgress = Math.min(elapsed / duration, 1);
            this.animationProgress = this.applyEasing(rawProgress, this.currentAnimation.options.easing);
            
            // Update positions
            this.updateNodePositions();
            
            // Update performance metrics
            this.updatePerformanceMetrics(frameStartTime);
            
            // Call update callback
            if (this.currentAnimation.options.onUpdate) {
                this.currentAnimation.options.onUpdate({
                    progress: this.animationProgress,
                    elapsed,
                    remaining: duration - elapsed
                });
            }
            
            // Emit progress event
            this.emit('animationProgress', {
                animationId: this.currentAnimation.id,
                progress: this.animationProgress,
                elapsed,
                remaining: duration - elapsed
            });
            
            // Check if animation is complete
            if (rawProgress >= 1) {
                this.completeAnimation();
            } else {
                // Continue animation
                this.frameId = requestAnimationFrame(animate);
            }
        };
        
        this.frameId = requestAnimationFrame(animate);
    }
    
    /**
     * Apply easing function to progress
     */
    applyEasing(progress, easingType) {
        switch (easingType) {
            case 'linear':
                return progress;
                
            case 'easeInQuad':
                return progress * progress;
                
            case 'easeOutQuad':
                return progress * (2 - progress);
                
            case 'easeInOutQuad':
                return progress < 0.5 
                    ? 2 * progress * progress 
                    : -1 + (4 - 2 * progress) * progress;
                    
            case 'easeInCubic':
                return progress * progress * progress;
                
            case 'easeOutCubic':
                return 1 + (--progress) * progress * progress;
                
            case 'easeInOutCubic':
                return progress < 0.5 
                    ? 4 * progress * progress * progress 
                    : 1 + (--progress) * (2 * (--progress)) * (2 * progress);
                    
            case 'easeInQuart':
                return progress * progress * progress * progress;
                
            case 'easeOutQuart':
                return 1 - (--progress) * progress * progress * progress;
                
            case 'easeInOutQuart':
                return progress < 0.5 
                    ? 8 * progress * progress * progress * progress 
                    : 1 - 8 * (--progress) * progress * progress * progress;
                    
            case 'easeInElastic':
                const c4 = (2 * Math.PI) / 3;
                return progress === 0 ? 0 : progress === 1 ? 1 : 
                    -Math.pow(2, 10 * progress - 10) * Math.sin((progress * 10 - 10.75) * c4);
                    
            case 'easeOutElastic':
                const c4_out = (2 * Math.PI) / 3;
                return progress === 0 ? 0 : progress === 1 ? 1 :
                    Math.pow(2, -10 * progress) * Math.sin((progress * 10 - 0.75) * c4_out) + 1;
                    
            case 'easeInBounce':
                return 1 - this.applyEasing(1 - progress, 'easeOutBounce');
                
            case 'easeOutBounce':
                const n1 = 7.5625;
                const d1 = 2.75;
                if (progress < 1 / d1) {
                    return n1 * progress * progress;
                } else if (progress < 2 / d1) {
                    return n1 * (progress -= 1.5 / d1) * progress + 0.75;
                } else if (progress < 2.5 / d1) {
                    return n1 * (progress -= 2.25 / d1) * progress + 0.9375;
                } else {
                    return n1 * (progress -= 2.625 / d1) * progress + 0.984375;
                }
                
            case 'spring':
                return 1 - Math.cos(progress * Math.PI * 2) * Math.exp(-progress * 6);
                
            default:
                // Default to easeInOutCubic
                return this.applyEasing(progress, 'easeInOutCubic');
        }
    }
    
    /**
     * Update node positions during animation
     */
    updateNodePositions() {
        const stagger = this.currentAnimation.options.stagger || 0;
        const nodeCount = this.sourceLayout.size;
        let updatedCount = 0;
        
        this.sourceLayout.forEach((sourcePos, nodeId) => {
            const targetPos = this.targetLayout.get(nodeId);
            if (!targetPos) return;
            
            // Calculate staggered progress
            let nodeProgress = this.animationProgress;
            if (stagger > 0) {
                const nodeIndex = Array.from(this.sourceLayout.keys()).indexOf(nodeId);
                const delay = (nodeIndex / nodeCount) * stagger;
                const adjustedProgress = Math.max(0, this.animationProgress - delay);
                nodeProgress = Math.min(1, adjustedProgress * (1 + stagger));
            }
            
            // Interpolate position
            const currentPos = {
                x: sourcePos.x + (targetPos.x - sourcePos.x) * nodeProgress,
                y: sourcePos.y + (targetPos.y - sourcePos.y) * nodeProgress
            };
            
            // Update current positions
            this.currentPositions.set(nodeId, currentPos);
            
            // Apply to actual node
            this.graphRenderer.setNodePosition(nodeId, currentPos);
            
            // Update trails if enabled
            if (this.config.enableTrails) {
                this.updateNodeTrail(nodeId, currentPos);
            }
            
            updatedCount++;
        });
        
        // Update edges to follow nodes
        this.graphRenderer.updateEdgePositions();
        
        // Batch DOM updates if enabled
        if (this.config.batchUpdates && updatedCount > 100) {
            this.graphRenderer.flushBatchedUpdates();
        }
        
        // Update ghost nodes opacity
        if (this.config.enableGhostNodes) {
            this.updateGhostNodes();
        }
    }
    
    /**
     * Update ghost nodes during animation
     */
    updateGhostNodes() {
        const reverseProgress = 1 - this.animationProgress;
        const opacity = this.config.previewOpacity * reverseProgress;
        
        this.ghostNodes.forEach((ghostNode, nodeId) => {
            if (ghostNode) {
                ghostNode.style.opacity = opacity;
                
                if (opacity <= 0.01) {
                    ghostNode.remove();
                    this.ghostNodes.delete(nodeId);
                }
            }
        });
    }
    
    /**
     * Update node trail effect
     */
    updateNodeTrail(nodeId, currentPos) {
        if (!this.trails.has(nodeId)) {
            this.trails.set(nodeId, []);
        }
        
        const trail = this.trails.get(nodeId);
        trail.push({ ...currentPos, timestamp: performance.now() });
        
        // Limit trail length
        const maxTrailLength = 20;
        if (trail.length > maxTrailLength) {
            trail.shift();
        }
        
        // Render trail
        this.renderNodeTrail(nodeId, trail);
    }
    
    /**
     * Render visual trail for a node
     */
    renderNodeTrail(nodeId, trail) {
        // This would integrate with the graph renderer's trail system
        // For now, just log the trail data
        if (trail.length > 5) {
            console.debug(`🎬 Node ${nodeId} trail:`, trail.slice(-3));
        }
    }
    
    /**
     * Update performance metrics
     */
    updatePerformanceMetrics(frameStartTime) {
        const frameEndTime = performance.now();
        const frameTime = frameEndTime - frameStartTime;
        
        this.performanceMetrics.frameCount++;
        this.performanceMetrics.lastFrameTime = frameTime;
        
        // Calculate rolling average
        const alpha = 0.1; // Smoothing factor
        this.performanceMetrics.averageFrameTime = 
            this.performanceMetrics.averageFrameTime * (1 - alpha) + frameTime * alpha;
        
        // Detect dropped frames
        const targetFrameTime = 1000 / this.config.frameRate;
        if (frameTime > targetFrameTime * 1.5) {
            this.performanceMetrics.droppedFrames++;
        }
        
        // Emit performance warning if needed
        if (this.performanceMetrics.frameCount % 60 === 0) { // Every 60 frames
            const dropRate = this.performanceMetrics.droppedFrames / this.performanceMetrics.frameCount;
            if (dropRate > 0.1) { // More than 10% dropped frames
                this.emit('performanceWarning', {
                    frameTime: this.performanceMetrics.averageFrameTime,
                    dropRate,
                    recommendation: 'Consider reducing animation quality'
                });
            }
        }
    }
    
    /**
     * Complete the current animation
     */
    completeAnimation() {
        if (!this.currentAnimation) return;
        
        // Ensure final positions are exact
        this.targetLayout.forEach((targetPos, nodeId) => {
            this.graphRenderer.setNodePosition(nodeId, targetPos);
        });
        
        // Clean up ghost nodes
        this.ghostNodes.forEach(ghostNode => {
            if (ghostNode) ghostNode.remove();
        });
        this.ghostNodes.clear();
        
        // Clean up trails
        this.trails.clear();
        
        // Emit completion event
        this.emit('animationComplete', {
            animationId: this.currentAnimation.id,
            duration: performance.now() - this.currentAnimation.startTime,
            performanceMetrics: { ...this.performanceMetrics }
        });
        
        // Call completion callback
        if (this.currentAnimation.options.onComplete) {
            this.currentAnimation.options.onComplete({
                duration: performance.now() - this.currentAnimation.startTime,
                performanceMetrics: { ...this.performanceMetrics }
            });
        }
        
        // Resolve promise
        this.currentAnimation.resolve({
            completed: true,
            duration: performance.now() - this.currentAnimation.startTime
        });
        
        console.log(`🎬 Animation completed (${this.currentAnimation.id})`);
        
        // Reset animation state
        this.isAnimating = false;
        this.currentAnimation = null;
        
        // Process queued animations
        this.processAnimationQueue();
    }
    
    /**
     * Cancel the current animation
     */
    cancelCurrentAnimation() {
        if (!this.currentAnimation) return;
        
        // Cancel animation frame
        if (this.frameId) {
            cancelAnimationFrame(this.frameId);
            this.frameId = null;
        }
        
        // Clean up
        this.ghostNodes.forEach(ghostNode => {
            if (ghostNode) ghostNode.remove();
        });
        this.ghostNodes.clear();
        this.trails.clear();
        
        // Emit cancellation event
        this.emit('animationCancel', {
            animationId: this.currentAnimation.id,
            progress: this.animationProgress
        });
        
        // Call cancellation callback
        if (this.currentAnimation.options.onCancel) {
            this.currentAnimation.options.onCancel({
                progress: this.animationProgress
            });
        }
        
        // Reject promise
        this.currentAnimation.reject(new Error('Animation cancelled'));
        
        console.log(`🎬 Animation cancelled (${this.currentAnimation.id})`);
        
        // Reset state
        this.isAnimating = false;
        this.currentAnimation = null;
    }
    
    /**
     * Queue animation for later execution
     */
    queueAnimation(layout, options) {
        return new Promise((resolve, reject) => {
            this.animationQueue.push({
                layout,
                options,
                resolve,
                reject,
                timestamp: performance.now()
            });
            
            console.log(`🎬 Animation queued (${this.animationQueue.length} in queue)`);
        });
    }
    
    /**
     * Process queued animations
     */
    processAnimationQueue() {
        if (this.animationQueue.length === 0 || this.isAnimating) {
            return;
        }
        
        const nextAnimation = this.animationQueue.shift();
        console.log(`🎬 Processing queued animation (${this.animationQueue.length} remaining)`);
        
        this.startLayoutAnimation(nextAnimation.layout, nextAnimation.options)
            .then(nextAnimation.resolve)
            .catch(nextAnimation.reject);
    }
    
    /**
     * Preview layout without animating
     */
    previewLayout(layout, duration = 2000) {
        if (!this.config.enablePreview) {
            console.warn('Layout preview is disabled');
            return;
        }
        
        // Setup ghost nodes at target positions
        this.targetLayout.clear();
        for (const [nodeId, position] of Object.entries(layout)) {
            this.targetLayout.set(nodeId, position);
        }
        
        this.setupGhostNodes();
        
        // Auto-hide preview after duration
        setTimeout(() => {
            this.hidePreview();
        }, duration);
        
        console.log('🎬 Layout preview shown');
    }
    
    /**
     * Hide layout preview
     */
    hidePreview() {
        this.ghostNodes.forEach(ghostNode => {
            if (ghostNode) ghostNode.remove();
        });
        this.ghostNodes.clear();
        
        console.log('🎬 Layout preview hidden');
    }
    
    /**
     * Get animation performance metrics
     */
    getPerformanceMetrics() {
        return {
            ...this.performanceMetrics,
            isAnimating: this.isAnimating,
            queueLength: this.animationQueue.length,
            currentFPS: this.performanceMetrics.lastFrameTime > 0 
                ? 1000 / this.performanceMetrics.lastFrameTime 
                : 0
        };
    }
    
    /**
     * Event system
     */
    on(eventType, handler) {
        if (!this.eventHandlers.has(eventType)) {
            this.eventHandlers.set(eventType, []);
        }
        this.eventHandlers.get(eventType).push(handler);
    }
    
    off(eventType, handler) {
        if (this.eventHandlers.has(eventType)) {
            const handlers = this.eventHandlers.get(eventType);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }
    
    emit(eventType, data) {
        if (this.eventHandlers.has(eventType)) {
            this.eventHandlers.get(eventType).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in ${eventType} handler:`, error);
                }
            });
        }
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        console.log('🎬 LayoutAnimator config updated:', this.config);
    }
    
    /**
     * Stop all animations and clean up
     */
    destroy() {
        this.cancelCurrentAnimation();
        this.animationQueue = [];
        this.ghostNodes.forEach(ghostNode => {
            if (ghostNode) ghostNode.remove();
        });
        this.ghostNodes.clear();
        this.trails.clear();
        this.eventHandlers.clear();
        
        console.log('🎬 LayoutAnimator destroyed');
    }
}

/**
 * Layout Transition Presets
 * Common animation configurations for different scenarios
 */
class LayoutTransitionPresets {
    static get SMOOTH() {
        return {
            duration: 1500,
            easing: 'easeInOutCubic',
            enablePreview: true,
            enableTrails: false
        };
    }
    
    static get QUICK() {
        return {
            duration: 800,
            easing: 'easeOutQuad',
            enablePreview: false,
            enableTrails: false
        };
    }
    
    static get DRAMATIC() {
        return {
            duration: 2500,
            easing: 'easeInOutElastic',
            enablePreview: true,
            enableTrails: true,
            stagger: 0.3
        };
    }
    
    static get PRECISE() {
        return {
            duration: 1200,
            easing: 'easeInOutQuart',
            enablePreview: true,
            enableTrails: false
        };
    }
    
    static get BOUNCY() {
        return {
            duration: 2000,
            easing: 'easeOutBounce',
            enablePreview: true,
            enableTrails: true,
            stagger: 0.1
        };
    }
    
    static get SPRING() {
        return {
            duration: 1800,
            easing: 'spring',
            enablePreview: true,
            enableTrails: true,
            stagger: 0.05
        };
    }
    
    static get PERFORMANCE() {
        return {
            duration: 600,
            easing: 'linear',
            enablePreview: false,
            enableTrails: false,
            enableGhostNodes: false
        };
    }
}

/**
 * Layout Comparison Tool
 * Allows side-by-side comparison of different layouts
 */
class LayoutComparator {
    constructor(graphRenderer) {
        this.graphRenderer = graphRenderer;
        this.layouts = new Map();
        this.currentComparison = null;
        this.splitViewEnabled = false;
    }
    
    /**
     * Add a layout for comparison
     */
    addLayout(name, layout, metadata = {}) {
        this.layouts.set(name, {
            layout,
            metadata: {
                algorithm: metadata.algorithm || 'unknown',
                parameters: metadata.parameters || {},
                computeTime: metadata.computeTime || 0,
                quality: metadata.quality || 0,
                ...metadata
            },
            timestamp: Date.now()
        });
        
        console.log(`📊 Added layout "${name}" for comparison`);
    }
    
    /**
     * Compare two layouts side by side
     */
    compare(layout1Name, layout2Name, options = {}) {
        const layout1 = this.layouts.get(layout1Name);
        const layout2 = this.layouts.get(layout2Name);
        
        if (!layout1 || !layout2) {
            throw new Error('One or both layouts not found');
        }
        
        this.currentComparison = {
            layout1: { name: layout1Name, ...layout1 },
            layout2: { name: layout2Name, ...layout2 },
            options,
            startTime: performance.now()
        };
        
        // Enable split view if requested
        if (options.splitView) {
            this.enableSplitView();
        }
        
        console.log(`📊 Comparing layouts: ${layout1Name} vs ${layout2Name}`);
        
        return this.currentComparison;
    }
    
    /**
     * Enable split-screen comparison view
     */
    enableSplitView() {
        // This would modify the renderer to show two views
        this.splitViewEnabled = true;
        console.log('📊 Split view enabled for layout comparison');
    }
    
    /**
     * Disable split view
     */
    disableSplitView() {
        this.splitViewEnabled = false;
        console.log('📊 Split view disabled');
    }
    
    /**
     * Get comparison metrics
     */
    getComparisonMetrics() {
        if (!this.currentComparison) {
            return null;
        }
        
        const { layout1, layout2 } = this.currentComparison;
        
        return {
            algorithms: [layout1.metadata.algorithm, layout2.metadata.algorithm],
            computeTimes: [layout1.metadata.computeTime, layout2.metadata.computeTime],
            qualityScores: [layout1.metadata.quality, layout2.metadata.quality],
            nodeCount: Object.keys(layout1.layout).length,
            comparison: {
                fasterAlgorithm: layout1.metadata.computeTime < layout2.metadata.computeTime 
                    ? layout1.name : layout2.name,
                higherQuality: layout1.metadata.quality > layout2.metadata.quality 
                    ? layout1.name : layout2.name
            }
        };
    }
}

// Global instances
window.GroggyLayoutAnimator = null;
window.GroggyLayoutComparator = null;

/**
 * Initialize layout animation system
 */
function initializeLayoutAnimation(graphRenderer, config = {}) {
    if (window.GroggyLayoutAnimator) {
        console.warn('Layout animator already initialized');
        return window.GroggyLayoutAnimator;
    }
    
    // Create animator
    window.GroggyLayoutAnimator = new LayoutAnimator(graphRenderer, config);
    
    // Create comparator
    window.GroggyLayoutComparator = new LayoutComparator(graphRenderer);
    
    console.log('🎬 Layout animation system initialized');
    
    return {
        animator: window.GroggyLayoutAnimator,
        comparator: window.GroggyLayoutComparator,
        presets: LayoutTransitionPresets
    };
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        LayoutAnimator,
        LayoutTransitionPresets,
        LayoutComparator,
        initializeLayoutAnimation
    };
}