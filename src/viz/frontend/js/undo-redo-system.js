/**
 * Undo/Redo System
 * Provides comprehensive undo/redo functionality using the Command pattern
 * with support for complex operations, batching, and history management.
 */

class UndoRedoSystem {
    constructor(graphRenderer, config = {}) {
        this.graphRenderer = graphRenderer;
        
        // Configuration with intelligent defaults
        this.config = {
            enableUndoRedo: true,
            maxHistorySize: 100,
            enableBatching: true,
            batchTimeout: 500,
            enableCompression: true,
            compressionThreshold: 10,
            
            // Keyboard shortcuts
            shortcuts: {
                undo: ['Ctrl+Z', 'Cmd+Z'],
                redo: ['Ctrl+Y', 'Cmd+Y', 'Ctrl+Shift+Z', 'Cmd+Shift+Z']
            },
            
            // Auto-save integration
            enableAutoSave: false,
            autoSaveInterval: 30000,
            
            // History persistence
            enablePersistence: false,
            persistenceKey: 'groggy_history',
            
            // UI integration
            enableHistoryPanel: false,
            showTooltips: true,
            
            ...config
        };
        
        // History management
        this.history = {
            undoStack: [],
            redoStack: [],
            currentBatch: null,
            batchTimer: null,
            lastOperation: null,
            isExecuting: false
        };
        
        // Command registry
        this.commandRegistry = new Map();
        
        // State tracking
        this.state = {
            canUndo: false,
            canRedo: false,
            isRecording: true,
            historySize: 0,
            currentPosition: 0
        };
        
        // Performance tracking
        this.performance = {
            operationTimes: [],
            compressionRatio: 0,
            memoryUsage: 0
        };
        
        this.initializeUndoRedoSystem();
    }
    
    /**
     * Initialize the undo/redo system
     */
    initializeUndoRedoSystem() {
        this.registerBuiltInCommands();
        this.bindEventListeners();
        this.setupKeyboardShortcuts();
        
        if (this.config.enablePersistence) {
            this.loadPersistedHistory();
        }
        
        if (this.config.enableAutoSave) {
            this.startAutoSave();
        }
    }
    
    /**
     * Register built-in command types
     */
    registerBuiltInCommands() {
        // Node operations
        this.registerCommand('addNode', AddNodeCommand);
        this.registerCommand('removeNode', RemoveNodeCommand);
        this.registerCommand('moveNode', MoveNodeCommand);
        this.registerCommand('updateNode', UpdateNodeCommand);
        
        // Edge operations
        this.registerCommand('addEdge', AddEdgeCommand);
        this.registerCommand('removeEdge', RemoveEdgeCommand);
        this.registerCommand('updateEdge', UpdateEdgeCommand);
        
        // Selection operations
        this.registerCommand('select', SelectCommand);
        this.registerCommand('deselect', DeselectCommand);
        
        // Batch operations
        this.registerCommand('batch', BatchCommand);
        
        // Layout operations
        this.registerCommand('applyLayout', ApplyLayoutCommand);
        
        // Property operations
        this.registerCommand('updateProperty', UpdatePropertyCommand);
        this.registerCommand('updateProperties', UpdatePropertiesCommand);
        
        // Graph operations
        this.registerCommand('clear', ClearGraphCommand);
        this.registerCommand('import', ImportGraphCommand);
    }
    
    /**
     * Bind event listeners
     */
    bindEventListeners() {
        // Listen to graph events
        this.graphRenderer.canvas.addEventListener('nodeAdd', (e) => {
            this.recordCommand('addNode', e.detail);
        });
        
        this.graphRenderer.canvas.addEventListener('nodeRemove', (e) => {
            this.recordCommand('removeNode', e.detail);
        });
        
        this.graphRenderer.canvas.addEventListener('nodeUpdate', (e) => {
            this.recordCommand('updateNode', e.detail);
        });
        
        this.graphRenderer.canvas.addEventListener('nodeDragEnd', (e) => {
            this.recordCommand('moveNode', e.detail);
        });
        
        this.graphRenderer.canvas.addEventListener('edgeAdd', (e) => {
            this.recordCommand('addEdge', e.detail);
        });
        
        this.graphRenderer.canvas.addEventListener('edgeRemove', (e) => {
            this.recordCommand('removeEdge', e.detail);
        });
        
        this.graphRenderer.canvas.addEventListener('edgeUpdate', (e) => {
            this.recordCommand('updateEdge', e.detail);
        });
        
        this.graphRenderer.canvas.addEventListener('selectionChanged', (e) => {
            this.recordSelectionChange(e.detail);
        });
        
        this.graphRenderer.canvas.addEventListener('elementEditElementChanged', (e) => {
            this.recordPropertyChange(e.detail);
        });
        
        this.graphRenderer.canvas.addEventListener('layoutApply', (e) => {
            this.recordLayoutChange(e.detail);
        });
    }
    
    /**
     * Setup keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (event) => {
            if (!this.config.enableUndoRedo) return;
            
            const keyCombo = this.getKeyCombo(event);
            
            if (this.config.shortcuts.undo.includes(keyCombo)) {
                this.undo();
                event.preventDefault();
            } else if (this.config.shortcuts.redo.includes(keyCombo)) {
                this.redo();
                event.preventDefault();
            }
        });
    }
    
    /**
     * Core undo/redo methods
     */
    undo() {
        if (!this.canUndo()) return false;
        
        this.finalizeBatch(); // Complete any ongoing batch
        
        const command = this.history.undoStack.pop();
        if (!command) return false;
        
        this.history.isExecuting = true;
        
        try {
            const startTime = performance.now();
            
            // Execute undo
            const result = command.undo();
            
            // Track performance
            this.performance.operationTimes.push(performance.now() - startTime);
            
            // Move to redo stack
            this.history.redoStack.push(command);
            
            // Update state
            this.updateState();
            
            // Emit events
            this.emitEvent('undo', {
                command: command,
                result: result,
                canUndo: this.canUndo(),
                canRedo: this.canRedo()
            });
            
            // Request redraw
            this.graphRenderer.requestRedraw();
            
            return true;
            
        } catch (error) {
            console.error('Error during undo operation:', error);
            this.emitEvent('undoError', { error, command });
            return false;
        } finally {
            this.history.isExecuting = false;
        }
    }
    
    redo() {
        if (!this.canRedo()) return false;
        
        const command = this.history.redoStack.pop();
        if (!command) return false;
        
        this.history.isExecuting = true;
        
        try {
            const startTime = performance.now();
            
            // Execute redo
            const result = command.execute();
            
            // Track performance
            this.performance.operationTimes.push(performance.now() - startTime);
            
            // Move back to undo stack
            this.history.undoStack.push(command);
            
            // Update state
            this.updateState();
            
            // Emit events
            this.emitEvent('redo', {
                command: command,
                result: result,
                canUndo: this.canUndo(),
                canRedo: this.canRedo()
            });
            
            // Request redraw
            this.graphRenderer.requestRedraw();
            
            return true;
            
        } catch (error) {
            console.error('Error during redo operation:', error);
            this.emitEvent('redoError', { error, command });
            return false;
        } finally {
            this.history.isExecuting = false;
        }
    }
    
    /**
     * Command recording and execution
     */
    recordCommand(commandType, data) {
        if (!this.config.enableUndoRedo || this.history.isExecuting || !this.state.isRecording) {
            return;
        }
        
        const CommandClass = this.commandRegistry.get(commandType);
        if (!CommandClass) {
            console.warn(`Unknown command type: ${commandType}`);
            return;
        }
        
        try {
            const command = new CommandClass(this.graphRenderer, data);
            
            // Check if this should be batched with previous command
            if (this.config.enableBatching && this.shouldBatch(command)) {
                this.addToBatch(command);
            } else {
                this.finalizeBatch(); // Complete any ongoing batch
                this.addCommand(command);
            }
            
        } catch (error) {
            console.error('Error creating command:', error);
            this.emitEvent('commandError', { error, commandType, data });
        }
    }
    
    addCommand(command) {
        // Clear redo stack when new command is added
        this.history.redoStack = [];
        
        // Add to undo stack
        this.history.undoStack.push(command);
        
        // Enforce history size limit
        this.enforceHistoryLimit();
        
        // Update state
        this.updateState();
        
        // Store last operation for batching
        this.history.lastOperation = {
            command: command,
            timestamp: Date.now()
        };
        
        // Emit event
        this.emitEvent('commandAdded', {
            command: command,
            historySize: this.history.undoStack.length
        });
        
        // Auto-save if enabled
        if (this.config.enablePersistence) {
            this.persistHistory();
        }
    }
    
    /**
     * Batching system
     */
    shouldBatch(command) {
        if (!this.history.lastOperation) return false;
        
        const timeDiff = Date.now() - this.history.lastOperation.timestamp;
        if (timeDiff > this.config.batchTimeout) return false;
        
        const lastCommand = this.history.lastOperation.command;
        
        // Check if commands can be batched
        return command.canBatchWith && command.canBatchWith(lastCommand);
    }
    
    addToBatch(command) {
        if (!this.history.currentBatch) {
            // Start new batch with the last command
            const lastCommand = this.history.undoStack.pop();
            this.history.currentBatch = new BatchCommand(this.graphRenderer, {
                commands: [lastCommand]
            });
        }
        
        // Add command to current batch
        this.history.currentBatch.addCommand(command);
        
        // Reset batch timer
        this.resetBatchTimer();
    }
    
    finalizeBatch() {
        if (this.history.currentBatch) {
            // Add the batch as a single command
            this.history.undoStack.push(this.history.currentBatch);
            this.history.currentBatch = null;
            
            this.updateState();
        }
        
        this.clearBatchTimer();
    }
    
    resetBatchTimer() {
        this.clearBatchTimer();
        
        this.history.batchTimer = setTimeout(() => {
            this.finalizeBatch();
        }, this.config.batchTimeout);
    }
    
    clearBatchTimer() {
        if (this.history.batchTimer) {
            clearTimeout(this.history.batchTimer);
            this.history.batchTimer = null;
        }
    }
    
    /**
     * History management
     */
    enforceHistoryLimit() {
        while (this.history.undoStack.length > this.config.maxHistorySize) {
            this.history.undoStack.shift();
        }
        
        // Compress history if enabled
        if (this.config.enableCompression && 
            this.history.undoStack.length >= this.config.compressionThreshold) {
            this.compressHistory();
        }
    }
    
    compressHistory() {
        // Compress older commands to save memory
        const oldSize = this.calculateHistorySize();
        
        // Implement compression logic (e.g., merge similar commands)
        const compressedCommands = this.mergeCompatibleCommands(
            this.history.undoStack.slice(0, Math.floor(this.history.undoStack.length / 2))
        );
        
        // Replace compressed portion
        this.history.undoStack = [
            ...compressedCommands,
            ...this.history.undoStack.slice(Math.floor(this.history.undoStack.length / 2))
        ];
        
        const newSize = this.calculateHistorySize();
        this.performance.compressionRatio = (oldSize - newSize) / oldSize;
        
        this.emitEvent('historyCompressed', {
            oldSize: oldSize,
            newSize: newSize,
            compressionRatio: this.performance.compressionRatio
        });
    }
    
    mergeCompatibleCommands(commands) {
        // Group commands by type and element
        const groups = new Map();
        
        commands.forEach(command => {
            const key = `${command.type}_${command.elementId || 'global'}`;
            if (!groups.has(key)) {
                groups.set(key, []);
            }
            groups.get(key).push(command);
        });
        
        const mergedCommands = [];
        
        groups.forEach((groupCommands, key) => {
            if (groupCommands.length > 1 && groupCommands[0].canMerge) {
                // Merge commands in this group
                const merged = groupCommands[0].merge(groupCommands.slice(1));
                mergedCommands.push(merged);
            } else {
                // Keep commands separate
                mergedCommands.push(...groupCommands);
            }
        });
        
        return mergedCommands;
    }
    
    clearHistory() {
        this.finalizeBatch();
        this.history.undoStack = [];
        this.history.redoStack = [];
        this.updateState();
        
        this.emitEvent('historyCleared', {});
    }
    
    /**
     * State management
     */
    updateState() {
        this.state.canUndo = this.history.undoStack.length > 0;
        this.state.canRedo = this.history.redoStack.length > 0;
        this.state.historySize = this.history.undoStack.length;
        this.state.currentPosition = this.history.undoStack.length;
        
        this.emitEvent('stateChanged', { ...this.state });
    }
    
    canUndo() {
        return this.state.canUndo && this.config.enableUndoRedo;
    }
    
    canRedo() {
        return this.state.canRedo && this.config.enableUndoRedo;
    }
    
    /**
     * Command registration
     */
    registerCommand(commandType, CommandClass) {
        this.commandRegistry.set(commandType, CommandClass);
    }
    
    unregisterCommand(commandType) {
        this.commandRegistry.delete(commandType);
    }
    
    /**
     * Event recording helpers
     */
    recordSelectionChange(detail) {
        if (detail.mode === 'clear') {
            this.recordCommand('deselect', { elements: detail.previousSelection });
        } else {
            this.recordCommand('select', {
                elements: detail.selectedElements,
                mode: detail.mode
            });
        }
    }
    
    recordPropertyChange(detail) {
        this.recordCommand('updateProperty', {
            element: detail.element,
            property: detail.property,
            oldValue: detail.oldValue,
            newValue: detail.newValue
        });
    }
    
    recordLayoutChange(detail) {
        // Capture positions before layout change
        const oldPositions = this.captureNodePositions();
        
        // Record layout command with position data
        setTimeout(() => {
            const newPositions = this.captureNodePositions();
            this.recordCommand('applyLayout', {
                layoutType: detail.layoutType,
                oldPositions: oldPositions,
                newPositions: newPositions
            });
        }, 100); // Small delay to ensure layout is applied
    }
    
    captureNodePositions() {
        const positions = new Map();
        const nodes = this.graphRenderer.getNodes();
        
        nodes.forEach(node => {
            positions.set(node.id, { x: node.x, y: node.y });
        });
        
        return positions;
    }
    
    /**
     * Persistence
     */
    persistHistory() {
        if (!this.config.enablePersistence) return;
        
        try {
            const historyData = {
                undoStack: this.serializeCommands(this.history.undoStack),
                timestamp: Date.now(),
                version: '1.0'
            };
            
            localStorage.setItem(this.config.persistenceKey, JSON.stringify(historyData));
        } catch (error) {
            console.error('Error persisting history:', error);
        }
    }
    
    loadPersistedHistory() {
        try {
            const data = localStorage.getItem(this.config.persistenceKey);
            if (!data) return;
            
            const historyData = JSON.parse(data);
            
            // Validate version compatibility
            if (historyData.version !== '1.0') {
                console.warn('Incompatible history version, clearing persisted data');
                localStorage.removeItem(this.config.persistenceKey);
                return;
            }
            
            // Deserialize commands
            this.history.undoStack = this.deserializeCommands(historyData.undoStack);
            this.updateState();
            
            this.emitEvent('historyLoaded', {
                commandCount: this.history.undoStack.length,
                timestamp: historyData.timestamp
            });
            
        } catch (error) {
            console.error('Error loading persisted history:', error);
            localStorage.removeItem(this.config.persistenceKey);
        }
    }
    
    serializeCommands(commands) {
        return commands.map(command => ({
            type: command.constructor.name,
            data: command.serialize ? command.serialize() : command.data,
            timestamp: command.timestamp
        }));
    }
    
    deserializeCommands(serializedCommands) {
        return serializedCommands.map(serialized => {
            const CommandClass = this.commandRegistry.get(serialized.type);
            if (!CommandClass) {
                console.warn(`Unknown command type during deserialization: ${serialized.type}`);
                return null;
            }
            
            try {
                const command = new CommandClass(this.graphRenderer, serialized.data);
                command.timestamp = serialized.timestamp;
                return command;
            } catch (error) {
                console.error('Error deserializing command:', error);
                return null;
            }
        }).filter(command => command !== null);
    }
    
    /**
     * Auto-save functionality
     */
    startAutoSave() {
        this.autoSaveInterval = setInterval(() => {
            this.performAutoSave();
        }, this.config.autoSaveInterval);
    }
    
    stopAutoSave() {
        if (this.autoSaveInterval) {
            clearInterval(this.autoSaveInterval);
            this.autoSaveInterval = null;
        }
    }
    
    performAutoSave() {
        this.emitEvent('autoSave', {
            historySize: this.state.historySize,
            timestamp: Date.now()
        });
    }
    
    /**
     * Utility methods
     */
    getKeyCombo(event) {
        const parts = [];
        
        if (event.ctrlKey) parts.push('Ctrl');
        if (event.metaKey) parts.push('Cmd');
        if (event.shiftKey) parts.push('Shift');
        if (event.altKey) parts.push('Alt');
        
        parts.push(event.key.toUpperCase());
        
        return parts.join('+');
    }
    
    calculateHistorySize() {
        // Estimate memory usage of history
        let size = 0;
        
        this.history.undoStack.forEach(command => {
            size += this.estimateCommandSize(command);
        });
        
        this.history.redoStack.forEach(command => {
            size += this.estimateCommandSize(command);
        });
        
        return size;
    }
    
    estimateCommandSize(command) {
        // Rough estimation of command memory usage
        if (command.estimateSize) {
            return command.estimateSize();
        }
        
        return JSON.stringify(command.data || {}).length;
    }
    
    /**
     * Public API methods
     */
    
    /**
     * Execute a command immediately
     */
    executeCommand(commandType, data) {
        const CommandClass = this.commandRegistry.get(commandType);
        if (!CommandClass) {
            throw new Error(`Unknown command type: ${commandType}`);
        }
        
        const command = new CommandClass(this.graphRenderer, data);
        
        try {
            const result = command.execute();
            this.addCommand(command);
            this.graphRenderer.requestRedraw();
            return result;
        } catch (error) {
            console.error('Error executing command:', error);
            throw error;
        }
    }
    
    /**
     * Start/stop recording
     */
    startRecording() {
        this.state.isRecording = true;
    }
    
    stopRecording() {
        this.state.isRecording = false;
        this.finalizeBatch();
    }
    
    /**
     * Get history information
     */
    getHistory() {
        return {
            undoStack: this.history.undoStack.map(cmd => ({
                type: cmd.constructor.name,
                timestamp: cmd.timestamp,
                description: cmd.getDescription ? cmd.getDescription() : 'Unknown operation'
            })),
            redoStack: this.history.redoStack.map(cmd => ({
                type: cmd.constructor.name,
                timestamp: cmd.timestamp,
                description: cmd.getDescription ? cmd.getDescription() : 'Unknown operation'
            })),
            state: { ...this.state }
        };
    }
    
    /**
     * Performance metrics
     */
    getPerformanceMetrics() {
        const times = this.performance.operationTimes;
        return {
            averageOperationTime: times.length > 0 ? times.reduce((a, b) => a + b) / times.length : 0,
            totalOperations: times.length,
            compressionRatio: this.performance.compressionRatio,
            memoryUsage: this.calculateHistorySize(),
            historySize: this.state.historySize
        };
    }
    
    /**
     * Configuration updates
     */
    updateConfig(newConfig) {
        Object.assign(this.config, newConfig);
        
        if (!this.config.enableUndoRedo) {
            this.clearHistory();
        }
        
        if (this.config.enableAutoSave && !this.autoSaveInterval) {
            this.startAutoSave();
        } else if (!this.config.enableAutoSave && this.autoSaveInterval) {
            this.stopAutoSave();
        }
    }
    
    /**
     * Event system
     */
    emitEvent(eventType, data) {
        const event = new CustomEvent(`undoRedo${eventType.charAt(0).toUpperCase() + eventType.slice(1)}`, {
            detail: data
        });
        this.graphRenderer.canvas.dispatchEvent(event);
    }
    
    /**
     * Cleanup and destroy
     */
    destroy() {
        this.finalizeBatch();
        this.clearBatchTimer();
        this.stopAutoSave();
        
        // Remove event listeners
        document.removeEventListener('keydown', this.handleKeydown);
        
        // Clear history
        this.clearHistory();
        
        // Clear command registry
        this.commandRegistry.clear();
    }
}

/**
 * Base Command Class
 */
class Command {
    constructor(graphRenderer, data) {
        this.graphRenderer = graphRenderer;
        this.data = data;
        this.timestamp = Date.now();
        this.type = this.constructor.name;
    }
    
    execute() {
        throw new Error('Execute method must be implemented');
    }
    
    undo() {
        throw new Error('Undo method must be implemented');
    }
    
    canBatchWith(otherCommand) {
        return false;
    }
    
    canMerge(otherCommands) {
        return false;
    }
    
    merge(otherCommands) {
        return this;
    }
    
    getDescription() {
        return this.type;
    }
    
    estimateSize() {
        return JSON.stringify(this.data).length;
    }
}

/**
 * Specific Command Implementations
 */

class AddNodeCommand extends Command {
    execute() {
        this.graphRenderer.addNode(this.data.node);
        return this.data.node;
    }
    
    undo() {
        this.graphRenderer.removeNode(this.data.node.id);
        return this.data.node.id;
    }
    
    getDescription() {
        return `Add node "${this.data.node.label || this.data.node.id}"`;
    }
}

class RemoveNodeCommand extends Command {
    constructor(graphRenderer, data) {
        super(graphRenderer, data);
        // Store connected edges for restoration
        this.data.connectedEdges = graphRenderer.getConnectedEdges(data.node.id);
    }
    
    execute() {
        this.graphRenderer.removeNode(this.data.node.id);
        return this.data.node.id;
    }
    
    undo() {
        // Restore node
        this.graphRenderer.addNode(this.data.node);
        
        // Restore connected edges
        this.data.connectedEdges.forEach(edge => {
            this.graphRenderer.addEdge(edge);
        });
        
        return this.data.node;
    }
    
    getDescription() {
        return `Remove node "${this.data.node.label || this.data.node.id}"`;
    }
}

class MoveNodeCommand extends Command {
    constructor(graphRenderer, data) {
        super(graphRenderer, data);
        this.elementId = data.node.id;
    }
    
    execute() {
        const node = this.graphRenderer.getNode(this.data.node.id);
        if (node) {
            node.x = this.data.newPosition.x;
            node.y = this.data.newPosition.y;
        }
        return node;
    }
    
    undo() {
        const node = this.graphRenderer.getNode(this.data.node.id);
        if (node) {
            node.x = this.data.oldPosition.x;
            node.y = this.data.oldPosition.y;
        }
        return node;
    }
    
    canBatchWith(otherCommand) {
        return otherCommand instanceof MoveNodeCommand &&
               otherCommand.data.node.id === this.data.node.id;
    }
    
    getDescription() {
        return `Move node "${this.data.node.label || this.data.node.id}"`;
    }
}

class UpdateNodeCommand extends Command {
    constructor(graphRenderer, data) {
        super(graphRenderer, data);
        this.elementId = data.node.id;
    }
    
    execute() {
        const node = this.graphRenderer.getNode(this.data.node.id);
        if (node) {
            Object.assign(node, this.data.newProperties);
        }
        return node;
    }
    
    undo() {
        const node = this.graphRenderer.getNode(this.data.node.id);
        if (node) {
            Object.assign(node, this.data.oldProperties);
        }
        return node;
    }
    
    getDescription() {
        const props = Object.keys(this.data.newProperties).join(', ');
        return `Update node "${this.data.node.label || this.data.node.id}" (${props})`;
    }
}

class AddEdgeCommand extends Command {
    execute() {
        this.graphRenderer.addEdge(this.data.edge);
        return this.data.edge;
    }
    
    undo() {
        this.graphRenderer.removeEdge(this.data.edge.id);
        return this.data.edge.id;
    }
    
    getDescription() {
        return `Add edge "${this.data.edge.label || this.data.edge.id}"`;
    }
}

class RemoveEdgeCommand extends Command {
    execute() {
        this.graphRenderer.removeEdge(this.data.edge.id);
        return this.data.edge.id;
    }
    
    undo() {
        this.graphRenderer.addEdge(this.data.edge);
        return this.data.edge;
    }
    
    getDescription() {
        return `Remove edge "${this.data.edge.label || this.data.edge.id}"`;
    }
}

class UpdateEdgeCommand extends Command {
    constructor(graphRenderer, data) {
        super(graphRenderer, data);
        this.elementId = data.edge.id;
    }
    
    execute() {
        const edge = this.graphRenderer.getEdge(this.data.edge.id);
        if (edge) {
            Object.assign(edge, this.data.newProperties);
        }
        return edge;
    }
    
    undo() {
        const edge = this.graphRenderer.getEdge(this.data.edge.id);
        if (edge) {
            Object.assign(edge, this.data.oldProperties);
        }
        return edge;
    }
    
    getDescription() {
        const props = Object.keys(this.data.newProperties).join(', ');
        return `Update edge "${this.data.edge.label || this.data.edge.id}" (${props})`;
    }
}

class SelectCommand extends Command {
    execute() {
        if (this.graphRenderer.selectionManager) {
            this.graphRenderer.selectionManager.setSelection(
                this.data.elements.filter(el => el.type === 'node').map(el => el.id),
                this.data.elements.filter(el => el.type === 'edge').map(el => el.id)
            );
        }
        return this.data.elements;
    }
    
    undo() {
        if (this.graphRenderer.selectionManager) {
            this.graphRenderer.selectionManager.clearSelection();
        }
        return [];
    }
    
    getDescription() {
        return `Select ${this.data.elements.length} element(s)`;
    }
}

class DeselectCommand extends Command {
    execute() {
        if (this.graphRenderer.selectionManager) {
            this.graphRenderer.selectionManager.clearSelection();
        }
        return [];
    }
    
    undo() {
        if (this.graphRenderer.selectionManager) {
            this.graphRenderer.selectionManager.setSelection(
                this.data.elements.filter(el => el.type === 'node').map(el => el.id),
                this.data.elements.filter(el => el.type === 'edge').map(el => el.id)
            );
        }
        return this.data.elements;
    }
    
    getDescription() {
        return `Deselect ${this.data.elements.length} element(s)`;
    }
}

class BatchCommand extends Command {
    constructor(graphRenderer, data) {
        super(graphRenderer, data);
        this.commands = data.commands || [];
    }
    
    addCommand(command) {
        this.commands.push(command);
    }
    
    execute() {
        const results = [];
        for (const command of this.commands) {
            results.push(command.execute());
        }
        return results;
    }
    
    undo() {
        const results = [];
        // Undo in reverse order
        for (let i = this.commands.length - 1; i >= 0; i--) {
            results.push(this.commands[i].undo());
        }
        return results;
    }
    
    getDescription() {
        return `Batch operation (${this.commands.length} commands)`;
    }
    
    estimateSize() {
        return this.commands.reduce((total, cmd) => total + cmd.estimateSize(), 0);
    }
}

class ApplyLayoutCommand extends Command {
    execute() {
        // Apply new positions
        this.data.newPositions.forEach((position, nodeId) => {
            const node = this.graphRenderer.getNode(nodeId);
            if (node) {
                node.x = position.x;
                node.y = position.y;
            }
        });
        return this.data.newPositions;
    }
    
    undo() {
        // Restore old positions
        this.data.oldPositions.forEach((position, nodeId) => {
            const node = this.graphRenderer.getNode(nodeId);
            if (node) {
                node.x = position.x;
                node.y = position.y;
            }
        });
        return this.data.oldPositions;
    }
    
    getDescription() {
        return `Apply ${this.data.layoutType} layout`;
    }
}

class UpdatePropertyCommand extends Command {
    constructor(graphRenderer, data) {
        super(graphRenderer, data);
        this.elementId = data.element.id;
    }
    
    execute() {
        this.data.element[this.data.property] = this.data.newValue;
        return this.data.newValue;
    }
    
    undo() {
        this.data.element[this.data.property] = this.data.oldValue;
        return this.data.oldValue;
    }
    
    canBatchWith(otherCommand) {
        return otherCommand instanceof UpdatePropertyCommand &&
               otherCommand.data.element.id === this.data.element.id &&
               otherCommand.data.property === this.data.property;
    }
    
    getDescription() {
        return `Update ${this.data.property} of ${this.data.element.type} "${this.data.element.label || this.data.element.id}"`;
    }
}

class UpdatePropertiesCommand extends Command {
    constructor(graphRenderer, data) {
        super(graphRenderer, data);
        this.elementId = data.element.id;
    }
    
    execute() {
        Object.assign(this.data.element, this.data.newProperties);
        return this.data.newProperties;
    }
    
    undo() {
        Object.assign(this.data.element, this.data.oldProperties);
        return this.data.oldProperties;
    }
    
    getDescription() {
        const props = Object.keys(this.data.newProperties).join(', ');
        return `Update properties (${props}) of ${this.data.element.type} "${this.data.element.label || this.data.element.id}"`;
    }
}

class ClearGraphCommand extends Command {
    constructor(graphRenderer, data) {
        super(graphRenderer, data);
        // Capture current graph state
        this.data.nodes = [...graphRenderer.getNodes()];
        this.data.edges = [...graphRenderer.getEdges()];
    }
    
    execute() {
        this.graphRenderer.clear();
        return { nodes: [], edges: [] };
    }
    
    undo() {
        // Restore all nodes and edges
        this.data.nodes.forEach(node => this.graphRenderer.addNode(node));
        this.data.edges.forEach(edge => this.graphRenderer.addEdge(edge));
        return { nodes: this.data.nodes, edges: this.data.edges };
    }
    
    getDescription() {
        return `Clear graph (${this.data.nodes.length} nodes, ${this.data.edges.length} edges)`;
    }
}

class ImportGraphCommand extends Command {
    constructor(graphRenderer, data) {
        super(graphRenderer, data);
        // Capture current graph state for undo
        this.data.oldNodes = [...graphRenderer.getNodes()];
        this.data.oldEdges = [...graphRenderer.getEdges()];
    }
    
    execute() {
        this.graphRenderer.clear();
        this.data.newNodes.forEach(node => this.graphRenderer.addNode(node));
        this.data.newEdges.forEach(edge => this.graphRenderer.addEdge(edge));
        return { nodes: this.data.newNodes, edges: this.data.newEdges };
    }
    
    undo() {
        this.graphRenderer.clear();
        this.data.oldNodes.forEach(node => this.graphRenderer.addNode(node));
        this.data.oldEdges.forEach(edge => this.graphRenderer.addEdge(edge));
        return { nodes: this.data.oldNodes, edges: this.data.oldEdges };
    }
    
    getDescription() {
        return `Import graph (${this.data.newNodes.length} nodes, ${this.data.newEdges.length} edges)`;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UndoRedoSystem;
}