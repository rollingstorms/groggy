/**
 * 📦 Batch Export System
 * Part of Groggy Phase 12: Static Export System
 * 
 * Comprehensive batch export functionality with:
 * - Multiple format export (SVG, PNG, PDF)
 * - Layout variation exports
 * - Quality/resolution variants
 * - Parallel processing and queue management
 * - Progress tracking and error handling
 * - Export templates and presets
 * - Archive creation (ZIP)
 */

class BatchExportSystem {
    constructor(exportSystem, config = {}) {
        this.exportSystem = exportSystem;
        this.config = {
            // Batch processing
            maxConcurrentExports: 3,
            batchSizeLimit: 50,
            retryAttempts: 3,
            retryDelay: 1000, // ms
            
            // Archive options
            createArchive: true,
            archiveFormat: 'zip',
            archiveCompression: 0.6,
            
            // Performance
            enableProgressTracking: true,
            enableErrorRecovery: true,
            pauseOnError: false,
            
            // File naming
            useTimestamp: true,
            filePrefix: 'groggy-export',
            includeBatchId: true,
            
            ...config
        };
        
        // Batch state
        this.isRunning = false;
        this.currentBatch = null;
        this.exportQueue = [];
        this.activeExports = new Map();
        this.completedExports = [];
        this.failedExports = [];
        
        // Progress tracking
        this.batchProgress = {
            total: 0,
            completed: 0,
            failed: 0,
            current: null,
            startTime: 0,
            estimatedCompletion: 0
        };
        
        // Event handlers
        this.eventHandlers = new Map();
        
        // Archive creation
        this.archiveWorker = null;
        
        console.log('📦 BatchExportSystem initialized with config:', this.config);
        
        this.initializeArchiveWorker();
    }
    
    /**
     * Initialize web worker for archive creation
     */
    initializeArchiveWorker() {
        // In a real implementation, this would load JSZip or similar
        // For now, we'll simulate the archive creation
        this.archiveLibrary = {
            createArchive: (files) => {
                console.log('📦 Creating archive with', files.length, 'files');
                
                // Simulate archive creation
                return new Promise((resolve) => {
                    setTimeout(() => {
                        const archiveBlob = new Blob(['Mock ZIP archive'], { type: 'application/zip' });
                        resolve({
                            blob: archiveBlob,
                            filename: this.generateArchiveFilename(),
                            size: archiveBlob.size,
                            fileCount: files.length
                        });
                    }, 1000);
                });
            }
        };
    }
    
    /**
     * Start batch export with configuration
     */
    async startBatchExport(batchConfig) {
        if (this.isRunning) {
            throw new Error('Batch export already running');
        }
        
        // Validate batch configuration
        this.validateBatchConfig(batchConfig);
        
        // Generate export jobs
        const exportJobs = this.generateExportJobs(batchConfig);
        
        if (exportJobs.length === 0) {
            throw new Error('No export jobs generated');
        }
        
        if (exportJobs.length > this.config.batchSizeLimit) {
            throw new Error(`Batch size limit exceeded (${exportJobs.length} > ${this.config.batchSizeLimit})`);
        }
        
        // Initialize batch
        this.initializeBatch(exportJobs, batchConfig);
        
        console.log(`📦 Starting batch export with ${exportJobs.length} jobs`);
        
        try {
            await this.processBatch();
            return this.completeBatch();
        } catch (error) {
            this.handleBatchError(error);
            throw error;
        }
    }
    
    /**
     * Validate batch configuration
     */
    validateBatchConfig(config) {
        const required = ['formats', 'name'];
        const missing = required.filter(field => !config[field]);
        
        if (missing.length > 0) {
            throw new Error(`Missing required batch config fields: ${missing.join(', ')}`);
        }
        
        if (!Array.isArray(config.formats) || config.formats.length === 0) {
            throw new Error('At least one export format must be specified');
        }
        
        const supportedFormats = ['svg', 'png', 'pdf'];
        const unsupported = config.formats.filter(format => !supportedFormats.includes(format));
        if (unsupported.length > 0) {
            throw new Error(`Unsupported formats: ${unsupported.join(', ')}`);
        }
    }
    
    /**
     * Generate export jobs from batch configuration
     */
    generateExportJobs(batchConfig) {
        const jobs = [];
        const timestamp = Date.now();
        
        // Generate jobs for each format
        batchConfig.formats.forEach((format, formatIndex) => {
            const baseConfig = this.getBaseConfigForFormat(format, batchConfig);
            
            // Generate variants if specified
            if (batchConfig.variants && batchConfig.variants[format]) {
                const variants = batchConfig.variants[format];
                
                variants.forEach((variant, variantIndex) => {
                    const jobConfig = { ...baseConfig, ...variant };
                    const job = this.createExportJob(
                        format, 
                        jobConfig, 
                        formatIndex, 
                        variantIndex, 
                        timestamp
                    );
                    jobs.push(job);
                });
            } else {
                // Single export for this format
                const job = this.createExportJob(format, baseConfig, formatIndex, 0, timestamp);
                jobs.push(job);
            }
        });
        
        // Generate layout variants if specified
        if (batchConfig.layouts && batchConfig.layouts.length > 1) {
            const originalJobs = [...jobs];
            jobs.length = 0; // Clear array
            
            batchConfig.layouts.forEach((layout, layoutIndex) => {
                originalJobs.forEach(job => {
                    const layoutJob = {
                        ...job,
                        id: `${job.id}_layout_${layoutIndex}`,
                        config: {
                            ...job.config,
                            layout,
                            layoutIndex
                        }
                    };
                    jobs.push(layoutJob);
                });
            });
        }
        
        return jobs;
    }
    
    /**
     * Get base configuration for format
     */
    getBaseConfigForFormat(format, batchConfig) {
        const baseConfig = {
            format,
            autoDownload: false,
            filename: this.generateJobFilename(format, batchConfig),
            title: batchConfig.title || `${batchConfig.name} - ${format.toUpperCase()}`,
            description: batchConfig.description || `Batch export generated on ${new Date().toLocaleDateString()}`
        };
        
        // Add format-specific defaults
        switch (format) {
            case 'svg':
                Object.assign(baseConfig, {
                    embedStyles: true,
                    embedFonts: true,
                    optimizeSize: true,
                    includeMetadata: true
                });
                break;
                
            case 'png':
                Object.assign(baseConfig, {
                    dpi: 300,
                    compression: 0.9,
                    antialiasing: true
                });
                break;
                
            case 'pdf':
                Object.assign(baseConfig, {
                    pageFormat: 'A4',
                    orientation: 'landscape',
                    margins: 20,
                    fitToPage: true,
                    imageDPI: 150
                });
                break;
        }
        
        // Override with batch-specific settings
        if (batchConfig.globalSettings) {
            Object.assign(baseConfig, batchConfig.globalSettings);
        }
        
        return baseConfig;
    }
    
    /**
     * Create individual export job
     */
    createExportJob(format, config, formatIndex, variantIndex, timestamp) {
        const jobId = `${timestamp}_${format}_${formatIndex}_${variantIndex}`;
        
        return {
            id: jobId,
            format,
            config,
            status: 'pending',
            attempts: 0,
            createdAt: timestamp,
            startedAt: null,
            completedAt: null,
            result: null,
            error: null,
            progress: 0
        };
    }
    
    /**
     * Initialize batch processing
     */
    initializeBatch(exportJobs, batchConfig) {
        this.currentBatch = {
            id: this.generateBatchId(),
            name: batchConfig.name,
            config: batchConfig,
            jobs: exportJobs,
            createdAt: Date.now(),
            startedAt: null,
            completedAt: null
        };
        
        this.exportQueue = [...exportJobs];
        this.activeExports.clear();
        this.completedExports = [];
        this.failedExports = [];
        
        this.batchProgress = {
            total: exportJobs.length,
            completed: 0,
            failed: 0,
            current: null,
            startTime: Date.now(),
            estimatedCompletion: 0
        };
        
        this.isRunning = true;
        this.currentBatch.startedAt = Date.now();
        
        // Emit batch start event
        this.emit('batchStart', {
            batchId: this.currentBatch.id,
            totalJobs: exportJobs.length,
            config: batchConfig
        });
    }
    
    /**
     * Process the batch queue
     */
    async processBatch() {
        while (this.exportQueue.length > 0 || this.activeExports.size > 0) {
            // Start new exports if we have capacity
            while (this.exportQueue.length > 0 && this.activeExports.size < this.config.maxConcurrentExports) {
                const job = this.exportQueue.shift();
                this.startExportJob(job);
            }
            
            // Wait for any active export to complete
            if (this.activeExports.size > 0) {
                await this.waitForExportCompletion();
            }
            
            // Update progress
            this.updateBatchProgress();
            
            // Check if we should pause on error
            if (this.config.pauseOnError && this.failedExports.length > 0) {
                throw new Error(`Batch paused due to export failures (${this.failedExports.length} failed)`);
            }
        }
    }
    
    /**
     * Start individual export job
     */
    async startExportJob(job) {
        job.status = 'running';
        job.startedAt = Date.now();
        job.attempts++;
        
        this.activeExports.set(job.id, job);
        this.batchProgress.current = job;
        
        console.log(`📦 Starting export job: ${job.id} (${job.format})`);
        
        // Emit job start event
        this.emit('jobStart', {
            batchId: this.currentBatch.id,
            jobId: job.id,
            job
        });
        
        try {
            // Apply layout if specified
            if (job.config.layout && job.config.layoutIndex !== undefined) {
                await this.applyLayoutForJob(job);
            }
            
            // Start the export
            const exportPromise = this.exportSystem.exportGraph(job.format, job.config);
            
            // Handle completion
            exportPromise.then((result) => {
                this.handleJobSuccess(job, result);
            }).catch((error) => {
                this.handleJobError(job, error);
            });
            
        } catch (error) {
            this.handleJobError(job, error);
        }
    }
    
    /**
     * Apply layout for job if needed
     */
    async applyLayoutForJob(job) {
        // This would integrate with the layout system to apply the specified layout
        console.log(`📦 Applying layout ${job.config.layout} for job ${job.id}`);
        
        // Simulate layout application
        return new Promise((resolve) => {
            setTimeout(resolve, 500);
        });
    }
    
    /**
     * Handle successful job completion
     */
    handleJobSuccess(job, result) {
        job.status = 'completed';
        job.completedAt = Date.now();
        job.result = result;
        job.progress = 100;
        
        this.activeExports.delete(job.id);
        this.completedExports.push(job);
        this.batchProgress.completed++;
        
        console.log(`📦 Job completed: ${job.id} (${result.filename})`);
        
        // Emit job completion event
        this.emit('jobComplete', {
            batchId: this.currentBatch.id,
            jobId: job.id,
            job,
            result
        });
    }
    
    /**
     * Handle job error
     */
    async handleJobError(job, error) {
        console.error(`📦 Job failed: ${job.id}`, error);
        
        // Retry if we haven't exceeded retry attempts
        if (job.attempts < this.config.retryAttempts) {
            console.log(`📦 Retrying job: ${job.id} (attempt ${job.attempts + 1}/${this.config.retryAttempts})`);
            
            // Wait before retry
            await new Promise(resolve => setTimeout(resolve, this.config.retryDelay));
            
            // Remove from active exports and re-queue
            this.activeExports.delete(job.id);
            this.exportQueue.unshift(job); // Add to front of queue
            
            return;
        }
        
        // Mark as failed
        job.status = 'failed';
        job.completedAt = Date.now();
        job.error = error.message;
        
        this.activeExports.delete(job.id);
        this.failedExports.push(job);
        this.batchProgress.failed++;
        
        // Emit job error event
        this.emit('jobError', {
            batchId: this.currentBatch.id,
            jobId: job.id,
            job,
            error: error.message
        });
    }
    
    /**
     * Wait for any export to complete
     */
    async waitForExportCompletion() {
        return new Promise((resolve) => {
            const checkCompletion = () => {
                if (this.activeExports.size < this.config.maxConcurrentExports || this.activeExports.size === 0) {
                    resolve();
                } else {
                    setTimeout(checkCompletion, 100);
                }
            };
            checkCompletion();
        });
    }
    
    /**
     * Update batch progress
     */
    updateBatchProgress() {
        const progress = this.batchProgress;
        const elapsed = Date.now() - progress.startTime;
        const completed = progress.completed + progress.failed;
        
        if (completed > 0) {
            const avgTimePerJob = elapsed / completed;
            const remaining = progress.total - completed;
            progress.estimatedCompletion = Date.now() + (remaining * avgTimePerJob);
        }
        
        // Emit progress event
        this.emit('batchProgress', {
            batchId: this.currentBatch.id,
            progress: {
                total: progress.total,
                completed: progress.completed,
                failed: progress.failed,
                percentage: (completed / progress.total) * 100,
                estimatedCompletion: progress.estimatedCompletion,
                currentJob: progress.current
            }
        });
    }
    
    /**
     * Complete batch processing
     */
    async completeBatch() {
        this.isRunning = false;
        this.currentBatch.completedAt = Date.now();
        
        const duration = this.currentBatch.completedAt - this.currentBatch.startedAt;
        const successCount = this.completedExports.length;
        const failureCount = this.failedExports.length;
        
        console.log(`📦 Batch export completed: ${successCount} succeeded, ${failureCount} failed in ${duration}ms`);
        
        // Create archive if enabled and we have successful exports
        let archive = null;
        if (this.config.createArchive && this.completedExports.length > 0) {
            archive = await this.createExportArchive();
        }
        
        const result = {
            batchId: this.currentBatch.id,
            success: true,
            duration,
            totalJobs: this.batchProgress.total,
            successfulJobs: successCount,
            failedJobs: failureCount,
            successRate: (successCount / this.batchProgress.total) * 100,
            exports: this.completedExports.map(job => job.result),
            failures: this.failedExports.map(job => ({
                jobId: job.id,
                format: job.format,
                error: job.error
            })),
            archive
        };
        
        // Emit batch completion event
        this.emit('batchComplete', {
            batchId: this.currentBatch.id,
            result
        });
        
        // Reset state
        this.resetBatchState();
        
        return result;
    }
    
    /**
     * Create archive of exported files
     */
    async createExportArchive() {
        console.log('📦 Creating export archive...');
        
        try {
            const files = this.completedExports.map(job => ({
                name: job.result.filename,
                blob: job.result.blob,
                size: job.result.size
            }));
            
            const archive = await this.archiveLibrary.createArchive(files);
            
            // Auto-download archive if configured
            if (this.config.autoDownload !== false) {
                this.downloadArchive(archive);
            }
            
            return archive;
            
        } catch (error) {
            console.error('Failed to create archive:', error);
            return null;
        }
    }
    
    /**
     * Download archive file
     */
    downloadArchive(archive) {
        const url = URL.createObjectURL(archive.blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = archive.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log(`📦 Archive downloaded: ${archive.filename}`);
    }
    
    /**
     * Handle batch error
     */
    handleBatchError(error) {
        this.isRunning = false;
        
        if (this.currentBatch) {
            this.currentBatch.completedAt = Date.now();
        }
        
        console.error('📦 Batch export failed:', error);
        
        // Emit batch error event
        this.emit('batchError', {
            batchId: this.currentBatch?.id,
            error: error.message,
            completedJobs: this.completedExports.length,
            failedJobs: this.failedExports.length
        });
        
        this.resetBatchState();
    }
    
    /**
     * Reset batch state
     */
    resetBatchState() {
        this.currentBatch = null;
        this.exportQueue = [];
        this.activeExports.clear();
        this.completedExports = [];
        this.failedExports = [];
        this.batchProgress = {
            total: 0,
            completed: 0,
            failed: 0,
            current: null,
            startTime: 0,
            estimatedCompletion: 0
        };
    }
    
    /**
     * Cancel running batch
     */
    cancelBatch() {
        if (!this.isRunning) {
            return;
        }
        
        console.log('📦 Cancelling batch export...');
        
        // Clear queue
        this.exportQueue = [];
        
        // Cancel active exports (would need to be implemented in export system)
        this.activeExports.forEach((job) => {
            job.status = 'cancelled';
            // this.exportSystem.cancelExport(job.id); // If supported
        });
        
        this.handleBatchError(new Error('Batch export cancelled by user'));
    }
    
    /**
     * Generate batch ID
     */
    generateBatchId() {
        return `batch_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
    }
    
    /**
     * Generate archive filename
     */
    generateArchiveFilename() {
        const timestamp = this.config.useTimestamp ? `_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}` : '';
        const batchId = this.config.includeBatchId && this.currentBatch ? `_${this.currentBatch.id}` : '';
        
        return `${this.config.filePrefix}${batchId}${timestamp}.${this.config.archiveFormat}`;
    }
    
    /**
     * Generate job filename
     */
    generateJobFilename(format, batchConfig) {
        const timestamp = this.config.useTimestamp ? `_${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}` : '';
        const batchName = batchConfig.name ? `_${batchConfig.name.replace(/[^a-zA-Z0-9]/g, '_')}` : '';
        
        return `${this.config.filePrefix}${batchName}_${format}${timestamp}`;
    }
    
    /**
     * Get batch status
     */
    getBatchStatus() {
        if (!this.currentBatch) {
            return { status: 'idle' };
        }
        
        return {
            status: this.isRunning ? 'running' : 'completed',
            batchId: this.currentBatch.id,
            progress: this.batchProgress,
            activeJobs: Array.from(this.activeExports.values()),
            completedJobs: this.completedExports.length,
            failedJobs: this.failedExports.length,
            queuedJobs: this.exportQueue.length
        };
    }
    
    /**
     * Get export history
     */
    getExportHistory() {
        return {
            completed: [...this.completedExports],
            failed: [...this.failedExports],
            currentBatch: this.currentBatch
        };
    }
    
    /**
     * Create preset batch configurations
     */
    static createPresets() {
        return {
            'web-package': {
                name: 'Web Package',
                description: 'Optimized exports for web use',
                formats: ['svg', 'png'],
                variants: {
                    svg: [
                        { embedStyles: true, optimizeSize: true },
                    ],
                    png: [
                        { dpi: 96, compression: 0.8 }, // Web quality
                        { dpi: 192, compression: 0.9 }, // Retina quality
                    ]
                },
                globalSettings: {
                    backgroundColor: 'transparent'
                }
            },
            
            'print-ready': {
                name: 'Print Ready',
                description: 'High-quality exports for printing',
                formats: ['png', 'pdf'],
                variants: {
                    png: [
                        { dpi: 300, compression: 1.0 },
                        { dpi: 600, compression: 1.0 },
                    ],
                    pdf: [
                        { pageFormat: 'A4', orientation: 'landscape' },
                        { pageFormat: 'A3', orientation: 'landscape' },
                    ]
                },
                globalSettings: {
                    backgroundColor: '#ffffff'
                }
            },
            
            'publication': {
                name: 'Publication Package',
                description: 'Complete package for academic publication',
                formats: ['svg', 'png', 'pdf'],
                variants: {
                    svg: [
                        { embedStyles: true, includeMetadata: true },
                    ],
                    png: [
                        { dpi: 300, compression: 1.0 },
                        { dpi: 600, compression: 1.0 },
                    ],
                    pdf: [
                        { pageFormat: 'A4', margins: 25, imageDPI: 300 },
                    ]
                },
                globalSettings: {
                    backgroundColor: '#ffffff',
                    includeMetadata: true
                }
            },
            
            'layout-comparison': {
                name: 'Layout Comparison',
                description: 'Multiple layouts for comparison',
                formats: ['png'],
                layouts: ['force-directed', 'circular', 'hierarchical', 'grid'],
                variants: {
                    png: [
                        { dpi: 150, compression: 0.9 },
                    ]
                },
                globalSettings: {
                    backgroundColor: '#ffffff'
                }
            }
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
     * Cleanup and destroy
     */
    destroy() {
        this.cancelBatch();
        this.eventHandlers.clear();
        
        if (this.archiveWorker) {
            this.archiveWorker.terminate();
        }
        
        console.log('📦 BatchExportSystem destroyed');
    }
}

/**
 * Batch Export Dialog
 * UI for configuring and running batch exports
 */
class BatchExportDialog {
    constructor(batchExportSystem) {
        this.batchExportSystem = batchExportSystem;
        this.dialogElement = null;
        this.isVisible = false;
        
        this.createDialog();
        this.setupEventListeners();
        
        console.log('📦 BatchExportDialog initialized');
    }
    
    /**
     * Create dialog UI
     */
    createDialog() {
        this.dialogElement = document.createElement('div');
        this.dialogElement.className = 'batch-export-dialog';
        this.dialogElement.style.display = 'none';
        
        this.dialogElement.innerHTML = `
            <div class="dialog-overlay"></div>
            <div class="dialog-content">
                <div class="dialog-header">
                    <h2>📦 Batch Export</h2>
                    <button class="close-btn">✕</button>
                </div>
                
                <div class="dialog-body">
                    <div class="config-section">
                        <h3>Export Configuration</h3>
                        
                        <div class="form-group">
                            <label for="batch-name">Batch Name</label>
                            <input type="text" id="batch-name" placeholder="My Export Batch">
                        </div>
                        
                        <div class="form-group">
                            <label>Export Formats</label>
                            <div class="checkbox-group">
                                <label><input type="checkbox" value="svg" checked> SVG</label>
                                <label><input type="checkbox" value="png" checked> PNG</label>
                                <label><input type="checkbox" value="pdf"> PDF</label>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label>Presets</label>
                            <select id="preset-select">
                                <option value="">Custom Configuration</option>
                                <option value="web-package">Web Package</option>
                                <option value="print-ready">Print Ready</option>
                                <option value="publication">Publication</option>
                                <option value="layout-comparison">Layout Comparison</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="progress-section" id="progress-section" style="display: none;">
                        <h3>Export Progress</h3>
                        <div class="progress-bar">
                            <div class="progress-fill"></div>
                        </div>
                        <div class="progress-text">Ready to start...</div>
                        <div class="progress-details">
                            <span class="completed">0</span> completed, 
                            <span class="failed">0</span> failed, 
                            <span class="total">0</span> total
                        </div>
                    </div>
                </div>
                
                <div class="dialog-footer">
                    <button class="btn btn-secondary" id="cancel-btn">Cancel</button>
                    <button class="btn btn-primary" id="start-btn">Start Export</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(this.dialogElement);
        this.addDialogStyles();
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Close button
        this.dialogElement.querySelector('.close-btn').addEventListener('click', () => this.hide());
        this.dialogElement.querySelector('#cancel-btn').addEventListener('click', () => this.hide());
        
        // Overlay click to close
        this.dialogElement.querySelector('.dialog-overlay').addEventListener('click', () => this.hide());
        
        // Start export
        this.dialogElement.querySelector('#start-btn').addEventListener('click', () => this.startBatchExport());
        
        // Preset selection
        this.dialogElement.querySelector('#preset-select').addEventListener('change', (e) => {
            this.applyPreset(e.target.value);
        });
        
        // Listen to batch events
        this.batchExportSystem.on('batchProgress', (data) => this.updateProgress(data));
        this.batchExportSystem.on('batchComplete', (data) => this.handleCompletion(data));
        this.batchExportSystem.on('batchError', (data) => this.handleError(data));
    }
    
    /**
     * Show dialog
     */
    show() {
        this.isVisible = true;
        this.dialogElement.style.display = 'block';
        
        // Reset form
        this.resetForm();
    }
    
    /**
     * Hide dialog
     */
    hide() {
        this.isVisible = false;
        this.dialogElement.style.display = 'none';
    }
    
    /**
     * Reset form to defaults
     */
    resetForm() {
        document.getElementById('batch-name').value = '';
        document.getElementById('preset-select').value = '';
        
        // Reset checkboxes
        const checkboxes = this.dialogElement.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(cb => {
            cb.checked = cb.value === 'svg' || cb.value === 'png';
        });
        
        // Hide progress
        document.getElementById('progress-section').style.display = 'none';
    }
    
    /**
     * Apply preset configuration
     */
    applyPreset(presetName) {
        if (!presetName) return;
        
        const presets = BatchExportSystem.createPresets();
        const preset = presets[presetName];
        
        if (preset) {
            // Update batch name
            document.getElementById('batch-name').value = preset.name;
            
            // Update format checkboxes
            const checkboxes = this.dialogElement.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(cb => {
                cb.checked = preset.formats.includes(cb.value);
            });
        }
    }
    
    /**
     * Start batch export
     */
    async startBatchExport() {
        const config = this.getConfigFromForm();
        
        if (!config) return;
        
        // Show progress section
        document.getElementById('progress-section').style.display = 'block';
        
        // Disable start button
        const startBtn = document.getElementById('start-btn');
        startBtn.disabled = true;
        startBtn.textContent = 'Exporting...';
        
        try {
            const result = await this.batchExportSystem.startBatchExport(config);
            console.log('📦 Batch export result:', result);
        } catch (error) {
            console.error('📦 Batch export failed:', error);
            this.handleError({ error: error.message });
        }
    }
    
    /**
     * Get configuration from form
     */
    getConfigFromForm() {
        const name = document.getElementById('batch-name').value.trim();
        if (!name) {
            alert('Please enter a batch name');
            return null;
        }
        
        const formats = Array.from(this.dialogElement.querySelectorAll('input[type="checkbox"]:checked'))
            .map(cb => cb.value);
        
        if (formats.length === 0) {
            alert('Please select at least one export format');
            return null;
        }
        
        const presetName = document.getElementById('preset-select').value;
        
        if (presetName) {
            // Use preset configuration
            const presets = BatchExportSystem.createPresets();
            const preset = presets[presetName];
            return { ...preset, name };
        } else {
            // Create custom configuration
            return {
                name,
                formats,
                description: `Custom batch export created on ${new Date().toLocaleDateString()}`
            };
        }
    }
    
    /**
     * Update progress display
     */
    updateProgress(data) {
        const progressFill = this.dialogElement.querySelector('.progress-fill');
        const progressText = this.dialogElement.querySelector('.progress-text');
        const completedSpan = this.dialogElement.querySelector('.completed');
        const failedSpan = this.dialogElement.querySelector('.failed');
        const totalSpan = this.dialogElement.querySelector('.total');
        
        progressFill.style.width = `${data.progress.percentage}%`;
        progressText.textContent = `Exporting... ${Math.round(data.progress.percentage)}%`;
        
        completedSpan.textContent = data.progress.completed;
        failedSpan.textContent = data.progress.failed;
        totalSpan.textContent = data.progress.total;
    }
    
    /**
     * Handle batch completion
     */
    handleCompletion(data) {
        const progressText = this.dialogElement.querySelector('.progress-text');
        const startBtn = document.getElementById('start-btn');
        
        progressText.textContent = `✅ Batch export completed! ${data.result.successfulJobs}/${data.result.totalJobs} succeeded`;
        
        startBtn.disabled = false;
        startBtn.textContent = 'Start Export';
        
        // Auto-hide after delay
        setTimeout(() => {
            if (this.isVisible) {
                this.hide();
            }
        }, 3000);
    }
    
    /**
     * Handle batch error
     */
    handleError(data) {
        const progressText = this.dialogElement.querySelector('.progress-text');
        const startBtn = document.getElementById('start-btn');
        
        progressText.textContent = `❌ Export failed: ${data.error}`;
        
        startBtn.disabled = false;
        startBtn.textContent = 'Start Export';
    }
    
    /**
     * Add dialog styles
     */
    addDialogStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .batch-export-dialog {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 10000;
            }
            
            .dialog-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
            }
            
            .dialog-content {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: #2a2a2a;
                border-radius: 8px;
                min-width: 500px;
                max-width: 90vw;
                max-height: 90vh;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
                color: #ddd;
                font-family: 'Inter', sans-serif;
            }
            
            .dialog-header {
                padding: 20px 24px 16px;
                border-bottom: 1px solid #444;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .dialog-header h2 {
                margin: 0;
                font-size: 18px;
                color: #fff;
            }
            
            .close-btn {
                background: none;
                border: none;
                color: #aaa;
                font-size: 18px;
                cursor: pointer;
                padding: 4px;
            }
            
            .close-btn:hover {
                color: #fff;
            }
            
            .dialog-body {
                padding: 24px;
                max-height: 60vh;
                overflow-y: auto;
            }
            
            .config-section,
            .progress-section {
                margin-bottom: 24px;
            }
            
            .config-section h3,
            .progress-section h3 {
                margin: 0 0 16px 0;
                font-size: 16px;
                color: #fff;
            }
            
            .form-group {
                margin-bottom: 16px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 6px;
                font-weight: 500;
                color: #fff;
            }
            
            .form-group input,
            .form-group select {
                width: 100%;
                padding: 8px 12px;
                background: #444;
                border: 1px solid #555;
                border-radius: 4px;
                color: #fff;
                font-size: 14px;
            }
            
            .form-group input:focus,
            .form-group select:focus {
                outline: none;
                border-color: #4CAF50;
            }
            
            .checkbox-group {
                display: flex;
                gap: 16px;
                flex-wrap: wrap;
            }
            
            .checkbox-group label {
                display: flex;
                align-items: center;
                gap: 6px;
                margin-bottom: 0;
                cursor: pointer;
            }
            
            .progress-bar {
                background: #444;
                border-radius: 4px;
                height: 8px;
                overflow: hidden;
                margin-bottom: 12px;
            }
            
            .progress-fill {
                background: linear-gradient(90deg, #4CAF50, #8BC34A);
                height: 100%;
                width: 0%;
                transition: width 0.3s ease;
            }
            
            .progress-text {
                font-size: 14px;
                color: #fff;
                margin-bottom: 8px;
            }
            
            .progress-details {
                font-size: 12px;
                color: #aaa;
            }
            
            .dialog-footer {
                padding: 16px 24px;
                border-top: 1px solid #444;
                display: flex;
                justify-content: flex-end;
                gap: 12px;
            }
            
            .btn {
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s;
            }
            
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            
            .btn-secondary {
                background: #555;
                color: #fff;
            }
            
            .btn-secondary:hover:not(:disabled) {
                background: #666;
            }
            
            .btn-primary {
                background: #4CAF50;
                color: white;
            }
            
            .btn-primary:hover:not(:disabled) {
                background: #45a049;
            }
        `;
        
        document.head.appendChild(style);
    }
}

// Global instances
window.GroggyBatchExportSystem = null;
window.GroggyBatchExportDialog = null;

/**
 * Initialize batch export system
 */
function initializeBatchExportSystem(exportSystem, config = {}) {
    if (window.GroggyBatchExportSystem) {
        console.warn('Batch export system already initialized');
        return window.GroggyBatchExportSystem;
    }
    
    window.GroggyBatchExportSystem = new BatchExportSystem(exportSystem, config);
    window.GroggyBatchExportDialog = new BatchExportDialog(window.GroggyBatchExportSystem);
    
    console.log('📦 Batch export system initialized');
    
    return {
        batchSystem: window.GroggyBatchExportSystem,
        dialog: window.GroggyBatchExportDialog,
        presets: BatchExportSystem.createPresets()
    };
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        BatchExportSystem,
        BatchExportDialog,
        initializeBatchExportSystem
    };
}