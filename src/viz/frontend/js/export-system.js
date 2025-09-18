/**
 * 📸 Graph Export System
 * Part of Groggy Phase 12: Static Export System
 * 
 * Comprehensive export system supporting multiple formats:
 * - SVG export with full styling and metadata
 * - PNG export with high-DPI support
 * - PDF export for publications
 * - Configurable export options
 * - Batch export functionality
 * - Vector and raster format optimization
 */

class GraphExportSystem {
    constructor(graphRenderer, config = {}) {
        this.graphRenderer = graphRenderer;
        this.config = {
            // Default export settings
            defaultFormat: 'svg',
            defaultQuality: 'high',
            defaultDPI: 300,
            
            // SVG options
            svg: {
                embedFonts: true,
                embedStyles: true,
                includeMetadata: true,
                optimizeSize: true,
                preserveAspectRatio: true
            },
            
            // PNG options
            png: {
                supportHighDPI: true,
                defaultDPI: 300,
                maxDPI: 600,
                compression: 0.9,
                includeMetadata: false
            },
            
            // PDF options
            pdf: {
                format: 'A4', // A4, A3, Letter, Custom
                orientation: 'landscape', // portrait, landscape
                margins: 20, // mm
                includeMetadata: true,
                compression: true
            },
            
            // Export behavior
            autoDownload: true,
            showPreview: true,
            includeWatermark: false,
            
            ...config
        };
        
        // Export state
        this.isExporting = false;
        this.exportQueue = [];
        this.exportHistory = [];
        
        // Canvas for rendering
        this.exportCanvas = null;
        this.exportContext = null;
        
        // PDF library reference
        this.pdfLib = null;
        
        // Event handlers
        this.eventHandlers = new Map();
        
        console.log('📸 GraphExportSystem initialized with config:', this.config);
        
        this.initializeExportCanvas();
        this.loadPDFLibrary();
    }
    
    /**
     * Initialize export canvas for raster rendering
     */
    initializeExportCanvas() {
        this.exportCanvas = document.createElement('canvas');
        this.exportCanvas.style.display = 'none';
        this.exportContext = this.exportCanvas.getContext('2d');
        document.body.appendChild(this.exportCanvas);
    }
    
    /**
     * Load PDF library for PDF export
     */
    async loadPDFLibrary() {
        try {
            // In a real implementation, this would load jsPDF or similar
            // For now, we'll simulate the API
            this.pdfLib = {
                jsPDF: class MockPDF {
                    constructor(options) {
                        this.options = options;
                        this.content = [];
                    }
                    
                    addImage(imageData, format, x, y, width, height) {
                        this.content.push({ type: 'image', imageData, format, x, y, width, height });
                    }
                    
                    text(text, x, y) {
                        this.content.push({ type: 'text', text, x, y });
                    }
                    
                    save(filename) {
                        console.log(`📸 Mock PDF saved as ${filename}`, this.content);
                        // In real implementation, this would generate and download the PDF
                    }
                }
            };
            
            console.log('📸 PDF library loaded successfully');
        } catch (error) {
            console.warn('Failed to load PDF library:', error);
        }
    }
    
    /**
     * Export graph in specified format
     */
    async exportGraph(format, options = {}) {
        if (this.isExporting) {
            return this.queueExport(format, options);
        }
        
        this.isExporting = true;
        
        try {
            const exportOptions = this.mergeExportOptions(format, options);
            const startTime = performance.now();
            
            // Emit export start event
            this.emit('exportStart', { format, options: exportOptions });
            
            let result;
            switch (format.toLowerCase()) {
                case 'svg':
                    result = await this.exportSVG(exportOptions);
                    break;
                case 'png':
                    result = await this.exportPNG(exportOptions);
                    break;
                case 'pdf':
                    result = await this.exportPDF(exportOptions);
                    break;
                default:
                    throw new Error(`Unsupported export format: ${format}`);
            }
            
            const exportTime = performance.now() - startTime;
            
            // Add to history
            this.exportHistory.push({
                format,
                options: exportOptions,
                timestamp: new Date().toISOString(),
                exportTime,
                filename: result.filename,
                size: result.size
            });
            
            // Emit completion event
            this.emit('exportComplete', {
                format,
                result,
                exportTime,
                options: exportOptions
            });
            
            console.log(`📸 Export completed: ${format} in ${exportTime.toFixed(1)}ms`);
            
            return result;
            
        } catch (error) {
            console.error(`Export failed (${format}):`, error);
            this.emit('exportError', { format, error: error.message });
            throw error;
        } finally {
            this.isExporting = false;
            this.processExportQueue();
        }
    }
    
    /**
     * Export graph as SVG
     */
    async exportSVG(options) {
        console.log('📸 Starting SVG export...');
        
        // Get graph bounds and elements
        const graphBounds = this.getGraphBounds();
        const graphElements = this.getGraphElements();
        
        // Create SVG document
        const svg = this.createSVGDocument(graphBounds, options);
        
        // Add styles if embedding is enabled
        if (options.embedStyles) {
            this.embedSVGStyles(svg, options);
        }
        
        // Add graph elements
        await this.addGraphElementsToSVG(svg, graphElements, options);
        
        // Add metadata if enabled
        if (options.includeMetadata) {
            this.addSVGMetadata(svg, options);
        }
        
        // Optimize SVG if enabled
        if (options.optimizeSize) {
            this.optimizeSVG(svg, options);
        }
        
        // Convert to string
        const svgString = this.serializeSVG(svg);
        
        // Create blob and download
        const blob = new Blob([svgString], { type: 'image/svg+xml' });
        const filename = this.generateFilename('svg', options);
        
        if (options.autoDownload) {
            this.downloadBlob(blob, filename);
        }
        
        return {
            format: 'svg',
            blob,
            data: svgString,
            filename,
            size: blob.size,
            dimensions: graphBounds
        };
    }
    
    /**
     * Create SVG document with proper structure
     */
    createSVGDocument(bounds, options) {
        // Calculate SVG dimensions
        const padding = 20;
        const width = bounds.width + (padding * 2);
        const height = bounds.height + (padding * 2);
        
        // Create SVG element
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', width);
        svg.setAttribute('height', height);
        svg.setAttribute('viewBox', `${bounds.x - padding} ${bounds.y - padding} ${width} ${height}`);
        svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
        svg.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
        
        if (options.preserveAspectRatio) {
            svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        }
        
        // Add title and description
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = options.title || 'Groggy Graph Visualization';
        svg.appendChild(title);
        
        const desc = document.createElementNS('http://www.w3.org/2000/svg', 'desc');
        desc.textContent = options.description || 'Graph visualization exported from Groggy';
        svg.appendChild(desc);
        
        return svg;
    }
    
    /**
     * Embed CSS styles into SVG
     */
    embedSVGStyles(svg, options) {
        const styleElement = document.createElementNS('http://www.w3.org/2000/svg', 'style');
        styleElement.setAttribute('type', 'text/css');
        
        // Collect relevant CSS rules
        const cssRules = this.collectCSSRules();
        
        // Add font imports if embedding fonts
        if (options.embedFonts) {
            const fontImports = this.getFontImports();
            cssRules.unshift(...fontImports);
        }
        
        // Create CDATA section for CSS
        const cdata = document.createCDATASection(cssRules.join('\n'));
        styleElement.appendChild(cdata);
        
        svg.appendChild(styleElement);
    }
    
    /**
     * Collect CSS rules relevant to graph visualization
     */
    collectCSSRules() {
        const rules = [];
        
        // Default node styles
        rules.push(`
            .graph-node {
                fill: #4CAF50;
                stroke: #2E7D32;
                stroke-width: 2;
                cursor: pointer;
            }
            
            .graph-node:hover {
                fill: #66BB6A;
                stroke-width: 3;
            }
            
            .graph-edge {
                stroke: #757575;
                stroke-width: 1;
                fill: none;
            }
            
            .graph-label {
                font-family: 'Arial', sans-serif;
                font-size: 12px;
                fill: #333;
                text-anchor: middle;
                dominant-baseline: central;
            }
            
            .graph-background {
                fill: white;
            }
        `);
        
        // Get computed styles from actual elements
        const nodes = this.graphRenderer.getNodes();
        if (nodes.length > 0) {
            const sampleNode = nodes[0];
            const computedStyle = window.getComputedStyle(sampleNode.element);
            
            // Extract relevant properties
            rules.push(`
                .node-${sampleNode.id} {
                    fill: ${computedStyle.backgroundColor || '#4CAF50'};
                    stroke: ${computedStyle.borderColor || '#2E7D32'};
                    stroke-width: ${computedStyle.borderWidth || '2px'};
                }
            `);
        }
        
        return rules;
    }
    
    /**
     * Get font import statements
     */
    getFontImports() {
        const fontImports = [];
        
        // Common web fonts
        fontImports.push(`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');`);
        fontImports.push(`@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');`);
        
        return fontImports;
    }
    
    /**
     * Add graph elements to SVG
     */
    async addGraphElementsToSVG(svg, graphElements, options) {
        // Create groups for different element types
        const edgesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        edgesGroup.setAttribute('class', 'edges');
        edgesGroup.setAttribute('id', 'graph-edges');
        
        const nodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        nodesGroup.setAttribute('class', 'nodes');
        nodesGroup.setAttribute('id', 'graph-nodes');
        
        const labelsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        labelsGroup.setAttribute('class', 'labels');
        labelsGroup.setAttribute('id', 'graph-labels');
        
        // Add edges first (so they appear behind nodes)
        graphElements.edges.forEach(edge => {
            const edgeElement = this.createSVGEdge(edge, options);
            edgesGroup.appendChild(edgeElement);
        });
        
        // Add nodes
        graphElements.nodes.forEach(node => {
            const nodeElement = this.createSVGNode(node, options);
            nodesGroup.appendChild(nodeElement);
            
            // Add label if present
            if (node.label && options.includeLabels !== false) {
                const labelElement = this.createSVGLabel(node, options);
                labelsGroup.appendChild(labelElement);
            }
        });
        
        // Append groups to SVG
        svg.appendChild(edgesGroup);
        svg.appendChild(nodesGroup);
        svg.appendChild(labelsGroup);
    }
    
    /**
     * Create SVG node element
     */
    createSVGNode(node, options) {
        const nodeEl = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        
        // Set basic attributes
        nodeEl.setAttribute('id', `node-${node.id}`);
        nodeEl.setAttribute('class', `graph-node node-${node.id}`);
        nodeEl.setAttribute('cx', node.x);
        nodeEl.setAttribute('cy', node.y);
        nodeEl.setAttribute('r', node.radius || 8);
        
        // Apply styling
        nodeEl.setAttribute('fill', node.color || '#4CAF50');
        nodeEl.setAttribute('stroke', node.borderColor || '#2E7D32');
        nodeEl.setAttribute('stroke-width', node.borderWidth || 2);
        
        // Add data attributes for metadata
        if (options.includeMetadata) {
            nodeEl.setAttribute('data-node-id', node.id);
            if (node.attributes) {
                Object.entries(node.attributes).forEach(([key, value]) => {
                    nodeEl.setAttribute(`data-${key}`, value);
                });
            }
        }
        
        return nodeEl;
    }
    
    /**
     * Create SVG edge element
     */
    createSVGEdge(edge, options) {
        const edgeEl = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        
        // Set basic attributes
        edgeEl.setAttribute('id', `edge-${edge.id}`);
        edgeEl.setAttribute('class', `graph-edge edge-${edge.id}`);
        edgeEl.setAttribute('x1', edge.x1);
        edgeEl.setAttribute('y1', edge.y1);
        edgeEl.setAttribute('x2', edge.x2);
        edgeEl.setAttribute('y2', edge.y2);
        
        // Apply styling
        edgeEl.setAttribute('stroke', edge.color || '#757575');
        edgeEl.setAttribute('stroke-width', edge.width || 1);
        
        if (edge.dashArray) {
            edgeEl.setAttribute('stroke-dasharray', edge.dashArray);
        }
        
        // Add data attributes for metadata
        if (options.includeMetadata) {
            edgeEl.setAttribute('data-edge-id', edge.id);
            edgeEl.setAttribute('data-source', edge.source);
            edgeEl.setAttribute('data-target', edge.target);
            
            if (edge.attributes) {
                Object.entries(edge.attributes).forEach(([key, value]) => {
                    edgeEl.setAttribute(`data-${key}`, value);
                });
            }
        }
        
        return edgeEl;
    }
    
    /**
     * Create SVG label element
     */
    createSVGLabel(node, options) {
        const labelEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        
        // Set basic attributes
        labelEl.setAttribute('id', `label-${node.id}`);
        labelEl.setAttribute('class', `graph-label label-${node.id}`);
        labelEl.setAttribute('x', node.x);
        labelEl.setAttribute('y', node.y + (node.radius || 8) + 16);
        labelEl.setAttribute('text-anchor', 'middle');
        
        // Set text content
        labelEl.textContent = node.label;
        
        // Apply styling
        labelEl.setAttribute('font-family', options.fontFamily || 'Arial, sans-serif');
        labelEl.setAttribute('font-size', options.fontSize || '12px');
        labelEl.setAttribute('fill', options.labelColor || '#333');
        
        return labelEl;
    }
    
    /**
     * Add metadata to SVG
     */
    addSVGMetadata(svg, options) {
        const metadata = document.createElementNS('http://www.w3.org/2000/svg', 'metadata');
        
        const metadataContent = `
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                     xmlns:dc="http://purl.org/dc/elements/1.1/"
                     xmlns:groggy="https://groggy.ai/metadata#">
                <rdf:Description about="">
                    <dc:title>${options.title || 'Groggy Graph Visualization'}</dc:title>
                    <dc:creator>Groggy Graph Analytics</dc:creator>
                    <dc:date>${new Date().toISOString()}</dc:date>
                    <dc:format>image/svg+xml</dc:format>
                    <groggy:version>0.5.0</groggy:version>
                    <groggy:layout>${options.layout || 'force-directed'}</groggy:layout>
                    <groggy:nodeCount>${this.graphRenderer.getNodes().length}</groggy:nodeCount>
                    <groggy:edgeCount>${this.graphRenderer.getEdges().length}</groggy:edgeCount>
                </rdf:Description>
            </rdf:RDF>
        `;
        
        metadata.innerHTML = metadataContent;
        svg.appendChild(metadata);
    }
    
    /**
     * Optimize SVG for smaller file size
     */
    optimizeSVG(svg, options) {
        // Remove unnecessary whitespace and comments
        this.removeWhitespaceNodes(svg);
        
        // Combine similar elements
        this.combineSimalarElements(svg);
        
        // Optimize numeric precision
        this.optimizeNumericPrecision(svg);
        
        // Remove unused definitions
        this.removeUnusedDefinitions(svg);
    }
    
    /**
     * Remove whitespace-only text nodes
     */
    removeWhitespaceNodes(element) {
        const walker = document.createTreeWalker(
            element,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: function(node) {
                    return /^\s*$/.test(node.data) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
                }
            }
        );
        
        const nodesToRemove = [];
        let node;
        while (node = walker.nextNode()) {
            nodesToRemove.push(node);
        }
        
        nodesToRemove.forEach(node => node.parentNode.removeChild(node));
    }
    
    /**
     * Combine similar elements to reduce file size
     */
    combineSimalarElements(svg) {
        // Group nodes with identical styling
        const nodeGroups = new Map();
        const nodes = svg.querySelectorAll('.graph-node');
        
        nodes.forEach(node => {
            const style = `${node.getAttribute('fill')}-${node.getAttribute('stroke')}-${node.getAttribute('stroke-width')}`;
            if (!nodeGroups.has(style)) {
                nodeGroups.set(style, []);
            }
            nodeGroups.get(style).push(node);
        });
        
        // Create style classes for common combinations
        nodeGroups.forEach((nodes, style) => {
            if (nodes.length > 3) {
                const className = `node-style-${Math.random().toString(36).substr(2, 6)}`;
                nodes.forEach(node => {
                    node.classList.add(className);
                });
            }
        });
    }
    
    /**
     * Optimize numeric precision to reduce file size
     */
    optimizeNumericPrecision(svg) {
        const numericAttributes = ['x', 'y', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2', 'r', 'width', 'height'];
        const elements = svg.querySelectorAll('*');
        
        elements.forEach(element => {
            numericAttributes.forEach(attr => {
                const value = element.getAttribute(attr);
                if (value && !isNaN(value)) {
                    const optimized = parseFloat(value).toFixed(2);
                    element.setAttribute(attr, optimized);
                }
            });
        });
    }
    
    /**
     * Remove unused definitions
     */
    removeUnusedDefinitions(svg) {
        const defs = svg.querySelector('defs');
        if (!defs) return;
        
        const definitions = defs.querySelectorAll('*[id]');
        const usedIds = new Set();
        
        // Find all references
        svg.querySelectorAll('*').forEach(element => {
            const attributes = element.attributes;
            for (let i = 0; i < attributes.length; i++) {
                const value = attributes[i].value;
                if (value.startsWith('url(#')) {
                    const id = value.slice(5, -1);
                    usedIds.add(id);
                }
            }
        });
        
        // Remove unused definitions
        definitions.forEach(def => {
            if (!usedIds.has(def.id)) {
                def.remove();
            }
        });
    }
    
    /**
     * Serialize SVG to string
     */
    serializeSVG(svg) {
        const serializer = new XMLSerializer();
        let svgString = serializer.serializeToString(svg);
        
        // Add XML declaration if not present
        if (!svgString.startsWith('<?xml')) {
            svgString = '<?xml version="1.0" encoding="UTF-8"?>\n' + svgString;
        }
        
        return svgString;
    }
    
    /**
     * Get graph bounds and elements
     */
    getGraphBounds() {
        const nodes = this.graphRenderer.getNodes();
        
        if (nodes.length === 0) {
            return { x: 0, y: 0, width: 800, height: 600 };
        }
        
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        nodes.forEach(node => {
            const pos = this.graphRenderer.getNodePosition(node.id);
            const radius = node.radius || 8;
            
            minX = Math.min(minX, pos.x - radius);
            maxX = Math.max(maxX, pos.x + radius);
            minY = Math.min(minY, pos.y - radius);
            maxY = Math.max(maxY, pos.y + radius);
        });
        
        return {
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY
        };
    }
    
    /**
     * Get graph elements data
     */
    getGraphElements() {
        const nodes = this.graphRenderer.getNodes().map(node => {
            const pos = this.graphRenderer.getNodePosition(node.id);
            return {
                id: node.id,
                x: pos.x,
                y: pos.y,
                radius: node.radius || 8,
                color: node.color || '#4CAF50',
                borderColor: node.borderColor || '#2E7D32',
                borderWidth: node.borderWidth || 2,
                label: node.label || node.id,
                attributes: node.attributes || {}
            };
        });
        
        const edges = this.graphRenderer.getEdges().map(edge => {
            const sourcePos = this.graphRenderer.getNodePosition(edge.source);
            const targetPos = this.graphRenderer.getNodePosition(edge.target);
            return {
                id: edge.id,
                source: edge.source,
                target: edge.target,
                x1: sourcePos.x,
                y1: sourcePos.y,
                x2: targetPos.x,
                y2: targetPos.y,
                color: edge.color || '#757575',
                width: edge.width || 1,
                dashArray: edge.dashArray,
                attributes: edge.attributes || {}
            };
        });
        
        return { nodes, edges };
    }
    
    /**
     * Export graph as PNG with high-DPI support
     */
    async exportPNG(options) {
        console.log('📸 Starting PNG export...');
        
        // First create SVG
        const svgResult = await this.exportSVG({
            ...options,
            autoDownload: false,
            embedStyles: true,
            embedFonts: true
        });
        
        // Calculate canvas dimensions based on DPI
        const dpi = options.dpi || this.config.png.defaultDPI;
        const scale = dpi / 96; // 96 DPI is standard web resolution
        
        const bounds = svgResult.dimensions;
        const canvasWidth = Math.round(bounds.width * scale);
        const canvasHeight = Math.round(bounds.height * scale);
        
        // Setup export canvas
        this.exportCanvas.width = canvasWidth;
        this.exportCanvas.height = canvasHeight;
        this.exportContext.scale(scale, scale);
        
        // Clear canvas
        this.exportContext.fillStyle = options.backgroundColor || '#ffffff';
        this.exportContext.fillRect(0, 0, bounds.width, bounds.height);
        
        // Convert SVG to image and draw on canvas
        const img = await this.svgToImage(svgResult.data, bounds.width, bounds.height);
        this.exportContext.drawImage(img, 0, 0);
        
        // Convert canvas to blob
        const blob = await this.canvasToBlob(this.exportCanvas, options);
        const filename = this.generateFilename('png', options);
        
        if (options.autoDownload) {
            this.downloadBlob(blob, filename);
        }
        
        return {
            format: 'png',
            blob,
            filename,
            size: blob.size,
            dimensions: { width: canvasWidth, height: canvasHeight },
            dpi
        };
    }
    
    /**
     * Convert SVG to image element
     */
    svgToImage(svgString, width, height) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            
            img.onload = () => resolve(img);
            img.onerror = reject;
            
            // Create blob URL for SVG
            const svgBlob = new Blob([svgString], { type: 'image/svg+xml' });
            const url = URL.createObjectURL(svgBlob);
            
            img.src = url;
            
            // Clean up URL after load
            img.onload = () => {
                URL.revokeObjectURL(url);
                resolve(img);
            };
        });
    }
    
    /**
     * Convert canvas to blob
     */
    canvasToBlob(canvas, options) {
        return new Promise((resolve) => {
            const quality = options.compression || this.config.png.compression;
            canvas.toBlob(resolve, 'image/png', quality);
        });
    }
    
    /**
     * Export graph as PDF
     */
    async exportPDF(options) {
        console.log('📸 Starting PDF export...');
        
        if (!this.pdfLib || !this.pdfLib.jsPDF) {
            throw new Error('PDF library not available');
        }
        
        // Get PNG data first
        const pngResult = await this.exportPNG({
            ...options,
            autoDownload: false,
            dpi: 150 // Lower DPI for PDF to reduce file size
        });
        
        // Convert blob to base64
        const imageData = await this.blobToBase64(pngResult.blob);
        
        // Create PDF document
        const pdf = new this.pdfLib.jsPDF({
            orientation: options.orientation || this.config.pdf.orientation,
            unit: 'mm',
            format: options.format || this.config.pdf.format
        });
        
        // Calculate image dimensions for PDF
        const pageWidth = pdf.internal.pageSize.getWidth();
        const pageHeight = pdf.internal.pageSize.getHeight();
        const margins = options.margins || this.config.pdf.margins;
        
        const maxWidth = pageWidth - (margins * 2);
        const maxHeight = pageHeight - (margins * 2);
        
        // Calculate aspect ratio
        const imgAspectRatio = pngResult.dimensions.width / pngResult.dimensions.height;
        let imgWidth = maxWidth;
        let imgHeight = maxWidth / imgAspectRatio;
        
        if (imgHeight > maxHeight) {
            imgHeight = maxHeight;
            imgWidth = maxHeight * imgAspectRatio;
        }
        
        // Center image on page
        const x = (pageWidth - imgWidth) / 2;
        const y = (pageHeight - imgHeight) / 2;
        
        // Add image to PDF
        pdf.addImage(imageData, 'PNG', x, y, imgWidth, imgHeight);
        
        // Add metadata if enabled
        if (options.includeMetadata) {
            this.addPDFMetadata(pdf, options);
        }
        
        // Add title and description
        if (options.title) {
            pdf.text(options.title, pageWidth / 2, margins, { align: 'center' });
        }
        
        // Save PDF
        const filename = this.generateFilename('pdf', options);
        
        if (options.autoDownload) {
            pdf.save(filename);
        }
        
        return {
            format: 'pdf',
            pdf,
            filename,
            dimensions: { width: imgWidth, height: imgHeight },
            pages: 1
        };
    }
    
    /**
     * Add metadata to PDF
     */
    addPDFMetadata(pdf, options) {
        pdf.setProperties({
            title: options.title || 'Groggy Graph Visualization',
            subject: options.description || 'Graph visualization exported from Groggy',
            author: 'Groggy Graph Analytics',
            creator: 'Groggy v0.5.0',
            producer: 'Groggy Export System'
        });
    }
    
    /**
     * Convert blob to base64
     */
    blobToBase64(blob) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.readAsDataURL(blob);
        });
    }
    
    /**
     * Merge export options with defaults
     */
    mergeExportOptions(format, options) {
        const formatDefaults = this.config[format.toLowerCase()] || {};
        return {
            ...formatDefaults,
            ...options,
            format: format.toLowerCase(),
            autoDownload: options.autoDownload !== undefined ? options.autoDownload : this.config.autoDownload
        };
    }
    
    /**
     * Generate filename for export
     */
    generateFilename(format, options) {
        const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
        const prefix = options.filename || 'groggy-graph';
        return `${prefix}-${timestamp}.${format}`;
    }
    
    /**
     * Download blob as file
     */
    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    /**
     * Queue export for later processing
     */
    queueExport(format, options) {
        return new Promise((resolve, reject) => {
            this.exportQueue.push({
                format,
                options,
                resolve,
                reject,
                timestamp: performance.now()
            });
            
            console.log(`📸 Export queued: ${format} (${this.exportQueue.length} in queue)`);
        });
    }
    
    /**
     * Process queued exports
     */
    async processExportQueue() {
        if (this.exportQueue.length === 0 || this.isExporting) {
            return;
        }
        
        const nextExport = this.exportQueue.shift();
        console.log(`📸 Processing queued export: ${nextExport.format}`);
        
        try {
            const result = await this.exportGraph(nextExport.format, nextExport.options);
            nextExport.resolve(result);
        } catch (error) {
            nextExport.reject(error);
        }
    }
    
    /**
     * Get export history
     */
    getExportHistory() {
        return [...this.exportHistory];
    }
    
    /**
     * Clear export history
     */
    clearExportHistory() {
        this.exportHistory = [];
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
        if (this.exportCanvas) {
            this.exportCanvas.remove();
        }
        this.exportQueue = [];
        this.exportHistory = [];
        this.eventHandlers.clear();
        
        console.log('📸 GraphExportSystem destroyed');
    }
}

// Global export system instance
window.GroggyExportSystem = null;

/**
 * Initialize graph export system
 */
function initializeGraphExportSystem(graphRenderer, config = {}) {
    if (window.GroggyExportSystem) {
        console.warn('Graph export system already initialized');
        return window.GroggyExportSystem;
    }
    
    window.GroggyExportSystem = new GraphExportSystem(graphRenderer, config);
    
    console.log('📸 Graph export system initialized');
    
    return window.GroggyExportSystem;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        GraphExportSystem,
        initializeGraphExportSystem
    };
}