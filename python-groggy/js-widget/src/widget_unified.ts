/**
 * 🧠 Unified Jupyter Widget using GroggyVizCore
 * 
 * Refactored to use the unified JavaScript core for consistency
 * across all visualization environments. This replaces the original
 * widget.ts with a version that leverages the unified architecture.
 */

import { DOMWidgetModel, DOMWidgetView } from '@jupyter-widgets/base';
import { GroggyVizCore } from './core/VizCore.js';

export const MODULE_NAME = 'groggy-widgets';
export const MODULE_VERSION = '0.1.0';

export class GroggyGraphModel extends DOMWidgetModel {
  static model_name = 'GroggyGraphModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;

  defaults() {
    return {
      ...super.defaults(),
      _model_name: 'GroggyGraphModel',
      _model_module: MODULE_NAME,
      _model_module_version: MODULE_VERSION,
      _view_name: 'GroggyGraphView',
      _view_module: MODULE_NAME,
      _view_module_version: MODULE_VERSION,
      
      // Graph data traits
      nodes: [],
      edges: [],
      title: 'Graph Visualization',
      
      // Configuration traits
      width: 800,
      height: 600,
      layout_algorithm: 'force-directed',
      theme: 'light',
      
      // Physics configuration
      physics_enabled: true,
      force_strength: 30,
      link_distance: 50,
      link_strength: 0.1,
      charge_strength: -300,
      center_strength: 0.1,
      
      // Interaction configuration
      enable_drag: true,
      enable_zoom: true,
      enable_pan: true,
      enable_selection: true,
      enable_hover: true,
      
      // State traits
      selected_nodes: [],
      hovered_node: '',
      camera_position: { scale: 1, translate: { x: 0, y: 0 } },
      is_dragging: false,
      
      // Styling
      node_color_scheme: 'default',
      background_color: '#ffffff',
      enable_shadows: true,
      enable_animations: true
    };
  }
}

export class GroggyGraphView extends DOMWidgetView {
  static view_name = 'GroggyGraphView';
  static view_module = MODULE_NAME;
  static view_module_version = MODULE_VERSION;
  
  private vizCore: GroggyVizCore | null = null;
  private container: HTMLElement | null = null;

  render() {
    console.log('🧠 Initializing Groggy Jupyter widget with unified core');
    
    // Create main container
    this.el.style.cssText = `
      border: 2px solid #4CAF50;
      padding: 10px;
      margin: 10px;
      border-radius: 8px;
      background: #f9f9f9;
      font-family: Arial, sans-serif;
      min-height: 400px;
      position: relative;
      user-select: none;
    `;
    
    // Create header
    const header = document.createElement('div');
    header.style.cssText = `
      font-weight: bold;
      margin-bottom: 10px;
      color: #333;
      display: flex;
      justify-content: space-between;
      align-items: center;
    `;
    
    const title = document.createElement('span');
    title.textContent = this.model.get('title') || 'Graph Visualization';
    header.appendChild(title);
    
    const status = document.createElement('span');
    status.style.cssText = `
      font-size: 12px;
      color: #666;
      background: rgba(255,255,255,0.8);
      padding: 2px 6px;
      border-radius: 3px;
    `;
    status.textContent = 'Powered by GroggyVizCore';
    header.appendChild(status);
    
    this.el.appendChild(header);
    
    // Create visualization container
    this.container = document.createElement('div');
    this.container.style.cssText = `
      width: 100%;
      height: ${this.model.get('height') || 600}px;
      border: 1px solid #ddd;
      background: ${this.model.get('background_color') || '#ffffff'};
      position: relative;
      overflow: hidden;
      border-radius: 4px;
    `;
    this.el.appendChild(this.container);
    
    // Initialize unified core
    this.initializeVizCore();
    
    // Set up model change listeners
    this.setupModelListeners();
    
    // Initial data load
    this.updateData();
  }
  
  /**
   * 🧠 Initialize the unified visualization core
   */
  private initializeVizCore() {
    const config = {
      width: this.model.get('width') || 800,
      height: this.model.get('height') || 600,
      
      physics: {
        enabled: this.model.get('physics_enabled') !== false,
        forceStrength: this.model.get('force_strength') || 30,
        linkDistance: this.model.get('link_distance') || 50,
        linkStrength: this.model.get('link_strength') || 0.1,
        chargeStrength: this.model.get('charge_strength') || -300,
        centerStrength: this.model.get('center_strength') || 0.1
      },
      
      rendering: {
        backgroundColor: this.model.get('background_color') || '#ffffff',
        nodeColorScheme: this.model.get('node_color_scheme') || 'default',
        enableShadows: this.model.get('enable_shadows') !== false,
        enableAnimations: this.model.get('enable_animations') !== false
      },
      
      interaction: {
        enableDrag: this.model.get('enable_drag') !== false,
        enableZoom: this.model.get('enable_zoom') !== false,
        enablePan: this.model.get('enable_pan') !== false,
        enableSelection: this.model.get('enable_selection') !== false,
        enableHover: this.model.get('enable_hover') !== false
      }
    };
    
    // Create unified core
    this.vizCore = new GroggyVizCore([], [], config);
    
    // Set up event handlers
    this.setupCoreEventHandlers();
    
    // Attach to DOM
    if (this.container) {
      this.vizCore.attachToDOM(this.container);
    }
    
    console.log('🧠 GroggyVizCore initialized for Jupyter widget');
  }
  
  /**
   * 🎧 Set up event handlers for core → Jupyter synchronization
   */
  private setupCoreEventHandlers() {
    if (!this.vizCore) return;
    
    // Node interactions
    this.vizCore.on('nodeClick', (data: any) => {
      this.model.send({
        type: 'node_click',
        node_id: data.nodeId,
        event: {
          ctrlKey: data.event?.ctrlKey || false,
          metaKey: data.event?.metaKey || false,
          shiftKey: data.event?.shiftKey || false
        }
      });
    });
    
    this.vizCore.on('nodeHover', (data: any) => {
      this.model.set('hovered_node', data.nodeId || '');
      this.model.save_changes();
    });
    
    this.vizCore.on('selectionChanged', (data: any) => {
      this.model.set('selected_nodes', data.selectedNodes || []);
      this.model.save_changes();
    });
    
    // Physics and camera events
    this.vizCore.on('simulationStarted', () => {
      this.updateStatus('Physics simulation running...');
    });
    
    this.vizCore.on('simulationStopped', () => {
      this.updateStatus('Physics simulation stopped');
    });
    
    this.vizCore.on('zoom', (data: any) => {
      const camera = this.model.get('camera_position');
      camera.scale = data.scale;
      this.model.set('camera_position', camera);
      this.model.save_changes();
    });
    
    this.vizCore.on('pan', (data: any) => {
      const camera = this.model.get('camera_position');
      camera.translate.x += data.delta.x;
      camera.translate.y += data.delta.y;
      this.model.set('camera_position', camera);
      this.model.save_changes();
    });
    
    // Drag events
    this.vizCore.on('nodeDrag', (data: any) => {
      if (data.isDragging) {
        this.model.set('is_dragging', true);
        this.model.save_changes();
      }
    });
    
    this.vizCore.on('nodeDragEnd', () => {
      this.model.set('is_dragging', false);
      this.model.save_changes();
    });
    
    // Update events for performance monitoring
    this.vizCore.on('update', (data: any) => {
      if (data.metadata.frameCount % 60 === 0) { // Every 60 frames
        this.updateStatus(`FPS: ${Math.round(data.metadata.fps)} | Nodes: ${data.nodes.length}`);
      }
    });
  }
  
  /**
   * 🔄 Set up model change listeners for Jupyter → core synchronization
   */
  private setupModelListeners() {
    // Data changes
    this.model.on('change:nodes change:edges', () => {
      this.updateData();
    });
    
    // Configuration changes
    this.model.on('change:width change:height', () => {
      this.updateDimensions();
    });
    
    this.model.on('change:physics_enabled change:force_strength change:link_distance change:link_strength change:charge_strength change:center_strength', () => {
      this.updatePhysicsConfig();
    });
    
    this.model.on('change:background_color change:node_color_scheme change:enable_shadows change:enable_animations', () => {
      this.updateRenderingConfig();
    });
    
    this.model.on('change:enable_drag change:enable_zoom change:enable_pan change:enable_selection change:enable_hover', () => {
      this.updateInteractionConfig();
    });
    
    // Selection changes from Python
    this.model.on('change:selected_nodes', () => {
      if (this.vizCore) {
        const selectedNodes = this.model.get('selected_nodes') || [];
        // Update visual selection without triggering events
        this.vizCore.selectedNodes.clear();
        selectedNodes.forEach((nodeId: string) => {
          this.vizCore!.selectedNodes.add(nodeId);
        });
        this.vizCore.update(); // Force re-render
      }
    });
    
    // Title changes
    this.model.on('change:title', () => {
      const titleElement = this.el.querySelector('span');
      if (titleElement) {
        titleElement.textContent = this.model.get('title') || 'Graph Visualization';
      }
    });
  }
  
  /**
   * 📊 Update graph data
   */
  private updateData() {
    if (!this.vizCore) return;
    
    const nodes = this.model.get('nodes') || [];
    const edges = this.model.get('edges') || [];
    
    this.vizCore.setData(nodes, edges);
    
    console.log(`🧠 Updated data: ${nodes.length} nodes, ${edges.length} edges`);
  }
  
  /**
   * 📐 Update canvas dimensions
   */
  private updateDimensions() {
    if (!this.vizCore || !this.container) return;
    
    const width = this.model.get('width') || 800;
    const height = this.model.get('height') || 600;
    
    this.container.style.height = `${height}px`;
    
    this.vizCore.updateConfig({
      width,
      height
    });
    
    console.log(`🧠 Updated dimensions: ${width}x${height}`);
  }
  
  /**
   * ⚛️ Update physics configuration
   */
  private updatePhysicsConfig() {
    if (!this.vizCore) return;
    
    const physicsConfig = {
      enabled: this.model.get('physics_enabled') !== false,
      forceStrength: this.model.get('force_strength') || 30,
      linkDistance: this.model.get('link_distance') || 50,
      linkStrength: this.model.get('link_strength') || 0.1,
      chargeStrength: this.model.get('charge_strength') || -300,
      centerStrength: this.model.get('center_strength') || 0.1
    };
    
    this.vizCore.updateConfig({ physics: physicsConfig });
    
    console.log('🧠 Updated physics configuration');
  }
  
  /**
   * 🎨 Update rendering configuration
   */
  private updateRenderingConfig() {
    if (!this.vizCore || !this.container) return;
    
    const renderingConfig = {
      backgroundColor: this.model.get('background_color') || '#ffffff',
      nodeColorScheme: this.model.get('node_color_scheme') || 'default',
      enableShadows: this.model.get('enable_shadows') !== false,
      enableAnimations: this.model.get('enable_animations') !== false
    };
    
    this.vizCore.updateConfig({ rendering: renderingConfig });
    
    // Update container background
    this.container.style.background = renderingConfig.backgroundColor;
    
    console.log('🧠 Updated rendering configuration');
  }
  
  /**
   * 🖱️ Update interaction configuration
   */
  private updateInteractionConfig() {
    if (!this.vizCore) return;
    
    const interactionConfig = {
      enableDrag: this.model.get('enable_drag') !== false,
      enableZoom: this.model.get('enable_zoom') !== false,
      enablePan: this.model.get('enable_pan') !== false,
      enableSelection: this.model.get('enable_selection') !== false,
      enableHover: this.model.get('enable_hover') !== false
    };
    
    this.vizCore.updateConfig({ interaction: interactionConfig });
    
    console.log('🧠 Updated interaction configuration');
  }
  
  /**
   * 📊 Update status display
   */
  private updateStatus(text: string) {
    const statusElement = this.el.querySelector('span:last-child') as HTMLElement;
    if (statusElement) {
      statusElement.textContent = text;
    }
  }
  
  /**
   * 📨 Handle custom messages from Python
   */
  handleCustomMessage(content: any) {
    const { type } = content;
    
    switch (type) {
      case 'set_layout':
        // Layout changes would be handled by physics config updates
        this.updatePhysicsConfig();
        break;
        
      case 'focus_node':
        if (this.vizCore && content.node_id) {
          // Find node position and adjust camera
          const pos = this.vizCore.nodePositions.get(content.node_id);
          if (pos) {
            const zoom = content.zoom || 2.0;
            const centerX = this.vizCore.config.width / 2;
            const centerY = this.vizCore.config.height / 2;
            
            // Calculate transform to center node
            const translateX = centerX - pos.x * zoom;
            const translateY = centerY - pos.y * zoom;
            
            this.vizCore.updateConfig({
              camera: {
                scale: zoom,
                translate: { x: translateX, y: translateY }
              }
            });
          }
        }
        break;
        
      case 'reset_camera':
        if (this.vizCore) {
          this.vizCore.updateConfig({
            camera: {
              scale: 1,
              translate: { x: 0, y: 0 }
            }
          });
        }
        break;
        
      case 'export_state':
        if (this.vizCore) {
          const state = this.vizCore.exportState();
          this.model.send({
            type: 'state_exported',
            state: state
          });
        }
        break;
        
      default:
        console.warn(`🧠 Unknown message type: ${type}`);
    }
  }
  
  /**
   * 🧹 Cleanup when widget is destroyed
   */
  remove() {
    if (this.vizCore) {
      this.vizCore.destroy();
      this.vizCore = null;
    }
    
    super.remove();
    console.log('🧠 Unified Jupyter widget destroyed');
  }
}