import { DOMWidgetModel, DOMWidgetView } from '@jupyter-widgets/base';

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
    };
  }
}

export class GroggyGraphView extends DOMWidgetView {
  static view_name = 'GroggyGraphView';
  static view_module = MODULE_NAME;
  static view_module_version = MODULE_VERSION;
  
  private nodePositions = new Map<string, {x: number, y: number}>();
  private nodeVelocities = new Map<string, {vx: number, vy: number}>();
  private isDragging = false;
  private draggedNode: string | null = null;
  private dragOffset = {x: 0, y: 0};
  private animationId: number | null = null;
  private simulation = {
    alpha: 1.0,
    alphaDecay: 0.0228,
    alphaMin: 0.001,
    velocityDecay: 0.4,
    forceStrength: 30,
    linkDistance: 50,
    linkStrength: 0.1,
    chargeStrength: -300,
    centerStrength: 0.1
  };

  render() {
    // Create container
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
    
    // Create a header
    const header = document.createElement('div');
    header.style.cssText = `
      font-weight: bold;
      margin-bottom: 10px;
      color: #333;
    `;
    header.textContent = this.model.get('title') || 'Graph Visualization';
    this.el.appendChild(header);
    
    // Create graph container
    const graphContainer = document.createElement('div');
    graphContainer.style.cssText = `
      width: 100%;
      height: 350px;
      border: 1px solid #ddd;
      background: white;
      position: relative;
      overflow: hidden;
      cursor: grab;
    `;
    this.el.appendChild(graphContainer);
    
    // Render basic graph visualization
    this.renderGraph(graphContainer);
    
    // Listen for model changes
    this.model.on('change', this.onModelChange, this);
  }
  
  private renderGraph(container: HTMLElement) {
    const nodes = this.model.get('nodes') || [];
    const edges = this.model.get('edges') || [];
    
    // Clear container
    container.innerHTML = '';
    
    if (nodes.length === 0) {
      container.innerHTML = `
        <div style="
          display: flex; 
          align-items: center; 
          justify-content: center; 
          height: 100%; 
          color: #666;
          font-style: italic;
        ">
          No graph data provided. Add nodes and edges to see visualization.
        </div>
      `;
      return;
    }
    
    // Create simple SVG visualization
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.style.cursor = 'grab';
    container.appendChild(svg);
    
    // Initialize force-directed layout
    const centerX = container.clientWidth / 2;
    const centerY = container.clientHeight / 2;
    
    // Initialize positions and velocities (if not already positioned)
    if (this.nodePositions.size === 0) {
      nodes.forEach((node: any) => {
        // Random initial positions near center
        const angle = Math.random() * 2 * Math.PI;
        const radius = Math.random() * 100;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        
        this.nodePositions.set(node.id, { x, y });
        this.nodeVelocities.set(node.id, { vx: 0, vy: 0 });
      });
      
      // Start physics simulation
      this.startForceSimulation(nodes, edges, centerX, centerY);
    }
    
    // Draw edges first (so they appear behind nodes)
    const edgeElements: SVGLineElement[] = [];
    edges.forEach((edge: any) => {
      const sourcePos = this.nodePositions.get(edge.source);
      const targetPos = this.nodePositions.get(edge.target);
      
      if (sourcePos && targetPos) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', sourcePos.x.toString());
        line.setAttribute('y1', sourcePos.y.toString());
        line.setAttribute('x2', targetPos.x.toString());
        line.setAttribute('y2', targetPos.y.toString());
        line.setAttribute('stroke', '#999');
        line.setAttribute('stroke-width', '2');
        line.setAttribute('opacity', '0.6');
        svg.appendChild(line);
        edgeElements.push(line);
      }
    });
    
    // Draw nodes with drag functionality
    nodes.forEach((node: any) => {
      const pos = this.nodePositions.get(node.id);
      if (!pos) return;
      
      // Create node group for easier manipulation
      const nodeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      nodeGroup.style.cursor = 'grab';
      
      // Node circle
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', pos.x.toString());
      circle.setAttribute('cy', pos.y.toString());
      circle.setAttribute('r', (node.size || 20).toString());
      circle.setAttribute('fill', this.getNodeColor(node.group));
      circle.setAttribute('stroke', '#333');
      circle.setAttribute('stroke-width', '2');
      circle.style.filter = 'drop-shadow(2px 2px 4px rgba(0,0,0,0.3))';
      
      // Node label
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', pos.x.toString());
      text.setAttribute('y', (pos.y + 5).toString());
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('font-family', 'Arial, sans-serif');
      text.setAttribute('font-size', '12');
      text.setAttribute('fill', '#333');
      text.setAttribute('pointer-events', 'none'); // Don't interfere with dragging
      text.textContent = node.label || node.id;
      
      nodeGroup.appendChild(circle);
      nodeGroup.appendChild(text);
      svg.appendChild(nodeGroup);
      
      // Add drag functionality
      this.addDragBehavior(nodeGroup, node, circle, text, edgeElements, edges);
    });
    
    // Add stats
    const stats = document.createElement('div');
    stats.style.cssText = `
      position: absolute;
      top: 5px;
      right: 5px;
      background: rgba(255,255,255,0.9);
      padding: 5px;
      border-radius: 3px;
      font-size: 12px;
      color: #666;
    `;
    stats.innerHTML = `${nodes.length} nodes, ${edges.length} edges<br><small>Drag nodes to reposition</small>`;
    container.appendChild(stats);
  }
  
  private addDragBehavior(nodeGroup: SVGGElement, node: any, circle: SVGCircleElement, text: SVGTextElement, edgeElements: SVGLineElement[], edges: any[]) {
    let startX = 0, startY = 0;
    let isDragging = false;
    
    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      nodeGroup.style.cursor = 'grabbing';
      circle.setAttribute('stroke-width', '3');
      e.preventDefault();
    };
    
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      
      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;
      
      const currentPos = this.nodePositions.get(node.id);
      const newX = currentPos.x + deltaX;
      const newY = currentPos.y + deltaY;
      
      // Update node position
      this.nodePositions.set(node.id, { x: newX, y: newY });
      
      // Update visual elements
      circle.setAttribute('cx', newX.toString());
      circle.setAttribute('cy', newY.toString());
      text.setAttribute('x', newX.toString());
      text.setAttribute('y', (newY + 5).toString());
      
      // Update connected edges
      this.updateEdges(node.id, edgeElements, edges);
      
      startX = e.clientX;
      startY = e.clientY;
    };
    
    const onMouseUp = () => {
      isDragging = false;
      nodeGroup.style.cursor = 'grab';
      circle.setAttribute('stroke-width', '2');
    };
    
    // Mouse events
    nodeGroup.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    
    // Touch events for mobile
    nodeGroup.addEventListener('touchstart', (e) => {
      const touch = e.touches[0];
      onMouseDown({ clientX: touch.clientX, clientY: touch.clientY, preventDefault: () => e.preventDefault() } as any);
    });
    
    document.addEventListener('touchmove', (e) => {
      if (isDragging) {
        const touch = e.touches[0];
        onMouseMove({ clientX: touch.clientX, clientY: touch.clientY } as any);
        e.preventDefault();
      }
    });
    
    document.addEventListener('touchend', onMouseUp);
    
    // Hover effects
    nodeGroup.addEventListener('mouseenter', () => {
      if (!isDragging) {
        circle.setAttribute('opacity', '0.8');
        circle.style.filter = 'drop-shadow(3px 3px 6px rgba(0,0,0,0.4))';
      }
    });
    
    nodeGroup.addEventListener('mouseleave', () => {
      if (!isDragging) {
        circle.setAttribute('opacity', '1');
        circle.style.filter = 'drop-shadow(2px 2px 4px rgba(0,0,0,0.3))';
      }
    });
  }
  
  private updateEdges(nodeId: string, edgeElements: SVGLineElement[], edges: any[]) {
    edges.forEach((edge: any, index: number) => {
      if (edge.source === nodeId || edge.target === nodeId) {
        const sourcePos = this.nodePositions.get(edge.source);
        const targetPos = this.nodePositions.get(edge.target);
        
        if (sourcePos && targetPos && edgeElements[index]) {
          edgeElements[index].setAttribute('x1', sourcePos.x.toString());
          edgeElements[index].setAttribute('y1', sourcePos.y.toString());
          edgeElements[index].setAttribute('x2', targetPos.x.toString());
          edgeElements[index].setAttribute('y2', targetPos.y.toString());
        }
      }
    });
  }
  
  private getNodeColor(group: string): string {
    const colors: { [key: string]: string } = {
      'person': '#FF6B6B',
      'team': '#4ECDC4', 
      'project': '#45B7D1',
      'executive': '#96CEB4',
      'manager': '#FFEAA7',
      'lead': '#DDA0DD',
      'developer': '#98D8C8',
      'product': '#F7DC6F',
      'application': '#85C1E9',
      'service': '#F8C471',
      'storage': '#82E0AA',
      'infrastructure': '#D7BDE2',
      'core': '#F1948A',
      'satellite': '#AED6F1'
    };
    return colors[group] || '#BDC3C7';
  }
  
  private onModelChange() {
    // Re-render when model data changes
    const graphContainer = this.el.querySelector('div:last-child') as HTMLElement;
    if (graphContainer) {
      this.renderGraph(graphContainer);
    }
    
    // Update title
    const header = this.el.querySelector('div:first-child') as HTMLElement;
    if (header) {
      header.textContent = this.model.get('title') || 'Graph Visualization';
    }
  }
}