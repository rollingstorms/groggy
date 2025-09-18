/**
 * Accessibility Tests for Groggy Visualization System
 * 
 * Tests keyboard navigation, screen reader support, and WCAG 2.1 AA compliance
 * for the graph visualization interface.
 */

describe('Accessibility Tests', () => {
    let renderer;
    let testGraph;
    let accessibilityUtils;
    
    beforeAll(async () => {
        // Initialize renderer with accessibility features enabled
        renderer = new GraphRenderer({
            canvas: document.createElement('canvas'),
            enableAccessibility: true,
            accessibilityConfig: {
                enableKeyboardNavigation: true,
                enableScreenReader: true,
                enableHighContrast: true,
                enableFocusTrapping: true,
                announceChanges: true
            }
        });
        
        // Create test graph for accessibility testing
        testGraph = TestGraphGenerator.createGraph({
            nodeCount: 20,
            edgeCount: 30,
            nodeLabels: true,
            edgeLabels: true,
            distribution: 'small-world'
        });
        
        // Utility for accessibility testing
        accessibilityUtils = new AccessibilityTestUtils();
        
        await renderer.loadGraph(testGraph);
        await renderer.render();
    });
    
    beforeEach(() => {
        // Reset focus and state before each test
        renderer.resetAccessibilityState();
        document.activeElement?.blur();
    });
    
    // =========================================================================
    // KEYBOARD NAVIGATION TESTS
    // =========================================================================
    
    describe('Keyboard Navigation', () => {
        test('Tab key navigates through focusable elements', async () => {
            const focusableElements = renderer.getFocusableElements();
            expect(focusableElements.length).toBeGreaterThan(0);
            
            // Simulate Tab navigation
            let currentIndex = 0;
            for (const element of focusableElements) {
                const tabEvent = new KeyboardEvent('keydown', {
                    key: 'Tab',
                    code: 'Tab',
                    keyCode: 9
                });
                
                document.dispatchEvent(tabEvent);
                await accessibilityUtils.waitForFocusChange();
                
                const focusedElement = renderer.getFocusedElement();
                expect(focusedElement).toBe(element);
                currentIndex++;
            }
            
            expect(currentIndex).toBe(focusableElements.length);
        });
        
        test('Shift+Tab navigates backwards through elements', async () => {
            const focusableElements = renderer.getFocusableElements();
            
            // Start at the last element
            renderer.setFocusedElement(focusableElements[focusableElements.length - 1]);
            
            // Navigate backwards with Shift+Tab
            for (let i = focusableElements.length - 2; i >= 0; i--) {
                const shiftTabEvent = new KeyboardEvent('keydown', {
                    key: 'Tab',
                    code: 'Tab',
                    keyCode: 9,
                    shiftKey: true
                });
                
                document.dispatchEvent(shiftTabEvent);
                await accessibilityUtils.waitForFocusChange();
                
                const focusedElement = renderer.getFocusedElement();
                expect(focusedElement).toBe(focusableElements[i]);
            }
        });
        
        test('Arrow keys navigate between graph nodes', async () => {
            const nodes = testGraph.nodes;
            
            // Focus first node
            renderer.focusNode(nodes[0].id);
            let currentNode = renderer.getFocusedNode();
            expect(currentNode.id).toBe(nodes[0].id);
            
            // Test arrow key navigation
            const arrowKeys = [
                { key: 'ArrowRight', direction: 'right' },
                { key: 'ArrowDown', direction: 'down' },
                { key: 'ArrowLeft', direction: 'left' },
                { key: 'ArrowUp', direction: 'up' }
            ];
            
            for (const arrow of arrowKeys) {
                const arrowEvent = new KeyboardEvent('keydown', {
                    key: arrow.key,
                    code: arrow.key,
                    keyCode: arrow.key === 'ArrowRight' ? 39 : 
                            arrow.key === 'ArrowDown' ? 40 :
                            arrow.key === 'ArrowLeft' ? 37 : 38
                });
                
                const oldNode = renderer.getFocusedNode();
                document.dispatchEvent(arrowEvent);
                await accessibilityUtils.waitForFocusChange();
                
                const newNode = renderer.getFocusedNode();
                
                // Should have moved to a different node
                expect(newNode.id).not.toBe(oldNode.id);
                
                // Should have moved in the correct spatial direction
                const moved = accessibilityUtils.verifyDirectionalMovement(
                    oldNode, newNode, arrow.direction
                );
                expect(moved).toBe(true);
            }
        });
        
        test('Enter key activates focused element', async () => {
            const testNode = testGraph.nodes[0];
            renderer.focusNode(testNode.id);
            
            const enterEvent = new KeyboardEvent('keydown', {
                key: 'Enter',
                code: 'Enter',
                keyCode: 13
            });
            
            const activationSpy = jest.spyOn(renderer, 'activateNode');
            document.dispatchEvent(enterEvent);
            
            expect(activationSpy).toHaveBeenCalledWith(testNode.id);
        });
        
        test('Space key selects/deselects focused element', async () => {
            const testNode = testGraph.nodes[0];
            renderer.focusNode(testNode.id);
            
            const spaceEvent = new KeyboardEvent('keydown', {
                key: ' ',
                code: 'Space',
                keyCode: 32
            });
            
            // First space - should select
            document.dispatchEvent(spaceEvent);
            expect(renderer.isNodeSelected(testNode.id)).toBe(true);
            
            // Second space - should deselect
            document.dispatchEvent(spaceEvent);
            expect(renderer.isNodeSelected(testNode.id)).toBe(false);
        });
        
        test('Escape key cancels current operation', async () => {
            // Start a multi-select operation
            renderer.startMultiSelect();
            expect(renderer.isInMultiSelectMode()).toBe(true);
            
            const escapeEvent = new KeyboardEvent('keydown', {
                key: 'Escape',
                code: 'Escape',
                keyCode: 27
            });
            
            document.dispatchEvent(escapeEvent);
            expect(renderer.isInMultiSelectMode()).toBe(false);
        });
        
        test('Ctrl+A selects all nodes', async () => {
            const selectAllEvent = new KeyboardEvent('keydown', {
                key: 'a',
                code: 'KeyA',
                keyCode: 65,
                ctrlKey: true
            });
            
            document.dispatchEvent(selectAllEvent);
            
            const selectedNodes = renderer.getSelectedNodes();
            expect(selectedNodes.length).toBe(testGraph.nodes.length);
        });
        
        test('Delete key removes selected elements', async () => {
            const testNode = testGraph.nodes[0];
            renderer.selectNode(testNode.id);
            
            const deleteEvent = new KeyboardEvent('keydown', {
                key: 'Delete',
                code: 'Delete',
                keyCode: 46
            });
            
            const deleteSpy = jest.spyOn(renderer, 'deleteSelectedElements');
            document.dispatchEvent(deleteEvent);
            
            expect(deleteSpy).toHaveBeenCalled();
        });
        
        test('Home/End keys navigate to first/last elements', async () => {
            const nodes = testGraph.nodes;
            
            // Press Home key
            const homeEvent = new KeyboardEvent('keydown', {
                key: 'Home',
                code: 'Home',
                keyCode: 36
            });
            
            document.dispatchEvent(homeEvent);
            await accessibilityUtils.waitForFocusChange();
            
            const firstFocused = renderer.getFocusedNode();
            expect(firstFocused.id).toBe(nodes[0].id);
            
            // Press End key
            const endEvent = new KeyboardEvent('keydown', {
                key: 'End',
                code: 'End',
                keyCode: 35
            });
            
            document.dispatchEvent(endEvent);
            await accessibilityUtils.waitForFocusChange();
            
            const lastFocused = renderer.getFocusedNode();
            expect(lastFocused.id).toBe(nodes[nodes.length - 1].id);
        });
    });
    
    // =========================================================================
    // SCREEN READER SUPPORT TESTS
    // =========================================================================
    
    describe('Screen Reader Support', () => {
        test('nodes have proper ARIA labels', () => {
            const nodeElements = renderer.getNodeElements();
            
            nodeElements.forEach((element, index) => {
                const node = testGraph.nodes[index];
                
                // Should have aria-label
                expect(element.getAttribute('aria-label')).toBeTruthy();
                
                // Should include node information
                const ariaLabel = element.getAttribute('aria-label');
                expect(ariaLabel).toContain(node.id);
                if (node.label) {
                    expect(ariaLabel).toContain(node.label);
                }
                
                // Should have role
                expect(element.getAttribute('role')).toBe('button');
                
                // Should be focusable
                expect(element.getAttribute('tabindex')).toBe('0');
            });
        });
        
        test('edges have proper ARIA descriptions', () => {
            const edgeElements = renderer.getEdgeElements();
            
            edgeElements.forEach((element, index) => {
                const edge = testGraph.edges[index];
                
                // Should have aria-label describing the connection
                const ariaLabel = element.getAttribute('aria-label');
                expect(ariaLabel).toBeTruthy();
                expect(ariaLabel).toContain(edge.source);
                expect(ariaLabel).toContain(edge.target);
                expect(ariaLabel).toContain('connects to');
            });
        });
        
        test('graph structure is announced to screen readers', () => {
            const graphContainer = renderer.getGraphContainer();
            
            // Should have aria-label with graph summary
            const ariaLabel = graphContainer.getAttribute('aria-label');
            expect(ariaLabel).toContain(`${testGraph.nodes.length} nodes`);
            expect(ariaLabel).toContain(`${testGraph.edges.length} edges`);
            
            // Should have proper role
            expect(graphContainer.getAttribute('role')).toBe('application');
            
            // Should have aria-description
            const ariaDescription = graphContainer.getAttribute('aria-description');
            expect(ariaDescription).toContain('interactive graph visualization');
        });
        
        test('live regions announce dynamic changes', async () => {
            const liveRegion = renderer.getLiveRegion();
            expect(liveRegion).toBeTruthy();
            expect(liveRegion.getAttribute('aria-live')).toBe('polite');
            
            // Add a new node and verify announcement
            const newNode = {
                id: 'new_node',
                x: 100,
                y: 100,
                label: 'New Node'
            };
            
            renderer.addNode(newNode);
            await accessibilityUtils.waitForLiveRegionUpdate();
            
            expect(liveRegion.textContent).toContain('Node added');
            expect(liveRegion.textContent).toContain('New Node');
        });
        
        test('focus changes are announced', async () => {
            const liveRegion = renderer.getLiveRegion();
            const testNode = testGraph.nodes[0];
            
            renderer.focusNode(testNode.id);
            await accessibilityUtils.waitForLiveRegionUpdate();
            
            expect(liveRegion.textContent).toContain('Focused on node');
            expect(liveRegion.textContent).toContain(testNode.id);
        });
        
        test('selection changes are announced', async () => {
            const liveRegion = renderer.getLiveRegion();
            const testNode = testGraph.nodes[0];
            
            renderer.selectNode(testNode.id);
            await accessibilityUtils.waitForLiveRegionUpdate();
            
            expect(liveRegion.textContent).toContain('Node selected');
            expect(liveRegion.textContent).toContain(testNode.id);
        });
        
        test('toolbar buttons have proper ARIA attributes', () => {
            const toolbar = renderer.getToolbar();
            const toolbarButtons = toolbar.querySelectorAll('button');
            
            toolbarButtons.forEach(button => {
                // Should have aria-label
                expect(button.getAttribute('aria-label')).toBeTruthy();
                
                // Should have proper role
                expect(button.getAttribute('role')).toBe('button');
                
                // Toggle buttons should have aria-pressed
                if (button.classList.contains('toggle')) {
                    expect(button.getAttribute('aria-pressed')).toMatch(/true|false/);
                }
                
                // Disabled buttons should have aria-disabled
                if (button.disabled) {
                    expect(button.getAttribute('aria-disabled')).toBe('true');
                }
            });
        });
    });
    
    // =========================================================================
    // HIGH CONTRAST AND VISUAL ACCESSIBILITY TESTS
    // =========================================================================
    
    describe('Visual Accessibility', () => {
        test('high contrast mode provides sufficient color contrast', () => {
            renderer.enableHighContrastMode();
            
            const nodeElements = renderer.getNodeElements();
            const edgeElements = renderer.getEdgeElements();
            
            // Test node contrast ratios
            nodeElements.forEach(element => {
                const styles = window.getComputedStyle(element);
                const backgroundColor = styles.backgroundColor;
                const borderColor = styles.borderColor;
                
                const contrastRatio = accessibilityUtils.calculateContrastRatio(
                    backgroundColor, borderColor
                );
                
                // Should meet WCAG AA standard (4.5:1)
                expect(contrastRatio).toBeGreaterThanOrEqual(4.5);
            });
            
            // Test edge contrast ratios
            edgeElements.forEach(element => {
                const styles = window.getComputedStyle(element);
                const strokeColor = styles.stroke;
                const backgroundColor = '#ffffff'; // Assuming white background
                
                const contrastRatio = accessibilityUtils.calculateContrastRatio(
                    strokeColor, backgroundColor
                );
                
                expect(contrastRatio).toBeGreaterThanOrEqual(3.0); // Lower standard for non-text
            });
        });
        
        test('focus indicators are clearly visible', () => {
            const testNode = testGraph.nodes[0];
            const nodeElement = renderer.getNodeElement(testNode.id);
            
            renderer.focusNode(testNode.id);
            
            const styles = window.getComputedStyle(nodeElement);
            
            // Should have visible focus outline
            expect(styles.outline).not.toBe('none');
            expect(styles.outlineWidth).not.toBe('0px');
            
            // Focus outline should have good contrast
            const outlineColor = styles.outlineColor;
            const backgroundColor = styles.backgroundColor;
            
            const contrastRatio = accessibilityUtils.calculateContrastRatio(
                outlineColor, backgroundColor
            );
            expect(contrastRatio).toBeGreaterThanOrEqual(3.0);
        });
        
        test('text labels are readable at all zoom levels', () => {
            const zoomLevels = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0];
            
            zoomLevels.forEach(zoom => {
                renderer.setZoom(zoom);
                
                const nodeElements = renderer.getNodeElements();
                
                nodeElements.forEach(element => {
                    const labelElement = element.querySelector('.node-label');
                    if (labelElement) {
                        const styles = window.getComputedStyle(labelElement);
                        const fontSize = parseFloat(styles.fontSize);
                        
                        // Font size should be at least 12px at all zoom levels
                        expect(fontSize).toBeGreaterThanOrEqual(12);
                        
                        // Should not be clipped or overlapping
                        const bounds = labelElement.getBoundingClientRect();
                        expect(bounds.width).toBeGreaterThan(0);
                        expect(bounds.height).toBeGreaterThan(0);
                    }
                });
            });
        });
        
        test('reduced motion mode respects user preferences', () => {
            // Mock reduced motion preference
            Object.defineProperty(window, 'matchMedia', {
                value: jest.fn(() => ({
                    matches: true, // Prefers reduced motion
                    addListener: jest.fn(),
                    removeListener: jest.fn()
                }))
            });
            
            renderer.applyAccessibilityPreferences();
            
            // Animations should be disabled or reduced
            const animationConfig = renderer.getAnimationConfig();
            expect(animationConfig.enableTransitions).toBe(false);
            expect(animationConfig.animationDuration).toBeLessThanOrEqual(200);
        });
    });
    
    // =========================================================================
    // FOCUS MANAGEMENT TESTS
    // =========================================================================
    
    describe('Focus Management', () => {
        test('focus is trapped within modal dialogs', async () => {
            // Open a context menu (modal dialog)
            const testNode = testGraph.nodes[0];
            renderer.showContextMenu(testNode.id, { x: 100, y: 100 });
            
            const contextMenu = renderer.getContextMenu();
            const focusableElements = contextMenu.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            
            expect(focusableElements.length).toBeGreaterThan(0);
            
            // Focus should be on first element
            expect(document.activeElement).toBe(focusableElements[0]);
            
            // Tab to last element
            for (let i = 1; i < focusableElements.length; i++) {
                const tabEvent = new KeyboardEvent('keydown', {
                    key: 'Tab',
                    code: 'Tab',
                    keyCode: 9
                });
                document.dispatchEvent(tabEvent);
            }
            
            expect(document.activeElement).toBe(focusableElements[focusableElements.length - 1]);
            
            // One more tab should wrap to first element
            const tabEvent = new KeyboardEvent('keydown', {
                key: 'Tab',
                code: 'Tab',
                keyCode: 9
            });
            document.dispatchEvent(tabEvent);
            
            expect(document.activeElement).toBe(focusableElements[0]);
        });
        
        test('focus returns to trigger element when dialog closes', async () => {
            const testNode = testGraph.nodes[0];
            const nodeElement = renderer.getNodeElement(testNode.id);
            
            // Focus on node and open context menu
            nodeElement.focus();
            renderer.showContextMenu(testNode.id, { x: 100, y: 100 });
            
            const contextMenu = renderer.getContextMenu();
            expect(contextMenu).toBeVisible();
            
            // Close context menu
            const escapeEvent = new KeyboardEvent('keydown', {
                key: 'Escape',
                code: 'Escape',
                keyCode: 27
            });
            document.dispatchEvent(escapeEvent);
            
            // Focus should return to the node
            await accessibilityUtils.waitForFocusChange();
            expect(document.activeElement).toBe(nodeElement);
        });
        
        test('skip links allow bypassing complex navigation', () => {
            const skipLinks = renderer.getSkipLinks();
            expect(skipLinks.length).toBeGreaterThan(0);
            
            const skipToGraphLink = skipLinks.find(link => 
                link.textContent.includes('Skip to graph')
            );
            expect(skipToGraphLink).toBeTruthy();
            
            // Clicking skip link should focus the graph
            skipToGraphLink.click();
            
            const graphContainer = renderer.getGraphContainer();
            expect(document.activeElement).toBe(graphContainer);
        });
    });
    
    // =========================================================================
    // WCAG COMPLIANCE TESTS
    // =========================================================================
    
    describe('WCAG 2.1 AA Compliance', () => {
        test('all interactive elements are keyboard accessible', () => {
            const interactiveElements = renderer.getAllInteractiveElements();
            
            interactiveElements.forEach(element => {
                // Should be focusable
                const tabIndex = element.getAttribute('tabindex');
                expect(tabIndex).not.toBe('-1');
                
                // Should have keyboard event handlers
                const hasKeyboardHandler = element.onkeydown !== null || 
                                         element.onkeyup !== null ||
                                         element.onkeypress !== null;
                expect(hasKeyboardHandler).toBe(true);
            });
        });
        
        test('all content is accessible to screen readers', () => {
            const allElements = renderer.getAllContentElements();
            
            allElements.forEach(element => {
                // Should have accessible name
                const accessibleName = accessibilityUtils.getAccessibleName(element);
                expect(accessibleName).toBeTruthy();
                
                // Should not be hidden from screen readers
                expect(element.getAttribute('aria-hidden')).not.toBe('true');
            });
        });
        
        test('timing is not essential for functionality', () => {
            // Test that auto-advance features can be paused
            const autoAdvanceControls = renderer.getAutoAdvanceControls();
            
            if (autoAdvanceControls.length > 0) {
                autoAdvanceControls.forEach(control => {
                    expect(control.querySelector('.pause-button')).toBeTruthy();
                    expect(control.querySelector('.speed-control')).toBeTruthy();
                });
            }
        });
        
        test('error messages are clearly communicated', async () => {
            // Trigger an error condition
            try {
                renderer.performInvalidOperation();
            } catch (error) {
                // Error should be announced to screen readers
                const liveRegion = renderer.getLiveRegion();
                await accessibilityUtils.waitForLiveRegionUpdate();
                
                expect(liveRegion.textContent).toContain('Error');
                
                // Error should be visually indicated
                const errorElements = document.querySelectorAll('[aria-invalid="true"]');
                expect(errorElements.length).toBeGreaterThan(0);
            }
        });
    });
    
    // =========================================================================
    // UTILITY CLASSES
    // =========================================================================
    
    /**
     * Accessibility testing utilities
     */
    class AccessibilityTestUtils {
        async waitForFocusChange(timeout = 1000) {
            return new Promise((resolve) => {
                const originalActiveElement = document.activeElement;
                let timeoutId;
                
                const checkFocus = () => {
                    if (document.activeElement !== originalActiveElement) {
                        clearTimeout(timeoutId);
                        resolve();
                    }
                };
                
                timeoutId = setTimeout(() => resolve(), timeout);
                
                // Check immediately and then every 10ms
                checkFocus();
                const intervalId = setInterval(() => {
                    checkFocus();
                    if (document.activeElement !== originalActiveElement) {
                        clearInterval(intervalId);
                    }
                }, 10);
                
                setTimeout(() => clearInterval(intervalId), timeout);
            });
        }
        
        async waitForLiveRegionUpdate(timeout = 1000) {
            return new Promise((resolve) => {
                const liveRegion = renderer.getLiveRegion();
                const originalContent = liveRegion.textContent;
                
                const observer = new MutationObserver(() => {
                    if (liveRegion.textContent !== originalContent) {
                        observer.disconnect();
                        resolve();
                    }
                });
                
                observer.observe(liveRegion, { 
                    childList: true, 
                    subtree: true, 
                    characterData: true 
                });
                
                setTimeout(() => {
                    observer.disconnect();
                    resolve();
                }, timeout);
            });
        }
        
        verifyDirectionalMovement(fromNode, toNode, direction) {
            const dx = toNode.x - fromNode.x;
            const dy = toNode.y - fromNode.y;
            
            switch (direction) {
                case 'right':
                    return dx > 0;
                case 'left':
                    return dx < 0;
                case 'down':
                    return dy > 0;
                case 'up':
                    return dy < 0;
                default:
                    return false;
            }
        }
        
        calculateContrastRatio(color1, color2) {
            // Convert colors to RGB values
            const rgb1 = this.parseColor(color1);
            const rgb2 = this.parseColor(color2);
            
            // Calculate relative luminance
            const l1 = this.getRelativeLuminance(rgb1);
            const l2 = this.getRelativeLuminance(rgb2);
            
            // Calculate contrast ratio
            const lighter = Math.max(l1, l2);
            const darker = Math.min(l1, l2);
            
            return (lighter + 0.05) / (darker + 0.05);
        }
        
        parseColor(color) {
            // Simple RGB color parser (extend as needed)
            const canvas = document.createElement('canvas');
            canvas.width = 1;
            canvas.height = 1;
            const ctx = canvas.getContext('2d');
            
            ctx.fillStyle = color;
            ctx.fillRect(0, 0, 1, 1);
            
            const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
            return { r, g, b };
        }
        
        getRelativeLuminance({ r, g, b }) {
            // Convert to 0-1 range
            r /= 255;
            g /= 255;
            b /= 255;
            
            // Apply gamma correction
            r = r <= 0.03928 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4);
            g = g <= 0.03928 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4);
            b = b <= 0.03928 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4);
            
            // Calculate luminance
            return 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
        
        getAccessibleName(element) {
            // Get the accessible name using ARIA computation rules
            return element.getAttribute('aria-label') ||
                   element.getAttribute('aria-labelledby') ||
                   element.getAttribute('title') ||
                   element.textContent?.trim() ||
                   element.getAttribute('alt') ||
                   '';
        }
    }
});

/**
 * Accessibility Test Configuration
 */
class AccessibilityTestConfig {
    static getWCAGRequirements() {
        return {
            // Color contrast ratios
            normalTextContrast: 4.5,     // AA standard
            largeTextContrast: 3.0,      // AA standard for large text
            nonTextContrast: 3.0,        // AA standard for UI components
            
            // Timing requirements
            maxAutoAdvanceTime: 20000,   // 20 seconds max before user control required
            
            // Focus requirements
            focusIndicatorMinWidth: 2,   // Minimum focus indicator width in pixels
            
            // Interactive target size
            minTargetSize: 44,           // Minimum touch target size in pixels
            
            // Animation requirements
            maxAnimationDuration: 5000,  // 5 seconds max for essential animations
            reducedMotionDuration: 200   // Max duration when reduced motion preferred
        };
    }
    
    static getKeyboardShortcuts() {
        return {
            'Tab': 'Next focusable element',
            'Shift+Tab': 'Previous focusable element',
            'Enter': 'Activate focused element',
            'Space': 'Select/deselect focused element',
            'Escape': 'Cancel current operation',
            'ArrowUp': 'Move focus up',
            'ArrowDown': 'Move focus down', 
            'ArrowLeft': 'Move focus left',
            'ArrowRight': 'Move focus right',
            'Home': 'Focus first element',
            'End': 'Focus last element',
            'Ctrl+A': 'Select all',
            'Delete': 'Delete selected elements',
            'F1': 'Show keyboard shortcuts help'
        };
    }
}