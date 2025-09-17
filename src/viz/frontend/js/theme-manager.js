/**
 * 🎨 Phase 9: Visual Design & Themes - ThemeManager
 * 
 * Comprehensive theme management system for Groggy Graph Visualization
 * Handles theme switching, responsive design, and visual transitions
 * 
 * Features:
 * - 5 built-in themes (Light, Dark, Publication, Minimal, Neon)
 * - Smooth theme transitions with animations
 * - Responsive design adaptation
 * - Theme persistence in localStorage
 * - Professional node/edge styling coordination
 * - Theme-aware component styling
 */

class ThemeManager {
    constructor() {
        this.currentTheme = 'light';
        this.previousTheme = null;
        this.isTransitioning = false;
        this.prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        // Theme definitions with comprehensive styling options
        this.themes = {
            'light': {
                name: 'Light',
                description: 'Clean, professional light theme',
                category: 'standard',
                accessibility: 'high',
                nodeStyles: {
                    defaultColor: '#007AFF',
                    selectedColor: '#FF3B30',
                    hoverColor: '#34C759',
                    strokeColor: '#FFFFFF',
                    strokeWidth: 2,
                    radius: 8,
                    fontSize: 12,
                    fontColor: '#000000'
                },
                edgeStyles: {
                    defaultColor: '#8E8E93',
                    selectedColor: '#FF3B30',
                    hoverColor: '#007AFF',
                    width: 2,
                    selectedWidth: 3,
                    opacity: 0.7,
                    selectedOpacity: 1.0
                },
                graphStyles: {
                    backgroundColor: '#F2F2F7',
                    gridColor: '#E5E5EA',
                    selectionColor: '#007AFF',
                    highlightColor: '#34C759'
                }
            },
            'dark': {
                name: 'Dark',
                description: 'Modern dark theme with blue accents',
                category: 'standard',
                accessibility: 'high',
                nodeStyles: {
                    defaultColor: '#0A84FF',
                    selectedColor: '#FF453A',
                    hoverColor: '#30D158',
                    strokeColor: '#1C1C1E',
                    strokeWidth: 2,
                    radius: 8,
                    fontSize: 12,
                    fontColor: '#FFFFFF'
                },
                edgeStyles: {
                    defaultColor: '#8E8E93',
                    selectedColor: '#FF453A',
                    hoverColor: '#0A84FF',
                    width: 2,
                    selectedWidth: 3,
                    opacity: 0.7,
                    selectedOpacity: 1.0
                },
                graphStyles: {
                    backgroundColor: '#1C1C1E',
                    gridColor: '#38383A',
                    selectionColor: '#0A84FF',
                    highlightColor: '#30D158'
                }
            },
            'publication': {
                name: 'Publication',
                description: 'Academic publication ready',
                category: 'professional',
                accessibility: 'very-high',
                nodeStyles: {
                    defaultColor: '#5E81AC',
                    selectedColor: '#BF616A',
                    hoverColor: '#A3BE8C',
                    strokeColor: '#2E3440',
                    strokeWidth: 1.5,
                    radius: 6,
                    fontSize: 10,
                    fontColor: '#2E3440'
                },
                edgeStyles: {
                    defaultColor: '#6B7280',
                    selectedColor: '#BF616A',
                    hoverColor: '#5E81AC',
                    width: 1.5,
                    selectedWidth: 2.5,
                    opacity: 0.8,
                    selectedOpacity: 1.0
                },
                graphStyles: {
                    backgroundColor: '#FAFAFA',
                    gridColor: '#E5E7EB',
                    selectionColor: '#5E81AC',
                    highlightColor: '#A3BE8C'
                }
            },
            'minimal': {
                name: 'Minimal',
                description: 'Ultra-clean minimal design',
                category: 'artistic',
                accessibility: 'medium',
                nodeStyles: {
                    defaultColor: '#000000',
                    selectedColor: '#000000',
                    hoverColor: '#666666',
                    strokeColor: '#FFFFFF',
                    strokeWidth: 1,
                    radius: 4,
                    fontSize: 10,
                    fontColor: '#000000'
                },
                edgeStyles: {
                    defaultColor: '#CCCCCC',
                    selectedColor: '#000000',
                    hoverColor: '#666666',
                    width: 1,
                    selectedWidth: 2,
                    opacity: 0.5,
                    selectedOpacity: 1.0
                },
                graphStyles: {
                    backgroundColor: '#FFFFFF',
                    gridColor: '#F0F0F0',
                    selectionColor: '#000000',
                    highlightColor: '#666666'
                }
            },
            'neon': {
                name: 'Neon',
                description: 'High-contrast cyberpunk aesthetic',
                category: 'artistic',
                accessibility: 'low',
                nodeStyles: {
                    defaultColor: '#00FFFF',
                    selectedColor: '#FF00FF',
                    hoverColor: '#00FF00',
                    strokeColor: '#0A0A0A',
                    strokeWidth: 2,
                    radius: 10,
                    fontSize: 12,
                    fontColor: '#00FFFF',
                    glowIntensity: 10
                },
                edgeStyles: {
                    defaultColor: '#666666',
                    selectedColor: '#FF00FF',
                    hoverColor: '#00FFFF',
                    width: 2,
                    selectedWidth: 4,
                    opacity: 0.8,
                    selectedOpacity: 1.0,
                    glowIntensity: 5
                },
                graphStyles: {
                    backgroundColor: '#0A0A0A',
                    gridColor: '#1A1A1A',
                    selectionColor: '#00FFFF',
                    highlightColor: '#00FF00'
                }
            }
        };
        
        // Responsive breakpoints
        this.breakpoints = {
            mobile: 768,
            tablet: 1024,
            desktop: 1440
        };
        
        // Animation settings
        this.animationSettings = {
            duration: 300,
            easing: 'ease-out',
            stagger: 50
        };
        
        this.init();
    }
    
    init() {
        console.log('🎨 Initializing ThemeManager for Phase 9');
        
        this.detectSystemPreference();
        this.loadSavedTheme();
        this.setupEventListeners();
        this.setupResponsiveHandlers();
        this.initializeThemeControls();
        
        // Apply initial theme
        this.applyTheme(this.currentTheme, false);
        
        console.log(`✅ ThemeManager initialized with theme: ${this.currentTheme}`);
    }
    
    /**
     * Detect system theme preference
     */
    detectSystemPreference() {
        const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
        this.prefersDarkMode = darkModeQuery.matches;
        
        // Listen for system theme changes
        darkModeQuery.addEventListener('change', (e) => {
            this.prefersDarkMode = e.matches;
            if (this.shouldFollowSystemTheme()) {
                this.applyTheme(e.matches ? 'dark' : 'light');
            }
        });
        
        console.log(`🌓 System preference: ${this.prefersDarkMode ? 'dark' : 'light'}`);
    }
    
    /**
     * Load theme from localStorage
     */
    loadSavedTheme() {
        try {
            const savedTheme = localStorage.getItem('groggy_theme');
            const savedSettings = localStorage.getItem('groggy_theme_settings');
            
            if (savedTheme && this.themes[savedTheme]) {
                this.currentTheme = savedTheme;
                console.log(`📂 Loaded saved theme: ${savedTheme}`);
            } else if (this.prefersDarkMode) {
                this.currentTheme = 'dark';
                console.log('🌙 Applied dark theme based on system preference');
            }
            
            if (savedSettings) {
                const settings = JSON.parse(savedSettings);
                this.animationSettings = { ...this.animationSettings, ...settings.animations };
            }
            
        } catch (error) {
            console.warn('⚠️  Failed to load saved theme:', error);
            this.currentTheme = this.prefersDarkMode ? 'dark' : 'light';
        }
    }
    
    /**
     * Set up event listeners for theme management
     */
    setupEventListeners() {
        // Theme selector in header
        const themeSelect = document.getElementById('theme-select');
        if (themeSelect) {
            themeSelect.value = this.currentTheme;
            themeSelect.addEventListener('change', (e) => {
                this.applyTheme(e.target.value);
            });
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case '1':
                        e.preventDefault();
                        this.applyTheme('light');
                        break;
                    case '2':
                        e.preventDefault();
                        this.applyTheme('dark');
                        break;
                    case '3':
                        e.preventDefault();
                        this.applyTheme('publication');
                        break;
                    case '4':
                        e.preventDefault();
                        this.applyTheme('minimal');
                        break;
                    case '5':
                        e.preventDefault();
                        this.applyTheme('neon');
                        break;
                }
            }
        });
        
        // Theme toggle button (if exists)
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }
        
        console.log('🎛️  Theme event listeners set up');
    }
    
    /**
     * Set up responsive design handlers
     */
    setupResponsiveHandlers() {
        // Window resize handler
        window.addEventListener('resize', () => {
            this.handleResponsiveChanges();
        });
        
        // Orientation change handler
        window.addEventListener('orientationchange', () => {
            setTimeout(() => {
                this.handleResponsiveChanges();
            }, 100);
        });
        
        // Initial responsive setup
        this.handleResponsiveChanges();
        
        console.log('📱 Responsive handlers set up');
    }
    
    /**
     * Initialize theme controls and indicators
     */
    initializeThemeControls() {
        this.createThemeInfoPanel();
        this.createThemeShortcutsHelp();
        this.updateThemeIndicators();
    }
    
    /**
     * Apply a theme with smooth transitions
     */
    applyTheme(themeName, animate = true) {
        if (!this.themes[themeName]) {
            console.error(`❌ Unknown theme: ${themeName}`);
            return false;
        }
        
        if (this.isTransitioning) {
            console.log('⏳ Theme transition in progress, skipping');
            return false;
        }
        
        console.log(`🎨 Applying theme: ${themeName}`);
        
        this.previousTheme = this.currentTheme;
        this.currentTheme = themeName;
        
        if (animate) {
            this.animateThemeTransition(themeName);
        } else {
            this.applyThemeStyles(themeName);
        }
        
        this.saveTheme();
        this.updateThemeIndicators();
        this.notifyThemeChange();
        
        return true;
    }
    
    /**
     * Animate theme transition
     */
    animateThemeTransition(themeName) {
        this.isTransitioning = true;
        
        // Add transition class to document
        document.documentElement.classList.add('theme-transitioning');
        
        // Apply the new theme
        setTimeout(() => {
            this.applyThemeStyles(themeName);
        }, 50);
        
        // Remove transition class after animation
        setTimeout(() => {
            document.documentElement.classList.remove('theme-transitioning');
            this.isTransitioning = false;
        }, this.animationSettings.duration);
    }
    
    /**
     * Apply theme styles to the document
     */
    applyThemeStyles(themeName) {
        const theme = this.themes[themeName];
        
        // Set data-theme attribute on document
        document.documentElement.setAttribute('data-theme', themeName);
        
        // Update theme selector
        const themeSelect = document.getElementById('theme-select');
        if (themeSelect) {
            themeSelect.value = themeName;
        }
        
        // Apply graph-specific styles
        this.applyGraphStyles(theme);
        
        // Handle theme-specific animations
        this.applyThemeAnimations(theme);
        
        console.log(`✅ Applied ${theme.name} theme styles`);
    }
    
    /**
     * Apply graph-specific styling
     */
    applyGraphStyles(theme) {
        // Update canvas background if canvas exists
        const canvas = document.getElementById('graph-canvas');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            if (ctx) {
                // This would be called by the graph renderer
                this.notifyGraphRenderer(theme);
            }
        }
        
        // Update CSS custom properties for graph elements
        const root = document.documentElement;
        const styles = theme.graphStyles;
        
        Object.entries(styles).forEach(([property, value]) => {
            root.style.setProperty(`--graph-${property.replace(/([A-Z])/g, '-$1').toLowerCase()}`, value);
        });
    }
    
    /**
     * Apply theme-specific animations
     */
    applyThemeAnimations(theme) {
        if (theme.category === 'artistic') {
            // Add special effects for artistic themes
            if (this.currentTheme === 'neon') {
                this.enableNeonEffects();
            } else if (this.currentTheme === 'minimal') {
                this.enableMinimalEffects();
            }
        } else {
            this.disableSpecialEffects();
        }
    }
    
    /**
     * Enable neon theme special effects
     */
    enableNeonEffects() {
        document.documentElement.classList.add('neon-effects');
        
        // Add glow effects to interactive elements
        const buttons = document.querySelectorAll('.control-button');
        buttons.forEach(button => {
            button.style.filter = 'drop-shadow(0 0 5px currentColor)';
        });
    }
    
    /**
     * Enable minimal theme effects
     */
    enableMinimalEffects() {
        document.documentElement.classList.add('minimal-effects');
        
        // Reduce visual noise
        const decorativeElements = document.querySelectorAll('.decorative');
        decorativeElements.forEach(el => {
            el.style.display = 'none';
        });
    }
    
    /**
     * Disable special effects
     */
    disableSpecialEffects() {
        document.documentElement.classList.remove('neon-effects', 'minimal-effects');
        
        // Reset filters
        const buttons = document.querySelectorAll('.control-button');
        buttons.forEach(button => {
            button.style.filter = '';
        });
        
        // Show decorative elements
        const decorativeElements = document.querySelectorAll('.decorative');
        decorativeElements.forEach(el => {
            el.style.display = '';
        });
    }
    
    /**
     * Handle responsive design changes
     */
    handleResponsiveChanges() {
        const width = window.innerWidth;
        let newBreakpoint;
        
        if (width <= this.breakpoints.mobile) {
            newBreakpoint = 'mobile';
        } else if (width <= this.breakpoints.tablet) {
            newBreakpoint = 'tablet';
        } else {
            newBreakpoint = 'desktop';
        }
        
        if (this.currentBreakpoint !== newBreakpoint) {
            this.currentBreakpoint = newBreakpoint;
            this.applyResponsiveStyles();
            console.log(`📱 Responsive breakpoint: ${newBreakpoint}`);
        }
    }
    
    /**
     * Apply responsive styles
     */
    applyResponsiveStyles() {
        const root = document.documentElement;
        root.setAttribute('data-breakpoint', this.currentBreakpoint);
        
        // Handle sidebar behavior on mobile
        if (this.currentBreakpoint === 'mobile') {
            this.enableMobileSidebar();
        } else {
            this.disableMobileSidebar();
        }
        
        // Adjust theme for small screens
        if (this.currentBreakpoint === 'mobile' && this.currentTheme === 'neon') {
            // Reduce neon intensity on mobile for better readability
            document.documentElement.style.setProperty('--neon-intensity', '0.5');
        } else {
            document.documentElement.style.setProperty('--neon-intensity', '1');
        }
    }
    
    /**
     * Enable mobile sidebar behavior
     */
    enableMobileSidebar() {
        const sidebar = document.querySelector('.viz-sidebar');
        const toggleButton = document.getElementById('toggle-sidebar');
        
        if (sidebar && toggleButton) {
            // Close sidebar by default on mobile
            sidebar.classList.remove('open');
            
            // Update toggle button behavior
            toggleButton.onclick = () => {
                sidebar.classList.toggle('open');
                
                // Add overlay
                if (sidebar.classList.contains('open')) {
                    this.addMobileOverlay();
                } else {
                    this.removeMobileOverlay();
                }
            };
        }
    }
    
    /**
     * Disable mobile sidebar behavior
     */
    disableMobileSidebar() {
        const sidebar = document.querySelector('.viz-sidebar');
        const toggleButton = document.getElementById('toggle-sidebar');
        
        if (sidebar) {
            sidebar.classList.remove('open');
            this.removeMobileOverlay();
        }
        
        if (toggleButton) {
            toggleButton.onclick = () => {
                sidebar.classList.toggle('collapsed');
            };
        }
    }
    
    /**
     * Add mobile overlay for sidebar
     */
    addMobileOverlay() {
        if (document.querySelector('.mobile-overlay')) return;
        
        const overlay = document.createElement('div');
        overlay.className = 'mobile-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 850;
            backdrop-filter: blur(4px);
        `;
        
        overlay.addEventListener('click', () => {
            document.querySelector('.viz-sidebar').classList.remove('open');
            this.removeMobileOverlay();
        });
        
        document.body.appendChild(overlay);
    }
    
    /**
     * Remove mobile overlay
     */
    removeMobileOverlay() {
        const overlay = document.querySelector('.mobile-overlay');
        if (overlay) {
            overlay.remove();
        }
    }
    
    /**
     * Toggle between light and dark theme
     */
    toggleTheme() {
        const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.applyTheme(newTheme);
    }
    
    /**
     * Cycle through all themes
     */
    cycleTheme() {
        const themeNames = Object.keys(this.themes);
        const currentIndex = themeNames.indexOf(this.currentTheme);
        const nextIndex = (currentIndex + 1) % themeNames.length;
        this.applyTheme(themeNames[nextIndex]);
    }
    
    /**
     * Create theme information panel
     */
    createThemeInfoPanel() {
        const existingPanel = document.getElementById('theme-info-panel');
        if (existingPanel) return;
        
        const panel = document.createElement('div');
        panel.id = 'theme-info-panel';
        panel.className = 'theme-info-panel';
        panel.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background-color: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: var(--border-radius);
            padding: 12px;
            font-size: 12px;
            color: var(--color-text-secondary);
            z-index: var(--z-tooltip);
            opacity: 0;
            transform: translateX(20px);
            transition: all var(--transition-normal);
            pointer-events: none;
        `;
        
        document.body.appendChild(panel);
    }
    
    /**
     * Create theme shortcuts help
     */
    createThemeShortcutsHelp() {
        // Add to help system or create tooltip
        const shortcuts = {
            'Ctrl/Cmd + 1': 'Light Theme',
            'Ctrl/Cmd + 2': 'Dark Theme',
            'Ctrl/Cmd + 3': 'Publication Theme',
            'Ctrl/Cmd + 4': 'Minimal Theme',
            'Ctrl/Cmd + 5': 'Neon Theme'
        };
        
        // Store shortcuts for help system
        this.themeShortcuts = shortcuts;
    }
    
    /**
     * Update theme indicators in UI
     */
    updateThemeIndicators() {
        const theme = this.themes[this.currentTheme];
        
        // Update theme info panel
        const panel = document.getElementById('theme-info-panel');
        if (panel) {
            panel.innerHTML = `
                <strong>${theme.name}</strong><br>
                ${theme.description}<br>
                <small>Accessibility: ${theme.accessibility}</small>
            `;
        }
        
        // Update document title to include theme
        const baseTitle = document.title.split(' - ')[0];
        document.title = `${baseTitle} - ${theme.name} Theme`;
        
        // Update meta theme color for mobile browsers
        let metaThemeColor = document.querySelector('meta[name="theme-color"]');
        if (!metaThemeColor) {
            metaThemeColor = document.createElement('meta');
            metaThemeColor.name = 'theme-color';
            document.head.appendChild(metaThemeColor);
        }
        
        const themeColor = getComputedStyle(document.documentElement)
            .getPropertyValue('--color-primary').trim();
        metaThemeColor.content = themeColor;
    }
    
    /**
     * Save current theme to localStorage
     */
    saveTheme() {
        try {
            localStorage.setItem('groggy_theme', this.currentTheme);
            localStorage.setItem('groggy_theme_settings', JSON.stringify({
                animations: this.animationSettings,
                timestamp: Date.now()
            }));
            console.log(`💾 Saved theme: ${this.currentTheme}`);
        } catch (error) {
            console.warn('⚠️  Failed to save theme:', error);
        }
    }
    
    /**
     * Check if should follow system theme
     */
    shouldFollowSystemTheme() {
        // Only follow system theme if user hasn't explicitly set a preference
        return !localStorage.getItem('groggy_theme');
    }
    
    /**
     * Notify graph renderer of theme change
     */
    notifyGraphRenderer(theme) {
        // Dispatch custom event for graph renderer
        const event = new CustomEvent('themeChanged', {
            detail: {
                theme: this.currentTheme,
                nodeStyles: theme.nodeStyles,
                edgeStyles: theme.edgeStyles,
                graphStyles: theme.graphStyles
            }
        });
        
        document.dispatchEvent(event);
    }
    
    /**
     * Notify other components of theme change
     */
    notifyThemeChange() {
        const event = new CustomEvent('groggyThemeChanged', {
            detail: {
                currentTheme: this.currentTheme,
                previousTheme: this.previousTheme,
                themeData: this.themes[this.currentTheme]
            }
        });
        
        document.dispatchEvent(event);
        console.log(`📢 Theme change notification sent: ${this.currentTheme}`);
    }
    
    /**
     * Get current theme data
     */
    getCurrentTheme() {
        return {
            name: this.currentTheme,
            data: this.themes[this.currentTheme]
        };
    }
    
    /**
     * Get all available themes
     */
    getAvailableThemes() {
        return Object.entries(this.themes).map(([key, theme]) => ({
            key,
            name: theme.name,
            description: theme.description,
            category: theme.category,
            accessibility: theme.accessibility
        }));
    }
    
    /**
     * Check if theme supports feature
     */
    supportsFeature(feature) {
        const theme = this.themes[this.currentTheme];
        
        switch (feature) {
            case 'animations':
                return theme.category !== 'minimal';
            case 'glow-effects':
                return theme.category === 'artistic';
            case 'high-contrast':
                return theme.accessibility === 'very-high';
            default:
                return true;
        }
    }
    
    /**
     * Export theme configuration
     */
    exportThemeConfig() {
        return {
            currentTheme: this.currentTheme,
            customSettings: this.animationSettings,
            responsive: {
                currentBreakpoint: this.currentBreakpoint,
                prefersDarkMode: this.prefersDarkMode
            },
            timestamp: Date.now()
        };
    }
    
    /**
     * Import theme configuration
     */
    importThemeConfig(config) {
        try {
            if (config.currentTheme && this.themes[config.currentTheme]) {
                this.applyTheme(config.currentTheme);
            }
            
            if (config.customSettings) {
                this.animationSettings = { ...this.animationSettings, ...config.customSettings };
            }
            
            console.log('✅ Imported theme configuration');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to import theme configuration:', error);
            return false;
        }
    }
    
    /**
     * Show theme info temporarily
     */
    showThemeInfo(duration = 3000) {
        const panel = document.getElementById('theme-info-panel');
        if (panel) {
            panel.style.opacity = '1';
            panel.style.transform = 'translateX(0)';
            
            setTimeout(() => {
                panel.style.opacity = '0';
                panel.style.transform = 'translateX(20px)';
            }, duration);
        }
    }
    
    /**
     * Cleanup method
     */
    destroy() {
        // Remove event listeners
        window.removeEventListener('resize', this.handleResponsiveChanges);
        window.removeEventListener('orientationchange', this.handleResponsiveChanges);
        
        // Remove theme info panel
        const panel = document.getElementById('theme-info-panel');
        if (panel) {
            panel.remove();
        }
        
        // Remove mobile overlay
        this.removeMobileOverlay();
        
        console.log('🧹 ThemeManager cleaned up');
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThemeManager;
}