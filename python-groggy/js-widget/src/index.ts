// src/index.ts

// Export original widget for backward compatibility
export { 
    GroggyGraphModel as GroggyGraphModelLegacy, 
    GroggyGraphView as GroggyGraphViewLegacy,
    MODULE_NAME as MODULE_NAME_LEGACY,
    MODULE_VERSION as MODULE_VERSION_LEGACY
} from './widget';

// Export unified widget (recommended for new usage)
export { 
    GroggyGraphModel, 
    GroggyGraphView,
    MODULE_NAME,
    MODULE_VERSION
} from './widget_unified';

// Export core components for advanced usage
export * from './core/index.js';

export { default } from './plugin'; // default export is only the plugin