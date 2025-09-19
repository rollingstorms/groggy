/**
 * Global type declarations for Jupyter widget development
 */

// AMD module system types
declare const define: {
    (deps: string[], callback: (...args: any[]) => any): void;
    amd?: boolean;
};

declare const require: {
    (module: string): any;
    (deps: string[], callback: (...args: any[]) => any): void;
};

// Browser timeout types (override Node.js types)
declare function setTimeout(callback: () => void, ms: number): any;
declare function clearTimeout(timeoutId: any): void;

// Widget view types
interface WidgetView {
    model: any;
    el: HTMLElement;
    render(): void;
    remove(): void;
}

// Global widget registration
declare const widgets: {
    registerWidget(config: {
        model_name: string;
        model_module: string;
        model_module_version: string;
        view_name: string;
        view_module: string;
        view_module_version: string;
        model: any;
        view: any;
    }): void;
};