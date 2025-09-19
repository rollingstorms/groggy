import { IJupyterWidgetRegistry, DOMWidgetModel, DOMWidgetView } from '@jupyter-widgets/base';
import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { GroggyGraphModel, GroggyGraphView, MODULE_NAME, MODULE_VERSION } from './widget';

function activate(app: JupyterFrontEnd, registry: IJupyterWidgetRegistry) {
  console.log('[groggy-widgets] export shape fix - checking prototype chain');
  console.log('types:', typeof GroggyGraphModel, typeof GroggyGraphView); // expect function,function
  console.log('is Model subclass:', GroggyGraphModel.prototype instanceof DOMWidgetModel);
  console.log('is View  subclass:', GroggyGraphView.prototype  instanceof DOMWidgetView);

  registry.registerWidget({
    name: MODULE_NAME,
    version: MODULE_VERSION,
    exports: { GroggyGraphModel, GroggyGraphView },
  });
  
  console.log('✅ Export shape fix: registered actual GroggyGraphView');
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'groggy-widgets:plugin',
  requires: [IJupyterWidgetRegistry as any],
  autoStart: true,
  activate,
};

export default plugin;