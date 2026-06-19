"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const modelEditorProvider_1 = require("./modelEditorProvider");
function activate(context) {
    context.subscriptions.push(modelEditorProvider_1.ModelEditorProvider.register(context));
}
function deactivate() {
    // no-op
}
//# sourceMappingURL=extension.js.map