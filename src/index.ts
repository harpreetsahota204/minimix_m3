import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import MiniMaxChatPanel from "./MiniMaxChatPanel";

/**
 * Register the MiniMaxChatPanel React component.
 *
 * The ``name`` must match the ``component`` kwarg in
 * ``MiniMaxChatPanel.render()`` in ``chat_panel.py``.
 */
registerComponent({
  name: "MiniMaxChatPanel",
  component: MiniMaxChatPanel,
  type: PluginComponentType.Component,
});
