declare module "@fiftyone/operators" {
  export function usePanelEvent(): (
    methodName: string,
    options: {
      operator: string;
      params: Record<string, unknown>;
      callback: (result: unknown) => void;
    }
  ) => void;
}

declare module "@fiftyone/plugins" {
  export enum PluginComponentType {
    Component = "Component",
  }

  export function registerComponent(options: {
    name: string;
    component: unknown;
    type: PluginComponentType;
  }): void;
}
