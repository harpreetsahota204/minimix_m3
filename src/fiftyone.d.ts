declare module "@fiftyone/operators" {
  export function usePanelEvent(): (
    methodName: string,
    options: {
      operator: string;
      params: Record<string, unknown>;
      callback: (result: unknown) => void;
    }
  ) => void;

  export interface OperatorResult {
    result?: (Record<string, unknown> & { error?: string }) | null;
    error?: unknown;
    errorMessage?: string;
  }

  export interface OperatorExecutor {
    execute: (
      params: Record<string, unknown>,
      options?: { callback?: (result: OperatorResult) => void }
    ) => void;
    isExecuting?: boolean;
  }

  export function useOperatorExecutor(
    uri: string,
    handlers?: unknown
  ): OperatorExecutor;
}

declare module "@fiftyone/state" {
  /** The App's per-sample JSON (what the modal looker renders). */
  export type AppSample = Record<string, unknown> & { _id: string };

  export interface ModalSample {
    id: string;
    sample: AppSample;
    urls?: unknown;
  }

  /** Current modal sample, or undefined while loading / when closed. */
  export function useModalSample(): ModalSample | undefined;

  /**
   * Refresh a sample in both the modal and the grid in place (updates the
   * cached sample and bumps the refresher). Does not close the modal.
   */
  export function useRefreshSample(): (sample: AppSample) => void;
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
