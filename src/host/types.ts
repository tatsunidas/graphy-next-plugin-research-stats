/**
 * このプラグインが使う host API の最小の型（本体の契約のうち、実際に呼ぶものだけ）。
 * 本体の契約は GRAPHY-Next の frontend/src/plugins/pluginTypes.ts。
 */
export interface PluginWindowHandle {
  /** プラグインが自由に使ってよい DOM。ここより外は触らない。 */
  container: HTMLElement;
  close(): void;
  onClose(listener: () => void): void;
  readonly closed: boolean;
}

export interface PluginWindowOptions {
  title?: string;
  width?: number;
  height?: number;
}

export interface ResearchHost {
  pluginId: string;
  /** 活性化した時点の表示言語（"ja" / "en"）。 */
  locale: string;
  notify: (message: string) => void;
  surface: string;
  /** メイン画面の host が持つ（H42）。 */
  openWindow?: (opts?: PluginWindowOptions) => PluginWindowHandle;
}
