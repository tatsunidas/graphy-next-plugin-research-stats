/**
 * レシピの宣言。レシピ = 調べたい問い + 固定した手順 + 既定値 + 出典。利用者は手法でなく問いを選ぶ。
 * 手順（Step）は実行エンジン（engine.ts）が順に呼び、設定・結果の要約・警告・エラーを記録する。
 */
import type { TableSource } from "./dataset";
import type { Rng } from "./rng";

export type Locale = "ja" | "en";
export type LocalText = Record<Locale, string>;
export type ParamValue = string | number | boolean | null | string[];

export interface RunInputs {
  features: TableSource[];
  /** 臨床の表。結合を使わないレシピ（再現性）では無くてよい。 */
  clinical?: TableSource;
  /** 変化率を使うとき: 前の時点・後の時点の特徴量の表。 */
  baseline?: TableSource[];
  followup?: TableSource[];
  /** 品質管理で除く特徴量の一覧（再現性のレシピの除外候補 CSV など。「feature」列）。 */
  exclusion?: TableSource;
  /** テスト集合の固定ファイル（JSON の文字列）。 */
  lock?: { name: string; text: string };
}

export interface StepOutcome {
  summary: Record<string, string | number | boolean | null>;
  warnings: string[];
  /** 1 つでもあれば、その手順は失敗として記録され、以降は実行しない。 */
  errors?: string[];
  /** 何もしなかったとき（条件が合わない）に true。 */
  skipped?: boolean;
}

export interface StepContext {
  inputs: RunInputs;
  params: Record<string, ParamValue>;
  rng: Rng;
  /** 手順どうしで受け渡す結果（データセットなど）。 */
  state: Record<string, unknown>;
  /** 実行の開始時刻（記録と同じ文字列）とプラグインの版。 */
  now: string;
  pluginVersion: string;
  /**
   * 長い計算の途中で呼ぶ。進み具合を知らせ、画面に制御を返し、取り消されていれば例外を投げる。
   * 呼ぶ頻度は気にしなくてよい（制御を返すのは一定時間ごと）。
   */
  tick(label: string, fraction: number): Promise<void>;
}

export interface Step {
  id: string;
  title: LocalText;
  run(ctx: StepContext): StepOutcome | Promise<StepOutcome>;
}

export interface Recipe {
  id: string;
  title: LocalText;
  question: LocalText;
  /** 出典（論文など）。空なら「出典なし」と表示する（この段の基盤レシピは統計の手法を持たない）。 */
  sources: string[];
  defaults: Record<string, ParamValue>;
  steps: Step[];
}
