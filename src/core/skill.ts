/**
 * スキルの宣言。スキル = 調べたい問い + 固定した手順 + 既定値 + 出典。利用者は手法でなく問いを選ぶ。
 * 手順（Step）は実行エンジン（engine.ts）が順に呼び、設定・結果の要約・警告・エラーを記録する。
 */
import type { TableSource } from "./dataset";
import type { Rng } from "./rng";

export type Locale = "ja" | "en";
export type LocalText = Record<Locale, string>;
export type ParamValue = string | number | boolean | null | string[] | { column: string; values: string[] }[];

export interface RunInputs {
  features: TableSource[];
  /** 臨床の表。結合を使わないスキル（再現性）では無くてよい。 */
  clinical?: TableSource;
  /** 変化率を使うとき: 前の時点・後の時点の特徴量の表。 */
  baseline?: TableSource[];
  followup?: TableSource[];
  /** 品質管理で除く特徴量の一覧（再現性のスキルの除外候補 CSV など。「feature」列）。 */
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
  /** 記録に残す構造のある詳細（除外した特徴量と理由など。summary は平らな値だけなので別に持つ）。 */
  detail?: Record<string, unknown>;
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

export interface Skill {
  id: string;
  title: LocalText;
  question: LocalText;
  /** 出典（論文など）。空なら「出典なし」と表示する（この段の基盤スキルは統計の手法を持たない）。 */
  sources: string[];
  defaults: Record<string, ParamValue>;
  /** 既定値を持たない任意の設定（渡したときだけ記録に載る）。既定の実行の記録を変えないために defaults へ入れない。 */
  optional?: string[];
  steps: Step[];
}
