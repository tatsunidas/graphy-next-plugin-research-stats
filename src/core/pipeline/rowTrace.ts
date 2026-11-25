/**
 * 前処理・特徴量選択・当てはめが、どの行から推定したかの記録。行番号は X の行番号。
 * 呼び出しの場所で、推定に渡した行をそのまま記録する（実行後に leak.ts が訓練側の行かを検査する）。
 */
export type RowStage = "qc" | "impute" | "rank" | "fit" | "pdrop" | "pdropPaired";

/** 目的変数（結果ラベル・時間・イベント）を使う段階か。補完（中央値）・品質管理（分散 0・相関）・対応ありの p 値による除外（前後の値の差）は特徴量の値だけを読む。対応なしの p 値による除外（pdrop）は結果ラベルの群を使う。 */
export const USES_OUTCOME: Record<RowStage, boolean> = { qc: false, impute: false, rank: true, fit: true, pdrop: true, pdropPaired: false };

export interface RowUse {
  /** "pre"（交差検証の前）・"r{反復}f{分割}"（外側の分割）・"final"（最終モデル）。内側の分割は末尾に ".i{番号}" を付ける。 */
  scope: string;
  stage: RowStage;
  /** 目的変数を使う段階か（USES_OUTCOME）。 */
  usesOutcome: boolean;
  rows: readonly number[];
}

export interface RowContext {
  /** その範囲で推定に使ってよい行（これ以外の行は、外側の検証の行・テスト集合）。 */
  train: readonly number[];
}

export class RowTrace {
  scope = "pre";
  readonly uses: RowUse[] = [];
  readonly contexts = new Map<string, RowContext>();
  private readonly stack: string[] = [];
  /** 内側の分割に入る。train はその内側の訓練の行。 */
  enter(inner: number, train: readonly number[]): void {
    this.stack.push(this.scope);
    this.scope = `${this.scope}.i${inner}`;
    this.contexts.set(this.scope, { train });
  }
  leave(): void {
    this.scope = this.stack.pop() as string;
  }
  note(stage: RowStage, rows: readonly number[]): void {
    this.uses.push({ scope: this.scope, stage, usesOutcome: USES_OUTCOME[stage], rows });
  }
}
