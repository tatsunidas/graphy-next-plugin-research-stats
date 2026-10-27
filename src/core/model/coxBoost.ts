/**
 * 成分ごとの尤度ベースの Cox ブースティング（線形の基底、同時刻は Breslow）。
 * 各反復で、負の勾配（マルチンゲール残差）を各特徴量の単回帰（切片なし・標準化済み）で当て、
 * 残差平方和が最小の 1 つだけを ν 倍して係数に足す。初期値は全係数 0。
 * 変数重要度は、その変数を更新したときに減った −対数部分尤度の合計。
 *
 * 論文で使われた R の mboost と**同じ値になるとは言えない**（論文に式が無く、R での照合もしていない）。
 */
import type { Surv } from "../survival/types";
import { breslow, riskSets, subSurv } from "./breslow";
import { fitStd, zRow, type Std } from "./lasso";

export interface BoostOptions {
  iterations: number;
  /** 学習率 ν（0 より大きく 1 以下）。 */
  nu: number;
}

export interface BoostResult {
  std: Std;
  /** 標準化した尺度の係数（std.cols の並び）。 */
  beta: number[];
  /** 変数重要度（std.cols の並び）。 */
  importance: number[];
  /** 反復ごとに選んだ列（std.cols の中の位置）。 */
  selected: number[];
  /** 反復ごとの −対数部分尤度（反復 0 は全係数 0）。 */
  risk: number[];
  /** 係数が 0 でない列（X の列番号）を重要度の大きい順に。 */
  ranking: number[];
}

export function coxBoost(surv: Surv, X: number[][], rows: number[], cols: number[], o: BoostOptions): BoostResult | { error: string } {
  if (!Number.isInteger(o.iterations) || o.iterations < 1 || o.iterations > 100000) return { error: "反復回数は 1 以上 100000 以下の整数にしてください" };
  if (!(o.nu > 0 && o.nu <= 1)) return { error: "学習率 ν は 0 より大きく 1 以下にしてください" };
  const std = fitStd(X, rows, cols);
  if ("error" in std) return std;
  const q = std.cols.length;
  if (!q) return { error: "値がばらつく特徴量がありません" };
  const s = subSurv(surv, rows);
  if (!s.event.some((e) => e === 1)) return { error: "イベントがありません" };
  const r = riskSets(s);
  const Z = rows.map((i) => zRow(std, X[i]));
  const ss = Array.from({ length: q }, (_, j) => Z.reduce((a, z) => a + z[j] * z[j], 0));
  const beta = new Array<number>(q).fill(0), importance = new Array<number>(q).fill(0);
  const eta = new Array<number>(Z.length).fill(0);
  const selected: number[] = [];
  let ev = breslow(s, r, eta);
  const risk = [-ev.ll];
  for (let m = 0; m < o.iterations; m++) {
    let best = -1, bestScore = -Infinity, bestB = 0;
    for (let j = 0; j < q; j++) {
      let xu = 0;
      for (let k = 0; k < Z.length; k++) xu += Z[k][j] * ev.g[k];
      // 残差平方和の減少 = (Σxu)²/Σx²。最大のものが残差平方和が最小
      const sc = (xu * xu) / ss[j];
      if (best < 0 || sc > bestScore * (1 + 1e-15)) { best = j; bestScore = sc; bestB = xu / ss[j]; }
    }
    const step = o.nu * bestB;
    beta[best] += step;
    for (let k = 0; k < Z.length; k++) eta[k] += step * Z[k][best];
    const before = -ev.ll;
    ev = breslow(s, r, eta);
    if (!Number.isFinite(ev.ll)) return { error: "対数部分尤度が数値になりませんでした（ν を小さくしてください）" };
    importance[best] += before - -ev.ll;
    selected.push(best);
    risk.push(-ev.ll);
  }
  const ranking = std.cols.map((c, a) => ({ c, a })).filter((x) => beta[x.a] !== 0)
    .sort((x, y) => importance[y.a] - importance[x.a] || x.a - y.a).map((x) => x.c);
  return { std, beta, importance, selected, risk, ranking };
}
