/**
 * Cox の後退選択（出る p > pOut、戻る p < pIn）。p は多変量 Cox（Efron）の Wald の p。
 * 1 回に 1 つ、p が最大で pOut を超える変数を除く。除くたびに、除いた変数を 1 つずつ戻したときの p を見て、
 * 最小が pIn 未満ならそれを戻す。同じ組み合わせに戻ったら（循環）止めて記録する。
 * 最後に maxVars を超えていれば、p の大きい順に maxVars 個になるまで除き、そのことを記録する。
 * 変数は当てはめに使う行で標準化してから当てはめる（p は尺度によらない。係数の発散の判定を安定させるため）。
 */
import type { Surv } from "../survival/types";
import { coxFit, type CoxFit } from "../survival/cox";
import { subSurv } from "./breslow";
import { fitStd, zRow } from "./lasso";

export interface BackwardOptions {
  pIn: number;
  pOut: number;
  /** 最後に残す変数の最大の数。 */
  maxVars: number;
}

export interface BackwardStep {
  action: "remove" | "add" | "cap" | "cycle";
  col: number;
  p: number | null;
}

export interface BackwardResult {
  /** 残った列（X の列番号、候補の並び）。 */
  cols: number[];
  steps: BackwardStep[];
  /** 残った列での当てはめ（標準化した尺度）。残りが 0 個なら null。 */
  fit: CoxFit | null;
}

export function backwardSelect(surv: Surv, X: number[][], rows: number[], candidates: number[], o: BackwardOptions): BackwardResult | { error: string } {
  if (!(o.pIn > 0 && o.pIn < 1 && o.pOut > 0 && o.pOut < 1)) return { error: "入る・出る p は 0 と 1 の間にしてください" };
  if (o.pIn >= o.pOut) return { error: "入る p は出る p より小さくしてください（同じか大きいと出入りを繰り返します）" };
  if (!Number.isInteger(o.maxVars) || o.maxVars < 1) return { error: "残す変数の最大の数は 1 以上の整数にしてください" };
  const s = subSurv(surv, rows);
  const fitOn = (cols: number[]): CoxFit | { error: string } => {
    const st = fitStd(X, rows, cols);
    if ("error" in st) return st;
    if (st.cols.length !== cols.length) return { error: "当てはめに使う行で値が同じ特徴量があります" };
    const r = coxFit(s, rows.map((i) => zRow(st, X[i])));
    return r.ok ? r.fit : { error: r.error };
  };
  let cur = [...candidates];
  const removed: number[] = [];
  const steps: BackwardStep[] = [];
  const seen = new Set<string>([key(cur)]);
  const order = (cols: number[]) => candidates.filter((c) => cols.includes(c));
  for (let guard = 0; guard < 10 * candidates.length + 10 && cur.length; guard++) {
    const f = fitOn(cur);
    if ("error" in f) return { error: `${cur.length} 変数の Cox: ${f.error}` };
    let w = 0;
    f.p.forEach((p, a) => { if (p > f.p[w]) w = a; });
    if (!(f.p[w] > o.pOut)) break;
    const out = cur[w];
    cur = cur.filter((c) => c !== out);
    removed.push(out);
    steps.push({ action: "remove", col: out, p: f.p[w] });
    let best: { col: number; p: number } | null = null;
    for (const c of removed) {
      if (c === out) continue;
      const g = fitOn(order([...cur, c]));
      if ("error" in g) continue;
      const p = g.p[order([...cur, c]).indexOf(c)];
      if (p < o.pIn && (!best || p < best.p)) best = { col: c, p };
    }
    if (best) {
      const next = order([...cur, best.col]);
      if (seen.has(key(next))) { steps.push({ action: "cycle", col: best.col, p: best.p }); break; }
      cur = next;
      removed.splice(removed.indexOf(best.col), 1);
      steps.push({ action: "add", col: best.col, p: best.p });
    }
    seen.add(key(cur));
  }
  while (cur.length > o.maxVars) {
    const f = fitOn(cur);
    if ("error" in f) return { error: `${cur.length} 変数の Cox: ${f.error}` };
    let w = 0;
    f.p.forEach((p, a) => { if (p > f.p[w]) w = a; });
    steps.push({ action: "cap", col: cur[w], p: f.p[w] });
    cur = cur.filter((_, a) => a !== w);
  }
  if (!cur.length) return { cols: [], steps, fit: null };
  const f = fitOn(cur);
  if ("error" in f) return { error: `${cur.length} 変数の Cox: ${f.error}` };
  return { cols: cur, steps, fit: f };
}

const key = (cols: number[]) => [...cols].sort((a, b) => a - b).join(",");
