/**
 * Harrell の C-index と、ブートストラップによる楽観の補正（Harrell の方法）。
 * 比較できる組: 時間が短い方がイベント、かつ時間が異なる。リスクが高い方が短い時間なら一致、同点は 0.5。
 * 同じ時間の組は数えない（流儀が分かれるため、ここで決めて記録する）。
 */
import type { Rng } from "../rng";
import type { Surv } from "./types";

export function cIndex(s: Surv, risk: number[]): number | null {
  let conc = 0, comp = 0;
  const n = s.time.length;
  for (let i = 0; i < n; i++) {
    if (s.event[i] !== 1) continue;
    for (let j = 0; j < n; j++) {
      if (s.time[j] <= s.time[i]) continue;
      comp++;
      if (risk[i] > risk[j]) conc++;
      else if (risk[i] === risk[j]) conc += 0.5;
    }
  }
  return comp === 0 ? null : conc / comp;
}

/** モデルの当てはめ。失敗は null（選択を含む手順全体を 1 つの関数として渡せる）。 */
export type FitFn = (idx: number[]) => ((i: number) => number) | null;

export interface CorrectedC {
  apparent: number;
  optimism: number;
  corrected: number;
  bootstraps: number;
  failed: number;
}

/**
 * fit(idx) は idx（元の症例番号の並び、重複あり）でモデルを当てはめ、
 * 元の症例 i のリスクを返す関数を返す。特徴量の選択もこの中に含めること
 * （選択を含めないと楽観を小さく見積もる）。
 */
export async function bootstrapCorrectedC(
  s: Surv, fit: FitFn, B: number, rng: Rng, tick?: (done: number, of: number) => Promise<void>,
): Promise<CorrectedC | { error: string }> {
  const n = s.time.length;
  const all = Array.from({ length: n }, (_, i) => i);
  const full = fit(all);
  if (!full) return { error: "全データでモデルを当てはめられませんでした" };
  const apparent = cIndex(s, all.map(full));
  if (apparent === null) return { error: "比較できる組がありません" };
  let sum = 0, ok = 0, failed = 0;
  for (let b = 0; b < B; b++) {
    if (tick) await tick(b, B);
    const idx = Array.from({ length: n }, () => rng.int(n));
    const model = fit(idx);
    if (!model) { failed++; continue; }
    const bs: Surv = { time: idx.map((i) => s.time[i]), event: idx.map((i) => s.event[i]) };
    const cBoot = cIndex(bs, idx.map(model));
    const cTest = cIndex(s, all.map(model));
    if (cBoot === null || cTest === null) { failed++; continue; }
    sum += cBoot - cTest;
    ok++;
  }
  if (ok === 0) return { error: "ブートストラップがすべて失敗しました" };
  const optimism = sum / ok;
  return { apparent, optimism, corrected: apparent - optimism, bootstraps: ok, failed };
}
