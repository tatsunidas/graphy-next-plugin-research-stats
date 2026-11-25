/**
 * Harrell の C-index と、ブートストラップによる楽観の補正（Harrell の方法）。
 * 比較できる組: 時間が短い方がイベント。リスクが高い方が短い時間なら一致、同点は 0.5。
 * 同じ時間の組は、片方がイベント・もう片方が打ち切りなら数える（イベントの方が先とみなす。lifelines の
 * concordance_index と同じ定義）。同じ時間のイベント同士・打ち切り同士は数えない。
 */
import type { Rng } from "../rng";
import type { Surv } from "./types";

export function cIndex(s: Surv, risk: number[]): number | null {
  let conc = 0, comp = 0;
  const n = s.time.length;
  for (let i = 0; i < n; i++) {
    if (s.event[i] !== 1) continue;
    for (let j = 0; j < n; j++) {
      if (s.time[j] < s.time[i] || (s.time[j] === s.time[i] && s.event[j] === 1)) continue;
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
  /**
   * ブートストラップに入らなかった患者（OOB。1 回あたり約 37%）での C-index の、回をまたいだ平均。モデルが見ていない患者での値なので、
   * 過学習が強く Harrell の補正が楽観を取りきれないとき（イベントが少ない・特徴量が多い）でも 0.5 に近づく。
   * OOB にイベントが無い・比べられる組が無い回は含めない（oobN はその数）。無ければ null。
   */
  oob: number | null;
  oobN: number;
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
  let sum = 0, ok = 0, failed = 0, oobSum = 0, oobN = 0;
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
    const inBag = new Set(idx);
    const out = all.filter((i) => !inBag.has(i));
    const cOob = out.length ? cIndex({ time: out.map((i) => s.time[i]), event: out.map((i) => s.event[i]) }, out.map((i) => model(i))) : null;
    if (cOob !== null) { oobSum += cOob; oobN++; }
  }
  if (ok === 0) return { error: "ブートストラップがすべて失敗しました" };
  const optimism = sum / ok;
  return { apparent, optimism, corrected: apparent - optimism, bootstraps: ok, failed, oob: oobN ? oobSum / oobN : null, oobN };
}
