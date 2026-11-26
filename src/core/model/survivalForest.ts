/**
 * ランダムサバイバルフォレスト（Ishwaran ら 2008）。scikit-survival の RandomSurvivalForest と同じ定義:
 * - 各木は、訓練の行から復元抽出で n 回引き、引かれた回数を重みにする（bootstrap=false なら全行を重み 1）。
 * - 分岐は、ログランク統計の絶対値が最大になる（特徴量・しきい値）。しきい値は隣り合う値の中点。
 *   特徴量は分岐ごとに無作為に並べ替え、先頭から mtry 個を見る（その中に有効な分岐が無ければ、有効な分岐が見つかるまで続ける）。
 * - 葉の最小の行数 minLeaf（引かれた行の数。重みではない）。ノードの行数が minSplit 未満なら分けない。
 * - 葉は、その葉の行（重みつき）の Nelson–Aalen の累積ハザード H(u)。木の予測（リスク）は、訓練の全行の**イベント時刻**
 *   u について H(u) を足した値。森の予測は、木の予測の平均。
 *
 * 乱数を使わない設定（bootstrap=false・mtry=列の数）で scikit-survival と予測が一致することを、試験で確かめている。
 */
import type { Rng } from "../rng";
import type { Surv } from "../survival/types";

export interface SurvForestOptions {
  trees: number;
  /** 分岐ごとに最低限見る特徴量の数（1 以上、列の数以下）。 */
  mtry: number;
  /** 葉の最小の行数（1 以上）。 */
  minLeaf: number;
  /** ノードを分ける最小の行数。省略は 2 × minLeaf（scikit-survival の既定 6・3 と同じ比）。 */
  minSplit?: number;
  /** 既定 true。false は試験用（乱数を使わない形にして scikit-survival と数値で照合する）。 */
  bootstrap?: boolean;
}

export interface Node {
  feat: number;
  thr: number;
  left: number;
  right: number;
  /** 葉のリスク（葉でなければ 0）。 */
  value: number;
}

export interface SurvForest {
  cols: number[];
  nTrees: number;
  /** 木（ノードの配列。先頭が根）。試験・確認用。 */
  trees: Node[][];
  /** X の行と同じ並びの 1 行から、リスク（累積ハザードのイベント時刻での和の、木の平均）。 */
  predict(x: ArrayLike<number>): number;
}

/** 重みつきのログランク統計の絶対値（群 = 左）。at-risk・イベントは、ノードのイベント時刻ごとの配列。 */
function logrankAbs(Y: Float64Array, D: Float64Array, Y1: Float64Array, D1: Float64Array): number {
  let num = 0, v = 0;
  for (let k = 0; k < Y.length; k++) {
    const y = Y[k], d = D[k];
    if (!(y > 0)) continue;
    const f = Y1[k] / y;
    num += D1[k] - d * f;
    if (y > 1) v += d * f * (1 - f) * (y - d) / (y - 1);
  }
  return v > 0 ? Math.abs(num) / Math.sqrt(v) : 0;
}

/** surv は X の行に揃えた時間・イベント。rows は訓練の行、cols は使う列（欠損の無い列）。 */
export function survivalForest(surv: Surv, X: number[][], rows: number[], cols: number[], o: SurvForestOptions, rng: Rng): SurvForest {
  if (!Number.isInteger(o.trees) || o.trees < 1) throw new RangeError("木の数は 1 以上の整数です");
  if (!Number.isInteger(o.mtry) || o.mtry < 1 || o.mtry > cols.length) throw new RangeError(`mtry は 1 以上 ${cols.length} 以下の整数です`);
  if (!Number.isInteger(o.minLeaf) || o.minLeaf < 1) throw new RangeError("葉の最小の行数は 1 以上の整数です");
  const minSplit = o.minSplit ?? 2 * o.minLeaf;
  const n = rows.length;
  const time = surv.time, event = surv.event;
  // 訓練の全行のイベント時刻（昇順）。木の予測は、この時刻での累積ハザードの和
  const U = [...new Set(rows.filter((i) => event[i] === 1).map((i) => time[i]))].sort((a, b) => a - b);
  if (!U.length) throw new RangeError("訓練の行にイベントがありません");
  /** U の中で u 以上の時刻の数。 */
  const countGe = (u: number): number => { let lo = 0, hi = U.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (U[mid] < u) lo = mid + 1; else hi = mid; } return U.length - lo; };
  const w = new Float64Array(X.length);
  const trees: Node[][] = [];

  const leafValue = (S: number[]): number => {
    // その葉のイベント時刻ごとに d/Y を求め、（その時刻以降の U の数）をかけて足す = Σ_{u∈U} H(u)
    const ts = [...new Set(S.filter((i) => event[i] === 1).map((i) => time[i]))].sort((a, b) => a - b);
    let v = 0;
    for (const u of ts) {
      let Y = 0, d = 0;
      for (const i of S) { if (time[i] >= u) Y += w[i]; if (event[i] === 1 && time[i] === u) d += w[i]; }
      if (Y > 0) v += (d / Y) * countGe(u);
    }
    return v;
  };

  const build = (S: number[], nodes: Node[]): number => {
    const id = nodes.length;
    nodes.push({ feat: -1, thr: 0, left: -1, right: -1, value: 0 });
    const leaf = () => { nodes[id].value = leafValue(S); return id; };
    const m = S.length;
    if (m < minSplit || m < 2 * o.minLeaf) return leaf();
    // このノードのイベント時刻
    const ut = [...new Set(S.filter((i) => event[i] === 1).map((i) => time[i]))].sort((a, b) => a - b);
    if (!ut.length) return leaf();
    const K = ut.length;
    const idxOf = new Map(ut.map((u, k) => [u, k]));
    const kmax = S.map((i) => { let c = 0; while (c < K && ut[c] <= time[i]) c++; return c; }); // 危険にさらされるイベント時刻の数
    const Y = new Float64Array(K), D = new Float64Array(K);
    S.forEach((i, q) => { for (let k = 0; k < kmax[q]; k++) Y[k] += w[i]; if (event[i] === 1) D[idxOf.get(time[i]) as number] += w[i]; });
    // 特徴量を無作為に並べ替える（先頭から mtry 個。有効な分岐が無ければ続ける）
    const perm = cols.slice();
    for (let a = 0; a < perm.length - 1; a++) { const r = a + rng.int(perm.length - a); [perm[a], perm[r]] = [perm[r], perm[a]]; }
    let best: { s: number; j: number; thr: number } | null = null;
    const Y1 = new Float64Array(K), D1 = new Float64Array(K);
    for (let a = 0; a < perm.length; a++) {
      if (a >= o.mtry && best !== null) break;
      const j = perm[a];
      const order = S.map((_, q) => q).sort((p, q) => X[S[p]][j] - X[S[q]][j] || p - q);
      Y1.fill(0); D1.fill(0);
      for (let p = 0; p < m - 1; p++) {
        const q = order[p], i = S[q];
        for (let k = 0; k < kmax[q]; k++) Y1[k] += w[i];
        if (event[i] === 1) D1[idxOf.get(time[i]) as number] += w[i];
        const xa = X[i][j], xb = X[S[order[p + 1]]][j];
        if (p + 1 < o.minLeaf || m - (p + 1) < o.minLeaf || !(xa < xb)) continue;
        const s = logrankAbs(Y, D, Y1, D1);
        if (best === null || s > best.s + 1e-12) {
          let thr = (xa + xb) / 2;
          if (!(thr < xb)) thr = xa; // 丸めで xb に達したとき
          best = { s, j, thr };
        }
      }
    }
    if (best === null || !(best.s > 0)) return leaf();
    const L = S.filter((i) => X[i][best!.j] <= best!.thr), R = S.filter((i) => X[i][best!.j] > best!.thr);
    nodes[id].feat = best.j;
    nodes[id].thr = best.thr;
    nodes[id].left = build(L, nodes);
    nodes[id].right = build(R, nodes);
    return id;
  };

  for (let t = 0; t < o.trees; t++) {
    w.fill(0);
    if (o.bootstrap === false) for (const i of rows) w[i] = 1;
    else for (let d = 0; d < n; d++) w[rows[rng.int(n)]] += 1;
    const inBag = rows.filter((i) => w[i] > 0);
    const nodes: Node[] = [];
    build(inBag, nodes);
    trees.push(nodes);
  }
  const predict = (x: ArrayLike<number>): number => {
    let s = 0;
    for (const nodes of trees) {
      let k = 0;
      while (nodes[k].left >= 0) k = x[nodes[k].feat] <= nodes[k].thr ? nodes[k].left : nodes[k].right;
      s += nodes[k].value;
    }
    return s / trees.length;
  };
  return { cols, nTrees: trees.length, predict, trees };
}
