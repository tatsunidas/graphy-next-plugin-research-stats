/**
 * LASSO（L1 罰則）。二値はロジスティック回帰（切片には罰則をかけない）、生存は Cox（同時刻は Breslow）。
 *
 * 目的関数: −(1/n)·対数尤度 ＋ λ·Σ|β_j|。n は当てはめに使う行の数。
 * 解き方: 尤度を η の対角ヘッセで二次近似し（glmnet と同じ考え方）、重み付き最小二乗の LASSO を座標降下で解く。
 * 近似の解への方向で、目的関数が下がるまでステップを半分にする（目的関数は凸なので、一意の最小に収束する）。
 *
 * 標準化は**当てはめに使う行だけ**の平均・SD（n−1）で行い、係数は標準化した尺度で返す。
 * λ の列は λmax（全係数が 0 になる最小の λ）から λmax·ratio まで対数で等間隔。
 * 交差検証の評価は偏差: 二値は検証の行の平均偏差（分割の人数で重み付け）、
 * 生存は Verweij–van Houwelingen の部分尤度の差 −2{l_全体(β_−k) − l_訓練(β_−k)} をイベント数で割ったもの（イベント数で重み付け）。
 */
import type { Rng } from "../rng";
import type { Surv } from "../survival/types";
import { stratifiedFolds } from "../pipeline/split";
import { breslow, riskSets, subSurv, type Risk } from "./breslow";

export type LassoTask = { kind: "binary"; y: number[] } | { kind: "survival"; surv: Surv };
export type LambdaRule = "min" | "1se";

const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
const log1pexp = (z: number) => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));
const soft = (x: number, l: number) => (x > l ? x - l : x < -l ? x + l : 0);

/** 当てはめに使う行（行列 Z は標準化済み）での尤度。 */
interface Lik {
  n: number;
  intercept: boolean;
  /** 対数尤度と、η についての勾配・対角の −ヘッセ。 */
  eval(eta: number[]): { ll: number; g: number[]; h: number[] };
  /** 飽和モデルの対数尤度（偏差の基準）。 */
  llSat: number;
}

function likOf(task: LassoTask, rows: number[]): Lik {
  if (task.kind === "binary") {
    const y = rows.map((i) => task.y[i]);
    return {
      n: rows.length, intercept: true, llSat: 0,
      eval(eta) {
        let ll = 0;
        const g = new Array<number>(eta.length), h = new Array<number>(eta.length);
        for (let k = 0; k < eta.length; k++) {
          const p = sigmoid(eta[k]);
          ll += y[k] * eta[k] - log1pexp(eta[k]);
          g[k] = y[k] - p;
          h[k] = p * (1 - p);
        }
        return { ll, g, h };
      },
    };
  }
  const s = subSurv(task.surv, rows);
  const r: Risk = riskSets(s);
  let llSat = 0;
  for (const grp of r.groups) if (grp.d > 0) llSat -= grp.d * Math.log(grp.d);
  return { n: rows.length, intercept: false, llSat, eval: (eta) => breslow(s, r, eta) };
}

export interface Coef {
  b0: number;
  beta: number[];
}

function etaOf(Z: number[][], c: Coef): number[] {
  return Z.map((row) => { let e = c.b0; for (let j = 0; j < row.length; j++) if (c.beta[j] !== 0) e += row[j] * c.beta[j]; return e; });
}

const penalty = (beta: number[]) => beta.reduce((a, b) => a + Math.abs(b), 0);

/** 一つの λ での解。init から始める（経路では一つ前の λ の解）。 */
export function solveLasso(lik: Lik, Z: number[][], lambda: number, init: Coef): { coef: Coef; iterations: number } | { error: string } {
  const n = lik.n, q = Z[0]?.length ?? 0;
  let cur: Coef = { b0: init.b0, beta: [...init.beta] };
  let eta = etaOf(Z, cur);
  let ev = lik.eval(eta);
  let F = -ev.ll / n + lambda * penalty(cur.beta);
  const colA = new Array<number>(q);
  for (let it = 0; it < 500; it++) {
    const w = ev.h.map((h) => (h > 1e-12 ? h / n : 0));
    const z = eta.map((e, k) => (w[k] > 0 ? e + ev.g[k] / ev.h[k] : e));
    // 重み付き最小二乗の LASSO: (1/2)Σ w_k (z_k − b0 − Z_k β)² + λ|β|₁
    const nxt: Coef = { b0: cur.b0, beta: [...cur.beta] };
    const res = z.map((zk, k) => zk - eta[k]);
    for (let j = 0; j < q; j++) { let a = 0; for (let k = 0; k < Z.length; k++) a += w[k] * Z[k][j] * Z[k][j]; colA[j] = a; }
    const sweep = (only: number[] | null): number => {
      let maxd = 0;
      if (lik.intercept) {
        let sw = 0, sr = 0;
        for (let k = 0; k < Z.length; k++) { sw += w[k]; sr += w[k] * res[k]; }
        if (sw > 0) { const d = sr / sw; nxt.b0 += d; for (let k = 0; k < Z.length; k++) res[k] -= d; maxd = Math.max(maxd, Math.abs(d)); }
      }
      const js = only ?? Array.from({ length: q }, (_, j) => j);
      for (const j of js) {
        if (!(colA[j] > 0)) continue;
        let c = 0;
        for (let k = 0; k < Z.length; k++) c += w[k] * Z[k][j] * res[k];
        c += colA[j] * nxt.beta[j];
        const b = soft(c, lambda) / colA[j];
        const d = b - nxt.beta[j];
        if (d !== 0) { for (let k = 0; k < Z.length; k++) res[k] -= d * Z[k][j]; nxt.beta[j] = b; maxd = Math.max(maxd, Math.abs(d) * Math.sqrt(colA[j])); }
      }
      return maxd;
    };
    for (let outer = 0; outer < 1000; outer++) {
      const full = sweep(null);
      if (full < 1e-13) break;
      const active = nxt.beta.map((b, j) => (b !== 0 ? j : -1)).filter((j) => j >= 0);
      for (let inner = 0; inner < 10000; inner++) if (sweep(active) < 1e-13) break;
    }
    // 近似の解への方向で、目的関数が下がるまで半分にする
    const d0 = nxt.b0 - cur.b0, d = nxt.beta.map((b, j) => b - cur.beta[j]);
    const step = Math.max(Math.abs(d0), ...d.map(Math.abs), 0);
    if (step < 1e-11) return { coef: cur, iterations: it };
    let t = 1, accepted = false;
    for (let hh = 0; hh < 50; hh++) {
      const cand: Coef = { b0: cur.b0 + t * d0, beta: cur.beta.map((b, j) => (t === 1 ? nxt.beta[j] : b + t * d[j])) };
      const e2 = etaOf(Z, cand);
      const ev2 = lik.eval(e2);
      const F2 = -ev2.ll / n + lambda * penalty(cand.beta);
      if (Number.isFinite(F2) && F2 <= F + 1e-15 * Math.abs(F)) {
        const small = t * step < 1e-11;
        cur = cand; eta = e2; ev = ev2; F = F2; accepted = true;
        if (small) return { coef: cur, iterations: it + 1 };
        break;
      }
      t /= 2;
    }
    if (!accepted) return step < 1e-6 ? { coef: cur, iterations: it } : { error: "LASSO の目的関数が下がらず、収束しませんでした" };
  }
  return { error: "LASSO が 500 回で収束しませんでした" };
}

export interface Std {
  /** 使った列（X の列番号）。訓練の行で SD が 0 の列は除く。 */
  cols: number[];
  means: number[];
  sds: number[];
}

/** 訓練の行で標準化を求める。値は欠けていないこと（呼び出し側で補完か完全症例に）。 */
export function fitStd(X: number[][], rows: number[], cols: number[]): Std | { error: string } {
  const out: Std = { cols: [], means: [], sds: [] };
  for (const j of cols) {
    const v = rows.map((i) => X[i][j]);
    if (v.some((x) => !Number.isFinite(x))) return { error: `欠損のある列があります（列 ${j + 1}）` };
    const mu = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / (v.length - 1));
    if (sd > 0) { out.cols.push(j); out.means.push(mu); out.sds.push(sd); }
  }
  return out;
}

export const zRow = (st: Std, row: number[]) => st.cols.map((j, a) => (row[j] - st.means[a]) / st.sds[a]);

/** 全係数が 0 になる最小の λ（二値は切片だけのモデル、生存は β=0 での勾配から）。 */
export function lambdaMax(lik: Lik, Z: number[][]): { lmax: number; b0: number } {
  let b0 = 0;
  if (lik.intercept) {
    // 切片だけのロジスティック: b0 = logit(平均)
    const ev0 = lik.eval(new Array<number>(Z.length).fill(0));
    const mean = ev0.g.reduce((a, g) => a + g, 0) / Z.length + 0.5;
    b0 = Math.log(mean / (1 - mean));
  }
  const ev = lik.eval(new Array<number>(Z.length).fill(b0));
  const q = Z[0]?.length ?? 0;
  let lmax = 0;
  for (let j = 0; j < q; j++) { let s = 0; for (let k = 0; k < Z.length; k++) s += Z[k][j] * ev.g[k]; lmax = Math.max(lmax, Math.abs(s) / lik.n); }
  return { lmax, b0 };
}

export function lambdaSeq(lmax: number, count: number, ratio: number): number[] {
  if (count === 1) return [lmax];
  return Array.from({ length: count }, (_, i) => lmax * Math.pow(ratio, i / (count - 1)));
}

export interface PathPoint {
  lambda: number;
  coef: Coef;
  /** 偏差の説明率（1 − 偏差/空モデルの偏差）。 */
  devRatio: number;
}

export interface Path {
  points: PathPoint[];
  /** 経路を途中で止めた理由（無ければ null）。 */
  stopped: string | null;
}

/**
 * λ の大きい順に、一つ前の解から始めて解く。偏差の説明率が 0.999 を超えたら止める（glmnet と同じ止め方。
 * 二値の完全分離などで係数が発散するのを防ぐ）。
 */
export function lassoPath(lik: Lik, Z: number[][], lambdas: number[], b0Init: number): Path {
  const q = Z[0]?.length ?? 0;
  const ll0 = lik.eval(new Array<number>(Z.length).fill(b0Init)).ll;
  const nullDev = 2 * (lik.llSat - ll0);
  let c: Coef = { b0: b0Init, beta: new Array<number>(q).fill(0) };
  const points: PathPoint[] = [];
  for (const l of lambdas) {
    const r = solveLasso(lik, Z, l, c);
    if ("error" in r) return { points, stopped: `λ=${l.toPrecision(4)}: ${r.error}` };
    c = r.coef;
    const ll = lik.eval(etaOf(Z, c)).ll;
    const devRatio = nullDev > 0 ? 1 - (2 * (lik.llSat - ll)) / nullDev : 0;
    points.push({ lambda: l, coef: { b0: c.b0, beta: [...c.beta] }, devRatio });
    if (devRatio > 0.999) return { points, stopped: "偏差の説明率が 0.999 を超えたため、これより小さい λ は解きません" };
  }
  return { points, stopped: null };
}

export interface LassoOptions {
  /** λ の点の数。 */
  nLambda: number;
  /** 最小の λ ÷ λmax。 */
  ratio: number;
  /** 交差検証の分割数。 */
  folds: number;
  rule: LambdaRule;
}

export interface LassoCvResult {
  std: Std;
  lambdas: number[];
  /** λ ごとの交差検証の偏差（全分割で解けた λ だけ。それ以外は null）。 */
  cvm: (number | null)[];
  cvsd: (number | null)[];
  idxMin: number;
  idx1se: number;
  /** 選んだ λ の位置と値。 */
  chosen: number;
  lambda: number;
  /** 当てはめに使った全行での経路。 */
  path: Path;
  coef: Coef;
  /** 係数が 0 でない列（X の列番号）。 */
  nonzero: number[];
  /** 行ごとの分割番号（rows の並び）。 */
  foldOf: number[];
  /** X の 1 行の予測（二値は確率、生存は線形予測子）。 */
  predict(row: number[]): number;
}

function validate(o: LassoOptions, nRows: number): string | null {
  if (!Number.isInteger(o.nLambda) || o.nLambda < 2 || o.nLambda > 1000) return "λ の点の数は 2 以上 1000 以下の整数にしてください";
  if (!(o.ratio > 0 && o.ratio < 1)) return "最小の λ の比は 0 より大きく 1 より小さい値にしてください";
  if (!Number.isInteger(o.folds) || o.folds < 2 || o.folds > nRows) return `交差検証の分割数は 2 以上 ${nRows} 以下の整数にしてください`;
  return null;
}

/** 当てはめに使う行 rows で、標準化 → λ の列 → k 分割の交差検証 → 選んだ λ の解。 */
export function lassoCv(task: LassoTask, X: number[][], rows: number[], cols: number[], o: LassoOptions, rng: Rng): LassoCvResult | { error: string } {
  const bad = validate(o, rows.length);
  if (bad) return { error: bad };
  const std = fitStd(X, rows, cols);
  if ("error" in std) return std;
  if (!std.cols.length) return { error: "値がばらつく特徴量がありません" };
  const Z = rows.map((i) => zRow(std, X[i]));
  const lik = likOf(task, rows);
  if (task.kind === "binary") {
    const pos = rows.reduce((a, i) => a + task.y[i], 0);
    if (pos === 0 || pos === rows.length) return { error: "片方の群しかありません" };
  } else if (!rows.some((i) => task.surv.event[i] === 1)) return { error: "イベントがありません" };
  const { lmax, b0 } = lambdaMax(lik, Z);
  if (!(lmax > 0)) return { error: "λmax が 0 です（どの特徴量も結果と関係が見えません）" };
  const lambdas = lambdaSeq(lmax, o.nLambda, o.ratio);
  const path = lassoPath(lik, Z, lambdas, b0);
  if (!path.points.length) return { error: path.stopped ?? "経路を解けませんでした" };

  const foldOf = stratifiedFolds(rows.length, null, o.folds, rng);
  const L = lambdas.length;
  const raw: { w: number; v: (number | null)[] }[] = [];
  for (let f = 0; f < o.folds; f++) {
    const tr = rows.filter((_, p) => foldOf[p] !== f), te = rows.filter((_, p) => foldOf[p] === f);
    const st = fitStd(X, tr, std.cols);
    if ("error" in st) return st;
    // 分割の訓練の行で SD が 0 になった列は、その分割では使わない（係数 0 と同じ）
    const Ztr = tr.map((i) => zRow(st, X[i]));
    const lk = likOf(task, tr);
    const b0f = task.kind === "binary" ? lambdaMax(lk, Ztr).b0 : 0;
    const pf = lassoPath(lk, Ztr, lambdas, b0f);
    const v: (number | null)[] = new Array<number | null>(L).fill(null);
    let w: number;
    if (task.kind === "binary") {
      w = te.length;
      const Zte = te.map((i) => zRow(st, X[i]));
      pf.points.forEach((pt, a) => {
        let dev = 0;
        Zte.forEach((z, k) => {
          let e = pt.coef.b0;
          for (let j = 0; j < z.length; j++) e += z[j] * pt.coef.beta[j];
          dev += -2 * (task.y[te[k]] * e - log1pexp(e));
        });
        v[a] = dev / te.length;
      });
    } else {
      w = te.reduce((a, i) => a + task.surv.event[i], 0);
      if (w > 0) {
        const Zall = rows.map((i) => zRow(st, X[i]));
        const lkAll = likOf(task, rows);
        pf.points.forEach((pt, a) => {
          const lAll = lkAll.eval(etaOf(Zall, pt.coef)).ll;
          const lTr = lk.eval(etaOf(Ztr, pt.coef)).ll;
          v[a] = (-2 * (lAll - lTr)) / w;
        });
      }
    }
    if (w > 0) raw.push({ w, v });
  }
  if (raw.length < 2) return { error: "交差検証で評価できた分割が 2 つ未満です（検証の分割にイベントが無いなど）" };
  const cvm: (number | null)[] = [], cvsd: (number | null)[] = [];
  for (let a = 0; a < L; a++) {
    if (a >= path.points.length || raw.some((r) => r.v[a] === null)) { cvm.push(null); cvsd.push(null); continue; }
    const W = raw.reduce((s, r) => s + r.w, 0);
    const m = raw.reduce((s, r) => s + r.w * (r.v[a] as number), 0) / W;
    const varw = raw.reduce((s, r) => s + r.w * ((r.v[a] as number) - m) ** 2, 0) / W;
    cvm.push(m);
    cvsd.push(Math.sqrt(varw / (raw.length - 1)));
  }
  let idxMin = -1;
  cvm.forEach((m, a) => { if (m !== null && (idxMin < 0 || m < (cvm[idxMin] as number))) idxMin = a; });
  if (idxMin < 0) return { error: "交差検証で評価できた λ がありません" };
  const lim = (cvm[idxMin] as number) + (cvsd[idxMin] as number);
  let idx1se = idxMin;
  for (let a = 0; a <= idxMin; a++) if (cvm[a] !== null && (cvm[a] as number) <= lim) { idx1se = a; break; }
  const chosen = o.rule === "min" ? idxMin : idx1se;
  const coef = path.points[chosen].coef;
  const nonzero = std.cols.filter((_, a) => coef.beta[a] !== 0);
  const predict = (row: number[]) => {
    const z = zRow(std, row);
    let e = coef.b0;
    for (let j = 0; j < z.length; j++) e += z[j] * coef.beta[j];
    return task.kind === "binary" ? sigmoid(e) : e;
  };
  return { std, lambdas, cvm, cvsd, idxMin, idx1se, chosen, lambda: lambdas[chosen], path, coef, nonzero, foldOf, predict };
}

/** 固定した λ での解（照合と試験用）。 */
export function lassoAt(task: LassoTask, X: number[][], rows: number[], cols: number[], lambda: number): { std: Std; coef: Coef; iterations: number } | { error: string } {
  const std = fitStd(X, rows, cols);
  if ("error" in std) return std;
  const Z = rows.map((i) => zRow(std, X[i]));
  const lik = likOf(task, rows);
  const b0 = task.kind === "binary" ? lambdaMax(lik, Z).b0 : 0;
  const r = solveLasso(lik, Z, lambda, { b0, beta: new Array<number>(std.cols.length).fill(0) });
  return "error" in r ? r : { std, coef: r.coef, iterations: r.iterations };
}

export { likOf };
