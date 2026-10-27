/**
 * 特徴量の品質管理（教師なしの絞り込み）。**開発集合の行だけ**で行い、テスト集合の値は見ない。
 * 結果ラベルを使わないので、交差検証のループの前に行ってよい（選択と違い、楽観的な偏りを生まない）。
 *
 * 順番: 除外リスト → 値が 1 種類しかない列 → 相関による冗長。
 * 相関で残す側は「欠損の少ない列、同じなら列の順が先の列」（決まった流儀が無いので、ここで決めて記録する）。
 */

export interface QcOptions {
  /** 開発集合で値（欠損を除く）が 1 種類以下の列を除く。 */
  dropConstant: boolean;
  /** |Spearman の順位相関| がこの値以上の組の片方を除く。null なら行わない。 */
  correlationThreshold: number | null;
  /** 名前で除く列（再現性の除外候補など）。null なら行わない。 */
  exclude: string[] | null;
}

export interface QcRemoval {
  feature: string;
  reason: string;
}

export interface QcResult {
  /** 残した列の番号（元の並び）。 */
  keep: number[];
  removed: QcRemoval[];
  /** 除外リストにあるが表に無かった名前。 */
  unknownExcluded: string[];
}

/** 平均順位（同順位は平均）。 */
export function ranks(x: number[]): number[] {
  const o = x.map((_, i) => i).sort((a, b) => x[a] - x[b]);
  const r = new Array<number>(x.length);
  for (let i = 0; i < o.length;) {
    let j = i;
    while (j < o.length && x[o[j]] === x[o[i]]) j++;
    for (let q = i; q < j; q++) r[o[q]] = (i + 1 + j) / 2;
    i = j;
  }
  return r;
}

function pearson(a: number[], b: number[]): number | null {
  const n = a.length;
  if (n < 3) return null;
  const ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : null;
}

/** 両方が欠けていない行での Spearman の順位相関。計算できなければ null。 */
export function spearman(x: (number | null)[], y: (number | null)[]): number | null {
  const a: number[] = [], b: number[] = [];
  for (let i = 0; i < x.length; i++) if (x[i] !== null && y[i] !== null) { a.push(x[i] as number); b.push(y[i] as number); }
  return pearson(ranks(a), ranks(b));
}

/** X は [行][列]、rows は開発集合の行番号。 */
export function qcFeatures(X: (number | null)[][], rows: number[], names: string[], o: QcOptions): QcResult {
  const removed: QcRemoval[] = [];
  let cand = names.map((_, j) => j);
  const unknownExcluded: string[] = [];
  if (o.exclude) {
    const ex = new Set(o.exclude);
    for (const nm of ex) if (!names.includes(nm)) unknownExcluded.push(nm);
    cand = cand.filter((j) => {
      if (!ex.has(names[j])) return true;
      removed.push({ feature: names[j], reason: "除外リストにある" });
      return false;
    });
  }
  const colOf = (j: number) => rows.map((i) => X[i][j]);
  if (o.dropConstant) {
    cand = cand.filter((j) => {
      const vals = colOf(j).filter((v): v is number => v !== null);
      if (new Set(vals).size >= 2) return true;
      removed.push({ feature: names[j], reason: vals.length ? "開発集合で値が 1 種類しかない" : "開発集合で全部欠損" });
      return false;
    });
  }
  if (o.correlationThreshold !== null) {
    const thr = o.correlationThreshold;
    const cols = new Map(cand.map((j) => [j, colOf(j)]));
    const miss = (j: number) => cols.get(j)!.filter((v) => v === null).length;
    const order = [...cand].sort((a, b) => miss(a) - miss(b) || a - b);
    const kept: number[] = [];
    for (const j of order) {
      let hit: { k: number; r: number } | null = null;
      for (const k of kept) {
        const r = spearman(cols.get(j)!, cols.get(k)!);
        if (r !== null && Math.abs(r) >= thr) { hit = { k, r }; break; }
      }
      if (hit) removed.push({ feature: names[j], reason: `「${names[hit.k]}」と |ρ| = ${Math.abs(hit.r).toFixed(3)} ≥ ${thr}` });
      else kept.push(j);
    }
    cand = kept.sort((a, b) => a - b);
  }
  return { keep: cand, removed, unknownExcluded };
}
