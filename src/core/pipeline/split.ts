/**
 * 分割: 層化した k 分割・層化したホールドアウト。乱数は {@link Rng}（同じシードで同じ分割）。
 * 層ごとに症例を並べ替え、層を順に連ねた通し番号を K で割った余りを分割番号にする。
 * こうすると、各分割の大きさの差も、層ごとの人数の差も 1 以内になる。
 */
import type { Rng } from "../rng";

export function shuffle(ix: number[], rng: Rng): number[] {
  const a = [...ix];
  for (let i = a.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const strataKeys = (n: number, strata: number[] | null) => {
  if (strata && strata.length !== n) throw new RangeError("層の長さが症例数と違います");
  return strata ? [...new Set(strata)].sort((a, b) => a - b) : [0];
};

const members = (n: number, strata: number[] | null, key: number) =>
  Array.from({ length: n }, (_, i) => i).filter((i) => (strata ? strata[i] : 0) === key);

/** 症例ごとの分割番号（0..K-1）。strata が null なら層化しない。K = n なら leave-one-out と同じ。 */
export function stratifiedFolds(n: number, strata: number[] | null, K: number, rng: Rng): number[] {
  if (!Number.isInteger(K) || K < 2 || K > n) throw new RangeError(`分割数は 2 以上 ${n} 以下の整数にしてください: ${K}`);
  const fold = new Array<number>(n).fill(-1);
  let pos = 0;
  for (const key of strataKeys(n, strata)) for (const i of shuffle(members(n, strata, key), rng)) fold[i] = pos++ % K;
  return fold;
}

/** 症例ごとに、テスト集合なら true。各層から round(層の人数 × fraction) 人をテストにする。 */
export function stratifiedHoldout(n: number, strata: number[] | null, fraction: number, rng: Rng): boolean[] {
  if (!(fraction > 0 && fraction < 1)) throw new RangeError(`テストの割合は 0 より大きく 1 より小さくしてください: ${fraction}`);
  const test = new Array<boolean>(n).fill(false);
  for (const key of strataKeys(n, strata)) {
    const m = shuffle(members(n, strata, key), rng);
    const t = Math.round(m.length * fraction);
    for (let q = 0; q < t; q++) test[m[q]] = true;
  }
  return test;
}
