/**
 * 最適しきい値の探索（log-rank のχ²が最大のしきい値）と、並べ替え検定による補正 p。
 *
 * 補正 p: 「しきい値を探して最大のχ²を取る」手順そのものを、特徴量の値を症例間で並べ替えて
 * B 回繰り返し、観測の最大χ²以上になった割合 (1 + 該当数) / (B + 1) を p とする。
 * 探索で p が小さく出る分（多重比較）を含めた p になる。
 */
import { chi2Sf } from "./dist";
import type { Rng } from "../rng";
import type { Surv } from "./types";

export interface ScanResult {
  cut: number;
  chi2: number;
  /** 候補が無い（群の最小割合を満たすしきい値が無い）とき true。 */
  empty: boolean;
}

export class CutScanner {
  /** 時間の昇順に並べた症例番号。 */
  readonly order: number[];
  private readonly groups: { start: number; end: number; d: number }[] = [];
  private readonly eventInOrder: Uint8Array;
  readonly n: number;

  constructor(s: Surv, private readonly minFraction: number) {
    this.n = s.time.length;
    this.order = Array.from({ length: this.n }, (_, i) => i).sort((a, b) => s.time[a] - s.time[b]);
    this.eventInOrder = Uint8Array.from(this.order.map((i) => s.event[i]));
    for (let i = 0; i < this.n;) {
      let j = i, d = 0;
      while (j < this.n && s.time[this.order[j]] === s.time[this.order[i]]) { d += s.event[this.order[j]]; j++; }
      this.groups.push({ start: i, end: j, d });
      i = j;
    }
  }

  /** 元の並びの値を、時間の昇順に並べ替える。 */
  inTimeOrder(values: number[]): number[] {
    return this.order.map((i) => values[i]);
  }

  private chi2For(high: Uint8Array): number {
    let nHigh = 0, nTot = 0, O = 0, E = 0, V = 0;
    for (let gi = this.groups.length - 1; gi >= 0; gi--) {
      const g = this.groups[gi];
      let dHigh = 0;
      for (let q = g.start; q < g.end; q++) {
        nTot++;
        if (high[q]) { nHigh++; if (this.eventInOrder[q]) dHigh++; }
      }
      if (g.d === 0) continue;
      const f = nHigh / nTot;
      O += dHigh;
      E += g.d * f;
      if (nTot > 1) V += (g.d * (nTot - g.d) / (nTot - 1)) * f * (1 - f);
    }
    return V > 0 ? (O - E) ** 2 / V : 0;
  }

  /** 時間順に並べた値に対し、最大のχ²を与えるしきい値を返す。 */
  scan(valuesInTimeOrder: number[]): ScanResult {
    const v = valuesInTimeOrder;
    const sorted = [...v].sort((a, b) => a - b);
    const minCount = Math.max(1, Math.ceil(this.minFraction * this.n));
    const high = new Uint8Array(this.n);
    let best: ScanResult = { cut: NaN, chi2: -1, empty: true };
    for (let k = minCount - 1; k < this.n - minCount; k++) {
      if (sorted[k] === sorted[k + 1]) continue; // 同じ値の間では分けられない
      const cut = (sorted[k] + sorted[k + 1]) / 2;
      for (let i = 0; i < this.n; i++) high[i] = v[i] > cut ? 1 : 0;
      const chi2 = this.chi2For(high);
      if (chi2 > best.chi2) best = { cut, chi2, empty: false };
    }
    return best;
  }
}

export interface PermutationResult {
  observed: ScanResult;
  rawP: number;
  correctedP: number;
  permutations: number;
}

/** 並べ替えを少しずつ進める（画面を固めないよう、呼び出し側が間に制御を返す）。 */
export class PermutationRun {
  private readonly values: number[];
  private done = 0;
  private atLeast = 0;
  readonly observed: ScanResult;

  constructor(private readonly scanner: CutScanner, valuesOriginalOrder: number[], readonly B: number, private readonly rng: Rng) {
    this.values = scanner.inTimeOrder(valuesOriginalOrder);
    this.observed = scanner.scan(this.values);
  }

  get finished(): boolean { return this.done >= this.B; }
  get progress(): number { return this.B === 0 ? 1 : this.done / this.B; }

  advance(count: number): void {
    const v = [...this.values];
    for (let c = 0; c < count && this.done < this.B; c++, this.done++) {
      for (let i = v.length - 1; i > 0; i--) {
        const j = this.rng.int(i + 1);
        [v[i], v[j]] = [v[j], v[i]];
      }
      if (this.scanner.scan(v).chi2 >= this.observed.chi2 - 1e-9) this.atLeast++;
    }
  }

  result(): PermutationResult | { error: string } {
    if (this.observed.empty) return { error: "しきい値の候補がありません（各群の最小割合を満たす分け方が無い）" };
    return {
      observed: this.observed,
      rawP: chi2Sf(this.observed.chi2, 1),
      correctedP: (1 + this.atLeast) / (this.done + 1),
      permutations: this.done,
    };
  }
}

export function permutationCorrectedP(scanner: CutScanner, values: number[], B: number, rng: Rng): PermutationResult | { error: string } {
  const run = new PermutationRun(scanner, values, B, rng);
  run.advance(B);
  return run.result();
}
