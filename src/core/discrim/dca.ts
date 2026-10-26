/**
 * 決定曲線。しきい確率 pt で「予測確率 >= pt なら介入」としたときの純便益
 * NB = TP/n − FP/n · pt/(1−pt)。基準線は「全員に介入」と「誰にも介入しない（0）」。
 */
export interface DcaPoint {
  threshold: number;
  model: number;
  all: number;
  none: number;
}

export function decisionCurve(pred: number[], label: number[], thresholds: number[]): DcaPoint[] {
  const n = pred.length, prev = label.reduce((a, b) => a + b, 0) / n;
  return thresholds.map((pt) => {
    let tp = 0, fp = 0;
    pred.forEach((p, i) => { if (p >= pt) { if (label[i] === 1) tp++; else fp++; } });
    const w = pt / (1 - pt);
    return { threshold: pt, model: tp / n - (fp / n) * w, all: prev - (1 - prev) * w, none: 0 };
  });
}

export const defaultThresholds = (): number[] => Array.from({ length: 99 }, (_, i) => (i + 1) / 100);
