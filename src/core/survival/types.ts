/** 生存データ。time は正の有限値、event は 1（イベント）か 0（打ち切り）。 */
export interface Surv {
  time: number[];
  event: number[];
}

export function checkSurv(s: Surv): string | null {
  if (s.time.length !== s.event.length) return "時間とイベントの長さが違います";
  if (!s.time.length) return "症例がありません";
  for (let i = 0; i < s.time.length; i++) {
    if (!Number.isFinite(s.time[i]) || s.time[i] <= 0) return `時間は正の有限値にしてください（${i + 1} 番目: ${s.time[i]}）`;
    if (s.event[i] !== 0 && s.event[i] !== 1) return `イベントは 0 か 1 にしてください（${i + 1} 番目: ${s.event[i]}）`;
  }
  return null;
}
