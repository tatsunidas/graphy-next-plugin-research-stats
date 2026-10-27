/**
 * テスト集合の固定ファイル（JSON）。テスト集合に入れた症例と、テスト集合で評価した履歴を持ち運ぶ。
 *
 * 固定するもの: 対象の症例（ID と結果）の指紋と、テスト症例の ID。対象が違うデータでは使えない。
 * 特徴量は固定しない（同じテスト集合で別の特徴量を試すことはあり、そのたびに評価の回数が増える）。
 * 評価のたびに回数を 1 増やした新しいファイルを書き出す。
 *
 * 全体の SHA-256 は「手で書き換えた・壊れた」を見つけるためのもので、改ざんは防げない
 * （作り直せば合う。古いファイルを読ませれば回数は戻る）。画面と README にそう書く。
 */
import { sha256Hex } from "../digest";

export const LOCK_FORMAT = "research-stats-test-lock";
export const LOCK_VERSION = 1;

export interface LockEvaluation {
  /** 何回目の評価か（1 から）。 */
  n: number;
  at: string;
  /** 評価に使った特徴量の表の指紋。 */
  featuresFingerprint: string;
  /** 主な結果（AUC・C など）。 */
  summary: Record<string, number | string | null>;
}

export interface LockBody {
  format: typeof LOCK_FORMAT;
  version: typeof LOCK_VERSION;
  createdAt: string;
  pluginVersion: string;
  outcomeKind: "binary" | "survival";
  seed: number;
  testFraction: number;
  /** 層化に使ったもの（"outcome"・"event"）。層化しなければ null。 */
  stratifiedBy: string | null;
  cohortFingerprint: string;
  cohortSize: number;
  /** 並べ替えた ID。 */
  testIds: string[];
  evaluations: LockEvaluation[];
}

export interface LockFile extends LockBody {
  sha256: string;
}

/** キーを並べた JSON（同じ中身なら同じ文字列）。 */
export function canonicalJson(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(canonicalJson).join(",")}]`;
  if (v !== null && typeof v === "object") {
    const o = v as Record<string, unknown>;
    return `{${Object.keys(o).filter((k) => o[k] !== undefined).sort().map((k) => `${JSON.stringify(k)}:${canonicalJson(o[k])}`).join(",")}}`;
  }
  if (typeof v === "number" && !Number.isFinite(v)) throw new RangeError("有限でない数は指紋にできません");
  return JSON.stringify(v);
}

async function hashOf(v: unknown): Promise<string> {
  const h = await sha256Hex(new TextEncoder().encode(canonicalJson(v)));
  if (h === null) throw new Error("この環境では SHA-256 を計算できないため、固定ファイルを扱えません");
  return h;
}

/** 対象の症例の指紋。outcome は症例ごとの結果（二値はラベル、生存は [時間, イベント]）。 */
export function fingerprint(ids: string[], outcome: unknown[]): Promise<string> {
  if (ids.length !== outcome.length) throw new RangeError("ID と結果の長さが違います");
  const rows = ids.map((id, i) => [id, outcome[i]]).sort((a, b) => ((a[0] as string) < (b[0] as string) ? -1 : (a[0] as string) > (b[0] as string) ? 1 : 0));
  return hashOf(rows);
}

export async function sealLock(body: LockBody): Promise<LockFile> {
  return { ...body, sha256: await hashOf(body) };
}

const isStr = (v: unknown): v is string => typeof v === "string";
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** 読み込んで形と全体の SHA-256 を確かめる。 */
export async function parseLock(text: string): Promise<LockFile | { error: string }> {
  let o: Record<string, unknown>;
  try { o = JSON.parse(text); } catch { return { error: "固定ファイルが JSON として読めません" }; }
  if (!o || typeof o !== "object" || o.format !== LOCK_FORMAT) return { error: "テスト集合の固定ファイルではありません" };
  if (o.version !== LOCK_VERSION) return { error: `固定ファイルの版 ${String(o.version)} は扱えません（${LOCK_VERSION} だけ）` };
  const ok = isStr(o.createdAt) && isStr(o.pluginVersion) && (o.outcomeKind === "binary" || o.outcomeKind === "survival") && isNum(o.seed)
    && isNum(o.testFraction) && (o.stratifiedBy === null || isStr(o.stratifiedBy)) && isStr(o.cohortFingerprint) && isNum(o.cohortSize)
    && Array.isArray(o.testIds) && o.testIds.every(isStr) && Array.isArray(o.evaluations) && isStr(o.sha256);
  if (!ok) return { error: "固定ファイルの項目が欠けているか、型が違います" };
  const evs = o.evaluations as LockEvaluation[];
  if (!evs.every((e, q) => e && e.n === q + 1 && isStr(e.at) && isStr(e.featuresFingerprint) && e.summary && typeof e.summary === "object")) {
    return { error: "固定ファイルの評価の履歴が壊れています" };
  }
  const { sha256, ...body } = o as unknown as LockFile;
  if ((await hashOf(body)) !== sha256) return { error: "固定ファイルの中身が書き換えられています（SHA-256 が合いません）。元のファイルを使ってください" };
  return o as unknown as LockFile;
}

/** 今のデータに使えるか。使えなければ理由。 */
export function checkLock(lock: LockFile, outcomeKind: "binary" | "survival", cohortFingerprint: string, cohortIds: string[]): string | null {
  if (lock.outcomeKind !== outcomeKind) return `固定ファイルは結果の型が「${lock.outcomeKind}」のものです`;
  if (lock.cohortFingerprint !== cohortFingerprint) return "固定ファイルを作ったときと、対象の症例（ID・結果）が違います。同じデータで使うか、新しく固定し直してください（作り直したことは報告に書いてください）";
  const ids = new Set(cohortIds);
  const missing = lock.testIds.filter((id) => !ids.has(id));
  if (missing.length) return `テスト症例のうち ${missing.length} 人が今のデータにいません`;
  return null;
}

/** 評価を 1 回足した新しい固定ファイル。 */
export function addEvaluation(lock: LockFile, e: Omit<LockEvaluation, "n">): Promise<LockFile> {
  const { sha256: _old, ...body } = lock;
  return sealLock({ ...body, evaluations: [...body.evaluations, { ...e, n: body.evaluations.length + 1 }] });
}
