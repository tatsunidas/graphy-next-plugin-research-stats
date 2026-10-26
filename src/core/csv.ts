/**
 * CSV の読み書き（RFC 4180）。引用符の中のカンマ・改行、二重引用符のエスケープ、BOM、CRLF／LF を扱う。
 *
 * 欠損は**欠損のまま**持つ（0 や空文字列に変えない）。解析の側が「欠損を使う症例を除く」と決める。
 */

export interface CsvTable {
  header: string[];
  rows: string[][];
}

export class CsvError extends Error {}

/** 欠損を表す記法（大文字小文字は区別しない。空欄も欠損）。 */
const MISSING = new Set(["", "na", "n/a", "nan", "null", "none"]);

export type Cell = { kind: "number"; value: number } | { kind: "missing" } | { kind: "text"; value: string };

/** 1 セルを解釈する。数として読めないもの（欠損の記法を除く）は text。 */
export function parseCell(raw: string): Cell {
  const s = raw.trim();
  if (MISSING.has(s.toLowerCase())) return { kind: "missing" };
  // 数値: 10 進（指数あり）。"0x10" や "1,5" は数にしない
  if (/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(s)) {
    const v = Number(s);
    if (Number.isFinite(v)) return { kind: "number", value: v };
  }
  if (/^[+-]?inf(inity)?$/i.test(s)) return { kind: "text", value: s };
  return { kind: "text", value: raw };
}

/** 数または null（欠損）。数でも欠損でもなければ undefined（呼び出し側が不正として扱う）。 */
export function numberOrNull(raw: string): number | null | undefined {
  const c = parseCell(raw);
  if (c.kind === "number") return c.value;
  if (c.kind === "missing") return null;
  return undefined;
}

/** CSV 全体を表にする。行の列数が見出しと違えば、行番号つきで止める（黙って詰めない）。 */
export function parseCsv(text: string): CsvTable {
  const src = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  const records: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  let fieldStarted = false;
  const endField = () => {
    row.push(field);
    field = "";
    fieldStarted = false;
  };
  const endRow = () => {
    endField();
    // 末尾の空行は捨てる（中間の空行は列数の検査で拾う）
    if (!(row.length === 1 && row[0] === "")) records.push(row);
    row = [];
  };
  for (let i = 0; i < src.length; i++) {
    const ch = src[i];
    if (inQuotes) {
      if (ch === '"') {
        if (src[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
      continue;
    }
    if (ch === '"' && !fieldStarted) {
      inQuotes = true;
      fieldStarted = true;
    } else if (ch === ",") {
      endField();
    } else if (ch === "\r") {
      if (src[i + 1] === "\n") i++;
      endRow();
    } else if (ch === "\n") {
      endRow();
    } else {
      field += ch;
      fieldStarted = true;
    }
  }
  if (inQuotes) throw new CsvError("引用符が閉じていません（ファイルの末尾に達しました）");
  if (fieldStarted || field !== "" || row.length > 0) endRow();
  if (records.length === 0) throw new CsvError("CSV が空です");
  const header = records[0].map((h) => h.trim());
  if (header.some((h) => h === "")) throw new CsvError("見出しに空の列名があります");
  const dup = header.find((h, i) => header.indexOf(h) !== i);
  if (dup !== undefined) throw new CsvError(`見出しに同じ列名があります: ${dup}`);
  const rows = records.slice(1);
  rows.forEach((r, i) => {
    if (r.length !== header.length) {
      throw new CsvError(`${i + 2} 行目の列数が見出しと違います（${r.length} 列、見出しは ${header.length} 列）`);
    }
  });
  return { header, rows };
}

/** 書き出し用に 1 セルを整える。null・undefined は空欄、数はそのまま、NaN は空欄にしない（"NaN"）。 */
function cellText(v: string | number | null | undefined): string {
  if (v === null || v === undefined) return "";
  const s = typeof v === "number" ? String(v) : v;
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** CSV に書き出す（CRLF・BOM なし。列名は呼び出し側が ASCII を守る）。 */
export function toCsv(header: readonly string[], rows: readonly (readonly (string | number | null | undefined)[])[]): string {
  const lines = [header.map(cellText).join(","), ...rows.map((r) => r.map(cellText).join(","))];
  return `${lines.join("\r\n")}\r\n`;
}
