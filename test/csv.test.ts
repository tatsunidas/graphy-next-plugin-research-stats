import { describe, expect, it } from "vitest";
import { CsvError, numberOrNull, parseCell, parseCsv, toCsv } from "../src/core/csv";

describe("parseCsv", () => {
  it("reads quoted commas, quotes and line breaks inside a field", () => {
    const t = parseCsv('id,note\r\na,"x,y"\r\nb,"he said ""hi""\nsecond line"\r\n');
    expect(t.header).toEqual(["id", "note"]);
    expect(t.rows).toEqual([["a", "x,y"], ["b", 'he said "hi"\nsecond line']]);
  });
  it("accepts a BOM, LF-only lines and no final newline", () => {
    const t = parseCsv("﻿a,b\n1,2\n3,4");
    expect(t.header).toEqual(["a", "b"]);
    expect(t.rows).toEqual([["1", "2"], ["3", "4"]]);
  });
  it("keeps empty fields as empty strings", () => {
    expect(parseCsv("a,b,c\n1,,3\n").rows).toEqual([["1", "", "3"]]);
  });
  it("stops with the line number when a row has the wrong number of columns (negative)", () => {
    expect(() => parseCsv("a,b\n1,2\n3\n")).toThrow(/3 行目/);
    expect(() => parseCsv("a,b\n1,2,3\n")).toThrow(CsvError);
  });
  it("refuses an unclosed quote, an empty file and duplicate or empty column names", () => {
    expect(() => parseCsv('a,b\n1,"2\n')).toThrow(/引用符/);
    expect(() => parseCsv("")).toThrow(/空/);
    expect(() => parseCsv("a,a\n1,2\n")).toThrow(/同じ列名/);
    expect(() => parseCsv("a,\n1,2\n")).toThrow(/空の列名/);
  });
});

describe("cells", () => {
  it("reads missing markers as missing, never as 0", () => {
    for (const s of ["", " ", "NA", "na", "N/A", "NaN", "null", "None"]) {
      expect(parseCell(s).kind).toBe("missing");
      expect(numberOrNull(s)).toBeNull();
    }
  });
  it("reads decimal numbers, exponents and signs", () => {
    expect(numberOrNull("12")).toBe(12);
    expect(numberOrNull("-0.5")).toBe(-0.5);
    expect(numberOrNull(".5")).toBe(0.5);
    expect(numberOrNull("1e3")).toBe(1000);
    expect(numberOrNull("+2.50")).toBe(2.5);
  });
  it("does not read other text as a number (negative)", () => {
    for (const s of ["0x10", "1,5", "abc", "Infinity", "1 2", "--1"]) expect(numberOrNull(s)).toBeUndefined();
  });
});

describe("toCsv", () => {
  it("writes CRLF, quotes where needed and leaves null empty", () => {
    expect(toCsv(["a", "b"], [[1, null], ['x,"y"', "z"]])).toBe('a,b\r\n1,\r\n"x,""y""",z\r\n');
  });
  it("round-trips through parseCsv", () => {
    const rows = [["p1", "1.5", ""], ["p2", "a\nb", "3"]];
    const back = parseCsv(toCsv(["id", "x", "y"], rows));
    expect(back.rows).toEqual(rows);
  });
});
