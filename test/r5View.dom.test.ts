// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { runSkill } from "../src/core/engine";
import { R5_MODEL, type R5State } from "../src/core/skills";
import { makeT } from "../src/ui/i18n";
import { r5Panel } from "../src/ui/r5View";
import { multiRow, opts, roles } from "./helpers/multiRow";

describe("R5 result panel", () => {
  it("shows the corrected C-index with the OOB C-index (ja and en); the OOB line is absent when no bootstrap was run", async () => {
    const d = multiRow(70, 5);
    const run = async (B: number) => (await runSkill(R5_MODEL, d, { ...roles, r5Corr: 0.9, r5Screen: "boost", r5BoostNu: 0.1, r5BootstrapB: B } as never, opts)).state.r5 as R5State;
    const withB = await run(20);
    for (const loc of ["ja", "en"] as const) {
      const el = r5Panel(withB, makeT(loc))!;
      document.body.replaceChildren(el);
      expect(el.querySelector('[data-testid="rs-r5-ccorr"]')).not.toBeNull();
      const oob = el.querySelector('[data-testid="rs-r5-oob"]')!;
      expect(oob.textContent).toMatch(/\d\.\d{3}/);
      expect(oob.textContent).not.toContain("{");
    }
    const none = r5Panel(await run(0), makeT("ja"))!;
    expect(none.querySelector('[data-testid="rs-r5-oob"]')).toBeNull();
    expect(none.querySelector('[data-testid="rs-r5-ccorr"]')).toBeNull();
  }, 120000);
});
