import type { ResearchHost } from "./host/types";
import { toLocale } from "./ui/i18n";
import { mountWizard } from "./ui/wizard";

/** メイン画面のメニュー「研究統計」を押したときに呼ばれる。専用ウィンドウを開いて中身を組み立てる。 */
export function activate(host: ResearchHost): void {
  if (!host.openWindow) {
    host.notify("research-stats: この画面では窓を開けません（メイン画面から開いてください）");
    return;
  }
  const locale = toLocale(host.locale);
  const win = host.openWindow({ title: locale === "ja" ? "研究統計" : "Research Stats", width: 900, height: 760 });
  mountWizard(win.container, locale);
}

export default { activate };
