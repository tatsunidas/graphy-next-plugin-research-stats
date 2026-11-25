# 実データでの Skill 検証（NSCLC Radiogenomics 56 例）

合成データの試験（`test/*.test.ts`・`tools/gen_*_fixture.py`）とは別に、**実データ**で `src/core` の各 Skill（データセットの確認・R1〜R3・P4・R5）を、**TypeScript を見ずに定義から書いた Python の参照実装**と突き合わせる。UI は通さない（UI の配線は実アプリで別に確かめる）。製品のコード・既存の試験・fixture は変えない。

## 構成

| ファイル | 役割 |
| --- | --- |
| `run_core.mts` / `run_core.sh` | `runSkill` と各 Skill を Node から直接呼び、固定シードで結果（書き出し機能と同じ CSV と、状態の要約）を JSON に出す。実行する設定（シード・分割数・しきい値・負例の入れ替え）は、すべて `run_core.mts` の `RUNS` と出力 JSON の `params` に残る |
| `verify_real.py` | JSON を読み、参照実装（lifelines・scipy・scikit-learn・statsmodels・pingouin と、定義から書いた自前の Cox（Efron）・log-rank・DeLong・C-index 等）と突き合わせて `REPORT.md` を書く。合格・失敗・未達（前提待ち）を分けて数える |

## 環境変数

| 名前 | 内容 |
| --- | --- |
| `RS_VERIFY_DATA` | `LGN-NSCLC-EGFR/runs`（読み取りだけ。コピーしない）。`default56/nsclc.csv`・`iso1mm_bw25/nsclc.csv`・`skill_verify/clinical_surv56_subject.csv` を使う |
| `RS_VERIFY_OUT` | `run_core` の出力先 = `verify_real.py` の入力（JSON）。リポジトリの外に置く |
| `RS_VERIFY_REPORT` | `REPORT.md` の書き出し先（既定 `RS_VERIFY_OUT/REPORT.md`） |
| `RS_VERIFY_BUNDLE` | （任意）esbuild で束ねた `run_core.mjs` の置き場 |

## 必要な環境

- Node 20 以上と、リポジトリの `npm ci` 済みの `node_modules`（esbuild を使う。`tsx` は要らない）。
- Python 3.10 以上。numpy・scipy・pandas・scikit-learn・statsmodels に、`lifelines`・`pingouin`。システムの環境を変えないよう、使い捨ての venv に入れる:

```
python3 -m venv --system-site-packages /tmp/rs-venv     # 既に numpy・scipy 等があるとき。無ければ --system-site-packages を外して pip で全部入れる
/tmp/rs-venv/bin/pip install lifelines pingouin
```

実行時の版（今回: lifelines 0.30.3・pingouin 0.5.5・scikit-learn 1.6.1・statsmodels 0.15.0・scipy 1.15.3・pandas 2.3.3）は `REPORT.md` の先頭に出る。

## 実行

```
export RS_VERIFY_DATA=/media/<...>/LGN-NSCLC-EGFR/runs
export RS_VERIFY_OUT=/tmp/rs-verify-out
export RS_VERIFY_REPORT=/tmp/rs-verify-out/REPORT.md

# 1. コアの実行（全部で約 30 分。R5 の LASSO-Cox のブートストラップが 1 回約 2 分で最も重い）
timeout 3600 tools/verify_real/run_core.sh                       # 全部
timeout 600  tools/verify_real/run_core.sh '^(dc|r1_|r2_|r3)'    # 引数は run id の正規表現（一部だけ）

# 2. 突き合わせ（約 3〜5 分）
timeout 1500 /tmp/rs-verify-out/../rs-venv/bin/python -I tools/verify_real/verify_real.py
```

同じコマンドを 2 回実行すると、同じ結果（固定シード）になる（`*_again` の run が 2 回目で、表と出力の完全一致を検査する）。

## 何を比べるか

- **Dataset check**: 結合の人数・片方だけの患者・欠損の列を `pandas` で数え直す。臨床の表から 3 人を除き ID を 1 つ小文字にした表（負例）で人数と警告が変わる。
- **R1**: 各特徴量（中央値・三分位）の KM（群別の生存確率・Greenwood の se・log(−log) 区間・人数）、log-rank、Cox（群の HR・SD あたり HR）、BH。最適しきい値は log-rank χ² の総当たりと、自前の並べ替え検定（B=1000）。共変量つきの多変量 Cox・C-index・追加価値の検定と、自前のブートストラップによる補正 C-index。
- **R2**（結果 EGFR・陽性 Mutant）: AUC・DeLong（総当たりのペア比較）・Mann–Whitney・BH・最適な cut の混同行列と Clopper–Pearson・上位 k のロジスティック回帰・LOO（選択を各折でやり直す）。
- **R3**（条件 A = default56、条件 B = iso1mm_bw25）: ICC（pingouin の ICC2・ICC2k と、McGraw–Wong の F 近似の自前式）・Bland–Altman。
- **P4**（入れ子の交差検証。2 値 EGFR・生存 OS。topk・lasso）: (a) 2 回実行で同じ、(b) 外側の分割が全症例を 1 回ずつ検証に使い・訓練と重ならない、(c) 出力された予測から AUC・C-index を再計算して一致、(d) ラベルの入れ替え（負例）。前処理（分散 0・相関・p 値による除外）を有効にした run でも (b)(c)(d)。
- **R5**（生存。boost・lasso）: 選ばれた特徴量だけで Cox に当て直して係数・p・C-index を比べ、後退選択の規則を確かめる。時間・イベントの組を入れ替えた負例で補正 C-index。

## 許容の決め方

数値の許容は実装の性質から決める。閉じた式（KM・log-rank・AUC・ICC・Bland–Altman など）は倍精度の丸め（1e-9〜1e-12）、p 値は実装が自前の不完全ガンマ関数を使うため相対 1e-8。Cox は実装の停止規則（部分対数尤度の変化 < 1e-10(|ll|+1)）が許す係数のずれ √(2ε·(I⁻¹)jj) を特徴量ごとに計算して許容にする。乱数を使う量（並べ替え p・補正 C-index・並べ替えの負例）は、再現できないので、独立に実装した推定との差を標準誤差で測る。

## 注意

- C-index の同時刻の扱いが参照によって違う（実装は同時刻の組を数えない。lifelines・R の `survival::concordance` は「イベント vs 打ち切り」の同時刻の組を数える）。`REPORT.md` では、自前の総当たりで両方を再現して原因を特定している。
- `lifelines` の Cox は、歪みの強い共変量（z の最大が 7 程度）で最大値に届かないことがある。Cox の基準には、定義から書いた自前の Newton 法（Efron）を使い、lifelines は交差確認に使う。
- データ（特徴量・臨床の表）と `REPORT.md` はリポジトリに入れない。患者に紐づく値を持ち込まない。
