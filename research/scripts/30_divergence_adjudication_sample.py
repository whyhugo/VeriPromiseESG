"""A sample of the disputed rows, laid out for a person to adjudicate.

Script 29 found 149 test paragraphs where all four frontier models agree with
each other and disagree with the gold label, and that those rows carry far fewer
numbers, years and percentages than the rows where the models are unanimously
right. The reading offered was that the models judge adequacy by surface
specificity while the annotators do not.

That reading assumes the annotators are right and the models wrong. The opposite
account fits the same numbers: the annotators may have been lenient, marking as
Clear some evidence that carries no figures at all, in which case the models are
correct and the finding is about the dataset rather than about the models.

The two lead to opposite conclusions and cannot both be claimed, so this builds
a sheet for human adjudication. It deliberately does not use another model to
decide -- a model asked to arbitrate would bring the same specificity prior that
is under examination.

The sample is stratified over the disagreement types and the specificity range,
so it is not all easy cases or all hard ones. Each row is printed with the
paragraph, both annotated spans, the gold label, what the models said, and the
guideline definitions, with a blank for the verdict.
"""
import csv
import io
import json
import os
import re
import sys
from collections import Counter

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
S26 = os.path.join(ROOT, "research", "artifacts", "s26")
os.makedirs(OUT, exist_ok=True)
SEED = 20260828
N_SAMPLE = 30
SEP = "｜"
EQ = ["Clear", "Not Clear", "Misleading"]

DEFN = {
    "Clear": "資訊充分且具體，足以支持該承諾，內容可理解且合乎邏輯。",
    "Not Clear": "資訊缺漏，以致所述內容只能部分理解，或流於表面。",
    "Misleading": "該證據無論真假，與所主張的承諾之間沒有清楚關聯，看起來像是在轉移注意力。",
}


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


def clean(raw):
    out = []
    for p in (raw or "").split(SEP):
        p = re.sub(r"\s*\d+\s*$", "", p.strip()).strip()
        if p:
            out.append(p)
    return "　／　".join(out)


def digits_per_100(t):
    return 100.0 * len(re.findall(r"[0-9０-９]", t or "")) / max(1, len(t or ""))


def main():
    te = load("vpesg4k_test_2000.csv")
    rows = {r["id"]: r for r in te
            if na(r["evidence_status"]) == "Yes" and na(r["evidence_quality"]) in EQ}

    preds = {}
    for f in sorted(os.listdir(S26)):
        if f.endswith(".json"):
            got = json.load(open(os.path.join(S26, f), encoding="utf-8"))
            preds[f.split("__")[0]] = {k: (v[0] or "") for k, v in got.items()}
    models = sorted(preds)

    ids = [i for i in rows if all(i in preds[m] for m in models)]
    unan_wrong = [i for i in ids
                  if len({preds[m][i] for m in models}) == 1
                  and preds[models[0]][i] != na(rows[i]["evidence_quality"])]

    # stratify: disagreement type x specificity tercile
    rng = np.random.default_rng(SEED)
    groups = {}
    dens = {i: digits_per_100(rows[i]["data"]) for i in unan_wrong}
    cuts = np.percentile([dens[i] for i in unan_wrong], [33, 67])
    for i in unan_wrong:
        gold = na(rows[i]["evidence_quality"])
        said = preds[models[0]][i]
        tier = 0 if dens[i] <= cuts[0] else (1 if dens[i] <= cuts[1] else 2)
        groups.setdefault(("%s->%s" % (gold, said), tier), []).append(i)

    # Proportional allocation by disagreement type, so the verdicts estimate the
    # population rather than merely cover it, with a floor of one per type so no
    # kind of disagreement goes unseen. Within a type, spread over the
    # specificity range, since that is the variable under examination.
    by_type = {}
    for i in unan_wrong:
        by_type.setdefault(
            "%s->%s" % (na(rows[i]["evidence_quality"]), preds[models[0]][i]),
            []).append(i)
    quota = {}
    for t, pool in by_type.items():
        quota[t] = max(1, int(round(N_SAMPLE * len(pool) / len(unan_wrong))))
    while sum(quota.values()) > N_SAMPLE:
        t = max(quota, key=lambda k: (quota[k], len(by_type[k])))
        if quota[t] > 1:
            quota[t] -= 1
        else:
            break

    picked = []
    for t, pool in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        pool = sorted(pool, key=lambda i: dens[i])
        n = min(quota[t], len(pool))
        if n >= len(pool):
            picked += pool
            continue
        # take evenly spaced positions along the specificity ordering
        idx = np.linspace(0, len(pool) - 1, n).round().astype(int)
        picked += [pool[j] for j in dict.fromkeys(idx.tolist())]

    L = []
    A = L.append
    A("# 待人工裁決：模型一致、但與標註不符的段落")
    A("")
    A("由 `research/scripts/30_divergence_adjudication_sample.py` 產生。")
    A("")
    A("## 這份清單要判斷什麼")
    A("")
    A("script 29 找出 149 個測試段落，四個前沿商用語言模型（%s）"
      "**彼此判斷完全一致，但與人工標註不符**。" % "、".join(models))
    A("")
    A("那批段落的數字密度明顯較低（每 100 字 2.00 個數字，對照組是 6.18），"
      "因此我先前的解讀是「模型把表面具體性當判準，標註者不是」。")
    A("")
    A("**但同一組數字有另一個解釋**：標註者可能過於寬鬆，把沒有任何數據的佐證也標成"
      "`Clear`，那麼模型才是對的，而這個發現就變成是關於**資料集**而非關於**模型**的。")
    A("")
    A("| 若標註正確 | 若標註有瑕疵 |")
    A("|---|---|")
    A("| 模型用錯判準（看具體性而非相關性） | 資料集的 `Clear` 定義不一致 |")
    A("| → 這是關於模型的發現 | → 這是關於資料集的發現 |")
    A("")
    A("兩者都有價值，但**不能兩者都宣稱**。所以需要人工逐筆判斷。")
    A("")
    A("> 刻意不用另一個語言模型來裁決——被檢驗的正是「具體性優先」這個偏好，"
      "而任何語言模型都可能帶著同樣的偏好。")
    A("")

    A("## 官方準則的三個定義")
    A("")
    for k in EQ:
        A("- **%s**：%s" % (k, DEFN[k]))
    A("")

    A("## 怎麼填")
    A("")
    A("每一筆看完後，在「你的判斷」欄填入下列之一：")
    A("")
    A("| 填入 | 意思 |")
    A("|---|---|")
    A("| `標註對` | 標註是對的，模型錯了 |")
    A("| `模型對` | 模型是對的，標註有瑕疵 |")
    A("| `都可` | 兩個判斷都說得過去，這一筆本質模糊 |")
    A("| `不確定` | 資訊不足以判斷 |")
    A("")
    A("樣本 %d 筆，**依分歧類型的母體比例抽樣**（每種類型至少 1 筆），"
      "型內再依數字密度均勻取樣。因此你的判斷結果可以直接推估到全部 %d 筆。"
      % (len(picked), len(unan_wrong)))
    A("")
    A("---")
    A("")

    for n, i in enumerate(picked, 1):
        r = rows[i]
        gold = na(r["evidence_quality"])
        said = preds[models[0]][i]
        A("## %d. id `%s`　　標註：**%s**　／　四個模型一致：**%s**" % (n, i, gold, said))
        A("")
        A("- 公司：%s（%s）　段落 %d 字，每 100 字 %.1f 個數字"
          % (r.get("company", ""), r.get("ticker", ""),
             len(r["data"] or ""), digits_per_100(r["data"])))
        A("")
        A("**承諾（標註跨度）**")
        A("")
        A("> %s" % (clean(r.get("promise_string")) or "（無）"))
        A("")
        A("**佐證（標註跨度）**")
        A("")
        A("> %s" % (clean(r.get("evidence_string")) or "（無）"))
        A("")
        A("<details><summary>完整段落</summary>")
        A("")
        A("> %s" % (r["data"] or "").replace("\n", " "))
        A("")
        A("</details>")
        A("")
        A("**你的判斷**：`________`　備註：")
        A("")
        A("---")
        A("")

    A("## 統計用：分歧類型的分布")
    A("")
    A("| 分歧類型 | 全部 149 筆中 | 本樣本中 |")
    A("|---|---:|---:|")
    allc = Counter("%s→%s" % (na(rows[i]["evidence_quality"]), preds[models[0]][i])
                   for i in unan_wrong)
    smpc = Counter("%s→%s" % (na(rows[i]["evidence_quality"]), preds[models[0]][i])
                   for i in picked)
    for k, v in allc.most_common():
        A("| %s | %d | %d |" % (k, v, smpc.get(k, 0)))
    A("")
    A("填完之後，如果 `標註對` 佔多數，第二部分關於「模型判準」的結論成立；"
      "如果 `模型對` 佔多數，結論要改寫成關於資料集標註一致性的發現；"
      "如果 `都可` 佔多數，則兩者都不宜宣稱，該類別的邊界本身就是模糊的。")
    A("")

    dest = os.path.join(OUT, "divergence_adjudication_sample.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("樣本 %d 筆，來自 %d 筆一致但錯的段落" % (len(picked), len(unan_wrong)))
    for k, v in allc.most_common():
        print("  %-24s 全部 %3d  樣本 %2d" % (k, v, smpc.get(k, 0)))


if __name__ == "__main__":
    main()
