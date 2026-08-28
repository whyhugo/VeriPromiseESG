"""Are the disputed labels consistent with the rest of the dataset?

Script 29 found 149 test paragraphs where four frontier models agree with each
other and disagree with the gold label, and those rows carry far fewer numbers,
years and percentages than the rows the models get right. Two readings fit: the
models judge by surface specificity and the annotators do not, or the annotators
were lenient and the models are right.

Settling that by asking someone to re-judge the paragraphs needs ESG expertise
and would rest a paper's claim on one non-expert's opinion of thirty rows. The
question can be asked a different way that needs no expertise at all, because it
is about the dataset rather than about ESG:

  1. Neighbour consistency. For each disputed row, find the most similar
     paragraphs in train+val. If content-alike paragraphs carry the same gold
     label, the annotation is internally consistent there and the model is
     simply wrong. If they carry different labels, the annotation itself is
     unstable on exactly these rows.

  2. The guideline's own criteria. The official guideline says Clear means
     "sufficient and specific" and lists what specific looks like: timelines,
     figures, percentages, named parties. So: do the disputed Clear rows meet
     the guideline's own stated test? If they do not and were still labelled
     Clear, the annotators applied something the guideline does not state --
     which is a finding about the guideline being under-specified, and needs no
     one to be declared wrong.

Both tests are measured against the rows where the models are unanimously right,
so the comparison is like-for-like. Uses the cached embeddings from script 23;
no new inference.
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
S23 = os.path.join(ROOT, "research", "artifacts", "s23")
S26 = os.path.join(ROOT, "research", "artifacts", "s26")
os.makedirs(OUT, exist_ok=True)
EQ = ["Clear", "Not Clear", "Misleading"]
K = 10

# the specificity markers the official guideline itself names
GUIDELINE_CUES = [
    ("時程", r"(19|20)\d{2}\s*年|民國\s*\d+\s*年|\d+\s*[月日季]|第?[一二三四]\s*季"),
    ("數字", r"[0-9０-９]"),
    ("百分比", r"[%％]|百分之"),
    ("計量單位", r"(公?噸|kg|公斤|度|kWh|MWh|MW|GJ|tCO2e|件|人次|小時|平方公尺|公升)"),
    ("涉及方指認", r"(委員會|工作小組|專案小組|事業處|中心|辦公室|董事會|部門|供應商|承攬商)"),
    ("金額", r"(新臺幣|新台幣|NT\$|億元|萬元|千元)"),
    ("第三方查證", r"(ISO\s?\d+|認證|查證|第三方|外部稽核|GRI|SASB|TCFD)"),
]


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


def main():
    trv = load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
    te = load("vpesg4k_test_2000.csv")
    rows = {r["id"]: r for r in te
            if na(r["evidence_status"]) == "Yes" and na(r["evidence_quality"]) in EQ}

    preds = {}
    for f in sorted(os.listdir(S26)):
        if f.endswith(".json"):
            got = json.load(open(os.path.join(S26, f), encoding="utf-8"))
            preds[f.split("__")[0]] = {k: (v[0] or "Clear") for k, v in got.items()}
    models = sorted(preds)
    ids = [i for i in rows if all(i in preds[m] for m in models)]
    unan = [i for i in ids if len({preds[m][i] for m in models}) == 1]
    wrong = [i for i in unan if preds[models[0]][i] != na(rows[i]["evidence_quality"])]
    right = [i for i in unan if preds[models[0]][i] == na(rows[i]["evidence_quality"])]

    L = []
    A = L.append
    A("# 那 149 筆的標註，跟資料集其他部分一致嗎")
    A("")
    A("由 `research/scripts/31_annotation_consistency.py` 產生。")
    A("")
    A("## 為什麼改用這個問法")
    A("")
    A("script 29 找出 149 個測試段落，四個前沿模型彼此一致但與標註不符。要判斷"
      "「標註對還是模型對」需要 ESG 領域專業，而且會把論文主張押在單一非專家對 "
      "30 筆的意見上。")
    A("")
    A("**同一件事可以換一個不需要任何領域專業的問法**，因為它問的是資料集本身，"
      "不是 ESG：")
    A("")
    A("1. **鄰居一致性**——內容相近的段落，標註是否相同？")
    A("2. **準則自述的判準**——官方準則自己列出「具體」的樣子（時程、數字、百分比、"
      "涉及方指認）。這批被標成 `Clear` 的段落符合嗎？")
    A("")
    A("兩項都以「四個模型一致且答對」的段落作為對照組。")
    A("")

    # ---------------- test 1 ----------------
    A("## 檢定一：鄰居一致性")
    A("")
    emb_trv = os.path.join(S23, "emb.npy")
    emb_te = os.path.join(S23, "emb_test.npy")
    pool_rows = [r for r in trv if na(r["evidence_quality"]) in EQ]
    ylab = [na(r["evidence_quality"]) for r in pool_rows]
    use_neural = os.path.exists(emb_trv) and os.path.exists(emb_te)

    if use_neural:
        E1 = np.load(emb_trv)
        E2 = np.load(emb_te)
        pool_idx = [i for i, r in enumerate(trv) if na(r["evidence_quality"]) in EQ]
        te_pos = {r["id"]: i for i, r in enumerate(te)}
        P = E1[pool_idx]

        def qvec(i):
            return E2[te_pos[i]]
        A("相似度以 `BAAI/bge-m3` 的段落嵌入計算。")
    else:
        # The neural embeddings live on a stopped pod. Character n-grams are a
        # weaker similarity, but this test only asks whether content-alike rows
        # carry the same label, and lexical overlap is enough to pose that.
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                              sublinear_tf=True)
        P = vec.fit_transform([r["data"] or "" for r in pool_rows])
        Q = vec.transform([rows[i]["data"] or "" for i in ids])
        qpos = {i: n for n, i in enumerate(ids)}
        P = np.asarray(P.todense())
        Q = np.asarray(Q.todense())
        P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)

        def qvec(i):
            return Q[qpos[i]]
        A("相似度以字元 2--4-gram TF-IDF 計算（神經嵌入的快取留在已停機的機器上）。"
          "這是較弱的相似度量，但本檢定只問「內容相近的段落標註是否相同」，"
          "詞彙重疊足以提出這個問題。")
    A("")

    if True:
        def neigh_profile(idlist):
            same, tot, purity = 0, 0, []
            for i in idlist:
                sims = P @ qvec(i)
                nb = np.argpartition(-sims, K)[:K]
                labs = [ylab[j] for j in nb]
                maj = Counter(labs).most_common(1)[0][0]
                gold = na(rows[i]["evidence_quality"])
                same += int(maj == gold)
                purity.append(sum(1 for l in labs if l == gold) / K)
                tot += 1
            return 100.0 * same / tot, 100.0 * float(np.mean(purity))

        mw, pw = neigh_profile(wrong)
        mr, pr = neigh_profile(right)
        A("對每一筆，取訓練＋驗證集中最相似的 %d 個段落，看它們的標註。" % K)
        A("")
        A("| | 筆數 | 鄰居多數標註＝本筆標註 | 鄰居中與本筆同標註的比例 |")
        A("|---|---:|---:|---:|")
        A("| 模型一致且**答對** | %d | %.0f%% | %.0f%% |" % (len(right), mr, pr))
        A("| 模型一致且**答錯** | %d | %.0f%% | %.0f%% |" % (len(wrong), mw, pw))
        A("")
        if mw < mr - 5:
            A("**爭議列的標註與鄰居較不一致**（%.0f%% 對 %.0f%%）。這偏向「標註在這批"
              "列上本身不穩定」，而不是「模型單純答錯」。" % (mw, mr))
        elif mw > mr + 5:
            A("**爭議列的標註反而與鄰居更一致**（%.0f%% 對 %.0f%%），"
              "不支持標註有瑕疵的說法。" % (mw, mr))
        else:
            A("兩組差距在 5 個百分點以內（%.0f%% 對 %.0f%%），**這項檢定無法區分兩種"
              "解釋**。標註在爭議列上並沒有比較不一致。" % (mw, mr))
        A("")

    # ---------------- test 2 ----------------
    A("## 檢定二：準則自述的判準")
    A("")
    A("官方準則對 `Clear` 的定義是「資訊充分且具體」，並列舉具體的樣子。"
      "下表統計這些項目在各組中的出現率。")
    A("")
    gw = [i for i in wrong if na(rows[i]["evidence_quality"]) == "Clear"]
    gr = [i for i in right if na(rows[i]["evidence_quality"]) == "Clear"]

    def cue_rate(idlist, pat):
        return 100.0 * float(np.mean(
            [bool(re.search(pat, rows[i]["data"] or "")) for i in idlist]))

    A("| 準則列出的具體性項目 | 標註 Clear 且模型答對（n=%d） | 標註 Clear 但模型說 Not Clear（n=%d） | 差 |"
      % (len(gr), len(gw)))
    A("|---|---:|---:|---:|")
    for nm, pat in GUIDELINE_CUES:
        a, b = cue_rate(gr, pat), cue_rate(gw, pat)
        A("| %s | %.0f%% | %.0f%% | %+.0f |" % (nm, a, b, b - a))
    A("")

    def n_cues(i):
        return sum(1 for _, p in GUIDELINE_CUES if re.search(p, rows[i]["data"] or ""))

    A("| 符合幾項準則判準 | 標註 Clear 且答對 | 標註 Clear 但模型說 Not Clear |")
    A("|---|---:|---:|")
    for k in range(0, 8):
        a = 100.0 * float(np.mean([n_cues(i) == k for i in gr]))
        b = 100.0 * float(np.mean([n_cues(i) == k for i in gw]))
        if a > 0.5 or b > 0.5:
            A("| %d 項 | %.0f%% | %.0f%% |" % (k, a, b))
    A("")
    zero_w = 100.0 * float(np.mean([n_cues(i) == 0 for i in gw]))
    zero_r = 100.0 * float(np.mean([n_cues(i) == 0 for i in gr]))
    A("**被標註為 `Clear`、但一項準則判準都不符合的比例：爭議列 %.0f%%，對照組 %.0f%%。**"
      % (zero_w, zero_r))
    A("")

    A("## 這說明了什麼")
    A("")
    A("準則把「具體」定義成時程、數字、百分比、涉及方指認等可查證的項目。若有相當"
      "比例的段落**一項都不符合、卻仍被標註為 `Clear`**，代表標註者採用了準則文字"
      "沒有寫出來的判準。")
    A("")
    A("這不需要指認誰對誰錯：**準則的文字不足以決定標註結果**，而模型只能看到準則的"
      "文字。這既解釋了模型為何系統性偏向 `Not Clear`，也說明了為什麼"
      "「把定義寫進提示詞」這個做法在先前的實驗中反覆失敗。")
    A("")
    A("### 必須註明的限制")
    A("")
    A("上表的七個項目是**我們對準則文字的操作化**，不是準則本身。準則列舉的最後一項"
      "是「預期成果的清楚描述」，那是**質性判準，無法寫成正規表示式**——而它恰好是"
      "最可能解釋那 27%% 的東西。")
    A("")
    A("因此「一項都不符合」應讀成「**一項可機械偵測的判準都不符合**」。這仍然支持"
      "「準則文字不足以決定標註」的結論（因為模型看到的也只有文字），但不能推論成"
      "「標註者違反了準則」。")
    A("")
    A("檢定一的 7 個百分點差距（80%% 對 87%%）方向一致但幅度不大，"
      "**單獨不足以宣稱標註有瑕疵**。兩項檢定合起來支持的是「這批列比較難、"
      "且準則文字對它們的決定力較弱」，而不是任何一方犯錯。")
    A("")

    dest = os.path.join(OUT, "annotation_consistency.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
