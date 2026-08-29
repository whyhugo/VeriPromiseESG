"""When the models disagree with the annotators, what reason do they give?

Scripts 33 and 34 inferred the mechanism from the outside. The rows on which
four frontier models agree with each other and differ from the annotators carry
fewer numbers, years, units and percentages, and a logistic regression on those
cues reproduces the models' consensus at 0.786 balanced accuracy against 0.690
for the gold label. That is correlational: the cues co-vary with a great deal
else, and nothing so far rules out the models attending to something different
that merely tracks them.

This asks the models directly. Each is given the same paragraph and the same
options as in script 26, but is now required to return a one-sentence reason
along with the label. The reasons are then coded automatically into criterion
categories and the rates compared between the disputed rows and a control
sample of rows the models got right.

Two of the categories are the criteria the organisers actually wrote down in
`data/DATA.md` -- vague wording, and the strength of the link to the promise.
One is the criterion they never mention and that script 34 found to explain the
models best -- the absence of concrete figures. The question is which of the
three the models reach for, and whether they reach for it more on the rows where
they and the annotators part company.

A second reading comes free. The label is requested again here, so comparing it
with the cached label from script 26 measures how stable these judgements are
under re-query, on rows that by construction were unanimous the first time.

Keys come from the environment and are never written to disk. Every response is
cached by (model, row) so an interrupted run resumes without paying twice.
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
S26 = os.path.join(ROOT, "research", "artifacts", "s26")
CACHE = os.path.join(ROOT, "research", "artifacts", "s35")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

SEED = 20260829
N_CONTROL = 200
EQ = ["Clear", "Not Clear", "Misleading"]
LABELS = ["N/A", "Not Clear", "Clear", "Misleading"]

# Identical wording to script 26, so the judgement task is unchanged.
OPTS = [
    "N/A：沒有承諾或沒有佐證，不適用",
    "Not Clear：資訊缺漏，內容只能部分理解，或流於表面",
    "Clear：具體且充分，足以支持該承諾，內容可理解且合乎邏輯",
    "Misleading：該佐證無論真假，與所主張的承諾之間沒有清楚關聯，看起來像在轉移注意力",
]
SYS = ("你是台灣上市公司永續報告書的稽核專家。"
       "先判斷標籤，再用一句話說明你判斷的**依據**——你是看了段落中的什麼，"
       "才做出這個判斷的。只輸出 JSON，不要任何其他文字。")

# --- criterion categories ----------------------------------------------------
# Coding a Chinese reason by keyword needs negation scope, not a fixed window.
# "未指出具體風險項目、情境假設、執行頻率、負責單位或任何量化結果" negates every
# item in the list, and the last one is twenty characters from the negator. So a
# reason is split into segments at sentence ends and contrastive markers, and
# within a segment everything after a negator counts as negated until the
# segment ends.
NEGATORS = (r"(未能|未見|未曾|未|沒有|沒|無法|尚無|並無|不具|不足|不夠|缺乏|欠缺|"
            r"欠奉|付之闕如|匱乏|薄弱|有限|無)")   # bare 無 last: 無法 must win
SEG_SPLIT = r"[。！？!?；;]|但是|但|然而|不過|惟(?!有)|雖然|雖"
POSITIVE = r"(提供|列出|載明|附有|具備|揭露|說明|提及|已有|有)"

# Objects, by criterion. The first two are the criteria data/DATA.md actually
# states; the third is the one it never mentions and that script 34 found to
# explain the models best. The rest are here so the first three are not
# credited with everything. Categories are not mutually exclusive.
OBJ = {
    "stated: link to the promise":
        r"(承諾|所述承諾|目標|主張)",
    "unstated: no concrete figures":
        r"(數字|數據|量化|統計|指標|比例|百分比|金額|數值|績效數|成果數)",
    "no timeline":
        r"(時程|期程|時間表|年份|完成時間|期限|時間點|截止)",
    "insufficient detail":
        r"(細節|做法|方法|措施|步驟|執行方式|具體內容|執行情形|實際作為|機制)",
    "no third-party assurance":
        r"(第三方|外部稽核|外部查證|查證|認證|驗證|稽核)",
}
# Vague wording is lexical rather than a negated object: the reason says the
# text is vague, it does not say something is missing.
VAGUE = (r"(模糊|籠統|空泛|含糊|抽象|流於表面|表面|口號|概括|概念性|概念說明|"
         r"僅(泛稱|宣稱|敘述|概述|說明|提及|列出|描述|表示|指出)|停留在)")
CATS = [(k, None) for k in
        ("stated: vague wording", "stated: link to the promise",
         "unstated: no concrete figures", "no timeline",
         "insufficient detail", "no third-party assurance",
         "cites concrete figures present")]


def code(reason):
    """Which criteria does this reason invoke? Returns a set of category names."""
    hit = set()
    if re.search(VAGUE, reason):
        hit.add("stated: vague wording")
    for seg in re.split(SEG_SPLIT, reason):
        if not seg:
            continue
        m = re.search(NEGATORS, seg)
        neg = seg[m.end():] if m else ""
        pos = seg if not m else seg[:m.start()]
        for name, pat in OBJ.items():
            if neg and re.search(pat, neg):
                hit.add(name)
        if re.search(POSITIVE, pos) and re.search(OBJ["unstated: no concrete figures"], pos):
            hit.add("cites concrete figures present")
    # "no link to the promise" needs the promise mentioned under negation, but
    # the plain word 承諾 under negation also fires for "no promise stated";
    # require a relational word somewhere in the reason to keep it honest.
    if "stated: link to the promise" in hit and not re.search(
            r"(關聯|連結|相關|對應|支持|支撐|呼應|扣合|無關|偏題)", reason):
        hit.discard("stated: link to the promise")
    return hit

_lock = threading.Lock()
_usage = {}


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


def post(url, payload, headers, tries=5):
    body = json.dumps(payload).encode("utf-8")
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers,
                                         method="POST")
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            txt = e.read()[:200].decode("utf-8", "replace")
            if e.code in (429, 500, 502, 503, 529) and k < tries - 1:
                time.sleep(3 * (k + 1))
                continue
            return {"_err": "HTTP %s %s" % (e.code, txt)}
        except Exception as e:
            if k < tries - 1:
                time.sleep(3 * (k + 1))
                continue
            return {"_err": "%s: %s" % (type(e).__name__, str(e)[:150])}
    return {"_err": "exhausted"}


def note(model, d):
    with _lock:
        u = _usage.setdefault(model, {"in": 0, "out": 0, "calls": 0, "err": 0})
        u["calls"] += 1
        if isinstance(d, dict):
            us = d.get("usage") or d.get("usageMetadata") or {}
            u["in"] += us.get("input_tokens") or us.get("prompt_tokens") or \
                us.get("promptTokenCount") or 0
            u["out"] += us.get("output_tokens") or us.get("completion_tokens") or \
                us.get("candidatesTokenCount") or 0


def ask(model, prompt):
    if model.startswith("claude"):
        d = post("https://api.anthropic.com/v1/messages",
                 {"model": model, "max_tokens": 400, "system": SYS,
                  "messages": [{"role": "user", "content": prompt}]},
                 {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                  "anthropic-version": "2023-06-01",
                  "content-type": "application/json"})
        note(model, d)
        if "_err" in d:
            return None, d["_err"]
        return "".join(b.get("text", "") for b in d.get("content", [])), None

    if model.startswith("gpt"):
        d = post("https://api.openai.com/v1/responses",
                 {"model": model, "max_output_tokens": 800,
                  "input": [{"role": "system", "content": SYS},
                            {"role": "user", "content": prompt}]},
                 {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                  "content-type": "application/json"})
        note(model, d)
        if "_err" in d:
            return None, d["_err"]
        txt = []
        for o in d.get("output", []):
            for c in o.get("content", []) or []:
                if c.get("type") in ("output_text", "text"):
                    txt.append(c.get("text", ""))
        return "".join(txt), None

    d = post("https://generativelanguage.googleapis.com/v1beta/models/"
             "%s:generateContent?key=%s" % (model, os.environ["GEMINI_API_KEY"]),
             {"systemInstruction": {"parts": [{"text": SYS}]},
              "contents": [{"role": "user", "parts": [{"text": prompt}]}],
              "generationConfig": {"maxOutputTokens": 800}},
             {"content-type": "application/json"})
    note(model, d)
    if "_err" in d:
        return None, d["_err"]
    try:
        parts = d["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts), None
    except Exception:
        return "", None


def build(row):
    opts = "\n".join("- " + o for o in OPTS)
    return ("段落：\n%s\n\n問題：這個段落所附的佐證，品質如何？\n\n"
            "可選標籤：\n%s\n\n"
            "輸出格式（只輸出這個 JSON，不要程式碼區塊）：\n"
            '{"label": "標籤名稱", "reason": "一句話說明你依據段落中的什麼做出判斷"}'
            % ((row["data"] or "")[:1500], opts))


def parse(txt):
    if not txt:
        return None, ""
    t = txt.strip()
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            lab = str(d.get("label") or "").strip()
            for L in sorted(LABELS, key=len, reverse=True):
                if re.fullmatch(re.escape(L), lab, re.I):
                    return L, str(d.get("reason") or "").strip()
            for L in sorted(LABELS, key=len, reverse=True):
                if re.search(re.escape(L), lab, re.I):
                    return L, str(d.get("reason") or "").strip()
            return None, str(d.get("reason") or "").strip()
        except Exception:
            pass
    lab = None
    for L in sorted(LABELS, key=len, reverse=True):
        if re.search(re.escape(L), t, re.I):
            lab = L
            break
    return lab, t


def two_prop_p(k1, n1, k2, n2):
    from math import erfc, sqrt
    if n1 == 0 or n2 == 0:
        return 1.0
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(max(1e-12, p * (1 - p) * (1 / n1 + 1 / n2)))
    return erfc(abs((k1 / n1 - k2 / n2) / se) / sqrt(2))


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, out, prev = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        v = min(1.0, max(prev, (m - rank) * pvals[i]))
        out[i], prev = v, v
    return out


def run(model, rows, workers):
    cf = os.path.join(CACHE, "%s.json" % model.replace("/", "_"))
    done = json.load(open(cf, encoding="utf-8")) if os.path.exists(cf) else {}
    todo = [r for r in rows if r["id"] not in done or not done[r["id"]][1]]
    print("  [%s] cached %d, to fetch %d" % (model, len(done), len(todo)), flush=True)

    def one(r):
        txt, err = ask(model, build(r))
        lab, reason = parse(txt)
        return r["id"], [lab or "", reason, err or ""]

    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for n, (rid, val) in enumerate(ex.map(one, todo), 1):
                done[rid] = val
                if n % 50 == 0:
                    json.dump(done, open(cf, "w", encoding="utf-8"))
                    print("    %d/%d" % (n, len(todo)), flush=True)
        json.dump(done, open(cf, "w", encoding="utf-8"))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["claude-opus-5", "claude-sonnet-5",
                             "gpt-5.6-sol", "gemini-3.7-flash"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: N rows per group")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    te = load("vpesg4k_test_2000.csv")
    rows = {r["id"]: r for r in te
            if na(r["evidence_status"]) == "Yes" and na(r["evidence_quality"]) in EQ}

    old = {}
    for f in sorted(os.listdir(S26)):
        if f.endswith(".json"):
            got = json.load(open(os.path.join(S26, f), encoding="utf-8"))
            old[f.split("__")[0]] = {k: (v[0] or "Clear") for k, v in got.items()}
    base = sorted(old)
    ids = [i for i in rows if all(i in old[m] for m in base)]
    unan = [i for i in ids if len({old[m][i] for m in base}) == 1]
    wrong = [i for i in unan if old[base[0]][i] != na(rows[i]["evidence_quality"])]
    right = [i for i in unan if old[base[0]][i] == na(rows[i]["evidence_quality"])]
    ctrl = [right[k] for k in rng.choice(len(right), size=min(N_CONTROL, len(right)),
                                         replace=False)]
    if args.limit:
        wrong, ctrl = wrong[:args.limit], ctrl[:args.limit]
    probe = wrong + ctrl
    print("disputed %d, control %d of %d, total %d rows x %d models = %d calls"
          % (len(wrong), len(ctrl), len(right), len(probe), len(args.models),
             len(probe) * len(args.models)), flush=True)

    got = {}
    for m in args.models:
        cf = os.path.join(CACHE, "%s.json" % m.replace("/", "_"))
        if args.report_only:
            got[m] = json.load(open(cf, encoding="utf-8")) if os.path.exists(cf) else {}
        else:
            got[m] = run(m, [rows[i] for i in probe], args.workers)

    if _usage:
        print("\nusage:", flush=True)
        for m, u in sorted(_usage.items()):
            print("  %-20s calls %5d  in %8d  out %7d"
                  % (m, u["calls"], u["in"], u["out"]), flush=True)

    # --- report --------------------------------------------------------------
    L = []
    A = L.append
    A("# What reason do the models give?")
    A("")
    A("Generated by `research/scripts/35_explanation_probe.py`. Each of %d "
      "frontier models judged the same paragraphs again, this time returning a "
      "one-sentence reason with the label. Two groups: the **%d** rows where "
      "all four had agreed with each other and differed from the gold label, "
      "and a random **%d**-row control drawn from the %d rows where they agreed "
      "and matched it (seed %d)."
      % (len(args.models), len(wrong), len(ctrl), len(right), SEED))
    A("")
    A("Categories are assigned by regular expression over the reason text. The "
      "first two are the criteria `data/DATA.md` actually states; the third is "
      "the one it never mentions and that script 34 found to explain the "
      "models' consensus best. The rest are listed so the first three are not "
      "credited with everything.")
    A("")

    A("## Did the label hold up on re-query?")
    A("")
    A("These rows were unanimous the first time, so any change is instability "
      "rather than disagreement between models.")
    A("")
    A("| model | reasons returned | label unchanged vs script 26 | errors |")
    A("|---|---:|---:|---:|")
    for m in args.models:
        d = got[m]
        ok = [i for i in probe if i in d and d[i][1]]
        same = sum(1 for i in ok if d[i][0] == old[m].get(i))
        errs = sum(1 for i in probe if i in d and d[i][2])
        A("| %s | %d / %d | %.1f%% | %d |"
          % (m, len(ok), len(probe), 100.0 * same / max(1, len(ok)), errs))
    A("")

    A("## Which criterion does each model cite?")
    A("")
    A("Rate at which a category appears in the reason, disputed rows versus "
      "control. Holm-Bonferroni within each model over the %d categories."
      % len(CATS))
    A("")
    pooled = {c: [[], []] for c, _ in CATS}
    for m in args.models:
        d = got[m]
        gw = [i for i in wrong if i in d and d[i][1]]
        gc = [i for i in ctrl if i in d and d[i][1]]
        if not gw or not gc:
            continue
        raw, cells = [], []
        ca = {i: code(d[i][1]) for i in gc}
        cb = {i: code(d[i][1]) for i in gw}
        for cname, _ in CATS:
            a = [cname in ca[i] for i in gc]
            b = [cname in cb[i] for i in gw]
            pooled[cname][0].extend(a)
            pooled[cname][1].extend(b)
            raw.append(two_prop_p(sum(a), len(a), sum(b), len(b)))
            cells.append((cname, float(np.mean(a)), float(np.mean(b))))
        adj = holm(raw)
        A("### %s" % m)
        A("")
        A("| criterion cited | control (right) | disputed (wrong) | difference | p (Holm) |")
        A("|---|---:|---:|---:|---:|")
        for (cname, ma, mb), q in zip(cells, adj):
            A("| %s | %.3f | %.3f | %+.3f | %s |"
              % (cname, ma, mb, mb - ma,
                 ("**%.2g**" % q) if q < 0.05 else "%.2g" % q))
        A("")

    A("## Pooled over all models")
    A("")
    raw, cells = [], []
    for cname, _ in CATS:
        a, b = pooled[cname]
        raw.append(two_prop_p(sum(a), len(a), sum(b), len(b)))
        cells.append((cname, float(np.mean(a)) if a else 0.0,
                      float(np.mean(b)) if b else 0.0))
    adj = holm(raw)
    A("| criterion cited | control (right) | disputed (wrong) | difference | p (Holm) |")
    A("|---|---:|---:|---:|---:|")
    for (cname, ma, mb), q in zip(cells, adj):
        A("| %s | %.3f | %.3f | %+.3f | %s |"
          % (cname, ma, mb, mb - ma,
             ("**%.2g**" % q) if q < 0.05 else "%.2g" % q))
    A("")

    A("## Are the two leading categories the same sentence?")
    A("")
    A("`vague wording` is a criterion the organisers state; `no concrete "
      "figures` is one they never mention. They fire at almost the same rate, "
      "which would be a coincidence unless the models are saying both in one "
      "breath. This measures that directly.")
    A("")
    A("| group | cites vague wording | of those, also cites missing figures |")
    A("|---|---:|---:|")
    for gname, ids in (("control (right)", ctrl), ("disputed (wrong)", wrong)):
        n = v = 0
        for m in args.models:
            d = got[m]
            for i in ids:
                if i in d and d[i][1] and "stated: vague wording" in code(d[i][1]):
                    n += 1
                    v += int("unstated: no concrete figures" in code(d[i][1]))
        A("| %s | %d | %.3f |" % (gname, n, v / max(1, n)))
    A("")

    A("## Is the coder trustworthy?")
    A("")
    allr = [d[i][1] for m in args.models for d in [got[m]]
            for i in probe if i in d and d[i][1]]
    nocat = sum(1 for r in allr if not code(r))
    A("Of %d coded reasons, %d (%.1f%%) matched no category at all. The "
      "categories are not mutually exclusive, and a reason can invoke several."
      % (len(allr), nocat, 100.0 * nocat / max(1, len(allr))))
    A("")
    A("A sample, so the coding can be checked by eye:")
    A("")
    for r in allr[::max(1, len(allr) // 6)][:6]:
        A("- `%s` -- %s" % (", ".join(sorted(code(r))) or "(none)", r[:150]))
    A("")

    A("## Sample reasons on the disputed rows")
    A("")
    for m in args.models:
        d = got[m]
        ex = [i for i in wrong if i in d and d[i][1]][:3]
        for i in ex:
            A("- **%s** on row `%s` (gold %s, said %s): %s"
              % (m, i, na(rows[i]["evidence_quality"]), d[i][0] or "?",
                 d[i][1][:180]))
    A("")

    dest = os.path.join(OUT, "explanation_probe.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
