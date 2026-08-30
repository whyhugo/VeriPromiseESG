"""Is Misleading undetectable, or just beyond the models we could run locally?

Everything tried so far says the class cannot be found. A fine-tuned three-
backbone ensemble predicts it zero times in 2,000 rows. Semantic relatedness
between the promise and its evidence scores AUC 0.4671, below chance. Qwen3-8B
given the guideline's own definition finds none of five. Three hundred synthetic
examples produce zero recall. Retrieved neighbours as prompt examples produce
zero predictions.

Every one of those used a model of at most 32B, quantised, run on one consumer
GPU. So the honest state of the claim is "nothing *we* could run finds it",
which is weaker than it needs to be. If a frontier model also fails, the claim
becomes a property of the task rather than a limitation of our compute -- and
that is what makes eight and a quarter points of a shared evaluation resting on
three rows a finding rather than a complaint.

Two runs:

  1. evidence_quality over all 1,335 test rows with gold evidence_status=Yes.
     This contains all three test-set Misleading instances, so precision and
     recall are both measurable rather than just "did it ever say the word".
  2. all four subtasks over the same 601-row train+val subsample the seven local
     models were scored on, for a like-for-like comparison.

Keys come from the environment and are never written to disk. Every response is
cached by (model, row, subtask) so an interrupted run resumes without paying
twice, and token usage is totalled so the cost is on the record.
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

from sklearn.metrics import classification_report, f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts", "s26")
S21 = os.path.join(ROOT, "research", "artifacts", "s21")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

EQ = ["Clear", "Not Clear", "Misleading"]
LABELS = {
    "promise_status": ["No", "Yes"],
    "evidence_status": ["N/A", "No", "Yes"],
    "evidence_quality": ["N/A", "Not Clear", "Clear", "Misleading"],
    "verification_timeline": ["N/A", "already", "within_2_years",
                              "between_2_and_5_years", "more_than_5_years"],
}
QUESTION = {
    "promise_status": "這個段落有沒有提出一項 ESG（環境、社會或公司治理）方面的承諾、目標或未來計畫？",
    "evidence_status": "這個段落所提出的承諾，有沒有附上具體的佐證或實績？",
    "evidence_quality": "這個段落所附的佐證，品質如何？",
    "verification_timeline": "這個段落所提出的承諾，何時可以被驗證？",
}
OPTS = {
    "promise_status": ["No：沒有提出承諾", "Yes：有提出承諾"],
    "evidence_status": ["N/A：這個段落根本沒有提出承諾",
                        "No：有提出承諾，但沒有附上任何佐證",
                        "Yes：有提出承諾，也附上了佐證"],
    "evidence_quality": [
        "N/A：沒有承諾或沒有佐證，不適用",
        "Not Clear：資訊缺漏，內容只能部分理解，或流於表面",
        "Clear：具體且充分，足以支持該承諾，內容可理解且合乎邏輯",
        "Misleading：該佐證無論真假，與所主張的承諾之間沒有清楚關聯，看起來像在轉移注意力"],
    "verification_timeline": [
        "N/A：這個段落根本沒有提出承諾",
        "already：已經可以驗證（描述已完成或正在進行的事）",
        "within_2_years：兩年內可以驗證",
        "between_2_and_5_years：兩到五年內可以驗證",
        "more_than_5_years：五年以上才能驗證"],
}
SYS = ("你是台灣上市公司永續報告書的稽核專家，依照給定的標註準則判斷。"
       "只輸出一個標籤，不要任何說明。")

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
                 {"model": model, "max_tokens": 2048,
                  "system": SYS,
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
                 {"model": model, "max_output_tokens": 2048,
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
              "generationConfig": {"maxOutputTokens": 2048}},
             {"content-type": "application/json"})
    note(model, d)
    if "_err" in d:
        return None, d["_err"]
    try:
        parts = d["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts), None
    except Exception:
        return "", None


def build(row, field):
    opts = "\n".join("- " + o for o in OPTS[field])
    return ("段落：\n%s\n\n問題：%s\n\n可選標籤：\n%s\n\n"
            "只輸出標籤名稱本身（例如 Clear），不要輸出冒號後的說明。"
            % ((row["data"] or "")[:1500], QUESTION[field], opts))


def parse(txt, field):
    if not txt:
        return None
    t = txt.strip()
    # longest label first so "Not Clear" is not swallowed by "Clear"
    for lab in sorted(LABELS[field], key=len, reverse=True):
        if re.search(re.escape(lab), t, re.I):
            return lab
    return None


def run(model, rows, field, workers):
    cf = os.path.join(CACHE, "%s__%s.json" % (model.replace("/", "_"), field))
    done = json.load(open(cf, encoding="utf-8")) if os.path.exists(cf) else {}
    todo = [r for r in rows if r["id"] not in done]
    print("  [%s|%s] cached %d, to fetch %d" % (model, field, len(done), len(todo)),
          flush=True)

    def one(r):
        txt, err = ask(model, build(r, field))
        return r["id"], (parse(txt, field) or "", err or "")

    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for n, (rid, val) in enumerate(ex.map(one, todo), 1):
                done[rid] = val
                if n % 100 == 0:
                    with _lock:
                        json.dump(done, open(cf, "w", encoding="utf-8"))
                    print("    %d/%d" % (n, len(todo)), flush=True)
        json.dump(done, open(cf, "w", encoding="utf-8"))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["claude-sonnet-5", "gpt-5.6-sol", "gemini-3.7-flash"])
    ap.add_argument("--run", default="1", choices=["1", "2", "12"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    te = load("vpesg4k_test_2000.csv")
    eq_rows = [r for r in te if na(r["evidence_status"]) == "Yes"
               and na(r["evidence_quality"]) in EQ]
    if args.limit:
        mis = [r for r in eq_rows if na(r["evidence_quality"]) == "Misleading"]
        rest = [r for r in eq_rows if na(r["evidence_quality"]) != "Misleading"]
        eq_rows = mis + rest[:args.limit]
    print("run 1 pool: %d test rows (Misleading %d)"
          % (len(eq_rows), sum(1 for r in eq_rows
                               if na(r["evidence_quality"]) == "Misleading")),
          flush=True)

    L = []
    A = L.append
    A("# Frontier API models on the undetectable class")
    A("")
    A("Generated by `research/scripts/26_frontier_api_ceiling.py`. Every local "
      "attempt so far ran a model of at most 32B, quantised, on one consumer "
      "GPU, so \"nothing finds Misleading\" has meant \"nothing we could run\". "
      "These are the current frontier models from three providers.")
    A("")

    if "1" in args.run:
        A("## `evidence_quality` on the test rows with gold evidence")
        A("")
        A("%d rows, containing all %d test-set `Misleading` instances."
          % (len(eq_rows), sum(1 for r in eq_rows
                               if na(r["evidence_quality"]) == "Misleading")))
        A("")
        A("| model | 計分列數 | macro-F1 | Clear | Not Clear | **Misleading** | 預測 Misleading 次數 | 抓到 |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|")
        for m in args.models:
            got = run(m, eq_rows, "evidence_quality", args.workers)
            # a failed call is excluded, not scored as a free `Clear`; the gold
            # vector has to be rebuilt on the same rows, per model
            usable = [r for r in eq_rows
                      if (got.get(r["id"], ["", ""])[0] or "")]
            y = [na(r["evidence_quality"]) for r in usable]
            p = [got[r["id"]][0] for r in usable]
            rep = classification_report(y, p, labels=EQ, output_dict=True,
                                        zero_division=0)
            npred = sum(1 for v in p if v == "Misleading")
            hit = sum(1 for a, b in zip(y, p)
                      if a == "Misleading" and b == "Misleading")
            A("| %s | %d | %.4f | %.4f | %.4f | **%.4f** | %d | **%d / %d** |"
              % (m, len(usable),
                 f1_score(y, p, labels=EQ, average="macro", zero_division=0),
                 rep["Clear"]["f1-score"], rep["Not Clear"]["f1-score"],
                 rep["Misleading"]["f1-score"], npred, hit,
                 sum(1 for v in y if v == "Misleading")))
            print("[%s] done" % m, flush=True)
        A("")
        A("Local reference on the same subtask: the fine-tuned three-backbone "
          "ensemble predicts `Misleading` 0 times in 2,000 rows; Qwen3-8B given "
          "the guideline definition finds 0 of 5; retrieved examples produce 0 "
          "predictions.")
        A("")

    A("## Token usage")
    A("")
    A("| model | calls | input tokens | output tokens |")
    A("|---|---:|---:|---:|")
    for m, u in _usage.items():
        A("| %s | %d | %d | %d |" % (m, u["calls"], u["in"], u["out"]))
    A("")

    dest = os.path.join(OUT, "frontier_api_ceiling.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
