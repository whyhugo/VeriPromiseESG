"""Can Misleading be detected at all, without training examples?

Misleading has 2 training and 3 test instances, so nothing can be learned from
examples. Two zero-shot routes exist and one is already closed:

  * relatedness (script 09) -- refuted. The five instances sit at cosine
    percentiles 54.9, 10.8, 72.8, 8.5 and 90.7; one is among the most related
    rows in the corpus.
  * definition-following -- an instruction-tuned LLM given the guideline's own
    definition. That is what this script tests.

The definition used is the guideline's, verbatim in substance: "The evidence,
whether true or not, has no clear connection with the claim and appears to
divert attention."

Why bother when the ceiling is only 3 test rows? Because the metric critique
depends on it. Claiming that 8.75 points of the official score are decided by an
undetectable class is much stronger if the best available zero-shot method has
been tried and shown to fail. A negative result here is the point.

Reports precision/recall on the five known instances, the false-positive rate,
and what the official score would do if these predictions were merged in.
"""
import argparse
import csv
import io
import json
import os
import sys

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

CLASSES3 = ["Clear", "Not Clear", "Misleading"]

SYS = "你是企業永續報告書的稽核專家，依照給定準則判斷「證據品質」。只輸出一個標籤。"

TMPL = """判斷下列「承諾」與其「證據」之間的關係，屬於哪一個類別。

類別定義：
- Clear：證據具體且充分，足以支持該承諾；內容可理解且合乎邏輯。
- Not Clear：資訊缺漏，以致所述內容只能部分理解，或流於表面。
- Misleading：該證據無論真假，與所主張的承諾之間沒有清楚的關聯，看起來像是在轉移注意力。

承諾：
{promise}

證據：
{evidence}

只輸出 Clear、Not Clear 或 Misleading 其中一個，不要輸出其他文字。
標籤："""


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def ok(v):
    return (v or "").strip() not in ("", "N/A")


def rows(rs, split):
    out = []
    for r in rs:
        q = (r["evidence_quality"] or "").strip()
        if r["evidence_status"] != "Yes" or q not in CLASSES3:
            continue
        if not (ok(r["evidence_string"]) and ok(r["promise_string"])):
            continue
        out.append({"id": r["id"], "split": split, "y": q,
                    "pr": r["promise_string"].strip(),
                    "ev": r["evidence_string"].strip()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--maxnew", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = (rows(load("vpesg4k_train_1000"), "train")
            + rows(load("vpesg4k_val_1000"), "val")
            + rows(load("vpesg4k_test_2000.csv"), "test"))
    if args.limit:
        mis = [r for r in data if r["y"] == "Misleading"]
        rest = [r for r in data if r["y"] != "Misleading"][:args.limit]
        data = mis + rest
    print("scoring %d rows with %s" % (len(data), args.model), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    preds = []
    for i in range(0, len(data), args.bs):
        chunk = data[i:i + args.bs]
        texts = []
        for r in chunk:
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": TMPL.format(promise=r["pr"][:600],
                                                            evidence=r["ev"][:900])}]
            texts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False))
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to("cuda")
        with torch.no_grad():
            gen = mdl.generate(**enc, max_new_tokens=args.maxnew, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        for j, g in enumerate(gen):
            txt = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            t = txt.strip()
            lab = "Clear"
            if "Misleading" in t or "誤導" in t:
                lab = "Misleading"
            elif "Not Clear" in t or "不清楚" in t or "NotClear" in t:
                lab = "Not Clear"
            preds.append(lab)
        if i % (args.bs * 20) == 0:
            print("  %d/%d" % (i + len(chunk), len(data)), flush=True)

    y = [r["y"] for r in data]
    json.dump({"ids": [r["id"] for r in data], "split": [r["split"] for r in data],
               "y": y, "pred": preds},
              open(os.path.join(CACHE, "misleading_zeroshot.json"), "w"),
              ensure_ascii=False)

    from sklearn.metrics import f1_score, classification_report
    L = []
    A = L.append
    A("# Zero-shot Misleading detection with the guideline definition")
    A("")
    A("Generated by `research/scripts/11_misleading_zeroshot.py`, model `%s`, "
      "greedy decoding, no examples in the prompt." % args.model)
    A("")
    A("n = %d rows with `evidence_status=Yes` and both gold spans." % len(data))
    A("")
    A("| class | gold n | predicted n | precision | recall | F1 |")
    A("|---|---:|---:|---:|---:|---:|")
    rep = classification_report(y, preds, labels=CLASSES3, zero_division=0,
                                output_dict=True)
    from collections import Counter
    pc = Counter(preds)
    for c in CLASSES3:
        A("| `%s` | %d | %d | %.4f | %.4f | %.4f |"
          % (c, int(rep[c]["support"]), pc.get(c, 0), rep[c]["precision"],
             rep[c]["recall"], rep[c]["f1-score"]))
    A("")
    A("3-class macro-F1: **%.4f**." % f1_score(y, preds, labels=CLASSES3,
                                               average="macro", zero_division=0))
    A("")
    A("## The five Misleading instances")
    A("")
    A("| id | split | predicted |")
    A("|---|---|---|")
    for r, p in zip(data, preds):
        if r["y"] == "Misleading":
            A("| %s | %s | `%s` |" % (r["id"], r["split"], p))
    A("")
    tp = sum(1 for r, p in zip(data, preds) if r["y"] == "Misleading" and p == "Misleading")
    fp = sum(1 for r, p in zip(data, preds) if r["y"] != "Misleading" and p == "Misleading")
    A("Caught %d of %d, at the cost of %d false positives (%.1f%% of the "
      "non-Misleading rows)."
      % (tp, sum(1 for v in y if v == "Misleading"), fp,
         100.0 * fp / max(1, len(y) - sum(1 for v in y if v == "Misleading"))))
    A("")
    A("With 3 Misleading rows in the official test set, F1 for the class is "
      "2*tp / (3 + predicted_count). Any usable detector has to be almost exact: "
      "flagging 30 rows to catch all 3 still only gives F1 0.18.")
    A("")

    dest = os.path.join(OUT, "misleading_zeroshot.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
