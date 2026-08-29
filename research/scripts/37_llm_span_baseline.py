"""Can a language model just be asked to copy out the promise sentence?

Section 5 of the side report compared the clause tagger against three hand-built
heuristics and reported a clear gain, but it is missing the control a reader
will ask for first: a language model, given the paragraph and the task in words,
told to quote the span back. If a prompt does as well as a trained tagger, the
tagger is not worth its cost.

The comparison has to be fair in both directions. The model is asked for
verbatim fragments and is scored the same way the tagger is -- character
precision, recall, F1, intersection over union, and exact match against the
annotated span, on the same test rows. Fragments the model returns that are not
actually substrings of the paragraph are dropped and counted separately, since
a system that invents text is failing at extraction whatever its overlap score.

The model is Qwen3-8B, the same one used for the zero-shot `evidence_quality`
attempt in the main report, so the two zero-shot results sit on one scale.
Thinking mode is disabled: the task is extraction, and a reasoning trace would
make the comparison one of inference budget rather than of formulation.
"""
import argparse
import csv
import importlib.util
import io
import json
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts", "s37")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

SEED = 20260829
DEV = "cuda" if torch.cuda.is_available() else "cpu"

_keep_ours = sys.stdout
_spec = importlib.util.spec_from_file_location(
    "s32", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "32_clause_tagger.py"))
s32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s32)
_keep_theirs = sys.stdout

# Definitions quoted verbatim from data/DATA.md, the only official description
# of these two fields, so the prompt is not a paraphrase of our own making.
WHAT = {
    "promise_string": (
        "承諾",
        "判斷文本是否表達企業對未來行動的具體承諾，並摘錄該承諾對應之語句。"
        "「將」、「計畫」、「目標」、「承諾」、「建立」等表述通常表示承諾。"
        "承諾不必是未來式；若某行動正在執行且屬於已表明的承諾，同樣算數。"),
    "evidence_string": (
        "證據",
        "判斷已識別的承諾語句是否附有行動規劃、現有紀錄支持佐證"
        "（例如：數據、方法、措施，或任何具體執行資訊），"
        "並摘取該證據對應之語句。"),
}


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def shot_pool(field, n, seed=SEED):
    """Worked examples, drawn from the training split only."""
    if n <= 0:
        return []
    tr = [r for r in load("vpesg4k_train_1000")
          if s32.ok(r.get(field)) and 60 <= len(r["data"] or "") <= 260]
    idx = np.random.default_rng(seed).permutation(len(tr))[:n]
    return [tr[i] for i in idx]


def build(para, field, shots=()):
    name, desc = WHAT[field]
    ex = ""
    for r in shots:
        got = json.dumps({"spans": s32.pieces_of(r.get(field))},
                         ensure_ascii=False)
        ex += "段落：\n%s\n答案：%s\n\n" % (r["data"] or "", got)
    return ("以下是一家台灣上市公司永續報告書中的一個段落。\n\n"
            "任務：找出段落中屬於「%s」的文字。%s\n\n"
            "規則：\n"
            "1. 只能逐字複製段落中原有的文字，一個字都不能改寫、"
            "增加或省略。\n"
            "2. 如果分散在不同位置，可以回傳多個片段。\n"
            "3. 如果段落中沒有，回傳空陣列。\n\n"
            "%s"
            "段落：\n%s\n"
            '答案（只輸出這個 JSON，不要其他文字）：'
            % (name, desc, ex, para[:1500]))


def parse(txt, para):
    """Fragments the model returned that really occur in the paragraph."""
    if not txt:
        return [], 0
    m = re.search(r"\{.*\}", txt, re.S)
    got = []
    if m:
        try:
            got = json.loads(m.group(0)).get("spans") or []
        except Exception:
            got = []
    if not got:
        got = re.findall(r'"([^"]{4,})"', txt)
    keep, bad = [], 0
    for g in got:
        g = str(g).strip()
        if len(g) >= 2 and g in para:
            keep.append(g)
        elif g:
            bad += 1
    return keep, bad


def mask_of(para, frags):
    m = np.zeros(len(para), dtype=bool)
    for f in frags:
        i = para.find(f)
        if i >= 0:
            m[i:i + len(f)] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--fields", nargs="*",
                    default=["promise_string", "evidence_string"])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--shots", nargs="*", type=int, default=[0, 4],
                    help="numbers of worked examples to compare")
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    tok = mdl = None
    if not args.report_only:
        tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=DEV).eval()

    L = []
    A = L.append
    A("# A language model asked to quote the span back")
    A("")
    A("Generated by `research/scripts/37_llm_span_baseline.py` with `%s`, "
      "greedy decoding, thinking mode off. The task description in the prompt "
      "is quoted verbatim from `data/DATA.md`, and worked examples, where used, "
      "are drawn from the training split only. Scored exactly as the clause "
      "tagger is scored in script 32, on the same test rows." % args.model)
    A("")

    for field in args.fields:
        te = [r for r in load("vpesg4k_test_2000.csv") if s32.ok(r.get(field))]
        if args.limit:
            te = te[:args.limit]
        A("## `%s`" % field)
        A("")
        A("| system | char precision | char recall | char F1 | IoU | exact match |")
        A("|---|---:|---:|---:|---:|---:|")
        notes = []
        for k in args.shots:
            shots = shot_pool(field, k)
            cf = os.path.join(CACHE, "%s__%s__%dshot.json"
                              % (args.model.replace("/", "_"), field, k))
            done = (json.load(open(cf, encoding="utf-8"))
                    if os.path.exists(cf) else {})
            todo = [r for r in te if r["id"] not in done]
            print("[%s %d-shot] %d rows, %d cached, %d to generate"
                  % (field, k, len(te), len(te) - len(todo), len(todo)), flush=True)
            for i in range(0, len(todo), args.bs):
                chunk = todo[i:i + args.bs]
                msgs = [tok.apply_chat_template(
                    [{"role": "user",
                      "content": build(r["data"] or "", field, shots)}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False) for r in chunk]
                enc = tok(msgs, return_tensors="pt", padding=True,
                          truncation=True, max_length=4096).to(DEV)
                with torch.no_grad():
                    gen = mdl.generate(
                        **enc, max_new_tokens=args.max_new, do_sample=False,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
                for n, r in enumerate(chunk):
                    done[r["id"]] = tok.decode(
                        gen[n][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                if (i // args.bs) % 10 == 0:
                    json.dump(done, open(cf, "w", encoding="utf-8"))
                    print("  %d/%d" % (i + len(chunk), len(todo)), flush=True)
            if todo:
                json.dump(done, open(cf, "w", encoding="utf-8"))

            P, R, F, I, EM, bad, empty = [], [], [], [], 0, 0, 0
            for r in te:
                para = r["data"] or ""
                frags, nb = parse(done.get(r["id"], ""), para)
                bad += nb
                empty += int(not frags)
                pred = mask_of(para, frags)
                gold = s32.gold_mask(para, r.get(field))
                p, rc, f, iou = s32.prf(pred, gold)
                P.append(p); R.append(rc); F.append(f); I.append(iou)
                EM += int(np.array_equal(pred, gold))
            A("| %s, %s | %.4f | %.4f | **%.4f** | %.4f | %.1f%% |"
              % (args.model.split("/")[-1],
                 "zero-shot" if k == 0 else "%d-shot" % k,
                 np.mean(P), np.mean(R), np.mean(F), np.mean(I),
                 100.0 * EM / len(te)))
            notes.append("%s: %d invented fragments dropped, %d rows (%.1f%%) "
                         "with no usable fragment"
                         % ("zero-shot" if k == 0 else "%d-shot" % k, bad,
                            empty, 100.0 * empty / len(te)))
        A("")
        A("Over %d test rows -- %s." % (len(te), "; ".join(notes)))
        A("")

    if mdl is not None:
        del mdl
        torch.cuda.empty_cache()

    A("## Reading")
    A("")
    A("Compare these numbers against the clause tagger and the three heuristics "
      "in `clause_tagger_promise_string.md` and `clause_tagger_evidence_string.md`. "
      "The tagger is only worth its training cost if it beats a prompt.")
    A("")
    dest = os.path.join(OUT, "llm_span_baseline.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
