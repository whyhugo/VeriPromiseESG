"""Can an LLM manufacture the training data the rare classes are missing?

Three classes carry most of the remaining error, and all three are starved:

    evidence_quality = Misleading    2 training instances, F1 0.0000
    evidence_quality = Not Clear   225 training instances, F1 0.3556
    evidence_status  = No          282 training instances, recall 0.514

Script 17 put the reachable headroom at +0.1155 for the gate and +0.1452 for the
Clear/Not Clear judgement, with +0.2503 locked in the three Misleading rows.
Augmentation is the one family of methods that attacks the starvation directly
rather than the model.

Generation is *guided* rather than free. Free generation from a definition tends
to produce prototypes that sit far from the decision boundary and teach the model
nothing; instead the LLM is given a real paragraph of the adjacent class and
asked for the minimal edit that flips the label under the guideline's own
wording. That keeps register, domain and length close to the corpus and puts the
synthetic example near the boundary where it can matter.

Misleading is expected to fail. Scripts 09 and 11 showed the class is invisible
to relatedness scoring and to the same model prompted with the defining
criterion, so there is little reason to think it can generate faithful instances
of a category it cannot recognise. That negative is worth having explicitly: it
converts "we could not detect Misleading" into "the class cannot be reached from
its definition at all, in either direction".

Every evaluation is on the real test set. Synthetic data never enters it.
"""
import argparse
import csv
import io
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts", "s19")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
SEED = 20260824
BACKBONE = "hfl/chinese-roberta-wwm-ext-large"

GUIDE = {
    "Not Clear": ("證據資訊缺漏，以致所述內容只能部分理解，或流於表面。"
                  "例如只有空泛宣示、沒有時程、沒有數字、沒有指明負責單位。"),
    "Clear": ("證據具體且充分，足以支持該承諾，內容可理解且合乎邏輯。"
              "通常包含時程、數字、百分比、涉及方的指認或預期成果的清楚描述。"),
    "Misleading": ("該證據無論真假，與所主張的承諾之間沒有清楚的關聯，"
                   "看起來像是在轉移注意力。"),
    "no_evidence": "段落提出了承諾或目標，但完全沒有提供任何佐證或實績。",
}

EDIT_PROMPT = """以下是一段台灣上市公司永續報告書的內容，目前的標註是「{src_label}」。

原文：
{text}

請將它改寫成符合「{dst_label}」的版本。「{dst_label}」的定義是：
{dst_def}

改寫要求：
1. 盡量保留原文的主題、產業用語與長度，只做達成標籤所需的最小改動。
2. 讀起來必須像真實的永續報告書段落，不要出現「假設」「範例」等字眼。
3. 只輸出改寫後的段落本文，不要任何說明或標題。

改寫後："""


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


# ----------------------------------------------------------------- generation
def generate(pairs, model_name, bs, max_new):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    out = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        texts = []
        for p in chunk:
            msg = [{"role": "user", "content": EDIT_PROMPT.format(
                src_label=p["src_label"], dst_label=p["dst_label"],
                dst_def=GUIDE[p["dst_key"]], text=p["text"][:900])}]
            texts.append(tok.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True,
                enable_thinking=False))
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to("cuda")
        with torch.no_grad():
            gen = mdl.generate(**enc, max_new_tokens=max_new, do_sample=True,
                               temperature=0.9, top_p=0.95,
                               pad_token_id=tok.pad_token_id)
        for j, g in enumerate(gen):
            txt = tok.decode(g[enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
            if len(txt) >= 30:
                out.append(dict(chunk[j], generated=txt))
        print("  generated %d/%d" % (min(i + bs, len(pairs)), len(pairs)), flush=True)
    del mdl
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------- training
class DS(Dataset):
    def __init__(self, texts, ys, tok, maxlen):
        self.t, self.y, self.tok, self.m = texts, ys, tok, maxlen

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        e = self.tok(self.t[i], truncation=True, max_length=self.m,
                     padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()}, torch.tensor(self.y[i])


def train_eval(tr_txt, tr_y, te_txt, nlab, maxlen, epochs, bs, lr):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE, num_labels=nlab).float().to("cuda")
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    dl = DataLoader(DS(tr_txt, tr_y, tok, maxlen), batch_size=bs, shuffle=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, len(dl) * epochs), pct_start=0.1)
    cnt = np.bincount(tr_y, minlength=nlab).astype(float)
    w = torch.tensor(len(tr_y) / (nlab * np.maximum(cnt, 1)),
                     dtype=torch.float32, device="cuda")
    lossf = nn.CrossEntropyLoss(weight=w)
    mdl.train()
    for _ in range(epochs):
        for enc, y in dl:
            enc = {k: v.to("cuda") for k, v in enc.items()}
            y = y.to("cuda")
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(mdl(**enc).logits.float(), y)
            loss.backward()
            opt.step()
            sched.step()
    mdl.eval()
    dle = DataLoader(DS(te_txt, [0] * len(te_txt), tok, maxlen), batch_size=bs * 2)
    ps = []
    with torch.no_grad():
        for enc, _ in dle:
            enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ps.append(torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy())
    del mdl
    torch.cuda.empty_cache()
    return np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--gen-bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--maxlen", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--skip-gen", action="store_true")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    trv = load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
    te = load("vpesg4k_test_2000.csv")

    # ------------------------------------------------------- build edit pairs
    gen_file = os.path.join(CACHE, "generated.json")
    if os.path.exists(gen_file) or args.skip_gen:
        gen = json.load(open(gen_file, encoding="utf-8"))
        print("loaded %d cached generations" % len(gen), flush=True)
    else:
        pairs = []
        clear = [r for r in trv if na(r["evidence_quality"]) == "Clear"]
        random.shuffle(clear)
        for r in clear[:args.n_per_class]:
            pairs.append({"text": r["data"] or "", "src_label": "Clear",
                          "dst_label": "Not Clear", "dst_key": "Not Clear",
                          "target": "eq_notclear"})
        for r in clear[args.n_per_class:2 * args.n_per_class]:
            pairs.append({"text": r["data"] or "", "src_label": "Clear",
                          "dst_label": "Misleading", "dst_key": "Misleading",
                          "target": "eq_misleading"})
        yes = [r for r in trv if na(r["promise_status"]) == "Yes"
               and na(r["evidence_status"]) == "Yes"]
        random.shuffle(yes)
        for r in yes[:args.n_per_class]:
            pairs.append({"text": r["data"] or "", "src_label": "有具體佐證",
                          "dst_label": "沒有任何佐證", "dst_key": "no_evidence",
                          "target": "es_no"})
        print("generating %d synthetic paragraphs with %s"
              % (len(pairs), args.gen_model), flush=True)
        gen = generate(pairs, args.gen_model, args.gen_bs, args.max_new)
        json.dump(gen, open(gen_file, "w", encoding="utf-8"), ensure_ascii=False)
        print("kept %d" % len(gen), flush=True)

    by_target = Counter(g["target"] for g in gen)
    print("generated per target:", dict(by_target), flush=True)

    L = []
    A = L.append
    A("# LLM augmentation for the starved classes")
    A("")
    A("Generated by `research/scripts/19_llm_augmentation.py`. Generator `%s`, "
      "guided minimal-edit prompting from a real paragraph of the adjacent "
      "class. Classifier `%s`. All evaluation is on the real 2,000-row test "
      "set; synthetic data is only ever added to training."
      % (args.gen_model, BACKBONE))
    A("")
    A("| target class | synthetic paragraphs kept |")
    A("|---|---:|")
    for k, v in sorted(by_target.items()):
        A("| `%s` | %d |" % (k, v))
    A("")

    # --------------------------------------------- experiment 1: evidence_quality
    EQL = ["N/A", "Not Clear", "Clear", "Misleading"]
    base_rows = [(r["data"] or "", EQL.index(na(r["evidence_quality"])))
                 for r in trv if na(r["evidence_quality"]) in EQL]
    te_txt = [r["data"] or "" for r in te]
    y_te = np.array([EQL.index(na(r["evidence_quality"])) for r in te])

    syn_nc = [(g["generated"], EQL.index("Not Clear"))
              for g in gen if g["target"] == "eq_notclear"]
    syn_mi = [(g["generated"], EQL.index("Misleading"))
              for g in gen if g["target"] == "eq_misleading"]

    conds = [("real only", base_rows),
             ("+ synthetic Not Clear", base_rows + syn_nc),
             ("+ synthetic Misleading", base_rows + syn_mi),
             ("+ both", base_rows + syn_nc + syn_mi)]
    A("## `evidence_quality`")
    A("")
    A("| training data | n | macro-F1 | Clear | Not Clear | Misleading | N/A |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for name, rows in conds:
        txt = [t for t, _ in rows]
        yy = np.array([v for _, v in rows])
        pr = train_eval(txt, yy, te_txt, len(EQL), args.maxlen, args.epochs,
                        args.bs, args.lr)
        pred = pr.argmax(1)
        rep = classification_report(y_te, pred, labels=list(range(len(EQL))),
                                    target_names=EQL, output_dict=True,
                                    zero_division=0)
        A("| %s | %d | **%.4f** | %.4f | %.4f | %.4f | %.4f |"
          % (name, len(rows),
             f1_score(y_te, pred, average="macro", zero_division=0),
             rep["Clear"]["f1-score"], rep["Not Clear"]["f1-score"],
             rep["Misleading"]["f1-score"], rep["N/A"]["f1-score"]))
        print("[eq] %s done" % name, flush=True)
    A("")

    # ---------------------------------------------- experiment 2: evidence_status
    ESL = ["N/A", "No", "Yes"]
    base_es = [(r["data"] or "", ESL.index(na(r["evidence_status"])))
               for r in trv if na(r["evidence_status"]) in ESL]
    y_es = np.array([ESL.index(na(r["evidence_status"])) for r in te])
    syn_no = [(g["generated"], ESL.index("No"))
              for g in gen if g["target"] == "es_no"]
    A("## `evidence_status`")
    A("")
    A("| training data | n | macro-F1 | Yes | No | N/A | recall on `No` |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for name, rows in [("real only", base_es),
                       ("+ synthetic No", base_es + syn_no)]:
        txt = [t for t, _ in rows]
        yy = np.array([v for _, v in rows])
        pr = train_eval(txt, yy, te_txt, len(ESL), args.maxlen, args.epochs,
                        args.bs, args.lr)
        pred = pr.argmax(1)
        rep = classification_report(y_es, pred, labels=list(range(len(ESL))),
                                    target_names=ESL, output_dict=True,
                                    zero_division=0)
        ino = ESL.index("No")
        A("| %s | %d | **%.4f** | %.4f | %.4f | %.4f | %.4f |"
          % (name, len(rows),
             f1_score(y_es, pred, average="macro", zero_division=0),
             rep["Yes"]["f1-score"], rep["No"]["f1-score"], rep["N/A"]["f1-score"],
             rep["No"]["recall"]))
        print("[es] %s done" % name, flush=True)
    A("")
    A("Reference: the competition ensemble scores `evidence_status` macro-F1 "
      "0.7132 with recall 0.514 on `No`, and `evidence_quality` macro-F1 0.4642 "
      "with `Misleading` at 0.0000.")
    A("")

    dest = os.path.join(OUT, "llm_augmentation.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
