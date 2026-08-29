"""Does retrieval give the model information it does not already have?

Eight interventions have failed on evidence_quality, and they share a shape:
more training, more data for the rare class, a different threshold, a different
architecture, a bigger model, a vote. Every one of them changes how the model
learns or how its outputs are combined. None changes what it can see. The
training curve says that is the binding constraint -- evidence_quality reaches
0.501 on the third epoch and 0.505 on the fifteenth, while the other subtasks
keep climbing.

Retrieval is the first intervention that adds information. This script runs the
two stages that need no external corpus:

  A. retrieval alone. Embed every paragraph, and for each one predict from the
     labels of its nearest neighbours among the others, leave-one-out. No
     training, no LLM. This tests the premise the whole idea rests on: that
     paragraphs which look alike carry the same label. If the neighbours of a
     Not Clear paragraph are no more Not Clear than chance, retrieval has
     nothing to offer here and stage B is pointless.

  B. retrieved examples in the prompt. The same neighbours, with their gold
     labels, placed in front of the query for a language model scored the same
     constrained way as script 21. Script 11 showed that giving the model the
     guideline's *definition* of Misleading finds none of five. This asks
     whether giving it *examples* does any better -- a different question, and
     the one k-nearest-neighbour prompting actually tests.

Retrieval is always leave-one-out over train+val, which is what a deployed
system would face on unseen text. Test is untouched.
"""
import argparse
import csv
import io
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts", "s23")
S18 = os.path.join(ROOT, "research", "artifacts", "s18")
S21 = os.path.join(ROOT, "research", "artifacts", "s21")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
SEED = 20260824

LABELS = {
    "promise_status": ["No", "Yes"],
    "evidence_status": ["N/A", "No", "Yes"],
    "evidence_quality": ["N/A", "Not Clear", "Clear", "Misleading"],
    "verification_timeline": ["N/A", "already", "within_2_years",
                              "between_2_and_5_years", "more_than_5_years"],
}
ZH = {
    "promise_status": {"No": "沒有承諾", "Yes": "有承諾"},
    "evidence_status": {"N/A": "不適用", "No": "無佐證", "Yes": "有佐證"},
    "evidence_quality": {"N/A": "不適用", "Not Clear": "不清楚",
                         "Clear": "清楚", "Misleading": "誤導"},
    "verification_timeline": {"N/A": "不適用", "already": "已可驗證",
                              "within_2_years": "兩年內",
                              "between_2_and_5_years": "兩到五年",
                              "more_than_5_years": "五年以上"},
}
QUESTION = {
    "promise_status": "這個段落有沒有提出一項 ESG 承諾、目標或未來計畫？",
    "evidence_status": "這個段落所提出的承諾，有沒有附上具體的佐證或實績？",
    "evidence_quality": "這個段落所附的佐證，品質如何？",
    "verification_timeline": "這個段落所提出的承諾，何時可以被驗證？",
}
LETTERS = "ABCDE"
EMBED_MODEL = "BAAI/bge-m3"
LLM = "Qwen/Qwen3-8B"


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


def embed(texts, bs=16):
    cf = os.path.join(CACHE, "emb.npy")
    if os.path.exists(cf):
        e = np.load(cf)
        if len(e) == len(texts):
            print("[cached] embeddings", e.shape, flush=True)
            return e
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBED_MODEL, device="cuda")
    e = m.encode(texts, batch_size=bs, normalize_embeddings=True,
                 show_progress_bar=False, convert_to_numpy=True)
    del m
    torch.cuda.empty_cache()
    np.save(cf, e)
    print("embedded", e.shape, flush=True)
    return e


# --------------------------------------------------------------- stage A
def knn_predict(sim, y, k, nlab):
    """Leave-one-out similarity-weighted vote. `sim` has its diagonal masked."""
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    out = np.zeros((len(y), nlab))
    for i in range(len(y)):
        nb = idx[i]
        w = np.clip(sim[i, nb], 0.0, None)
        np.add.at(out[i], y[nb], w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 3, 5, 10, 20])
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--stage", default="AB")
    args = ap.parse_args()

    trv = load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
    texts = [r["data"] or "" for r in trv]
    emb = embed(texts)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)   # leave-one-out

    L = []
    A = L.append
    A("# Retrieval augmentation: stages A and B")
    A("")
    A("Generated by `research/scripts/23_retrieval_augmentation.py`. Embeddings "
      "from `%s` over all %d train+val paragraphs; retrieval is leave-one-out, "
      "so no paragraph can retrieve itself. Test is untouched."
      % (EMBED_MODEL, len(trv)))
    A("")

    # ---- the premise: do neighbours share the label? ----
    A("## Does retrieval find label-consistent neighbours?")
    A("")
    A("For each paragraph, the share of its `k` nearest neighbours carrying the "
      "same label, against the share expected if neighbours were drawn at "
      "random. A ratio near 1.0 means retrieval carries no label information "
      "and nothing downstream can help.")
    A("")
    A("| subtask | base rate | k=1 | k=5 | k=10 | ratio at k=10 |")
    A("|---|---:|---:|---:|---:|---:|")
    order = np.argsort(-sim, axis=1)
    for f in LABELS:
        y = np.array([LABELS[f].index(na(r[f])) for r in trv])
        cnt = Counter(y.tolist())
        base = sum((c / len(y)) ** 2 for c in cnt.values())
        cells = []
        for k in (1, 5, 10):
            nb = order[:, :k]
            cells.append(float((y[nb] == y[:, None]).mean()))
        A("| `%s` | %.4f | %.4f | %.4f | %.4f | **%.2fx** |"
          % (f, base, cells[0], cells[1], cells[2], cells[2] / base))
    A("")

    # ---- stage A: retrieval-only classifier ----
    A("## Stage A: retrieval alone as a classifier")
    A("")
    A("Similarity-weighted vote over the k nearest neighbours, no training. "
      "The encoder row is the supervised out-of-fold baseline on the same rows.")
    A("")
    A("| subtask | encoder (OOF) | %s |" % " | ".join("k=%d" % k for k in args.ks))
    A("|---|---:|%s" % ("---:|" * len(args.ks)))
    knn_prob = {}
    for f in LABELS:
        y = np.array([LABELS[f].index(na(r[f])) for r in trv])
        eo = np.load(os.path.join(S18, "%s.npz" % f.replace("_", "")))["oof"]
        row = ["%.4f" % f1_score(y, eo.argmax(1), average="macro", zero_division=0)]
        best, bestk = None, None
        for k in args.ks:
            p = knn_predict(sim, y, k, len(LABELS[f]))
            s = f1_score(y, p.argmax(1), average="macro", zero_division=0)
            row.append("%.4f" % s)
            if best is None or s > best:
                best, bestk, knn_prob[f] = s, k, p / np.maximum(p.sum(1, keepdims=True), 1e-9)
        A("| `%s` | %s |" % (f, " | ".join(row)))
    A("")

    A("### Complementarity of the retrieval classifier with the encoder")
    A("")
    A("| subtask | agreement | both right | retrieval only right | encoder only right |")
    A("|---|---:|---:|---:|---:|")
    for f in LABELS:
        y = np.array([LABELS[f].index(na(r[f])) for r in trv])
        eo = np.load(os.path.join(S18, "%s.npz" % f.replace("_", "")))["oof"].argmax(1)
        kp = knn_prob[f].argmax(1)
        A("| `%s` | %.3f | %.3f | **%.3f** | %.3f |"
          % (f, float((kp == eo).mean()), float(((kp == y) & (eo == y)).mean()),
             float(((kp == y) & (eo != y)).mean()),
             float(((kp != y) & (eo == y)).mean())))
    A("")
    np.savez(os.path.join(CACHE, "knn_probs.npz"), **knn_prob)

    if "B" not in args.stage:
        dest = os.path.join(OUT, "retrieval_augmentation.md")
        open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
        print("wrote", dest, flush=True)
        print("\n".join(L))
        return

    # ---- stage B: retrieved examples in the prompt ----
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    pick = json.load(open(os.path.join(S21, "subsample.json")))["picked_rows"]
    tok = AutoTokenizer.from_pretrained(LLM)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    mdl = AutoModelForCausalLM.from_pretrained(
        LLM, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
        device_map="cuda").eval()

    A("## Stage B: retrieved examples in the prompt")
    A("")
    A("`%s`, %d retrieved neighbours shown with their gold labels before the "
      "query, scored the same constrained way as script 21 so the zero-shot "
      "column is directly comparable. Same %d-row subsample."
      % (LLM, args.shots, len(pick)))
    A("")
    A("| subtask | zero-shot (script 21) | %d-shot retrieved | delta |"
      % args.shots)
    A("|---|---:|---:|---:|")
    zs = np.load(os.path.join(S21, "Qwen3-8B.npz"))
    for f in LABELS:
        labels = LABELS[f]
        lids = []
        for i in range(len(labels)):
            cand = set()
            for form in (LETTERS[i], " " + LETTERS[i]):
                e = tok.encode(form, add_special_tokens=False)
                if e:
                    cand.add(e[0])
            lids.append(sorted(cand))
        opts = "\n".join("%s. %s" % (LETTERS[i], ZH[f][l])
                         for i, l in enumerate(labels))
        prompts = []
        for qi in pick:
            nb = order[qi, :args.shots]
            ex = "\n\n".join(
                "段落：%s\n答案：%s" % ((trv[j]["data"] or "")[:280],
                                    ZH[f][na(trv[j][f])])
                for j in reversed(nb) if na(trv[j][f]) in labels)
            user = ("以下是幾個已標註的例子：\n\n%s\n\n---\n\n"
                    "現在判斷這個段落。\n段落：%s\n\n問題：%s\n\n%s\n\n只回答選項字母。"
                    % (ex, (trv[qi]["data"] or "")[:900], QUESTION[f], opts))
            msgs = [{"role": "user", "content": user}]
            try:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
            prompts.append(t + "答案：")
        acc = []
        for i in range(0, len(prompts), args.bs):
            enc = tok(prompts[i:i + args.bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=4096).to("cuda")
            with torch.no_grad():
                lg = mdl(**enc).logits[:, -1, :].float()
            acc.append(torch.softmax(
                torch.stack([lg[:, ids].max(-1).values for ids in lids], -1),
                -1).cpu().numpy())
        p = np.concatenate(acc)
        np.savez(os.path.join(CACHE, "fewshot_%s.npz" % f.replace("_", "")), p=p)
        yq = np.array([labels.index(na(trv[i][f])) for i in pick])
        few = f1_score(yq, p.argmax(1), average="macro", zero_division=0)
        zero = f1_score(yq, zs[f].argmax(1), average="macro", zero_division=0)
        A("| `%s` | %.4f | **%.4f** | %+.4f |" % (f, zero, few, few - zero))
        print("[B] %s zero %.4f few %.4f" % (f, zero, few), flush=True)
    A("")
    A("### Does retrieval find the Misleading instances?")
    A("")
    q = np.load(os.path.join(CACHE, "fewshot_evidencequality.npz"))["p"]
    labs = LABELS["evidence_quality"]
    yq = np.array([labs.index(na(trv[i]["evidence_quality"])) for i in pick])
    mi = labs.index("Misleading")
    A("| | count |")
    A("|---|---:|")
    A("| gold Misleading in the subsample | %d |" % int((yq == mi).sum()))
    A("| predicted Misleading | %d |" % int((q.argmax(1) == mi).sum()))
    A("| correctly predicted | %d |" % int(((q.argmax(1) == mi) & (yq == mi)).sum()))
    A("")
    A("Script 11 gave the model the guideline's definition and it found none of "
      "five. This gives it retrieved examples instead.")
    A("")

    dest = os.path.join(OUT, "retrieval_augmentation.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
