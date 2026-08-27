"""Which local LLM, if any, is worth adding to the ensemble?

Ensembling is the only intervention that has produced a stable gain in this
project (+0.0142 for three backbones over one, script 15). But those three
backbones are all 300M-class BERT-family encoders trained on the same data with
the same objective, so they fail in correlated ways. An 8-32B generative model
fails differently, which is the property a vote actually needs.

This sweeps six candidates before committing compute to any one of them.

Scoring is *constrained*, not generative. Each subtask is posed as a lettered
multiple choice, the prompt ends on an answer prefix, and the distribution is
read off the logits at that single position restricted to the option letters.
One forward pass per (row, subtask), no sampling, no parsing of free text, and
a calibrated distribution that can go straight into a weighted vote. It also
sidesteps the failure mode of script 11, where the model's answers had to be
recovered from generated prose.

All six are quantised the same way, with bitsandbytes NF4 applied on load,
rather than pulled as pre-quantised AWQ checkpoints built by different people
from different calibration sets. Qwen3-8B is additionally run unquantised so the
cost of the quantisation itself is visible rather than assumed.

Two consequences worth stating. Constrained scoring bypasses chain-of-thought
entirely, which handicaps a reasoning-distilled model such as
DeepSeek-R1-Distill; that is a real limitation of this sweep rather than a
property of the model. And Qwen3's template is invoked with thinking disabled so
the answer position is immediately after the prefix.

The sweep runs on a stratified subsample of **train+val**, never test, so the
model choice it informs stays inside the protocol. Complementarity against the
encoder is measured against the encoder's own out-of-fold predictions on the
same rows, cached by script 18 -- a model that merely agrees with the encoder
adds nothing to a vote no matter how good its own score is.
"""
import argparse
import csv
import io
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "veripromiseesgdataset")
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts", "s21")
S18 = os.path.join(ROOT, "research", "artifacts", "s18")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
SEED = 20260824

# AWQ was the plan, but transformers 5.16 routes AWQ through gptqmodel, which
# needs torch >= 2.11 against the 2.8 every encoder script in this project is
# validated on. bitsandbytes NF4 quantises any bf16 checkpoint on load instead,
# so the same recipe applies to all six and none of them depends on who happened
# to publish a pre-quantised copy. Qwen3-8B runs in both bf16 and NF4 to price
# the quantisation itself.
MODELS = [
    ("Qwen3-8B",            "Qwen/Qwen3-8B",                                 "bf16"),
    ("Qwen3-8B-nf4",        "Qwen/Qwen3-8B",                                 "nf4"),
    ("Qwen3-14B",           "Qwen/Qwen3-14B",                                "nf4"),
    ("Qwen3-32B",           "Qwen/Qwen3-32B",                                "nf4"),
    ("Gemma-3-12B-it",      "unsloth/gemma-3-12b-it",                        "nf4"),
    ("Gemma-3-27B-it",      "unsloth/gemma-3-27b-it",                        "nf4"),
    ("R1-Distill-Qwen-32B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",      "nf4"),
]

LETTERS = "ABCDE"
SYS = "你是台灣上市公司永續報告書的稽核專家，依照給定準則作答。只輸出一個選項字母。"
ANSWER_PREFIX = "答案："

TASKS = {
    "promise_status": {
        "labels": ["No", "Yes"],
        "q": "這個段落有沒有提出一項 ESG（環境、社會或公司治理）方面的承諾、目標或未來計畫？",
        "opts": ["沒有", "有"],
    },
    "evidence_status": {
        "labels": ["N/A", "No", "Yes"],
        "q": "這個段落所提出的承諾，有沒有附上具體的佐證或實績？",
        "opts": ["這個段落根本沒有提出承諾（不適用）",
                 "有提出承諾，但沒有附上任何佐證",
                 "有提出承諾，也附上了佐證"],
    },
    "evidence_quality": {
        "labels": ["N/A", "Not Clear", "Clear", "Misleading"],
        "q": "這個段落所附的佐證，品質如何？",
        "opts": ["沒有承諾或沒有佐證（不適用）",
                 "資訊缺漏，內容只能部分理解，或流於表面",
                 "具體且充分，足以支持該承諾，內容可理解且合乎邏輯",
                 "該佐證無論真假，與所主張的承諾之間沒有清楚關聯，看起來像在轉移注意力"],
    },
    "verification_timeline": {
        "labels": ["N/A", "already", "within_2_years",
                   "between_2_and_5_years", "more_than_5_years"],
        "q": "這個段落所提出的承諾，何時可以被驗證？",
        "opts": ["這個段落根本沒有提出承諾（不適用）",
                 "已經可以驗證（描述的是已完成或正在進行的事）",
                 "兩年內可以驗證",
                 "兩到五年內可以驗證",
                 "五年以上才能驗證"],
    },
}


def load(fn):
    return list(csv.DictReader(open(os.path.join(D, fn), encoding="utf-8-sig")))


def na(v):
    v = (v or "").strip()
    return "N/A" if v in ("", "N/A") else v


def build_prompt(tok, para, task):
    spec = TASKS[task]
    opts = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(spec["opts"]))
    user = "段落：\n%s\n\n問題：%s\n\n%s\n\n只回答一個選項字母。" % (
        para[:1200], spec["q"], opts)
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
    kw = {}
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # a few templates reject a system role; fold it into the user turn
        msgs = [{"role": "user", "content": SYS + "\n\n" + user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text + ANSWER_PREFIX


def letter_ids(tok, n):
    """Token ids that begin each option letter, with and without a leading space."""
    ids = []
    for i in range(n):
        cand = set()
        for form in (LETTERS[i], " " + LETTERS[i]):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                cand.add(enc[0])
        ids.append(sorted(cand))
    return ids


def score_batch(mdl, tok, prompts, lids, device="cuda"):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=2048).to(device)
    with torch.no_grad():
        logits = mdl(**enc).logits
    # left padding puts the answer position at the end of every row
    last = logits[:, -1, :].float()
    cols = []
    for ids in lids:
        cols.append(last[:, ids].max(dim=-1).values)
    stacked = torch.stack(cols, dim=-1)
    return torch.softmax(stacked, dim=-1).cpu().numpy()


def purge_hub(repo):
    """Drop a model's Hub cache once its scores are saved.

    Seven configurations at bf16 download weight totals around 230 GB. Since the
    per-model probability matrices are cached separately, nothing is lost by
    deleting the weights as soon as a model is scored, and peak usage falls to
    whichever single model is largest.
    """
    import shutil
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    d = os.path.join(home, "hub", "models--" + repo.replace("/", "--"))
    if os.path.isdir(d):
        n = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(d) for f in fs
                if os.path.exists(os.path.join(r, f)))
        shutil.rmtree(d, ignore_errors=True)
        print("  purged %s (%.1f GB)" % (repo, n / 1e9), flush=True)


LOCAL_ROOT = "/workspace/models"


def resolve(repo):
    """Prefer a locally fetched copy over the Hub id.

    The hub client could not pull the larger checkpoints on a small pod -- it
    was OOM-killed on every 4.85 GB shard and opened a fresh .incomplete on
    restart instead of resuming, so 42 GB on disk yielded no complete file.
    Those were fetched with curl into plain directories instead, and
    from_pretrained takes a path as readily as a repo id.
    """
    d = os.path.join(LOCAL_ROOT, repo.split("/")[-1])
    if os.path.isdir(d) and any(f.endswith(".safetensors") for f in os.listdir(d)):
        return d
    return repo


def load_model(repo, kind):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    repo = resolve(repo)
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = dict(device_map="cuda", trust_remote_code=True)
    if kind == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    else:
        kw["dtype"] = torch.bfloat16
    try:
        mdl = AutoModelForCausalLM.from_pretrained(repo, **kw).eval()
    except (ValueError, KeyError):
        # Gemma-3 ships as a multimodal checkpoint; the text tower still takes
        # input_ids and returns logits, so the image-text class works unchanged.
        from transformers import AutoModelForImageTextToText
        mdl = AutoModelForImageTextToText.from_pretrained(repo, **kw).eval()
    return mdl, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="stratified subsample size")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--purge", action="store_true",
                    help="delete each model's Hub cache after scoring it")
    args = ap.parse_args()

    trv = load("vpesg4k_train_1000") + load("vpesg4k_val_1000")
    rng = np.random.default_rng(SEED)
    # stratify on evidence_quality, the most skewed field, and keep every
    # Misleading row since there are only two
    by = {}
    for i, r in enumerate(trv):
        by.setdefault(na(r["evidence_quality"]), []).append(i)
    pick = []
    for k, idxs in by.items():
        if k == "Misleading":
            pick += idxs
            continue
        take = max(1, round(args.n * len(idxs) / len(trv)))
        pick += list(rng.choice(idxs, size=min(take, len(idxs)), replace=False))
    pick = sorted(set(pick))
    rows = [trv[i] for i in pick]
    print("subsample %d rows from train+val" % len(rows), flush=True)
    for f in TASKS:
        c = {}
        for r in rows:
            c[na(r[f])] = c.get(na(r[f]), 0) + 1
        print("  %-24s %s" % (f, c), flush=True)

    # encoder out-of-fold predictions on the same rows, from script 18
    enc_pred = {}
    for f in TASKS:
        p = os.path.join(S18, "%s.npz" % f.replace("_", ""))
        if os.path.exists(p):
            z = np.load(p)
            if len(z["oof"]) == len(trv):
                enc_pred[f] = z["oof"].argmax(1)[pick]
    print("encoder OOF available for: %s" % sorted(enc_pred), flush=True)

    todo = [m for m in MODELS if not args.only or m[0] in args.only]
    results = {}
    for ti, (name, repo, kind) in enumerate(todo):
        cf = os.path.join(CACHE, "%s.npz" % name.replace("/", "_"))
        if os.path.exists(cf):
            z = np.load(cf, allow_pickle=True)
            results[name] = {f: z[f] for f in TASKS if f in z}
            print("[cached] %s" % name, flush=True)
            continue
        print("=== %s (%s, %s) ===" % (name, repo, kind), flush=True)
        try:
            mdl, tok = load_model(repo, kind)
        except Exception as e:
            print("  LOAD FAILED: %s: %s" % (type(e).__name__, str(e)[:300]), flush=True)
            continue

        probs, bs = {}, args.bs
        ok = True
        for f, spec in TASKS.items():
            lids = letter_ids(tok, len(spec["labels"]))
            prompts = [build_prompt(tok, r["data"] or "", f) for r in rows]
            acc, i = [], 0
            while i < len(prompts):
                try:
                    acc.append(score_batch(mdl, tok, prompts[i:i + bs], lids))
                    i += bs
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if bs == 1:
                        print("  OOM at batch 1, skipping %s" % name, flush=True)
                        ok = False
                        break
                    bs = max(1, bs // 2)
                    print("  OOM -> batch %d" % bs, flush=True)
            if not ok:
                break
            probs[f] = np.concatenate(acc)
            print("  %-24s done (batch %d)" % (f, bs), flush=True)
        del mdl
        torch.cuda.empty_cache()
        if not ok:
            if args.purge:
                purge_hub(repo)
            continue
        np.savez(cf, **probs)
        results[name] = probs
        # Qwen3-8B appears twice, bf16 and NF4, so only purge a repo once no
        # later entry still needs it.
        if args.purge and not any(r == repo for _, r, _ in todo[ti + 1:]):
            purge_hub(repo)

    L = []
    A = L.append
    A("# Local LLM sweep: who is worth a vote?")
    A("")
    A("Generated by `research/scripts/21_llm_vote_sweep.py`. Constrained "
      "multiple-choice scoring: one forward pass per (row, subtask), "
      "distribution read from the logits at the answer position over the option "
      "letters. Stratified subsample of **train+val**, n=%d; test is untouched, "
      "so the choice this informs stays inside the protocol." % len(rows))
    A("")
    A("## Zero-shot macro-F1 by subtask")
    A("")
    A("| model | %s |" % " | ".join("`%s`" % f for f in TASKS))
    A("|---|%s" % ("---:|" * len(TASKS)))
    enc_row = []
    for f in TASKS:
        if f in enc_pred:
            y = np.array([TASKS[f]["labels"].index(na(r[f])) for r in rows])
            enc_row.append("%.4f" % f1_score(y, enc_pred[f], average="macro",
                                             zero_division=0))
        else:
            enc_row.append("n/a")
    A("| *encoder (out-of-fold, supervised)* | %s |" % " | ".join(enc_row))
    for name in results:
        cells = []
        for f in TASKS:
            if f not in results[name]:
                cells.append("n/a")
                continue
            y = np.array([TASKS[f]["labels"].index(na(r[f])) for r in rows])
            p = results[name][f].argmax(1)
            cells.append("%.4f" % f1_score(y, p, average="macro", zero_division=0))
        A("| %s | %s |" % (name, " | ".join(cells)))
    A("")
    A("The encoder row is supervised and out-of-fold; the LLM rows are zero-shot. "
      "A zero-shot model that merely loses is still useful to a vote if it loses "
      "*differently*, which the next table measures.")
    A("")

    A("## Complementarity with the encoder")
    A("")
    A("`LLM only right` is the share of rows the LLM gets right and the encoder "
      "gets wrong -- the headroom a vote could actually capture.")
    A("")
    for f in TASKS:
        if f not in enc_pred:
            continue
        A("### `%s`" % f)
        A("")
        A("| model | agreement | both right | LLM only right | encoder only right | neither |")
        A("|---|---:|---:|---:|---:|---:|")
        y = np.array([TASKS[f]["labels"].index(na(r[f])) for r in rows])
        e = enc_pred[f]
        for name in results:
            if f not in results[name]:
                continue
            p = results[name][f].argmax(1)
            A("| %s | %.3f | %.3f | **%.3f** | %.3f | %.3f |"
              % (name, float((p == e).mean()),
                 float(((p == y) & (e == y)).mean()),
                 float(((p == y) & (e != y)).mean()),
                 float(((p != y) & (e == y)).mean()),
                 float(((p != y) & (e != y)).mean())))
        A("")

    A("## Does a simple average vote help?")
    A("")
    A("Encoder out-of-fold probability averaged with each LLM's, equal weight, "
      "on the same subsample.")
    A("")
    A("| model | %s |" % " | ".join("`%s`" % f for f in TASKS))
    A("|---|%s" % ("---:|" * len(TASKS)))
    for name in results:
        cells = []
        for f in TASKS:
            p18 = os.path.join(S18, "%s.npz" % f.replace("_", ""))
            if f not in results[name] or not os.path.exists(p18):
                cells.append("n/a")
                continue
            z = np.load(p18)
            eo = z["oof"][pick]
            y = np.array([TASKS[f]["labels"].index(na(r[f])) for r in rows])
            mix = (eo + results[name][f]) / 2.0
            cells.append("%.4f" % f1_score(y, mix.argmax(1), average="macro",
                                           zero_division=0))
        A("| encoder + %s | %s |" % (name, " | ".join(cells)))
    A("| *encoder alone* | %s |" % " | ".join(enc_row))
    A("")
    A("An equal-weight average is the crudest possible vote and mainly indicates "
      "whether the LLM's probabilities are on a compatible scale. Weight fitting "
      "and per-subtask model assignment come after a model is chosen.")
    A("")

    dest = os.path.join(OUT, "llm_vote_sweep.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"picked_rows": [int(i) for i in pick]},
              open(os.path.join(CACHE, "subsample.json"), "w"))
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
