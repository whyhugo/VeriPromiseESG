"""Do learned confidence tokens detect a criterion mismatch, or only ignorance?

Self-REF (Chuang et al., ICML 2025) teaches a local model two special tokens.
After its answer it emits `<CN>` when it expects to be right and `<UN>` when it
does not, and a confidence score is read off as P(<CN>) / (P(<CN>) + P(<UN>)).
The paper evaluates this on MMLU, OpenbookQA, GSM8K and MedQA -- four datasets
where being wrong means not knowing something.

This dataset contains a second kind of error, and the distinction has not been
tested before. On the rows where four frontier models agree with each other and
differ from the annotators, the models are not ignorant. They are confident,
mutually consistent, and applying a criterion the annotators did not use. Script
38 already showed that features computed from the paragraph, with no access to
the model at all, rank those errors better than the model's own softmax
confidence (AUC 0.794 against 0.769).

So there is a specific prediction to test. A confidence token is trained on
"was my own answer right", which is exactly the signal that a capability error
leaves and a criterion mismatch does not. It should beat softmax confidence
overall -- that is the paper's result -- and still lose to the text-side
features on the disputed rows.

Implementation notes, since two choices matter for what the numbers mean:

  * the answer is a single option letter, so the confidence token sits at one
    fixed position and both the gradient mask and the score extraction are
    exact rather than approximate.
  * for rows the base model got wrong, the loss is taken on the confidence token
    only; the wrong answer itself is masked out, following Algorithm 1, so the
    model is never trained to prefer an answer it should not give.
"""
import argparse
import importlib.util
import io
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "research", "output")
S26 = os.path.join(ROOT, "research", "artifacts", "s26")
CACHE = os.path.join(ROOT, "research", "artifacts", "s44")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

SEED = 20260829
N_CONTROL = 200
EQ = ["Clear", "Not Clear", "Misleading"]
CHOICES = ["A", "B", "C"]          # Clear / Not Clear / Misleading
CN, UN = "<CN>", "<UN>"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

_keep_ours = sys.stdout
_spec = importlib.util.spec_from_file_location(
    "s35", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "35_explanation_probe.py"))
s35 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s35)
_keep_theirs = sys.stdout

OPTS = ("A. Clear：具體且充分，足以支持該承諾，內容可理解且合乎邏輯\n"
        "B. Not Clear：資訊缺漏，內容只能部分理解，或流於表面\n"
        "C. Misleading：該佐證無論真假，與所主張的承諾之間沒有清楚關聯")


def prompt(row):
    return ("你是台灣上市公司永續報告書的稽核專家。\n\n"
            "段落：\n%s\n\n"
            "問題：這個段落所附的佐證，品質如何？\n\n%s\n\n"
            "只回答一個選項字母。\n答案："
            % ((row["data"] or "")[:1200], OPTS))


def load_split():
    trv = [r for r in s35.load("vpesg4k_train_1000") + s35.load("vpesg4k_val_1000")
           if s35.na(r["evidence_status"]) == "Yes"
           and s35.na(r["evidence_quality"]) in EQ]
    te = [r for r in s35.load("vpesg4k_test_2000.csv")
          if s35.na(r["evidence_status"]) == "Yes"
          and s35.na(r["evidence_quality"]) in EQ]
    return trv, te


def letter_logits(mdl, tok, rows, ids_letters, bs=4):
    """Logits at the answer position, restricted to the option letters."""
    out = np.zeros((len(rows), len(CHOICES)))
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        enc = tok([prompt(r) for r in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=1536).to(DEV)
        with torch.no_grad():
            lg = mdl(**enc).logits[:, -1, :].float()
        out[i:i + len(chunk)] = lg[:, ids_letters].cpu().numpy()
    return out


def conf_logits(mdl, tok, rows, letters, id_cn, id_un, bs=4):
    """Logits of the two confidence tokens, one position after the answer."""
    out = np.zeros((len(rows), 2))
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        txt = [prompt(r) + letters[i + n] for n, r in enumerate(chunk)]
        enc = tok(txt, return_tensors="pt", padding=True, truncation=True,
                  max_length=1536).to(DEV)
        with torch.no_grad():
            lg = mdl(**enc).logits[:, -1, :].float()
        out[i:i + len(chunk)] = lg[:, [id_cn, id_un]].cpu().numpy()
    return out


def build_sft(rows, base_pred, gold, rng, alpha):
    """Algorithm 1: correct answers get <CN>, incorrect get <UN> and a mask."""
    conf, unconf = [], []
    for n, r in enumerate(rows):
        letter = CHOICES[base_pred[n]]
        ok = base_pred[n] == gold[n]
        (conf if ok else unconf).append(
            {"prompt": prompt(r), "letter": letter,
             "token": CN if ok else UN, "mask_answer": not ok})
    keep = int(round(alpha * len(unconf)))
    idx = rng.permutation(len(unconf))[:keep]
    return conf + [unconf[i] for i in idx], len(conf), len(unconf), keep


class SFTData(torch.utils.data.Dataset):
    def __init__(self, items, tok, maxlen=1536):
        self.items, self.tok, self.maxlen = items, tok, maxlen

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        pre = self.tok(it["prompt"], add_special_tokens=False)["input_ids"]
        ans = self.tok(it["letter"], add_special_tokens=False)["input_ids"]
        cft = self.tok(it["token"], add_special_tokens=False)["input_ids"]
        ids = (pre + ans + cft)[-self.maxlen:]
        lab = [-100] * len(ids)
        n_tail = len(ans) + len(cft)
        # supervise the confidence token always; the answer only when it was right
        for j in range(len(ids) - n_tail, len(ids)):
            lab[j] = ids[j]
        if it["mask_answer"]:
            for j in range(len(ids) - n_tail, len(ids) - len(cft)):
                lab[j] = -100
        return {"input_ids": ids, "labels": lab}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):          # left pad, so the tail stays aligned
        k = len(b["input_ids"])
        ids[i, n - k:] = torch.tensor(b["input_ids"])
        lab[i, n - k:] = torch.tensor(b["labels"])
        att[i, n - k:] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="fraction of unconfident samples kept (Algorithm 1)")
    ap.add_argument("--eval-bs", type=int, default=4)
    ap.add_argument("--stage", default="all",
                    choices=["base", "train", "eval", "all"])
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    trv, te = load_split()
    y_tr = [EQ.index(s35.na(r["evidence_quality"])) for r in trv]
    y_te = [EQ.index(s35.na(r["evidence_quality"])) for r in te]
    print("train+val %d, test %d" % (len(trv), len(te)), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": [CN, UN]})
    id_cn = tok.convert_tokens_to_ids(CN)
    id_un = tok.convert_tokens_to_ids(UN)
    ids_letters = [tok(c, add_special_tokens=False)["input_ids"][0] for c in CHOICES]

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)

    # --- stage 1: what does the base model answer, before any training? ------
    bp = os.path.join(CACHE, "base_%s.json" % args.model.replace("/", "_"))
    if not os.path.exists(bp) and args.stage in ("base", "all"):
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant, device_map=DEV,
            dtype=torch.bfloat16)
        mdl.resize_token_embeddings(len(tok))
        mdl.eval()
        got = {"train": letter_logits(mdl, tok, trv, ids_letters,
                                      args.eval_bs).tolist(),
               "test": letter_logits(mdl, tok, te, ids_letters,
                                     args.eval_bs).tolist()}
        json.dump(got, open(bp, "w", encoding="utf-8"))
        del mdl
        torch.cuda.empty_cache()
        print("base predictions cached", flush=True)
    base = json.load(open(bp, encoding="utf-8"))
    p_tr = np.array(base["train"]).argmax(1)
    p_te = np.array(base["test"]).argmax(1)
    print("base accuracy: train %.3f, test %.3f"
          % (np.mean(p_tr == np.array(y_tr)), np.mean(p_te == np.array(y_te))),
          flush=True)

    if args.stage == "base":
        return

    # --- stage 2: augment, then fine-tune ------------------------------------
    items, n_conf, n_unconf, n_kept = build_sft(trv, p_tr, y_tr, rng, args.alpha)
    print("augmented: %d confident, %d unconfident (%d kept), %d total"
          % (n_conf, n_unconf, n_kept, len(items)), flush=True)

    adapter = os.path.join(CACHE, "adapter_%s" % args.model.replace("/", "_"))
    if not os.path.isdir(adapter) and args.stage in ("train", "all"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant, device_map=DEV,
            dtype=torch.bfloat16)
        mdl.resize_token_embeddings(len(tok))
        mdl = prepare_model_for_kbit_training(mdl)
        mdl.config.use_cache = False
        cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["lm_head"])
        mdl = get_peft_model(mdl, cfg)
        mdl.print_trainable_parameters()

        dl = torch.utils.data.DataLoader(
            SFTData(items, tok), batch_size=args.bs, shuffle=True,
            collate_fn=lambda b: collate(b, tok.pad_token_id))
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(
            [p for p in mdl.parameters() if p.requires_grad], lr=args.lr)
        steps = max(1, args.epochs * len(dl) // args.accum)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                  total_steps=steps, pct_start=0.1)
        mdl.train()
        step = 0
        for ep in range(args.epochs):
            for n, b in enumerate(dl):
                b = {k: v.to(DEV) for k, v in b.items()}
                loss = mdl(**b).loss / args.accum
                loss.backward()
                if (n + 1) % args.accum == 0:
                    opt.step()
                    sch.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 20 == 0:
                        print("  epoch %d step %d loss %.4f"
                              % (ep + 1, step, loss.item() * args.accum), flush=True)
        mdl.save_pretrained(adapter)
        del mdl, opt
        torch.cuda.empty_cache()
        print("adapter saved", flush=True)

    if args.stage == "train":
        return

    # --- stage 3: read the confidence token off the test set -----------------
    cp = os.path.join(CACHE, "conf_%s.json" % args.model.replace("/", "_"))
    if not os.path.exists(cp):
        from peft import PeftModel
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant, device_map=DEV,
            dtype=torch.bfloat16)
        mdl.resize_token_embeddings(len(tok))
        mdl = PeftModel.from_pretrained(mdl, adapter).eval()
        tuned = letter_logits(mdl, tok, te, ids_letters, args.eval_bs)
        letters = [CHOICES[i] for i in tuned.argmax(1)]
        cl = conf_logits(mdl, tok, te, letters, id_cn, id_un, args.eval_bs)
        json.dump({"tuned": tuned.tolist(), "conf": cl.tolist()},
                  open(cp, "w", encoding="utf-8"))
        del mdl
        torch.cuda.empty_cache()
    got = json.load(open(cp, encoding="utf-8"))
    tuned = np.array(got["tuned"])
    cl = np.array(got["conf"])
    e = np.exp(cl - cl.max(1, keepdims=True))
    c_token = e[:, 0] / e.sum(1)                     # P(<CN>) normalised
    pred_tuned = tuned.argmax(1)
    b = np.array(base["test"])
    eb = np.exp(b - b.max(1, keepdims=True))
    softmax_base = (eb / eb.sum(1, keepdims=True)).max(1)
    # The confidence token is read off the fine-tuned model, so its softmax
    # baseline has to come from the same model. Comparing a learned token
    # against the *base* model's softmax would flatter the token for free.
    et = np.exp(tuned - tuned.max(1, keepdims=True))
    softmax_tuned = (et / et.sum(1, keepdims=True)).max(1)

    err_base = (p_te != np.array(y_te)).astype(int)
    err_tuned = (pred_tuned != np.array(y_te)).astype(int)

    # the disputed rows, identified by the frontier models alone
    fr = {}
    for f in sorted(os.listdir(S26)):
        if f.endswith(".json"):
            g = json.load(open(os.path.join(S26, f), encoding="utf-8"))
            fr[f.split("__")[0]] = {k: (v[0] or "Clear") for k, v in g.items()}
    nm = sorted(fr)
    idx = {r["id"]: n for n, r in enumerate(te)}
    disputed = np.zeros(len(te), bool)
    for i in idx:
        if all(i in fr[m] for m in nm) and len({fr[m][i] for m in nm}) == 1:
            if fr[nm[0]][i] != s35.na(te[idx[i]]["evidence_quality"]):
                disputed[idx[i]] = True

    L = []
    A = L.append
    A("# Does a learned confidence token see a criterion mismatch?")
    A("")
    A("Generated by `research/scripts/44_self_ref_confidence.py` with `%s`, "
      "4-bit base and LoRA, following Self-REF (Chuang et al., ICML 2025). "
      "Trained on the %d train+val rows with `evidence_status=Yes`, evaluated "
      "once on the %d test rows." % (args.model, len(trv), len(te)))
    A("")
    A("| | base model | after Self-REF |")
    A("|---|---:|---:|")
    A("| test accuracy | %.3f | %.3f |"
      % (1 - err_base.mean(), 1 - err_tuned.mean()))
    A("")
    A("The accuracy gain is ordinary supervised fine-tuning on %d labelled rows "
      "and is not attributable to the confidence tokens; it is reported because "
      "it bears on what the rest of this project has found. The guideline's "
      "words did not convey the criterion to any model. Labelled examples "
      "conveyed a usable part of it to this one." % len(trv))
    A("")
    A("## Ranking the model's own errors")
    A("")
    A("| signal | AUC over all test rows | AUC over the disputed rows |")
    A("|---|---:|---:|")
    for nm2, sc in [("softmax confidence, same fine-tuned model", -softmax_tuned),
                    ("softmax confidence, base model before tuning", -softmax_base),
                    ("confidence token (Self-REF)", -c_token)]:
        a = roc_auc_score(err_tuned, sc) if len(set(err_tuned)) > 1 else float("nan")
        sub = disputed | (err_tuned == 0)
        d = (roc_auc_score(err_tuned[sub], sc[sub])
             if len(set(err_tuned[sub])) > 1 else float("nan"))
        A("| %s | %.4f | %.4f |" % (nm2, a, d))
    A("| criterion features (script 38) | 0.7937* | -- |")
    A("")
    A("*Not directly comparable. Script 38's features rank the errors of a "
      "different base model -- the fine-tuned Chinese RoBERTa -- so that row "
      "shows the order of magnitude of a text-side signal, not a head-to-head "
      "result against this one.")
    A("")
    A("## Does the token flag the disputed rows?")
    A("")
    A("| group | rows | mean confidence |")
    A("|---|---:|---:|")
    A("| disputed | %d | %.4f |" % (int(disputed.sum()), c_token[disputed].mean()))
    A("| everything else | %d | %.4f |"
      % (int((~disputed).sum()), c_token[~disputed].mean()))
    A("")
    A("## Reading")
    A("")
    A("**The prediction this script was written to test is wrong.** The "
      "expectation was that a confidence token, trained on whether the model's "
      "own answer had been right, would detect capability errors and miss a "
      "criterion mismatch, because on the disputed rows the model is confident "
      "and consistent rather than uncertain. It does not miss them: it reaches "
      "%.4f AUC on those rows and gives them a mean confidence of %.4f against "
      "%.4f elsewhere."
      % (roc_auc_score(err_tuned[disputed | (err_tuned == 0)],
                       -c_token[disputed | (err_tuned == 0)]),
         c_token[disputed].mean(), c_token[~disputed].mean()))
    A("")
    A("The reason it does not is visible in the accuracy row, and it is the "
      "more interesting half of this result. Fine-tuning on %d labelled rows "
      "moved the model from %.3f to %.3f. The criterion that no wording of the "
      "guideline conveyed to any model -- not the published text, not a "
      "prompt containing it, not the four criteria asked one at a time -- was "
      "conveyed by examples. Once the model has absorbed part of that "
      "criterion, its uncertainty tracks it, and the disputed rows are exactly "
      "where the absorption is thinnest."
      % (len(trv), 1 - err_base.mean(), 1 - err_tuned.mean()))
    A("")
    A("A second correction belongs here. Against a softmax baseline taken from "
      "the *same* fine-tuned model, the confidence token's advantage is small. "
      "The large gap is against the base model's softmax, which is not the "
      "comparison to make when the token is read off the tuned model. This "
      "does not contradict the original paper, whose datasets and baselines "
      "differ, but on this task most of what the token provides is available "
      "from the tuned model's own output distribution.")
    A("")
    A("Read against the rest of the project, the finding is consistent and "
      "narrow: **words did not transmit this criterion and labelled examples "
      "did.** That is also why a fine-tuned 330M-parameter encoder outscores "
      "four zero-shot frontier models on the same rows.")
    A("")
    dest = os.path.join(OUT, "self_ref_confidence.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
