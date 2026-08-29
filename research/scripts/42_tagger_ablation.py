"""Which of the clause tagger's design choices is doing the work?

Section 5 of the side report gives one number per field and no account of where
it comes from. Three choices were made when the model was built, each defended
in prose and none measured:

  the unit      clauses, split on commas as well as full stops, rather than
                sentences. Script 28 argued for this from an alignment ceiling
                of 0.9748 against 0.5414, but that was a property of the
                annotation, not of a trained model.
  the context   the paragraph is encoded once and each clause is read out of
                that encoding, rather than each clause being encoded alone.
  the encoder   `chinese-roberta-wwm-ext-large`, chosen because it was already
                the strongest backbone in the main report.

This removes them one at a time. It also repeats the published configuration
under three random seeds, because every figure this project reports rests on a
single seed and a reader cannot currently tell a real difference from noise.
Wall-clock per fold is recorded throughout, since the venue's papers report it
and this one so far does not.

The context ablation is done by masking attention so that tokens may attend only
within their own clause. Parameter count, pooling and training budget are
identical to the full model; the only thing removed is the ability to look
outside the clause.
"""
import argparse
import importlib.util
import io
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import KFold

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "research", "output")
CACHE = os.path.join(ROOT, "research", "artifacts", "s42")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

_keep_ours = sys.stdout
_spec = importlib.util.spec_from_file_location(
    "s32", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "32_clause_tagger.py"))
s32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s32)
_keep_theirs = sys.stdout

PUBLISHED_SEED = 20260828
PUBLISHED_BACKBONE = "hfl/chinese-roberta-wwm-ext-large"


def clause_block_mask(sel, pad):
    """Block-diagonal attention: a token may attend only inside its clause.

    `sel` is the mean-pooling matrix, non-zero exactly where a token belongs to
    a clause, so it already carries the assignment the mask needs.
    """
    memb = (sel > 0).float()                       # B x C x L
    same = torch.einsum("bcl,bcm->blm", memb, memb).clamp(max=1.0)
    # tokens in no clause (specials, padding) keep their normal visibility
    none = (memb.sum(1) == 0).float()              # B x L
    same = torch.maximum(same, none.unsqueeze(1))
    same = torch.maximum(same, none.unsqueeze(2))
    return same * pad.unsqueeze(1).float()


def run_fold(tr_rows, eval_sets, field, epochs, bs, lr, backbone, no_context):
    """Script 32's fold, with an optional attention mask over clause blocks."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(backbone)
    mdl = s32.Tagger(backbone).float().to("cuda")
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    from torch.utils.data import DataLoader
    dl = DataLoader(s32.ClauseDS(tr_rows, tok, field), batch_size=bs, shuffle=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, len(dl) * epochs), pct_start=0.1)

    pos = sum(sum(s32.clause_labels(r["data"] or "", r.get(field))) for r in tr_rows)
    tot = sum(len(s32.clauses_of(r["data"] or "")) for r in tr_rows)
    w = torch.tensor([tot / (2 * max(1, tot - pos)), tot / (2 * max(1, pos))],
                     dtype=torch.float32, device="cuda")
    lossf = torch.nn.CrossEntropyLoss(weight=w, ignore_index=-100)

    def forward(enc, sel):
        if no_context:
            enc = dict(enc)
            enc["attention_mask"] = clause_block_mask(sel, enc["attention_mask"])
        return mdl(enc, sel)

    t0 = time.time()
    mdl.train()
    for _ in range(epochs):
        for enc, sel, y, _ in dl:
            enc = {k: v.to("cuda") for k, v in enc.items()}
            sel, y = sel.to("cuda"), y.to("cuda")
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logit = forward(enc, sel).float()
            loss = lossf(logit.reshape(-1, 2), y.reshape(-1))
            loss.backward()
            opt.step()
            sched.step()
    train_s = time.time() - t0

    mdl.eval()
    outs, t1 = [], time.time()
    for rows in eval_sets:
        probs = np.zeros((len(rows), s32.MAX_CLAUSES))
        dle = DataLoader(s32.ClauseDS(rows, tok, field), batch_size=bs * 2)
        with torch.no_grad():
            for enc, sel, _, idx in dle:
                enc = {k: v.to("cuda") for k, v in enc.items()}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = forward(enc, sel.to("cuda")).float()
                p = torch.softmax(lg, -1)[:, :, 1].cpu().numpy()
                for n, j in enumerate(idx.numpy()):
                    probs[j] = p[n]
        outs.append(probs)
    infer_s = time.time() - t1
    del mdl
    torch.cuda.empty_cache()
    return outs, train_s, infer_s


def one_variant(name, field, args, seed=PUBLISHED_SEED,
                backbone=PUBLISHED_BACKBONE, sentence_unit=False,
                no_context=False):
    cf = os.path.join(CACHE, "%s__%s.json" % (name.replace(" ", "_"), field))
    if os.path.exists(cf):
        print("[%s] cached" % name, flush=True)
        return json.load(open(cf, encoding="utf-8"))

    # Script 32 splits on CLAUSE_END; setting it to SENT_END gives the sentence
    # unit with every other part of the pipeline untouched.
    saved = s32.CLAUSE_END
    if sentence_unit:
        s32.CLAUSE_END = s32.SENT_END
    torch.manual_seed(seed)
    np.random.seed(seed)

    trv = [r for r in s32.load("vpesg4k_train_1000") + s32.load("vpesg4k_val_1000")
           if s32.ok(r.get(field))]
    te = [r for r in s32.load("vpesg4k_test_2000.csv") if s32.ok(r.get(field))]
    oof = np.zeros((len(trv), s32.MAX_CLAUSES))
    tp = np.zeros((len(te), s32.MAX_CLAUSES))
    train_s = infer_s = 0.0
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=seed)
    for k, (ia, ib) in enumerate(kf.split(trv)):
        (pv, pt), ts, isec = run_fold(
            [trv[i] for i in ia], [[trv[i] for i in ib], te], field,
            args.epochs, args.bs, args.lr, backbone, no_context)
        for n, j in enumerate(ib):
            oof[j] = pv[n]
        tp += pt / args.folds
        train_s += ts
        infer_s += isec
        print("  [%s] fold %d/%d done (%.0fs train)"
              % (name, k + 1, args.folds, ts), flush=True)

    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.1, 0.91, 0.05):
        f = s32.score(trv, oof, field, float(t))[2]
        if f > best_f:
            best_f, best_t = f, float(t)
    p, r, f1, iou, em = s32.score(te, tp, field, best_t)
    got = {"name": name, "field": field, "seed": seed, "backbone": backbone,
           "threshold": best_t, "oof_char_f1": best_f,
           "test_p": p, "test_r": r, "test_char_f1": f1, "test_iou": iou,
           "test_em": em, "train_s": train_s, "infer_s": infer_s,
           "n_clauses": sum(len(s32.clauses_of(x["data"] or "")) for x in te)}
    s32.CLAUSE_END = saved
    json.dump(got, open(cf, "w", encoding="utf-8"))
    print("[%s] test char-F1 %.4f, EM %.1f%%, %.0fs" % (name, f1, em, train_s),
          flush=True)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="promise_string")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", nargs="*", type=int,
                    default=[PUBLISHED_SEED, 20260829, 20260830])
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    F = args.field

    runs = []
    if not args.report_only:
        for i, sd in enumerate(args.seeds):
            runs.append(one_variant("full seed %d" % sd, F, args, seed=sd))
        runs.append(one_variant("sentence unit", F, args, sentence_unit=True))
        runs.append(one_variant("no paragraph context", F, args, no_context=True))
        runs.append(one_variant("macbert backbone", F, args,
                                backbone="hfl/chinese-macbert-large"))
    else:
        for f in sorted(os.listdir(CACHE)):
            if f.endswith("__%s.json" % F):
                runs.append(json.load(open(os.path.join(CACHE, f),
                                           encoding="utf-8")))

    by = {r["name"]: r for r in runs}
    seeds = [by[k] for k in by if k.startswith("full seed")]

    L = []
    A = L.append
    A("# What each design choice in the clause tagger is worth")
    A("")
    A("Generated by `research/scripts/42_tagger_ablation.py` on `%s`. Every "
      "variant uses the same folds, epochs, batch size and learning rate as the "
      "published run; only the named choice changes. The decision threshold is "
      "refitted out-of-fold for each variant, so no variant is helped by a "
      "threshold tuned for another." % F)
    A("")

    if len(seeds) > 1:
        f1s = [r["test_char_f1"] for r in seeds]
        ems = [r["test_em"] for r in seeds]
        A("## How much of a difference is noise?")
        A("")
        A("The published configuration, repeated under %d seeds." % len(seeds))
        A("")
        A("| measure | mean | standard deviation | range |")
        A("|---|---:|---:|---|")
        A("| test char-F1 | %.4f | %.4f | %.4f -- %.4f |"
          % (np.mean(f1s), np.std(f1s, ddof=1), min(f1s), max(f1s)))
        A("| test exact match | %.1f%% | %.1f | %.1f -- %.1f |"
          % (np.mean(ems), np.std(ems, ddof=1), min(ems), max(ems)))
        A("")
        A("Any ablation below that moves char-F1 by less than about %.4f is "
          "inside seed noise and should not be read as an effect."
          % (2 * np.std(f1s, ddof=1)))
        A("")

    base = np.mean([r["test_char_f1"] for r in seeds]) if seeds else float("nan")
    A("## Removing one choice at a time")
    A("")
    A("| variant | char-F1 | IoU | exact match | vs published |")
    A("|---|---:|---:|---:|---:|")
    if seeds:
        A("| **published** (clause unit, paragraph context, roberta) | "
          "**%.4f** | %.4f | %.1f%% | -- |"
          % (base, np.mean([r["test_iou"] for r in seeds]),
             np.mean([r["test_em"] for r in seeds])))
    for k in ["sentence unit", "no paragraph context", "macbert backbone"]:
        if k not in by:
            continue
        r = by[k]
        A("| %s | %.4f | %.4f | %.1f%% | %+.4f |"
          % (k, r["test_char_f1"], r["test_iou"], r["test_em"],
             r["test_char_f1"] - base))
    A("")

    A("## What it costs to run")
    A("")
    A("One NVIDIA RTX 4090 (24 GB). Training time is the total across %d folds; "
      "inference is over the out-of-fold and test sets together." % args.folds)
    A("")
    A("| variant | training | inference | clauses scored | per 1,000 clauses |")
    A("|---|---:|---:|---:|---:|")
    for r in runs:
        A("| %s | %.0f s | %.0f s | %d | %.2f s |"
          % (r["name"], r["train_s"], r["infer_s"], r["n_clauses"],
             1000.0 * r["infer_s"] / max(1, r["n_clauses"] * (args.folds + 1))))
    A("")

    A("## Reading")
    A("")
    A("The unit and the context are the two claims section 5 makes in prose. "
      "This is where they are either earned or withdrawn. The backbone row is "
      "included because a reader would otherwise wonder how much of the result "
      "is the encoder rather than the formulation.")
    A("")
    dest = os.path.join(OUT, "tagger_ablation_%s.md" % F)
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
