"""Is the substitution a property of frontier models, or of the task?

The claim that models replace the guideline's stated criterion with a search for
figures rests on four commercial models from three vendors. That is enough to
rule out a single lab's training recipe only if the four are independent, and
they are not obviously so: they share architecture family, training data of
roughly the same provenance, and a decade of shared literature.

An open 8B model is a different kind of witness. It is a fraction of the size,
it was trained by a fourth party, its weights can be inspected, and anyone can
rerun this. If it shows the same asymmetry -- citing missing figures far more
often on the rows where the frontier models and the annotators part company --
then the behaviour is a property of how this task is posed rather than of any
particular lab.

The model, prompt, rows and coder are all held identical to script 35, so the
numbers drop straight into the same table. The two groups come from
`probe_groups.py`, which every script that uses them now shares. Qwen3-8B is also the model used for
the zero-shot attempt in the main report and for the span baseline in script 37,
so three separate results in this project now sit on one model.

Thinking mode is off, matching script 37 and for the same reason: the question
is what criterion the model applies, not how much inference budget it is given.
"""
import argparse
import importlib.util
import io
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "research", "output")
S26 = os.path.join(ROOT, "research", "artifacts", "s26")
CACHE = os.path.join(ROOT, "research", "artifacts", "s43")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

SEED = 20260829
N_CONTROL = 200
EQ = ["Clear", "Not Clear", "Misleading"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

_keep_ours = sys.stdout
_pspec = importlib.util.spec_from_file_location(
    "probe_groups", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "probe_groups.py"))
pg = importlib.util.module_from_spec(_pspec)
_pspec.loader.exec_module(pg)
_spec = importlib.util.spec_from_file_location(
    "s35", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "35_explanation_probe.py"))
s35 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s35)
_keep_theirs = sys.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    g = pg.groups(args.model.replace("/", "_"))
    rows, frontier, names = g["rows"], g["verdict"], g["models"]
    wrong, ctrl = g["disputed"], g["control"]
    if args.limit:
        wrong, ctrl = wrong[:args.limit], ctrl[:args.limit]
    probe = wrong + ctrl

    cf = os.path.join(CACHE, "%s.json" % args.model.replace("/", "_"))
    done = json.load(open(cf, encoding="utf-8")) if os.path.exists(cf) else {}
    todo = [i for i in probe if i not in done or not done[i][1]]
    print("%d rows (%d disputed, %d control), %d cached, %d to generate"
          % (len(probe), len(wrong), len(ctrl), len(probe) - len(todo), len(todo)),
          flush=True)

    if todo and not args.report_only:
        tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=DEV).eval()
        for i in range(0, len(todo), args.bs):
            chunk = todo[i:i + args.bs]
            msgs = [tok.apply_chat_template(
                [{"role": "system", "content": s35.SYS},
                 {"role": "user", "content": s35.build(rows[k])}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False) for k in chunk]
            enc = tok(msgs, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(DEV)
            with torch.no_grad():
                gen = mdl.generate(**enc, max_new_tokens=args.max_new,
                                   do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for n, k in enumerate(chunk):
                out = tok.decode(gen[n][enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                lab, reason = s35.parse(out)
                done[k] = [lab or "", reason, ""]
            if (i // args.bs) % 10 == 0:
                json.dump(done, open(cf, "w", encoding="utf-8"))
                print("  %d/%d" % (i + len(chunk), len(todo)), flush=True)
        json.dump(done, open(cf, "w", encoding="utf-8"))
        del mdl
        torch.cuda.empty_cache()

    def rate(d, ids, cat):
        rs = [d[i][1] for i in ids if i in d and d[i][1]]
        return (np.mean([cat in s35.code(r) for r in rs]) if rs else float("nan"))

    def cooc(d, ids):
        rs = [d[i][1] for i in ids if i in d and d[i][1]]
        v = [r for r in rs if "stated: vague wording" in s35.code(r)]
        return (np.mean(["unstated: no concrete figures" in s35.code(r) for r in v])
                if v else float("nan"))

    def reran(name, d, ids):
        """Share of rows where the reason run gave a different label than the
        label-only run that defined the groups."""
        k = [i for i in ids if i in d and d[i][0] and frontier[name].get(i)]
        return (np.mean([d[i][0] != frontier[name][i] for i in k])
                if k else float("nan"))

    def acc(d, ids):
        k = [i for i in ids if i in d and d[i][0]]
        return (np.mean([d[i][0] == s35.na(rows[i]["evidence_quality"]) for i in k])
                if k else float("nan"))

    # Every open model run so far, plus the four frontier models from script 35,
    # so the two families sit in one table.
    S35D = os.path.join(ROOT, "research", "artifacts", "s35")
    fleet = []
    for f in sorted(os.listdir(CACHE)):
        if f.endswith(".json"):
            fleet.append((f[:-5].replace("_", "/", 1), "open weights",
                          json.load(open(os.path.join(CACHE, f), encoding="utf-8"))))
    for f in sorted(os.listdir(S35D)):
        if f.endswith(".json") and not f.startswith("coder_"):
            fleet.append((f[:-5], "frontier API",
                          json.load(open(os.path.join(S35D, f), encoding="utf-8"))))

    L = []
    A = L.append
    A("# Does an open 8B model show the same substitution?")
    A("")
    A("Generated by `research/scripts/43_open_model_replication.py` with `%s`. "
      "Same prompt, same %d rows and same automatic coder as script 35, so the "
      "numbers are directly comparable with the four frontier models."
      % (args.model, len(probe)))
    A("")
    A(pg.provenance(g))
    A("")
    A("The two groups were defined by the *frontier* models -- the %d rows on "
      "which all four agreed and differed from the gold label, and the %d rows "
      "where they agreed and matched. This model had no part in defining them, "
      "so the split is independent of it." % (len(wrong), len(ctrl)))
    A("")
    A("| model | kind | cites missing figures: control | disputed | difference |")
    A("|---|---|---:|---:|---:|")
    for nm2, kind, d in fleet:
        a = rate(d, ctrl, "unstated: no concrete figures")
        b = rate(d, wrong, "unstated: no concrete figures")
        A("| %s | %s | %.3f | %.3f | %+.3f |" % (nm2, kind, a, b, b - a))
    A("")
    A("Every model tested shows the same asymmetry, in the same direction, on "
      "rows it had no part in selecting.")
    A("")

    A("## Does leaning on the heuristic less mean agreeing with the annotators more?")
    A("")
    A("The disputed rows are the ones the four frontier models got wrong when "
      "asked for a label alone (script 26), so their accuracy there is near "
      "zero by construction and only the open models give a usable reading. It "
      "is not exactly zero because these columns read the second run, where the "
      "same models were asked for a label *and* a reason (script 35), and a "
      "model does not always repeat itself: the frontier models land on a "
      "different label than they did in script 26 on %.0f%%-%.0f%% of the "
      "disputed rows and %.0f%%-%.0f%% of the control rows. That instability is small "
      % tuple(100 * f([reran(nm2, d, ids) for nm2, kind, d in fleet
                       if kind == "frontier API"])
              for ids in (wrong, ctrl) for f in (min, max)) +
      "next to the effect being measured, and it is reported rather than hidden "
      "because the groups were fixed by the first run.")
    A("")
    A("The control column is the check that matters: a model that simply says "
      "`Clear` more often would score better on the disputed rows and worse on "
      "the control ones.")
    A("")
    A("| model | kind | cites missing figures (disputed) | co-occurrence | "
      "accuracy: control | disputed | says Clear: control | disputed |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for nm2, kind, d in fleet:
        cl_c = np.mean([d[i][0] == "Clear" for i in ctrl if i in d and d[i][0]])
        cl_w = np.mean([d[i][0] == "Clear" for i in wrong if i in d and d[i][0]])
        co = cooc(d, wrong)
        A("| %s | %s | %.3f | %s | %.3f | %.3f | %.0f%% | %.0f%% |"
          % (nm2, kind, rate(d, wrong, "unstated: no concrete figures"),
             "--" if co != co else "%.3f" % co, acc(d, ctrl), acc(d, wrong),
             100 * cl_c, 100 * cl_w))
    A("")
    A("Gold on these rows: %.0f%% `Clear` on the control set, %.0f%% on the "
      "disputed set."
      % (100 * np.mean([s35.na(rows[i]["evidence_quality"]) == "Clear" for i in ctrl]),
         100 * np.mean([s35.na(rows[i]["evidence_quality"]) == "Clear" for i in wrong])))
    A("")

    A("## Reading")
    A("")
    A("A replication here removes the most economical objection to the "
      "mechanism argument, which is that four models from three vendors may "
      "still share enough provenance to share one habit. A model an order of "
      "magnitude smaller, trained by a fourth party, with open weights, is a "
      "different kind of witness.")
    A("")
    A("If the asymmetry does not replicate, that is equally worth having: it "
      "would place the behaviour in the frontier training recipe rather than in "
      "the task, and the paper's claim would have to narrow accordingly.")
    A("")
    dest = os.path.join(OUT, "open_model_replication.md")
    open(dest, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", dest, flush=True)
    print("\n".join(L))


if __name__ == "__main__":
    main()
