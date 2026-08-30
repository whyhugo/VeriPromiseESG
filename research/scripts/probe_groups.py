"""One definition of the two paragraph groups, shared by every script that uses
them.

Four scripts (43, 48, 49, 50) all need the same two sets of paragraphs: the ones
where four frontier models agreed with each other and with the annotators, and
the ones where they agreed with each other and not with the annotators. They had
four copies of the selection code, and the copies had drifted apart in two ways
that changed the numbers.

**A failed API call is not an answer.** `gemini-3.7-flash` returned HTTP 503 on
80 rows. Reading an empty verdict as `Clear` manufactures unanimity out of three
models answering and one not, which inflated both groups.

**The control group is the rows that were actually probed.** The control group
was originally a seeded draw of 200 from the pool, and the reasons in
`artifacts/s35` were generated for exactly that draw. Re-drawing from the
corrected pool with the same seed lands on a mostly different 200 rows, of which
only some have a cached reason -- so a script that re-draws silently evaluates
the control stratum on a fraction of the sample it claims. The rule here is the
other way round: the probe set is the set of rows that were probed, and the
groups are that set partitioned by the corrected criteria. Rows that lose
unanimity once failed calls are excluded are reported as dropped rather than
silently reassigned.

Nothing here is a judgement call that could have gone the other way after seeing
the numbers: the probe set is fixed by what was generated, and the partition is
fixed by the labels.
"""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
S26 = os.path.join(ROOT, "research", "artifacts", "s26")
S35 = os.path.join(ROOT, "research", "artifacts", "s35")
S43 = os.path.join(ROOT, "research", "artifacts", "s43")
EQ = ["Clear", "Not Clear", "Misleading"]

_keep = sys.stdout
_spec = importlib.util.spec_from_file_location(
    "s35", os.path.join(HERE, "35_explanation_probe.py"))
s35 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s35)
_keep2 = sys.stdout   # script 35 rebinds stdout; hold both so neither closes it


def groups(open_model="Qwen_Qwen3-8B"):
    """The probe set, partitioned. See the module docstring for the rules."""
    te = s35.load("vpesg4k_test_2000.csv")
    rows = {r["id"]: r for r in te
            if s35.na(r["evidence_status"]) == "Yes"
            and s35.na(r["evidence_quality"]) in EQ}

    verdict = {}
    for f in sorted(os.listdir(S26)):
        if f.endswith(".json"):
            g = json.load(open(os.path.join(S26, f), encoding="utf-8"))
            verdict[f.split("__")[0]] = {k: (v[0] or "") for k, v in g.items()}
    models = sorted(verdict)

    reasons = {}
    for m in models:
        p = os.path.join(S35, "%s.json" % m)
        if os.path.exists(p):
            reasons[m] = json.load(open(p, encoding="utf-8"))

    open_reasons = {}
    p = os.path.join(S43, "%s.json" % open_model)
    if os.path.exists(p):
        open_reasons[open_model.split("_")[-1]] = json.load(
            open(p, encoding="utf-8"))

    # the probe set: every row a reason was generated for, in any of the four
    probe = sorted({i for d in reasons.values() for i, v in d.items() if v[1]}
                   & set(rows))

    def answered(i):
        return all(i in verdict[m] and verdict[m][i] for m in models)

    def unanimous(i):
        return answered(i) and len({verdict[m][i] for m in models}) == 1

    disputed, control, dropped = [], [], []
    for i in probe:
        if not unanimous(i):
            dropped.append(i)
        elif verdict[models[0]][i] == s35.na(rows[i]["evidence_quality"]):
            control.append(i)
        else:
            disputed.append(i)

    return {
        "rows": rows, "verdict": verdict, "models": models,
        "reasons": reasons, "open_reasons": open_reasons,
        "probe": probe, "disputed": disputed, "control": control,
        "dropped": dropped, "s35": s35,
    }


def unanimous_all():
    """Every eligible row, not just the probed ones, under the same rule.

    Script 34 fits its regressions over all 1,335 eligible rows and needs the
    unanimous set among them; the probe set is a sample of that. The rule about
    failed calls is identical, so the two cannot disagree about a row.
    """
    te = s35.load("vpesg4k_test_2000.csv")
    rows = {r["id"]: r for r in te
            if s35.na(r["evidence_status"]) == "Yes"
            and s35.na(r["evidence_quality"]) in EQ}
    verdict = {}
    for f in sorted(os.listdir(S26)):
        if f.endswith(".json"):
            g = json.load(open(os.path.join(S26, f), encoding="utf-8"))
            verdict[f.split("__")[0]] = {k: (v[0] or "") for k, v in g.items()}
    models = sorted(verdict)
    answered = [i for i in rows
                if all(i in verdict[m] and verdict[m][i] for m in models)]
    unan = [i for i in answered
            if len({verdict[m][i] for m in models}) == 1]
    right = [i for i in unan
             if verdict[models[0]][i] == s35.na(rows[i]["evidence_quality"])]
    wrong = [i for i in unan if i not in set(right)]
    return {"rows": rows, "verdict": verdict, "models": models,
            "answered": answered, "unanimous": unan,
            "control": right, "disputed": wrong, "s35": s35}


def provenance(g):
    """One sentence a report can print so the sample size is never implicit."""
    return ("The probe set is the %d rows a reason was generated for. Excluding "
            "rows where any of the four models failed to answer leaves %d "
            "unanimous, split into %d disputed and %d control; the remaining %d "
            "lose unanimity once the failed calls are excluded and are dropped."
            % (len(g["probe"]), len(g["disputed"]) + len(g["control"]),
               len(g["disputed"]), len(g["control"]), len(g["dropped"])))
