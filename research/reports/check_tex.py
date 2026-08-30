# -*- coding: utf-8 -*-
"""Structural check on the draft: environments, labels, references.

Run before handing the file to Overleaf. It catches the two failure modes that
have actually happened here -- an unclosed environment after a large edit, and a
`\\ref` to a label that a rewrite removed.
"""
import collections
import io
import os
import re
import sys

F = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "paper_draft_zh_2026-08-30.tex")
s = io.open(F, encoding="utf-8").read()
out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

labs = set(re.findall(re.escape("\\label") + r"\{([^}]+)\}", s))
refs = set(re.findall(re.escape("\\ref") + r"\{([^}]+)\}", s))

depth = collections.Counter()
stack = []
for m in re.finditer(re.escape("\\") + r"(begin|end)\{([^}]+)\}", s):
    if m.group(1) == "begin":
        stack.append((m.group(2), s[:m.start()].count("\n") + 1))
        depth[m.group(2)] += 1
    else:
        depth[m.group(2)] -= 1
        if stack and stack[-1][0] == m.group(2):
            stack.pop()

bad = {k: v for k, v in depth.items() if v}
out.write("labels defined : %d\n" % len(labs))
out.write("refs used      : %d\n" % len(refs))
out.write("unresolved refs: %s\n" % (sorted(refs - labs) or "none"))
out.write("unused labels  : %s\n" % (sorted(labs - refs) or "none"))
out.write("unbalanced env : %s\n" % (bad or "none"))
if bad and stack:
    out.write("  still open at EOF: %s\n" % stack)
out.write("lines          : %d\n" % s.count("\n"))
out.flush()
sys.exit(1 if (bad or refs - labs) else 0)
