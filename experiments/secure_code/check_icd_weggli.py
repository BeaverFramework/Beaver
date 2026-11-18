"""Check ext_constraint_fn's ICD (regex + weggli) against the upstream ICD code.

The reference is the insecure_code_detector package from meta-llama/PurpleLlama
@ 905fd6a686 (CybersecurityBenchmarks/insecure_code_detector), imported as is
with oss.ENABLE_WEGGLI = True. At that commit C/C++ used only the regex and
weggli analyzers, so semgrep is never run; its oss.py still wants a
semgrep-core binary at import time, so a dummy one is put on PATH.

Three checks, each comparing the multiset of pattern ids per input:
  1. upstream's own C/C++ unit tests (tests/test_{c,cpp}_insecure_code_detector.py),
     against their expected counts and against ext_constraint_fn;
  2. every c/cpp base in data/autocomplete.json: generated = line_text,
     code_before = origin_code.split(line_text)[0] (the benchmark's call);
  3. a sample of statement-complete prefixes and violation leaves from the
     dist300 seccode logs, with and without the prescan.

Example (fetch the reference first, e.g. with
`git clone https://github.com/meta-llama/PurpleLlama && git -C PurpleLlama checkout 905fd6a686`):
    python -m experiments.secure_code.check_icd_weggli \\
        --ref-dir PurpleLlama/CybersecurityBenchmarks
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import json
import os
import random
import re
import stat
import sys
import tempfile
import time
from pathlib import Path

from experiments.secure_code.ext_constraint_fn import icd_regex_hits, icd_weggli_hits

_THIS_DIR = Path(__file__).parent
_REPO = _THIS_DIR.parent.parent


def load_reference(ref_dir: str):
    shim = tempfile.mkdtemp(prefix="fake_semgrep_")
    fake = os.path.join(shim, "semgrep-core")
    with open(fake, "w") as f:
        f.write("#!/bin/sh\nexit 1\n")
    os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)
    os.environ["PATH"] = shim + os.pathsep + os.environ["PATH"]
    sys.path.insert(0, ref_dir)
    from insecure_code_detector import insecure_code_detector as icd  # noqa: E402
    from insecure_code_detector import oss  # noqa: E402
    from insecure_code_detector.languages import Language  # noqa: E402

    oss.ENABLE_WEGGLI = True
    oss.ENABLE_SEMGREP = False
    return icd, Language, ref_dir


def ref_ids(ref, language: str, code: str, before: str | None, after: str | None = None) -> list[str]:
    icd, Language, _ = ref
    lang = Language.CPP if language == "cpp" else Language.C
    issues = asyncio.run(icd.analyze(lang, code, before, after))
    return sorted(i.pattern_id for i in issues)


def ours_ids(language: str, codes: list[str], before: str | None, after: str | None = None, prescan=True):
    wg = icd_weggli_hits(language, codes, before, prescan=prescan, code_after=after)
    return [sorted(icd_regex_hits(language, c) + w) for c, w in zip(codes, wg)]


def check_unit_tests(ref) -> list[str]:
    _, _, ref_dir = ref
    tests = Path(ref_dir) / "insecure_code_detector" / "tests"
    out = []
    n = bad_ref = bad_ours = diff = 0
    for lang in ("c", "cpp"):
        src = (tests / f"test_{lang}_insecure_code_detector.py").read_text()
        ns: dict = {"oss": sys.modules["insecure_code_detector.oss"], "Language": ref[1]}
        body = src[src.index(f"{lang.upper()}_REGEX_TEST_CASES"):]
        body = body[: body.index("class ")]
        exec(body, ns)
        for kind in ("REGEX", "WEGGLI"):
            for code, expected in ns[f"{lang.upper()}_{kind}_TEST_CASES"]:
                before, after = ("void foo(){\n", "\n}") if kind == "WEGGLI" else (None, None)
                r = ref_ids(ref, lang, code, before, after)
                o = ours_ids(lang, [code], before, after)[0]
                if kind == "REGEX":  # the regex tests run the full analyze(); count regex issues only
                    r = [p for p in r if "-W-" not in p]
                    o = [p for p in o if "-W-" not in p]
                else:
                    r = [p for p in r if "-W-" in p]
                    o = [p for p in o if "-W-" in p]
                n += 1
                bad_ref += len(r) != expected
                bad_ours += len(o) != expected
                if r != o:
                    diff += 1
                    out.append(f"  DIFF {lang} {kind} {code!r}: upstream {r} ours {o} expected {expected}")
    out.insert(0, f"unit tests: {n} cases; upstream wrong on {bad_ref}, ours wrong on {bad_ours}, "
                  f"upstream != ours on {diff}")
    return out


def check_bases(ref) -> list[str]:
    data = json.load(open(_THIS_DIR / "data" / "autocomplete.json"))
    bases = [x for x in data if x["language"] in ("c", "cpp")]
    diff = []
    per_rule = collections.defaultdict(lambda: [0, 0, 0])  # n, fired (any), fired own rule
    for x in bases:
        before = x["origin_code"].split(x["line_text"])[0]
        r = ref_ids(ref, x["language"], x["line_text"], before)
        o = ours_ids(x["language"], [x["line_text"]], before)[0]
        if r != o:
            diff.append(f"  DIFF base {x['prompt_id']} {x['pattern_id']}: upstream {r} ours {o}")
        s = per_rule[f"{x['analyzer']}:{x['pattern_id']}"]
        s[0] += 1
        s[1] += bool(r)
        s[2] += x["pattern_id"] in r
    out = [f"bases: {len(bases)} c/cpp bases; upstream != ours on {len(diff)}"] + diff[:20]
    out.append("  base insecure line flagged by upstream ICD (given its context), by labelled rule:")
    for k, (n, a, own) in sorted(per_rule.items()):
        out.append(f"    {k:18s} n={n:3d} any={a:3d} own={own:3d}")
    return out


def _records(path):
    txt = open(path).read()
    dec = json.JSONDecoder()
    i = 0
    while i < len(txt):
        while i < len(txt) and txt[i].isspace():
            i += 1
        if i >= len(txt):
            break
        obj, i = dec.raw_decode(txt, i)
        yield obj


def check_logs(ref, n_per_instance: int, n_instances: int, seed: int) -> list[str]:
    ds = {x["dist_rank"]: x for x in json.load(open(_THIS_DIR / "data" / "secure_code_dist_n300_seed0.json"))}
    files = sorted(glob.glob(str(_REPO / "logging/runs/dist300_*_seccode_pc_eps1e-3_d2500/vllm_frontier/logs_*/*.jsonl")))
    rng = random.Random(seed)
    files = rng.sample(files, min(n_instances, len(files)))
    stmt_end = (";", "{", "}", ")", ">", "*/", "```")
    diff_ref, diff_pre, n, t_ref, t_ours = [], [], 0, 0.0, 0.0
    for f in files:
        x = ds[int(Path(f).stem)]
        before = x["origin_code"].split(x["line_text"])[0]
        texts = set()
        for r in _records(f):
            if r["event"] == "transition":
                texts.update(e["decoded"] for e in r["expanded elements"])
                texts.update(v["decoded"] for v in r.get("violation_texts", []))
        cand = sorted(t for t in texts if t.rstrip().endswith(stmt_end))
        cand = rng.sample(cand, min(n_per_instance, len(cand)))
        codes = [(re.findall(r"```(.*?)```", t, re.DOTALL) or [t])[0] for t in cand]
        t0 = time.perf_counter()
        ours = ours_ids(x["language"], codes, before)
        t_ours += time.perf_counter() - t0
        nopre = ours_ids(x["language"], codes, before, prescan=False)
        for c, o, npre in zip(codes, ours, nopre):
            t0 = time.perf_counter()
            r = ref_ids(ref, x["language"], c, before)
            t_ref += time.perf_counter() - t0
            n += 1
            if r != o:
                diff_ref.append(f"  DIFF base {x['source_idx']}: upstream {r} ours {o}  code tail {c[-80:]!r}")
            if o != npre:
                diff_pre.append(f"  PRESCAN base {x['source_idx']}: with {o} without {npre}")
    return [
        f"log sample: {n} texts from {len(files)} instances; upstream != ours on {len(diff_ref)}; "
        f"prescan changed the result on {len(diff_pre)}",
        f"  time: upstream one file per call {1e3 * t_ref / max(n, 1):.1f} ms/text, "
        f"ours batched+prescan {1e3 * t_ours / max(n, 1):.1f} ms/text",
    ] + diff_ref[:20] + diff_pre[:20]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", required=True, help="PurpleLlama/CybersecurityBenchmarks @ 905fd6a686")
    ap.add_argument("--log-instances", type=int, default=100)
    ap.add_argument("--per-instance", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    ref = load_reference(args.ref_dir)
    for section in (check_unit_tests(ref), check_bases(ref),
                    check_logs(ref, args.per_instance, args.log_instances, args.seed)):
        print("\n".join(section))
        print()


if __name__ == "__main__":
    main()
