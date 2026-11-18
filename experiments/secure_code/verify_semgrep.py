"""Verify semgrep actually FIRES, not merely that it imports.

A version check is not enough. semgrep 1.157 loads the rules, accepts
--experimental, and then scans zero files because it only walks git-TRACKED
paths while insecure_code_detector hands it NamedTemporaryFile targets -- so
labels silently degrade to the regex analyser. See the pin comment in
pyproject.toml.

The payload matters too: a hashlib-md5 snippet passes even when semgrep is
completely broken, because _is_semgrep_scan_recommended prescans with regexes
and can skip the semgrep pass entirely. subprocess(shell=True) is a rule only
semgrep carries.

Run from experiments/secure_code -- insecure_code_detector is a bare
top-level module there, not part of the beaver package.
"""
import asyncio, sys

from insecure_code_detector import insecure_code_detector as icd
from insecure_code_detector import oss
from insecure_code_detector.languages import Language

print("semgrep binary :", oss.SEMGREP_BINARY_PATH)
print("project-root   :", oss.INCLUDE_SEMGREP_PROJECT_ROOT_PARAMETER)

CODE = "import subprocess\ndef f(cmd):\n    subprocess.run(cmd, shell=True)\n"
issues = asyncio.run(icd.analyze(Language.PYTHON, CODE))
for i in issues:
    print(" found:", i.analyzer, i.pattern_id, i.cwe_id)

analyzers = {str(i.analyzer) for i in issues}
if not any("SEMGREP" in a.upper() for a in analyzers):
    sys.exit(
        f"FAIL: no SEMGREP finding (analyzers={analyzers or '{}'}).\n"
        "semgrep is not firing -- labels would come from the regex analyser "
        "alone. Check the semgrep pin in pyproject.toml."
    )
print("OK: semgrep fired")
