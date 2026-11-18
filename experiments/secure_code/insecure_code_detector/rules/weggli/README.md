c.yaml and cpp.yaml are verbatim copies of CyberSecEval's weggli rules:
https://github.com/meta-llama/PurpleLlama/tree/905fd6a686/CybersecurityBenchmarks/insecure_code_detector/rules/weggli
(git blob 350f946c for c.yaml, 9099d4df for cpp.yaml). 905fd6a686 is the last
upstream commit that shipped the weggli analyzer; df6a487973 (Nov 2024)
removed it. These are the rules the autocomplete dataset's C-W-* / CPP-W-*
labels come from. Used by experiments/secure_code/ext_constraint_fn.py.
