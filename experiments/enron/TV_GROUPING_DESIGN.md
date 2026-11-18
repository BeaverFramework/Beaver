# Designing distributions for the cached-frontier (TV-grouped) verifier

Everything below was derived empirically in the Enron privacy task with
`Qwen/Qwen3-4B-Instruct-2507`, `gen_length=16`, ε (`early_termination_threshold`)
= 0.1. The companion builder is [`build_tv_grouped_distribution.py`](build_tv_grouped_distribution.py).

Cost is measured in **forward passes (FP)** per instance (`cost_analysis.py`,
`total_fp`).

---

## 1. The cost model (grounded in the code)

- **Uncached** (`verifier: frontier`): every instance pays a full cold frontier
  search. Total = `Σ_i C_cold(i)`.
- **Cached** (`verifier: cachedfrontier`,
  [`cached_frontier_verifier.py`](../../beaver/verifiers/cached_frontier_verifier.py)):
  instances are bucketed by `instance["group"]`; **each group has its own
  `TokenTrieCache` and caches never cross groups**. Within a group instances run
  sequentially: the first (seed) pays `C_cold`; each later member warm-starts
  from the cache the earlier ones grew.
  - A member **warm-skips** (near-free) when the cache alone trips the
    early-termination gate — `v_prob ≥ 1−ε` (cached **violation**) or
    `c_prob > ε` (cached completion) — in `build_frontier_from_cache`.
  - Otherwise the warm-start **re-scores the cached antichain** (cost ∝ cache
    size) and then continues exploring.

**The gap you are maximizing:**

```
savings = uncached − cached ≈ (N − K) · (C_cold − C_warm)
```

where `N` = #instances, `K` = #groups. So you want **many non-seed members**
(few, large groups), **large `C_cold`** (expensive cold search), and **small
`C_warm`** (the cache actually decides the member cheaply).

---

## 2. Learnings (each backed by a run in this session)

### L1 — The cache is per-group; only *intra*-group structure matters.
Different groups never share a cache. Making distinct groups resemble each other
buys nothing. Spend all "similarity budget" **inside** groups.

### L2 — Singletons are pure waste; prefer few, large groups.
A singleton group is its own seed and reuses nothing. i.i.d. adversarial
sampling (the stock `build_distribution.py --mode adversarial`) TV-scatters into
mostly singletons — a 200-prompt sample was **~58% singletons**, so caching had
almost nothing to reuse. Build **families**, don't sample independently.

### L3 — TV-tightness is *necessary*: within-group TV must be < ε.
The `cachedfrontier` grouping only helps when members drive near-identical
next-token distributions (within-group TV ≤ ε), so the diameter-capped TV
clustering keeps them together instead of splitting them.
- **Distinct behavioral themes → clean between-group separation.** Distinct
  target emails give between-group TV ≈ **1.0**; TV never merges different
  targets.
- **Mild within-theme variation keeps groups tight.** Shuffling the few-shot
  order (framing fixed) keeps within-group TV small.

### L4 — Structural grouping (shared target) is *not* the same as TV-tight.
Grouping only by "same target email" guarantees a shared **violation sequence**
but **not** TV-tightness, because the *jailbreak strength* reshapes the
next-token distribution independently of the target. A shared-target group with
varied jailbreaks (`crossover_prob=0.3`) had within-group TV mean **0.13** (only
65% of pairs ≤ ε) and the cached run came out **−3.5% (caching LOST)**. You need
**both** a shared violation **and** TV-tightness. Vary framing *mildly*, or not
at all.

### L5 — A single global cache is actively harmful on diverse data.
Putting all instances in one group (one global cache) over TV-dissimilar prompts
(between-group TV ≈ 1.0):
- Warm-starts re-score a huge, polluted antichain drawn from unrelated prompts.
- Causes **false early terminations** with wide, uninformative bounds
  (`UB−LB` blew up from ~0.12 to ~0.71; safe determinations collapsed).
- Costs **more** than no cache at all — **2.1×** worse in the 8×12 test
  (9134 vs 4359 FP).
This is the empirical proof that **per-TV-cluster grouping — not one big cache —
is the correct structure.**

### L6 — The biggest savings need instances that actually VIOLATE.
Warm-skips that save the most fire on a cached **violation** (`v_prob ≥ 1−ε`).
- **Safe** instances get little: proving safety needs the full incomplete
  frontier, and warm-start re-scores it at ≈ cold cost. TV-*identical* duplicate
  prompts that were all safe saved only **+2.2%**, and only **1** instance
  warm-skipped in the whole run.
- Our Qwen runs were **0% risky** (the model refuses), so we only ever measured
  the **safe floor** (+13–15%). On a distribution that reliably leaks a shared
  target, warm members would gate-trip on the cached violation and the gap would
  be much larger.

### L7 — Make the cold search expensive so there's something to amortize.
`savings ∝ C_cold − C_warm`. If prompts are decided in ~1 step, caching saves
almost nothing. Favor **borderline** prompts (violation mass near ε) and a
**longer `gen_length`** so the cold seed search is deep.

### L8 — Savings grow with group size.
Same design, bigger groups: 8×12 (max group 12) → **+13.0%**; 15×24 (max group
24) → **+15.1%**. More populated groups ⇒ more warm-skips ⇒ larger gap.

---

## 3. The optimal recipe

1. **Distinct behavioral themes** (e.g. one target email per theme) →
   between-group TV ≈ 1, clean separation, no cross-group merges. *(L1, L3)*
2. **Large groups** of many members; minimize the number of groups for a given
   `N`; drive singletons toward zero. *(L2, L8)*
3. **Members share the violation sequence** (same target) → one seed's cached
   violation covers the whole group. *(L4, L6)*
4. **Vary members only mildly** (shuffle few-shot order; framing fixed or a
   *single* shared jailbreak) so within-group TV ≤ ε. Never give members
   *different* jailbreak strengths. *(L3, L4)*
5. For the biggest win, **make members reliably violate** (strong-but-shared
   framing) and **borderline/deep** (leak mass ≈ ε, long `gen_length`). *(L6, L7)*
6. **Group with the TV tools, not by hand:** run
   [`tv_distance`](../../beaver/grouping/tv_distance.py) then
   [`tv_visualize`](../../beaver/grouping/tv_visualize.py) at threshold ε
   (complete-linkage diameter cap) to get the actual `group` labels and to
   *verify* few singletons. Structural theme labels are a good approximation
   only when variation is mild. *(L3, L5)*
7. **Never use one global cache** across dissimilar themes. *(L5)*

## 4. Measured results (this session)

| distribution | design | singletons | cached vs uncached |
|---|---|---|---|
| i.i.d. adversarial (sample 200) | independent sampling | ~58% | caching ~useless |
| structural 5×10, `crossover 0.3` | shared target, varied jailbreak (TV-loose) | — | **−3.5%** (LOST) |
| TV-identical 5×10 (duplicates) | tight but all safe | 0 | **+2.2%** |
| **tv_demo 8×12** | distinct target + shot-shuffle | 2 / 14 clusters | **+13.0%** |
| **tv_demo 15×24** | distinct target + shot-shuffle | 16 (4.4% of prompts) | **+15.1%** |
| one global cache (8×12) | all in one group | — | **2.1× worse** than no cache |

All Qwen runs were 0% risky ⇒ these are the **safe-instance floor**; a leaking
configuration should do substantially better (L6).
