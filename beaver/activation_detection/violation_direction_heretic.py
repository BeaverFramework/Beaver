# %%
"""Extract a "violation-generation direction" from BEAVER logs, HERETIC-NATIVE.

Same goal as violation_direction_detection.py, but residuals come from
`heretic.model.Model.get_residuals_batched`, which runs one forward pass per
batch, generates a single token, and returns the residual stream at the LAST
token of each input, stacked over every layer (shape [N, L, d], L = n_layers+1
because layer 0 is the embedding). Winsorization / float32 upcast / CPU offload
are inherited from Heretic.

Heretic's `Model.generate` always re-templates a Prompt(system, user); BEAVER
already stores the fully chat-templated string (`exact_prompt`), so re-templating
would double-wrap it. We subclass Model and feed the pre-templated text verbatim:

  READ_POSITIONS
    "risk_prior" -> residual at the last token of `exact_prompt` (a PROMPT-LEVEL
                    prior: identical for every generation of a given prompt, so
                    its effective N is the number of prompts, not rows).
    "gen_last"   -> residual at the last token of `exact_prompt + generation`
                    (generation truncated to MAX_GEN_TOKENS) -> per-row detector.

Backend-agnostic pieces (build_labels, analyze_layers, probe/CV testing, plots)
are imported from the activation_detection package and shared with the
TransformerLens script. Run with
`python -m activation_detection.violation_direction_heretic` or
`python activation_detection/violation_direction_heretic.py` (defaults), or pass
--log-file / --dataset-file. Or import `main()` / the individual functions.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

from heretic.config import Settings
from heretic.model import Model

from beaver.utils import new_log_dir, log_json
from beaver.activation_detection.data import build_labels
from beaver.activation_detection.analysis import analyze_layers
from beaver.activation_detection.plots import plot_metrics, plot_heldout
from beaver.activation_detection.probes import grouped_cv_auc

# %% ----------------------------------------------------------------- config
# LOG_FILE = Path("logging/logs_20260531222846")
# DATASET_FILE = Path("experiments/toxicity/data/distribution_uniform100.json")

LOG_FILE = Path("logging/qwen3-4b-secure_coder_samples")
DATASET_FILE = Path("experiments/secure_code/data/autocomplete_ccpp_uniform_100.json")

# Heretic-native read positions (see module docstring).
READ_POSITIONS = ("risk_prior", "gen_last")
MAX_GEN_TOKENS = 64          # truncate the generation before reading gen_last
BATCH_SIZE = 16              # prompts per forward pass (Heretic's batch_size)
WINSORIZATION_QUANTILE = 1.0  # 1.0 == off; <1 clamps per-layer residual tails
ORTHOGONALIZE_DIRECTION = False  # projected-abliteration orthogonalization

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path(".residual_cache")
OUT_ROOT = Path("logging/direction_detection")


# %% --------------------------------------------------------- the heretic model
@dataclass
class RawPrompt:
    """Carries a pre-templated string instead of Heretic's Prompt(system, user).
    BeaverModel.generate reads `.text` directly, so no re-templating happens."""

    text: str


class BeaverModel(Model):
    """Heretic Model that reads residuals over verbatim pre-templated text.

    All residual machinery (`get_residuals`, `get_residuals_batched`, last-token
    extraction, winsorization, float32 upcast, CPU offload) is inherited. We only
    override `generate` to tokenize `RawPrompt.text` directly. Heretic uses
    left-padding, so the last-token slice `[:, -1, :]` in `get_residuals` is the
    true final token of every input regardless of batch padding.
    """

    def generate(self, prompts, **kwargs):  # type: ignore[override]
        # Heretic's __init__ self-tests with real Prompt(system, user) objects;
        # only RawPrompt carries pre-templated `.text`. Delegate the former to
        # Heretic's normal chat-templating path.
        if not all(isinstance(p, RawPrompt) for p in prompts):
            return super().generate(prompts, **kwargs)
        texts = [p.text for p in prompts]
        if self.settings.response_prefix:
            texts = [t + self.settings.response_prefix for t in texts]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            add_special_tokens=False,  # exact_prompt already carries template/BOS
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
        )
        return inputs, outputs


def load_model(log_file: Path) -> BeaverModel:
    with open(log_file / "run_args.json") as f:
        run_args = json.load(f)
    # Heretic's Settings hardcodes cli_parse_args=True, so constructing it reads
    # sys.argv -- which under this script's own --log-file/--dataset-file/--out-root
    # flags makes heretic's CLI parser abort on "unrecognized arguments". Neutralize
    # argv for just this construction (kwargs still take effect; they are the
    # highest-priority settings source). Same workaround as probe_ranker.py's
    # _build_model, for the identical heretic quirk.
    import sys

    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        settings = Settings(
            model=run_args["model"],
            batch_size=BATCH_SIZE,
            max_response_length=1,          # we only need one generated token
            winsorization_quantile=WINSORIZATION_QUANTILE,
            offload_outputs_to_cpu=True,    # keep VRAM flat: residuals -> CPU per batch
        )
    finally:
        sys.argv = saved_argv
    return BeaverModel(settings)


def _raw_text(model: BeaverModel, item: dict, read_pos: str) -> str:
    """Build the pre-templated string whose last token we read."""
    if read_pos == "risk_prior":
        return item["prompt"]
    if read_pos == "gen_last":
        gen_ids = model.tokenizer(item["generation"], add_special_tokens=False)[
            "input_ids"
        ][:MAX_GEN_TOKENS]
        return item["prompt"] + model.tokenizer.decode(gen_ids)
    raise ValueError(read_pos)


# %% --------------------------------------------------------- read residuals
def _cache_path(log_file, read_positions, n_labels):
    pos = "-".join(read_positions)
    stem = (
        f"{Path(log_file).name}__{pos}__g{MAX_GEN_TOKENS}"
        f"__w{WINSORIZATION_QUANTILE}__n{n_labels}__heretic_v1"
    )
    return CACHE_DIR / f"{stem}.npz"


def extract_residuals(
    model, labels, read_positions=READ_POSITIONS, log_file=LOG_FILE, use_cache=True
):
    """Heretic-native residuals via `get_residuals_batched`, one call per read
    position. Returns (acts {pos: [N, L, d]}, y, cwes, groups). Cached as .npz."""
    path = _cache_path(log_file, read_positions, len(labels))
    if use_cache and path.exists():
        print(f"loading cached residuals <- {path}")
        z = np.load(path, allow_pickle=True)
        acts = {p: z[f"acts__{p}"] for p in read_positions}
        return acts, z["y"], z["groups"]

    acts = {}
    for p in read_positions:
        prompts = [
            RawPrompt(_raw_text(model, it, p)) for it in tqdm(labels, desc=f"text:{p}")
        ]
        # Heretic's own batched residual extraction: [N, L, d], L = n_layers+1.
        res = model.get_residuals_batched(prompts)  # last-token residuals, all layers
        acts[p] = res.float().cpu().numpy()
        print(f"  {p}: residuals {acts[p].shape}")

    y = np.array([0 if it["is_secure"] else 1 for it in labels])
    groups = np.array([it["group"] for it in labels])

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        save = {f"acts__{p}": v for p, v in acts.items()}
        np.savez_compressed(path, y=y, groups=groups, **save)
        print(f"saved residuals -> {path}")
    return acts, y, groups


# %% ===================================================================== main
def main(
    log_file=LOG_FILE,
    dataset_file=DATASET_FILE,
    out_root=OUT_ROOT,
    read_positions=READ_POSITIONS,
    orthogonalize=ORTHOGONALIZE_DIRECTION,
    calibrate=False,  # rank-only by default; pass calibrate=True / --calibrate to add P(violation)
    calibrator_method="logistic",
):
    """Heretic-native pipeline. Returns the run directory + per-position CV
    summary. Each step is also callable on its own (build_labels, load_model,
    extract_residuals, analyze_layers, grouped_cv_auc, ...)."""
    log_file, dataset_file = Path(log_file), Path(dataset_file)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = new_log_dir(out_root)
    print(f"run dir -> {out_dir}")

    labels = build_labels(log_file, dataset_file)
    n_ins = sum(not l["is_secure"] for l in labels)
    print(f"{len(labels)} samples: {n_ins} insecure / {len(labels) - n_ins} secure")
    log_json(
        dict(
            log_file=str(log_file),
            dataset_file=str(dataset_file),
            read_positions=list(read_positions),
            max_gen_tokens=MAX_GEN_TOKENS,
            winsorization_quantile=WINSORIZATION_QUANTILE,
            orthogonalize_direction=orthogonalize,
            n_samples=len(labels),
            n_insecure=n_ins,
        ),
        out_dir / "config.json",
    )

    model = load_model(log_file)
    acts, y, groups = extract_residuals(
        model, labels, read_positions=read_positions, log_file=log_file
    )
    for p, v in acts.items():
        print("acts", p, v.shape)
    print("groups:", len(np.unique(groups)), "prompts")

    # per-layer diff-of-means directions (Heretic-native) + in-sample metrics
    dirs_by_pos, rows_by_pos = {}, {}
    for p in read_positions:
        print(f"\n=== {p} ===")
        dirs_by_pos[p], rows_by_pos[p] = analyze_layers(
            acts[p], y, orthogonalize=orthogonalize
        )

    log_json(rows_by_pos, out_dir / "metrics.json")
    np.savez_compressed(out_dir / "directions.npz", **dirs_by_pos)
    plot_metrics(rows_by_pos, save_path=out_dir / "metrics.png")

    # grouped held-out CV for every position
    cv_by_pos, cv_summary = {}, {}
    for p in read_positions:
        print(f"\n=== grouped CV: {p} ===")
        cv_by_pos[p] = grouped_cv_auc(acts[p], y, groups, n_splits=5, seed=0)
        for det in ("diffmeans", "probe"):
            m = cv_by_pos[p][det]["mean"]
            bl = int(np.nanargmax(m))
            cv_summary.setdefault(p, {})[det] = dict(
                best_layer=bl,
                best_auc=float(m[bl]),
                best_std=float(cv_by_pos[p][det]["std"][bl]),
            )
            print(
                f"  {det:9s} best layer {bl:2d}  "
                f"heldout_auc={m[bl]:.3f} +/- {cv_by_pos[p][det]['std'][bl]:.3f}"
            )

    log_json(cv_summary, out_dir / "heldout_cv_summary.json")
    plot_heldout(cv_by_pos, rows_by_pos, save_path=out_dir / "heldout_cv.png")

    # ── ranker artifacts: pick the risk_prior diff-of-means layer (what the
    # probe ranker reads -- a prompt-level direction, no generation needed) and
    # calibrate its projection into P(violation) from the data already in hand.
    decision = {"read_positions": list(read_positions)}
    if "risk_prior" in cv_summary:
        rp_layer = cv_summary["risk_prior"]["diffmeans"]["best_layer"]
        rp_dir = dirs_by_pos["risk_prior"][rp_layer]  # [d] unit vector
        decision["risk_prior"] = dict(
            read_pos="risk_prior",
            detector_type="diffmeans",
            layer=int(rp_layer),
            heldout_auc=cv_summary["risk_prior"]["diffmeans"]["best_auc"],
            n_layers_convention="hidden_states (L = n_layers+1; layer 0 = embedding)",
        )

        if calibrate:
            from beaver.activation_detection.calibration import (
                fit_calibrator,
                prompt_fractions,
            )

            # risk_prior residual is constant within a prompt, so per-row proj is
            # too -> take one proj per prompt and pair with its violation fraction.
            proj_row = acts["risk_prior"][:, rp_layer, :] @ rp_dir  # [N]
            gids, frac, n = prompt_fractions(y, groups)
            g2p = {int(g): float(proj_row[groups == g][0]) for g in gids}
            proj_prompt = np.array([g2p[int(g)] for g in gids])

            cal = fit_calibrator(
                proj_prompt, frac, n,
                method=calibrator_method, read_pos="risk_prior", layer=int(rp_layer),
            )
            joblib.dump(cal, out_dir / "calibrator.joblib")
            decision["calibrator"] = dict(file="calibrator.joblib", **cal.meta)
            print(
                f"\nCALIBRATOR ({calibrator_method}) risk_prior L{rp_layer}  "
                f"brier={cal.meta['brier']:.4f}  loo_brier={cal.meta['loo_brier']:.4f}  "
                f"(mean-baseline {cal.meta['baseline_brier']:.4f})"
            )
    else:
        print("\n[warn] 'risk_prior' not in read_positions; no ranker direction/calibrator exported")

    log_json(decision, out_dir / "detector_decision.json")

    print(f"\nall artifacts -> {out_dir}")
    return dict(out_dir=out_dir, cv_summary=cv_summary, decision=decision)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", default=str(LOG_FILE), help="BEAVER log dir")
    parser.add_argument("--dataset-file", default=str(DATASET_FILE), help="dataset json")
    parser.add_argument("--out-root", default=str(OUT_ROOT), help="run-dir root")
    parser.add_argument(
        "--read-positions",
        nargs="+",
        default=list(READ_POSITIONS),
        help="residual read positions",
    )
    parser.add_argument(
        "--orthogonalize", action="store_true", help="projected-abliteration"
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="also fit + save the probe calibrator (rank-only without this)",
    )
    parser.add_argument(
        "--calibrator-method", default="logistic", choices=["logistic", "isotonic"]
    )
    args = parser.parse_args()
    main(
        log_file=args.log_file,
        dataset_file=args.dataset_file,
        out_root=args.out_root,
        read_positions=tuple(args.read_positions),
        orthogonalize=args.orthogonalize,
        calibrate=args.calibrate,
        calibrator_method=args.calibrator_method,
    )
