"""Plotting helpers, shared by both direction scripts.

plot_metrics  — in-sample per-layer metrics; optional embedding lexical floor.
plot_heldout  — grouped held-out AUC (mean +/- std band); optional bag-of-words.
Both fall back to plt.show() when save_path is None.
"""

import numpy as np
import matplotlib.pyplot as plt


def _save_or_show(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved plot -> {save_path}")
    else:
        plt.show()


def plot_metrics(rows_by_pos, emb_floor=None, save_path=None):
    """In-sample per-layer metrics, one line per read position. When emb_floor
    (a {pos: float} dict) is given, the dotted horizontal lines mark each
    position's embedding lexical floor."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for pos, rows in rows_by_pos.items():
        layers = [r["layer"] for r in rows]
        ax[0].plot(layers, [r["r_norm"] for r in rows], "o-", label=pos)
        ax[1].plot(layers, [r["fisher"] for r in rows], "o-", label=pos)
        (line,) = ax[2].plot(layers, [r["proj_auc"] for r in rows], "o-", label=pos)
        if emb_floor is not None:
            ax[2].axhline(emb_floor[pos], ls=":", color=line.get_color(), alpha=0.7)
    auc_title = "proj AUC + lexical floor" if emb_floor is not None else "diff-of-means proj AUC"
    for a, t in zip(ax, ("|r| (not quality)", "Fisher ratio (1-D)", auc_title)):
        a.set_title(t)
        a.set_xlabel("layer")
        a.grid(alpha=0.3)
        a.legend()
    ax[2].axhline(0.5, color="k", ls="--", alpha=0.5)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_heldout(cv_by_pos, insample_by_pos, bow_auc=None, save_path=None):
    """One panel per position: grouped held-out AUC (mean +/- std band) for the
    diff-of-means and probe detectors, the in-sample proj AUC (optimistic
    reference), an optional bag-of-words baseline, and chance.

    cv_by_pos[pos][det] = {'mean': [L], 'std': [L]}."""
    positions = list(cv_by_pos)
    fig, ax = plt.subplots(
        1, len(positions), figsize=(5 * len(positions), 4), squeeze=False
    )
    for a, pos in zip(ax[0], positions):
        L = len(cv_by_pos[pos]["probe"]["mean"])
        xs = np.arange(L)
        for det, style in (("diffmeans", "o-"), ("probe", "s-")):
            m = cv_by_pos[pos][det]["mean"]
            s = cv_by_pos[pos][det]["std"]
            (line,) = a.plot(xs, m, style, label=f"{det} (CV)")
            a.fill_between(xs, m - s, m + s, alpha=0.15, color=line.get_color())
        a.plot(
            xs,
            [r["proj_auc"] for r in insample_by_pos[pos]],
            ":",
            color="gray",
            label="proj AUC (in-sample)",
        )
        if bow_auc is not None:
            a.axhline(bow_auc, color="purple", ls="-.", alpha=0.7, label="bag-of-words")
        a.axhline(0.5, color="k", ls="--", alpha=0.5)
        a.set_title(pos)
        a.set_xlabel("layer")
        a.set_ylabel("AUC")
        a.set_ylim(0.4, 1.02)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    _save_or_show(fig, save_path)
