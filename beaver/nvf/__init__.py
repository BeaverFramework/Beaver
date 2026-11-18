"""Violation probability probe — lightweight models on transformer hidden states.

This module is imported by both the training pipeline and the beaver
representation verifier at runtime, so it must have minimal dependencies
(just torch).

Three estimator types:
  ViolationProbe  — linear (proj_dim=0) or shallow MLP (proj_dim>0)
  NonlinearProbe  — bottleneck MLP with dropout, for when linear isn't enough
  KNNEstimator    — non-parametric KNN on training hidden states
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ViolationProbe(nn.Module):
    """Probe: hidden_state → predicted violation probability.

    proj_dim > 0:  two-layer MLP (hidden_dim → proj_dim → 1)
    proj_dim = 0:  linear probe  (hidden_dim → 1)

    forward() returns probabilities in [0, 1].
    logits() returns raw pre-sigmoid values (for BCEWithLogitsLoss).
    """

    def __init__(self, hidden_dim: int = 4096, proj_dim: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        if proj_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, 1),
            )
        else:
            self.net = nn.Linear(hidden_dim, 1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: [hidden_dim] or [batch, hidden_dim] → raw scalar(s)."""
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [hidden_dim] or [batch, hidden_dim] → probability in [0,1]."""
        return torch.sigmoid(self.logits(x))

    def save(self, path: str):
        torch.save(
            {
                "hidden_dim": self.hidden_dim,
                "proj_dim": self.proj_dim,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ViolationProbe":
        data = torch.load(path, map_location=device, weights_only=True)
        probe = cls(hidden_dim=data["hidden_dim"], proj_dim=data["proj_dim"])
        probe.load_state_dict(data["state_dict"])
        probe.to(device)
        probe.eval()
        return probe


class NonlinearProbe(nn.Module):
    """Bottleneck MLP with dropout for violation probability prediction.

    Architecture: hidden_dim → bottleneck → bottleneck → 1
    Dropout after each hidden layer to regularize on small datasets.

    Use this when linear probe shows some signal but underfits —
    it can capture nonlinear interactions between features while
    staying small enough to not overfit on ~1000 samples.
    """

    def __init__(
        self, hidden_dim: int = 4096, bottleneck: int = 128, dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck = bottleneck
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(bottleneck, bottleneck),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 1),
        )

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x))

    def save(self, path: str):
        torch.save(
            {
                "hidden_dim": self.hidden_dim,
                "bottleneck": self.bottleneck,
                "dropout": self.dropout,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NonlinearProbe":
        data = torch.load(path, map_location=device, weights_only=True)
        probe = cls(
            hidden_dim=data["hidden_dim"],
            bottleneck=data["bottleneck"],
            dropout=data["dropout"],
        )
        probe.load_state_dict(data["state_dict"])
        probe.to(device)
        probe.eval()
        return probe


class KNNEstimator:
    """KNN-based violation rate estimator using training hidden states.

    Loads training data from a .pt file (output of extract_data.py),
    predicts violation rate for new hidden states via K nearest neighbors.

    No trainable parameters — directly uses the geometry of the hidden
    state space to estimate behavioral risk.
    """

    def __init__(self, X_train, y_train, k=5, normalize=False):
        self.k = k
        self.normalize = normalize
        self.X_train = X_train.float()
        self.y_train = y_train.float()
        if normalize:
            self.X_train = self.X_train / self.X_train.norm(dim=1, keepdim=True)
        # Precompute squared norms for fast L2 distance
        self._train_sq = (self.X_train**2).sum(dim=1)  # [n_train]

    def predict(self, hidden_state, return_info=False):
        """Predict violation rate for a single hidden state.

        Args:
            hidden_state: torch.Tensor of shape [dim] or [1, dim]

        Returns:
            float: predicted violation rate (mean of K nearest neighbor VRs)
        """
        x = hidden_state.float().to(self.X_train.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self.normalize:
            x = x / x.norm(dim=1, keepdim=True)

        # L2 distance: ||x - t||^2 = ||x||^2 + ||t||^2 - 2*x·t
        x_sq = (x**2).sum(dim=1)  # [1]
        cross = x @ self.X_train.T  # [1, n_train]
        dists = (
            x_sq.unsqueeze(1) + self._train_sq.unsqueeze(0) - 2 * cross
        )  # [1, n_train]

        _, nn_idx = dists.topk(self.k, dim=1, largest=False)  # [1, k]
        neighbours = None
        if return_info:
            neighbours = (self.y_train[nn_idx[0]], dists[0, nn_idx[0]])
        return self.y_train[nn_idx[0]].mean().item(), neighbours

    def to(self, device):
        """Move training data to device."""
        self.X_train = self.X_train.to(device)
        self.y_train = self.y_train.to(device)
        self._train_sq = self._train_sq.to(device)
        return self

    @classmethod
    def load(cls, pt_path, k=3, normalize=True, layer_idx=None, device="cpu"):
        """Load from a .pt file produced by extract_data.py.

        Args:
            pt_path: Path to .pt file with hidden_states, violation_rates, metadata.
            k: Number of neighbors.
            normalize: L2-normalize hidden states (cosine distance).
            layer_idx: For multi-layer data [n, n_layers, dim]:
                       int = use that layer index, None = concatenate all layers.

        Returns:
            KNNEstimator with model_layers attribute (list of HF model layer
            indices needed at runtime for hidden state extraction).
        """
        data = torch.load(pt_path, map_location=device, weights_only=False)
        hs = data["hidden_states"]
        vr = data["violation_ratios"]
        meta = data["metadata"]

        if hs.dim() == 3:
            # Multi-layer: [n_samples, n_layers, hidden_dim]
            if layer_idx is not None:
                X = hs[:, layer_idx, :]
                model_layers = [meta["layer_indices"][layer_idx]]
            else:
                X = hs.reshape(hs.shape[0], -1)  # concat all
                model_layers = meta["layer_indices"]
        else:
            # Single layer: [n_samples, hidden_dim]
            X = hs
            layer = meta.get("layer")
            indices = meta.get("layer_indices")
            if layer is not None:
                model_layers = [layer]
            elif indices is not None:
                model_layers = indices
            else:
                model_layers = [None]

        estimator = cls(X, vr, k=k, normalize=normalize)
        # estimator.model_layers = model_layers
        # estimator.meta = meta
        # print(
        #     f"KNNEstimator: {len(vr)} training points, k={k}, "
        #     f"dim={X.shape[1]}, model_layers={model_layers}"
        # )
        return estimator


# ---------------------------------------------------------------------------
# DirectionNVF — arXiv:2607.20286 (Nazeri et al.) contrastive-direction scoring
# ---------------------------------------------------------------------------


def load_direction_stack(probe_dir, key="risk_prior"):
    """Load and L2-normalize the full per-layer direction stack from a
    heretic-produced `directions.npz` (see violation_direction_heretic.py).

    Args:
        probe_dir: directory containing directions.npz.
        key: which named direction to load (matches detector_decision.json's
            top-level keys, e.g. "risk_prior").

    Returns:
        np.ndarray [L, d] of unit-norm direction vectors, one row per
        heretic-space layer index (row 0 = the embedding layer, per
        heretic's `get_residuals_batched` convention: L = n_layers + 1).
    """
    probe_dir = Path(probe_dir)
    directions_path = probe_dir / "directions.npz"
    if not directions_path.exists():
        raise FileNotFoundError(
            f"probe_dir {probe_dir} must contain directions.npz "
            f"(produced by violation_direction_heretic.py)"
        )
    z = np.load(directions_path, allow_pickle=True)
    if key not in z:
        # Callers disagree on the default key ("risk_prior" here vs
        # "gen_last" in DirectionNVF.load), and a given artifact usually
        # holds exactly one. Fall back to it rather than failing on a
        # default nobody chose.
        if len(z.files) == 1:
            key = z.files[0]
        else:
            raise KeyError(
                f"directions.npz in {probe_dir} has no {key!r} key; "
                f"available: {z.files}"
            )
    stack = np.asarray(z[key], dtype=np.float32)  # [L, d]
    norms = np.linalg.norm(stack, axis=1, keepdims=True)
    return stack / (norms + 1e-8)


class DirectionNVF:
    """Fitness scorer from arXiv:2607.20286 (Nazeri et al.): cosine similarity
    to a per-layer mean-difference ("harmful direction") vector, summed across
    layers, plus a length-decayed offset — Eq. 3 of the paper, dropping its
    explicit "* P(y|x)" factor since Frontier's built-in "violation-aware" pick
    strategy already multiplies by exp(child.logprob).

    Loaded from heretic-produced direction artifacts (directions.npz). Heretic's
    hidden-state stack is [n_layers+1, dim] with index 0 = the embedding layer;
    hf_server's forward-hook hidden states index transformer blocks directly (no
    embedding entry), so heretic index i (i>=1) maps to hf_server hook layer
    i-1. `hf_server_layers` exposes that mapped, 0-based hf_server layer list;
    `predict()` expects a dict keyed by those same hf_server layer indices.
    """

    def __init__(self, directions, b=1e-2, layers=None, layer_weights=None,
                 strict=True):
        """directions: torch.Tensor [L, d], heretic layer indexing (row 0 =
        embedding, dropped here since hf_server has no hook for it).

        layers: which hf_server layer indices to actually score. None = all.
            Restricting to one well-chosen layer both sharpens the signal
            (correlated layers add little) and shrinks the hidden-state
            payload by the same factor.
        layer_weights: per-layer multipliers, aligned with ``layers``. None =
            uniform. Normalized to sum to 1 so the score's scale does not
            depend on how many layers are in play -- otherwise a 28-layer sum
            and a 1-layer sum are not comparable and any fixed threshold
            calibrated on one is meaningless on the other.
        strict: raise if a requested layer is absent from the hidden states
            handed to predict(). The server's hook layers are fixed at launch
            and are NOT negotiated with the client, so a mismatch otherwise
            silently sums fewer terms and shifts every score.
        """
        self.directions = directions[1:]  # [L-1, d], heretic indices 1..L-1
        n = self.directions.shape[0]
        self.hf_server_layers = list(range(n)) if layers is None else list(layers)
        bad = [l for l in self.hf_server_layers if not 0 <= l < n]
        if bad:
            raise ValueError(
                f"direction layers {bad} out of range for a {n}-layer stack"
            )
        if layer_weights is None:
            w = np.ones(len(self.hf_server_layers), dtype=np.float32)
        else:
            w = np.asarray(layer_weights, dtype=np.float32)
            if w.shape[0] != len(self.hf_server_layers):
                raise ValueError(
                    f"layer_weights has {w.shape[0]} entries but "
                    f"{len(self.hf_server_layers)} layers were requested"
                )
        total = float(np.abs(w).sum())
        self.layer_weights = (w / total) if total > 0 else w
        # Scale the length-decay term by the SAME factor as the cosine term.
        # Normalising the weights alone would shrink the cosine sum ~n_layers-
        # fold while leaving b/token_count untouched, turning a term worth
        # ~1% of a score std into one worth ~34% at shallow depth -- i.e. a
        # depth bias nobody asked for. Scaling both keeps the score an exact
        # rescale of the unnormalised one, so every ranking is preserved.
        self.b = (b / total) if total > 0 else b
        self.strict = strict

    @classmethod
    def load(cls, probe_dir, key="gen_last", b=1e-2, device="cpu", layers=None,
             layer_weights=None, strict=True):
        stack = load_direction_stack(probe_dir, key=key)  # np.ndarray [L, d]
        directions = torch.from_numpy(stack).float().to(device)
        return cls(directions, b=b, layers=layers, layer_weights=layer_weights,
                   strict=strict)

    def predict(self, hidden_states_by_layer, token_count, return_info=False):
        """hidden_states_by_layer: {hf_server_layer_idx: torch.Tensor}.

        Returns (score, None) — score = b/token_count + sum of per-layer
        cosine similarities to the direction vectors present in
        hidden_states_by_layer (layers missing from the dict are skipped,
        not treated as zero).
        """
        if not hidden_states_by_layer:
            return 0.0, None
        missing = [l for l in self.hf_server_layers if l not in hidden_states_by_layer]
        if missing and self.strict:
            raise ValueError(
                f"DirectionNVF needs layers {self.hf_server_layers} but the "
                f"backend returned {sorted(hidden_states_by_layer)}; missing "
                f"{missing}. Launch hf_server with --hidden-state-layers "
                f"covering them, or pass hf_local_hidden_layers."
            )
        cos_sum = 0.0
        for hf_layer, weight in zip(self.hf_server_layers, self.layer_weights):
            h = hidden_states_by_layer.get(hf_layer)
            if h is None:
                continue
            direction = self.directions[hf_layer]
            h = h.float().to(direction.device).flatten()
            cos_sum += float(weight) * torch.nn.functional.cosine_similarity(
                h.unsqueeze(0), direction.unsqueeze(0)
            ).item()
        score = (self.b / max(token_count, 1)) + cos_sum
        return score, None
