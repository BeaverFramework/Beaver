"""Response parsing shared by every model backend.

These helpers all operate on the *choice* shape produced by
``http_backend._wrap_choice`` (and mirrored by ``local_vllm``): an
attribute-accessible object with ``.text``, ``.hidden_states``, and
``.logprobs.{tokens, token_logprobs, top_logprobs}``, where each
``top_logprobs`` entry is a plain ``{"token_id:N": float}`` dict.

Keeping both backends on that one shape is what lets every ``ops.py``
function stay backend-agnostic.

Moved verbatim from beaver/verifiers/worker_common.py.
"""

import base64

import numpy as np
import torch


def parse_logprobs(response_choice):
    """Extract logprobs array from an OpenAI-compatible response choice.

    Returns:
        list of np.ndarray, each shape [N, 2] with [token_id, logprob] pairs
        (one array per position present in ``top_logprobs``).
    """
    logprobs_obj = response_choice.logprobs
    if (
        logprobs_obj is None
        or not hasattr(logprobs_obj, "top_logprobs")
        or logprobs_obj.top_logprobs is None
    ):
        raise ValueError(
            "No logprobs returned from server. "
            "Make sure server was started with --max-logprobs -1"
        )
    all_logprobs = []
    for top_logprobs_dict in logprobs_obj.top_logprobs:
        logprobs = []
        for key, logprob in top_logprobs_dict.items():
            if key.startswith("token_id:"):
                token_id = int(key.split(":")[1])
                logprobs.append([token_id, logprob])

        all_logprobs.append(np.array(logprobs))

    if not all_logprobs:
        raise ValueError(
            "No token_id logprobs found in response. "
            "Check if server was started with --max-logprobs flag."
        )
    return all_logprobs


def top_logprobs_to_array(top_logprobs_dict):
    """Parse one position's ``top_logprobs`` dict into an ``[N, 2]`` array of
    ``[token_id, logprob]`` rows. Returns an empty ``(0, 2)`` array when the
    position carries no distribution (e.g. the very first prompt token, which
    echo reports as ``None``)."""
    if not top_logprobs_dict:
        return np.empty((0, 2))
    rows = [
        [int(key.split(":")[1]), logprob]
        for key, logprob in top_logprobs_dict.items()
        if key.startswith("token_id:")
    ]
    return np.array(rows) if rows else np.empty((0, 2))


def parse_hidden_states(hidden_states_raw):
    """Decode base64-encoded hidden states from server response.

    Args:
        hidden_states_raw: List of dicts with keys ``data`` (base64),
            ``shape``, ``dtype``, ``layer``.  Returned by beaver.hf_server.

    Returns:
        dict mapping ``{layer_idx: torch.Tensor}``.  Each tensor has the
        shape recorded by the server (typically ``[hidden_dim]``).
    """
    result = {}
    for hs in hidden_states_raw:
        data_bytes = base64.b64decode(hs["data"])
        tensor = torch.from_numpy(
            np.frombuffer(data_bytes, dtype=np.float32).copy()
        ).reshape(hs["shape"])
        result[hs["layer"]] = tensor
    return result


def parse_sample_choice(choice, tokenizer):
    """Extract (token_ids, token_logprobs) from a single completion choice.

    ``tokenizer`` is only consulted for the legacy path where a server returns
    literal token *text* rather than ``token_id:N`` markers.
    """
    logprobs_obj = choice.logprobs
    if logprobs_obj is None or not logprobs_obj.token_logprobs:
        return [], []

    token_ids = []
    for token_str in logprobs_obj.tokens:
        if token_str.startswith("token_id:"):
            token_ids.append(int(token_str.split(":")[1]))
        else:
            enc_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if enc_ids:
                token_ids.append(enc_ids[0])

    return token_ids, logprobs_obj.token_logprobs
