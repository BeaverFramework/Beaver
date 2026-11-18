"""In-process hidden-state capture for the ``vllm-local`` backend.

Why this exists
---------------
vLLM's ``RequestOutput``/``CompletionOutput`` carry no hidden states, which is
why ``HIDDEN_STATE_BACKENDS`` historically excluded both vLLM backends and the
representation verifier could only run on ``hf``/``hf-local``.

vLLM 0.21 does expose them, just not through the output objects: the
``extract_hidden_states`` speculative-decoding method stores per-token states
for chosen layers into a dedicated ``CacheOnlyAttentionLayer``, and a KV
connector is invoked to ship them somewhere. vLLM's own
``ExampleHiddenStatesConnector`` ships them to disk as safetensors -- fine as a
demo, useless per-request in a search loop.

This connector is the same mechanism with the disk write replaced by a dict in
this module. That only works because the states are produced inside the
EngineCore process: with ``VLLM_ENABLE_V1_MULTIPROCESSING=1`` (vLLM's default)
EngineCore is a *separate* process and this dict would be written in one
process and read in another. ``ensure_in_process()`` therefore pins
EngineCore in-process, and must be called before the engine is constructed.

This is also exactly why the ``vllm`` (HTTP server) backend cannot use this:
its EngineCore is always a separate process from the API server, and the
OpenAI completions response has nowhere to put a hidden-state payload, so the
states would have to travel out-of-band and be re-associated by request id.
"""

import os

import torch

# req_id -> hidden states for that request's tokens, [num_tokens, n_layers, d]
_CAPTURED: dict[str, torch.Tensor] = {}
_REGISTERED = False

CONNECTOR_NAME = "BeaverHiddenStatesConnector"


def ensure_in_process() -> None:
    """Pin EngineCore into this process so ``_CAPTURED`` is readable here.

    Must run before the engine is built; afterwards the setting is baked in.
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def register() -> None:
    """Register this connector with vLLM's factory (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    KVConnectorFactory.register_connector(
        CONNECTOR_NAME, __name__, "BeaverHiddenStatesConnector"
    )
    _REGISTERED = True


def _base_id(req_id: str) -> str:
    """Strip vLLM's internal hash suffix from a request id.

    The scheduler tags requests as ``"{request_id}-{hash}"`` (e.g.
    ``"0-bc9ad1ec"``) while ``RequestOutput.request_id`` carries the bare
    ``"0"``. Keying on the bare id is what lets the caller look up its own
    request; storing under both forms keeps the lookup working if a future
    vLLM drops or changes the suffix.
    """
    head, sep, tail = req_id.rpartition("-")
    # Only treat it as a suffix if it actually looks like the hex hash vLLM
    # appends -- never chop a legitimate id that happens to contain a dash.
    if sep and head and all(c in "0123456789abcdef" for c in tail) and tail:
        return head
    return req_id


def pop(req_id: str):
    """Take (and forget) the captured states for ``req_id``, or None."""
    hit = _CAPTURED.pop(req_id, None)
    if hit is None:
        hit = _CAPTURED.pop(_base_id(req_id), None)
    return hit


def pop_last(req_id: str):
    """Last token's hidden states for ``req_id`` as [n_layers, d], or None.

    The last prompt position is what the representation verifier scores: with
    ``max_tokens=1`` the whole scored sequence is the prompt, so this is the
    state after the final token of prompt+path.
    """
    hs = pop(req_id)
    return None if hs is None else hs[-1]


def clear() -> None:
    _CAPTURED.clear()


def _build_connector_class():
    from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
        ExampleHiddenStatesConnector,
        extract_from_kv_cache,
    )

    class BeaverHiddenStatesConnector(ExampleHiddenStatesConnector):
        """``ExampleHiddenStatesConnector`` with the disk write replaced.

        Only ``save_kv_layer`` differs: scheduler-side bookkeeping, layer
        discovery and metadata construction are all inherited unchanged.
        """

        def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
            if layer_name not in self.cache_layers:
                return
            meta = self._get_connector_metadata()
            for request in getattr(meta, "requests", []):
                hs = extract_from_kv_cache(
                    kv_layer, request.slot_mapping, request.token_ids.shape[0]
                )
                # .clone(): the extraction indexes into vLLM's paged buffer,
                # which is reused by the next step. Without a copy the stored
                # tensor would silently change under us.
                _CAPTURED[_base_id(request.req_id)] = hs.detach().clone()

    return BeaverHiddenStatesConnector


def __getattr__(name):
    # Built lazily so importing this module never drags in vLLM (the hf and
    # hf-local backends import beaver.model without vLLM installed).
    if name == "BeaverHiddenStatesConnector":
        return _build_connector_class()
    raise AttributeError(name)
