"""
BEAVER: Formal verification bounds for LLM generation.

Quick start:
    import beaver

    results = beaver.run(
        prompts=[{"question": "What is 2+2?"}],
        constraint_fn=lambda instance, seq: "4" in seq,
        model="meta-llama/Llama-3.1-8B-Instruct",
        server_addr="http://localhost:8000",
    )
"""

__version__ = "0.1.0"

from .api import run
from .batch_runner import run_batch

__all__ = ["run", "run_batch", "__version__"]
