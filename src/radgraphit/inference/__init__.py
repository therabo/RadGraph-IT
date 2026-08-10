"""Inference orchestration: tokenizer, backend boundary, decoder, and the service.

Importing this package is torch-free; the torch-dependent DyGIE-v2 backend under
``backends/dygie_v2`` is imported lazily by the loader, never at package import time.
"""
