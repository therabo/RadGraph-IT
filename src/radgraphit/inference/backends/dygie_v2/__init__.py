"""DyGIE-v2 backend: an inference-only PyTorch port of ``training_v2/src/dygie``.

Private to the package — none of these classes are part of the public API. The parameter-bearing
modules are faithful ports of the training code so a v2 checkpoint loads with ``strict=True``.
Importing this package pulls in torch + transformers; it is imported lazily by the API facade.
"""

from .backend import DyGIEv2Backend
from .loader import load_backend

__all__ = ["DyGIEv2Backend", "load_backend"]
