"""radgraphit — RadGraph-XL inference for Italian radiology reports.

Public API::

    from radgraphit import RadGraphIT

Importing this package has no side effects: no network I/O, no model download, no logging to
stdout, and no torch import. Everything model-related is loaded lazily on first use.
"""

from __future__ import annotations

import logging

from .api import RadGraphIT
from .core.errors import (
    BackendError,
    CheckpointError,
    DeviceError,
    InvalidReportError,
    ManifestError,
    ModelConfigurationError,
    ModelNotConfiguredError,
    ModelNotFoundError,
    RadGraphITError,
)

# A library must not configure logging handlers for the application; attach a NullHandler so that
# emitting logs never prints a "No handlers could be found" warning if the app hasn't configured
# logging (see https://docs.python.org/3/howto/logging.html#library-config).
logging.getLogger("radgraphit").addHandler(logging.NullHandler())

__version__ = "1.0.1"

__all__ = [
    "BackendError",
    "CheckpointError",
    "DeviceError",
    "InvalidReportError",
    "ManifestError",
    "ModelConfigurationError",
    "ModelNotConfiguredError",
    "ModelNotFoundError",
    "RadGraphIT",
    "RadGraphITError",
    "__version__",
]
