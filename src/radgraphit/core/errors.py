"""Domain exception hierarchy for predictable public loading and inference failures."""

from __future__ import annotations


class RadGraphITError(Exception):
    """Base class for every error raised by radgraphit."""


class InvalidReportError(RadGraphITError, ValueError):
    """Input is not a non-empty ``str`` or non-empty sequence of non-empty ``str``.

    Raised deterministically instead of the upstream package's behaviour of silently turning an
    empty report into the literal string ``"None"``.
    """


class ModelConfigurationError(RadGraphITError):
    """The model artifact could not be located or is misconfigured."""


class ModelNotConfiguredError(ModelConfigurationError):
    """The caller supplied an empty model identifier or revision."""


class ModelNotFoundError(ModelConfigurationError):
    """A configured model id / revision could not be downloaded from the Hugging Face Hub, or a
    local model directory does not exist or is missing required files."""


class ManifestError(ModelConfigurationError):
    """``model_manifest.json`` is missing, malformed, incompatible, or fails checksum
    verification."""


class DeviceError(RadGraphITError):
    """An invalid device string was requested, or CUDA was requested but is unavailable."""


class CheckpointError(ModelConfigurationError):
    """The checkpoint could not be loaded onto the reconstructed model topology (e.g. a
    ``strict=True`` state-dict mismatch)."""


class BackendError(RadGraphITError):
    """The inference backend named by the manifest is unknown or unsupported."""
