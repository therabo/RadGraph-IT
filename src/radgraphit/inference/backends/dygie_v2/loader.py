"""Load a v2 checkpoint into a runnable backend.

Reconstructs the model topology from ``config.json`` + ``vocab.json``, then loads the weights with
``strict=True`` so any mismatch between the checkpoint and the reconstructed graph fails loudly and
comprehensibly. For ``.pt`` checkpoints the safe ``torch.load(..., weights_only=True)`` path is
used; the released checkpoint uses the safer ``.safetensors`` format.

This is the only module that reads the config schema. It accepts every option v2 configs can carry
— ``encoder``, ``max_span_width``, ``feature_size``, ``feedforward_params``, ``loss_weights`` (read
but not needed at inference), ``relation_spans_per_word``, and, when present, ``span_pooling``,
``transformer_params``, ``relation_context``, ``relation_feedforward_params``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from ....core.errors import CheckpointError
from .backend import DyGIEv2Backend
from .device import resolve_device
from .model import DyGIEModel
from .tokenizer_embedder import MismatchedEmbedder
from .vocab import Vocabulary, derive_dataset

logger = logging.getLogger("radgraphit")


def build_model(
    config: dict[str, Any],
    vocab: Vocabulary,
    dataset: str,
    *,
    encoder_revision: str | None = None,
    embedder: MismatchedEmbedder | None = None,
) -> DyGIEModel:
    """Reconstruct the model topology from a v2 config dict + loaded vocab.

    ``embedder`` is injectable for testing (to avoid downloading an encoder); production leaves it
    ``None`` so the default :class:`MismatchedEmbedder` is built from ``config["encoder"]``.
    """
    try:
        encoder = config["encoder"]
        model = DyGIEModel(
            vocab,
            dataset,
            encoder_name=encoder["model_name"],
            encoder_revision=encoder_revision,
            max_length=encoder["max_length"],
            max_span_width=config["max_span_width"],
            feature_size=config["feature_size"],
            feedforward_params=config["feedforward_params"],
            relation_spans_per_word=config["relation_spans_per_word"],
            span_pooling=config.get("span_pooling", False),
            transformer_params=config.get("transformer_params"),
            relation_context=config.get("relation_context", False),
            relation_feedforward_params=config.get("relation_feedforward_params"),
            train_encoder=encoder.get("train_parameters", True),
            embedder=embedder,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(
            f"config.json is missing or has an invalid required field: {exc}. A valid v2 config "
            "needs at least "
            "'encoder', 'max_span_width', 'feature_size', 'feedforward_params', and "
            "'relation_spans_per_word'."
        ) from exc
    except Exception as exc:
        raise CheckpointError(
            f"Could not construct the encoder/model topology described by config.json: {exc}"
        ) from exc
    return model


def _read_state_dict(weights_path: Path) -> dict[str, torch.Tensor]:
    """Read a state dict from a ``.pt`` (safe, weights-only) or ``.safetensors`` checkpoint."""
    if weights_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - exercised only when safetensors is absent
            raise CheckpointError(
                f"{weights_path.name} is a safetensors checkpoint but the 'safetensors' package is "
                "not installed. Reinstall or upgrade RadGraph-IT."
            ) from exc
        try:
            state_dict = load_file(str(weights_path))
        except Exception as exc:
            raise CheckpointError(
                f"Could not read safetensors checkpoint {weights_path}: {exc}"
            ) from exc
        return state_dict

    # .pt path: weights_only=True refuses to execute arbitrary pickled code (safe for untrusted
    # bundles). The v2 trainer saves {"epoch": int, "model_state_dict": OrderedDict[str, Tensor]}.
    try:
        obj = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"Could not safely read checkpoint {weights_path}: {exc}") from exc
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    if not isinstance(obj, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in obj.items()
    ):
        raise CheckpointError(
            f"Unexpected checkpoint structure in {weights_path.name}: expected a mapping from "
            "parameter names to tensors, optionally under 'model_state_dict'."
        )
    return obj


def load_weights(model: DyGIEModel, weights_path: Path) -> None:
    """Load weights into ``model`` with ``strict=True``, raising a clear error on mismatch."""
    state_dict = _read_state_dict(weights_path)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise CheckpointError(
            f"Failed to load {weights_path.name} into the reconstructed model (strict=True): "
            f"{exc}. This usually means config.json / vocab.json do not describe the same "
            "architecture the checkpoint was trained with."
        ) from exc


def load_backend(
    *,
    config_path: Path,
    vocab_path: Path,
    weights_path: Path,
    device: str = "auto",
    encoder_revision: str | None = None,
    embedder: MismatchedEmbedder | None = None,
) -> DyGIEv2Backend:
    """Build and return a ready-to-run :class:`DyGIEv2Backend` from a verified bundle's files."""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(
            f"Could not read a valid JSON config at {config_path}: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise CheckpointError(f"config.json at {config_path} must contain a JSON object.")
    vocab = Vocabulary.load(vocab_path)
    dataset = derive_dataset(vocab)
    torch_device = resolve_device(device)

    model = build_model(
        config,
        vocab,
        dataset,
        encoder_revision=encoder_revision,
        embedder=embedder,
    )
    load_weights(model, weights_path)
    try:
        model.to(torch_device).eval()
    except Exception as exc:
        raise CheckpointError(
            f"Could not move the reconstructed model to {torch_device}: {exc}"
        ) from exc
    logger.debug("Loaded dygie_v2 backend (dataset=%s) onto %s", dataset, torch_device)
    return DyGIEv2Backend(model, torch_device)
