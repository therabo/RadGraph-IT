"""The single source of truth for the default model location and supported bundle schema."""

from __future__ import annotations

DEFAULT_MODEL_ID = "radgraphIT/Radgraph-IT"
DEFAULT_REVISION = "11f691cd0e42985b87330e197507c91b43b9bfd8"
ENCODER_REVISIONS = {
    "IVN-RIN/medBIT-r3-plus": "0f1c91e551551869a72935962b54c1f5247333a4",
}

#: Files the model bundle must contain on the Hub / in a local directory.
MANIFEST_FILENAME = "model_manifest.json"
CONFIG_FILENAME = "config.json"
VOCAB_FILENAME = "vocab.json"

#: Weight file candidates, in preference order. The released model uses ``model.safetensors``;
#: ``best.pt`` remains supported for compatible local bundles.
WEIGHT_FILENAMES = ("model.safetensors", "best.pt")

#: Backend identifiers this package version knows how to build.
SUPPORTED_BACKENDS = frozenset({"dygie_v2"})

#: Manifest ``schema`` values this package version can serialize to.
SUPPORTED_SCHEMAS = frozenset({"radgraph-xl"})
