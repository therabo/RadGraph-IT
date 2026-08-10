# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.1] - 2026-08-10

### Changed
- Use the final RadGraph-IT checkpoint trained on all 2,850 reports for 13 epochs, pinned to the
  immutable Hugging Face revision `11f691cd0e42985b87330e197507c91b43b9bfd8`.
- Make `safetensors` a core dependency because it is the release checkpoint format.
- Pin the model bundle and encoder to immutable Hugging Face commit revisions.
- Require a checksum-backed manifest for every local and remote model bundle.
- Download only manifest-declared bundle files and construct the encoder without redundant
  pretrained-weight downloads.

### Fixed
- Correct the installation command to use the published `RadGraph-IT` distribution name.
- Harden manifest paths, checksums, schema/version parsing, and encoder consistency checks.
- Normalize checkpoint, vocabulary, I/O, and inference failures into the public error hierarchy.
- Run the real pinned checkpoint against a committed synthetic golden output in CI.
- Align pre-commit, Python 3.13 coverage, CLI input validation, and documented cache size.

## [1.0.0]

First public release.

### Added
- Public Python API `radgraphit.RadGraphIT`: `from_pretrained()` (Hugging Face Hub),
  `from_local()` (local bundle), and direct lazy construction; `predict()` accepts a string or a
  sequence of strings and preserves input order; `predictor(...)` is an alias of `predict()`.
- Output in the canonical **RadGraph-XL** dictionary format: stable entity ordering, deterministic
  relation ordering, no dangling edges.
- Italian RadGraph-XL compatibility tokenizer (exact equivalent of
  `radgraph_xl_preprocess_report`).
- `dygie_v2` inference backend: pure-PyTorch port of the Italian v2 model (shared encoder, span
  extractor, NER head, relation head), on CPU or CUDA.
- Automatic anonymous model download from the public Hugging Face Hub repository
  (`radgraphIT/Radgraph-IT-v1`), cached after the first use.
- CLI `radgraphit predict`: reports as arguments, from `--input-file` (one per line), or from
  stdin; JSON output on stdout.
