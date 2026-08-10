"""Command-line interface: ``radgraphit predict``.

The CLI is the one place the package writes to stdout (the library itself never prints). It reads
one or more reports — from positional arguments, a ``--input-file`` (one report per line), or
stdin — runs inference, and writes the RadGraph-XL dictionary as JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .api import RadGraphIT
from .core.errors import RadGraphITError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radgraphit",
        description="Extract the RadGraph-XL graph from Italian radiology reports.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Annotate one or more reports.")
    source = predict.add_mutually_exclusive_group()
    source.add_argument(
        "--model-dir",
        help="Path to a local model bundle (config.json, vocab.json, "
        "model_manifest.json, weights). Use this for development/offline.",
    )
    source.add_argument(
        "--model-id",
        help="Hugging Face model repository (standard path). Default: the registry.",
    )
    predict.add_argument(
        "--revision",
        help="Revision (commit sha or tag) of the Hugging Face repository. Used with --model-id.",
    )
    predict.add_argument(
        "--cache-dir", help="Cache directory for the model download (default: HF cache)."
    )
    predict.add_argument(
        "--device",
        default="auto",
        help="Device: auto (default), cpu, cuda, cuda:<index>.",
    )
    predict.add_argument(
        "--input-file",
        help="Text file with one report per line (alternative to positional arguments).",
    )
    predict.add_argument(
        "--indent", type=int, default=2, help="Indentation of the output JSON (default: 2)."
    )
    predict.add_argument(
        "reports",
        nargs="*",
        help="One or more reports as arguments. If absent, reads from --input-file or stdin.",
    )
    return parser


def _collect_reports(args: argparse.Namespace) -> list[str]:
    """Gather reports from the single input source validated by :func:`main`."""
    if args.reports:
        return list(args.reports)
    if args.input_file:
        with Path(args.input_file).open(encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    data = sys.stdin.read()
    return [line.strip() for line in data.splitlines() if line.strip()]


def _build_predictor(args: argparse.Namespace) -> RadGraphIT:
    if args.model_dir:
        return RadGraphIT.from_local(args.model_dir, device=args.device)
    kwargs: dict[str, object] = {"device": args.device, "cache_dir": args.cache_dir}
    if args.model_id:
        kwargs["model_id"] = args.model_id
    if args.revision:
        kwargs["revision"] = args.revision
    return RadGraphIT.from_pretrained(**kwargs)  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        if args.reports and args.input_file:
            print(
                "radgraphit: positional reports and --input-file are mutually exclusive.",
                file=sys.stderr,
            )
            return 2
        if args.model_dir and (args.revision or args.cache_dir):
            print(
                "radgraphit: --revision/--cache-dir cannot be used with --model-dir.",
                file=sys.stderr,
            )
            return 2
        try:
            reports = _collect_reports(args)
            if not reports:
                print("radgraphit: no report provided.", file=sys.stderr)
                return 2
            predictor = _build_predictor(args)
            annotations = predictor.predict(reports)
        except RadGraphITError as exc:
            print(f"radgraphit: {exc}", file=sys.stderr)
            return 1
        except (OSError, UnicodeError) as exc:
            print(f"radgraphit: {exc}", file=sys.stderr)
            return 1
        json.dump(annotations, sys.stdout, indent=args.indent, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    parser.error(f"unknown command {args.command!r}")  # argparse exits; unreachable return below
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
