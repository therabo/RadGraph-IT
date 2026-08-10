# RadGraph-IT 🇮🇹

**RadGraph** graph inference for **Italian** radiology reports, with output in the canonical
**RadGraph-XL** format.

`radgraphit` takes one or more Italian radiology reports, runs the Italian DyGIE v2 model
(PyTorch: shared encoder, span extractor, NER head, and relation head), and returns the graph of
entities and relations as the RadGraph-XL-compatible dictionary.

- **PyPI distribution:** `RadGraph-IT` · **Import / CLI:** `radgraphit` · **Python:** `>= 3.10`
  · **Device:** CPU or CUDA
- **Model:** downloaded automatically from pinned Hugging Face revisions on first use
  (~445 MB / 424 MiB plus small tokenizer/config files, then cached)

## Installation

```bash
pip install RadGraph-IT
```

## Usage — Python API

```python
from radgraphit import RadGraphIT

predictor = RadGraphIT()
annotations = predictor.predict(
    "Non si evidenziano segni di pneumotorace dopo la rimozione del drenaggio toracico."
)
```

`predict` accepts a **string** or a **sequence of strings** and preserves order.
`predictor(report)` is an alias for `predictor.predict(report)`:

```python
annotations = predictor(["referto 1", "referto 2"])  # batch, order preserved
```

## Usage — CLI

```bash
# report as an argument
radgraphit predict "Non si evidenzia pneumotorace."

# from a file (one report per line), or from stdin
radgraphit predict --input-file reports.txt
echo "Nessuna evidenza di pneumotorace." | radgraphit predict
```

The result is written as JSON to stdout.

## Output

For the input `["Non si evidenzia pneumotorace ..."]` the output has this shape (top-level keys
`"0"`, `"1"`, … in input order):

```python
{
    "0": {
        "text": "<normalized tokens, space-separated>",
        "entities": {
            "1": {
                "tokens": "<span tokens>",
                "label": "Observation::definitely absent",
                "start_ix": 3,
                "end_ix": 3,
                "relations": [["located_at", "2"]],
            }
        },
        "data_source": None,
        "data_split": "inference",
    },
    "1": {"...": "..."},
}
```

- The indices (`start_ix`, `end_ix`) are **token-level, zero-based, and inclusive**.
- Relations are directed *source → target* lists `[label, target_entity_id]`.
- The output is deterministic: entities and relations are stably ordered.

## Citation

The output format and the serializer derive from the official **RadGraph / RadGraph-XL** package
(see [`Third-Party`](LICENSES/Third-Party.md)). If you use this package, please cite
both this work and the RadGraph-XL paper:

```bibtex
@software{radgraphit,
    author = "Daniel Rabottini and Edoardo Avenia and Rocco Angelella",
    title = "{RadGraph-IT}: {RadGraph} graph inference for {Italian} radiology reports",
    year = "2026",
    version = "1.0.1",
    url = "https://github.com/therabo/RadGraph-IT",
}
```

```bibtex
@inproceedings{delbrouck-etal-2024-radgraph,
    title = "{R}ad{G}raph-{XL}: A Large-Scale Expert-Annotated Dataset for Entity and Relation
             Extraction from Radiology Reports",
    author = "Delbrouck, Jean-Benoit and Chambon, Pierre and Chen, Zhihong and Varma, Maya and
              Johnston, Andrew and Blankemeier, Louis and Van Veen, Dave and Bui, Tan and
              Truong, Steven and Langlotz, Curtis",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    year = "2024",
    url = "https://aclanthology.org/2024.findings-acl.765",
    pages = "12902--12915",
}
```

## License

See the [`MIT License`](LICENSE) and [`Third-Party`](LICENSES/Third-Party.md) notices.

> **Not a medical device.** The output is an automatic extraction of entities and relations from
> text and does not constitute a diagnosis, a report, or a clinical opinion.
