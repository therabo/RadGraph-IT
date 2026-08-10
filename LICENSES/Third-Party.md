# Third-Party

This file collects provenance and attributions. It does **not** declare weights, datasets, or
clinical texts redistributable, and it does **not** automatically copy any third-party license. The
package code itself is licensed under MIT (see the root `LICENSE` file); this file concerns the
third-party components it builds on.

## RadGraph-XL output format and serializer

The output dictionary and the serialization logic (`postprocess_reports` / `get_entity`) derive from
the official **RadGraph / RadGraph-XL** package from Stanford AIMI / StanfordMIMI.

- Upstream repository: `radgraph` package (Stanford), distributed under the MIT license.

Citation:

> Delbrouck et al. "RadGraph-XL: A Large-Scale Expert-Annotated Dataset for Entity and Relation
> Extraction from Radiology Reports." Findings of the ACL 2024, pp. 12902–12915.
> https://aclanthology.org/2024.findings-acl.765

## Inference backend (`dygie_v2`)

An inference-only PyTorch port of the Italian v2 model (internal project "RadGraph-XL-IT",
`training_v2`), which is itself a pure-PyTorch reimplementation of the DyGIE++ architecture
(NER + relations) without AllenNLP.

- DyGIE / DyGIE++ architecture: Wadden et al., "Entity, Relation, and Event Extraction with
  Contextualized Span Representations", EMNLP 2019.
- Pre-trained encoder: fetched at runtime from the Hugging Face Hub (e.g. `IVN-RIN/medBIT-r3-plus`);
  subject to the license of its own repository on the Hub. Not redistributed by this package.

## Model, weights, and data

The model weights, the datasets, and any clinical text are **not** included in this repository or in
the wheel/sdist, and are not declared redistributable. The model bundle is hosted separately on the
Hugging Face Hub and is subject to the licenses/terms declared in that repository.

## Runtime dependencies

`torch`, `transformers`, `nltk`, `huggingface_hub` are subject to their respective licenses
(BSD-3-Clause, Apache-2.0, Apache-2.0, Apache-2.0 respectively, unless changed upstream).
