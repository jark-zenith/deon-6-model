# DEON 6 Model

DEON 6 is Jark AI Tech's model-development repository. It implements the engineering pipeline described by the DEON 6 Model Blueprint.

## Status

**Engineering scaffold / M1 preparation.** The repository is ready for tokenizer experiments, CPU smoke tests, reproducible training, evaluation, and later GPU training. No trained DEON 6 model weights are claimed by this repository.

## Targets

- Decoder-only Transformer
- RMSNorm, pre-norm
- RoPE positional embeddings
- Grouped-query attention (GQA)
- SwiGLU feed-forward layers
- 8K launch context
- bf16 training when supported, with development fallbacks
- ~64K byte-level BPE tokenizer
- English + Swahili + Sheng + code focus
- Nano → Mini → Base → Code → optional Pro

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m deon6.tools.smoke_test
python -m deon6.tokenizer.train --input data/examples/corpus.txt --output artifacts/tokenizer
python -m deon6.training.pretrain --config configs/nano-smoke.yaml
```

The smoke configuration is intentionally tiny. It validates the pipeline; it is not a useful language model.

## GPU readiness

The training code automatically selects CUDA when available. A future GPU machine can run the same training entry point. GitHub Actions contains a manual GPU workflow targeting a self-hosted runner labeled `gpu`.

GitHub supports self-hosted runners and lets workflows target custom labels such as `[self-hosted, linux, x64, gpu]`. See the official documentation: https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/use-in-a-workflow

Do not commit private datasets, credentials, API keys, or large checkpoints to Git.

## Layout

```text
configs/       reproducible model/training configs
data/          manifests and tiny development examples
deon6/         model, tokenizer, training, evaluation, inference, serving
scripts/       operational helpers
tests/         unit tests
.github/       CI and manual GPU workflow
docs/          engineering and data governance
```

## Model status

DEON 6 is not trained by simply uploading instructions. This repository contains the software needed to train it once suitable data and GPU compute are connected.

## License

Software licensing and model/dataset licensing will be finalized before a public model release.
