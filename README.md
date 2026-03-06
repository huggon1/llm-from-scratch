# llm-from-scratch

Small, readable experiments that walk through a mini LLM pipeline:

1. train a tokenizer
2. pretrain a compact language model
3. run supervised fine-tuning
4. try DPO
5. try LoRA fine-tuning

## Layout

```text
llm-from-scratch/
  tokenizer/
  pretrain/
  sft/
  dpo/
  lora/
  docs/
```

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Tokenizers
- Pandas and NumPy

Install:

```bash
pip install -r requirements.txt
```

## What's Included

- Training scripts and model definitions for each stage
- Small sample datasets for demonstration
- Notes copied from the original study folders under `docs/`
- Minimal public-safe data samples that keep the repository lightweight

## What's Omitted

- Large checkpoints and `.pth` weights
- Full training corpora and large raw datasets
- Temporary training outputs
- Cache files and Python bytecode

Some scripts expect base model weights produced by an earlier stage. Those weights are intentionally not committed, so you should place them in the expected local path before training or inference.

## Suggested Order

### 1. Tokenizer

```bash
cd tokenizer
python main.py
```

### 2. Pretrain

Use the tokenizer artifacts under `pretrain/tokenizer/`, then run:

```bash
cd pretrain
python main.py
```

### 3. SFT / DPO / LoRA

These stages depend on locally available base weights from prior training.

```bash
cd sft
python main.py
```

```bash
cd dpo
python main.py
```

```bash
cd lora
python main.py
```

## Notes

- The repository keeps each stage self-contained so the training flow stays easy to follow.
- The original Chinese notes are preserved as markdown files in `docs/`.
- Sample datasets are intentionally small and mainly serve as runnable examples.
- The original larger datasets used during experimentation came from open-source or publicly available materials.
- Only a few sanitized sample rows are committed here so the repository stays lightweight and easy to share.
