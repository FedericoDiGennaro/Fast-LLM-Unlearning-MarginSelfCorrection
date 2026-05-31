# Scaling Experiments on TOFU

This folder scaffolds the scaling experiment described in the project notes:

1. Download a family of base checkpoints across parameter scales.
2. Construct a task model for each scale by fine-tuning on the full TOFU dataset.
3. Start selective unlearning from those fine-tuned task models with a fixed objective and stopping rule.

The setup here is intentionally thin: it reuses OpenUnlearning's existing TOFU finetuning pipeline instead of creating a parallel training stack.

## Folder layout

- `manifests/qwen25_tofu_scaling.json`
  Scale manifest for the Qwen2.5 family.
- `scripts/download_family_models.py`
  Downloads all base checkpoints listed in a manifest into `exp_scaling/cache/hf/`.
- `scripts/run_download_qwen25_family.sh`
  Convenience launcher for downloading the Qwen2.5 family.
- `scripts/run_tofu_full_finetune.py`
  Launches OpenUnlearning full-TOFU supervised fine-tuning for one or more scales.

The current Qwen2.5 manifest includes:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

## TOFU fine-tuning protocol

The finetuning launcher follows OpenUnlearning's TOFU finetune recipe:

- experiment: `finetune/tofu/default.yaml`
- dataset: `TOFU_QA_full`, split `full`
- learning rate: `1e-5`
- weight decay: `0.01`
- warmup: `1.0` epochs
- epochs: `5`

That comes from:

- [default.yaml](../open-unlearning/configs/experiment/finetune/tofu/default.yaml)
- [finetune.yaml](../open-unlearning/configs/trainer/finetune.yaml)

For reliability in this workspace, the launcher also disables `flash_attention_2` via a Hydra override, since the local environment has previously failed on that path.

## Typical usage

Download the Qwen2.5 family:

```bash
cd .
source ./venv_open_unlearning/bin/activate
python exp_scaling/scripts/download_family_models.py \
  --manifest exp_scaling/manifests/qwen25_tofu_scaling.json
```

Or with the convenience launcher:

```bash
cd .
bash exp_scaling/scripts/run_download_qwen25_family.sh
```

Dry-run the TOFU full-SFT commands:

```bash
cd .
source ./venv_open_unlearning/bin/activate
python exp_scaling/scripts/run_tofu_full_finetune.py \
  --manifest exp_scaling/manifests/qwen25_tofu_scaling.json \
  --dry_run
```

Run the full-SFT jobs sequentially:

```bash
cd .
source ./venv_open_unlearning/bin/activate
python exp_scaling/scripts/run_tofu_full_finetune.py \
  --manifest exp_scaling/manifests/qwen25_tofu_scaling.json
```

## On "apples-to-apples" stopping comparisons

The number of optimizer steps per epoch is determined by:

- dataset size
- effective batch size
- number of processes / GPUs
- sampler/drop-last behavior

It is **not** determined directly by model size.

So:

- comparing **stop step**, **fraction of epoch**, or **examples processed** is apples-to-apples across scales
  if the data pipeline and effective batch size are held fixed;
- comparing **wall-clock time** is **not** apples-to-apples across scales,
  because larger models have slower forward/backward passes.

In other words:

- `stop_step` is a fair optimization-dynamics comparison,
- `seconds_to_stop` is a compute-efficiency comparison.

Both are useful, but they answer different questions.
