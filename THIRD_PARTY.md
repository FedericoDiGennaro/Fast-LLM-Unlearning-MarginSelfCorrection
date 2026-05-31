# Third-Party Code Notice

This repository bundles a small amount of code copied from local external project folders so the experiments can be run from a single checkout.

## Included Third-Party Folders

- `open-unlearning/`: trimmed from the local `open-unlearning` project folder. It keeps only the TOFU finetuning/evaluation runtime, minimal configs, one retain90 TOFU reference-eval log, and its original `LICENSE`.
- `muse_bench/`: trimmed from the local `muse_bench` project folder. It keeps only the MUSE News data materialization code, evaluator, and metrics used by the reproduction scripts. No model weights or benchmark data are included.

## Included Project Artifact

- `assets/metric_runtime_sixpanel.png`: a single static README image copied from this project's generated figures. The plotting scripts and generated figure directories are not included.

## Not Included

- No files from `Unlearn-Simple/` are included.
- No files from `FailureLLMUnlearning/` are included.
- No plotting scripts or generated plot/figure directories are included.
- No checkpoints, model weights, Hugging Face tokens, API keys, personal cache files, or local cluster paths are intentionally included.

Before publishing, verify that the licenses of all bundled third-party code are compatible with your intended GitHub license.
