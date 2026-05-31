from .metrics.verbmem import eval as eval_verbmem
from .metrics.privleak import eval as eval_privleak
from .metrics.knowmem import eval as eval_knowmem
from .utils import load_model, load_tokenizer, write_csv, read_json, write_json
from .constants import SUPPORTED_METRICS, CORPORA, LLAMA_DIR, DEFAULT_DATA, AUC_RETRAIN

import os
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Any, List, Dict, Literal
from pandas import DataFrame


def _metric_value_key_from_agg_key(agg_key: str) -> str:
    """Map an aggregate key such as mean_rougeL to the per-example log key."""
    for prefix in ("mean_", "max_"):
        if agg_key.startswith(prefix):
            return agg_key[len(prefix):]
    return agg_key


def _per_example_values_from_log(log: List[Dict[str, Any]], agg_key: str) -> List[float]:
    value_key = _metric_value_key_from_agg_key(agg_key)
    values = []
    for idx, item in enumerate(log):
        if value_key not in item:
            raise KeyError(
                f"Cannot bootstrap aggregate key {agg_key!r}: per-example "
                f"log item {idx} does not contain {value_key!r}."
            )
        values.append(float(item[value_key]))
    return values


def eval_model(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer = LLAMA_DIR,
    metrics: List[str] = SUPPORTED_METRICS,
    corpus: Literal['news'] | None = None,
    privleak_auc_key: str = 'forget_holdout_Min-40%',
    verbmem_agg_key: str = 'mean_rougeL',
    verbmem_max_new_tokens: int = 128,
    knowmem_agg_key: str = 'mean_rougeL',
    knowmem_max_new_tokens: int = 32,
    verbmem_forget_file: str | None = None,
    privleak_forget_file: str | None = None,
    privleak_retain_file: str | None = None,
    privleak_holdout_file: str | None = None,
    knowmem_forget_qa_file: str | None = None,
    knowmem_forget_qa_icl_file: str | None = None,
    knowmem_retain_qa_file: str | None = None,
    knowmem_retain_qa_icl_file: str | None = None,
    temp_dir: str | None = None,
    data_root: str | None = None,
    print_metrics_live: bool = False,
    model_name: str | None = None,
    per_example_out_file: str | None = None,
) -> Dict[str, float]:
    # Argument sanity check
    if not metrics:
        raise ValueError(f"Specify `metrics` to be a non-empty list.")
    for metric in metrics:
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"Given metric {metric} is not supported.")
    if corpus is not None and corpus not in CORPORA:
        raise ValueError("Invalid corpus. This bundle includes only the MUSE News evaluator.")
    if corpus is not None:
        verbmem_forget_file = DEFAULT_DATA[corpus]['verbmem_forget_file'] if verbmem_forget_file is None else verbmem_forget_file
        privleak_forget_file = DEFAULT_DATA[corpus]['privleak_forget_file'] if privleak_forget_file is None else privleak_forget_file
        privleak_retain_file = DEFAULT_DATA[corpus]['privleak_retain_file'] if privleak_retain_file is None else privleak_retain_file
        privleak_holdout_file = DEFAULT_DATA[corpus]['privleak_holdout_file'] if privleak_holdout_file is None else privleak_holdout_file
        knowmem_forget_qa_file = DEFAULT_DATA[corpus]['knowmem_forget_qa_file'] if knowmem_forget_qa_file is None else knowmem_forget_qa_file
        knowmem_forget_qa_icl_file = DEFAULT_DATA[corpus]['knowmem_forget_qa_icl_file'] if knowmem_forget_qa_icl_file is None else knowmem_forget_qa_icl_file
        knowmem_retain_qa_file = DEFAULT_DATA[corpus]['knowmem_retain_qa_file'] if knowmem_retain_qa_file is None else knowmem_retain_qa_file
        knowmem_retain_qa_icl_file = DEFAULT_DATA[corpus]['knowmem_retain_qa_icl_file'] if knowmem_retain_qa_icl_file is None else knowmem_retain_qa_icl_file

    out = {}
    per_example_metrics: Dict[str, Dict[str, List[float]]] = {corpus: {}} if corpus is not None else {}
    if data_root is not None:
        def resolve_path(path: str | None) -> str | None:
            if path is None or os.path.isabs(path):
                return path
            return os.path.join(data_root, path)

        verbmem_forget_file = resolve_path(verbmem_forget_file)
        privleak_forget_file = resolve_path(privleak_forget_file)
        privleak_retain_file = resolve_path(privleak_retain_file)
        privleak_holdout_file = resolve_path(privleak_holdout_file)
        knowmem_forget_qa_file = resolve_path(knowmem_forget_qa_file)
        knowmem_forget_qa_icl_file = resolve_path(knowmem_forget_qa_icl_file)
        knowmem_retain_qa_file = resolve_path(knowmem_retain_qa_file)
        knowmem_retain_qa_icl_file = resolve_path(knowmem_retain_qa_icl_file)

    if not getattr(model, "is_loaded_in_4bit", False) and not getattr(model, "is_loaded_in_8bit", False):
        model = model.to('cuda')

    label = model_name or "model"

    def maybe_print(metric_name: str, value: float):
        if print_metrics_live:
            print(f"[{label}] {metric_name} = {value:.4f}", flush=True)

    # 1. verbmem_f
    if 'verbmem_f' in metrics:
        data = read_json(verbmem_forget_file)
        agg, log = eval_verbmem(
            prompts=[d['prompt'] for d in data],
            gts=[d['gt'] for d in data],
            model=model, tokenizer=tokenizer,
            max_new_tokens=verbmem_max_new_tokens
        )
        if temp_dir is not None:
            write_json(agg, os.path.join(temp_dir, "verbmem_f/agg.json"))
            write_json(log, os.path.join(temp_dir, "verbmem_f/log.json"))
        out['verbmem_f'] = agg[verbmem_agg_key] * 100
        if corpus is not None:
            per_example_metrics[corpus]['verbmem_f'] = _per_example_values_from_log(log, verbmem_agg_key)
        maybe_print('verbmem_f', out['verbmem_f'])

    # 2. privleak
    if 'privleak' in metrics:
        auc, log = eval_privleak(
            forget_data=read_json(privleak_forget_file),
            retain_data=read_json(privleak_retain_file),
            holdout_data=read_json(privleak_holdout_file),
            model=model, tokenizer=tokenizer
        )
        if temp_dir is not None:
            write_json(auc, os.path.join(temp_dir, "privleak/auc.json"))
            write_json(log, os.path.join(temp_dir, "privleak/log.json"))
        out['privleak'] = (auc[privleak_auc_key] - AUC_RETRAIN[corpus][privleak_auc_key]) / AUC_RETRAIN[corpus][privleak_auc_key] * 100
        # privleak is an AUC over score distributions, not a per-example mean.
        maybe_print('privleak', out['privleak'])

    # 3. knowmem_f
    if 'knowmem_f' in metrics:
        qa = read_json(knowmem_forget_qa_file)
        icl = read_json(knowmem_forget_qa_icl_file)
        agg, log = eval_knowmem(
            questions=[d['question'] for d in qa],
            answers=[d['answer'] for d in qa],
            icl_qs=[d['question'] for d in icl],
            icl_as=[d['answer'] for d in icl],
            model=model, tokenizer=tokenizer,
            max_new_tokens=knowmem_max_new_tokens
        )
        if temp_dir is not None:
            write_json(agg, os.path.join(temp_dir, "knowmem_f/agg.json"))
            write_json(log, os.path.join(temp_dir, "knowmem_f/log.json"))
        out['knowmem_f'] = agg[knowmem_agg_key] * 100
        if corpus is not None:
            per_example_metrics[corpus]['knowmem_f'] = _per_example_values_from_log(log, knowmem_agg_key)
        maybe_print('knowmem_f', out['knowmem_f'])

    # 4. knowmem_r
    if 'knowmem_r' in metrics:
        qa = read_json(knowmem_retain_qa_file)
        icl = read_json(knowmem_retain_qa_icl_file)
        agg, log = eval_knowmem(
            questions=[d['question'] for d in qa],
            answers=[d['answer'] for d in qa],
            icl_qs=[d['question'] for d in icl],
            icl_as=[d['answer'] for d in icl],
            model=model, tokenizer=tokenizer,
            max_new_tokens=knowmem_max_new_tokens
        )
        if temp_dir is not None:
            write_json(agg, os.path.join(temp_dir, "knowmem_r/agg.json"))
            write_json(log, os.path.join(temp_dir, "knowmem_r/log.json"))
        out['knowmem_r'] = agg[knowmem_agg_key] * 100
        if corpus is not None:
            per_example_metrics[corpus]['knowmem_r'] = _per_example_values_from_log(log, knowmem_agg_key)
        maybe_print('knowmem_r', out['knowmem_r'])

    if per_example_out_file is not None and any(per_example_metrics.values()):
        write_json(per_example_metrics, per_example_out_file)

    return out


def load_then_eval_models(
    model_dirs: List[str],
    names: List[str],
    corpus: Literal['news'],
    tokenizer_dir: str = LLAMA_DIR,
    out_file: str | None = None,
    metrics: List[str] = SUPPORTED_METRICS,
    temp_dir: str = "temp",
    data_root: str | None = None,
    quantize_4bit: int = 0,
    quantize_8bit: int = 0,
    print_metrics_live: int = 0,
    save_per_example_metrics: int = 1,
    per_example_out_dir: str | None = None,
) -> DataFrame:
    # Argument sanity check
    if not model_dirs:
        raise ValueError(f"`model_dirs` should be non-empty.")
    if len(model_dirs) != len(names):
        raise ValueError(f"`model_dirs` and `names` should equal in length.")
    if out_file is not None and not out_file.endswith('.csv'):
        raise ValueError(f"The file extension of `out_file` should be '.csv'.")

    # Run evaluation
    out = []
    for model_dir, name in zip(model_dirs, names):
        if print_metrics_live:
            print(f"[{name}] loading model from {model_dir}", flush=True)
        model = load_model(model_dir, quantize_4bit=quantize_4bit, quantize_8bit=quantize_8bit)
        tokenizer = load_tokenizer(tokenizer_dir)
        model_temp_dir = os.path.join(temp_dir, name)
        per_example_out_file = None
        if save_per_example_metrics:
            if per_example_out_dir is not None:
                per_example_out_file = os.path.join(per_example_out_dir, name, "per_example_metrics.json")
            else:
                per_example_out_file = os.path.join(model_temp_dir, "per_example_metrics.json")
        res = eval_model(
            model, tokenizer, metrics, corpus,
            temp_dir=model_temp_dir,
            data_root=data_root,
            print_metrics_live=bool(print_metrics_live),
            model_name=name,
            per_example_out_file=per_example_out_file,
        )
        out.append({'name': name} | res)
        if print_metrics_live:
            print(f"[{name}] completed with metrics: {res}", flush=True)
            if per_example_out_file is not None:
                print(f"[{name}] per-example metrics saved to {per_example_out_file}", flush=True)
        if out_file is not None: write_csv(out, out_file)
    return DataFrame(out)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dirs', type=str, nargs='+', default=[])
    parser.add_argument('--names', type=str, nargs='+', default=[])
    parser.add_argument('--tokenizer_dir', type=str, default=LLAMA_DIR)
    parser.add_argument('--corpus', type=str, required=True, choices=CORPORA)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--metrics', type=str, nargs='+', default=SUPPORTED_METRICS)
    parser.add_argument('--temp_dir', type=str, default='temp')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--quantize_4bit', type=int, default=0)
    parser.add_argument('--quantize_8bit', type=int, default=0)
    parser.add_argument('--print_metrics_live', type=int, default=0)
    parser.add_argument('--save_per_example_metrics', type=int, default=1)
    parser.add_argument('--per_example_out_dir', type=str, default=None)
    args = parser.parse_args()
    load_then_eval_models(**vars(args))
