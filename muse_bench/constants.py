SUPPORTED_METRICS = ["verbmem_f", "privleak", "knowmem_f", "knowmem_r"]

CORPORA = ["news"]

LLAMA_DIR = "meta-llama/Llama-2-7b-hf"

DEFAULT_DATA = {
    "news": {
        "verbmem_forget_file": "data/news/verbmem/forget.json",
        "privleak_forget_file": "data/news/privleak/forget.json",
        "privleak_retain_file": "data/news/privleak/retain.json",
        "privleak_holdout_file": "data/news/privleak/holdout.json",
        "knowmem_forget_qa_file": "data/news/knowmem/forget_qa.json",
        "knowmem_forget_qa_icl_file": "data/news/knowmem/forget_qa_icl.json",
        "knowmem_retain_qa_file": "data/news/knowmem/retain_qa.json",
        "knowmem_retain_qa_icl_file": "data/news/knowmem/retain_qa_icl.json",
    }
}

# Official retrain baseline used to normalize the optional MUSE privacy metric.
AUC_RETRAIN = {
    "news": {
        "forget_holdout_Min-40%": 0.47719999999999996,
    }
}
