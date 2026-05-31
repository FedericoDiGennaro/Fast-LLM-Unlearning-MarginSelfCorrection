import os
import argparse
from datasets import load_dataset
from utils import write_json, write_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        choices=["news"],
        default="news",
        help="Benchmark corpus to materialize.",
    )
    return parser.parse_args()


def materialize_corpus(corpus: str, Corpus: str):
    for split in ['forget_qa', 'retain_qa', 'forget_qa_icl', 'retain_qa_icl']:
        data = load_dataset(f"muse-bench/MUSE-{Corpus}", 'knowmem', split=split)
        questions, answers = data['question'], data['answer']
        knowmem = [
            {'question': question, 'answer': answer}
            for question, answer in zip(questions, answers)
        ]
        write_json(knowmem, f"data/{corpus}/knowmem/{split}.json")

    for split in ['forget']:
        data = load_dataset(f"muse-bench/MUSE-{Corpus}", 'verbmem', split='forget')
        prompts, gts = data['prompt'], data['gt']
        verbmem = [
            {'prompt': prompt, 'gt': gt}
            for prompt, gt in zip(prompts, gts)
        ]
        write_json(verbmem, f"data/{corpus}/verbmem/forget.json")

    for split in ['forget', 'retain', 'holdout']:
        privleak = load_dataset(f"muse-bench/MUSE-{Corpus}", 'privleak', split=split)['text']
        write_json(privleak, f"data/{corpus}/privleak/{split}.json")

    for split in ['forget', 'holdout', 'retain1', 'retain2']:
        raw = load_dataset(f"muse-bench/MUSE-{Corpus}", 'raw', split=split)['text']
        write_json(raw, f"data/{corpus}/raw/{split}.json")
        write_text("\n\n".join(raw), f"data/{corpus}/raw/{split}.txt")


def materialize_news_auxiliary():
    for crit in ['scal', 'sust']:
        for fold in range(1, 5):
            data = load_dataset(f"muse-bench/MUSE-News", crit, split=f"forget_{fold}")['text']
            write_json(data, f"data/news/{crit}/forget_{fold}.json")
            write_text("\n\n".join(data), f"data/news/{crit}/forget_{fold}.txt")


def main():
    args = parse_args()
    os.makedirs('data', exist_ok=True)
    materialize_corpus("news", "News")
    materialize_news_auxiliary()


if __name__ == "__main__":
    main()
