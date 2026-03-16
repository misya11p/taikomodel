from pathlib import Path
import json
import random

from datasets import Dataset, Features, Value, Image
from datasets.table import embed_table_storage
import typer


DPATH_IMAGES = "data/preprocessed/"
FPATH_ANNOTATED = "data/annotated.json"
FPATH_TRAIN = "data/train.txt"
FPATH_EVAL = "data/eval.txt"
FPATH_PARQUET_TRAIN = "data/train.parquet"
FPATH_PARQUET_EVAL = "data/eval.parquet"

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    n_eval: int = typer.Option(
        80,
        "--n-eval", "-n",
        help="Number of evaluation samples.",
    ),
    seed: int = typer.Option(
        0,
        "--seed", "-s",
        help="Random seed for splitting the dataset.",
    ),
):
    with open(FPATH_ANNOTATED, "r") as f:
        data = json.load(f)
    train_data, eval_data = split(data, n_eval, seed)
    ds_train = create_dataset(train_data)
    ds_eval = create_dataset(eval_data)

    with open(FPATH_TRAIN, "w") as ft, open(FPATH_EVAL, "w") as fe:
        ft.write("\n".join(train_data.keys()))
        fe.write("\n".join(eval_data.keys()))

    ds_train.to_parquet(FPATH_PARQUET_TRAIN)
    ds_eval.to_parquet(FPATH_PARQUET_EVAL)


def split(data, n_eval, seed):
    files = list(data.keys())
    random.seed(seed)
    random.shuffle(files)
    eval_data = {k: data[k] for k in sorted(files[:n_eval])}
    train_data = {k: data[k] for k in sorted(files[n_eval:])}
    return train_data, eval_data


def create_dataset(data):
    dpath_images = Path(DPATH_IMAGES)
    texts, images = [], []
    for k, v in data.items():
        texts.append(json.dumps(v, ensure_ascii=False, indent=2))
        images.append(str(dpath_images / k))
    features = Features({
        "text": Value("string"),
        "image": Image(),
    })
    ds = Dataset.from_dict(
        {"text": texts, "image": images},
        features=features
    )
    format = ds.format
    ds = ds.with_format("arrow")
    ds = ds.map(embed_table_storage, batched=True)
    ds = ds.with_format(**format)
    return ds


if __name__ == "__main__":
    app()