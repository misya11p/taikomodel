from pathlib import Path
import json
import random

from datasets import Dataset, Features, Value, Image
import typer

from instruction import inference_instruction, ocr_instructions


DPATH_IMAGES = Path("data/preprocessed/")
DPATH_MIMIC = Path("data/synthetic/mimic/")
DPATH_RANDOM = Path("data/synthetic/random/")
FPATH_ANNOTATED = Path("data/annotated.json")
FPATH_TRAIN = Path("data/train.txt")

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    n_natural: int = typer.Option(
        0,
        "--n-natural", "-n",
        help="Number of natural samples.",
    ),
    n_random: int = typer.Option(
        0,
        "--n-random", "-r",
        help="Number of random samples.",
    ),
    n_mimic: int = typer.Option(
        0,
        "--n-mimic", "-m",
        help="Number of mimic samples.",
    ),
    dpath_output: Path = typer.Option(
        "data/sft/",
        "--dpath-output", "-o",
        help="Directory path to save the generated images. ",
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Random seed for reproducibility. ",
    )
):
    random.seed(seed)

    data = []
    data += get_natural_samples(n_natural)
    data += get_random_samples(n_random)
    data += get_mimic_samples(n_mimic)

    random.shuffle(data)

    features = Features({
        "instruction": Value("string"),
        "image": Image(),
        "label": Value("string"),
    })
    ds = Dataset.from_list(data, features=features)
    ds.save_to_disk(dpath_output)


def get_natural_samples(n):
    with open(FPATH_ANNOTATED, "r") as f:
        annotated = json.load(f)
    with open(FPATH_TRAIN, "r") as f:
        files = f.read().splitlines()
    n = min(n, len(files)) if n >= 0 else len(files)

    data = []
    for file in files[:n]:
        instruction = inference_instruction
        image = str(DPATH_IMAGES / file)
        label = json.dumps(annotated[file], ensure_ascii=False, indent=2)
        data.append({
            "instruction": instruction,
            "image": image,
            "label": label,
        })

    return data


def get_random_samples(n):
    files = list((DPATH_RANDOM).glob("*.png"))
    n = min(n, len(files)) if n >= 0 else len(files)

    data = []
    for file in files[:n]:
        instruction = random.choice(ocr_instructions)
        image = str(file)
        label = file.with_suffix(".txt").read_text(encoding="utf-8")
        data.append({
            "instruction": instruction,
            "image": image,
            "label": label,
        })

    return data


def get_mimic_samples(n):
    files = list((DPATH_MIMIC).glob("*.png"))
    n = min(n, len(files)) if n >= 0 else len(files)

    data = []
    for file in files[:n]:
        instruction = inference_instruction
        image = str(file)
        label = file.with_suffix(".txt").read_text(encoding="utf-8")
        data.append({
            "instruction": instruction,
            "image": image,
            "label": label,
        })

    return data


if __name__ == "__main__":
    app()