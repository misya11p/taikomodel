import base64
import json
import re
from pathlib import Path
import asyncio
from difflib import SequenceMatcher

from openai import AsyncOpenAI
from environs import env
import typer
from tqdm import tqdm


env.read_env()
BASE_URL_OPENROUTER = "https://openrouter.ai/api/v1"
BASE_URL_OLLAMA = "http://localhost:11434/v1"
DPATH_IMAGES = "data/preprocessed/"
DPATH_DEFAULT_OUTPUT = "data/experiments/"
FPATH_INSTRUCTION = "instruction.txt"
FPATH_ANNOTATED = "data/annotated.json"
COLUMNS_NUM = ["良", "可", "不可", "進捗率", "スコア", "最大コンボ数", "連打数"]

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    model: str = typer.Option(
        "qwen/qwen3.5-35b-a3b",
        "--model", "-m",
        help="Model id to use for inference.",
    ),
    n_samples: int = typer.Option(
        None,
        "--n-samples", "-n",
        help=(
            "Number of samples to run inference on. "
            "Defaults to all samples if not specified."
        ),
    ),
    reasoning_effort: str = typer.Option(
        "none",
        "--reasoning",
        help=(
            "The level of reasoning to enable in the API call. "
            "Options are 'high', 'medium', 'low', or 'none'. "
        ),
    ),
    ollama: bool = typer.Option(
        False,
        "--ollama",
        help=(
            "Whether to use Ollama for local inference. If enabled, "
            "the base URL will automatically be set to "
            "'http://localhost:11434/v1'."
        ),
    ),
    list: str = typer.Option(
        "data/eval.txt",
        "--list", "-l",
        help=(
            "File path containing the list of image file names to run "
            "inference on."
        ),
    ),
    fpath_output: str = typer.Option(
        None,
        "--output", "-o",
        help=(
            "File path to save the inference results. "
            f"Defaults to '{DPATH_DEFAULT_OUTPUT}<model_name>.json'."
        ),
    ),
    base_url: str = typer.Option(
        None,
        "--base-url", "-b",
        help=(
            "Base URL for the API. If not specified, it will be set "
            "to OpenRouter's URL."
        ),
    ),
    api_key: str = typer.Option(
        None,
        "--api-key", "-k",
        help=(
            "API key for authentication. Defaults to the value of the "
            "OPENROUTER_API_KEY environment variable."
        ),
    ),
):
    instruction = Path(FPATH_INSTRUCTION).read_text()
    base_url = base_url or (BASE_URL_OLLAMA if ollama else BASE_URL_OPENROUTER)
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or env("OPENROUTER_API_KEY")
    )

    dpath_images = Path(DPATH_IMAGES)
    with open(list) as f:
        images = [dpath_images / line.strip() for line in f if line.strip()]
    if n_samples is not None:
        images = images[:n_samples]

    if fpath_output:
        fpath_output = Path(fpath_output)
    else:
        fpath_output = Path(DPATH_DEFAULT_OUTPUT) / f"{model.split('/')[-1]}.json"
    fpath_output.parent.mkdir(parents=True, exist_ok=True)

    with open(FPATH_ANNOTATED) as f:
        annotated = json.load(f)

    results = asyncio.run(run(
        model, instruction, client, images, reasoning_effort
    ))
    cost = sum(result[2] for result in results)
    results = {fpath: data for fpath, data, _ in results}
    results = dict(sorted(results.items()))
    stats, classes = evaluate(results, annotated)

    out = {
        "stats": stats,
        "cost": cost,
        "results": results,
    }
    with open(fpath_output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    fpath_output_summary = fpath_output.with_suffix(".summary.txt")
    summarize(fpath_output_summary, stats, classes, cost)

    print(f"Total cost: ${cost:.4f}")
    print(f"Saved inference results to {fpath_output}")
    print(f"Saved summary to {fpath_output_summary}")


async def run(model, instruction, client, images, reasoning):
    pbar = tqdm(total=len(images))
    results = await asyncio.gather(*[
        call_api(client, model, instruction, image, reasoning, pbar)
        for image in images
    ])
    return results


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def call_api(client, model, instruction, fpath_image, reasoning_effort, pbar):
    image_url = f"data:image/jpeg;base64,{encode_image(fpath_image)}"
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction}]
            },
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_url}}],
            },
        ],
        reasoning_effort=reasoning_effort,
    )
    res = response.choices[0].message.content
    cost = response.usage.model_dump().get("cost", 0)
    if match := re.search(r'\{.*\}', res, re.DOTALL):
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            data = {"error": f"Invalid JSON: {json_str}"}
    else:
        data = {"error": f"No JSON found in response: {res}"}
    pbar.update(1)
    return fpath_image.name, data, cost


def evaluate(results, annotated):
    correct_keys = []
    name_errors = []
    num_errors = []
    json_errors = []
    n_matches = 0
    n_matches_name = 0
    n_matches_num = 0
    sum_score_name = 0
    sum_score_num = 0

    for key, pred in results.items():
        label = annotated[key]
        if "error" in pred:
            json_errors.append((key, pred["error"]))
            continue

        if pred.keys() != label.keys():
            json_errors.append((key, f"Key mismatch: {pred.keys()}"))
            continue

        try:
            score_name = SequenceMatcher(None, pred["曲名"], label["曲名"]).ratio()
            score_num = sum([pred[idx] == label[idx] for idx in COLUMNS_NUM])
        except Exception as e:
            json_errors.append((key, f"Error comparing predictions: {e}, pred: {pred}"))
            continue

        score_num /= len(COLUMNS_NUM)
        if score_name == 1.0 and score_num == 1.0:
            n_matches += 1
            correct_keys.append(key)
        if score_name == 1.0:
            n_matches_name += 1
        else:
            name_errors.append((key, pred["曲名"], label["曲名"], score_name))
        if score_num == 1.0:
            n_matches_num += 1
        else:
            num_errors.append((key, pred, label, score_num))
        sum_score_name += score_name
        sum_score_num += score_num

    n = len(results)
    rate_match = n_matches / n
    rate_match_name = n_matches_name / n
    rate_match_num = n_matches_num / n
    avg_score_name = sum_score_name / n
    avg_score_num = sum_score_num / n

    stats = {
        "完全一致率": rate_match,
        "曲名完全一致率": rate_match_name,
        "数値完全一致率": rate_match_num,
        "曲名平均スコア": avg_score_name,
        "数値平均スコア": avg_score_num,
        "サンプル数": n,
    }
    classes = {
        "完全一致": correct_keys,
        "曲名エラー": name_errors,
        "数値エラー": num_errors,
        "JSONエラー": json_errors,
    }
    return stats, classes


def summarize(fpath_output_summary, stats, classes, cost):
    with open(fpath_output_summary, "w") as f:
        f.write("Evaluation Summary\n")
        f.write("==================\n\n")
        f.write(f"Total Cost: ${cost:.4f}\n")
        f.write("\nStatistics:\n")
        for stat_name, stat_value in stats.items():
            f.write(f"- {stat_name}: {stat_value}\n")
        f.write("\n曲名エラー:\n")
        for key, pred_name, label_name, score in classes["曲名エラー"]:
            f.write(f"{key}: '{pred_name}' - '{label_name}' ({score:.4f})\n")
        f.write("\n数値エラー:\n")
        for key, pred, label, score in classes["数値エラー"]:
            f.write(f"{key}: {pred} - {label}\n")
        f.write("\nJSONエラー:\n")
        for key, error in classes["JSONエラー"]:
            f.write(f"{key}: {error}\n")
        f.write("\n完全一致:\n")
        for key in classes["完全一致"]:
            f.write(f"{key}\n")


if __name__ == "__main__":
    app()
