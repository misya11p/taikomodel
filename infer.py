import base64
import json
import re
from pathlib import Path
import asyncio
import time

from openai import AsyncOpenAI
from environs import env
import typer
from tqdm import tqdm


FPATH_INSTRUCTION = "instruction.txt"
DPATH_IMAGES = "data/preprocessed/"
env.read_env()

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    model: str = typer.Option(
        "qwen/qwen3.5-35b-a3b",
        "--model", "-m",
        help="OpenRouter model id to use for inference.",
    ),
    fpath_input: str = typer.Option(
        "data/eval.txt",
        "--input", "-i",
        help=(
            "File path containing the list of image file names to run "
            "inference on. "
        ),
    ),
    fpath_output: str = typer.Option(
        None,
        "--output", "-o",
        help=(
            "File path to save the inference results. "
            r"Defaults to 'results_{model_name}.json' if not specified."
        ),
    ),
    n_samples: int = typer.Option(
        None,
        "--n-samples", "-n",
        help=(
            "Number of samples to run inference on. "
            "Defaults to all samples if not specified."
        )
    ),
    reasoning: bool = typer.Option(
        False,
        "--reasoning",
        help="Whether to enable reasoning in the API call."
    )
):
    instruction = Path(FPATH_INSTRUCTION).read_text()
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=env("OPENROUTER_API_KEY")
    )
    dpath_images = Path(DPATH_IMAGES)
    with open(fpath_input) as f:
        images = [dpath_images / line.strip() for line in f if line.strip()]
    if n_samples is not None:
        images = images[:n_samples]
    fpath_output = fpath_output or f"results_{model.split('/')[-1]}.json"
    pbar = tqdm(total=len(images))
    asyncio.run(run(
        fpath_output, model, instruction, client, images, reasoning, pbar
    ))
    print(f"Saved results to {fpath_output}")


async def run(fpath_output, model, instruction, client, images, reasoning, pbar):
    results = await asyncio.gather(*[
        call_api(client, model, instruction, fpath, reasoning, pbar) for fpath in images
    ])

    cost = sum(result[2] for result in results)
    print(f"Total cost: ${cost:.4f}")

    results = {fpath: data for fpath, data, _ in results}
    results = dict(sorted(results.items()))
    with open(fpath_output, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def call_api(client, model, instruction, fpath_image, reasoning, pbar):
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": instruction,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(fpath_image)}"
                        },
                    }
                ],
            },
        ],
        extra_body={"reasoning": {"enabled": reasoning}},
    )
    res = response.choices[0].message.content
    cost = response.usage.cost
    json_str = re.search(r'\{.*\}', res, re.DOTALL).group(0)
    data = json.loads(json_str)
    pbar.update(1)
    return fpath_image.name, data, cost


if __name__ == "__main__":
    app()
