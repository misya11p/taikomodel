import base64
import json
import re
from pathlib import Path
import random
import asyncio

from openai import AsyncOpenAI
from environs import env

import typer


FPATH_INSTRUCTION = "instruction.txt"
DPATH_IMAGES = "data/preprocessed/"
env.read_env()

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    n_samples: int = typer.Option(
        100,
        "--n-samples", "-n",
        help="Number of samples to infer.",
    ),
    model: str = typer.Option(
        "qwen/qwen3.5-35b-a3b",
        "--model", "-m",
        help="OpenRouter model id to use for inference.",
    ),
    fpath_output: str = typer.Option(
        None,
        "--output", "-o",
        help=(
            "File path to save the inference results. "
            r"Defaults to 'results_{model_name}.json' if not specified."
        ),
    ),
):
    instruction = Path(FPATH_INSTRUCTION).read_text()
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=env("OPENROUTER_API_KEY")
    )
    images = list(Path(DPATH_IMAGES).glob("*.jpg"))
    images = random.sample(images, min(n_samples, len(images)))
    fpath_output = fpath_output or f"results_{model.split('/')[-1]}.json"

    asyncio.run(run(fpath_output, model, instruction, client, images))


async def run(fpath_output, model, instruction, client, images):
    results = await asyncio.gather(*[
        call_api(client, model, instruction, fpath) for fpath in images
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


async def call_api(client, model, instruction, fpath_image):
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
        extra_body={"reasoning": {"enabled": False}},
    )
    res = response.choices[0].message.content
    cost = response.usage.cost
    json_str = re.search(r'\{.*\}', res, re.DOTALL).group(0)
    data = json.loads(json_str)
    return fpath_image.name, data, cost


if __name__ == "__main__":
    app()
