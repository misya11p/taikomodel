from pathlib import Path

import numpy as np
from PIL import Image
import typer
from tqdm import tqdm


W, H = 1280, 720
DPATH_IMAGES = "data/images/"
DPATH_PREPROCESSED = "data/preprocessed/"


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    overwrite: bool = typer.Option(
        False,
        "--overwrite", "-o",
        help="Whether to overwrite existing preprocessed images.",
    )
):
    dpath_images = Path(DPATH_IMAGES)
    dpath_preprocessed = Path(DPATH_PREPROCESSED)
    dpath_preprocessed.mkdir(exist_ok=True)
    fpaths = list(dpath_images.glob("*.jpg"))
    for fpath in tqdm(fpaths):
        fpath_out = dpath_preprocessed / fpath.name
        if (not overwrite) and fpath_out.exists():
            continue
        img = Image.open(fpath)
        img = preprocess(img)
        img.save(fpath_out)


def preprocess(img):
    img = img.resize((W, H))
    imarr = np.array(img)
    imarr = imarr[:-220, 470:-40]
    imarr = np.concatenate((imarr[:90], imarr[280:]), axis=0)
    imarr[70:95] = 0
    imarr[130:160] = 0
    imarr[95:105, :430] = 0
    imarr[120:130, 492:] = 0
    imarr[95:120, 635:] = 0
    imarr[242:290, 15:250] = 0
    img = Image.fromarray(imarr)
    return img


if __name__ == "__main__":
    app()