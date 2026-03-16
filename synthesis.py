from pathlib import Path
import random
import csv
import json

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import typer


DPATH_IMAGES = Path("./data/preprocessed/")
DNAME_RANDOM = "random"
DNAME_MIMIC = "mimic"
FPATH_SONGS = Path("./data/songs.csv")
FPATH_FONT = "assets/FOT-大江戸勘亭流 Std E.otf"
FPATH_CHARACTERS = "assets/characters.txt"
W, H = 770, 310

with open(FPATH_SONGS, "r") as f:
    reader = csv.reader(f)
    next(reader)
    songs = [row[1] for row in reader]

with open(FPATH_CHARACTERS, "r") as f:
    chars = f.read().splitlines()
    chars_kana = chars[0]
    chars_kanji = chars[1]

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    n_samples_mimic: int = typer.Option(
        100,
        "--n-samples-mimic", "-m",
        help="Number of samples to generate for mimic dataset. ",
    ),
    n_samples_random: int = typer.Option(
        100,
        "--n-samples-random", "-r",
        help="Number of samples to generate for random dataset. ",
    ),
    dpath_output: Path = typer.Option(
        "data/synthetic/",
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
    dpath_output = Path(dpath_output)

    dpath_output_random = dpath_output / DNAME_RANDOM
    dpath_output_random.mkdir(parents=True, exist_ok=True)
    n_digits_random = len(str(n_samples_random))
    for i in tqdm(range(1, n_samples_random + 1), desc="Generating random dataset"):
        img, label = synthesize_random()
        fpath_output = dpath_output_random / f"{i:0{n_digits_random}d}.png"
        img.save(fpath_output)
        with open(fpath_output.with_suffix(".txt"), "w") as f:
            f.write(label)

    dpath_output_mimic = dpath_output / DNAME_MIMIC
    dpath_output_mimic.mkdir(parents=True, exist_ok=True)
    n_digits_mimic = len(str(n_samples_mimic))
    for i in tqdm(range(1, n_samples_mimic + 1), desc="Generating mimic dataset"):
        img, label = synthesize_mimic()
        fpath_output = dpath_output_mimic / f"{i:0{n_digits_mimic}d}.png"
        img.save(fpath_output)
        with open(fpath_output.with_suffix(".txt"), "w") as f:
            f.write(label)


def get_random_color(base=None):
    if base:
        return (
            min(255, max(0, base[0] + random.randint(-50, 50))),
            min(255, max(0, base[1] + random.randint(-50, 50))),
            min(255, max(0, base[2] + random.randint(-50, 50))),
        )
    else:
        return (
            random.randint(30, 255),
            random.randint(30, 255),
            random.randint(30, 255),
        )


def init_image(n_shape=10, black_line=True):
    base = get_random_color()
    img = Image.new("RGB", (W, H), base)
    draw = ImageDraw.Draw(img)

    for _ in range(random.randint(0, n_shape)):
        x1 = random.randint(0, W)
        y1 = random.randint(0, H)
        x2 = random.randint(x1, W)
        y2 = random.randint(y1, H)
        color = get_random_color(base)
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)

        x = random.randint(0, W)
        y = random.randint(0, H)
        r1 = random.randint(10, W // 4)
        r2 = random.randint(10, H // 4)
        color = get_random_color(base)
        draw.ellipse([x-r1, y-r2, x+r1, y+r2], fill=color, outline=None)

        x1 = random.randint(0, W)
        y1 = random.randint(0, H)
        x2 = random.randint(0, W)
        y2 = random.randint(0, H)
        color = get_random_color(base)
        draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 5))

    if black_line:
        for _ in range(10):
            y = random.randint(80, 130)
            h = random.randint(1, 5)
            draw.rectangle([0, y, W, y+h], fill="black", outline=None)

    return img


def insert_text(draw, pos, text, size, fill, stroke_width, random_offset):
    pos = (
        pos[0] + random.randint(-random_offset, random_offset),
        pos[1] + random.randint(-random_offset, random_offset),
    )
    size = size + random.randint(-2, 2)
    draw.text(
        pos,
        text,
        font=ImageFont.truetype(FPATH_FONT, size=size),
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill="black",
    )


def generate_label_random():
    seqs = []
    for _ in range(5):
        chars = []
        for _ in range(random.randint(10, 15)):
            if random.random() < 0.3:
                chars.append(random.choice(chars_kanji))
            else:
                chars.append(random.choice(chars_kana))
        seqs.append("".join(chars))
    return seqs


def generate_label_mimic():
    song = random.choice(songs)
    progress = random.randint(0, 100)
    score = random.randint(0, 1500000)
    ryo = random.randint(0, 1500)
    ka = random.randint(0, 1500)
    fuka = random.randint(0, 1500)
    combo = random.randint(0, 1500)
    renda = random.randint(0, 1500)
    return {
        "曲名": song,
        "進捗率": progress,
        "スコア": score,
        "良": ryo,
        "可": ka,
        "不可": fuka,
        "最大コンボ数": combo,
        "連打数": renda
    }


def synthesize_random():
    img = init_image(black_line=False)
    draw = ImageDraw.Draw(img)
    label = generate_label_random()

    for i, text in enumerate(label):
        x = random.randint(20, 100)
        y = (H // 5) * i + 10
        size = random.randint(30, 40)
        insert_text(
            draw,
            (x, y),
            text,
            size,
            "white",
            stroke_width=2,
            random_offset=10
        )
    label = "\n".join(label)
    return img, label


def synthesize_mimic():
    img = init_image()
    draw = ImageDraw.Draw(img)
    label = generate_label_mimic()

    font_size = random.randint(20, 40)
    while True:
        font = ImageFont.truetype(FPATH_FONT, size=font_size)
        text_width, _ = draw.textbbox((0, 0), label["曲名"], font=font)[2:]
        if text_width < W - 100:
            break
        font_size -= 2

    insert_text(draw, (W - text_width - 25, 25), label["曲名"], font_size, "white", 4, 3)

    x = random.randint(50, W - 200)
    insert_text(draw, (x, 100), f"{label['進捗率']}%", 17, "yellow", 3, 3)

    insert_text(draw, (80, 160), "スコア", 20, "white", 3, 3)
    insert_text(draw, (60, 200), str(label["スコア"]), 36, "white", 3, 3)

    color = get_random_color((240, 137, 22))
    insert_text(draw, (300, 160), "良", 26, color, 3, 3)
    insert_text(draw, (400, 160), str(label["良"]), 26, "white", 3, 3)

    insert_text(draw, (300, 200), "可", 26, "white", 3, 3)
    insert_text(draw, (400, 200), str(label["可"]), 26, "white", 3, 3)

    color = get_random_color((40, 42, 231))
    insert_text(draw, (300, 240), "不可", 26, color, 3, 3)
    insert_text(draw, (400, 240), str(label["不可"]), 26, "white", 3, 3)

    insert_text(draw, (520, 180), "最大コンボ数", 20, "black", 0, 3)
    insert_text(draw, (670, 180), str(label["最大コンボ数"]), 26, "white", 3, 3)

    insert_text(draw, (520, 220), "連打数", 20, "black", 0, 3)
    insert_text(draw, (670, 220), str(label["連打数"]), 26, "white", 3, 3)

    label = json.dumps(label, ensure_ascii=False, indent=2)
    return img, label


if __name__ == "__main__":
    app()