import json
from difflib import SequenceMatcher

import typer


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    fpath_input: str = typer.Argument(
        help="File path containing the inference results to evaluate."
    ),
    fpath_annotated: str = typer.Option(
        "data/annotated.json",
        "--annotated", "-a",
        help="File path containing the annotated labels for evaluation."
    ),
    fpath_output: str = typer.Option(
        "eval_results.txt",
        "--output", "-o",
        help="File path to save the evaluation results."
    )
):
    with open(fpath_input) as f:
        preds = json.load(f)
    with open(fpath_annotated) as f:
        annotated = json.load(f)

    name_errors = []
    num_errors = []
    n_matches = 0
    n_matches_name = 0
    n_matches_num = 0
    sum_score_name = 0
    sum_score_num = 0
    for key, pred in preds.items():
        label = annotated[key]
        score_name, score_num = get_score(pred, label)
        if score_name == 1.0 and score_num == 1.0:
            n_matches += 1
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

    n = len(preds)
    rate_match = n_matches / n
    rate_match_name = n_matches_name / n
    rate_match_num = n_matches_num / n
    avg_score_name = sum_score_name / n
    avg_score_num = sum_score_num / n

    lines = [
        f"完全一致率: {rate_match:.2%}",
        f"曲名完全一致率: {rate_match_name:.2%}",
        f"数値完全一致率: {rate_match_num:.2%}",
        f"曲名平均スコア: {avg_score_name:.4f}",
        f"数値平均スコア: {avg_score_num:.4f}",
    ]
    with open(fpath_output, "w") as f:
        f.write("\n".join(lines) + "\n\n")
        f.write(f"評価データ数: {n}\n")
        f.write("\n曲名エラー:\n")
        for key, pred_name, label_name, score in name_errors:
            f.write(f"{key}: '{pred_name}' - '{label_name}' ({score:.4f})\n")
        f.write("\n数値エラー:\n")
        for key, pred, label, score in num_errors:
            f.write(f"{key}: {pred} - {label}\n")

    print("\n".join(lines))
    print(f"Saeved evaluation results to {fpath_output}")


columns = ["良", "可", "不可", "進捗率", "スコア", "最大コンボ数", "連打数"]
def get_score(output, label):
    score_name = SequenceMatcher(None, output["曲名"], label["曲名"]).ratio()
    score_num = sum([output[idx] == label[idx] for idx in columns]) / len(columns)
    return score_name, score_num


if __name__ == "__main__":
    app()
