"""Lit les JSON de results/ et affiche le tableau comparatif (+ results/comparison.md)."""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import RESULTS_DIR

ORDER = ["custom_cnn", "faster_rcnn", "transfer_alexnet", "transfer_vgg16", "vit_scratch"]


def main():
    rows = []
    for name in ORDER:
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                rows.append(json.load(f))
        else:
            print(f"(absent : {name}, lance son script d'abord)")

    header = "| Modèle | Accuracy | F1 (macro) | Loss | Temps d'entraînement (s) |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        lines.append(f"| {r['model']} | {r['accuracy'] * 100:.2f}% | {r['f1_score']:.4f} | "
                     f"{r['loss']:.4f} | {r['train_time_s']:.1f} |")
    table = "\n".join(lines)
    print(table)

    with open(os.path.join(RESULTS_DIR, "comparison.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()