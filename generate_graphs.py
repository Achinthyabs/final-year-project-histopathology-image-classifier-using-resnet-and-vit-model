import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "graphs"
HISTORY_FILE = BASE_DIR / "prediction_history" / "history.json"
VIT_PREDICTIONS_CSV = (
    BASE_DIR
    / "vit-tiny-colorectal-histopathology-classification"
    / "vit_test_predictions.csv"
)

CNN_TEST_ACCURACY = 92.32
VIT_TEST_ACCURACY = 97.85
CNN_CLASS_NAMES = [
    "ADI",
    "Adenocarcinoma",
    "BACK",
    "DEB",
    "High-grade IN",
    "LYM",
    "Low-grade IN",
    "MUC",
    "MUS",
    "NORM",
    "Polyp",
    "STR",
]
CNN_COLOR = "#6f7cff"
VIT_COLOR = "#2dd4bf"
BACKGROUND = "#111827"
GRID = "#27314a"
TEXT = "#f7f8fc"
MUTED = "#cbd5e1"


def style_axis(axis):
    axis.set_facecolor(BACKGROUND)
    axis.tick_params(colors=MUTED)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(color=GRID, linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)


def style_legend(legend):
    if legend:
        for text in legend.get_texts():
            text.set_color(TEXT)


def save_figure(figure, filename):
    OUTPUT_DIR.mkdir(exist_ok=True)
    figure.savefig(OUTPUT_DIR / filename, dpi=180, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(figure)


def load_history():
    if not HISTORY_FILE.exists():
        return []
    with HISTORY_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def load_vit_predictions():
    if not VIT_PREDICTIONS_CSV.exists():
        return [], []

    true_labels = []
    predicted_labels = []
    with VIT_PREDICTIONS_CSV.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            true_label = row.get("True_Label")
            predicted_label = row.get("Predicted_Label")
            if true_label and predicted_label:
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)

    return true_labels, predicted_labels


def create_accuracy_graph():
    figure, axis = plt.subplots(figsize=(7, 4), facecolor=BACKGROUND)
    model_names = ["ResNet-8 CNN", "ViT-Tiny"]
    accuracies = [CNN_TEST_ACCURACY, VIT_TEST_ACCURACY]
    bars = axis.bar(model_names, accuracies, color=[CNN_COLOR, VIT_COLOR], width=0.55)

    style_axis(axis)
    axis.set_title("Held-out Test Accuracy", color=TEXT, fontsize=15, weight="bold", pad=12)
    axis.set_ylabel("Accuracy (%)", color=MUTED)
    axis.set_ylim(0, 105)
    axis.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)

    for bar, value in zip(bars, accuracies):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.2f}%",
            ha="center",
            color=TEXT,
            fontsize=12,
            weight="bold",
        )

    save_figure(figure, "model_accuracy.png")


def create_vit_metrics_graph(true_labels, predicted_labels):
    if not true_labels:
        return

    labels = sorted(set(true_labels) | set(predicted_labels))
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    precision = [report[label]["precision"] for label in labels]
    recall = [report[label]["recall"] for label in labels]
    f1_scores = [report[label]["f1-score"] for label in labels]

    figure, axis = plt.subplots(figsize=(12, 5), facecolor=BACKGROUND)
    positions = np.arange(len(labels))
    width = 0.25

    axis.bar(positions - width, precision, width, color="#f59e0b", label="Precision")
    axis.bar(positions, recall, width, color=VIT_COLOR, label="Recall")
    axis.bar(positions + width, f1_scores, width, color=CNN_COLOR, label="F1")

    style_axis(axis)
    axis.set_title("ViT-Tiny Class Metrics", color=TEXT, fontsize=15, weight="bold", pad=12)
    axis.set_ylabel("Score", color=MUTED)
    axis.set_ylim(0, 1.05)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=35, ha="right", color=MUTED)
    legend = axis.legend(facecolor="#1b2133", edgecolor=GRID)
    style_legend(legend)

    save_figure(figure, "vit_class_metrics.png")


def create_confusion_matrix_graph(true_labels, predicted_labels):
    if not true_labels:
        return

    labels = sorted(set(true_labels) | set(predicted_labels))
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)

    figure, axis = plt.subplots(figsize=(9, 7), facecolor=BACKGROUND)
    image = axis.imshow(matrix, cmap="magma")
    axis.set_title("ViT-Tiny Confusion Matrix", color=TEXT, fontsize=15, weight="bold", pad=12)
    axis.set_xlabel("Predicted Label", color=MUTED)
    axis.set_ylabel("True Label", color=MUTED)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", color=MUTED)
    axis.set_yticklabels(labels, color=MUTED)
    axis.tick_params(colors=MUTED)
    for spine in axis.spines.values():
        spine.set_visible(False)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="#111827" if value > threshold else TEXT,
                fontsize=8,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=MUTED)
    save_figure(figure, "vit_confusion_matrix.png")


def true_label_from_filename(file_name):
    normalized_name = file_name.replace("_", "-").upper()
    for class_name in CNN_CLASS_NAMES:
        normalized_class = class_name.replace(" ", "-").upper()
        if normalized_name.startswith(normalized_class):
            return class_name
    return None


def create_cnn_confusion_matrix_graph(history):
    true_labels = []
    predicted_labels = []

    for record in history:
        true_label = true_label_from_filename(record.get("file_name", ""))
        predicted_label = record.get("cnn", {}).get("class_name")
        if true_label and predicted_label:
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

    if not true_labels:
        return

    labels = [label for label in CNN_CLASS_NAMES if label in set(true_labels) | set(predicted_labels)]
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)

    figure, axis = plt.subplots(figsize=(9, 7), facecolor=BACKGROUND)
    image = axis.imshow(matrix, cmap="magma")
    axis.set_title("ResNet-8 CNN Confusion Matrix", color=TEXT, fontsize=15, weight="bold", pad=12)
    axis.set_xlabel("Predicted Label", color=MUTED)
    axis.set_ylabel("True Label", color=MUTED)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", color=MUTED)
    axis.set_yticklabels(labels, color=MUTED)
    axis.tick_params(colors=MUTED)
    for spine in axis.spines.values():
        spine.set_visible(False)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="#111827" if value > threshold else TEXT,
                fontsize=8,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=MUTED)
    save_figure(figure, "cnn_confusion_matrix.png")


def create_history_confidence_graph(history):
    if not history:
        return

    recent = history[-20:]
    positions = np.arange(1, len(recent) + 1)
    cnn_values = [record["cnn"]["confidence_percent"] for record in recent]
    vit_values = [record["vit"]["confidence_percent"] for record in recent]

    figure, axis = plt.subplots(figsize=(11, 4.5), facecolor=BACKGROUND)
    axis.plot(positions, cnn_values, "o-", color=CNN_COLOR, label="CNN Confidence")
    axis.plot(positions, vit_values, "s-", color=VIT_COLOR, label="ViT Confidence")

    style_axis(axis)
    axis.set_title("Recent Prediction Confidence", color=TEXT, fontsize=15, weight="bold", pad=12)
    axis.set_xlabel("Recent Upload Number", color=MUTED)
    axis.set_ylabel("Confidence (%)", color=MUTED)
    axis.set_ylim(0, 105)
    axis.set_xticks(positions)
    legend = axis.legend(facecolor="#1b2133", edgecolor=GRID)
    style_legend(legend)

    save_figure(figure, "recent_prediction_confidence.png")


def create_latest_prediction_graph(history):
    if not history:
        return

    latest = history[-1]
    labels = ["Confidence", "Final Score", "Stored Accuracy"]
    cnn_values = [
        latest["cnn"]["confidence_percent"],
        latest["cnn"]["score_percent"],
        CNN_TEST_ACCURACY,
    ]
    vit_values = [
        latest["vit"]["confidence_percent"],
        latest["vit"]["score_percent"],
        VIT_TEST_ACCURACY,
    ]
    positions = np.arange(len(labels))

    figure, axis = plt.subplots(figsize=(8, 4.5), facecolor=BACKGROUND)
    axis.bar(positions - 0.18, cnn_values, 0.36, color=CNN_COLOR, alpha=0.78, label="ResNet-8 CNN")
    axis.bar(positions + 0.18, vit_values, 0.36, color=VIT_COLOR, alpha=0.78, label="ViT-Tiny")

    style_axis(axis)
    axis.set_title(
        f"Latest Prediction: {latest['final']['class_name']}",
        color=TEXT,
        fontsize=15,
        weight="bold",
        pad=12,
    )
    axis.set_ylabel("Percent (%)", color=MUTED)
    axis.set_ylim(0, 105)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, color=MUTED)
    legend = axis.legend(facecolor="#1b2133", edgecolor=GRID)
    style_legend(legend)

    save_figure(figure, "latest_prediction_comparison.png")


def create_vit_class_distribution_graph(true_labels, predicted_labels):
    if not true_labels:
        return

    labels = sorted(set(true_labels) | set(predicted_labels))
    true_counts = [true_labels.count(label) for label in labels]
    predicted_counts = [predicted_labels.count(label) for label in labels]
    positions = np.arange(len(labels))

    figure, axis = plt.subplots(figsize=(12, 5), facecolor=BACKGROUND)
    axis.bar(positions - 0.18, true_counts, 0.36, color="#f59e0b", label="True")
    axis.bar(positions + 0.18, predicted_counts, 0.36, color=VIT_COLOR, label="Predicted")

    style_axis(axis)
    axis.set_title("ViT-Tiny Class Distribution", color=TEXT, fontsize=15, weight="bold", pad=12)
    axis.set_ylabel("Image Count", color=MUTED)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=35, ha="right", color=MUTED)
    style_legend(axis.legend(facecolor="#1b2133", edgecolor=GRID))

    save_figure(figure, "vit_class_distribution.png")


def create_prediction_outcome_graph(history):
    if not history:
        return

    final_models = [record.get("final", {}).get("model", "Unknown") for record in history]
    model_names = ["ResNet-8 CNN", "ViT-Tiny", "Unknown"]
    counts = [final_models.count(model_name) for model_name in model_names]
    model_names = [name for name, count in zip(model_names, counts) if count]
    counts = [count for count in counts if count]

    figure, axis = plt.subplots(figsize=(7, 4.5), facecolor=BACKGROUND)
    wedges, _, autotexts = axis.pie(
        counts,
        labels=model_names,
        autopct="%1.1f%%",
        colors=[CNN_COLOR, VIT_COLOR, "#94a3b8"][: len(counts)],
        textprops={"color": TEXT, "fontsize": 10},
        startangle=90,
    )
    for text in autotexts:
        text.set_color("#111827")
        text.set_weight("bold")
    for wedge in wedges:
        wedge.set_edgecolor(BACKGROUND)

    axis.set_title("Final Model Selection Share", color=TEXT, fontsize=15, weight="bold", pad=12)
    save_figure(figure, "final_model_selection_share.png")


def create_model_agreement_graph(history):
    if not history:
        return

    agree = 0
    disagree = 0
    for record in history:
        cnn_class = record.get("cnn", {}).get("class_name")
        vit_class = record.get("vit", {}).get("class_name")
        if not cnn_class or not vit_class:
            continue
        if cnn_class == vit_class:
            agree += 1
        else:
            disagree += 1

    if agree + disagree == 0:
        return

    figure, axis = plt.subplots(figsize=(7, 4), facecolor=BACKGROUND)
    bars = axis.bar(["Agree", "Disagree"], [agree, disagree], color=[VIT_COLOR, "#f87171"], width=0.55)

    style_axis(axis)
    axis.set_title("CNN vs ViT Prediction Agreement", color=TEXT, fontsize=15, weight="bold", pad=12)
    axis.set_ylabel("Prediction Count", color=MUTED)
    axis.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            str(int(bar.get_height())),
            ha="center",
            color=TEXT,
            fontsize=12,
            weight="bold",
        )

    save_figure(figure, "model_agreement.png")


def create_confidence_gap_graph(history):
    if not history:
        return

    recent = history[-20:]
    positions = np.arange(1, len(recent) + 1)
    gaps = [
        record.get("vit", {}).get("confidence_percent", 0)
        - record.get("cnn", {}).get("confidence_percent", 0)
        for record in recent
    ]
    colors = [VIT_COLOR if gap >= 0 else CNN_COLOR for gap in gaps]

    figure, axis = plt.subplots(figsize=(11, 4.5), facecolor=BACKGROUND)
    axis.bar(positions, gaps, color=colors, width=0.65)
    axis.axhline(0, color=MUTED, linewidth=1)

    style_axis(axis)
    axis.set_title("Recent Confidence Gap", color=TEXT, fontsize=15, weight="bold", pad=12)
    axis.set_xlabel("Recent Upload Number", color=MUTED)
    axis.set_ylabel("ViT - CNN Confidence (%)", color=MUTED)
    axis.set_xticks(positions)

    save_figure(figure, "recent_confidence_gap.png")


def create_history_class_frequency_graph(history):
    labels = []
    for record in history:
        class_name = record.get("final", {}).get("class_name")
        if class_name:
            labels.append(class_name)

    if not labels:
        return

    class_names = [label for label in CNN_CLASS_NAMES if label in set(labels)]
    counts = [labels.count(label) for label in class_names]
    positions = np.arange(len(class_names))

    figure, axis = plt.subplots(figsize=(10, 4.5), facecolor=BACKGROUND)
    bars = axis.bar(positions, counts, color="#f59e0b", width=0.6)

    style_axis(axis)
    axis.set_title("Prediction History Class Frequency", color=TEXT, fontsize=15, weight="bold", pad=12)
    axis.set_ylabel("Final Prediction Count", color=MUTED)
    axis.set_xticks(positions)
    axis.set_xticklabels(class_names, rotation=35, ha="right", color=MUTED)
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(int(bar.get_height())),
            ha="center",
            color=TEXT,
            fontsize=10,
            weight="bold",
        )

    save_figure(figure, "history_class_frequency.png")


def main():
    history = load_history()
    true_labels, predicted_labels = load_vit_predictions()

    create_accuracy_graph()
    create_vit_metrics_graph(true_labels, predicted_labels)
    create_confusion_matrix_graph(true_labels, predicted_labels)
    create_cnn_confusion_matrix_graph(history)
    create_history_confidence_graph(history)
    create_latest_prediction_graph(history)
    create_vit_class_distribution_graph(true_labels, predicted_labels)
    create_prediction_outcome_graph(history)
    create_model_agreement_graph(history)
    create_confidence_gap_graph(history)
    create_history_class_frequency_graph(history)

    print(f"Graphs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
