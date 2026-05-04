import json
import shutil
from io import BytesIO
from datetime import datetime
from pathlib import Path
from threading import Thread
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib

matplotlib.use("TkAgg")

import numpy as np
import tensorflow as tf
import timm
import torch
from matplotlib.figure import Figure
from PIL import Image, ImageOps
from sklearn.metrics import classification_report
from torchvision import transforms

from about import show_about_page as render_about_page
from front import show_front_page as render_front_page
from history import show_history_page as render_history_page
from home import show_home_page as render_home_page
from thermal_detection import create_thermal_overlay


BASE_DIR = Path(__file__).resolve().parent
CNN_MODEL_PATH = BASE_DIR / "resnet8_12class_model.keras"
VIT_MODEL_PATH = (
    BASE_DIR
    / "vit-tiny-colorectal-histopathology-classification"
    / "best_vit_tiny.pth"
)
VIT_PREDICTIONS_CSV = (
    BASE_DIR
    / "vit-tiny-colorectal-histopathology-classification"
    / "vit_test_predictions.csv"
)
HISTORY_DIR = BASE_DIR / "prediction_history"
HISTORY_FILE = HISTORY_DIR / "history.json"

IMAGE_SIZE = (224, 224)
DEVICE = "cpu"
CNN_TEST_ACCURACY = 92.32
VIT_TEST_ACCURACY = 97.85
MIN_MODEL_CONFIDENCE = 0.60
CNN_DISAGREEMENT_CONFIDENCE_THRESHOLD = 0.90
DISAGREEMENT_SCORE_MARGIN = 0.05

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

VIT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def preprocess_for_cnn(image):
    resized = image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def ranked_predictions(probabilities, class_names):
    top_indices = np.argsort(probabilities)[::-1]
    return [
        {
            "class_name": class_names[index],
            "confidence": float(probabilities[index]),
        }
        for index in top_indices
    ]


def select_final_prediction(cnn_best, vit_best, cnn_score, vit_score):
    cnn_confidence = cnn_best["confidence"]
    vit_confidence = vit_best["confidence"]
    cnn_is_reliable = cnn_confidence >= MIN_MODEL_CONFIDENCE
    vit_is_reliable = vit_confidence >= MIN_MODEL_CONFIDENCE
    models_agree = cnn_best["class_name"] == vit_best["class_name"]

    if models_agree:
        if vit_score > cnn_score:
            return (
                "ViT-Tiny",
                vit_best,
                vit_score,
                "Both models agree on the class; ViT has the stronger adjusted score.",
            )
        return (
            "ResNet-8 CNN",
            cnn_best,
            cnn_score,
            "Both models agree on the class; CNN has the stronger adjusted score.",
        )

    if vit_is_reliable and not cnn_is_reliable:
        return (
            "ViT-Tiny",
            vit_best,
            vit_score,
            f"Models disagree. CNN confidence is below the {MIN_MODEL_CONFIDENCE * 100:.0f}% reliability threshold, so ViT is selected.",
        )

    if cnn_is_reliable and not vit_is_reliable:
        return (
            "ResNet-8 CNN",
            cnn_best,
            cnn_score,
            f"Models disagree. ViT confidence is below the {MIN_MODEL_CONFIDENCE * 100:.0f}% reliability threshold, so CNN is selected.",
        )

    if vit_is_reliable and cnn_score > vit_score:
        score_gap = cnn_score - vit_score
        cnn_can_override_vit = (
            cnn_confidence >= CNN_DISAGREEMENT_CONFIDENCE_THRESHOLD
            and score_gap >= DISAGREEMENT_SCORE_MARGIN
        )
        if not cnn_can_override_vit:
            return (
                "ViT-Tiny",
                vit_best,
                vit_score,
                (
                    "Models disagree. CNN did not pass the stricter disagreement rule "
                    f"({CNN_DISAGREEMENT_CONFIDENCE_THRESHOLD * 100:.0f}% confidence and "
                    f"{DISAGREEMENT_SCORE_MARGIN * 100:.0f}% score margin), so ViT is selected."
                ),
            )

    if vit_score > cnn_score:
        return (
            "ViT-Tiny",
            vit_best,
            vit_score,
            "Models disagree; ViT has the stronger adjusted score.",
        )

    return (
        "ResNet-8 CNN",
        cnn_best,
        cnn_score,
        "Models disagree; CNN passed the threshold rule and has the stronger adjusted score.",
    )


def load_metrics_from_prediction_csv(csv_path):
    if not csv_path.exists():
        return {}

    true_labels = []
    predicted_labels = []

    with csv_path.open("r", encoding="utf-8") as file:
        header = file.readline().strip().split(",")
        try:
            true_index = header.index("True_Label")
            predicted_index = header.index("Predicted_Label")
        except ValueError:
            return {}

        for line in file:
            values = line.strip().split(",")
            if len(values) <= max(true_index, predicted_index):
                continue
            true_labels.append(values[true_index])
            predicted_labels.append(values[predicted_index])

    if not true_labels:
        return {}

    labels = sorted(set(true_labels) | set(predicted_labels))
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
        }
        for label in labels
        if label in report
    }


def load_prediction_history():
    if not HISTORY_FILE.exists():
        return []

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []

    return data if isinstance(data, list) else []


def save_prediction_history(history):
    HISTORY_DIR.mkdir(exist_ok=True)
    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def load_vit_model():
    checkpoint = torch.load(VIT_MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["classes"]

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=len(class_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


class ModelComparisonApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Histopathology Model Comparison")
        self.geometry("1200x760")
        self.minsize(1080, 680)
        self.configure(fg_color="#151927")

        self.cnn_model = None
        self.vit_model = None
        self.vit_class_names = []
        self.selected_image = None
        self.selected_image_path = None
        self.preview_image = None
        self.thermal_image = None
        self.chart_canvas = None
        self.chart_image = None
        self.chart_image_label = None
        self.sidebar_pie_canvas = None
        self.sidebar_pie_image = None
        self.sidebar_pie_label = None
        self.content_animation_job = None
        self.sidebar_width = 230
        self.sidebar_visible = True
        self.history_images = []
        self.about_images = []
        self.vit_class_metrics = load_metrics_from_prediction_csv(VIT_PREDICTIONS_CSV)
        self.prediction_history = load_prediction_history()

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self.build_top_nav()
        self.build_sidebar()
        self.build_main_area()
        self.load_models_async()

    def build_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self,
            width=self.sidebar_width,
            corner_radius=0,
            fg_color="#0d1220",
        )
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        self.model_status = ctk.CTkLabel(
            self.sidebar,
            text="Loading models...",
            font=ctk.CTkFont(size=13),
            text_color="#f4b860",
            justify="left",
        )
        self.model_status.pack(padx=24, pady=(30, 12), anchor="w")

        self.sidebar_info = ctk.CTkLabel(
            self.sidebar,
            text=(
                f"ResNet-8 accuracy\n{CNN_TEST_ACCURACY:.2f}%\n\n"
                f"ViT-Tiny accuracy\n{VIT_TEST_ACCURACY:.2f}%\n\n"
                f"Min confidence\n{MIN_MODEL_CONFIDENCE * 100:.0f}%\n\n"
                f"CNN override rule\n"
                f"{CNN_DISAGREEMENT_CONFIDENCE_THRESHOLD * 100:.0f}% confidence + "
                f"{DISAGREEMENT_SCORE_MARGIN * 100:.0f}% margin"
            ),
            font=ctk.CTkFont(size=13),
            text_color="#9aa6bd",
            justify="left",
        )
        self.sidebar_info.pack(padx=24, pady=8, anchor="w")

        ctk.CTkFrame(self.sidebar, height=1, fg_color="#222a3d").pack(
            fill="x", padx=22, pady=20
        )

        ctk.CTkLabel(
            self.sidebar,
            text="Uploaded Image",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#f7f8fc",
        ).pack(padx=24, pady=(0, 8), anchor="w")

        self.sidebar_details_label = ctk.CTkLabel(
            self.sidebar,
            text="No image uploaded yet.",
            font=ctk.CTkFont(size=12),
            text_color="#9aa6bd",
            justify="left",
            wraplength=175,
        )
        self.sidebar_details_label.pack(padx=24, pady=(0, 14), anchor="w")

        self.sidebar_pie_frame = ctk.CTkFrame(
            self.sidebar,
            height=170,
            corner_radius=18,
            fg_color="#111827",
            border_color="#27314a",
            border_width=1,
        )
        self.sidebar_pie_frame.pack(fill="x", padx=18, pady=(0, 14))
        self.sidebar_pie_frame.pack_propagate(False)

        self.sidebar_prediction_label = ctk.CTkLabel(
            self.sidebar,
            text="Prediction score share will appear here.",
            font=ctk.CTkFont(size=12),
            text_color="#9aa6bd",
            justify="left",
            wraplength=175,
        )
        self.sidebar_prediction_label.pack(padx=24, pady=(0, 14), anchor="w")
        self.draw_sidebar_pie()

    def add_nav_item(self, text, active=False):
        frame_color = "#7c5cff" if active else "#111827"
        text_color = "#ffffff" if active else "#9aa6bd"
        item = ctk.CTkLabel(
            self.sidebar,
            text=text,
            height=42,
            corner_radius=12,
            fg_color=frame_color,
            text_color=text_color,
            font=ctk.CTkFont(size=14, weight="bold" if active else "normal"),
        )
        item.pack(fill="x", padx=20, pady=6)

    def update_sidebar_image_info(self, file_path, image):
        file_size_kb = file_path.stat().st_size / 1024
        width, height = image.size
        aspect_ratio = width / height if height else 0

        self.sidebar_details_label.configure(
            text=(
                f"File: {file_path.name}\n"
                f"Size: {file_size_kb:.1f} KB\n"
                f"Dimensions: {width} x {height}\n"
                f"Aspect ratio: {aspect_ratio:.2f}"
            )
        )
        self.sidebar_prediction_label.configure(
            text="Image loaded. Waiting for model predictions."
        )
        self.draw_sidebar_pie()

    def update_sidebar_prediction_summary(
        self,
        cnn_best,
        vit_best,
        cnn_score,
        vit_score,
        winner_name,
        winner_result,
    ):
        agreement = "Yes" if cnn_best["class_name"] == vit_best["class_name"] else "No"
        self.sidebar_prediction_label.configure(
            text=(
                f"Final: {winner_result['class_name']}\n"
                f"Winner: {winner_name}\n"
                f"Agreement: {agreement}\n"
                f"CNN score: {cnn_score * 100:.2f}%\n"
                f"ViT score: {vit_score * 100:.2f}%"
            )
        )
        self.draw_sidebar_pie(cnn_score, vit_score)

    def draw_sidebar_pie(self, cnn_score=0, vit_score=0):
        if self.sidebar_pie_label is not None and self.sidebar_pie_label.winfo_exists():
            self.sidebar_pie_label.destroy()
            self.sidebar_pie_label = None

        figure = Figure(figsize=(1.8, 1.55), dpi=100, facecolor="#111827")
        axis = figure.add_subplot(111)
        axis.set_facecolor("#111827")

        values = [cnn_score, vit_score]
        if sum(values) <= 0:
            values = [1]
            colors = ["#27314a"]
            labels = ["No prediction"]
        else:
            colors = ["#6f7cff", "#2dd4bf"]
            labels = ["CNN", "ViT"]

        wedges, _ = axis.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.42, "edgecolor": "#111827"},
        )
        axis.legend(
            wedges,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=2,
            frameon=False,
            fontsize=7,
            labelcolor="#cbd5e1",
        )
        axis.set_aspect("equal")
        figure.tight_layout(pad=0.25)

        buffer = BytesIO()
        figure.savefig(buffer, format="png", facecolor="#111827", dpi=100)
        buffer.seek(0)
        image = Image.open(buffer).copy()
        self.sidebar_pie_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(180, 155),
        )
        self.sidebar_pie_label = ctk.CTkLabel(
            self.sidebar_pie_frame,
            image=self.sidebar_pie_image,
            text="",
        )
        self.sidebar_pie_label.pack(fill="both", expand=True)

    def build_main_area(self):
        self.main = ctk.CTkFrame(self, fg_color="#151927", corner_radius=0)
        self.main.grid(row=1, column=1, sticky="nsew", padx=22, pady=(0, 22))
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(0, weight=1)

        self.content = ctk.CTkScrollableFrame(
            self.main,
            fg_color="transparent",
            scrollbar_button_color="#2a344d",
            scrollbar_button_hover_color="#3a4663",
        )
        self.content.grid(row=0, column=0, sticky="nsew")
        self.content.grid_columnconfigure((0, 1, 2), weight=1)
        self.content.grid_rowconfigure(2, weight=1)
        self.content.grid_rowconfigure(3, weight=0)

        self.show_front_page()

    def build_top_nav(self):
        nav = ctk.CTkFrame(
            self,
            height=64,
            corner_radius=0,
            fg_color="#101624",
            border_color="#232b40",
            border_width=1,
        )
        nav.grid(row=0, column=0, columnspan=2, sticky="ew")
        nav.grid_columnconfigure(5, weight=1)
        nav.grid_propagate(False)

        logo = ctk.CTkButton(
            nav,
            text="H",
            width=40,
            height=40,
            corner_radius=14,
            fg_color="#7c5cff",
            hover_color="#6d4df2",
            text_color="#ffffff",
            border_color="#9d8cff",
            border_width=1,
            font=ctk.CTkFont(size=22, weight="bold"),
            command=self.show_front_page,
        )
        logo.grid(row=0, column=0, padx=(16, 8), pady=12)

        home_button = ctk.CTkButton(
            nav,
            text="Home",
            width=96,
            height=36,
            corner_radius=12,
            fg_color="transparent",
            hover_color="#1b2133",
            text_color="#ffffff",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.show_home_page,
        )
        home_button.grid(row=0, column=1, padx=6, pady=14)

        about_button = ctk.CTkButton(
            nav,
            text="About",
            width=96,
            height=36,
            corner_radius=12,
            fg_color="transparent",
            hover_color="#1b2133",
            text_color="#cbd5e1",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.show_about_page,
        )
        about_button.grid(row=0, column=2, padx=6, pady=14)

        ctk.CTkButton(
            nav,
            text="History",
            width=104,
            height=36,
            corner_radius=12,
            fg_color="transparent",
            hover_color="#1b2133",
            text_color="#cbd5e1",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.show_history_page,
        ).grid(row=0, column=3, padx=6, pady=14)

        ctk.CTkButton(
            nav,
            text="Refresh",
            width=104,
            height=36,
            corner_radius=12,
            fg_color="transparent",
            hover_color="#1b2133",
            text_color="#cbd5e1",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.refresh_current_page,
        ).grid(row=0, column=4, padx=6, pady=14)

        ctk.CTkLabel(
            nav,
            text="HistoScan",
            text_color="#8f9ab0",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=5, sticky="e", padx=(8, 18))

    def show_sidebar(self):
        self.sidebar_visible = True
        self.grid_columnconfigure(0, weight=0, minsize=self.sidebar_width)
        self.sidebar.configure(width=self.sidebar_width)
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.main.grid_configure(column=1, columnspan=1, padx=22, pady=(0, 22))

    def hide_sidebar(self):
        self.sidebar_visible = False
        self.sidebar.grid_remove()
        self.grid_columnconfigure(0, weight=0, minsize=0)
        self.main.grid_configure(column=0, columnspan=2, padx=22, pady=(0, 22))

    def animate_content_in(self):
        pass

    def clear_content(self):
        if self.content_animation_job is not None:
            self.after_cancel(self.content_animation_job)
            self.content_animation_job = None
        self.chart_canvas = None
        self.chart_image = None
        self.chart_image_label = None
        for widget in self.content.winfo_children():
            widget.destroy()

    def show_home_page(self):
        self.current_page = "home"
        render_home_page(self)

    def show_front_page(self):
        self.current_page = "front"
        render_front_page(self)

    def show_about_page(self):
        self.current_page = "about"
        render_about_page(self)

    def show_history_page(self):
        self.current_page = "history"
        render_history_page(self)

    def refresh_current_page(self):
        current_page = getattr(self, "current_page", "home")

        if current_page == "history":
            self.prediction_history = load_prediction_history()
            if hasattr(self, "history_visible_count"):
                del self.history_visible_count
            self.show_history_page()
            return

        if current_page == "about":
            self.show_about_page()
            return

        if current_page == "front":
            self.show_front_page()
            return

        self.selected_image = None
        self.selected_image_path = None
        self.preview_image = None
        self.thermal_image = None
        if hasattr(self, "sidebar_details_label"):
            self.sidebar_details_label.configure(text="No image uploaded yet.")
        if hasattr(self, "sidebar_prediction_label"):
            self.sidebar_prediction_label.configure(
                text="Prediction score share will appear here."
            )
        self.draw_sidebar_pie()
        self.show_home_page()

    def load_models_async(self):
        Thread(target=self.load_models, daemon=True).start()

    def load_models(self):
        try:
            if not CNN_MODEL_PATH.exists():
                raise FileNotFoundError(f"Missing file: {CNN_MODEL_PATH}")
            if not VIT_MODEL_PATH.exists():
                raise FileNotFoundError(f"Missing file: {VIT_MODEL_PATH}")

            cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
            vit_model, vit_class_names = load_vit_model()

            self.after(
                0,
                lambda: self.finish_model_load(cnn_model, vit_model, vit_class_names),
            )
        except Exception as exc:
            self.after(0, lambda: self.show_model_error(exc))

    def finish_model_load(self, cnn_model, vit_model, vit_class_names):
        self.cnn_model = cnn_model
        self.vit_model = vit_model
        self.vit_class_names = vit_class_names
        self.model_status.configure(text="Models ready", text_color="#2dd4bf")
        if hasattr(self, "upload_button") and self.upload_button.winfo_exists():
            self.upload_button.configure(state="normal")

    def show_model_error(self, exc):
        self.model_status.configure(text="Model loading failed", text_color="#f87171")
        messagebox.showerror("Model loading failed", str(exc))

    def select_image(self):
        if self.cnn_model is None or self.vit_model is None:
            messagebox.showinfo("Please wait", "Models are still loading.")
            return

        file_path = filedialog.askopenfilename(
            title="Upload your histopathology image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Invalid image", str(exc))
            return

        self.selected_image = image
        self.selected_image_path = Path(file_path)
        self.file_label.configure(text=Path(file_path).name)
        self.update_sidebar_image_info(self.selected_image_path, image)
        self.show_preview(image)
        self.set_busy_state()
        Thread(target=self.run_prediction, args=(image,), daemon=True).start()

    def show_preview(self, image):
        preview = ImageOps.contain(image.copy(), (330, 285))
        self.preview_image = ctk.CTkImage(
            light_image=preview,
            dark_image=preview,
            size=preview.size,
        )
        self.preview_label.configure(image=self.preview_image, text="")

    def set_busy_state(self):
        self.final_class.configure(text="Analyzing image...")
        self.final_message.configure(
            text=(
                "ResNet-8 and ViT-Tiny are processing the uploaded image. "
                "The winning model thermal map will be generated after prediction."
            )
        )
        self.final_score_bar.set(0)
        if hasattr(self, "thermal_label") and self.thermal_label.winfo_exists():
            self.thermal_label.configure(
                image=None,
                text="Generating after model decision...",
            )
        if hasattr(self, "thermal_note") and self.thermal_note.winfo_exists():
            self.thermal_note.configure(text="Uses the final winning model.")
        self.upload_button.configure(state="disabled", text="Processing...")

    def run_prediction(self, image):
        try:
            cnn_results = self.predict_cnn(image)
            vit_results = self.predict_vit(image)
            thermal_image = self.generate_winning_model_thermal_image(
                image,
                cnn_results,
                vit_results,
            )
            self.after(
                0,
                lambda: self.display_results(cnn_results, vit_results, thermal_image),
            )
        except Exception as exc:
            self.after(0, lambda: self.show_prediction_error(exc))

    def predict_cnn(self, image):
        probabilities = self.cnn_model.predict(preprocess_for_cnn(image), verbose=0)[0]
        return ranked_predictions(probabilities, CNN_CLASS_NAMES)

    @torch.no_grad()
    def predict_vit(self, image):
        tensor = VIT_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        outputs = self.vit_model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        return ranked_predictions(probabilities, self.vit_class_names)

    def generate_winning_model_thermal_image(self, image, cnn_results, vit_results):
        cnn_best = cnn_results[0]
        vit_best = vit_results[0]
        cnn_score = cnn_best["confidence"] * (CNN_TEST_ACCURACY / 100.0)
        vit_score = vit_best["confidence"] * (VIT_TEST_ACCURACY / 100.0)
        winner_name, winner_result, _, _ = (
            select_final_prediction(cnn_best, vit_best, cnn_score, vit_score)
        )
        return create_thermal_overlay(
            image=image,
            model_name=winner_name,
            winner_result=winner_result,
            cnn_model=self.cnn_model,
            vit_model=self.vit_model,
            vit_class_names=self.vit_class_names,
            cnn_class_names=CNN_CLASS_NAMES,
            vit_transform=VIT_TRANSFORM,
            preprocess_for_cnn=preprocess_for_cnn,
            image_size=IMAGE_SIZE,
            device=DEVICE,
        )

    def update_thermal_image(self, thermal_result):
        if (
            thermal_result is None
            or not hasattr(self, "thermal_label")
            or not self.thermal_label.winfo_exists()
        ):
            return

        overlay, model_name, class_name = thermal_result
        preview = ImageOps.contain(overlay, (520, 300))
        self.thermal_image = ctk.CTkImage(
            light_image=preview,
            dark_image=preview,
            size=preview.size,
        )
        self.thermal_label.configure(image=self.thermal_image, text="")
        self.thermal_note.configure(
            text=f"Generated using {model_name} for class {class_name}."
        )

    def display_results(self, cnn_results, vit_results, thermal_image=None):
        cnn_best = cnn_results[0]
        vit_best = vit_results[0]

        cnn_score = cnn_best["confidence"] * (CNN_TEST_ACCURACY / 100.0)
        vit_score = vit_best["confidence"] * (VIT_TEST_ACCURACY / 100.0)

        winner_name, winner_result, winner_score, decision_note = (
            select_final_prediction(cnn_best, vit_best, cnn_score, vit_score)
        )

        self.final_class.configure(text=winner_result["class_name"])
        self.final_message.configure(
            text=(
                f"Selected from {winner_name}. "
                f"Final score: {winner_score * 100:.2f}%. {decision_note}"
            )
        )
        self.final_score_bar.set(max(0.0, min(1.0, winner_score)))

        self.update_model_card(
            self.cnn_widgets,
            cnn_best,
            cnn_score,
            cnn_results,
            CNN_TEST_ACCURACY,
        )
        self.update_model_card(
            self.vit_widgets,
            vit_best,
            vit_score,
            vit_results,
            VIT_TEST_ACCURACY,
        )

        self.update_metrics_chart(cnn_best, vit_best, cnn_score, vit_score)
        self.update_thermal_image(thermal_image)
        self.update_sidebar_prediction_summary(
            cnn_best,
            vit_best,
            cnn_score,
            vit_score,
            winner_name,
            winner_result,
        )
        self.add_history_record(
            cnn_best,
            vit_best,
            cnn_score,
            vit_score,
            winner_name,
            winner_result,
            winner_score,
        )

        self.upload_button.configure(state="normal", text="Upload Another Image")

    def add_history_record(
        self,
        cnn_best,
        vit_best,
        cnn_score,
        vit_score,
        winner_name,
        winner_result,
        winner_score,
    ):
        HISTORY_DIR.mkdir(exist_ok=True)
        uploaded_at = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        source_path = self.selected_image_path
        original_name = source_path.name if source_path else "uploaded_image.png"
        suffix = source_path.suffix if source_path and source_path.suffix else ".png"
        safe_stem = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in Path(original_name).stem
        )
        saved_image = HISTORY_DIR / f"{timestamp}_{safe_stem}{suffix}"

        try:
            if source_path and source_path.exists():
                shutil.copy2(source_path, saved_image)
            elif self.selected_image is not None:
                self.selected_image.save(saved_image)
        except OSError:
            saved_image = None

        record = {
            "uploaded_at": uploaded_at,
            "file_name": original_name,
            "saved_image": str(saved_image) if saved_image else "",
            "final": {
                "model": winner_name,
                "class_name": winner_result["class_name"],
                "confidence_percent": round(winner_result["confidence"] * 100, 2),
                "score_percent": round(winner_score * 100, 2),
            },
            "cnn": {
                "class_name": cnn_best["class_name"],
                "confidence_percent": round(cnn_best["confidence"] * 100, 2),
                "score_percent": round(cnn_score * 100, 2),
            },
            "vit": {
                "class_name": vit_best["class_name"],
                "confidence_percent": round(vit_best["confidence"] * 100, 2),
                "score_percent": round(vit_score * 100, 2),
            },
        }

        self.prediction_history.append(record)
        save_prediction_history(self.prediction_history)

    def update_model_card(self, widgets, best, score, results, accuracy):
        widgets["class"].configure(text=f"Class: {best['class_name']}")
        widgets["confidence"].configure(
            text=f"Confidence: {best['confidence'] * 100:.2f}%"
        )
        widgets["bar"].set(max(0.0, min(1.0, best["confidence"])))
        widgets["score"].configure(
            text=f"Final score: {score * 100:.2f}%  |  Accuracy: {accuracy:.2f}%"
        )

        top_lines = [
            f"{item['class_name']}: {item['confidence'] * 100:.2f}%"
            for item in results[:4]
        ]
        widgets["top"].configure(text="\n".join(top_lines))

    def metric_values_for_chart(self, best, score, accuracy, metric_lookup=None):
        class_metrics = {}
        if metric_lookup:
            class_metrics = metric_lookup.get(best["class_name"], {})

        precision = class_metrics.get("precision", best["confidence"])
        recall = class_metrics.get("recall", accuracy / 100.0)
        f1_score = class_metrics.get("f1", score)

        return [
            precision,
            recall,
            f1_score,
            best["confidence"],
            accuracy / 100.0,
            score,
        ]

    def update_metrics_chart(self, cnn_best, vit_best, cnn_score, vit_score):
        if not hasattr(self, "chart_frame") or not self.chart_frame.winfo_exists():
            return

        labels = ["Precision", "Recall", "F1", "Confidence", "Accuracy", "Score"]
        cnn_values = self.metric_values_for_chart(
            cnn_best,
            cnn_score,
            CNN_TEST_ACCURACY,
        )
        vit_values = self.metric_values_for_chart(
            vit_best,
            vit_score,
            VIT_TEST_ACCURACY,
            self.vit_class_metrics,
        )

        self.metrics_note.configure(
            text=(
                "Overlapping bars compare both models for their predicted class. "
                "ViT precision/recall/F1 come from vit_test_predictions.csv; CNN uses confidence and stored accuracy where a saved class report is not available."
            )
        )
        self.draw_metrics_chart(labels, cnn_values, vit_values, cnn_best, vit_best)

    def draw_metrics_chart(
        self,
        labels=None,
        cnn_values=None,
        vit_values=None,
        cnn_best=None,
        vit_best=None,
    ):
        if self.chart_image_label is not None and self.chart_image_label.winfo_exists():
            self.chart_image_label.destroy()
            self.chart_image_label = None

        labels = labels or ["Precision", "Recall", "F1", "Confidence", "Accuracy", "Score"]
        cnn_values = cnn_values or [0, 0, 0, 0, CNN_TEST_ACCURACY / 100.0, 0]
        vit_values = vit_values or [0, 0, 0, 0, VIT_TEST_ACCURACY / 100.0, 0]

        figure = Figure(figsize=(9.2, 2.3), dpi=100, facecolor="#111827")
        axis = figure.add_subplot(111)
        axis.set_facecolor("#111827")

        x_positions = np.arange(len(labels))
        axis.bar(
            x_positions,
            cnn_values,
            width=0.62,
            color="#6f7cff",
            alpha=0.74,
            label="ResNet-8 CNN",
        )
        axis.bar(
            x_positions,
            vit_values,
            width=0.36,
            color="#2dd4bf",
            alpha=0.72,
            label="ViT-Tiny",
        )

        axis.set_ylim(0, 1.05)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, color="#cbd5e1", fontsize=9)
        axis.tick_params(axis="y", colors="#8e99ae", labelsize=8)
        axis.grid(axis="y", color="#27314a", linestyle="-", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)

        for spine in axis.spines.values():
            spine.set_visible(False)

        title = "Waiting for uploaded image"
        if cnn_best and vit_best:
            title = (
                f"CNN: {cnn_best['class_name']}   |   "
                f"ViT: {vit_best['class_name']}"
            )

        axis.set_title(title, color="#f7f8fc", fontsize=11, pad=8)
        legend = axis.legend(
            loc="upper right",
            frameon=True,
            facecolor="#1b2133",
            edgecolor="#27314a",
            fontsize=8,
        )
        for text in legend.get_texts():
            text.set_color("#d8dee9")

        figure.tight_layout(pad=1.0)

        buffer = BytesIO()
        figure.savefig(buffer, format="png", facecolor="#111827", dpi=100)
        buffer.seek(0)
        image = Image.open(buffer).copy()
        self.chart_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(920, 230),
        )
        self.chart_image_label = ctk.CTkLabel(
            self.chart_frame,
            image=self.chart_image,
            text="",
        )
        self.chart_image_label.pack(fill="both", expand=True, padx=8, pady=8)

    def show_prediction_error(self, exc):
        self.upload_button.configure(state="normal", text="Upload Image")
        messagebox.showerror("Prediction failed", str(exc))


if __name__ == "__main__":
    app = ModelComparisonApp()
    app.mainloop()
