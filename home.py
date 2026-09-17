from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageOps

from ui_helpers import build_model_card, make_card


GRAPH_DIR = Path(__file__).resolve().parent / "graphs"
SAVED_ANALYSIS_GRAPHS = (
    ("model_accuracy.png", "Held-out Model Accuracy"),
    ("vit_class_metrics.png", "ViT-Tiny Class Metrics"),
    ("vit_class_distribution.png", "ViT-Tiny Class Distribution"),
    ("vit_confusion_matrix.png", "ViT-Tiny Confusion Matrix"),
    ("cnn_confusion_matrix.png", "ResNet-8 CNN Confusion Matrix"),
    ("latest_prediction_comparison.png", "Latest Prediction Comparison"),
    ("model_agreement.png", "Model Agreement"),
    ("final_model_selection_share.png", "Final Model Selection Share"),
    ("recent_prediction_confidence.png", "Recent Prediction Confidence"),
    ("recent_confidence_gap.png", "Recent Confidence Gap"),
    ("history_class_frequency.png", "Prediction Class Frequency"),
)


def hide_graph_preview(app):
    """Close the enlarged graph preview, if one is open."""
    preview_window = getattr(app, "graph_preview_window", None)
    if preview_window is not None and preview_window.winfo_exists():
        try:
            preview_window.grab_release()
        except Exception:
            pass
        preview_window.destroy()
    app.graph_preview_window = None
    app.graph_preview_image = None


def show_graph_preview(app, graph_path, title, _event=None):
    """Open a readable full-size version of a saved analysis graph."""
    preview_window = getattr(app, "graph_preview_window", None)
    if preview_window is not None and preview_window.winfo_exists():
        hide_graph_preview(app)

    with Image.open(graph_path) as source:
        preview = ImageOps.contain(source.convert("RGB"), (820, 570))

    preview_window = ctk.CTkToplevel(app)
    preview_window.title(title)
    preview_window.transient(app)
    preview_window.configure(fg_color="#111827")

    frame = ctk.CTkFrame(
        preview_window,
        corner_radius=16,
        fg_color="#111827",
        border_color="#7c5cff",
        border_width=2,
    )
    frame.pack(fill="both", expand=True)

    header = ctk.CTkFrame(frame, fg_color="transparent")
    header.pack(fill="x", padx=14, pady=(10, 3))
    header.grid_columnconfigure(0, weight=1)
    ctk.CTkLabel(
        header,
        text=title,
        font=ctk.CTkFont(size=14, weight="bold"),
        text_color="#f7f8fc",
    ).grid(row=0, column=0, sticky="w")
    ctk.CTkButton(
        header,
        text="×",
        width=32,
        height=30,
        corner_radius=10,
        fg_color="#3f1f2a",
        hover_color="#7f1d1d",
        text_color="#fecaca",
        font=ctk.CTkFont(size=20, weight="bold"),
        command=lambda: hide_graph_preview(app),
    ).grid(row=0, column=1, sticky="e")
    rendered = ctk.CTkImage(
        light_image=preview,
        dark_image=preview,
        size=preview.size,
    )
    ctk.CTkLabel(frame, image=rendered, text="").pack(padx=12, pady=(0, 12))

    preview_window.update_idletasks()
    width = preview_window.winfo_reqwidth()
    height = preview_window.winfo_reqheight()
    x_position = max(12, (app.winfo_screenwidth() - width) // 2)
    y_position = max(12, (app.winfo_screenheight() - height) // 2)
    preview_window.geometry(f"{width}x{height}+{x_position}+{y_position}")
    preview_window.protocol("WM_DELETE_WINDOW", lambda: hide_graph_preview(app))
    preview_window.bind("<Escape>", lambda _event: hide_graph_preview(app))
    preview_window.grab_set()

    app.graph_preview_window = preview_window
    app.graph_preview_image = rendered
    app.graph_preview_path = graph_path


def show_home_page(app):
    app.show_sidebar()
    app.clear_content()
    app.content.grid_columnconfigure(0, weight=1, minsize=0)
    app.content.grid_columnconfigure(1, weight=1, minsize=0)
    app.content.grid_columnconfigure(2, weight=1, minsize=0)
    app.content.grid_columnconfigure((0, 1, 2), weight=1)
    app.content.grid_rowconfigure(2, weight=1)

    header = ctk.CTkFrame(app.content, fg_color="transparent")
    header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 18))
    header.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        header,
        text="Histopathology Image Classification",
        font=ctk.CTkFont(size=30, weight="bold"),
        text_color="#f7f8fc",
    ).grid(row=0, column=0, sticky="w")

    ctk.CTkLabel(
        header,
        text="Upload one image and compare predictions from ResNet-8 and ViT-Tiny.",
        font=ctk.CTkFont(size=14),
        text_color="#96a0b5",
    ).grid(row=1, column=0, sticky="w", pady=(6, 0))

    app.upload_card = make_card(app.content)
    app.upload_card.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=(0, 16))
    app.upload_card.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        app.upload_card,
        text="Upload Your Image",
        font=ctk.CTkFont(size=22, weight="bold"),
        text_color="#f7f8fc",
    ).pack(anchor="w", padx=24, pady=(24, 4))

    ctk.CTkLabel(
        app.upload_card,
        text="Choose a JPG, PNG, TIFF, or JPEG image from your computer.",
        font=ctk.CTkFont(size=13),
        text_color="#8e99ae",
        wraplength=290,
        justify="left",
    ).pack(anchor="w", padx=24, pady=(0, 18))

    app.preview_box = ctk.CTkFrame(
        app.upload_card,
        height=310,
        corner_radius=18,
        fg_color="#111827",
        border_color="#263049",
        border_width=1,
    )
    app.preview_box.pack(fill="x", padx=24, pady=(0, 18))
    app.preview_box.pack_propagate(False)

    app.preview_label = ctk.CTkLabel(
        app.preview_box,
        text="No image selected",
        text_color="#7f8aa3",
        font=ctk.CTkFont(size=15, weight="bold"),
    )
    app.preview_label.pack(expand=True)

    app.upload_button = ctk.CTkButton(
        app.upload_card,
        text="Upload Image",
        height=48,
        corner_radius=16,
        fg_color="#7c5cff",
        hover_color="#6b4df0",
        font=ctk.CTkFont(size=15, weight="bold"),
        command=app.select_image,
    )
    app.upload_button.pack(fill="x", padx=24, pady=(0, 12))

    app.file_label = ctk.CTkLabel(
        app.upload_card,
        text="Waiting for user upload",
        text_color="#8e99ae",
        font=ctk.CTkFont(size=12),
    )
    app.file_label.pack(anchor="w", padx=24, pady=(0, 24))

    app.decision_card = make_card(app.content)
    app.decision_card.grid(row=1, column=1, columnspan=2, sticky="nsew")
    app.decision_card.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        app.decision_card,
        text="Final Output",
        font=ctk.CTkFont(size=22, weight="bold"),
        text_color="#f7f8fc",
    ).pack(anchor="w", padx=24, pady=(24, 4))

    app.final_class = ctk.CTkLabel(
        app.decision_card,
        text="Upload an image to begin",
        font=ctk.CTkFont(size=34, weight="bold"),
        text_color="#ffffff",
    )
    app.final_class.pack(anchor="w", padx=24, pady=(8, 0))

    app.final_message = ctk.CTkLabel(
        app.decision_card,
        text="Both models will analyze the uploaded image. The higher confidence x accuracy score is selected.",
        font=ctk.CTkFont(size=14),
        text_color="#9aa6bd",
        wraplength=620,
        justify="left",
    )
    app.final_message.pack(anchor="w", padx=24, pady=(8, 18))

    app.final_score_bar = ctk.CTkProgressBar(
        app.decision_card,
        height=16,
        progress_color="#7c5cff",
        fg_color="#232b40",
    )
    app.final_score_bar.pack(fill="x", padx=24, pady=(0, 24))
    app.final_score_bar.set(0)

    app.cnn_card = make_card(app.content)
    app.cnn_card.grid(row=2, column=1, sticky="nsew", padx=(0, 16), pady=(16, 0))

    app.vit_card = make_card(app.content)
    app.vit_card.grid(row=2, column=2, sticky="nsew", pady=(16, 0))

    app.cnn_widgets = build_model_card(app.cnn_card, "ResNet-8 CNN", "#6f7cff")
    app.vit_widgets = build_model_card(app.vit_card, "ViT-Tiny", "#2dd4bf")

    app.pipeline_card = make_card(app.content)
    app.pipeline_card.grid(
        row=3,
        column=0,
        columnspan=3,
        sticky="ew",
        pady=(16, 0),
    )
    app.pipeline_card.grid_columnconfigure((0, 1, 2, 3), weight=1)

    ctk.CTkLabel(
        app.pipeline_card,
        text="Deep Learning Image Pipeline",
        font=ctk.CTkFont(size=20, weight="bold"),
        text_color="#f7f8fc",
    ).grid(row=0, column=0, columnspan=4, sticky="w", padx=22, pady=(18, 3))

    app.pipeline_note = ctk.CTkLabel(
        app.pipeline_card,
        text=(
            "Upload an image to inspect the 224 × 224 preprocessing pipeline before classification."
        ),
        font=ctk.CTkFont(size=12),
        text_color="#8e99ae",
        justify="left",
    )
    app.pipeline_note.grid(
        row=1, column=0, columnspan=4, sticky="w", padx=22, pady=(0, 12)
    )

    app.pipeline_stage_labels = []
    app.pipeline_image_labels = []
    for column, stage_name in enumerate(
        (
            "1. Image Resizing",
            "2. Color Normalization",
            "3. Histogram Equalization",
            "4. Image Compression",
        )
    ):
        stage = ctk.CTkFrame(
            app.pipeline_card,
            fg_color="#111827",
            corner_radius=12,
            border_color="#27314a",
            border_width=1,
        )
        stage.grid(row=2, column=column, sticky="nsew", padx=(22 if column == 0 else 7, 22 if column == 3 else 7), pady=(0, 18))
        image_label = ctk.CTkLabel(
            stage,
            text="Awaiting upload",
            text_color="#7f8aa3",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=145,
        )
        image_label.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        label = ctk.CTkLabel(
            stage,
            text=stage_name,
            text_color="#d8dee9",
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        label.pack(pady=(0, 8))
        app.pipeline_image_labels.append(image_label)
        app.pipeline_stage_labels.append(label)

    app.analysis_card = make_card(app.content)
    app.analysis_card.grid(
        row=4,
        column=0,
        columnspan=3,
        sticky="ew",
        pady=(16, 0),
    )
    app.analysis_card.grid_columnconfigure((0, 1, 2), weight=1)

    ctk.CTkLabel(
        app.analysis_card,
        text="Classifier Analysis Charts",
        font=ctk.CTkFont(size=20, weight="bold"),
        text_color="#f7f8fc",
    ).grid(row=0, column=0, columnspan=3, sticky="w", padx=22, pady=(18, 3))

    app.analysis_note = ctk.CTkLabel(
        app.analysis_card,
        text="Upload an image to compare both classifiers across probability, ranking, and accuracy.",
        font=ctk.CTkFont(size=12),
        text_color="#8e99ae",
        justify="left",
    )
    app.analysis_note.grid(
        row=1, column=0, columnspan=3, sticky="w", padx=22, pady=(0, 10)
    )

    app.analysis_chart_labels = []
    for column, chart_name in enumerate(
        ("Prediction heatmap", "Confidence curve", "Model accuracy")
    ):
        chart_box = ctk.CTkFrame(
            app.analysis_card,
            height=230,
            corner_radius=12,
            fg_color="#111827",
            border_color="#27314a",
            border_width=1,
        )
        chart_box.grid(
            row=2,
            column=column,
            sticky="nsew",
            padx=(22 if column == 0 else 7, 22 if column == 2 else 7),
            pady=(0, 18),
        )
        chart_box.grid_propagate(False)
        chart_label = ctk.CTkLabel(
            chart_box,
            text=f"{chart_name}\nAppears after prediction",
            text_color="#7f8aa3",
            font=ctk.CTkFont(size=12, weight="bold"),
            justify="center",
        )
        chart_label.pack(fill="both", expand=True, padx=6, pady=6)
        app.analysis_chart_labels.append(chart_label)

    ctk.CTkLabel(
        app.analysis_card,
        text="Saved Dataset and Prediction Graphs",
        font=ctk.CTkFont(size=16, weight="bold"),
        text_color="#d8dee9",
    ).grid(
        row=3, column=0, columnspan=3, sticky="w", padx=22, pady=(2, 3)
    )

    ctk.CTkLabel(
        app.analysis_card,
        text="Performance, class-distribution, confusion-matrix, and prediction-history summaries generated in the graphs folder.",
        font=ctk.CTkFont(size=12),
        text_color="#8e99ae",
        justify="left",
        wraplength=900,
    ).grid(row=4, column=0, columnspan=3, sticky="w", padx=22, pady=(0, 10))

    # Keep CTkImage references on the app; otherwise Tk can discard the images.
    app.saved_analysis_images = []
    for index, (filename, title) in enumerate(SAVED_ANALYSIS_GRAPHS):
        column = index % 3
        row = 5 + index // 3
        graph_box = ctk.CTkFrame(
            app.analysis_card,
            height=235,
            corner_radius=12,
            fg_color="#111827",
            border_color="#27314a",
            border_width=1,
        )
        graph_box.grid(
            row=row,
            column=column,
            sticky="nsew",
            padx=(22 if column == 0 else 7, 22 if column == 2 else 7),
            pady=(0, 14 if index < len(SAVED_ANALYSIS_GRAPHS) - 1 else 18),
        )
        graph_box.grid_propagate(False)

        graph_path = GRAPH_DIR / filename
        if graph_path.exists():
            with Image.open(graph_path) as source:
                preview = ImageOps.contain(source.convert("RGB"), (330, 185))
            rendered = ctk.CTkImage(
                light_image=preview,
                dark_image=preview,
                size=preview.size,
            )
            app.saved_analysis_images.append(rendered)
            image_label = ctk.CTkLabel(graph_box, image=rendered, text="")
            image_label.pack(
                fill="both", expand=True, padx=6, pady=(6, 0)
            )
        else:
            ctk.CTkLabel(
                graph_box,
                text="Graph unavailable",
                text_color="#7f8aa3",
                font=ctk.CTkFont(size=12, weight="bold"),
            ).pack(fill="both", expand=True, padx=6, pady=(6, 0))

        title_label = ctk.CTkLabel(
            graph_box,
            text=title,
            text_color="#d8dee9",
            font=ctk.CTkFont(size=11, weight="bold"),
            wraplength=310,
        )
        title_label.pack(padx=8, pady=(2, 9))

        if graph_path.exists():
            for widget in (image_label, title_label):
                widget.configure(cursor="hand2")
                widget.bind(
                    "<Button-1>",
                    lambda event, path=graph_path, graph_title=title: show_graph_preview(
                        app, path, graph_title, event
                    ),
                )

    app.metrics_card = make_card(app.content)
    app.metrics_card.grid(
        row=5,
        column=0,
        columnspan=3,
        sticky="ew",
        pady=(16, 0),
    )
    app.metrics_card.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        app.metrics_card,
        text="Model Performance Comparison",
        font=ctk.CTkFont(size=20, weight="bold"),
        text_color="#f7f8fc",
    ).pack(anchor="w", padx=22, pady=(18, 4))

    app.metrics_note = ctk.CTkLabel(
        app.metrics_card,
        text="Upload an image to compare precision, recall, F1 score, confidence, and accuracy for both models.",
        font=ctk.CTkFont(size=12),
        text_color="#8e99ae",
        wraplength=900,
        justify="left",
    )
    app.metrics_note.pack(anchor="w", padx=22, pady=(0, 10))

    app.chart_frame = ctk.CTkFrame(
        app.metrics_card,
        height=230,
        corner_radius=16,
        fg_color="#111827",
        border_color="#27314a",
        border_width=1,
    )
    app.chart_frame.pack(fill="x", padx=22, pady=(0, 18))
    app.chart_frame.pack_propagate(False)
    app.draw_metrics_chart()

    app.thermal_card = make_card(app.content)
    app.thermal_card.grid(
        row=6,
        column=0,
        columnspan=3,
        sticky="ew",
        pady=(16, 0),
    )
    app.thermal_card.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        app.thermal_card,
        text="Winning Model Thermal Detection",
        font=ctk.CTkFont(size=20, weight="bold"),
        text_color="#f7f8fc",
    ).pack(anchor="w", padx=22, pady=(18, 4))

    app.thermal_note = ctk.CTkLabel(
        app.thermal_card,
        text="Upload an image to generate a thermal map from the final winning model.",
        text_color="#8e99ae",
        font=ctk.CTkFont(size=12),
        wraplength=900,
        justify="left",
    )
    app.thermal_note.pack(anchor="w", padx=22, pady=(0, 10))

    app.thermal_box = ctk.CTkFrame(
        app.thermal_card,
        height=330,
        corner_radius=16,
        fg_color="#111827",
        border_color="#27314a",
        border_width=1,
    )
    app.thermal_box.pack(fill="x", padx=22, pady=(0, 18))
    app.thermal_box.pack_propagate(False)

    app.thermal_label = ctk.CTkLabel(
        app.thermal_box,
        text="Thermal map appears after prediction",
        text_color="#7f8aa3",
        font=ctk.CTkFont(size=14, weight="bold"),
        wraplength=520,
        justify="center",
    )
    app.thermal_label.pack(expand=True)
