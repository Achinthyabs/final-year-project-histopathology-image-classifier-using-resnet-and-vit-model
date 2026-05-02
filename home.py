import customtkinter as ctk

from animations import animate_content_in
from ui_helpers import build_model_card, make_card


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

    app.metrics_card = make_card(app.content)
    app.metrics_card.grid(
        row=3,
        column=0,
        columnspan=3,
        sticky="ew",
        pady=(16, 0),
    )
    app.metrics_card.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        app.metrics_card,
        text="Predicted Class Metrics",
        font=ctk.CTkFont(size=20, weight="bold"),
        text_color="#f7f8fc",
    ).pack(anchor="w", padx=22, pady=(18, 4))

    app.metrics_note = ctk.CTkLabel(
        app.metrics_card,
        text="Upload an image to view an overlapping bar graph for the selected class.",
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
    animate_content_in(app)
