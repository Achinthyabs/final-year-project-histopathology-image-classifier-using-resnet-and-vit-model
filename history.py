from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageOps

from animations import animate_content_in
from ui_helpers import make_card


def show_history_page(app):
    app.hide_sidebar()
    app.clear_content()
    app.history_images = []
    app.content.grid_columnconfigure(0, weight=1, minsize=0)
    app.content.grid_columnconfigure(1, weight=0, minsize=0)
    app.content.grid_columnconfigure(2, weight=0, minsize=0)
    app.content.grid_rowconfigure(0, weight=1)

    history_card = make_card(app.content)
    history_card.grid(row=0, column=0, sticky="nsew")
    history_card.grid_columnconfigure(0, weight=1)

    top = ctk.CTkFrame(history_card, fg_color="transparent")
    top.pack(fill="x", padx=28, pady=(28, 16))
    top.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        top,
        text="Prediction History",
        font=ctk.CTkFont(size=30, weight="bold"),
        text_color="#ffffff",
    ).grid(row=0, column=0, sticky="w")

    ctk.CTkButton(
        top,
        text="Home",
        width=120,
        height=40,
        corner_radius=14,
        fg_color="#7c5cff",
        hover_color="#6b4df0",
        font=ctk.CTkFont(size=14, weight="bold"),
        command=app.show_home_page,
    ).grid(row=0, column=1, sticky="e")

    ctk.CTkLabel(
        history_card,
        text="Each upload is saved with time, image, both model predictions, and the selected final output.",
        font=ctk.CTkFont(size=13),
        text_color="#9aa6bd",
        wraplength=860,
        justify="left",
    ).pack(anchor="w", padx=28, pady=(0, 18))

    if not app.prediction_history:
        ctk.CTkLabel(
            history_card,
            text="No uploaded images have been predicted yet.",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#8e99ae",
        ).pack(anchor="w", padx=28, pady=20)
        animate_content_in(app)
        return

    for record in reversed(app.prediction_history):
        build_history_record(app, history_card, record)
    animate_content_in(app)


def build_history_record(app, parent, record):
    row = ctk.CTkFrame(
        parent,
        corner_radius=18,
        fg_color="#111827",
        border_color="#27314a",
        border_width=1,
    )
    row.pack(fill="x", padx=28, pady=(0, 14))
    row.grid_columnconfigure(1, weight=1)

    image_path = Path(record.get("saved_image", ""))
    if image_path.exists():
        preview = ImageOps.contain(Image.open(image_path).convert("RGB"), (90, 90))
        photo = ctk.CTkImage(
            light_image=preview,
            dark_image=preview,
            size=preview.size,
        )
        app.history_images.append(photo)
        image_label = ctk.CTkLabel(row, image=photo, text="")
    else:
        image_label = ctk.CTkLabel(
            row,
            text="No image",
            width=90,
            height=90,
            fg_color="#1b2133",
            corner_radius=14,
            text_color="#7f8aa3",
        )
    image_label.grid(row=0, column=0, rowspan=2, padx=16, pady=16)

    final = record.get("final", {})
    cnn = record.get("cnn", {})
    vit = record.get("vit", {})
    uploaded_at = record.get("uploaded_at", "--")
    file_name = record.get("file_name", "Uploaded image")

    ctk.CTkLabel(
        row,
        text=f"{file_name}   |   {uploaded_at}",
        font=ctk.CTkFont(size=13),
        text_color="#8e99ae",
    ).grid(row=0, column=1, sticky="w", padx=(0, 16), pady=(16, 2))

    ctk.CTkLabel(
        row,
        text=(
            f"Final: {final.get('class_name', '--')} "
            f"({final.get('model', '--')}, {final.get('score_percent', '--')}%)"
        ),
        font=ctk.CTkFont(size=18, weight="bold"),
        text_color="#ffffff",
        wraplength=520,
        justify="left",
    ).grid(row=1, column=1, sticky="w", padx=(0, 16), pady=(0, 16))

    details = (
        f"CNN: {cnn.get('class_name', '--')} ({cnn.get('confidence_percent', '--')}%)\n"
        f"ViT: {vit.get('class_name', '--')} ({vit.get('confidence_percent', '--')}%)"
    )
    ctk.CTkLabel(
        row,
        text=details,
        font=ctk.CTkFont(size=13),
        text_color="#b4bfd4",
        justify="left",
    ).grid(row=0, column=2, rowspan=2, sticky="e", padx=16, pady=16)
