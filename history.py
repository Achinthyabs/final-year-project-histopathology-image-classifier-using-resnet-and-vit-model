import json
from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageOps

from ui_helpers import make_card


HISTORY_FILE = Path(__file__).resolve().parent / "prediction_history" / "history.json"
HISTORY_PAGE_SIZE = 8
HUMAN_OPINION_CONFIDENCE_THRESHOLD = 85.0
HUMAN_OPINION_SCORE_THRESHOLD = 80.0


def show_history_page(app):
    app.hide_sidebar()
    app.clear_content()
    app.history_images = []
    if not hasattr(app, "history_visible_count"):
        app.history_visible_count = HISTORY_PAGE_SIZE
    if not hasattr(app, "history_category"):
        app.history_category = "opinion"
    if not hasattr(app, "history_thumbnail_cache"):
        app.history_thumbnail_cache = {}
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
        return

    records = list(reversed(app.prediction_history))
    opinion_records, confident_records = categorize_history_records(records)
    selected_records = opinion_records if app.history_category == "opinion" else confident_records
    visible_records = selected_records[: app.history_visible_count]
    selected_title = (
        "Needs Human 2nd Opinion"
        if app.history_category == "opinion"
        else "Confident Class"
    )
    selected_subtitle = (
        "Model disagreement, low confidence, or weak final score."
        if app.history_category == "opinion"
        else "Models agree and final confidence is strong."
    )
    selected_color = "#f87171" if app.history_category == "opinion" else "#2dd4bf"

    switch = ctk.CTkFrame(history_card, fg_color="transparent")
    switch.pack(fill="x", padx=28, pady=(0, 14))

    ctk.CTkButton(
        switch,
        text=f"Needs 2nd Opinion ({len(opinion_records)})",
        height=38,
        corner_radius=14,
        fg_color="#f87171" if app.history_category == "opinion" else "#27314a",
        hover_color="#ef4444" if app.history_category == "opinion" else "#34405d",
        text_color="#ffffff" if app.history_category == "opinion" else "#cbd5e1",
        font=ctk.CTkFont(size=13, weight="bold"),
        command=lambda: switch_history_category(app, "opinion"),
    ).pack(side="left", padx=(0, 10))

    ctk.CTkButton(
        switch,
        text=f"Confident ({len(confident_records)})",
        height=38,
        corner_radius=14,
        fg_color="#2dd4bf" if app.history_category == "confident" else "#27314a",
        hover_color="#14b8a6" if app.history_category == "confident" else "#34405d",
        text_color="#0d1220" if app.history_category == "confident" else "#cbd5e1",
        font=ctk.CTkFont(size=13, weight="bold"),
        command=lambda: switch_history_category(app, "confident"),
    ).pack(side="left")

    ctk.CTkLabel(
        history_card,
        text=(
            f"Showing {len(visible_records)} of {len(selected_records)} selected records  |  "
            f"Needs opinion: {len(opinion_records)}  |  "
            f"Confident: {len(confident_records)}"
        ),
        font=ctk.CTkFont(size=12),
        text_color="#8e99ae",
    ).pack(anchor="w", padx=28, pady=(0, 12))

    build_history_section(
        app,
        history_card,
        selected_title,
        selected_subtitle,
        visible_records,
        selected_color,
    )

    if app.history_visible_count < len(selected_records):
        ctk.CTkButton(
            history_card,
            text="Show Older Records",
            height=40,
            corner_radius=14,
            fg_color="#27314a",
            hover_color="#34405d",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda: show_more_history(app),
        ).pack(anchor="center", padx=28, pady=(4, 24))


def show_more_history(app):
    app.history_visible_count += HISTORY_PAGE_SIZE
    show_history_page(app)


def switch_history_category(app, category):
    if app.history_category == category:
        return

    app.history_category = category
    app.history_visible_count = HISTORY_PAGE_SIZE
    show_history_page(app)


def delete_history_record(app, record):
    saved_image = Path(record.get("saved_image", ""))
    if saved_image.exists():
        try:
            saved_image.unlink()
        except OSError:
            pass

    app.prediction_history = [
        item
        for item in app.prediction_history
        if not is_same_history_record(item, record)
    ]
    save_history_records(app.prediction_history)
    show_history_page(app)


def is_same_history_record(left, right):
    return (
        left.get("uploaded_at") == right.get("uploaded_at")
        and left.get("file_name") == right.get("file_name")
        and left.get("saved_image") == right.get("saved_image")
    )


def save_history_records(records):
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)


def categorize_history_records(records):
    opinion_records = []
    confident_records = []

    for record in records:
        if needs_human_opinion(record):
            opinion_records.append(record)
        else:
            confident_records.append(record)

    return opinion_records, confident_records


def needs_human_opinion(record):
    final = record.get("final", {})
    cnn = record.get("cnn", {})
    vit = record.get("vit", {})

    cnn_class = cnn.get("class_name", "--")
    vit_class = vit.get("class_name", "--")
    final_confidence = float(final.get("confidence_percent") or 0)
    final_score = float(final.get("score_percent") or 0)
    cnn_confidence = float(cnn.get("confidence_percent") or 0)
    vit_confidence = float(vit.get("confidence_percent") or 0)

    models_disagree = cnn_class != vit_class
    final_is_weak = (
        final_confidence < HUMAN_OPINION_CONFIDENCE_THRESHOLD
        or final_score < HUMAN_OPINION_SCORE_THRESHOLD
    )
    one_model_is_uncertain = (
        cnn_confidence < HUMAN_OPINION_CONFIDENCE_THRESHOLD
        or vit_confidence < HUMAN_OPINION_CONFIDENCE_THRESHOLD
    )

    return models_disagree or final_is_weak or one_model_is_uncertain


def build_history_section(app, parent, title, subtitle, records, accent_color):
    section = ctk.CTkFrame(parent, fg_color="transparent")
    section.pack(fill="x", padx=28, pady=(4, 16))

    ctk.CTkLabel(
        section,
        text=f"{title} ({len(records)})",
        font=ctk.CTkFont(size=20, weight="bold"),
        text_color=accent_color,
    ).pack(anchor="w", pady=(0, 2))

    ctk.CTkLabel(
        section,
        text=subtitle,
        font=ctk.CTkFont(size=12),
        text_color="#8e99ae",
    ).pack(anchor="w", pady=(0, 10))

    if not records:
        ctk.CTkLabel(
            section,
            text="No records in this category.",
            font=ctk.CTkFont(size=13),
            text_color="#8e99ae",
        ).pack(anchor="w", pady=(0, 8))
        return

    for record in records:
        build_history_record(app, section, record, accent_color)


def build_history_record(app, parent, record, accent_color):
    row = ctk.CTkFrame(
        parent,
        corner_radius=18,
        fg_color="#111827",
        border_color="#27314a",
        border_width=1,
    )
    row.pack(fill="x", pady=(0, 14))
    row.grid_columnconfigure(1, weight=1)
    row.grid_columnconfigure(2, weight=0)
    row.grid_columnconfigure(3, weight=0)

    image_path = Path(record.get("saved_image", ""))
    if image_path.exists():
        cache_key = str(image_path)
        photo = app.history_thumbnail_cache.get(cache_key)
        if photo is None:
            with Image.open(image_path) as source:
                preview = ImageOps.contain(source.convert("RGB"), (72, 72))
            photo = ctk.CTkImage(
                light_image=preview,
                dark_image=preview,
                size=preview.size,
            )
            app.history_thumbnail_cache[cache_key] = photo
        app.history_images.append(photo)
        image_label = ctk.CTkLabel(row, image=photo, text="")
    else:
        image_label = ctk.CTkLabel(
            row,
            text="No image",
            width=72,
            height=72,
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

    category_text = "Needs 2nd Opinion" if accent_color == "#f87171" else "Confident"
    ctk.CTkLabel(
        row,
        text=category_text,
        font=ctk.CTkFont(size=12, weight="bold"),
        text_color=accent_color,
        fg_color="#1b2133",
        corner_radius=10,
        padx=10,
        pady=4,
    ).grid(row=0, column=2, sticky="e", padx=(0, 8), pady=(14, 2))

    ctk.CTkButton(
        row,
        text="Delete",
        width=76,
        height=30,
        corner_radius=10,
        fg_color="#3f1f2a",
        hover_color="#7f1d1d",
        text_color="#fecaca",
        font=ctk.CTkFont(size=12, weight="bold"),
        command=lambda record=record: delete_history_record(app, record),
    ).grid(row=0, column=3, sticky="e", padx=(0, 16), pady=(14, 2))

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
    ).grid(row=1, column=2, columnspan=2, sticky="e", padx=16, pady=(0, 16))
