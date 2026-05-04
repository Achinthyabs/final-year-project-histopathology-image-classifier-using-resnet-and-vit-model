import customtkinter as ctk
from PIL import Image, ImageDraw

from ui_helpers import make_card


def show_about_page(app):
    app.hide_sidebar()
    app.clear_content()
    app.about_images = []
    app.content.grid_columnconfigure(0, weight=1, minsize=0)
    app.content.grid_columnconfigure(1, weight=0, minsize=0)
    app.content.grid_columnconfigure(2, weight=0, minsize=0)
    app.content.grid_rowconfigure(0, weight=1)

    about_card = make_card(app.content)
    about_card.grid(row=0, column=0, sticky="nsew")
    about_card.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        about_card,
        text="HistoScan",
        font=ctk.CTkFont(size=16, weight="bold"),
        text_color="#7c5cff",
    ).pack(anchor="center", padx=34, pady=(34, 4))

    ctk.CTkLabel(
        about_card,
        text="Colorectal Histopathology Classification",
        font=ctk.CTkFont(size=32, weight="bold"),
        text_color="#ffffff",
        justify="center",
    ).pack(anchor="center", padx=34, pady=(0, 10))

    ctk.CTkLabel(
        about_card,
        text=(
            "A formal desktop interface for automated tissue-patch analysis. "
            "The system compares a ResNet-8 CNN and a ViT-Tiny model to support "
            "faster, more consistent colorectal cancer screening workflows."
        ),
        font=ctk.CTkFont(size=16),
        text_color="#b4bfd4",
        wraplength=760,
        justify="center",
    ).pack(anchor="center", padx=34, pady=(0, 24))

    visual_grid = ctk.CTkFrame(about_card, fg_color="transparent")
    visual_grid.pack(fill="x", padx=34, pady=(0, 22))
    visual_grid.grid_columnconfigure((0, 1), weight=1)

    build_about_image_card(
        app,
        visual_grid,
        "Microscopic Tissue Patch",
        "Histopathology image patches are resized to 224 x 224 before analysis.",
        create_about_visual("histology"),
        0,
    )
    build_about_image_card(
        app,
        visual_grid,
        "Model Comparison Pipeline",
        "CNN and Vision Transformer outputs are compared before the final class decision.",
        create_about_visual("pipeline"),
        1,
    )

    info_grid = ctk.CTkFrame(about_card, fg_color="transparent")
    info_grid.pack(fill="x", padx=74, pady=(0, 22))
    info_grid.grid_columnconfigure((0, 1, 2), weight=1)

    build_about_metric(info_grid, "Input", "Single uploaded image", 0)
    build_about_metric(info_grid, "Models", "ResNet-8 CNN + ViT-Tiny", 1)
    build_about_metric(info_grid, "Output", "Final class decision", 2)

    ctk.CTkLabel(
        about_card,
        text=(
            "Histopathological diagnosis can be time-consuming and subjective when "
            "performed manually. HistoScan presents the uploaded sample, both model "
            "predictions, confidence values, and supporting metrics in one clear "
            "view for project demonstration and decision-support explanation."
        ),
        font=ctk.CTkFont(size=15),
        text_color="#9aa6bd",
        wraplength=780,
        justify="center",
    ).pack(anchor="center", padx=34, pady=(0, 26))

    ctk.CTkButton(
        about_card,
        text="Home",
        width=180,
        height=48,
        corner_radius=16,
        fg_color="#7c5cff",
        hover_color="#6b4df0",
        font=ctk.CTkFont(size=15, weight="bold"),
        command=app.show_home_page,
    ).pack(anchor="center", padx=34, pady=(0, 34))


def build_about_metric(parent, title, value, column):
    card = ctk.CTkFrame(
        parent,
        corner_radius=18,
        fg_color="#111827",
        border_color="#27314a",
        border_width=1,
    )
    card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 10, 0))

    ctk.CTkLabel(
        card,
        text=title,
        font=ctk.CTkFont(size=13, weight="bold"),
        text_color="#7c5cff",
    ).pack(anchor="w", padx=18, pady=(18, 4))

    ctk.CTkLabel(
        card,
        text=value,
        font=ctk.CTkFont(size=17, weight="bold"),
        text_color="#ffffff",
        wraplength=230,
        justify="left",
    ).pack(anchor="w", padx=18, pady=(0, 18))


def build_about_image_card(app, parent, title, caption, image, column):
    card = ctk.CTkFrame(
        parent,
        corner_radius=20,
        fg_color="#111827",
        border_color="#27314a",
        border_width=1,
    )
    card.grid(
        row=0,
        column=column,
        sticky="nsew",
        padx=(0, 12) if column == 0 else (12, 0),
    )
    card.grid_columnconfigure(0, weight=1)

    photo = ctk.CTkImage(
        light_image=image,
        dark_image=image,
        size=(360, 190),
    )
    app.about_images.append(photo)

    ctk.CTkLabel(card, image=photo, text="").pack(
        anchor="center",
        padx=20,
        pady=(20, 12),
    )
    ctk.CTkLabel(
        card,
        text=title,
        font=ctk.CTkFont(size=17, weight="bold"),
        text_color="#ffffff",
        justify="center",
    ).pack(anchor="center", padx=20, pady=(0, 4))
    ctk.CTkLabel(
        card,
        text=caption,
        font=ctk.CTkFont(size=12),
        text_color="#9aa6bd",
        wraplength=320,
        justify="center",
    ).pack(anchor="center", padx=20, pady=(0, 18))


def create_about_visual(visual_type):
    image = Image.new("RGB", (720, 380), "#0f172a")
    draw = ImageDraw.Draw(image)

    if visual_type == "histology":
        draw.rounded_rectangle((28, 24, 692, 356), radius=30, fill="#172033")
        draw.rounded_rectangle(
            (50, 46, 670, 334),
            radius=24,
            fill="#f3d1df",
            outline="#f8a8c3",
            width=3,
        )

        cell_colors = ["#7c2d58", "#a8558a", "#5b2b6f", "#de6f9d", "#8b5cf6"]
        for index in range(54):
            x = 78 + ((index * 89) % 560)
            y = 72 + ((index * 47) % 230)
            radius_x = 16 + (index % 4) * 5
            radius_y = 11 + (index % 3) * 4
            color = cell_colors[index % len(cell_colors)]
            draw.ellipse(
                (x, y, x + radius_x * 2, y + radius_y * 2),
                fill=color,
                outline="#f7c6d7",
                width=2,
            )

        for index in range(18):
            x = 92 + ((index * 131) % 520)
            y = 88 + ((index * 73) % 210)
            draw.ellipse((x, y, x + 7, y + 7), fill="#2dd4bf")

        draw.rounded_rectangle((76, 270, 312, 316), radius=18, fill="#111827")
        draw.text((100, 284), "224 x 224 tissue patch", fill="#e5e7eb")
        return image

    draw.rounded_rectangle((28, 24, 692, 356), radius=30, fill="#172033")
    draw.rounded_rectangle((76, 82, 224, 164), radius=18, fill="#26314d")
    draw.rounded_rectangle((298, 62, 458, 144), radius=18, fill="#252452")
    draw.rounded_rectangle((298, 198, 458, 280), radius=18, fill="#153f43")
    draw.rounded_rectangle((532, 130, 644, 232), radius=22, fill="#5b4acb")

    draw.text((116, 112), "Input", fill="#f8fafc")
    draw.text((318, 92), "ResNet-8", fill="#f8fafc")
    draw.text((350, 226), "ViT", fill="#f8fafc")
    draw.text((552, 166), "Final", fill="#ffffff")
    draw.text((542, 188), "Class", fill="#ffffff")

    draw.line((224, 123, 298, 103), fill="#7c5cff", width=5)
    draw.line((224, 123, 298, 239), fill="#2dd4bf", width=5)
    draw.line((458, 103, 532, 181), fill="#a78bfa", width=5)
    draw.line((458, 239, 532, 181), fill="#5eead4", width=5)

    for x, y, color in [
        (230, 116, "#7c5cff"),
        (230, 132, "#2dd4bf"),
        (464, 110, "#a78bfa"),
        (464, 232, "#5eead4"),
    ]:
        draw.ellipse((x, y, x + 14, y + 14), fill=color)

    draw.rounded_rectangle((90, 258, 250, 306), radius=16, fill="#0f172a")
    draw.text((110, 274), "Confidence", fill="#cbd5e1")
    draw.rounded_rectangle((470, 276, 630, 318), radius=16, fill="#0f172a")
    draw.text((492, 288), "Accuracy score", fill="#cbd5e1")
    return image
