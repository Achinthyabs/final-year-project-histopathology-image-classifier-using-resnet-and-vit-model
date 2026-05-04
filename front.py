import customtkinter as ctk


def show_front_page(app):
    app.hide_sidebar()
    app.clear_content()

    app.content.grid_columnconfigure(0, weight=1, minsize=0)
    app.content.grid_columnconfigure(1, weight=0, minsize=0)
    app.content.grid_columnconfigure(2, weight=0, minsize=0)
    app.content.grid_rowconfigure(0, weight=1)

    available_height = max(
        app.winfo_height() - 86,
        app.winfo_screenheight() - 140,
        760,
    )
    shell = ctk.CTkFrame(
        app.content,
        height=available_height,
        corner_radius=0,
        fg_color="#f3f1ec",
        border_color="#252a38",
        border_width=1,
    )
    shell.grid(row=0, column=0, sticky="nsew")
    shell.grid_propagate(False)
    shell.grid_columnconfigure(0, weight=11)
    shell.grid_columnconfigure(1, weight=13)
    shell.grid_rowconfigure(0, weight=1)

    left_panel = ctk.CTkFrame(shell, corner_radius=0, fg_color="#f3f1ec")
    left_panel.grid(row=0, column=0, sticky="nsew")
    left_panel.grid_columnconfigure(0, weight=1)
    left_panel.grid_rowconfigure(0, weight=1)

    left_content = ctk.CTkFrame(left_panel, fg_color="transparent")
    left_content.grid(row=0, column=0, sticky="nsew", padx=(62, 34), pady=50)
    left_content.grid_columnconfigure(0, weight=1)
    left_content.grid_rowconfigure(0, weight=1)
    left_content.grid_rowconfigure(1, weight=0)
    left_content.grid_rowconfigure(2, weight=1)

    title_block = ctk.CTkFrame(left_content, fg_color="transparent")
    title_block.grid(row=1, column=0)

    ctk.CTkFrame(
        title_block,
        width=84,
        height=5,
        corner_radius=0,
        fg_color="#7c5cff",
    ).pack(anchor="w", pady=(0, 16))

    ctk.CTkLabel(
        title_block,
        text="HISTOSCAN",
        font=ctk.CTkFont(size=54, weight="bold", slant="italic"),
        text_color="#050505",
        padx=8,
    ).pack(anchor="center")
    ctk.CTkLabel(
        title_block,
        text="HISTOPATHOLOGY",
        font=ctk.CTkFont(size=38, weight="bold", slant="italic"),
        text_color="#050505",
        padx=8,
    ).pack(anchor="center", pady=(14, 0))
    ctk.CTkLabel(
        title_block,
        text="CLASSIFICATION",
        font=ctk.CTkFont(size=38, weight="bold", slant="italic"),
        text_color="#050505",
        padx=8,
    ).pack(anchor="center", pady=(6, 0))

    ctk.CTkLabel(
        left_content,
        text=(
            "Colorectal histopathology classification with model comparison, "
            "confidence rules, history tracking, and thermal detection."
        ),
        font=ctk.CTkFont(size=12),
        text_color="#292524",
        wraplength=430,
        justify="left",
    ).grid(row=3, column=0, sticky="sw")

    accent = ctk.CTkFrame(left_panel, height=5, fg_color="#7c5cff", corner_radius=0)
    accent.grid(row=1, column=0, sticky="ew")

    right_panel = ctk.CTkFrame(
        shell,
        corner_radius=0,
        fg_color="#111827",
        border_color="#252a38",
        border_width=1,
    )
    right_panel.grid(row=0, column=1, rowspan=2, sticky="nsew")
    right_panel.grid_columnconfigure(0, weight=1)
    right_panel.grid_rowconfigure(0, weight=1)

    center = ctk.CTkFrame(right_panel, fg_color="transparent")
    center.grid(row=0, column=0, sticky="nsew", padx=54, pady=46)
    center.grid_columnconfigure(0, weight=1)
    center.grid_rowconfigure(0, weight=1)

    content = ctk.CTkFrame(center, fg_color="transparent")
    content.grid(row=0, column=0, sticky="ew")

    ctk.CTkLabel(
        content,
        text="SYSTEM OVERVIEW",
        font=ctk.CTkFont(size=13, weight="bold"),
        text_color="#8d70ff",
    ).pack(anchor="w", pady=(0, 14))

    ctk.CTkLabel(
        content,
        text=(
            "HistoScan is a desktop diagnostic-support interface for colorectal "
            "histopathology images. It evaluates one uploaded tissue image using "
            "ResNet-8 CNN and ViT-Tiny, compares their confidence-adjusted scores, "
            "then presents a final class with visual model evidence and saved "
            "prediction history."
        ),
        font=ctk.CTkFont(size=16),
        text_color="#d7deeb",
        wraplength=560,
        justify="left",
    ).pack(anchor="w", pady=(0, 24))

    ctk.CTkButton(
        content,
        text="Enter Home",
        width=190,
        height=48,
        corner_radius=4,
        fg_color="#f3f1ec",
        hover_color="#d8d4ca",
        text_color="#050505",
        font=ctk.CTkFont(size=15, weight="bold"),
        command=app.show_home_page,
    ).pack(anchor="w")

    ctk.CTkLabel(
        right_panel,
        text="SYSTEM STATUS  /  Model engine ready. Awaiting image input.",
        font=ctk.CTkFont(size=11),
        text_color="#9aa6bd",
    ).grid(row=2, column=0, sticky="ew", padx=46, pady=(0, 28))
