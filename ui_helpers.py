import customtkinter as ctk


def make_card(parent):
    return ctk.CTkFrame(
        parent,
        corner_radius=24,
        fg_color="#1b2133",
        border_color="#252d44",
        border_width=1,
    )


def build_model_card(parent, title, color):
    ctk.CTkLabel(
        parent,
        text=title,
        font=ctk.CTkFont(size=20, weight="bold"),
        text_color="#f7f8fc",
    ).pack(anchor="w", padx=22, pady=(22, 8))

    class_label = ctk.CTkLabel(
        parent,
        text="Class: --",
        font=ctk.CTkFont(size=18, weight="bold"),
        text_color="#ffffff",
        wraplength=300,
        justify="left",
    )
    class_label.pack(anchor="w", padx=22, pady=(6, 2))

    confidence_label = ctk.CTkLabel(
        parent,
        text="Confidence: --",
        font=ctk.CTkFont(size=13),
        text_color="#9aa6bd",
    )
    confidence_label.pack(anchor="w", padx=22, pady=(0, 12))

    confidence_bar = ctk.CTkProgressBar(
        parent,
        height=14,
        progress_color=color,
        fg_color="#232b40",
    )
    confidence_bar.pack(fill="x", padx=22, pady=(0, 16))
    confidence_bar.set(0)

    score_label = ctk.CTkLabel(
        parent,
        text="Final score: --",
        font=ctk.CTkFont(size=13),
        text_color="#9aa6bd",
    )
    score_label.pack(anchor="w", padx=22, pady=(0, 14))

    top_label = ctk.CTkLabel(
        parent,
        text="Top predictions will appear here",
        font=ctk.CTkFont(size=12),
        text_color="#7f8aa3",
        justify="left",
        wraplength=300,
    )
    top_label.pack(anchor="w", padx=22, pady=(0, 22))

    return {
        "class": class_label,
        "confidence": confidence_label,
        "bar": confidence_bar,
        "score": score_label,
        "top": top_label,
    }
