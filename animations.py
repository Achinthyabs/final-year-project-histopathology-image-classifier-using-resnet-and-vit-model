def interpolate_hex_color(start, end, progress):
    start = start.lstrip("#")
    end = end.lstrip("#")
    channels = []

    for index in range(0, 6, 2):
        start_value = int(start[index : index + 2], 16)
        end_value = int(end[index : index + 2], 16)
        value = round(start_value + (end_value - start_value) * progress)
        channels.append(f"{value:02x}")

    return f"#{''.join(channels)}"


def animate_content_in(app):
    if app.content_animation_job is not None:
        app.after_cancel(app.content_animation_job)
        app.content_animation_job = None

    app.content.configure(fg_color="#111827")

    steps = 6
    duration_ms = 120
    interval_ms = max(1, duration_ms // steps)

    def step(frame=0):
        progress = frame / steps
        bg_color = interpolate_hex_color("#111827", "#151927", progress)
        app.content.configure(fg_color=bg_color)

        if frame < steps:
            app.content_animation_job = app.after(
                interval_ms,
                lambda: step(frame + 1),
            )
        else:
            app.content_animation_job = None
            app.content.configure(fg_color="transparent")

    step()
