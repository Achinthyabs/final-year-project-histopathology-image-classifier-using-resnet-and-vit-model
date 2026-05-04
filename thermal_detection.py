import numpy as np
import torch
from PIL import Image


THERMAL_GRID_SIZE = 7


def class_probability_for_model(
    image,
    model_name,
    class_name,
    cnn_model,
    vit_model,
    vit_class_names,
    cnn_class_names,
    vit_transform,
    preprocess_for_cnn,
    device,
):
    if model_name == "ViT-Tiny":
        class_index = vit_class_names.index(class_name)
        tensor = vit_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = vit_model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        return float(probabilities[class_index])

    class_index = cnn_class_names.index(class_name)
    probabilities = cnn_model.predict(preprocess_for_cnn(image), verbose=0)[0]
    return float(probabilities[class_index])


def create_thermal_overlay(
    image,
    model_name,
    winner_result,
    cnn_model,
    vit_model,
    vit_class_names,
    cnn_class_names,
    vit_transform,
    preprocess_for_cnn,
    image_size,
    device,
):
    source = image.convert("RGB").resize(image_size)
    class_name = winner_result["class_name"]
    baseline_probability = class_probability_for_model(
        source,
        model_name,
        class_name,
        cnn_model,
        vit_model,
        vit_class_names,
        cnn_class_names,
        vit_transform,
        preprocess_for_cnn,
        device,
    )

    heatmap = np.zeros((THERMAL_GRID_SIZE, THERMAL_GRID_SIZE), dtype=np.float32)
    patch_width = image_size[0] // THERMAL_GRID_SIZE
    patch_height = image_size[1] // THERMAL_GRID_SIZE
    neutral_color = tuple(np.asarray(source).mean(axis=(0, 1)).astype(np.uint8))

    for row in range(THERMAL_GRID_SIZE):
        for column in range(THERMAL_GRID_SIZE):
            occluded = source.copy()
            left = column * patch_width
            top = row * patch_height
            right = image_size[0] if column == THERMAL_GRID_SIZE - 1 else left + patch_width
            bottom = image_size[1] if row == THERMAL_GRID_SIZE - 1 else top + patch_height
            occluded.paste(neutral_color, (left, top, right, bottom))

            occluded_probability = class_probability_for_model(
                occluded,
                model_name,
                class_name,
                cnn_model,
                vit_model,
                vit_class_names,
                cnn_class_names,
                vit_transform,
                preprocess_for_cnn,
                device,
            )
            heatmap[row, column] = max(0.0, baseline_probability - occluded_probability)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8), mode="L")
    heatmap_image = heatmap_image.resize(image_size, Image.Resampling.BICUBIC)

    red_channel = Image.new("L", image_size, 255)
    green_channel = heatmap_image.point(lambda value: int(value * 0.72))
    blue_channel = Image.new("L", image_size, 0)
    alpha_channel = heatmap_image.point(lambda value: int(value * 0.62))
    thermal_layer = Image.merge(
        "RGBA",
        (red_channel, green_channel, blue_channel, alpha_channel),
    )

    overlay = source.convert("RGBA")
    overlay.alpha_composite(thermal_layer)
    return overlay.convert("RGB"), model_name, class_name
