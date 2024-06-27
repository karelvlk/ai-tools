"""Color palettes for visualizing data with human distinguishable colors."""

palette = {
    "16": [
        "#2f4f4f",
        "#7f0000",
        "#006400",
        "#bdb76b",
        "#4b0082",
        "#ff0000",
        "#ffa500",
        "#ffff00",
        "#00ff00",
        "#00fa9a",
        "#0000ff",
        "#ff00ff",
        "#1e90ff",
        "#87ceeb",
        "#ff1493",
        "#ffb6c1",
    ],
    "8": [
        "#191970",
        "#006400",
        "#ff0000",
        "#ffd700",
        "#00ff00",
        "#00ffff",
        "#ff00ff",
        "#ffb6c1",
    ],
    "5": [
        "#ffa500",
        "#00ff7f",
        "#00bfff",
        "#0000ff",
        "#ff1493",
    ],
}


def get_palette(max_colors=16):
    if max_colors <= 5:
        return palette["5"]
    elif max_colors <= 8:
        return palette["8"]
    elif max_colors <= 16:
        return palette["16"]
    else:
        return palette["16"]


def get_color(color_id, max_colors=16):
    return get_palette(max_colors)[color_id % max_colors]
