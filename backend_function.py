def is_black_or_white(image):
    mean_intensity = image.mean()

    black_threshold = 30
    white_threshold = 225

    if mean_intensity < black_threshold:
        return "Black"
    elif mean_intensity > white_threshold:
        return "White"
    else:
        return "Neither black nor white"