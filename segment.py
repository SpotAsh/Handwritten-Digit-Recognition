# segment.py
import cv2
import numpy as np

def segment_equation_image(image_path):
    """
    Load an equation image, preprocess it, find contours of symbols,
    crop each symbol, resize to 28x28, and return list of numpy arrays.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image {image_path}")

    # Invert colors: digits should be white on black background
    img = cv2.bitwise_not(img)

    # Apply adaptive threshold to get binary image
    thresh = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Find contours of connected components (symbols)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    symbol_images = []

    # Collect bounding boxes for all contours
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # Sort bounding boxes left to right (by x coordinate)
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    for (x, y, w, h) in bounding_boxes:
        # Ignore very small contours (noise)
        if w < 5 or h < 5:
            continue

        # Crop the symbol out of the original inverted image
        symbol_crop = img[y:y+h, x:x+w]

        # Resize to 28x28 while keeping aspect ratio and padding
        symbol_28 = resize_and_pad(symbol_crop, 28, 28)

        # Normalize to 0-1 float32
        symbol_28 = symbol_28.astype(np.float32) / 255.0

        symbol_images.append(symbol_28)

    return symbol_images


def resize_and_pad(img, target_w, target_h):
    """
    Resize img to fit in target_w x target_h while keeping aspect ratio.
    Pad with black pixels to target size.
    """
    h, w = img.shape

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black canvas of target size
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    # Compute top-left corner for centering resized img
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas
