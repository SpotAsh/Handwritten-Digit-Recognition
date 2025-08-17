# segment.py
import cv2
import numpy as np

def segment_equation_image(image_path, debug=False, return_full_image=False):
    """
    Load an equation image, preprocess it, find contours of symbols,
    crop each symbol with padding, and return list of normalized numpy arrays at original sizes.
    """

    # 1. Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image {image_path}")

    # 2. Invert colors: digits should be white, background black
    img = cv2.bitwise_not(img)

    if debug:
        cv2.imwrite("debug_inverted.png", img)
        print("Saved debug_inverted.png (white symbols, black background)")

    # 3. Adaptive thresholding → binary image
    thresh = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 17, 5)

    if debug:
        cv2.imwrite("debug_thresholded.png", thresh)
        print("Saved debug_thresholded.png (binary result after thresholding)")

    # 4. Morphological closing to connect broken strokes  - skip for now
    #kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small)
#
    #kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large)

    if debug:
        cv2.imwrite("debug_after_morph.png", thresh)
        print("Saved debug_after_morph.png (after morphology ops)")

    # Keep a copy of processed grayscale (for return if requested)
    full_processed_image = img.copy()

    # 5. Find contours (external → only outer shapes)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    print(f"[INFO] Total contours found: {len(contours)}")

    symbol_images = []

    if not contours:
        print("Warning: No contours found in the image")
        return symbol_images

    # Collect bounding boxes for all contours
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # 🚨 Skip merging for now — just use raw boxes
    merged_boxes = sorted(bounding_boxes, key=lambda b: b[0])


    # 9. Crop and normalize each symbol
    for (x, y, w, h) in merged_boxes:
        # Debug: Show bounding box info
        if debug:
            print(f"Box: x={x}, y={y}, w={w}, h={h}")

        # Filter tiny noise (can tweak this if digits are small)
        if w < 5 or h < 5:
            if debug:
                print(" → Skipped (too small, likely noise)")
            continue


        # Add 3px padding (prevent cutting edges)
        img_height, img_width = img.shape
        x_padded = max(0, x - 3)
        y_padded = max(0, y - 3)
        w_padded = min(img_width - x_padded, w + 6)
        h_padded = min(img_height - y_padded, h + 6)

        # Crop from the inverted grayscale image
        symbol_crop = img[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]

        # Normalize to [0,1]
        symbol_normalized = symbol_crop.astype(np.float32) / 255.0
        symbol_images.append(symbol_normalized)

        if debug:
            cv2.imwrite(f"debug_symbol_{len(symbol_images)}.png", symbol_crop)
            print(f"Saved debug_symbol_{len(symbol_images)}.png")

    # 10. Return symbols + full image (if requested)
    if return_full_image:
        full_normalized = full_processed_image.astype(np.float32) / 255.0
        return symbol_images, full_normalized
    else:
        return symbol_images



def merge_nearby_boxes(bounding_boxes, img_shape, horizontal_threshold=10, vertical_overlap_threshold=0.3):
    """
    Merge bounding boxes that are likely parts of the same digit.
    
    Args:
        bounding_boxes: List of (x, y, w, h) tuples
        img_shape: Shape of the image (height, width)
        horizontal_threshold: Maximum horizontal distance to consider for merging
        vertical_overlap_threshold: Minimum vertical overlap ratio to consider for merging
    
    Returns:
        List of merged bounding boxes
    """
    if not bounding_boxes:
        return []
    
    # Sort boxes by x coordinate
    boxes = sorted(bounding_boxes, key=lambda b: b[0])
    merged = []
    
    for box in boxes:
        x, y, w, h = box
        
        # Check if this box should be merged with the last merged box
        if merged and should_merge_boxes(merged[-1], box, horizontal_threshold, vertical_overlap_threshold):
            # Merge with the last box
            last_x, last_y, last_w, last_h = merged[-1]
            
            # Calculate the merged bounding box
            new_x = min(last_x, x)
            new_y = min(last_y, y)
            new_right = max(last_x + last_w, x + w)
            new_bottom = max(last_y + last_h, y + h)
            new_w = new_right - new_x
            new_h = new_bottom - new_y
            
            merged[-1] = (new_x, new_y, new_w, new_h)
        else:
            merged.append(box)
    
    return merged


def should_merge_boxes(box1, box2, horizontal_threshold, vertical_overlap_threshold):
    """
    Determine if two bounding boxes should be merged.
    
    Args:
        box1, box2: Bounding boxes as (x, y, w, h) tuples
        horizontal_threshold: Maximum horizontal distance to consider for merging
        vertical_overlap_threshold: Minimum vertical overlap ratio to consider for merging
    
    Returns:
        bool: True if boxes should be merged
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate horizontal distance between boxes
    horizontal_distance = x2 - (x1 + w1)
    
    # If boxes are too far apart horizontally, don't merge
    if horizontal_distance > horizontal_threshold:
        return False
    
    # Calculate vertical overlap
    y1_bottom = y1 + h1
    y2_bottom = y2 + h2
    
    overlap_top = max(y1, y2)
    overlap_bottom = min(y1_bottom, y2_bottom)
    overlap_height = max(0, overlap_bottom - overlap_top)
    
    # Calculate overlap ratio relative to the smaller box height
    min_height = min(h1, h2)
    overlap_ratio = overlap_height / min_height if min_height > 0 else 0
    
    return overlap_ratio >= vertical_overlap_threshold


def resize_and_pad(img, target_w, target_h):
    """
    Resize img to fit in target_w x target_h while keeping aspect ratio.
    Pad with black pixels to target size.
    """
    h, w = img.shape
    
    # Handle edge cases: empty or zero-sized images
    if w == 0 or h == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))  # Ensure minimum size of 1
    new_h = max(1, int(h * scale))  # Ensure minimum size of 1

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black canvas of target size
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    # Compute top-left corner for centering resized img
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas
