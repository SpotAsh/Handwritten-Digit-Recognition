import os
import cv2
from segment import segment_equation_image

# Paths
input_folder = "eq_input"
output_folder = "eq_output"
os.makedirs(output_folder, exist_ok=True)

# Pick one image with an equation in it
image_name = "eq1.png"  # put your test image here inside input folder
image_path = os.path.join(input_folder, image_name)

# Run segmentation to get list of symbol images (original size normalized numpy arrays)
# Also get the full processed grayscale image at original size
try:
    symbols, full_image = segment_equation_image(image_path, debug=True, return_full_image=True)
    if not symbols:
        print("No symbols found in the image")
        exit()
except Exception as e:
    print(f"Error processing image: {e}")
    exit()

# Save the full processed grayscale image at original size
full_image_uint8 = (full_image * 255).astype('uint8')
full_image_path = os.path.join(output_folder, "full_processed_image.png")
cv2.imwrite(full_image_path, full_image_uint8)
print(f"Saved full processed image: {full_image_path}")
print(f"Full image shape: {full_image.shape}")

# Save and show each symbol
for i, symbol_img in enumerate(symbols):
    # Convert back to uint8 [0,255] for saving and display
    symbol_uint8 = (symbol_img * 255).astype('uint8')

    save_path = os.path.join(output_folder, f"symbol_{i+1}.png")
    cv2.imwrite(save_path, symbol_uint8)
    print(f"Saved {save_path}")

    # Show the symbol image
    cv2.imshow(f"Symbol {i+1}", symbol_uint8)
    cv2.waitKey(0)  # press any key to continue

# Clean up all windows after the loop
cv2.destroyAllWindows()
