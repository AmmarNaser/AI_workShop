import cv2
import matplotlib.pyplot as plt

# Load the image
file_path = "PATH"
image = cv2.imread(file_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not read the image file. Check the file path and permissions.")
else:
    print("Image loaded successfully.")

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(image_rgb)
plt.title("Reference Image")
plt.axis('off')
plt.show()

# Get image dimensions
height, width, channels = image_rgb.shape
print(f"Image dimensions (HxW): {height} x {width}")
print(f"Number of channels: {channels}")

# Check if resizing is needed (DeepFace generally handles resizing internally, but we can ensure it's reasonable)
max_size = 1024
if height > max_size or width > max_size:
    scaling_factor = max_size / max(height, width)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    image_rgb = cv2.resize(image_rgb, new_size)
    print(f"Resized image to: {new_size}")

# Save the preprocessed image (if needed for verification purposes)
preprocessed_file_path = "//Users/apple/Documents/WS/ai_401/project/prepro_my2.jpg"
cv2.imwrite(preprocessed_file_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
print(f"Preprocessed image saved to: {preprocessed_file_path}")
