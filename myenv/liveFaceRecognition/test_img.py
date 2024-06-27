import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

# Paths to the reference and test images
reference_image_path = "/Users/apple/Documents/WS/ai_401/project/prepro_my2.jpg"
test_image_path = "/Users/apple/Documents/WS/ai_401/project/image.jpg"

# Load the reference image
reference_img = cv2.imread(reference_image_path)
if reference_img is None:
    print("Error: Could not load the reference image.")
else:
    print("Reference image loaded successfully.")

# Function to capture a test image from the webcam
def capture_test_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    ret, frame = cap.read()
    if ret:
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Save the test image
        cv2.imwrite(test_image_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        cap.release()
        return frame_rgb
    else:
        cap.release()
        print("Error: Could not capture image from webcam.")
        return None

# Capture a test image
test_img = capture_test_image()
if test_img is None:
    print("Failed to capture test image.")
else:
    print("Test image captured successfully.")

    # Display the reference and test images
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB))
    plt.title("Reference Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(test_img)
    plt.title("Test Image")
    plt.axis('off')

    plt.show()

    # Perform face verification
    try:
        result = DeepFace.verify(test_img, reference_img, model_name='VGG-Face', detector_backend='opencv')
        print(f"Verification result: {result}")
        if result['verified']:
            print("Faces match!")
        else:
            print("Faces do not match.")
    except Exception as e:
        print(f"Face verification error: {e}")
