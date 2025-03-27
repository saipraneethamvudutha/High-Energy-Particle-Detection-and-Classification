import tensorflow as tf
from PIL import Image
import numpy as np

# Load the Model
try:
    model = tf.keras.models.load_model("CNN_retrain.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Test Function (Model Predictions and Error Handling)
def test_image(image_path):
    try:
        # Load and prepare the image
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((32, 32))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make the prediction
        prediction = model.predict(image_array)[0][0]
        signal_percentage = prediction * 100
        background_percentage = (1 - prediction) * 100
        label = "signal" if prediction > 0.5 else "background"

        # Show the result
        print(f"\nTesting image: {image_path}")
        print(f"Prediction: {label}")
        print(f"Signal Percentage: {signal_percentage:.2f}%")
        print(f"Background Percentage: {background_percentage:.2f}%")

    except Exception as e:
        print(f"Error testing image {image_path}: {e}")

# Test Some Images
image_paths = [
    r"C:\Users\vudut\OneDrive\Desktop\Python\MINI Project\test_images_from_features\test_image_13171275.png",
    r"C:\Users\vudut\OneDrive\Desktop\Python\MINI Project\test_images_from_features\test_image_13281958.png"
]

for image_path in image_paths:
    test_image(image_path)