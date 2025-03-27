import gradio as gr
import tensorflow as tf
import os
from PIL import Image
import numpy as np

# Force CPU (Hugging Face issue workaround)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the Model
try:
    model = tf.keras.models.load_model("CNN_retrain.h5")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Define Prediction Function
def predict_image(image):
    try:
        print("📥 Received image for prediction.")
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((32, 32))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        print(f"🖼️ Processed Image Shape: {image_array.shape}")

        # Make prediction
        prediction = model.predict(image_array)[0][0]
        print(f"🔮 Raw Prediction Value: {prediction}")

        signal_percentage = prediction * 100
        background_percentage = (1 - prediction) * 100
        label = "signal" if prediction > 0.5 else "background"

        return f"Prediction: {label}\nSignal Percentage: {signal_percentage:.2f}%\nBackground Percentage: {background_percentage:.2f}%"
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return f"Error: {e}"

# Create Gradio Interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text"
)

# Launch the Interface with Debugging
if __name__ == "__main__":
    iface.launch(debug=True)
