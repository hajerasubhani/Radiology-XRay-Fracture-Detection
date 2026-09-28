import tensorflow as tf
import numpy as np

IMAGE_SIZE = (224, 224)
MODEL_PATH = "fracture_model.keras"

# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# Ask user for image
image_path = input("\nEnter the path of the X-ray image: ")

# Load image
image = tf.keras.utils.load_img(
    image_path,
    target_size=IMAGE_SIZE
)

# Convert image to array
image_array = tf.keras.utils.img_to_array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# Normalize
image_array = image_array / 255.0

# Predict
probability = model.predict(image_array, verbose=0)[0][0]

# Display result
if probability >= 0.5:
    prediction = "NOT FRACTURED"
    confidence = probability
else:
    prediction = "FRACTURED"
    confidence = 1 - probability

print("\n======================================")
print("X-RAY FRACTURE PREDICTION")
print("======================================")
print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2%}")
print("======================================")