import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

test_path = "test_clean"
model_path = "fracture_model.keras"

# ---------------------------------------
# Load test dataset
# ---------------------------------------

print("Loading test dataset...")

test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = test_data.class_names

print("Classes:", class_names)

# Ignore problematic images if TensorFlow encounters any
test_data = test_data.apply(tf.data.Dataset.ignore_errors)

# Improve performance
test_data = test_data.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------
# Load trained model
# ---------------------------------------

print("\nLoading trained model...")

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully!")


# ---------------------------------------
# Evaluate model
# ---------------------------------------

print("\nEvaluating model...\n")

loss, accuracy = model.evaluate(test_data, verbose=1)

print("\n======================================")
print("TEST RESULTS")
print("======================================")

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.2%}")


# ---------------------------------------
# Get predictions
# ---------------------------------------

y_true = []
y_prob = []

print("\nGenerating predictions...")

for images, labels in test_data:

    probabilities = model.predict(images, verbose=0).ravel()

    y_prob.extend(probabilities)
    y_true.extend(labels.numpy().ravel())


y_true = np.array(y_true).astype(int)
y_prob = np.array(y_prob)

# Probability >= 0.5 = class 1
y_pred = (y_prob >= 0.5).astype(int)


# ---------------------------------------
# Classification Report
# ---------------------------------------

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ---------------------------------------
# Confusion Matrix
# ---------------------------------------

cm = confusion_matrix(y_true, y_pred)

print("======================================")
print("CONFUSION MATRIX")
print("======================================")

print(cm)


# ---------------------------------------
# Display Confusion Matrix
# ---------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot()

plt.title("Bone Fracture Detection - Confusion Matrix")
plt.tight_layout()

plt.show()