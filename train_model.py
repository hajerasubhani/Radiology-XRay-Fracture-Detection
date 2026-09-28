import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# 1. SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

train_path = "train"
val_path = "val"


# ============================================================
# 2. LOAD DATASET
# ============================================================

train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

val_data = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)


# ============================================================
# 3. CHECK CLASS NAMES
# ============================================================

print("Classes:", train_data.class_names)


# ============================================================
# 4. IGNORE BAD IMAGES
# ============================================================

train_data = train_data.apply(
    tf.data.Dataset.ignore_errors
)

val_data = val_data.apply(
    tf.data.Dataset.ignore_errors
)


# ============================================================
# 5. IMPROVE DATA LOADING PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(
    buffer_size=AUTOTUNE
)

val_data = val_data.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
])


# ============================================================
# 7. LOAD PRETRAINED EFFICIENTNET MODEL
# ============================================================

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False


# ============================================================
# 8. BUILD OUR FRACTURE CLASSIFICATION MODEL
# ============================================================

model = models.Sequential([
    data_augmentation,

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 10. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "fracture_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nStarting model training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ============================================================
# 12. FINISHED
# ============================================================

print("\n======================================")
print("Training completed!")
print("======================================")
print("Model saved as: fracture_model.keras")