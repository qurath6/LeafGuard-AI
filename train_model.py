
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Settings
DATASET_PATH = "dataset_small"
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 8

os.makedirs("models", exist_ok=True)

print("Loading dataset...")

# Create training dataset (80%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Create validation dataset (20%)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
number_of_classes = len(class_names)

print("Classes:", class_names)
print("Total classes:", number_of_classes)

# Save class names for later prediction
with open("models/class_names.json", "w") as file:
    json.dump(class_names, file)

# Improve dataset loading speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Load pretrained MobileNetV2
print("Loading MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# Build complete model
inputs = layers.Input(shape=(160, 160, 3))

x = data_augmentation(inputs)
x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    number_of_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save the best model during training
checkpoint = ModelCheckpoint(
    "models/leafguard_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nStarting AI model training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# Save final model
model.save("models/leafguard_model.keras")

print("\nTraining completed!")
print("Model saved in models/leafguard_model.keras")
print("Class names saved in models/class_names.json")
python train_model.py