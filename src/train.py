import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import os

# ---------------------------------------------------------
# PATHS (IMPORTANT — MATCH YOUR FOLDER EXACTLY)
# ---------------------------------------------------------
DATA_DIR = "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
VAL_DIR  = "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"


# ---------------------------------------------------------
# TRAINING CONFIG
# ---------------------------------------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1  # You can increase later (10–20)

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
train_gen = ImageDataGenerator(rescale=1/255.0)
val_gen   = ImageDataGenerator(rescale=1/255.0)

train_data = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ---------------------------------------------------------
# SAVE CLASS LABELS
# ---------------------------------------------------------
label_path = "src/labels.txt"
with open(label_path, "w") as f:
    for cls in train_data.class_indices:
        f.write(cls + "\n")

print("✔ Labels saved at:", label_path)


# ---------------------------------------------------------
# BUILD MODEL (EfficientNetB0)
# ---------------------------------------------------------
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze base layers

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(len(train_data.class_indices), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------
os.makedirs("models", exist_ok=True)

# 1) Native Keras format (recommended)
model.save("models/efficientnet_model.keras")

# 2) TensorFlow SavedModel format (for TF Serving / TFLite etc.)
model.export("models/efficientnet_savedmodel")

print("🎉 Training Complete!")
print("✔ Keras model saved to models/efficientnet_model.keras")
print("✔ SavedModel exported to models/efficientnet_savedmodel/")

