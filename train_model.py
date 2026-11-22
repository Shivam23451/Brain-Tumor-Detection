import os
import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = os.path.join(
    "..",
    "archive (1)",
    "brisc2025",
    "classification_task"
)

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_PATH = "brain_tumor_model.keras"
CLASS_NAMES_PATH = "class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10


# ============================================================
# CHECK DATASET
# ============================================================

print("\nChecking dataset...")

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training folder not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test folder not found:\n{TEST_DIR}"
    )

print("Training directory:", TRAIN_DIR)
print("Test directory:", TEST_DIR)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,

    validation_split=0.20,

    subset="training",

    seed=SEED,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="categorical"
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,

    validation_split=0.20,

    subset="validation",

    seed=SEED,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="categorical"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,

    shuffle=False,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="categorical"
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_ds.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# Save class names for Flask

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=4)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)


# ============================================================
# RESNET50 BASE MODEL
# ============================================================

print("\nLoading ResNet50...")

base_model = ResNet50(
    include_top=False,

    weights="imagenet",

    input_shape=(224, 224, 3)
)

# Freeze pretrained layers

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = keras.Input(
    shape=(224, 224, 3)
)


x = data_augmentation(inputs)


x = layers.Lambda(
    preprocess_input,
    name="resnet_preprocess"
)(x)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dropout(0.40)(x)


x = layers.Dense(
    256,
    activation="relu"
)(x)


x = layers.Dropout(0.30)(x)


outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)


model = keras.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    ),

    keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=4,

        restore_best_weights=True,

        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=2,

        min_lr=1e-7,

        verbose=1
    )
]


# ============================================================
# INITIAL TRAINING
# ============================================================

print("\n======================================")
print("STARTING INITIAL TRAINING")
print("======================================\n")


history_initial = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=INITIAL_EPOCHS,

    callbacks=callbacks
)


# ============================================================
# FINE TUNING
# ============================================================

print("\n======================================")
print("STARTING FINE-TUNING")
print("======================================\n")


base_model.trainable = True


# Freeze first layers and fine-tune later layers

for layer in base_model.layers[:-30]:

    layer.trainable = False


# Keep BatchNormalization layers frozen

for layer in base_model.layers:

    if isinstance(layer, layers.BatchNormalization):

        layer.trainable = False


model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


history_fine = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=callbacks
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input,
        "function": preprocess_input
    }
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n======================================")
print("FINAL TEST EVALUATION")
print("======================================\n")


test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)


print(
    f"\nTest Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: {test_loss:.4f}"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []


for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    true_classes = np.argmax(
        labels.numpy(),
        axis=1
    )

    y_true.extend(
        true_classes
    )

    y_pred.extend(
        predicted_classes
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================\n")


print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n======================================")
print("CONFUSION MATRIX")
print("======================================\n")

print(cm)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n======================================")
print("TRAINING COMPLETED")
print("======================================")

print("\nModel saved at:")
print(MODEL_PATH)

print("\nClass names saved at:")
print(CLASS_NAMES_PATH)

print("\nClasses:")

for name in class_names:
    print("-", name)