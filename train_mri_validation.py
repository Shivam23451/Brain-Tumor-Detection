import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "validation_dataset"

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 10

MODEL_NAME = "mri_validator.keras"


# ============================================================
# CHECK DATASET
# ============================================================

MRI_DIR = os.path.join(
    DATASET_DIR,
    "mri"
)

NON_MRI_DIR = os.path.join(
    DATASET_DIR,
    "non-mri"
)


if not os.path.exists(MRI_DIR):

    raise FileNotFoundError(
        f"MRI folder not found: {MRI_DIR}"
    )


if not os.path.exists(NON_MRI_DIR):

    raise FileNotFoundError(
        f"Non-MRI folder not found: {NON_MRI_DIR}"
    )


print("=" * 60)
print("MRI / NON-MRI VALIDATOR")
print("=" * 60)

print()

print(
    "MRI folder:",
    MRI_DIR
)

print(
    "Non-MRI folder:",
    NON_MRI_DIR
)


# ============================================================
# LOAD DATASET
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_DIR,

    labels="inferred",

    label_mode="binary",

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    validation_split=0.20,

    subset="training",

    seed=42

)


validation_dataset = tf.keras.utils.image_dataset_from_directory(

    DATASET_DIR,

    labels="inferred",

    label_mode="binary",

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    validation_split=0.20,

    subset="validation",

    seed=42

)


print()

print(
    "Classes:",
    train_dataset.class_names
)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


train_dataset = train_dataset.prefetch(
    AUTOTUNE
)


validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.05
    ),

    layers.RandomZoom(
        0.10
    )

])


# ============================================================
# BASE MODEL
# ============================================================

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(
        224,
        224,
        3
    )

)


# Freeze pretrained layers

base_model.trainable = False


# ============================================================
# MODEL
# ============================================================

inputs = keras.Input(
    shape=(
        224,
        224,
        3
    )
)


x = data_augmentation(
    inputs
)


x = keras.applications.mobilenet_v2.preprocess_input(
    x
)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(
    x
)


x = layers.Dropout(
    0.30
)(
    x
)


outputs = layers.Dense(
    1,
    activation="sigmoid"
)(
    x
)


model = keras.Model(
    inputs,
    outputs
)

# COMPILE

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]

)

# MODEL SUMMARY

model.summary()

# CALLBACKS

callbacks = [

    keras.callbacks.ModelCheckpoint(

        MODEL_NAME,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max"

    ),

    keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True

    )

]

# TRAIN

print()
print("=" * 60)
print("STARTING MRI VALIDATOR TRAINING")
print("=" * 60)
print()


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=callbacks

)

# FINAL EVALUATION


print()
print("=" * 60)
print("VALIDATION RESULT")
print("=" * 60)


loss, accuracy = model.evaluate(
    validation_dataset
)


print()

print(
    "Validation Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)

# SAVE MODEL

model.save(
    MODEL_NAME
)

print()

print(
    "MRI validator saved successfully:"
)

print(
    MODEL_NAME
)

print()
print("Training completed.")