import os
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import image_dataset_from_directory

# =========================
# Config
# =========================
IMG_SIZE = 96
BATCH_SIZE = 32
NUM_CLASSES = 13
EPOCHS = 50

CLASS_NAMES = [
    "empty",
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
]

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset"
MODEL_PATH = BASE_DIR / "chess_piece_classifier.keras"


# =========================
# Preprocessing
# =========================
def preprocess_image_opencv(image_path: str, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Loads an image with OpenCV, applies light normalization, and returns
    a float32 image in range [0, 1].
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to fixed size
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Light denoising / smoothing can help when screenshots are compressed
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    return image


def build_tf_preprocess():
    """
    Keras preprocessing / augmentation stack.
    Mild augmentation keeps it realistic for chess square crops.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.08),
            layers.RandomContrast(0.08),
        ],
        name="augmentation",
    )


# =========================
# Model
# =========================
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES) -> tf.keras.Model:
    """
    TensorFlow CNN architecture for chess piece classification.
    Output classes include 'empty' for no-piece detection.
    """
    augmentation = build_tf_preprocess()

    inputs = layers.Input(shape=input_shape)

    x = augmentation(inputs)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ChessPieceClassifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =========================
# Dataset loading
# =========================
def load_datasets(dataset_dir: str):
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")

    train_ds = image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )

    val_ds = image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_ds = image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Normalize
    normalization_layer = layers.Rescaling(1.0 / 255.0)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


# =========================
# Training
# =========================
def train():
    train_ds, val_ds, test_ds = load_datasets(DATASET_DIR)

    model = build_model()

    checkpoint_cb = callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    return model, history


# =========================
# Inference
# =========================
def predict_image(model: tf.keras.Model, image_path: str, threshold: float = 0.60) -> dict:
    """
    Predicts the class of a single image.
    If confidence is too low, returns 'uncertain' to reduce bad guesses.
    """
    image = preprocess_image_opencv(image_path)
    image_batch = np.expand_dims(image, axis=0)

    probs = model.predict(image_batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    predicted_label = CLASS_NAMES[pred_idx]

    if confidence < threshold:
        return {
            "label": "uncertain",
            "confidence": confidence,
            "top_predictions": sorted(
                [
                    {"label": CLASS_NAMES[i], "confidence": float(probs[i])}
                    for i in range(len(CLASS_NAMES))
                ],
                key=lambda x: x["confidence"],
                reverse=True,
            )[:3],
        }

    return {
        "label": predicted_label,
        "confidence": confidence,
        "top_predictions": sorted(
            [
                {"label": CLASS_NAMES[i], "confidence": float(probs[i])}
                for i in range(len(CLASS_NAMES))
            ],
            key=lambda x: x["confidence"],
            reverse=True,
        )[:3],
    }


def load_trained_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)


# =========================
# Command line usage
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chess Piece Classifier")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, help="Path to image for prediction")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold")
    args = parser.parse_args()

    if args.train:
        train()

    elif args.predict:
        model = load_trained_model()
        result = predict_image(model, args.predict, threshold=args.threshold)
        print("\nPrediction Result")
        print("-----------------")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Top predictions:")
        for pred in result["top_predictions"]:
            print(f"  - {pred['label']}: {pred['confidence']:.4f}")

    else:
        print("Use --train to train or --predict <image_path> to classify an image.")