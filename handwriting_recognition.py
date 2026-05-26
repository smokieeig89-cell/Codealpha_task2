"""
Handwritten Character Recognition using CNN
Supports: MNIST (digits 0-9) and EMNIST (letters A-Z)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. LOAD & PREPROCESS DATA
# ──────────────────────────────────────────────
def load_data(dataset="mnist"):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        num_classes = 10
    else:  # emnist (letters)
        import tensorflow_datasets as tfds
        ds = tfds.load("emnist/letters", split=["train", "test"], as_supervised=True)
        def prep(ds):
            X, y = zip(*[(x.numpy(), l.numpy()) for x, l in ds.batch(10000).take(1)][0])
            return np.array(X), np.array(y) - 1  # labels 1-26 → 0-25
        (X_train, y_train), (X_test, y_test) = prep(ds[0]), prep(ds[1])
        num_classes = 26

    # Normalize & reshape: (N, 28, 28) → (N, 28, 28, 1)
    X_train = X_train[..., np.newaxis] / 255.0
    X_test  = X_test[...,  np.newaxis] / 255.0
    return X_train, y_train, X_test, y_test, num_classes


# ──────────────────────────────────────────────
# 2. BUILD CNN MODEL
# ──────────────────────────────────────────────
def build_cnn(num_classes):
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ──────────────────────────────────────────────
# 3. TRAIN
# ──────────────────────────────────────────────
def train(model, X_train, y_train, X_test, y_test, epochs=5):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(X_test, y_test),
        verbose=1,
    )
    return history


# ──────────────────────────────────────────────
# 4. EVALUATE & VISUALIZE
# ──────────────────────────────────────────────
def evaluate(model, X_test, y_test, num_classes):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {acc:.4f}  |  Loss: {loss:.4f}")

    # Show 10 predictions
    preds = model.predict(X_test[:10], verbose=0).argmax(axis=1)
    labels = list("0123456789") if num_classes == 10 else [chr(65 + i) for i in range(26)]

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, img, true, pred in zip(axes.flat, X_test[:10], y_test[:10], preds):
        ax.imshow(img.squeeze(), cmap="gray")
        color = "green" if true == pred else "red"
        ax.set_title(f"True:{labels[true]}  Pred:{labels[pred]}", color=color, fontsize=9)
        ax.axis("off")
    plt.suptitle("Handwritten Character Recognition — CNN Predictions", fontsize=13)
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=100)
    plt.show()
    print("📸 Saved predictions.png")


# ──────────────────────────────────────────────
# 5. PREDICT A SINGLE IMAGE
# ──────────────────────────────────────────────
def predict_image(model, img_array, num_classes):
    """img_array: grayscale numpy array (28×28)"""
    labels = list("0123456789") if num_classes == 10 else [chr(65 + i) for i in range(26)]
    img = img_array[..., np.newaxis] / 255.0  # normalize
    img = np.expand_dims(img, 0)              # add batch dim
    pred = model.predict(img, verbose=0).argmax()
    print(f"🔍 Predicted character: {labels[pred]}")
    return labels[pred]


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    DATASET = "mnist"   # change to "emnist" for letters A-Z

    print(f"📦 Loading {DATASET.upper()} dataset...")
    X_train, y_train, X_test, y_test, num_classes = load_data(DATASET)
    print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}  |  Classes: {num_classes}")

    print("\n🧠 Building CNN model...")
    model = build_cnn(num_classes)
    model.summary()

    print("\n🚀 Training...")
    train(model, X_train, y_train, X_test, y_test, epochs=5)

    print("\n📊 Evaluating...")
    evaluate(model, X_test, y_test, num_classes)

    # Save model
    model.save("handwriting_cnn.keras")
    print("💾 Model saved to handwriting_cnn.keras")
