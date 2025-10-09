import argparse
import json
import numpy as np
from src.task1.mnist_wrapper import MnistClassifier
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split


def load_data(limit=None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    x = x.astype('float32') / 255.0
    if limit:
        x = x[:limit]
        y = y[:limit]
    return train_test_split(x, y, test_size=0.2, random_state=42)


def main():
    parser = argparse.ArgumentParser(description="Train MNIST model")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (JSON)")
    parser.add_argument("--algo", choices=["rf", "nn", "cnn"], default=None, help="Algorithm to use")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for faster runs")

    args = parser.parse_args()

    # ---------- Загружаем конфиг (если указан) ----------
    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)

    # ---------- Параметры из аргументов или конфига ----------
    algo = args.algo or cfg.get("algorithm", "cnn")
    epochs = args.epochs or cfg.get("training", {}).get("epochs", 3)
    limit = args.limit or cfg.get("limit", None)

    print(f"⚙️ Using algorithm: {algo}")
    print(f"📚 Epochs: {epochs}")
    print(f"📉 Limit: {limit}")

    # ---------- Загружаем данные ----------
    X_train, X_val, y_train, y_val = load_data(limit=limit)

    # ---------- Обучение ----------
    if algo == "rf":
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_val_flat = X_val.reshape(len(X_val), -1)
        clf = MnistClassifier("rf")
        clf.train(X_train_flat, y_train, n_estimators=100)
        preds = clf.predict(X_val_flat)

    elif algo == "nn":
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_val_flat = X_val.reshape(len(X_val), -1)
        clf = MnistClassifier("nn")
        clf.train(X_train_flat, y_train, epochs=epochs)
        preds = clf.predict(X_val_flat)

    else:  # CNN
        X_train_c = X_train.reshape(-1, 28, 28, 1)
        X_val_c = X_val.reshape(-1, 28, 28, 1)
        clf = MnistClassifier("cnn")
        clf.train(X_train_c, y_train, epochs=epochs)
        preds = clf.predict(X_val_c)

    # ---------- Оценка ----------
    acc = (preds == y_val).mean()
    print("✅ Validation accuracy:", round(acc, 4))


if __name__ == "__main__":
    main()

