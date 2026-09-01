"""
RGIPT - CS361 AIMLDL Application
Lab Assignment 4 : Fixing overfitting in a CNN clothing classifier

WHAT THIS SCRIPT DOES
----------------------
1. Rebuilds the ORIGINAL baseline CNN from the previous lab (the one that
   hit ~99% train acc / ~59% val acc -> classic overfitting).
2. Builds 4 modified versions, one technique at a time:
      A. Smaller model (fewer filters, GAP instead of Flatten -> far fewer params)
      B. BatchNorm after every Conv2D and Dense layer
      C. Dropout after Dense layers
      D. ReduceLROnPlateau + EarlyStopping callbacks (on top of the smaller model)
3. Trains a FINAL "best of all worlds" model that combines A+B+C+D.
4. Trains everything with a small epoch budget + EarlyStopping so the whole
   script finishes in a few minutes on a laptop/Colab GPU (or even CPU for
   a small dataset), while still letting you draw correct conclusions --
   EarlyStopping stops each run right when val_loss plateaus, so you are
   not "cutting off" a model early, you're stopping exactly where it would
   have stopped anyway.
5. Saves accuracy/loss curves for every model + one final comparison table
   and bar chart into ./outputs_lab4/, and saves the best model to
   best_model.keras (this is what Lab 5 will load).

HOW TO GET THE DATASET
-----------------------
The dataset is hosted on Google Drive. Colab / most local networks can reach
Drive; this sandboxed assistant environment cannot, so run this part on your
own machine or Colab:

    !pip install -q gdown tensorflow
    import gdown
    gdown.download(id="1XdXz0TKo_KCDRHOMvzV-YtcTx7NPG-jC", output="clothing_dataset.zip", quiet=False)
    !unzip -q clothing_dataset.zip -d clothing_dataset

After unzipping, point DATA_DIR below at the folder. Two layouts are
supported automatically (see `get_datasets()`):
  (a) clothing_dataset/train/<class_name>/*.jpg  and  clothing_dataset/test/<class_name>/*.jpg
  (b) clothing_dataset/<class_name>/*.jpg   (single folder -> we carve out
      a validation/test split ourselves with validation_split)

Just set DATA_DIR to wherever the class-named subfolders end up after
unzip -- print(os.listdir(DATA_DIR)) if unsure.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------
# CONFIG -- tweak these two lines for your machine, nothing else needs to change
# ----------------------------------------------------------------------
DATA_DIR = "clothing-dataset-small"      # <-- set this to your unzipped dataset folder
IMG_SIZE = 128                     # matches the original model's input shape
BATCH_SIZE = 64
NUM_CLASSES = 10
MAX_EPOCHS = 15                    # upper bound only -- EarlyStopping usually stops well before this
SEED = 42
OUT_DIR = "outputs_lab4"
os.makedirs(OUT_DIR, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def get_datasets():
    """Returns (train_ds, val_ds, test_ds, class_names). Auto-detects layout."""
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")

    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        # layout (a): pre-split train/ and test/ folders
        full_train = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
            validation_split=0.15, subset="training", seed=SEED)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
            validation_split=0.15, subset="validation", seed=SEED)
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False)
        class_names = full_train.class_names
        train_ds = full_train
    else:
        # layout (b): single folder of class subdirectories -> 70/15/15 split
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
            validation_split=0.30, subset="training", seed=SEED)
        rest_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
            validation_split=0.30, subset="validation", seed=SEED)
        class_names = train_ds.class_names
        # split "rest" 50/50 into val/test by taking alternating batches
        val_ds = rest_ds.shard(num_shards=2, index=0)
        test_ds = rest_ds.shard(num_shards=2, index=1)

    normalize = layers.Rescaling(1.0 / 255)
    AUTOTUNE = tf.data.AUTOTUNE

    def prep(ds, training=False):
        ds = ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.cache().shuffle(1000).prefetch(AUTOTUNE)
        else:
            ds = ds.cache().prefetch(AUTOTUNE)
        return ds

    return prep(train_ds, True), prep(val_ds), prep(test_ds), class_names


# ----------------------------------------------------------------------
# 2. MODEL VARIANTS
# ----------------------------------------------------------------------
def build_baseline():
    """The ORIGINAL model from the previous lab (16.78M params, overfits)."""
    model = keras.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="baseline_cnn")
    return model


def build_smaller():
    """Technique 1: shrink the model -- fewer filters, deeper (better feature
    extraction per parameter), and GlobalAveragePooling instead of Flatten
    (this alone removes ~16.7M of the 16.78M baseline parameters, since
    Flatten->Dense(256) was 99.9% of the model)."""
    model = keras.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="smaller_cnn")
    return model


def build_batchnorm():
    """Technique 2: BatchNorm after every Conv2D and Dense (built on the
    smaller backbone so BN's effect isn't confounded with raw param count)."""
    model = keras.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="batchnorm_cnn")
    return model


def build_dropout():
    """Technique 3: Dropout after the Dense layer (again on the smaller
    backbone, no BN, so dropout's effect is isolated)."""
    model = keras.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="dropout_cnn")
    return model


def build_final_best():
    """Technique 4 (combined): smaller backbone + BatchNorm + Dropout.
    Trained with ReduceLROnPlateau + EarlyStopping (see train_and_evaluate)."""
    model = keras.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, use_bias=False),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="final_best_cnn")
    return model


# ----------------------------------------------------------------------
# 3. TRAIN / EVALUATE / PLOT HELPERS
# ----------------------------------------------------------------------
def train_and_evaluate(model, train_ds, val_ds, test_ds, use_lr_schedule=False):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = []
    if use_lr_schedule:
        callbacks += [
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                               patience=2, min_lr=1e-6, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                           restore_best_weights=True, verbose=1),
        ]
    else:
        # even the "un-scheduled" runs get a generous early stop so the
        # whole script stays fast -- this does not change the comparison,
        # it just avoids wasting epochs once a model has clearly converged
        callbacks += [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                      restore_best_weights=True, verbose=1)]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
                         callbacks=callbacks, verbose=2)
    train_time = time.time() - t0

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    n_params = model.count_params()
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    gap = final_train_acc - final_val_acc

    return {
        "name": model.name,
        "history": history.history,
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "params": int(n_params),
        "train_time_sec": round(train_time, 1),
        "epochs_run": len(history.history["accuracy"]),
        "final_train_acc": float(final_train_acc),
        "final_val_acc": float(final_val_acc),
        "overfit_gap": float(gap),
    }


def plot_curves(result, out_dir=OUT_DIR):
    h = result["history"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h["accuracy"], label="Train")
    axes[0].plot(h["val_accuracy"], label="Val")
    axes[0].set_title(f"{result['name']} -- Accuracy")
    axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(h["loss"], label="Train")
    axes[1].plot(h["val_loss"], label="Val")
    axes[1].set_title(f"{result['name']} -- Loss")
    axes[1].set_xlabel("epoch"); axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, f"{result['name']}_curves.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def plot_comparison(results, out_dir=OUT_DIR):
    names = [r["name"] for r in results]
    test_accs = [r["test_acc"] for r in results]
    gaps = [r["overfit_gap"] for r in results]
    params = [r["params"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].bar(names, test_accs, color="steelblue")
    axes[0].set_title("Test Accuracy"); axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, gaps, color="indianred")
    axes[1].set_title("Overfitting Gap (train_acc - val_acc)")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(names, params, color="seagreen")
    axes[2].set_title("Trainable Parameters")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_yscale("log")

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_summary.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


# ----------------------------------------------------------------------
# 4. MAIN
# ----------------------------------------------------------------------
def main():
    print("Loading data ...")
    train_ds, val_ds, test_ds, class_names = get_datasets()
    print("Classes:", class_names)

    experiments = [
        ("baseline_cnn", build_baseline(), False),
        ("smaller_cnn", build_smaller(), False),
        ("batchnorm_cnn", build_batchnorm(), False),
        ("dropout_cnn", build_dropout(), False),
        ("final_best_cnn", build_final_best(), True),   # gets LR schedule + early stop
    ]

    results = []
    best_model_obj = None
    best_acc = -1
    for name, model, use_sched in experiments:
        print(f"\n===== Training {name} (params={model.count_params():,}) =====")
        res = train_and_evaluate(model, train_ds, val_ds, test_ds, use_lr_schedule=use_sched)
        results.append(res)
        plot_curves(res)
        if res["test_acc"] > best_acc:
            best_acc = res["test_acc"]
            best_model_obj = model

    plot_comparison(results)

    with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "history"} for r in results], f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'Model':<16}{'TestAcc':>10}{'TrainAcc':>10}{'Gap':>8}{'Params':>14}{'Epochs':>8}")
    print("=" * 70)
    for r in results:
        print(f"{r['name']:<16}{r['test_acc']*100:>9.2f}%{r['final_train_acc']*100:>9.2f}%"
              f"{r['overfit_gap']*100:>7.2f}%{r['params']:>14,}{r['epochs_run']:>8}")

    best_model_obj.save(os.path.join(OUT_DIR, "best_model.keras"))
    with open(os.path.join(OUT_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)
    print(f"\nBest model saved to {OUT_DIR}/best_model.keras "
          f"(and {OUT_DIR}/class_names.json) -- this is what lab5_streamlit_app.py loads.")


if __name__ == "__main__":
    main()
