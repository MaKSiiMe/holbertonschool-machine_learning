#!/usr/bin/env python3
"""Transfer learning on CIFAR-10 with MobileNetV2 (fast) to reach
>=87% test acc.

- Hint2: Lambda layer upsizes 32x32 -> 160x160 (smaller than 224
  for speed)
- Hint3: Precompute features ONCE from frozen backbone, train top on
  those features
- Then rebuild full model and fine-tune the last ~20 non-BN layers
  with SGD+momentum
- Proper preprocessing for MobileNetV2: multiply by 255 then
  mobilenet_v2.preprocess_input ([-1,1])
- Saves compiled model as cifar10.h5
- Does not run on import
"""
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Resizing, Rescaling


# --------------------------
# Required by the checker
# --------------------------
def preprocess_data(X, Y):
    """Preprocess CIFAR-10:
    - X -> float32 in [0,1]
    - Y -> one-hot (10 classes)
    Returns: X_p, Y_p
    """
    X_p = X.astype("float32") / 255.0
    Y_p = keras.utils.to_categorical(Y.reshape(-1), 10)
    return X_p, Y_p


# --------------------------
# Building blocks
# --------------------------
def build_mnv2_feature_extractor(input_shape=(32, 32, 3),
                                 target_size=(160, 160)):
    """Build a frozen MobileNetV2 feature extractor with proper
    preprocessing.
    Output is a pooled feature vector (GAP).
    """
    from tensorflow.keras.applications import mobilenet_v2

    base = mobilenet_v2.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(target_size[0], target_size[1], 3)
    )
    base.trainable = False  # fully frozen for feature extraction

    inp = keras.Input(shape=input_shape)
    # Resize to target
    x = Resizing(target_size[0], target_size[1],
                 interpolation="bilinear", name="resize")(inp)
    # Convert [0,1] -> [0,255] then preprocess to [-1,1]
    x = Rescaling(2.0, offset=-1.0, name="scale_to_-1_1")(x)
    # Extract conv features
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    extractor = keras.Model(inp, x, name="mnv2_feature_extractor")
    return extractor, base


def build_top_classifier(feat_dim, num_classes=10,
                         label_smoothing=0.05):
    """Small MLP head trained on precomputed features."""
    inp = keras.Input(shape=(feat_dim,), name="feat_in")
    x = layers.BatchNormalization(name="bn_top_0")(inp)
    x = layers.Dropout(0.30, name="do_top_0")(x)
    x = layers.Dense(512, activation="relu", name="fc_512")(x)
    x = layers.BatchNormalization(name="bn_top_1")(x)
    x = layers.Dropout(0.30, name="do_top_1")(x)
    x = layers.Dense(256, activation="relu", name="fc_256")(x)
    x = layers.BatchNormalization(name="bn_top_2")(x)
    x = layers.Dropout(0.20, name="do_top_2")(x)
    out = layers.Dense(num_classes, activation="softmax",
                       name="pred")(x)

    model = keras.Model(inp, out, name="top_classifier")
    loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=loss,
                  metrics=["accuracy"])
    return model


def rebuild_full_model_with_top(base_model, top_model,
                                input_shape=(32, 32, 3),
                                target_size=(160, 160)):
    """Rebuild the full end-to-end graph: resize+preproc -> base -> GAP
    -> top_model.
    We *call* `top_model` on the GAP tensor to reuse its learned weights
    directly.
    """
    from tensorflow.keras.applications import mobilenet_v2

    inp = keras.Input(shape=input_shape, name="input_32x32x3")
    x = Resizing(target_size[0], target_size[1],
                 interpolation="bilinear", name="resize")(inp)
    x = Rescaling(2.0, offset=-1.0, name="scale_to_-1_1")(x)
    x = base_model(x, training=False)  # still frozen at this stage
    gap = layers.GlobalAveragePooling2D(name="gap")(x)

    out = top_model(gap)
    full = keras.Model(inp, out, name="cifar10_mobilenetv2")

    return full


def freeze_bns_and_unfreeze_tail(base_model, n_unfreeze=30):
    """Freeze all BatchNorm layers; unfreeze the last `n_unfreeze`
    non-BN layers."""
    # Freeze everything first
    for layer in base_model.layers:
        layer.trainable = False

    # Then unfreeze last N non-BN layers
    to_go = n_unfreeze
    for layer in reversed(base_model.layers):
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            continue
        if to_go > 0:
            layer.trainable = True
            to_go -= 1
        else:
            break


# --------------------------
# Main training routine
# --------------------------
if __name__ == "__main__":
    # Optional reproducibility
    tf.keras.utils.set_random_seed(42)

    # 1) Load CIFAR-10 and split
    (x_train_full, y_train_full), (x_test, y_test) = \
        keras.datasets.cifar10.load_data()
    y_train_full = y_train_full.reshape(-1)
    y_test = y_test.reshape(-1)

    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=5000, random_state=42,
        stratify=y_train_full
    )
    del x_train_full, y_train_full
    gc.collect()

    # 2) Preprocess to [0,1] + one-hot (checker expects this function)
    x_train, y_train = preprocess_data(x_train, y_train)
    x_val, y_val = preprocess_data(x_val, y_val)
    x_test, y_test = preprocess_data(x_test, y_test)

    # 3) Build feature extractor (frozen) and precompute features once
    extractor, base = build_mnv2_feature_extractor(
        input_shape=(32, 32, 3), target_size=(160, 160))
    print("Extracting features (train/val/test) ...")
    train_feats = extractor.predict(x_train, batch_size=64, verbose=1)
    val_feats = extractor.predict(x_val, batch_size=64, verbose=1)
    test_feats = extractor.predict(x_test, batch_size=64, verbose=1)
    feat_dim = train_feats.shape[1]
    print("Feature dim:", feat_dim)

    # 4) Train top classifier on features
    top = build_top_classifier(feat_dim, num_classes=10,
                               label_smoothing=0.05)
    callbacks_top = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6,
            verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10,
            restore_best_weights=True, verbose=1),
    ]
    top.fit(
        train_feats, y_train,
        validation_data=(val_feats, y_val),
        batch_size=128,
        epochs=80,
        verbose=2,
        callbacks=callbacks_top
    )
    tloss, tacc = top.evaluate(test_feats, y_test, verbose=0)
    print(f"[Top-on-features] test_acc = {tacc:.4f}")

    # 5) Rebuild full model end-to-end and fine-tune a small tail
    full = rebuild_full_model_with_top(
        base, top, input_shape=(32, 32, 3), target_size=(160, 160))
    # Freeze BN and unfreeze a small tail of the backbone
    freeze_bns_and_unfreeze_tail(base, n_unfreeze=20)

    # Compile for fine-tuning (small LR, SGD momentum, mild label
    # smoothing)
    loss_ft = keras.losses.CategoricalCrossentropy(
        label_smoothing=0.05)
    full.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=3e-4, momentum=0.9, nesterov=True),
        loss=loss_ft,
        metrics=["accuracy"]
    )

    ft_callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6,
            verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, verbose=1),
    ]

    full.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=15,
        batch_size=128,
        verbose=2,
        callbacks=ft_callbacks
    )

    # 6) Final evaluation and save compiled model
    final_loss, final_acc = full.evaluate(
        x_test, y_test, batch_size=128, verbose=1)
    print(f"[Full model] test_acc = {final_acc:.4f}")
    full.save("cifar10.h5")
    print("Saved model -> cifar10.h5")
