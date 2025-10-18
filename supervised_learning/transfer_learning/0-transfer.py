#!/usr/bin/env python3
"""
Script d'entraînement pour la classification sur CIFAR-10 en utilisant
le transfer learning avec MobileNetV2.

Ce script est structuré en deux étapes :
1. Entraînement d'une tête de classification sur les caractéristiques
   extraites d'un MobileNetV2 gelé.
2. Fine-tuning optionnel des dernières couches du modèle de base.
"""

import os
import json
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Désactivation des logs TensorFlow non critiques
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# --- CONSTANTES DE CONFIGURATION ---
TARGET_SIZE = (160, 160)
NUM_CLASSES = 10
FEATURE_DIM = 1280


# -------------------------
# Fonctions Utilitaires
# -------------------------

def set_seed(seed: int = 42) -> None:
    """Fixe les graines aléatoires pour la reproductibilité."""
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def preprocess_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalise les images et convertit les étiquettes en one-hot encoding."""
    Xp = X.astype("float32") / 255.0
    yp = keras.utils.to_categorical(y.reshape(-1), NUM_CLASSES)
    return Xp, yp


# -------------------------
# Fonctions de Construction du Modèle
# -------------------------

def build_feature_extractor(target_size: tuple = TARGET_SIZE) -> tuple[keras.Model, keras.Model]:
    """Construit l'extracteur de caractéristiques basé sur un MobileNetV2 gelé."""
    from tensorflow.keras.applications import mobilenet_v2
    
    base_model = mobilenet_v2.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(target_size[0], target_size[1], 3)
    )
    base_model.trainable = False

    inp = keras.Input(shape=(32, 32, 3))
    x = layers.Resizing(target_size[0], target_size[1], interpolation="bilinear")(inp)
    x = layers.Rescaling(2.0, offset=-1.0)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    extractor = keras.Model(inp, x, name="feature_extractor")
    return extractor, base_model


def build_top_classifier(feat_dim: int = FEATURE_DIM, num_classes: int = NUM_CLASSES) -> keras.Model:
    """Construit la tête de classification (MLP)."""
    inp = keras.Input(shape=(feat_dim,))
    x = layers.BatchNormalization()(inp)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inp, out, name="top_classifier")
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3), loss=loss, metrics=["accuracy"])
    return model


def rebuild_full_model(base_model: keras.Model, top_model: keras.Model, args: argparse.Namespace, target_size: tuple = TARGET_SIZE) -> keras.Model:
    """Reconstruit le modèle complet en combinant la base et la tête."""
    inp = keras.Input(shape=(32, 32, 3))
    x = inp

    # Affiche un avertissement si les options avancées sont utilisées, mais ne plante pas.
    if args.use_cutout or args.use_mixup:
        print("\n[AVERTISSEMENT] Les options --use_cutout et --use_mixup ne sont pas disponibles sur cette version de TensorFlow.")
        print("--> Ces options seront ignorées pour continuer l'entraînement.\n")

    x = layers.Resizing(target_size[0], target_size[1], interpolation="bilinear")(x)
    x = layers.Rescaling(2.0, offset=-1.0)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    out = top_model(x)
    
    full_model = keras.Model(inp, out, name="full_cifar10_model")
    return full_model


def unfreeze_model_tail(base_model: keras.Model, n_unfreeze: int = 0) -> None:
    """Dégèle les N dernières couches (non-BN) du modèle de base."""
    for layer in base_model.layers:
        layer.trainable = False
        
    layers_to_unfreeze = []
    for layer in reversed(base_model.layers):
        if not isinstance(layer, layers.BatchNormalization):
            layers_to_unfreeze.append(layer)
        if len(layers_to_unfreeze) >= n_unfreeze:
            break
            
    for layer in layers_to_unfreeze:
        layer.trainable = True


def _history_to_dict(hist: tf.keras.callbacks.History) -> dict:
    """Convertit un objet History de Keras en dictionnaire JSON-sérialisable."""
    if not hist:
        return {}
    h = hist.history
    # Convertit les np.float32 en float Python natif
    for k in h:
        h[k] = [float(v) for v in h[k]]
    return h


def _best_epoch(values: list) -> int:
    """Trouve l'époque avec la meilleure valeur (ex: val_accuracy)."""
    if not values:
        return 0
    return int(np.argmax(values) + 1)


# -------------------------
# Fonctions de Workflow Principal
# -------------------------

def load_and_prepare_data() -> dict:
    """Charge, pré-traite et divise les données CIFAR-10."""
    (x_tr_full, y_tr_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Garder une copie des étiquettes non-encodées pour la matrice de confusion
    y_test_labels = y_test.reshape(-1)

    x_train, x_val, y_train, y_val = train_test_split(
        x_tr_full, y_tr_full, test_size=5000, stratify=y_tr_full, random_state=42
    )

    return {
        "x_train": preprocess_data(x_train, y_train)[0],
        "y_train": preprocess_data(x_train, y_train)[1],
        "x_val": preprocess_data(x_val, y_val)[0],
        "y_val": preprocess_data(x_val, y_val)[1],
        "x_test": preprocess_data(x_test, y_test)[0],
        "y_test": preprocess_data(x_test, y_test)[1],
        "y_test_labels": y_test_labels,
    }


def run_stage1_training(data: dict, args: argparse.Namespace) -> dict:
    """Exécute l'extraction de caractéristiques et l'entraînement de la tête."""
    print(">> Stage 1: Extraction de caractéristiques et entraînement de la tête...")
    t0 = time.time()
    
    extractor, base_model = build_feature_extractor()
    
    f_train = extractor.predict(data["x_train"], batch_size=args.batch_size, verbose=1)
    f_val   = extractor.predict(data["x_val"],   batch_size=args.batch_size, verbose=1)
    f_test  = extractor.predict(data["x_test"],  batch_size=args.batch_size, verbose=1)
    t1 = time.time()

    top_model = build_top_classifier()

    # --- Création des callbacks ---
    callbacks_s1 = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            verbose=1
        )
    ]

    history = top_model.fit(
        f_train, data["y_train"],
        validation_data=(f_val, data["y_val"]),
        epochs=args.epochs_stage1,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks_s1 # AJOUT : On passe les callbacks ici
    )
    t2 = time.time()

    return {
        "base_model": base_model, "top_model": top_model, "history": history,
        "time_feature_extract": t1 - t0, "time_top_train": t2 - t1,
    }


def run_stage2_finetuning(data: dict, args: argparse.Namespace, stage1_results: dict) -> dict:
    """Exécute le fine-tuning du modèle complet, avec augmentation optionnelle."""
    print(f"\n>> Stage 2: Fine-tuning des {args.n_unfreeze} dernières couches...")
    t0 = time.time()
    
    full_model = rebuild_full_model(stage1_results["base_model"], stage1_results["top_model"], args)
    unfreeze_model_tail(stage1_results["base_model"], n_unfreeze=args.n_unfreeze)

    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    full_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss=loss, metrics=["accuracy"]
    )
    
    callbacks_s2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1
        )
    ]

    # --- Support de Mixup ---
    if args.use_mixup:
        print(f">> Utilisation de Mixup (alpha={getattr(args, 'mixup_alpha', 0.2)})...")
        
        # Prépare le datagen classique si l'augmentation est activée
        datagen = None
        if args.augment:
            print(">> + Data augmentation classique activée")
            datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=args.rotation,
                width_shift_range=args.width_shift,
                height_shift_range=args.height_shift,
                zoom_range=args.zoom,
                horizontal_flip=True
            )
            datagen.fit(data["x_train"])
        
        # Utilise le générateur Mixup
        train_generator = MixupDataGenerator(
            data["x_train"], 
            data["y_train"],
            batch_size=args.batch_size,
            datagen=datagen,
            alpha=getattr(args, 'mixup_alpha', 0.2),
            shuffle=True
        )
        
        history = full_model.fit(
            train_generator,
            validation_data=(data["x_val"], data["y_val"]),
            epochs=args.epochs_stage2,
            verbose=2,
            callbacks=callbacks_s2
        )
    
    elif args.augment:
        print(">> Utilisation de la data augmentation pour le Stage 2...")
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=args.rotation, width_shift_range=args.width_shift,
            height_shift_range=args.height_shift, zoom_range=args.zoom,
            horizontal_flip=True
        )
        datagen.fit(data["x_train"])
        history = full_model.fit(
            datagen.flow(data["x_train"], data["y_train"], batch_size=args.batch_size),
            steps_per_epoch=len(data["x_train"]) // args.batch_size,
            validation_data=(data["x_val"], data["y_val"]),
            epochs=args.epochs_stage2,
            verbose=2,
            callbacks=callbacks_s2
        )
    else:
        history = full_model.fit(
            data["x_train"], data["y_train"],
            validation_data=(data["x_val"], data["y_val"]),
            epochs=args.epochs_stage2,
            batch_size=args.batch_size,
            verbose=2,
            callbacks=callbacks_s2
        )
        
    t1 = time.time()

    print("\n>> Évaluation finale...")
    val_acc = full_model.evaluate(data["x_val"],  data["y_val"],  verbose=0)[1]
    test_acc = full_model.evaluate(data["x_test"], data["y_test"], verbose=0)[1]
    
    return {
        "full_model": full_model, "history": history, "val_acc": val_acc,
        "test_acc": test_acc, "time_end2end": t1 - t0,
    }


def save_results_as_json(path: str, args: argparse.Namespace, data: dict, s1: dict, s2: dict) -> None:
    """Collecte toutes les informations et les sauvegarde dans un fichier JSON."""
    if not path:
        return

    # Prédictions et matrice de confusion (avec TTA)
    print(">> Évaluation avec Test-Time Augmentation (TTA)...")

    # 1. Prédictions sur les images originales
    y_pred_probs_original = s2["full_model"].predict(data["x_test"])

    # 2. Prédictions sur les images retournées horizontalement
    x_test_flipped = tf.image.flip_left_right(data["x_test"])
    y_pred_probs_flipped = s2["full_model"].predict(x_test_flipped)

    # 3. Moyenne des prédictions
    y_pred_probs = (y_pred_probs_original + y_pred_probs_flipped) / 2.0

    # 4. Le reste ne change pas
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(data["y_test_labels"], y_pred_labels)

    hist_s1 = _history_to_dict(s1["history"])
    hist_s2 = _history_to_dict(s2["history"])
    
    timing = {
        "feature_extract": float(s1["time_feature_extract"]),
        "top_train": float(s1["time_top_train"]),
        "end2end": float(s2["time_end2end"]),
        "total": float(s1["time_feature_extract"] + s1["time_top_train"] + s2["time_end2end"])
    }
    
    # --- MODIFICATION DE LA SECTION 'config' ---
    config = {
        "base_model": "MobileNetV2", 
        "img_size": TARGET_SIZE[0],
        # Utilise vars(args) pour récupérer tous les arguments automatiquement
        **vars(args)
    }
    # Convertit 'augment' en une chaîne "True"/"False" pour le JSON
    config['augment'] = str(config.get('augment', False))
    # --- FIN DE LA MODIFICATION ---

    out = {
        "config": config,
        "metrics": {"val_end2end": float(s2["val_acc"]), "test_end2end": float(s2["test_acc"])},
        "curves": {"stage1": hist_s1, "stage2": hist_s2},
        "timing_sec": timing,
        "confusion_matrix": cm.tolist()
    }
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[JSON] Résultats sauvegardés -> {path}")


# -------------------------
# Fonctions d'Augmentation
# -------------------------


def mixup_batch(x_batch, y_batch, alpha=0.2):
    """
    Applique Mixup sur un batch d'images et de labels.
    
    Args:
        x_batch: Batch d'images (batch_size, H, W, C)
        y_batch: Batch de labels one-hot (batch_size, num_classes)
        alpha: Paramètre de la distribution Beta
    
    Returns:
        x_mixed, y_mixed: Images et labels mixés
    """
    batch_size = tf.shape(x_batch)[0]
    
    # Génère lambda depuis une distribution Beta
    lambda_param = np.random.beta(alpha, alpha)
    lambda_param = max(lambda_param, 1 - lambda_param)  # Pour garder lambda >= 0.5
    
    # Mélange aléatoire des indices
    indices = tf.random.shuffle(tf.range(batch_size))
    x_batch_shuffled = tf.gather(x_batch, indices)
    y_batch_shuffled = tf.gather(y_batch, indices)
    
    # Mixup
    x_mixed = lambda_param * x_batch + (1 - lambda_param) * x_batch_shuffled
    y_mixed = lambda_param * y_batch + (1 - lambda_param) * y_batch_shuffled
    
    return x_mixed, y_mixed


class MixupDataGenerator(keras.utils.Sequence):
    """Générateur de données avec Mixup et augmentation optionnelle."""
    
    def __init__(self, x, y, batch_size, datagen=None, alpha=0.2, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.datagen = datagen
        self.alpha = alpha
        self.shuffle = shuffle
        self.indices = np.arange(len(x))
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.x) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = self.x[batch_indices]
        y_batch = self.y[batch_indices]
        
        # Applique l'augmentation classique si un datagen est fourni
        if self.datagen is not None:
            x_batch = np.array([self.datagen.random_transform(img) for img in x_batch])
        
        # Applique Mixup
        x_mixed, y_mixed = mixup_batch(x_batch, y_batch, self.alpha)
        
        return x_mixed, y_mixed
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# -------------------------
# Exécution Principale
# -------------------------

def main():
    """Orchestre le workflow complet d'entraînement et de sauvegarde."""
    ap = argparse.ArgumentParser(description="Baseline de Transfer Learning sur CIFAR-10")
    ap.add_argument("--n_unfreeze", type=int, default=0, help="Nombre de couches à dé-geler")
    ap.add_argument("--epochs_stage1", type=int, default=10)
    ap.add_argument("--epochs_stage2", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json_output_path", type=str, default="", help="Chemin pour sauvegarder le résumé JSON")

    ap.add_argument("--augment", action='store_true', help="Activer la data augmentation")
    ap.add_argument("--rotation", type=int, default=15, help="Plage de rotation en degrés")
    ap.add_argument("--zoom", type=float, default=0.1, help="Plage de zoom")
    ap.add_argument("--width_shift", type=float, default=0.1, help="Plage de décalage en largeur")
    ap.add_argument("--height_shift", type=float, default=0.1, help="Plage de décalage en hauteur")

    ap.add_argument("--use_cutout", action='store_true', help="Activer l'augmentation Cutout")
    ap.add_argument("--use_mixup", action='store_true', help="Activer l'augmentation Mixup")
    ap.add_argument("--mixup_alpha", type=float, default=0.2, help="Paramètre alpha pour Mixup")

    args = ap.parse_args()

    set_seed(args.seed)
    
    # Chaque étape est maintenant une simple fonction
    data = load_and_prepare_data()
    stage1_results = run_stage1_training(data, args)
    stage2_results = run_stage2_finetuning(data, args, stage1_results)
    
    # Sauvegarde du modèle final et des résultats JSON
    stage2_results["full_model"].save("cifar10.h5")
    print("\n[SAVE] Modèle sauvegardé -> cifar10.h5")
    
    save_results_as_json(args.json_output_path, args, data, stage1_results, stage2_results)


if __name__ == "__main__":
    main()
