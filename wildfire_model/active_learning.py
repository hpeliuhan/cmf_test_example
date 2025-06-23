# -----------------------------------------------------------------------------
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import shutil
import logging
from typing import List, Dict, Any
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def load_fire_model(model_path: str):
    tf.keras.backend.clear_session()
    logger.info(f"Loading model from {model_path}")
    return load_model(model_path)


def get_image_files(directory: str, exts=("jpg", "png"), exclude_bases=None) -> list:
    files = []
    for ext in exts:
        pattern = os.path.join(directory, f"*.{ext}")
        for f in glob(pattern):
            if exclude_bases:
                base = os.path.splitext(os.path.basename(f))[0].split("_aug")[0].split("_orig")[0]
                if base in exclude_bases:
                    continue
            files.append(f)
    logger.info(f"Found {len(files)} files in {directory} (excluded: {len(exclude_bases) or set()})")
    return files


def group_augmentations(image_files: list) -> dict:
    groups = {}
    for path in image_files:
        name = os.path.basename(path)
        base = name.split("_aug")[0].split("_orig")[0]
        groups.setdefault(base, []).append(path)
    valid = {b: p for b, p in groups.items() if len(p) > 1}
    logger.info(f"Grouped into {len(valid)} augmentation sets")
    return valid


def preprocess_image(img_path: str, size=(128, 128)) -> np.ndarray:
    img = image.load_img(img_path, target_size=size)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, 0)

# -----------------------------------------------------------------------------
# Variance Analysis and Selection
# -----------------------------------------------------------------------------
def compute_variances(model, frames_path: str, exclude_bases=None) -> pd.DataFrame:
    files = get_image_files(frames_path, exclude_bases=exclude_bases)
    groups = group_augmentations(files)
    records = []
    for base, paths in groups.items():
        preds = [
            float(model.predict(preprocess_image(p), verbose=0).flatten()[0])
            for p in paths
        ]
        records.append({
            'base': base,
            'paths': paths,
            'mean_pred': np.mean(preds),
            'var_pred': np.var(preds),
            'preds': preds
        })
    df = pd.DataFrame(records)
    logger.info("Computed variances for all groups")
    return df.sort_values('var_pred', ascending=False).reset_index(drop=True)


def select_and_label(df: pd.DataFrame, top_k=10, low_k=10, thresh=0.5) -> pd.DataFrame:
    logger.info(f"Selecting top {top_k} high-variance and bottom {low_k} low-variance samples")
    high = df.head(top_k).copy()
    low = df.tail(low_k).copy()
    sel = pd.concat([high, low]).reset_index(drop=True)
    sel['ground_truth'] = (sel['mean_pred'] > thresh).astype(int)
    logger.info(f"Assigned ground truth labels based on threshold {thresh}")
    return sel

# -----------------------------------------------------------------------------
# Dataset Preparation
# -----------------------------------------------------------------------------
def prepare_dataset(labeled_df: pd.DataFrame, out_dir: str):
    fire_dir = os.path.join(out_dir, 'fire')
    nofire_dir = os.path.join(out_dir, 'no_fire')
    os.makedirs(fire_dir, exist_ok=True)
    os.makedirs(nofire_dir, exist_ok=True)
    for _, row in labeled_df.iterrows():
        src = row['paths'][0]
        tgt = fire_dir if row['ground_truth'] else nofire_dir
        dst = os.path.join(tgt, f"{row['base']}.jpg")
        shutil.copy2(src, dst)
    logger.info(f"Prepared dataset in {out_dir} (fire: {len(os.listdir(fire_dir))}, no_fire: {len(os.listdir(nofire_dir))})")

# -----------------------------------------------------------------------------
# Save Changed Prediction Examples
# -----------------------------------------------------------------------------
def save_changed_examples(before_df: pd.DataFrame, after_df: pd.DataFrame,
                           iteration: int, test_dir: str,
                           output_dir: str, max_examples: int = 2):
    merged = before_df.merge(after_df, on='image', suffixes=('_orig','_finetuned'))
    changed = merged[merged['pred_orig'] != merged['pred_finetuned']]
    out_folder = os.path.join(output_dir, f'iteration_{iteration}')
    os.makedirs(out_folder, exist_ok=True)
    examples = changed.head(max_examples)
    for idx, row in examples.iterrows():
        img_path = os.path.join(test_dir, row['image'])
        img = plt.imread(img_path)
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(img)
        axes[0].set_title(f"Orig: {row['pred_orig']}")
        axes[0].axis('off')
        axes[1].imshow(img)
        axes[1].set_title(f"New: {row['pred_finetuned']}")
        axes[1].axis('off')
        fig.tight_layout()
        save_path = os.path.join(out_folder, f"{row['image']}_iter{iteration}.png")
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Saved changed example to {save_path}")
    return examples

# -----------------------------------------------------------------------------
# Fine-tuning and Inference with Flexible Loss
# -----------------------------------------------------------------------------
def fine_tune_model(base_model, samples_dir: str, epochs=5, batch_size=16):
    logger.info("Starting fine-tuning")
    out_units = base_model.output_shape[-1]
    model = tf.keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())
    if out_units == 1:
        loss = 'binary_crossentropy'; class_mode='binary'
    else:
        loss = 'sparse_categorical_crossentropy'; class_mode='sparse'
    model.compile(optimizer=Adam(1e-3), loss=loss, metrics=['accuracy'])
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(samples_dir, target_size=(128,128),
        batch_size=batch_size, class_mode=class_mode, subset='training')
    val_gen = datagen.flow_from_directory(samples_dir, target_size=(128,128),
        batch_size=batch_size, class_mode=class_mode, subset='validation')
    callbacks = [ModelCheckpoint('al_best.h5', save_best_only=True, monitor='val_loss'),
                 EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks, verbose=1)
    if os.path.exists('al_best.h5'):
        logger.info("Loading best model checkpoint")
        return load_model('al_best.h5')
    logger.info("Fine-tuning complete")
    return model


def run_inference(model, test_dir: str, exclude_bases=None) -> pd.DataFrame:
    logger.info(f"Running inference on {test_dir} (excluding {len(exclude_bases) or set()})")
    files = get_image_files(test_dir, exclude_bases=exclude_bases)
    results = []
    out_units = model.output_shape[-1]
    for f in files:
        probs = model.predict(preprocess_image(f), verbose=0).flatten()
        if out_units == 1:
            prob = float(probs[0]); pred = int(prob > 0.5)
        else:
            pred = int(np.argmax(probs)); prob = float(probs[pred])
        results.append({'image': os.path.basename(f), 'pred': pred, 'prob': prob})
    df = pd.DataFrame(results)
    logger.info(f"Completed inference: {len(df)} samples")
    return df

# -----------------------------------------------------------------------------
# Compare Results
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def compare_results(before_df: pd.DataFrame, after_df: pd.DataFrame) -> float:
    merged = before_df.merge(after_df, on='image', suffixes=('_orig','_finetuned'))
    changed = merged[merged['pred_orig'] != merged['pred_finetuned']]
    ratio = len(changed) / len(merged) if len(merged) else 0.0
    logger.info(f"Changed {len(changed)}/{len(merged)} predictions ({ratio*100:.1f}%)")
    return ratio

# -----------------------------------------------------------------------------
# Active Learning Loop with Indexing
# -----------------------------------------------------------------------------
def active_learning_loop(
    model_path: str,
    frames_path: str,
    samples_dir: str,
    test_dir: str,
    iterations: int = 5,
    top_k: int = 10,
    low_k: int = 10
):
    logger.info("=== Active Learning Loop Start ===")
    base_model = load_fire_model(model_path)
    added_bases = set()
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    changed_rates = []
    for i in range(iterations):
        logger.info(f"--- Iteration {i+1}/{iterations} ---")
        # Compute variances excluding already added
        pool_df = compute_variances(base_model, frames_path, exclude_bases=added_bases)
        # Select and label
        sel = select_and_label(pool_df, top_k, low_k)
        # Add to index
        for base in sel['base']:
            added_bases.add(base)
        # Prepare dataset
        prepare_dataset(sel, samples_dir)
        # Fine-tune model
        fine_model = fine_tune_model(base_model, samples_dir)
        # Inference
        before = run_inference(base_model, test_dir, exclude_bases=added_bases)
        after = run_inference(fine_model, test_dir, exclude_bases=added_bases)
        rate = compare_results(before, after)
        changed_rates.append(rate * 100)
        # Save examples
        save_changed_examples(before, after, i+1, test_dir, output_dir)
        base_model = fine_model

    # Plot performance
    iters = list(range(1, iterations+1))
    plt.figure()
    plt.plot(iters, changed_rates, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Changed Predictions (%)')
    plt.title('Active Learning Impact Over Iterations')
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'active_learning_impact.png')
    plt.savefig(plot_path)
    logger.info(f"Saved performance plot to {plot_path}")
    plt.close()

    logger.info("=== Active Learning Loop End ===")
    print(f"Plots and examples available in {output_dir}")
    return changed_rates

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    active_learning_loop(
        model_path="best_model.h5",
        frames_path="../data/extracted_frames_with_augmentations",
        samples_dir="../data/al_samples",
        test_dir="../data/extracted_frames",
        iterations=5,
        top_k=10,
        low_k=10
    )