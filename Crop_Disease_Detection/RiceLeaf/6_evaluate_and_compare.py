# ============================================================
# Rice Leaf Disease Detection - Model Comparison
# Run this AFTER all 5 models have been trained
# Loads all saved models and generates a comparison report
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

# --- UPDATE THESE PATHS ---
TEST_DIR    = "/kaggle/input/rice-leaf-disease-dataset/test"
WORKING_DIR = "/kaggle/working"
OUTPUT_DIR  = os.path.join(WORKING_DIR, "model_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Brown Spot", "Healthy", "Hispa", "Leaf Blast", "Leaf Scald"]

# Model registry: name -> (model_path, uses_custom_preprocess, img_size)
MODELS = {
    "EfficientNetB0": (
        os.path.join(WORKING_DIR, "efficientnet_b0/efficientnet_b0_final.keras"),
        None, (224, 224)
    ),
    "EfficientNetB3": (
        os.path.join(WORKING_DIR, "efficientnet_b3/efficientnet_b3_final.keras"),
        None, (300, 300)
    ),
    "MobileNetV3Large": (
        os.path.join(WORKING_DIR, "mobilenetv3_large/mobilenetv3_large_final.keras"),
        None, (224, 224)
    ),
    "ResNet50": (
        os.path.join(WORKING_DIR, "resnet50/resnet50_final.keras"),
        resnet_preprocess, (224, 224)
    ),
    "VGG16": (
        os.path.join(WORKING_DIR, "vgg16/vgg16_final.keras"),
        vgg_preprocess, (224, 224)
    ),
}

# ============================================================
# EVALUATE EACH MODEL
# ============================================================
results = []

for model_name, (model_path, preprocess_fn, img_size) in MODELS.items():
    if not os.path.exists(model_path):
        print(f"[SKIP] {model_name} - model file not found: {model_path}")
        continue

    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Create test generator
    if preprocess_fn:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    test_gen = datagen.flow_from_directory(
        TEST_DIR,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)

    # Predictions
    test_gen.reset()
    preds = model.predict(test_gen, verbose=1)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # Per-class metrics
    report = classification_report(true_classes, pred_classes,
                                   target_names=class_labels, output_dict=True)

    # Model size
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    results.append({
        "Model": model_name,
        "Test Accuracy": round(test_acc * 100, 2),
        "Test Loss": round(test_loss, 4),
        "Model Size (MB)": round(model_size_mb, 1),
        "Macro F1": round(report['macro avg']['f1-score'] * 100, 2),
        "Weighted F1": round(report['weighted avg']['f1-score'] * 100, 2),
        "pred_classes": pred_classes,
        "true_classes": true_classes,
        "class_labels": class_labels,
        "report": report
    })

    print(f"  Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f} | Size: {model_size_mb:.1f}MB")
    print(classification_report(true_classes, pred_classes, target_names=class_labels))

    # Free memory
    del model
    tf.keras.backend.clear_session()

# ============================================================
# COMPARISON TABLE
# ============================================================
df = pd.DataFrame([{k: v for k, v in r.items()
                    if k not in ['pred_classes', 'true_classes', 'class_labels', 'report']}
                   for r in results])

df_sorted = df.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
df_sorted.index += 1  # Rank starts at 1

print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(df_sorted.to_string())

df_sorted.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=True)

# ============================================================
# COMPARISON PLOTS
# ============================================================
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
model_names = [r["Model"] for r in results]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Rice Leaf Disease - Model Comparison", fontsize=16, fontweight='bold')

# 1. Accuracy Bar Chart
accs = [r["Test Accuracy"] for r in results]
bars = axes[0, 0].bar(model_names, accs, color=colors, edgecolor='white', linewidth=0.5)
axes[0, 0].set_title("Test Accuracy (%)", fontweight='bold')
axes[0, 0].set_ylim([min(accs) - 5, 100])
axes[0, 0].set_ylabel("Accuracy (%)")
for bar, acc in zip(bars, accs):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=15)
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. F1 Score Comparison
macro_f1  = [r["Macro F1"] for r in results]
weight_f1 = [r["Weighted F1"] for r in results]
x = np.arange(len(model_names))
w = 0.35
axes[0, 1].bar(x - w/2, macro_f1, w, label='Macro F1', color='#2196F3', alpha=0.8)
axes[0, 1].bar(x + w/2, weight_f1, w, label='Weighted F1', color='#4CAF50', alpha=0.8)
axes[0, 1].set_title("F1 Scores (%)", fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(model_names, rotation=15)
axes[0, 1].set_ylabel("F1 Score (%)")
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Model Size vs Accuracy (scatter)
sizes = [r["Model Size (MB)"] for r in results]
scatter = axes[1, 0].scatter(sizes, accs, s=200, c=colors, zorder=5, edgecolors='white', linewidth=1.5)
for i, name in enumerate(model_names):
    axes[1, 0].annotate(name, (sizes[i], accs[i]),
                         textcoords="offset points", xytext=(8, 4), fontsize=9)
axes[1, 0].set_title("Model Size vs Accuracy", fontweight='bold')
axes[1, 0].set_xlabel("Model Size (MB)")
axes[1, 0].set_ylabel("Test Accuracy (%)")
axes[1, 0].grid(True, alpha=0.3)

# 4. Per-class F1 for best model
best = max(results, key=lambda r: r["Test Accuracy"])
class_f1 = [best["report"][cls]['f1-score'] * 100
            for cls in best["class_labels"] if cls in best["report"]]
axes[1, 1].barh(best["class_labels"], class_f1, color=colors[:len(class_f1)])
axes[1, 1].set_title(f"Per-Class F1: {best['Model']} (Best Model)", fontweight='bold')
axes[1, 1].set_xlabel("F1 Score (%)")
axes[1, 1].set_xlim([0, 105])
for i, v in enumerate(class_f1):
    axes[1, 1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_charts.png"), dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CONFUSION MATRICES - side by side for all models
# ============================================================
n = len(results)
cols = 3
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
axes = axes.flatten()

cmaps = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds']

for i, r in enumerate(results):
    cm = confusion_matrix(r["true_classes"], r["pred_classes"])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[i % len(cmaps)],
                xticklabels=r["class_labels"], yticklabels=r["class_labels"],
                ax=axes[i], cbar=False)
    axes[i].set_title(f'{r["Model"]}\nAcc: {r["Test Accuracy"]}%', fontweight='bold')
    axes[i].set_ylabel('True')
    axes[i].set_xlabel('Predicted')
    axes[i].tick_params(axis='x', rotation=30)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Confusion Matrices - All Models", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "all_confusion_matrices.png"), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nBest Model: {best['Model']} with {best['Test Accuracy']}% accuracy")
print(f"\nAll comparison outputs saved to: {OUTPUT_DIR}")
print("Done!")
