#!/usr/bin/env python3
"""
Generate Confusion Matrix for Trained Emotion Model
Shows detailed per-emotion accuracy breakdown
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from train_emotion_model import EmotionDatasetGenerator, EmotionLSTM, EmotionDataset, create_vocab

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Generate dataset (same as training)
print("\nGenerating dataset...")
generator = EmotionDatasetGenerator()
df = generator.generate_dataset(samples_per_emotion=300)

print(f"Dataset size: {len(df)} samples")
print(f"Emotions: {df['emotion'].nunique()}")

# Prepare data
texts = df['text']
emotions = df['emotion']

# Create vocabulary
vocab = create_vocab(texts)
vocab_size = len(vocab)

# Encode labels
label_encoder = LabelEncoder()
encoded_emotions = label_encoder.fit_transform(emotions)
num_classes = len(label_encoder.classes_)

print(f"Number of emotions: {num_classes}")
print(f"Emotions: {', '.join(label_encoder.classes_)}")

# Split data (same split as training)
X_train, X_test, y_train, y_test = train_test_split(
    texts, encoded_emotions, test_size=0.2, random_state=42, stratify=encoded_emotions
)

print(f"\nTest set size: {len(X_test)} samples")

# Create test dataset
test_dataset = EmotionDataset(X_test, y_test, vocab)

# Load model
print("\nLoading trained model...")
model = EmotionLSTM(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=256,
    num_classes=num_classes,
    num_layers=2,
    dropout=0.3
).to(device)

model.load_state_dict(torch.load('best_emotion_model.pth', map_location=device))
model.eval()

# Evaluate
print("Evaluating model on test set...")
all_predictions = []
all_labels = []

with torch.no_grad():
    for i in range(len(test_dataset)):
        text_tensor, label = test_dataset[i]
        text_tensor = text_tensor.unsqueeze(0).to(device)
        
        output = model(text_tensor)
        _, predicted = torch.max(output.data, 1)
        
        all_predictions.append(predicted.item())
        all_labels.append(label.item())

# Convert to emotion names
predicted_emotions = label_encoder.inverse_transform(all_predictions)
true_emotions = label_encoder.inverse_transform(all_labels)

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_predictions)
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Generate confusion matrix
cm = confusion_matrix(true_emotions, predicted_emotions, labels=label_encoder.classes_)

# Print classification report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(true_emotions, predicted_emotions, digits=4))

# Calculate per-class accuracy
print("\n" + "="*80)
print("PER-EMOTION ACCURACY")
print("="*80)
for i, emotion in enumerate(label_encoder.classes_):
    correct = cm[i, i]
    total = cm[i, :].sum()
    per_emotion_acc = correct / total if total > 0 else 0
    print(f"{emotion.capitalize():15s}: {per_emotion_acc:.4f} ({per_emotion_acc*100:.2f}%) - {correct}/{total} correct")

# Create confusion matrix heatmap
plt.figure(figsize=(14, 12))
sns.set(font_scale=1.2)

# Normalize confusion matrix for percentages
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Create heatmap with both counts and percentages
annot = np.empty_like(cm, dtype=object)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        pct = cm_normalized[i, j]
        annot[i, j] = f'{count}\n({pct:.1f}%)'

sns.heatmap(
    cm_normalized,
    annot=annot,
    fmt='',
    cmap='Blues',
    xticklabels=[e.capitalize() for e in label_encoder.classes_],
    yticklabels=[e.capitalize() for e in label_encoder.classes_],
    cbar_kws={'label': 'Percentage (%)'},
    linewidths=0.5,
    linecolor='gray'
)

plt.title(f'Confusion Matrix - Emotion Detection Model\nAccuracy: {accuracy*100:.2f}%', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Emotion', fontsize=14, fontweight='bold')
plt.ylabel('True Emotion', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save figure
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix saved as 'confusion_matrix.png'")

# Also create a simpler version with just percentages
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.1f',
    cmap='YlOrRd',
    xticklabels=[e.capitalize() for e in label_encoder.classes_],
    yticklabels=[e.capitalize() for e in label_encoder.classes_],
    cbar_kws={'label': 'Percentage (%)'},
    linewidths=1,
    linecolor='white'
)

plt.title(f'Emotion Classification Accuracy Heatmap\nOverall Accuracy: {accuracy*100:.2f}%', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Emotion', fontsize=14, fontweight='bold')
plt.ylabel('True Emotion', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig('confusion_matrix_percentage.png', dpi=300, bbox_inches='tight')
print(f"Percentage heatmap saved as 'confusion_matrix_percentage.png'")

# Print confusion matrix as table
print("\n" + "="*80)
print("CONFUSION MATRIX (Raw Counts)")
print("="*80)
cm_df = pd.DataFrame(cm, 
                     index=[f"True: {e.capitalize()}" for e in label_encoder.classes_],
                     columns=[f"Pred: {e.capitalize()}" for e in label_encoder.classes_])
print(cm_df.to_string())

print("\n" + "="*80)
print("CONFUSION MATRIX (Percentages)")
print("="*80)
cm_pct_df = pd.DataFrame(cm_normalized, 
                         index=[f"True: {e.capitalize()}" for e in label_encoder.classes_],
                         columns=[f"Pred: {e.capitalize()}" for e in label_encoder.classes_])
print(cm_pct_df.round(2).to_string())

# Identify most confused pairs
print("\n" + "="*80)
print("MOST COMMON MISCLASSIFICATIONS")
print("="*80)

misclassifications = []
for i in range(len(label_encoder.classes_)):
    for j in range(len(label_encoder.classes_)):
        if i != j and cm[i, j] > 0:
            misclassifications.append({
                'True': label_encoder.classes_[i].capitalize(),
                'Predicted': label_encoder.classes_[j].capitalize(),
                'Count': cm[i, j],
                'Percentage': cm_normalized[i, j]
            })

misclassifications_df = pd.DataFrame(misclassifications)
misclassifications_df = misclassifications_df.sort_values('Count', ascending=False).head(10)

for idx, row in misclassifications_df.iterrows():
    print(f"{row['True']:15s} -> {row['Predicted']:15s}: {row['Count']:3d} times ({row['Percentage']:.2f}%)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(f"Total Test Samples: {len(all_labels)}")
print(f"Correctly Classified: {sum(np.array(all_predictions) == np.array(all_labels))}")
print(f"Misclassified: {sum(np.array(all_predictions) != np.array(all_labels))}")
print("\nConfusion matrix images generated:")
print("  - confusion_matrix.png (with counts and percentages)")
print("  - confusion_matrix_percentage.png (percentages only)")

plt.show()

