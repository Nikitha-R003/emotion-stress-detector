#!/usr/bin/env python3
"""
Custom Emotion Detection Model Training
Trains a model from scratch to achieve 90%+ accuracy on emotion classification
"""

import os
import json
import random
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class EmotionDatasetGenerator:
    """Generate comprehensive synthetic emotion dataset"""
    
    def __init__(self):
        self.emotions = {
            'joy': {
                'keywords': ['happy', 'joyful', 'excited', 'cheerful', 'delighted', 'ecstatic', 'thrilled', 'elated', 'blissful', 'content'],
                'patterns': [
                    "I'm so {word} today!",
                    "This makes me feel {word}",
                    "I'm feeling {word} about this",
                    "What a {word} day!",
                    "I feel {word} and grateful",
                    "This is absolutely {word}!",
                    "I'm {word} beyond words",
                    "Feeling {word} and optimistic"
                ]
            },
            'sadness': {
                'keywords': ['sad', 'depressed', 'down', 'melancholy', 'gloomy', 'heartbroken', 'dejected', 'miserable', 'sorrowful', 'blue'],
                'patterns': [
                    "I'm feeling so {word} today",
                    "This makes me {word}",
                    "I feel {word} and lonely",
                    "I'm {word} about everything",
                    "This is making me {word}",
                    "I feel {word} and hopeless",
                    "I'm {word} and can't stop crying",
                    "Feeling {word} and empty inside"
                ]
            },
            'anger': {
                'keywords': ['angry', 'furious', 'mad', 'irritated', 'frustrated', 'enraged', 'livid', 'outraged', 'annoyed', 'hostile'],
                'patterns': [
                    "I'm so {word} right now!",
                    "This makes me {word}",
                    "I'm {word} about this situation",
                    "I feel {word} and frustrated",
                    "This is making me {word}",
                    "I'm {word} beyond belief",
                    "I feel {word} and want to scream",
                    "I'm {word} and can't take it anymore"
                ]
            },
            'fear': {
                'keywords': ['scared', 'afraid', 'terrified', 'anxious', 'worried', 'nervous', 'frightened', 'panicked', 'alarmed', 'petrified'],
                'patterns': [
                    "I'm so {word} about this",
                    "This makes me feel {word}",
                    "I'm {word} and anxious",
                    "I feel {word} and worried",
                    "This is making me {word}",
                    "I'm {word} and can't sleep",
                    "I feel {word} and helpless",
                    "I'm {word} and shaking"
                ]
            },
            'surprise': {
                'keywords': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered', 'startled', 'dumbfounded', 'flabbergasted', 'gobsmacked'],
                'patterns': [
                    "I'm so {word} by this!",
                    "This is {word}!",
                    "I'm {word} and confused",
                    "I feel {word} and excited",
                    "This is making me {word}",
                    "I'm {word} beyond words",
                    "I feel {word} and speechless",
                    "I'm {word} and can't believe it"
                ]
            },
            'disgust': {
                'keywords': ['disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated', 'appalled', 'horrified', 'offended', 'disturbed', 'grossed out'],
                'patterns': [
                    "I'm so {word} by this",
                    "This makes me feel {word}",
                    "I'm {word} and sickened",
                    "I feel {word} and repulsed",
                    "This is making me {word}",
                    "I'm {word} beyond belief",
                    "I feel {word} and want to vomit",
                    "I'm {word} and can't stand it"
                ]
            },
            'love': {
                'keywords': ['loving', 'adoring', 'cherishing', 'devoted', 'affectionate', 'caring', 'tender', 'romantic', 'passionate', 'fond'],
                'patterns': [
                    "I'm feeling so {word} today",
                    "This makes me feel {word}",
                    "I'm {word} and grateful",
                    "I feel {word} and blessed",
                    "This is making me {word}",
                    "I'm {word} and happy",
                    "I feel {word} and complete",
                    "I'm {word} and at peace"
                ]
            },
            'anxiety': {
                'keywords': ['anxious', 'worried', 'nervous', 'uneasy', 'restless', 'tense', 'stressed', 'overwhelmed', 'panicked', 'agitated'],
                'patterns': [
                    "I'm feeling so {word} today",
                    "This makes me {word}",
                    "I'm {word} and stressed",
                    "I feel {word} and overwhelmed",
                    "This is making me {word}",
                    "I'm {word} and can't relax",
                    "I feel {word} and restless",
                    "I'm {word} and on edge"
                ]
            },
            'calm': {
                'keywords': ['calm', 'peaceful', 'serene', 'tranquil', 'relaxed', 'composed', 'collected', 'centered', 'balanced', 'zen'],
                'patterns': [
                    "I'm feeling so {word} today",
                    "This makes me feel {word}",
                    "I'm {word} and content",
                    "I feel {word} and balanced",
                    "This is making me {word}",
                    "I'm {word} and at ease",
                    "I feel {word} and grounded",
                    "I'm {word} and peaceful"
                ]
            },
            'excitement': {
                'keywords': ['excited', 'thrilled', 'enthusiastic', 'eager', 'pumped', 'energized', 'animated', 'vibrant', 'lively', 'buzzing'],
                'patterns': [
                    "I'm so {word} about this!",
                    "This makes me {word}!",
                    "I'm {word} and ready",
                    "I feel {word} and motivated",
                    "This is making me {word}",
                    "I'm {word} and can't wait",
                    "I feel {word} and alive",
                    "I'm {word} and buzzing with energy"
                ]
            },
            'shame': {
                'keywords': ['ashamed', 'embarrassed', 'guilty', 'humiliated', 'mortified', 'disgraced', 'chagrined', 'sheepish', 'remorseful', 'contrite'],
                'patterns': [
                    "I'm feeling so {word} about this",
                    "This makes me {word}",
                    "I'm {word} and regretful",
                    "I feel {word} and guilty",
                    "This is making me {word}",
                    "I'm {word} and want to hide",
                    "I feel {word} and sorry",
                    "I'm {word} and disappointed in myself"
                ]
            },
            'gratitude': {
                'keywords': ['grateful', 'thankful', 'appreciative', 'blessed', 'indebted', 'obliged', 'moved', 'touched', 'humbled', 'gratified'],
                'patterns': [
                    "I'm feeling so {word} today",
                    "This makes me {word}",
                    "I'm {word} and blessed",
                    "I feel {word} and thankful",
                    "This is making me {word}",
                    "I'm {word} and moved",
                    "I feel {word} and humbled",
                    "I'm {word} and appreciative"
                ]
            }
        }
    
    def generate_dataset(self, samples_per_emotion=500):
        """Generate synthetic emotion dataset"""
        data = []
        
        for emotion, config in self.emotions.items():
            keywords = config['keywords']
            patterns = config['patterns']
            
            for _ in range(samples_per_emotion):
                # Random keyword and pattern
                keyword = random.choice(keywords)
                pattern = random.choice(patterns)
                
                # Generate text
                text = pattern.format(word=keyword)
                
                # Add variations
                variations = [
                    text,
                    text.lower(),
                    text.upper(),
                    text + "!",
                    text + ".",
                    text + "?",
                    "Really " + text.lower(),
                    "I can't believe " + text.lower(),
                    "Honestly, " + text.lower(),
                    "To be honest, " + text.lower()
                ]
                
                for variation in variations:
                    data.append({
                        'text': variation,
                        'emotion': emotion,
                        'length': len(variation.split())
                    })
        
        return pd.DataFrame(data)

class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels[idx]
        
        # Simple tokenization (you can use more sophisticated tokenizers)
        tokens = text.lower().split()[:self.max_length]
        
        # Convert to indices (simple word-to-index mapping)
        token_ids = [self.tokenizer.get(word, 0) for word in tokens]
        
        # Pad to max_length
        if len(token_ids) < self.max_length:
            token_ids += [0] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class EmotionLSTM(nn.Module):
    """LSTM-based emotion classification model"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(EmotionLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        output = self.dropout(lstm_out[:, -1, :])
        output = self.fc(output)
        return output

class EmotionCNN(nn.Module):
    """CNN-based emotion classification model"""
    
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.3):
        super(EmotionCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)  # (batch, embedding_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        concatenated = torch.cat(conv_outputs, dim=1)
        output = self.dropout(concatenated)
        output = self.fc(output)
        return output

def create_vocab(texts, min_freq=2):
    """Create vocabulary from texts"""
    word_freq = Counter()
    for text in texts:
        words = text.lower().split()
        word_freq.update(words)
    
    # Create word-to-index mapping
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the emotion classification model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        scheduler.step(avg_train_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_emotion_model.pth')
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_accuracies, best_val_acc

def evaluate_model(model, test_loader, label_encoder):
    """Evaluate the trained model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert back to emotion labels
    predicted_emotions = label_encoder.inverse_transform(all_predictions)
    true_emotions = label_encoder.inverse_transform(all_labels)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    
    print(f"\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(true_emotions, predicted_emotions))
    
    return accuracy, predicted_emotions, true_emotions

def plot_training_history(train_losses, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline"""
    print("Starting Emotion Detection Model Training")
    print("=" * 50)
    
    # Generate dataset
    print("Generating synthetic emotion dataset...")
    generator = EmotionDatasetGenerator()
    df = generator.generate_dataset(samples_per_emotion=300)  # 300 samples per emotion
    
    print(f"Dataset size: {len(df)} samples")
    print(f"Emotions: {df['emotion'].nunique()}")
    print(f"Emotion distribution:")
    print(df['emotion'].value_counts())
    
    # Prepare data
    print("\nPreparing data...")
    texts = df['text']
    emotions = df['emotion']
    
    # Create vocabulary
    vocab = create_vocab(texts)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_emotions = label_encoder.fit_transform(emotions)
    num_classes = len(label_encoder.classes_)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, encoded_emotions, test_size=0.2, random_state=42, stratify=encoded_emotions
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train, vocab)
    val_dataset = EmotionDataset(X_val, y_val, vocab)
    test_dataset = EmotionDataset(X_test, y_test, vocab)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Train LSTM model
    print(f"\nTraining LSTM model on {device}...")
    lstm_model = EmotionLSTM(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
    
    train_losses, val_accuracies, best_val_acc = train_model(
        lstm_model, train_loader, val_loader, num_epochs=100, learning_rate=0.001
    )
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    # Load best model and evaluate
    lstm_model.load_state_dict(torch.load('best_emotion_model.pth'))
    accuracy, predictions, true_labels = evaluate_model(lstm_model, test_loader, label_encoder)
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies)
    
    # Save model and metadata
    model_info = {
        'vocab': vocab,
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'accuracy': float(accuracy),
        'model_type': 'LSTM'
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Model saved as 'best_emotion_model.pth'")
    print(f"Model info saved as 'model_info.json'")
    
    return lstm_model, vocab, label_encoder, accuracy

if __name__ == "__main__":
    model, vocab, label_encoder, accuracy = main()
