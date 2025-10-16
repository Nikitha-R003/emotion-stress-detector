#!/usr/bin/env python3
"""
Custom Emotion Detection Model Inference
Loads and uses the trained custom model for emotion classification
"""

import json
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import re

class EmotionLSTM(nn.Module):
    """LSTM-based emotion classification model (same as training)"""
    
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

class CustomEmotionDetector:
    """Custom emotion detection using trained model"""
    
    def __init__(self, model_path='best_emotion_model.pth', model_info_path='model_info.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab = None
        self.label_encoder_classes = None
        self.vocab_size = None
        self.num_classes = None
        
        # Load model info
        try:
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            self.vocab = model_info['vocab']
            self.label_encoder_classes = model_info['label_encoder_classes']
            self.vocab_size = model_info['vocab_size']
            self.num_classes = model_info['num_classes']
            
            # Load model
            self.model = EmotionLSTM(
                vocab_size=self.vocab_size,
                embedding_dim=128,
                hidden_dim=256,
                num_classes=self.num_classes,
                num_layers=2,
                dropout=0.3
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            print(f"Custom emotion model loaded successfully!")
            print(f"Model accuracy: {model_info.get('accuracy', 'Unknown'):.4f}")
            print(f"Device: {self.device}")
            
        except Exception as e:
            print(f"Error loading custom model: {e}")
            print("Falling back to rule-based detection...")
            self.model = None
    
    def preprocess_text(self, text, max_length=128):
        """Preprocess text for model input"""
        if not text:
            return torch.zeros(max_length, dtype=torch.long)
        
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()[:max_length]
        
        # Convert to indices
        token_ids = [self.vocab.get(token, 1) for token in tokens]  # 1 for <UNK>
        
        # Pad to max_length
        if len(token_ids) < max_length:
            token_ids += [0] * (max_length - len(token_ids))
        else:
            token_ids = token_ids[:max_length]
        
        return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    
    def detect_emotion(self, text):
        """Detect emotion from text"""
        if self.model is None:
            return self._fallback_detection(text)
        
        try:
            # Preprocess text
            input_tensor = self.preprocess_text(text).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion = self.label_encoder_classes[predicted.item()]
                confidence_score = confidence.item()
            
            return emotion, confidence_score
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return self._fallback_detection(text)
    
    def _fallback_detection(self, text):
        """Fallback rule-based emotion detection"""
        if not text:
            return 'neutral', 0.5
        
        text_lower = text.lower()
        
        # Emotion keywords mapping
        emotion_keywords = {
            'joy': ['happy', 'joyful', 'excited', 'cheerful', 'delighted', 'ecstatic', 'thrilled', 'elated', 'blissful', 'content', 'great', 'amazing', 'wonderful', 'fantastic'],
            'sadness': ['sad', 'depressed', 'down', 'melancholy', 'gloomy', 'heartbroken', 'dejected', 'miserable', 'sorrowful', 'blue', 'terrible', 'awful', 'horrible'],
            'anger': ['angry', 'furious', 'mad', 'irritated', 'frustrated', 'enraged', 'livid', 'outraged', 'annoyed', 'hostile', 'hate', 'disgusting'],
            'fear': ['scared', 'afraid', 'terrified', 'anxious', 'worried', 'nervous', 'frightened', 'panicked', 'alarmed', 'petrified', 'worried'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered', 'startled', 'dumbfounded', 'flabbergasted', 'wow', 'unbelievable'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated', 'appalled', 'horrified', 'offended', 'disturbed', 'gross'],
            'love': ['loving', 'adoring', 'cherishing', 'devoted', 'affectionate', 'caring', 'tender', 'romantic', 'passionate', 'fond', 'love'],
            'anxiety': ['anxious', 'worried', 'nervous', 'uneasy', 'restless', 'tense', 'stressed', 'overwhelmed', 'panicked', 'agitated'],
            'calm': ['calm', 'peaceful', 'serene', 'tranquil', 'relaxed', 'composed', 'collected', 'centered', 'balanced', 'zen', 'fine', 'okay'],
            'excitement': ['excited', 'thrilled', 'enthusiastic', 'eager', 'pumped', 'energized', 'animated', 'vibrant', 'lively', 'buzzing'],
            'shame': ['ashamed', 'embarrassed', 'guilty', 'humiliated', 'mortified', 'disgraced', 'chagrined', 'sheepish', 'remorseful', 'contrite'],
            'gratitude': ['grateful', 'thankful', 'appreciative', 'blessed', 'indebted', 'obliged', 'moved', 'touched', 'humbled', 'gratified', 'thanks']
        }
        
        # Count keyword matches
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score
        
        # Find emotion with highest score
        if emotion_scores:
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[best_emotion]
            
            if max_score > 0:
                confidence = min(0.9, 0.5 + (max_score * 0.1))
                return best_emotion, confidence
        
        return 'neutral', 0.5

class CustomSentimentAnalyzer:
    """Custom sentiment analysis using simple rules"""
    
    def __init__(self):
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 'brilliant',
            'perfect', 'beautiful', 'lovely', 'nice', 'sweet', 'happy', 'joyful', 'pleased', 'satisfied',
            'love', 'like', 'enjoy', 'appreciate', 'admire', 'respect', 'proud', 'grateful', 'thankful'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike', 'angry', 'mad',
            'furious', 'sad', 'depressed', 'miserable', 'disappointed', 'frustrated', 'annoyed', 'upset',
            'worried', 'scared', 'afraid', 'nervous', 'anxious', 'stressed', 'overwhelmed', 'confused'
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if not text:
            return 'neutral', 0.5
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count > negative_count:
            return 'positive', min(0.9, 0.5 + (positive_count * 0.1))
        elif negative_count > positive_count:
            return 'negative', min(0.9, 0.5 + (negative_count * 0.1))
        else:
            return 'neutral', 0.5

# Global instances
emotion_detector = None
sentiment_analyzer = None

def initialize_models():
    """Initialize the custom models"""
    global emotion_detector, sentiment_analyzer
    
    try:
        emotion_detector = CustomEmotionDetector()
        sentiment_analyzer = CustomSentimentAnalyzer()
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

def detect_emotion(text):
    """Detect emotion using custom model"""
    global emotion_detector
    if emotion_detector is None:
        emotion_detector = CustomEmotionDetector()
    return emotion_detector.detect_emotion(text)

def analyze_sentiment(text):
    """Analyze sentiment using custom model"""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = CustomSentimentAnalyzer()
    return sentiment_analyzer.analyze_sentiment(text)

def estimate_stress_level(emotion):
    """Estimate stress level based on emotion"""
    stress_mapping = {
        'anger': 'high',
        'fear': 'high',
        'sadness': 'medium',
        'disgust': 'high',
        'joy': 'low',
        'anxiety': 'high',
        'excitement': 'low',
        'love': 'low',
        'shame': 'high',
        'surprise': 'medium',
        'calm': 'low',
        'gratitude': 'low',
        'neutral': 'medium'
    }
    return stress_mapping.get(emotion.lower(), 'medium')

def calculate_wellness_score(emotion, stress_level, sentiment):
    """Calculate wellness score based on emotion, stress, and sentiment"""
    emotion_scores = {
        'joy': 90, 'love': 90, 'gratitude': 85, 'calm': 80, 'excitement': 85,
        'surprise': 70, 'sadness': 40, 'anxiety': 30, 'anger': 25, 'fear': 20,
        'disgust': 30, 'shame': 35, 'neutral': 60
    }
    
    stress_scores = {'low': 80, 'medium': 50, 'high': 20}
    sentiment_scores = {'positive': 90, 'neutral': 50, 'negative': 20}
    
    emotion_score = emotion_scores.get(emotion.lower(), 50)
    stress_score = stress_scores.get(stress_level, 50)
    sentiment_score = sentiment_scores.get(sentiment, 50)
    
    # Weighted average
    wellness_score = (emotion_score * 0.4 + stress_score * 0.4 + sentiment_score * 0.2)
    return round(wellness_score, 1)
