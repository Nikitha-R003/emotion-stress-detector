# AI Mental Wellness Companion

A comprehensive, AI-powered mental wellness platform that detects emotions from text using advanced zero-shot classification, providing personalized insights and coping strategies.

## Features

- **Expanded Emotion Detection**: Now supports 24 emotions using Facebook's BART-large-MNLI zero-shot classification model, including: Anger, Fear, Sadness, Disgust, Happiness, Anxiety, Envy, Excitement, Love, Shame, Disappointment, Surprise, AWE, Calmness, Confusion, Empathy, Enjoyment, Gratitude, Joy, Acceptance, Amusement, Anticipation, Contempt, and Contentment.
- **Stress Level Estimation**: Intelligent stress assessment based on detected emotions with personalized coping strategies.
- **Comprehensive Analysis**: Multi-modal analysis combining emotion detection, sentiment analysis, and wellness scoring.
- **Therapeutic Journaling**: AI-powered journaling with personalized insights and emotional tracking.
- **Progress Dashboard**: Visual analytics showing mood trends, emotion distribution, and wellness score progression.
- **Wellness Tools**: Breathing exercises, mindfulness prompts, mood boosters, and crisis resources.
- **User Authentication**: Secure user accounts with data persistence.
- **Data Export/Import**: Backup and restore your emotional journey data.
- **Dark Theme UI**: Modern, accessible interface with floating character animations.

## Installation

1. Clone or download the project.
2. Navigate to the project directory: `cd emotion_stress_detector`
3. Install dependencies: `pip install -r requirements.txt`

## Usage

Run the app with: `streamlit run app.py`

Create an account or login, then explore the various features:
- **Mood Analysis**: Get comprehensive emotional analysis of your text
- **Journal**: Write freely and receive AI insights
- **Progress Dashboard**: Track your emotional journey over time
- **Wellness Tools**: Access personalized coping strategies and exercises

## How It Works

1. **Emotion Detection**: Uses zero-shot classification to identify emotions from text input
2. **Sentiment Analysis**: Analyzes overall sentiment polarity and subjectivity
3. **Stress Assessment**: Maps emotions to stress levels (low, medium, high)
4. **Wellness Scoring**: Calculates a composite wellness score from emotion, stress, and sentiment
5. **Personalized Insights**: Generates tailored advice and coping strategies
6. **Progress Tracking**: Maintains historical data for trend analysis

## Technologies Used

- **Streamlit**: Modern web interface with real-time interactivity
- **Hugging Face Transformers**: Advanced NLP models for emotion and sentiment analysis
- **Facebook BART-large-MNLI**: Zero-shot classification for broad emotion coverage
- **CardiffNLP Twitter RoBERTa**: Sentiment analysis optimized for social media text
- **Plotly**: Interactive data visualizations
- **SQLite**: Local database for user data persistence
- **bcrypt**: Secure password hashing

## Model Details

- **Primary Model**: `facebook/bart-large-mnli` for zero-shot emotion classification
- **Sentiment Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Supported Emotions**: 24 comprehensive emotions covering the full spectrum of human feelings
- **Confidence Scoring**: Provides confidence levels for all predictions

## Future Enhancements

- Voice input integration
- Multi-language support
- Advanced mood prediction algorithms
- Integration with wearable devices
- Professional therapist matching
- Group support communities
