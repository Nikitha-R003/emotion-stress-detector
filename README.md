# AI Mental Wellness Companion

A fun, intelligent, and visually appealing web app for emotion and stress detection using Transformer models.

## Features

- **Emotion Detection**: Uses the "bhadresh-savani/distilbert-base-uncased-emotion" model to detect emotions like joy, sadness, anger, fear, surprise, disgust, and neutral.
- **Stress Level Estimation**: Estimates stress levels (low, medium, high) based on detected emotions.
- **User-Friendly Interface**: Clean, modern Streamlit UI with friendly messages and emojis.
- **Instant Analysis**: Real-time analysis of user input sentences.

## Installation

1. Clone or download the project.
2. Navigate to the project directory: `cd emotion_stress_detector`
3. Install dependencies: `pip install -r requirements.txt`

## Usage

Run the app with: `streamlit run app.py`

Open your browser to the provided local URL and start analyzing your emotions!

## How It Works

1. Enter a sentence describing your day or feelings.
2. Click "Analyze My Vibe".
3. View your detected emotion, confidence score, stress level, and personalized friendly message.

## Technologies Used

- Streamlit for the web interface
- Hugging Face Transformers for emotion detection
- PyTorch as the backend for the model

## Future Enhancements

- Add more detailed stress management tips
- Integrate with journaling features
- Support for multiple languages
- Historical tracking of emotions over time
