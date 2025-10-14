import streamlit as st
from transformers import pipeline
import random
import time

# Load the emotion detection model
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

emotion_model = load_emotion_model()

# Function to detect emotion
def detect_emotion(text):
    result = emotion_model(text)
    emotion = result[0]['label']
    confidence = result[0]['score']
    return emotion, confidence

# Function to estimate stress level based on emotion
def estimate_stress_level(emotion):
    stress_mapping = {
        'joy': 'low',
        'sadness': 'medium',
        'anger': 'high',
        'fear': 'high',
        'surprise': 'medium',
        'disgust': 'medium',
        'neutral': 'low'
    }
    return stress_mapping.get(emotion.lower(), 'medium')

# Animated chibi characters - larger anime-style
def floating_characters():
    characters = [
        "🐱‍👤", "🐶‍👤", "🐰‍👤", "🐼‍👤", "🦊‍👤", "🐻‍👤", "🐨‍👤", "🐸‍👤",
        "🌸", "🌺", "🌻", "🌼", "🌹", "🌷", "🌻", "🌸"
    ]
    return random.choice(characters)

# CSS for animations
st.markdown("""
<style>
@keyframes float {
    0% { transform: translateY(0px) rotate(0deg); }
    33% { transform: translateY(-10px) rotate(5deg); }
    66% { transform: translateY(5px) rotate(-5deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}

.floating-char {
    animation: float 3s ease-in-out infinite;
    font-size: 4rem;
    display: inline-block;
    margin: 0 10px;
}

.character-container {
    position: fixed;
    top: 20%;
    left: 10%;
    z-index: -1;
    opacity: 0.3;
}

.character-container:nth-child(2) {
    top: 60%;
    left: 80%;
    animation-delay: 1s;
}

.character-container:nth-child(3) {
    top: 40%;
    right: 15%;
    animation-delay: 2s;
}

.character-container:nth-child(4) {
    bottom: 20%;
    left: 70%;
    animation-delay: 0.5s;
}

.character-container:nth-child(5) {
    top: 70%;
    left: 20%;
    animation-delay: 1.5s;
}

@media (max-width: 768px) {
    .character-container {
        display: none;
    }
}
</style>
""", unsafe_allow_html=True)

# Add floating characters
for i in range(5):
    st.markdown(f'<div class="character-container"><span class="floating-char">{floating_characters()}</span></div>', unsafe_allow_html=True)

# Streamlit UI
st.set_page_config(page_title="AI Mental Wellness Companion", page_icon="🧠", layout="centered")

st.title("🧠 AI Mental Wellness Companion")
st.subheader("Detect your emotions and stress levels from a simple sentence!")

st.markdown("""
Welcome to your personal AI companion for mental wellness! Just type or paste a sentence about how your day went, and I'll analyze your emotion and stress level.
""")

# Input text area
user_input = st.text_area("How was your day? Share a sentence:", height=100, placeholder="e.g., I had a great meeting today!")

if st.button("Analyze My Vibe"):
    if user_input.strip():
        with st.spinner("Analyzing your emotions..."):
            emotion, confidence = detect_emotion(user_input)
            stress_level = estimate_stress_level(emotion)

        # Display results with friendly text
        st.success("Analysis complete!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Detected Emotion", emotion.capitalize())
            st.write(f"Confidence: {confidence:.2%}")

        with col2:
            st.metric("Stress Level", stress_level.capitalize())

        # Friendly messages with animated emojis
        if emotion.lower() == 'joy':
            st.write("🌟 Your emotion vibe today looks like pure joy! Keep that positive energy flowing! 🎉")
        elif emotion.lower() == 'sadness':
            st.write("😢 Looks like you're feeling a bit down. Remember, it's okay to feel sad sometimes. Take care of yourself! 🤗")
        elif emotion.lower() == 'anger':
            st.write("😠 Anger detected! Take a deep breath and maybe go for a walk to cool off. 🌬️")
        elif emotion.lower() == 'fear':
            st.write("😨 Fear is showing up. You're brave for facing it – talk to someone if you need support. 💪")
        elif emotion.lower() == 'surprise':
            st.write("😲 Surprise! Life's full of unexpected moments – embrace them! ✨")
        elif emotion.lower() == 'disgust':
            st.write("🤢 Disgust noted. Sometimes things just don't sit right – process those feelings. 🤔")
        else:
            st.write("😐 Feeling neutral today? That's okay – balance is key! ⚖️")

        if stress_level == 'high':
            st.warning("🚨 High stress detected! Take a deep breath — maybe try some mindfulness or talk to a friend. 🧘‍♀️")
        elif stress_level == 'medium':
            st.info("⚠️ Medium stress level. Keep an eye on it and find ways to relax. ☕")
        else:
            st.success("✅ Low stress – you're doing great! Maintain that calm vibe. 😌")

    else:
        st.error("Please enter some text to analyze!")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Hugging Face Transformers")
