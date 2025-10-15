import streamlit as st
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import time
import json
from textblob import TextBlob
import nltk
import streamlit.components.v1 as components
from database import db_manager

# Set page config at the very beginning
st.set_page_config(page_title="AI Mental Wellness Companion", page_icon="🧠", layout="wide")

# Custom CSS for light theme
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #000000;
}
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
    color: #000000;
}
[data-testid="stHeader"] {
    background-color: #ffffff;
}
.stTextInput, .stTextArea, .stSelectbox, .stMultiselect {
    background-color: #ffffff;
    color: #000000;
}
.stButton button {
    background-color: #007bff;
    color: white;
}
.stSuccess, .stInfo, .stWarning, .stError {
    background-color: #f8f9fa;
    color: #000000;
}
</style>
""", unsafe_allow_html=True)

nltk.download('punkt', quiet=True)

# Load the emotion detection model
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

emotion_model = load_emotion_model()

# Function to detect emotion and map to user-friendly labels
def detect_emotion(text):
    # Pre-check for sleepy/tired keywords
    sleepy_keywords = ['sleepy', 'tired', 'exhausted', 'drowsy', 'fatigued', 'sleep', 'nap']
    if any(keyword in text.lower() for keyword in sleepy_keywords):
        return 'sleepy', 0.9

    result = emotion_model(text)
    raw_emotion = result[0]['label']
    confidence = result[0]['score']

    # Map model emotions to user-requested emotions
    emotion_mapping = {
        'joy': 'happy',
        'sadness': 'sad',
        'anger': 'angry',
        'fear': 'stressed',  # Map fear to stressed
        'surprise': 'surprised',  # Keep as surprised
        'disgust': 'disgusted',  # Keep as disgusted
        'neutral': 'calm'
    }

    emotion = emotion_mapping.get(raw_emotion.lower(), 'neutral')
    return emotion, confidence

# Function to estimate stress level based on emotion
def estimate_stress_level(emotion):
    stress_mapping = {
        'happy': 'low',
        'sad': 'medium',
        'angry': 'high',
        'stressed': 'high',
        'neutral': 'medium',
        'calm': 'low',
        'sleepy': 'low',
        'surprised': 'medium',
        'disgusted': 'high'
    }
    return stress_mapping.get(emotion.lower(), 'medium')

# Additional AI models for comprehensive analysis
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@st.cache_resource
def load_stress_intensity_model():
    # Using a general purpose model for stress intensity estimation
    return pipeline("text-classification", model="microsoft/DialoGPT-medium")

sentiment_model = load_sentiment_model()
stress_intensity_model = load_stress_intensity_model()

# Function to analyze sentiment polarity
def analyze_sentiment(text):
    result = sentiment_model(text)
    sentiment = result[0]['label']
    confidence = result[0]['score']
    return sentiment, confidence

# Function to calculate wellness score
def calculate_wellness_score(emotion, stress_level, sentiment):
    # Base scores - updated to match new emotion mapping
    emotion_scores = {
        'happy': 90, 'calm': 70, 'neutral': 60,
        'sad': 40, 'angry': 30, 'stressed': 20
    }

    stress_scores = {'low': 80, 'medium': 50, 'high': 20}

    sentiment_scores = {
        'LABEL_2': 90,  # Positive
        'LABEL_1': 50,  # Neutral
        'LABEL_0': 20   # Negative
    }

    emotion_score = emotion_scores.get(emotion.lower(), 50)
    stress_score = stress_scores.get(stress_level, 50)
    sentiment_score = sentiment_scores.get(sentiment, 50)

    # Weighted average
    wellness_score = (emotion_score * 0.4 + stress_score * 0.4 + sentiment_score * 0.2)
    return round(wellness_score, 1)

# Function to generate personalized insights
def generate_insights(emotion, stress_level, sentiment, text):
    insights = []

    # Analyze text for patterns
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if emotion.lower() == 'sad' and polarity < -0.3:
        insights.append("💙 Your words show deep sadness. Consider reaching out to a trusted friend or professional for support.")
    elif emotion.lower() == 'angry' and stress_level == 'high':
        insights.append("🔥 High anger + high stress detected. Try the 4-7-8 breathing technique: inhale for 4 seconds, hold for 7, exhale for 8.")
    elif emotion.lower() == 'stressed' and subjectivity > 0.7:
        insights.append("😨 Your writing shows anxious patterns. Grounding exercises might help - name 5 things you can see, 4 you can touch, etc.")
    elif emotion.lower() == 'happy' and polarity > 0.5:
        insights.append("🎉 You're experiencing genuine happiness! This is great for mental wellness - try to note what contributed to this feeling.")

    if len(text.split()) < 10:
        insights.append("📝 Consider writing more about your feelings - longer entries often reveal deeper insights.")

    return insights

# Function to get coping strategies
def get_coping_strategies(emotion, stress_level):
    strategies = {
        'joy': [
            "🎨 Express your joy through creative activities",
            "📝 Write down what made you happy to remember later",
            "🤝 Share your positive feelings with others"
        ],
        'sadness': [
            "🧘 Practice self-compassion meditation",
            "🚶‍♀️ Take a gentle walk in nature",
            "📞 Call a friend for connection",
            "🎵 Listen to uplifting music"
        ],
        'anger': [
            "🫁 Try deep breathing: 4 counts in, hold 4, out 4",
            "🏃‍♀️ Physical exercise to release tension",
            "✍️ Write down your thoughts without judgment",
            "⏰ Take a timeout before responding"
        ],
        'fear': [
            "🧘‍♀️ Grounding techniques: 5-4-3-2-1 sensory exercise",
            "📋 Break overwhelming tasks into small steps",
            "🗣️ Talk through your fears with someone trusted",
            "📚 Read about others who've overcome similar fears"
        ],
        'surprise': [
            "🤔 Reflect on what surprised you and why",
            "📝 Journal about unexpected events",
            "🎯 Use surprise as an opportunity for growth"
        ],
        'disgust': [
            "🧹 Clean or organize something to regain control",
            "🌱 Focus on what you can change positively",
            "🧘 Practice acceptance of things you can't control"
        ],
        'neutral': [
            "🎯 Set small, achievable goals for the day",
            "📚 Learn something new to stimulate your mind",
            "🤝 Connect with others for social energy"
        ]
    }

    base_strategies = strategies.get(emotion.lower(), strategies['neutral'])

    if stress_level == 'high':
        base_strategies.extend([
            "🧘‍♀️ Guided meditation (try apps like Headspace)",
            "💤 Prioritize sleep and rest",
            "🍎 Eat nourishing foods and stay hydrated",
            "🚫 Limit caffeine and screen time before bed"
        ])
    elif stress_level == 'medium':
        base_strategies.extend([
            "☕ Take short breaks throughout the day",
            "🌳 Spend time in nature",
            "🎵 Listen to calming music"
        ])

    return base_strategies[:5]  # Return top 5 strategies

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

# JavaScript component for localStorage persistence
def local_storage_component(key, data=None):
    """Component to save/load data from browser localStorage"""
    if data is not None:
        # Save data
        data_json = json.dumps(data, default=str)
        js_code = f"""
        <script>
        localStorage.setItem('{key}', '{data_json}');
        </script>
        """
    else:
        # Load data
        js_code = f"""
        <script>
        const data = localStorage.getItem('{key}');
        if (data) {{
            console.log('Loaded {key} from localStorage');
        }}
        </script>
        """

    components.html(js_code, height=0, width=0)

def load_from_local_storage(key):
    """Load data from localStorage via JavaScript"""
    # This is a simplified approach - in practice, we'd need bidirectional communication
    # For now, we'll use a placeholder and implement proper persistence
    return None

def save_to_local_storage(key, data):
    """Save data to localStorage"""
    local_storage_component(key, data)

# Initialize session state for authentication and data persistence
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'mood_history' not in st.session_state:
    st.session_state.mood_history = []
if 'journal_entries' not in st.session_state:
    st.session_state.journal_entries = []

# Authentication check
def check_authentication():
    """Check if user is logged in"""
    return st.session_state.user_id is not None

def login_page():
    """Display login/signup page"""
    st.title("🧠 AI Mental Wellness Companion")
    st.markdown("### Ahoy there! Ready to embark on your emotional adventure?")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.markdown("### Welcome back, old friend!")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", type="primary"):
            if login_username and login_password:
                user_id = db_manager.authenticate_user(login_username, login_password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.username = login_username
                    # Load user data from database
                    st.session_state.mood_history = db_manager.get_mood_history(user_id)
                    st.session_state.journal_entries = db_manager.get_journal_entries(user_id)
                    st.success("🎉 Welcome back, you magnificent human! Your emotional data awaits!")
                    st.rerun()
                else:
                    st.error("🤔 Hmm, those credentials don't ring a bell. Double-check and try again?")
            else:
                st.error("Hey, I need both username and password to let you in!")

    with tab2:
        st.markdown("### Join the emotional intelligence revolution!")
        signup_username = st.text_input("Choose a username", key="signup_username")
        signup_password = st.text_input("Choose a password", type="password", key="signup_password")
        signup_confirm = st.text_input("Confirm password", type="password", key="signup_confirm")

        if st.button("Create Account"):
            if signup_username and signup_password and signup_confirm:
                if signup_password != signup_confirm:
                    st.error("🕵️ Passwords playing hide and seek? They need to match!")
                elif len(signup_username) < 3:
                    st.error("Username too short! Aim for at least 3 characters - make it memorable!")
                elif len(signup_password) < 6:
                    st.error("Password needs some muscle! At least 6 characters, please.")
                elif db_manager.user_exists(signup_username):
                    st.error("That username's taken! Time for creative brainstorming...")
                else:
                    if db_manager.create_user(signup_username, signup_password):
                        st.success("🎊 Account created! You're officially part of the wellness squad. Login to begin!")
                    else:
                        st.error("🤖 Oops! Account creation failed. The robots are having a bad day - try again?")
            else:
                st.error("Fill 'er up! All fields are required for this emotional journey.")

# Main app logic
if not check_authentication():
    login_page()
else:
    # Sidebar with logout option
    st.sidebar.title("🧠 Wellness Hub")
    st.sidebar.markdown(f"**Welcome, {st.session_state.username}!**")

    if st.sidebar.button("Logout"):
        # Save current data to database before logout
        if st.session_state.mood_history:
            for entry in st.session_state.mood_history:
                db_manager.save_mood_entry(st.session_state.user_id, entry)
        if st.session_state.journal_entries:
            for entry in st.session_state.journal_entries:
                db_manager.save_journal_entry(st.session_state.user_id, entry)

        # Clear session state
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.mood_history = []
        st.session_state.journal_entries = []
        st.rerun()

    page = st.sidebar.radio("Navigate", ["Home", "Mood Analysis", "Journal", "Progress Dashboard", "Wellness Tools"])

    # Home page
    if page == "Home":
        st.title("🧠 Your Chill AI Wellness Buddy")
        st.markdown("### Hey there! Ready to chat about how you're feeling?")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            Welcome to your super-friendly mental wellness sidekick! We've got all the cool AI tools to help you understand and manage your emotions:

            🎯 **Multi-Modal Analysis**: We detect emotions, check sentiment, and gauge stress levels
            📊 **Progress Tracking**: See your mood patterns with fun charts over time
            📝 **Therapeutic Journaling**: Write freely and get AI insights on your thoughts
            🛠️ **Wellness Tools**: Get personalized coping strategies and chill exercises
            🎨 **Cheerful Vibes**: Floating characters and friendly interactions to keep it light

            **Let's Get Started:**
            1. Head to "Mood Analysis" to spill the tea on how you're feeling
            2. Try "Journal" to write and uncover some cool insights
            3. Check out "Progress Dashboard" for your emotional journey trends
            4. Explore "Wellness Tools" for tips to feel better
            """)

        with col2:
            st.markdown("### Quick Stats")
            if st.session_state.mood_history:
                recent_emotions = [entry['emotion'] for entry in st.session_state.mood_history[-7:]]
                most_common = max(set(recent_emotions), key=recent_emotions.count) if recent_emotions else "None"
                st.metric("Most Common Emotion (Last 7)", most_common.capitalize())
                st.metric("Total Entries", len(st.session_state.mood_history))
            else:
                st.info("Ready to dive in? Start by analyzing your mood!")

    # Mood Analysis page
    elif page == "Mood Analysis":
        st.title("🎭 Comprehensive Mood Analysis")

        # Input section
        st.markdown("### Spill the tea - how are you really feeling?")
        user_input = st.text_area(
            "Describe your day, thoughts, or emotions:",
            height=120,
            placeholder="e.g., I had a stressful meeting today and I'm feeling overwhelmed..."
        )

        # Voice input option (placeholder for future implementation)
        if st.checkbox("🎤 Enable Voice Input (Coming Soon)"):
            st.info("Voice-to-text integration will be available in the next update!")

        if st.button("🔍 Analyze My Emotional State", type="primary"):
            if user_input.strip():
                with st.spinner("🤖 Running comprehensive AI analysis..."):
                    # Multi-model analysis
                    emotion, emotion_conf = detect_emotion(user_input)
                    sentiment, sentiment_conf = analyze_sentiment(user_input)
                    stress_level = estimate_stress_level(emotion)
                    wellness_score = calculate_wellness_score(emotion, stress_level, sentiment)
                    insights = generate_insights(emotion, stress_level, sentiment, user_input)
                    coping_strategies = get_coping_strategies(emotion, stress_level)

                    # Text analysis
                    blob = TextBlob(user_input)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity

                # Save to history
                entry = {
                    'timestamp': datetime.now(),
                    'text': user_input,
                    'emotion': emotion,
                    'emotion_conf': emotion_conf,
                    'sentiment': sentiment,
                    'stress_level': stress_level,
                    'wellness_score': wellness_score,
                    'polarity': polarity,
                    'subjectivity': subjectivity
                }
                st.session_state.mood_history.append(entry)

                # Results display
                st.success("🎉 Analysis Complete! Here's your comprehensive emotional profile:")

                # Main metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("🎭 Primary Emotion", emotion.capitalize(), f"{emotion_conf:.1%}")
                with col2:
                    st.metric("📊 Sentiment", sentiment.replace("LABEL_", "").replace("2", "Positive").replace("1", "Neutral").replace("0", "Negative"), f"{sentiment_conf:.1%}")
                with col3:
                    st.metric("⚡ Stress Level", stress_level.capitalize())
                with col4:
                    st.metric("🌟 Wellness Score", f"{wellness_score}/100")

                # Detailed analysis
                st.markdown("### 📋 Detailed Analysis")

                tab1, tab2, tab3 = st.tabs(["Emotional Profile", "AI Insights", "Coping Strategies"])

                with tab1:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Text Characteristics:**")
                        st.write(f"📏 Length: {len(user_input.split())} words")
                        st.write(f"📈 Polarity: {polarity:.2f} (-1 to +1)")
                        st.write(f"🎯 Subjectivity: {subjectivity:.2f} (0 to 1)")

                    with col2:
                        st.markdown("**Emotional Breakdown:**")
                        # Simple emotion distribution (placeholder)
                        emotion_data = pd.DataFrame({
                            'Emotion': ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Neutral'],
                            'Intensity': [0.8 if emotion.lower() == 'joy' else 0.1,
                                        0.8 if emotion.lower() == 'sadness' else 0.1,
                                        0.8 if emotion.lower() == 'anger' else 0.1,
                                        0.8 if emotion.lower() == 'fear' else 0.1,
                                        0.8 if emotion.lower() == 'surprise' else 0.1,
                                        0.8 if emotion.lower() == 'disgust' else 0.1,
                                        0.8 if emotion.lower() == 'neutral' else 0.1]
                        })
                        st.bar_chart(emotion_data.set_index('Emotion'))

                with tab2:
                    st.markdown("**🤖 AI-Generated Insights:**")
                    if insights:
                        for insight in insights:
                            st.info(insight)
                    else:
                        st.write("✨ Your emotional state appears balanced. Keep up the good work!")

                    # Wellness interpretation
                    if wellness_score >= 80:
                        st.success("🌟 Excellent mental wellness! You're crushing it emotionally!")
                    elif wellness_score >= 60:
                        st.info("👍 Good mental wellness. Consider the coping strategies below.")
                    elif wellness_score >= 40:
                        st.warning("⚠️ Moderate mental wellness. Pay attention to your emotional needs.")
                    else:
                        st.error("🚨 Low mental wellness detected. Consider professional support.")

                with tab3:
                    st.markdown("**🛠️ Personalized Coping Strategies:**")
                    for i, strategy in enumerate(coping_strategies, 1):
                        st.write(f"{i}. {strategy}")

                    if stress_level == 'high':
                        st.warning("💡 For high stress, try immediate interventions like deep breathing or stepping away from triggers.")

            else:
                st.error("Hey, I need some words to work with! Share your thoughts so I can analyze them!")

    # Journal page
    elif page == "Journal":
        st.title("📝 Therapeutic Journal")

        st.markdown("### Write freely and receive AI-powered insights")

        journal_input = st.text_area(
            "What's on your mind today?",
            height=200,
            placeholder="Write about your day, your feelings, your thoughts... anything that comes to mind."
        )

        if st.button("💭 Get AI Journal Insights"):
            if journal_input.strip():
                with st.spinner("Analyzing your journal entry..."):
                    emotion, _ = detect_emotion(journal_input)
                    sentiment, _ = analyze_sentiment(journal_input)
                    stress_level = estimate_stress_level(emotion)
                    insights = generate_insights(emotion, stress_level, sentiment, journal_input)

                    # Save journal entry
                    journal_entry = {
                        'timestamp': datetime.now(),
                        'content': journal_input,
                        'emotion': emotion,
                        'sentiment': sentiment,
                        'insights': insights
                    }
                    st.session_state.journal_entries.append(journal_entry)

                st.success("Journal entry analyzed! You're doing great at self-reflection!")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Detected Emotion", emotion.capitalize())
                    st.metric("Overall Sentiment", sentiment.replace("LABEL_", "").replace("2", "Positive").replace("1", "Neutral").replace("0", "Negative"))

                with col2:
                    st.metric("Stress Indicators", stress_level.capitalize())
                    st.metric("Word Count", len(journal_input.split()))

                if insights:
                    st.markdown("### 🤖 AI Insights on Your Writing:")
                    for insight in insights:
                        st.info(insight)

                # Journal history
                if len(st.session_state.journal_entries) > 1:
                    st.markdown("### 📚 Recent Journal Entries")
                    for entry in st.session_state.journal_entries[-3:]:
                        with st.expander(f"📅 {entry['timestamp'].strftime('%Y-%m-%d %H:%M')} - {entry['emotion'].capitalize()}"):
                            st.write(entry['content'][:200] + "..." if len(entry['content']) > 200 else entry['content'])
                            if entry['insights']:
                                st.write("**AI Insights:**", entry['insights'][0])

        else:
            st.error("Hey, your journal is calling! Write something to get those insights flowing!")

    # Progress Dashboard
    elif page == "Progress Dashboard":
        st.title("📊 Mental Wellness Progress Dashboard")

        # Data Export/Import Section
        st.markdown("### 💾 Data Management")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📤 Export Your Data"):
                if st.session_state.mood_history or st.session_state.journal_entries:
                    export_data = {
                        'mood_history': st.session_state.mood_history,
                        'journal_entries': st.session_state.journal_entries,
                        'export_date': datetime.now().isoformat()
                    }
                    export_json = json.dumps(export_data, default=str, indent=2)
                    st.download_button(
                        label="📥 Download Data File",
                        data=export_json,
                        file_name=f"mental_wellness_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    st.success("✅ Data export ready! Click the download button above.")
                else:
                    st.warning("No data to export yet. Start by analyzing your mood!")

        with col2:
            uploaded_file = st.file_uploader("📥 Import Previous Data", type=['json'])
            if uploaded_file is not None:
                try:
                    imported_data = json.load(uploaded_file)
                    if 'mood_history' in imported_data:
                        # Merge with existing data, avoiding duplicates
                        existing_timestamps = {entry['timestamp'] for entry in st.session_state.mood_history}
                        new_entries = [entry for entry in imported_data['mood_history']
                                     if entry['timestamp'] not in existing_timestamps]
                        st.session_state.mood_history.extend(new_entries)
                        st.success(f"✅ Imported {len(new_entries)} mood analysis entries!")

                    if 'journal_entries' in imported_data:
                        existing_timestamps = {entry['timestamp'] for entry in st.session_state.journal_entries}
                        new_entries = [entry for entry in imported_data['journal_entries']
                                     if entry['timestamp'] not in existing_timestamps]
                        st.session_state.journal_entries.extend(new_entries)
                        st.success(f"✅ Imported {len(new_entries)} journal entries!")

                except Exception as e:
                    st.error(f"❌ Error importing data: {str(e)}")

        st.markdown("---")

        if not st.session_state.mood_history:
            st.info("Start by analyzing your mood in the 'Mood Analysis' tab to see your progress here!")
            st.markdown("""
            **💡 Pro Tip:** Your data is stored temporarily during your session. To keep your progress between visits:
            - Use the **Export** button above to save your data as a JSON file
            - Use the **Import** button to restore your data when you return
            """)
        else:
            # Convert history to DataFrame
            df = pd.DataFrame(st.session_state.mood_history)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            df['time'] = pd.to_datetime(df['timestamp']).dt.time

            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_wellness = df['wellness_score'].mean()
                st.metric("Avg Wellness Score", f"{avg_wellness:.1f}/100")

            with col2:
                most_common_emotion = df['emotion'].mode().iloc[0]
                st.metric("Dominant Emotion", most_common_emotion.capitalize())

            with col3:
                stress_counts = df['stress_level'].value_counts()
                most_common_stress = stress_counts.index[0]
                st.metric("Common Stress Level", most_common_stress.capitalize())

            with col4:
                total_entries = len(df)
                st.metric("Total Analyses", total_entries)

            # Charts
            st.markdown("### 📈 Mood Trends Over Time")

            # Emotion timeline
            emotion_timeline = df.groupby('date')['emotion'].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 'neutral').reset_index()
            emotion_timeline['emotion_code'] = emotion_timeline['emotion'].map({
                'happy': 5, 'calm': 4, 'neutral': 3,
                'sad': 1, 'angry': 0, 'stressed': 0
            })

            fig_emotion = px.line(emotion_timeline, x='date', y='emotion_code',
                                 title="Emotional Journey",
                                 labels={'emotion_code': 'Emotional State (Higher = More Positive)'})
            fig_emotion.update_yaxes(tickvals=[0,1,3,4,5],
                                    ticktext=['Angry/Stressed', 'Sad', 'Neutral', 'Calm', 'Happy'])
            st.plotly_chart(fig_emotion, use_container_width=True)

            # Wellness score trend
            fig_wellness = px.line(df, x='timestamp', y='wellness_score',
                                  title="Wellness Score Trend",
                                  labels={'wellness_score': 'Wellness Score', 'timestamp': 'Time'})
            st.plotly_chart(fig_wellness, use_container_width=True)

            # Emotion distribution
            col1, col2 = st.columns(2)

            with col1:
                emotion_counts = df['emotion'].value_counts()
                fig_pie = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                               title="Emotion Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                stress_counts = df['stress_level'].value_counts()
                fig_stress = px.bar(x=stress_counts.index, y=stress_counts.values,
                                  title="Stress Level Distribution",
                                  labels={'x': 'Stress Level', 'y': 'Count'})
                st.plotly_chart(fig_stress, use_container_width=True)

            # Insights
            st.markdown("### 💡 AI-Generated Insights")

            # Calculate trends
            recent_scores = df['wellness_score'].tail(5)
            if len(recent_scores) >= 2:
                trend = "improving" if recent_scores.iloc[-1] > recent_scores.iloc[0] else "declining"
                st.info(f"📈 Your wellness trend is {trend} over the last {len(recent_scores)} analyses.")

            # Most improved areas
            if len(df) >= 3:
                avg_emotion_positive = df[df['emotion'].isin(['joy', 'neutral'])]['wellness_score'].mean()
                avg_emotion_negative = df[~df['emotion'].isin(['joy', 'neutral'])]['wellness_score'].mean()

                if pd.notna(avg_emotion_positive) and pd.notna(avg_emotion_negative):
                    if avg_emotion_positive > avg_emotion_negative + 10:
                        st.success("🌟 You tend to have higher wellness scores during positive emotions - great awareness!")
                    elif avg_emotion_negative > avg_emotion_positive + 10:
                        st.warning("⚠️ Your wellness scores are lower during negative emotions. Consider building coping strategies.")

    # Wellness Tools page
    elif page == "Wellness Tools":
        st.title("🛠️ Wellness Tools & Resources")

        st.markdown("### Personalized tools based on your emotional patterns")

        if st.session_state.mood_history:
            # Get user's most common emotion and stress patterns
            recent_emotions = [entry['emotion'] for entry in st.session_state.mood_history[-10:]]
            recent_stress = [entry['stress_level'] for entry in st.session_state.mood_history[-10:]]

            most_common_emotion = max(set(recent_emotions), key=recent_emotions.count) if recent_emotions else 'neutral'
            most_common_stress = max(set(recent_stress), key=recent_stress.count) if recent_stress else 'low'

            st.markdown(f"**Based on your recent patterns:** Most common emotion is **{most_common_emotion}** with **{most_common_stress}** stress levels.")

        # Breathing exercises
        st.markdown("### 🫁 Breathing Exercises")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("4-7-8 Breathing"):
                st.success("**4-7-8 Technique:**\n1. Inhale quietly through nose for 4 seconds\n2. Hold breath for 7 seconds\n3. Exhale through mouth for 8 seconds\n\nRepeat 4 times. Great for stress relief!")

        with col2:
            if st.button("Box Breathing"):
                st.success("**Box Breathing:**\n1. Inhale for 4 counts\n2. Hold for 4 counts\n3. Exhale for 4 counts\n4. Hold for 4 counts\n\nRepeat. Excellent for focus and calm.")

        with col3:
            if st.button("Deep Breathing"):
                st.success("**Deep Breathing:**\n1. Place one hand on belly\n2. Inhale slowly through nose for 4 counts\n3. Feel belly rise\n4. Exhale slowly through mouth for 6 counts\n\nRepeat 5 times.")

        # Mindfulness prompts
        st.markdown("### 🧘 Mindfulness Prompts")

        prompts = [
            "What am I grateful for right now?",
            "What would I say to a friend feeling this way?",
            "What's one small thing I can do to feel better?",
            "What does my body need in this moment?",
            "What strengths have helped me through difficult times before?"
        ]

        if st.button("🎯 Get Random Mindfulness Prompt"):
            prompt = random.choice(prompts)
            st.info(f"**Today's Mindfulness Prompt:** {prompt}")

        # Quick mood boosters
        st.markdown("### ⚡ Quick Mood Boosters")

        boosters = [
            "☀️ Step outside for 5 minutes of sunlight",
            "🎵 Listen to your favorite uplifting song",
            "💃 Dance like nobody's watching for 2 minutes",
            "📞 Call or text someone you care about",
            "🎨 Draw or doodle for 5 minutes",
            "🚶‍♀️ Take a 10-minute walk",
            "📚 Read an inspiring quote or story",
            "🛀 Take a warm shower or bath",
            "🍎 Eat something nourishing",
            "😴 Take a 5-minute power nap"
        ]

        if st.button("🚀 Get Quick Mood Booster"):
            booster = random.choice(boosters)
            st.success(f"**Try this:** {booster}")

        # Emergency resources
        st.markdown("### 🚨 Crisis Resources")

        st.info("""
        **If you're in crisis or having thoughts of self-harm:**

        - **National Suicide Prevention Lifeline**: 988 (US)
        - **Crisis Text Line**: Text HOME to 741741
        - **International Association for Suicide Prevention**: Visit befrienders.org
        - **Emergency Services**: Call 911 (US) or your local emergency number

        Remember: You're not alone, and help is available 24/7.
        """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Hugging Face Transformers, and Plotly")
st.markdown("*This app is for informational purposes only and not a substitute for professional mental health care.*")
