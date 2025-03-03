import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import auth, credentials, firestore
import requests
import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize sentiment analysis
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Set up Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

# Define AI counselor personalities
counselor_personalities = {
    "CBT Therapist": "I use cognitive behavioral therapy techniques to help users challenge negative thoughts.",
    "Mindfulness Coach": "I focus on mindfulness and meditation practices to reduce stress and anxiety.",
    "Solution-Focused Counselor": "I provide actionable solutions to life’s problems using a positive and goal-oriented approach.",
    "Empathetic Listener": "I provide a safe space to talk about emotions and feelings with empathy and compassion."
}

# Streamlit UI configuration
st.set_page_config(page_title="AI Counselor", page_icon="🧠")

# Authentication state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.session_state.conversation_history = []

# Logout function
def logout():
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.session_state.conversation_history = []
    st.experimental_rerun()

# Sidebar authentication UI
if not st.session_state.authenticated:
    st.sidebar.header("Login Required")
    st.sidebar.warning("Please log in to access AI counseling.")
else:
    st.sidebar.success(f"Logged in as {st.session_state.user_email}")
    if st.sidebar.button("Logout"):
        logout()

    # Main dashboard
    st.title("🧠 AI Counseling Chat")
    st.write("Choose a counselor and start your conversation.")

    # Select counselor
    counselor_choice = st.selectbox("Choose an AI Counselor:", list(counselor_personalities.keys()))
    user_input = st.text_area("Write your message here:")

    # Send message to AI
    if st.button("Send Message"):
        if user_input.strip():
            counselor_prompt = f"You are a virtual {counselor_choice}. {counselor_personalities[counselor_choice]} Respond to the user’s concern in a helpful and supportive way."

            # Call Gemini API
            response = genai.generate_text(model="gemini-2.0-flash", prompt=f"{counselor_prompt}\nUser: {user_input}\nAI Counselor:")
            ai_response = response.text.strip()

            # Update conversation history
            st.session_state.conversation_history.append(("User", user_input))
            st.session_state.conversation_history.append((f"{counselor_choice}", ai_response))

            # Store conversation in Firestore
            user_doc = db.collection("users").where("email", "==", st.session_state.user_email).get()
            if user_doc:
                user_ref = user_doc[0].reference
                user_ref.update({"conversations": firestore.ArrayUnion([{"counselor": counselor_choice, "user_message": user_input, "ai_response": ai_response}] )})

        else:
            st.warning("Please enter a message.")

    # Display conversation history
    st.subheader("🗨️ Conversation History")
    for role, message in st.session_state.conversation_history:
        with st.chat_message(role):
            st.markdown(message)

    # Mood tracking & journaling
    st.header("📊 Mood Tracker & Journaling")

    # User enters journal entry
    journal_entry = st.text_area("Write about your day or how you're feeling:")

    if st.button("Analyze Mood & Save Entry"):
        if journal_entry.strip():
            # Perform sentiment analysis
            sentiment_score = sia.polarity_scores(journal_entry)["compound"]
            if sentiment_score >= 0.05:
                mood = "Happy 😊"
            elif sentiment_score <= -0.05:
                mood = "Sad 😢"
            else:
                mood = "Neutral 😐"

            # Store in Firestore
            user_doc = db.collection("users").where("email", "==", st.session_state.user_email).get()
            if user_doc:
                user_ref = user_doc[0].reference
                user_ref.update({
                    "mood_logs": firestore.ArrayUnion([{"mood": mood, "sentiment": sentiment_score, "entry": journal_entry}])
                })

            st.success(f"Mood logged as: {mood}")

        else:
            st.warning("Please enter some text before analyzing.")

    # Mood history visualization
    st.subheader("📈 Mood Trends Over Time")

    # Retrieve mood history
    user_doc = db.collection("users").where("email", "==", st.session_state.user_email).get()
    if user_doc:
        mood_data = user_doc[0].to_dict().get("mood_logs", [])
        
        if mood_data:
            dates = list(range(1, len(mood_data) + 1))
            sentiment_scores = [entry["sentiment"] for entry in mood_data]

            # Plot sentiment trend
            plt.figure(figsize=(6, 3))
            plt.plot(dates, sentiment_scores, marker="o", linestyle="-", color="b")
            plt.axhline(y=0, color="gray", linestyle="--")
            plt.xlabel("Entries")
            plt.ylabel("Sentiment Score")
            plt.title("Mood Sentiment Trend")
            st.pyplot(plt)
        else:
            st.write("No mood history available yet.")

    # View past journal entries
    st.subheader("📜 Past Journal Entries")

    if user_doc:
        journal_entries = user_doc[0].to_dict().get("mood_logs", [])
        
        if journal_entries:
            with st.expander("View Past Journals"):
                for entry in reversed(journal_entries):
                    st.write(f"**Mood:** {entry['mood']}")
                    st.write(f"📝 {entry['entry']}")
                    st.write("---")
        else:
            st.write("No journal entries yet.")
