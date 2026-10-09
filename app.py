import json
import os
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
KB_FILE = "knowledge_base.json"
UNHANDLED_FILE = "unhandled_questions.json"
SIMILARITY_THRESHOLD = 0.2  # Minimum similarity score required to return an answer


# --- HELPER FUNCTIONS ---
def load_knowledge_base():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def log_unhandled_question(question):
    unhandled = []
    if os.path.exists(UNHANDLED_FILE):
        with open(UNHANDLED_FILE, "r", encoding="utf-8") as f:
            unhandled = json.load(f)

    # Append question with count frequency
    for item in unhandled:
        if item["question"].lower() == question.lower():
            item["count"] += 1
            break
    else:
        unhandled.append({"question": question, "count": 1})

    with open(UNHANDLED_FILE, "w", encoding="utf-8") as f:
        json.dump(unhandled, f, indent=2)


def get_unhandled_questions():
    if os.path.exists(UNHANDLED_FILE):
        with open(UNHANDLED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def get_response(user_query, kb):
    if not kb:
        return "Knowledge base is empty.", 0.0

    questions = [item["question"] for item in kb]

    # Preprocess and calculate TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(questions + [user_query])

    # Compute cosine similarity between user query (last item) and KB questions
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    if best_score >= SIMILARITY_THRESHOLD:
        return kb[best_match_idx]["answer"], best_score
    else:
        log_unhandled_question(user_query)
        return (
            "I'm sorry, I couldn't find an accurate answer to your question. "
            "It has been logged for our team to review!",
            best_score,
        )


# --- STREAMLIT UI ---
st.set_page_config(
    page_title="Support Chatbot", page_icon="🤖", layout="wide"
)

st.title("🤖 AI Customer Support Chatbot")
st.write("Ask any question regarding our services or institution.")

# Sidebar - Admin Dashboard to track common unhandled questions
with st.sidebar:
    st.header("📊 Admin Insights")
    st.subheader("Logged / Unanswered Questions")
    unhandled_list = get_unhandled_questions()

    if unhandled_list:
        sorted_unhandled = sorted(
            unhandled_list, key=lambda x: x["count"], reverse=True
        )
        for item in sorted_unhandled:
            st.write(f"• **{item['question']}** (Asked {item['count']}x)")
    else:
        st.info("No unhandled questions recorded yet.")

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?",
        }
    ]

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input widget
if user_prompt := st.chat_input("Type your question here..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate response
    kb = load_knowledge_base()
    response_text, score = get_response(user_prompt, kb)

    # Render chatbot response
    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )