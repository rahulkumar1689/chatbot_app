# AI-Powered Customer Support Chatbot

An intelligent, lightweight customer support chatbot built with Python, Natural Language Processing (NLP), Scikit-learn, and Streamlit. This application answers user questions regarding an organization or institution by matching query intent against a structured knowledge base using TF-IDF vectorization and cosine similarity.

---

## Features

- **Interactive Chat Interface**: Modern UI powered by Streamlit's chat components.
- **Intent & Keyword Matching**: Uses `TfidfVectorizer` and `cosine_similarity` to identify relevant answers even if phrasing differs.
- **Graceful Fallback**: Detects unknown questions when similarity falls below a customizable confidence threshold.
- **Automatic Unhandled Question Logging**: Tracks unanswered user queries and frequency in real time.
- **Admin Dashboard**: Built-in sidebar to monitor top unanswered questions to help continuously improve the knowledge base.

---

## Project Structure

```text
chatbot-app/
│-- app.py                      # Main Streamlit application
│-- knowledge_base.json         # FAQ data store (Questions & Answers)
│-- unhandled_questions.json   # Auto-generated log for unanswered queries
│-- requirements.txt            # Project dependencies
└── README.md                   # Project documentation