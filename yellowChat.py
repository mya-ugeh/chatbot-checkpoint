import streamlit as st
import random
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Additional NLTK downloads
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize NLTK lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Read the FAQ data
df = pd.read_csv("AI_FAQ.csv", na_filter=False)
df.rename(columns={'Question': 'Questions', 'Answer': 'Answers'}, inplace=True)
df.drop('Unnamed: 0', axis=1, inplace=True)

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    return ' '.join(preprocessed_sentences)

df['tokenized Questions'] = df['Questions'].apply(preprocess_text)

# Create a corpus by flattening the preprocessed questions
corpus = df['tokenized Questions'].tolist()

# Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(corpus)

# Function to get bot response
def get_response(user_input):
    global most_similar_index
    user_input_processed = preprocess_text(user_input)
    user_input_vector = tfidf_vectorizer.transform([user_input_processed])
    similarity_scores = cosine_similarity(user_input_vector, X)
    most_similar_index = similarity_scores.argmax()
    return df['Answers'].iloc[most_similar_index]

# Greeting and farewell messages
greetings = ["Hey there, I am your assistant. How may I help you?",
             "Hi, how can I assist you today?",
             "Hello, how may I be of help?",
             "Good day! How can I help you?",
             "Hello there, how may I assist you today?"]
exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']
farewell = ['Thanks, see you soon!', 'Goodbye, see you later!', 'Farewell, come back soon!']
random_farewell = random.choice(farewell)
random_greetings = random.choice(greetings)

# Streamlit app
st.markdown("<h1 style='text-align: center; color: white; margin-top: -70px; font-family: Times New Roman;'>Artificial Intelligence Basics Q&A</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

# Function to add background image from local file
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Add background image
add_bg_from_local('am6eiddh.png')

st.markdown("<br><br>", unsafe_allow_html=True)

history = []
st.sidebar.markdown("<h2 style='text-align: center; margin-top: 0rem; color: #64CCC5;'>Chat History</h2>", unsafe_allow_html=True)

user_input = st.text_input('Ask Your Question')
if user_input:
    user_input_lower = user_input.lower()
    
    if user_input_lower in exits:
        bot_reply = random_farewell
    elif user_input_lower in greetings:
        bot_reply = random_greetings
        st.markdown("<br/>", unsafe_allow_html=True)
    else:
        response = get_response(user_input)
        bot_reply = response

    # Apply CSS style to the chat container
    chat_container_style = """
        <style>
            .chat-container {
                background-color: #e6f7ff;
                padding: 10px;
                border-radius: 10px;
                color: black;
            }
        </style>
    """
    st.markdown(chat_container_style, unsafe_allow_html=True)

    # Create a chat container div
    chat_container = f'<div class="chat-container">You: {user_input}<br>Bot: {bot_reply}</div>'
    st.markdown(chat_container, unsafe_allow_html=True)

    with open("chat_history.txt", "w") as file:
        file.write(user_input + "\n")

history.append(user_input)
with open("chat_history.txt", "r") as file:
    history = file.readlines()

for message in history:
    st.sidebar.write(message)
