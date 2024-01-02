import pickle

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Download the 'punkt'  and 'stopwords' resource if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    words = []
    for word in text:
        if word.isalnum():
            words.append(word)
    text = words[:]
    words.clear()
    for word in text:
        if word not in stopwords.words('english'):
            words.append(word)

    text = words[:]
    words.clear()
    ps = PorterStemmer()
    for word in text:
        words.append(ps.stem(word))

    return " ".join(words)

def predict_message(msg):
    transformed_email = transform_text(msg)
    # vectorize
    vector_input = tfidf.transform([transformed_email])
    # result
    result = model.predict(vector_input)[0]
    return result
