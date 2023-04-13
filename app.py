import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # To covert input Text to lower case Text
    text = nltk.word_tokenize(text)  # To covert input Text to Tokens

    y = []
    for i in text:
        if i.isalnum():  # To Remove special characters and append the alpha numeric this loop is used
            y.append(i)

    text = y[:]  # Text cloning (else both y and text will be cleared )
    y.clear()

    for i in text:
        if i not in stopwords.words(
                'english') and i not in string.punctuation:  # for removing stop words and punctuation
            y.append(i)

    text = y[:]  # reassigning text using y and then clearing y
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # for stemming

    return " ".join(y)  # to return as string whatever is inside y


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter  the Message")
if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorized
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
