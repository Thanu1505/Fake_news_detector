import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin

port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

# Load models and vectorizer only once
with open('vector.pkl', 'rb') as f:
    vector_form = pickle.load(f)

with open('model.pkl', 'rb') as f:
    load_model=pickle.load(open('model.pkl', 'rb'))

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

if __name__ == '__main__':
    st.title('Fake News Classification App')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "", height=200)
    predict_btt = st.button("Predict")

    if predict_btt:
        prediction_class = fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success('Reliable')
        elif prediction_class == [1]:
            st.warning('Unreliable')