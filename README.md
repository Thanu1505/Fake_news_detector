# Fake_news_detector
--A Natural Language Processing (NLP) based system for detecting fake news articles, utilizing a Decision Tree Classifier and TF-IDF vectorization.

# Project Overview
--This project aims to develop a fake news detection system that can classify news articles as reliable or unreliable. The system uses NLP techniques to preprocess and analyze text data, and a Decision Tree Classifier to make predictions. 

# The project includes:
--Data preprocessing: handling missing values, cleaning text data, tokenization, and stemming
--TF-IDF vectorization: converting text data into numerical vectors
--Decision Tree Classifier: training and evaluating the model on preprocessed and vectorized data
--Interactive Streamlit web app: allowing users to input news content for real-time predictions

# Technologies Used
--Python
--NLTK (Natural Language Toolkit) for text preprocessing
--scikit-learn for TF-IDF vectorization and Decision Tree Classifier
--Streamlit for building the interactive web app
--Pickle for saving the trained model and vectorizer

# Dataset
--The project uses a labeled dataset of news articles from Kaggle, classified as reliable or unreliable.

# Preprocessing Steps
--Handling missing values
--Cleaning text data by removing special characters and converting to lowercase
--Tokenization
--Stemming using NLTK
--Model Training and Evaluation
--TF-IDF vectorization: converting text data into numerical vectors
--Decision Tree Classifier: training and evaluating the model on preprocessed and vectorized data
--Accuracy metrics: evaluating the performance of the model

# Web App
--The project includes an interactive Streamlit web app that allows users to input news content for real-time predictions. The app utilizes the saved model and vectorizer to make predictions.

# Goals and Contributions
--The goal of this project is to contribute to the development of effective tools for combating the spread of misinformation in online platforms. By employing NLP techniques, this system provides a user-friendly interface to combat misinformation and ensure news credibility.

# License
This project is licensed under the MIT License. See LICENSE for details.

# Acknowledgments
Kaggle for providing the dataset
NLTK and scikit-learn for providing the necessary libraries for NLP tasks
Streamlit for providing the framework for building the interactive web app
