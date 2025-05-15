import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split  #type: ignore
from sklearn.feature_extraction.text import CountVectorizer #type: ignore
from sklearn.naive_bayes import MultinomialNB  #type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix  #type: ignore
from sklearn.feature_extraction.text import CountVectorizer  #type: ignore


vectorizer = joblib.load('vectorizer copy.pkl')
model = pickle.load(open("model copy.txt","rb"))
classes = model.classes_

while True:
    user_input = input("string >")
    if user_input == "End":
        break
    user_input_vectorized = vectorizer.transform([user_input])
    user_input_vectorized = vectorizer.transform([user_input])
    predicted_weights = model.predict_proba(user_input_vectorized)
    predicted_label = model.predict(user_input_vectorized)
    inSet = True
    weights = []
    for i in range(len(predicted_weights[0])):   
        pWeights = predicted_weights[0]
        k = abs(float(pWeights[i]))
        cl = str(classes[i])
        weights.append([k,cl])
        weights.sort(reverse=True)
    wNums = []
    for i in weights:
        wNums.append(i[0])
    if wNums[0] < 10:
        print(f"fuzzy results, close match {weights[:5]}")
    print(f"The input text belongs to the '{predicted_label[0]}' category.")
