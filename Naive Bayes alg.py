import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split  #type: ignore
from sklearn.feature_extraction.text import CountVectorizer #type: ignore
from sklearn.naive_bayes import MultinomialNB  #type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix  #type: ignore
from sklearn.feature_extraction.text import CountVectorizer  #type: ignore


data = pd.read_csv('datasetMaker/encoded_data.csv')
X = data['data']

y = data['label']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



vectorizer = CountVectorizer(analyzer='char',encoding="utf-16")

X_train_vectorized = vectorizer.fit_transform(X_train)
joblib.dump(vectorizer, 'vectorizer.pkl')


X_test_vectorized = vectorizer.transform(X_test)
print("vectorised")

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)
pickle.dump(model,open("model.txt","wb"))
print("trained")

y_pred = model.predict(X_test_vectorized)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred,labels=['base64','base16','base32','base_45','base85','base92','url','binary','html entity','charcode','octal','hex','morse','rot_13','rot_47','plain'])
print(conf_matrix)
print(f'Accuracy: {accuracy *100}%')


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
