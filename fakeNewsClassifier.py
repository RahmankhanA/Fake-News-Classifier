# Importing Libraries

import numpy as np    # for the access of N-d Array
import pandas as pd   # for the loading dataset
import itertools      # for itereting
from sklearn.model_selection import train_test_split   # split our dataset into train and test part
from sklearn.feature_extraction.text import TfidfVectorizer  # for converting the text into vector
from sklearn.linear_model import PassiveAggressiveClassifier  # for the classifieng fake news
from sklearn.metrics import accuracy_score, confusion_matrix  # checking the performence of the model


#Read the data
df=pd.read_csv('news.csv')

#Get shape and head
df.shape
df.head()
#Get the labels
labels=df.label
labels.head()


# Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english')

# Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])