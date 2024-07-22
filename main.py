import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import CountVectorizer

#Step 1: Creating the Data
movie_review = input("Hello, please provide a short review of the most recent movie you watched. ")
test_corpus = ["hate","I like this, but the production quality is terrible.","Adore","good","good"]
test_corpus.append(movie_review)
training_corpus = ['hate','like','terrible','adore','love','good','bad','amazing','horrible','great','atrocious','perfect','sucky','flawless']

#Step 2: Creating the Algorithm
vectorizer = CountVectorizer()

#Step 3: Training the Vectorizer
vectorizer.fit(training_corpus)

#Step 4: Doing the actual Transformation
X = vectorizer.transform(training_corpus)
X_train = X.toarray()
X_test = vectorizer.transform(test_corpus)
X_test = X_test.toarray()

#Machine Learning Portion
#Step 1: Creating the Data(data transformation is already done)
y_train = np.array(["bad","good","bad","good","good","good",'bad','good','bad','good','bad','good','bad','good'])
#Creating the Algorithm
algo = KNN(n_neighbors = 1)
#Fitting/Training the Algorithm
algo.fit(X_train,y_train)

#Prediction
results = algo.predict(X_test)
print("Your results have been classified. You thought that the movie was",results[5])
