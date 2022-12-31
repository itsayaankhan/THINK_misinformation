# Step 1: Install and set up the necessary libraries
!pip install numpy pandas scikit-learn tensorflow

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf

# Step 2: Collect and preprocess the data
# Load the dataset into a pandas DataFrame
df = pd.read_csv("misinformation_dataset.csv")

# Preprocess the text data
# Tokenize the text
df["tokens"] = df["text"].apply(lambda x: x.split())
# Stem the tokens
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
df["stemmed_tokens"] = df["tokens"].apply(lambda x: [stemmer.stem(token) for token in x])
# Vectorize the stemmed tokens
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
df["vectors"] = df["stemmed_tokens"].apply(lambda x: vectorizer.fit_transform(x).toarray())

# Step 3: Extract features from the text
X = np.concatenate(df["vectors"].to_numpy(), axis=0)
y = df["label"].to_numpy()

# Step 4: Train and test a machine learning algorithm
# Split the data into a training set and a test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a k-nearest neighbors classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Test the classifier on the test set
y_pred = knn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy:.2f}")

# Step 5: Train a DNN for sentiment analysis
# Preprocess the text data for the DNN
# Tokenize the text
df["dnn_tokens"] = df["text"].apply(lambda x: x.split())
# Convert the tokens to numerical indices
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df["dnn_tokens"])
df["dnn_sequences"] = df["dnn_tokens"].apply(lambda x: tokenizer.texts_to_sequences([x])[0])
# Pad the sequences to the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_length = max(df["dnn_sequences"].apply(lambda x: len(x)))
df["dnn_padded_sequences"] = df["dnn_sequences"].apply(lambda x: pad_sequences([x], maxlen=max_length, padding="post")[0])

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(df["dnn_padded_sequences"].to_numpy(), y, test_size=0.2, random_state=42)

# Build the DNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Test the model
y_pred = model.predict(X_test) > 0.5
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy:.2f}")
