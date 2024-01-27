# importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# data collection and processing
# loading csv data to a pandas data frame
heart_data = pd.read_csv("heart.csv")

# print first 5 rows of the dataset
# print(heart_data.head())

# print last 5 rows of the dataset
# print(heart_data.tail())

# number of rows and columns in the dataset
# print(heart_data.shape)

# getting more info about the data
# print(heart_data.info())
# non null values - no null values found

# check for missing values
# print(heart_data.isnull().sum())

# statistical measures about the data
# print(heart_data.describe())

# checking the distribution of target variable
# print(heart_data["target"])
# print(f"Value Count: \n{heart_data['target'].value_counts()}")

# splitting the features and target
X = heart_data.drop(columns="target", axis=1)
# print(f"features(X): \n{X}")
Y = heart_data["target"]
# print(f"Labels(Y): \n{Y}")

# splitting the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
# print(X.shape, X_train.shape, X_test.shape)
# print(Y.shape, Y_train.shape, Y_test.shape)

# Logistic Regression model
model = LogisticRegression()

# train the model with training data
model.fit(X_train, Y_train)

# Model Evaluation - Accuracy Score
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f"Training accuracy: {training_data_accuracy}")

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f"Test accuracy: {test_data_accuracy}")

# building a predictive system
input_data = (58, 0, 3, 150, 283, 1, 0, 162, 0, 1, 2, 0, 2)
# change input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only one instance
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshape)
print(f"prediction: {prediction}")
