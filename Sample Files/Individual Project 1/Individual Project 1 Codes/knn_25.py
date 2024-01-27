import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# Load data from data file
train_df = pd.read_csv("train.csv")
test_X_df = pd.read_csv("test_X.csv")
sample_y_df = pd.read_csv("sample_submission.csv")

def convert_categorical_to_numerical(df):
    new_df = df.copy()  # so operations on new_df will not influence df

    # convert categorical features
    sex = pd.get_dummies(new_df["Sex"], prefix="sex", dtype=int)
    chest = pd.get_dummies(new_df["ChestPainType"], prefix="chest", dtype=int)
    restECG = pd.get_dummies(new_df["RestingECG"], prefix="rest_ECG", dtype=int)
    exercise = pd.get_dummies(new_df["ExerciseAngina"], prefix="exercise", dtype=int)
    slope = pd.get_dummies(new_df["ST_Slope"], prefix="slope", dtype=int)

    # drop categorical features with their numerical values
    new_df.drop(columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], axis=1, inplace=True)

    # create new dataframe with only numerical values
    new_df = pd.concat([new_df, sex, chest, restECG, exercise, slope], axis=1)

    return new_df

# Convert features for training and testing data
my_train_df = convert_categorical_to_numerical(train_df)
my_test_X_df = convert_categorical_to_numerical(test_X_df)

# Replace zeroes with 25th percentile
percentile_25 = my_train_df["Cholesterol"].quantile(0.25)
my_train_df["Cholesterol"] = my_train_df["Cholesterol"].replace(0, percentile_25)


# Prepare features and labels for training/testing
train_X = my_train_df.drop(["HeartDisease", "PatientID"], axis=1)
train_y = my_train_df["HeartDisease"]
test_X = my_test_X_df.drop(["PatientID"], axis=1)

# Create a Random Forest Classifier
rf_model = RandomForestClassifier()

# # Perform k-fold cross-validation
# k = 11  # You can choose the number of folds (e.g., 5 for 5-fold cross-validation)
# cv_scores = cross_val_score(rf_model, train_X, train_y, cv=k, scoring='accuracy')
#
# # Print the cross-validation scores
# print(f"Cross-validation scores: {cv_scores}")
# print(f"Mean accuracy: {np.mean(cv_scores)}")
# print(f"Standard deviation of accuracy: {np.std(cv_scores)}")

# Fit the model on the entire training data
rf_model.fit(train_X, train_y)

# Model Evaluation - Accuracy Score
# Accuracy on training data
train_y_pred = rf_model.predict(train_X)
training_data_accuracy = accuracy_score(train_y_pred, train_y)
print(f"Training accuracy: {training_data_accuracy}")

# Predict on test data
test_y_pred = rf_model.predict(test_X)

# Prepare the prediction file to submit on Kaggle
submission_df = pd.DataFrame({
    'PatientID': my_test_X_df['PatientID'],
    'HeartDisease': test_y_pred
})
submission_df.to_csv("random_forest_q1.csv", index=False)
