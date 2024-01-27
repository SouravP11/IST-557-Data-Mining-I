import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("train.csv")
test_X_df = pd.read_csv("test_X.csv")
sample_y_df = pd.read_csv("sample_submission.csv")

def convert_categorical_to_numerical(df):
    new_df = df.copy()  # so operations on new_df will not influence df

    # convert categorical features
    sex = pd.get_dummies(new_df["Sex"], prefix="sex", dtype=float)
    chest = pd.get_dummies(new_df["ChestPainType"], prefix="chest", dtype=float)
    restECG = pd.get_dummies(new_df["RestingECG"], prefix="rest_ECG", dtype=float)
    exercise = pd.get_dummies(new_df["ExerciseAngina"], prefix="exercise", dtype=float)
    slope = pd.get_dummies(new_df["ST_Slope"], prefix="slope", dtype=float)

    # drop categorical features with their numerical values
    new_df.drop(columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], axis=1, inplace=True)

    # create new dataframe with only numerical values
    new_df = pd.concat([new_df, sex, chest,restECG,exercise,slope], axis=1)

    return new_df

# convert features for training and testing data
my_train_df = convert_categorical_to_numerical(train_df)
my_test_X_df = convert_categorical_to_numerical(test_X_df)

# prepare features and labels for training/testing
train_X = my_train_df.drop(["HeartDisease", "PatientID"], axis=1)
train_y = my_train_df["HeartDisease"]
test_X = my_test_X_df.drop(["PatientID"], axis=1)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(train_X)
X_test = sc.transform(test_X)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(train_y.astype(np.float32))