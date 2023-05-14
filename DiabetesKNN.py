# K nearest neighbors classification model to predict the risk of diabetes at early symptoms
# importing required libraries
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np

# reading the data from the .csv file
data = pd.read_csv("db.csv", sep=",")
data = data[["Age", "Gender", "Polyuria", "Polydipsia", "sudden weight loss", "weakness", "delayed healing", "Itching",
             "Obesity", "class"]]

# preprocessing the nominal data
p = preprocessing.LabelEncoder()
Age = p.fit_transform(list(data["Age"]))
Gender = p.fit_transform(list(data["Gender"]))
Polyuria = p.fit_transform(list(data["Polyuria"]))
Polydipsia = p.fit_transform(list(data["Polydipsia"]))
SWL = p.fit_transform(list(data["sudden weight loss"]))
weakness = p.fit_transform(list(data["weakness"]))
dl = p.fit_transform(list(data["delayed healing"]))
Itching = p.fit_transform(list(data["Itching"]))
Obesity = p.fit_transform(list(data["Obesity"]))
class1 = p.fit_transform(list(data["class"]))
print(class1)

# zipping and creating list for labels and attributes
x = list(zip(Age, Gender, Polyuria, Polydipsia, SWL, weakness, dl, Itching, Obesity))
y = list(class1)

# splitting the train and test data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# creating the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print("Accuracy: \n", accuracy)

# predicting using the test data
predictions = model.predict(x_test)
name = ["Positive", "Negative"]
for i in range(len(predictions)):
    print("Prediction: ", name[predictions[i]], "Actual: ", name[y_test[i]])
