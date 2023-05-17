# importing libraries
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

# loading breast_cancer_data from sklearn.dataset
cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target
# print(x)
# print(y)

# Data splitting
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# print(x_train, y_train)
classes = ["malignant", "benign"]

# Creating the model
clas = svm.SVC(kernel="linear") #using linear in the kernel function. 
clas.fit(x_train, y_train)
prediction = clas.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print("Accuracy =", accuracy)

# Displaying the predictions
for i in prediction:
    print(classes[i], y_test[i])
