import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv('gbsg.csv',sep=',')
data = data[["age","size","grade","nodes","rfstime","pgr"]]
print(data.head())

predict = "grade"
x=np.array(data.drop([predict],1))
y=np.array(data[predict])
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.5)

best=0
for i in range(50):
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.3)
    lm=linear_model.LinearRegression()
    lm.fit(x_train,y_train)
    ac=lm.score(x_test,y_test)
    print(ac)
    print("Coefficients: \n",lm.coef_)
    print("Intercept: \n",lm.intercept_)

    if(ac>best):
        with open("grade.pickle","wb") as f:
            pickle.dump(lm,f)
print(best)
pickle_open=open("grade.pickle","rb")
lm=pickle.load(pickle_open)

prediction=lm.predict(x_test)
for i in range(len(prediction)):
    print(prediction[i],y_test[i])

p='size'
style.use('ggplot')
pyplot.scatter(data[p],data['grade'])
pyplot.xlabel("Variable")
pyplot.ylabel("Cancer Grade")
pyplot.show()






