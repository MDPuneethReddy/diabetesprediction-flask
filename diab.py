import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, linear_model
import pickle
data=pd.read_csv("diabetes.csv")
x=data.drop("Outcome",axis=1)
y = data.Outcome
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
lm =  GaussianNB()
model = lm.fit(X_train, y_train)
pickle.dump(model,open("diabetes.pkl","wb"))
model=pickle.load(open("diabetes.pkl","rb"))

predictions = lm.predict(X_test)
#predictions[0:5]
#print (model.score(X_test, y_test))

