import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.datasets import load_iris

data= load_iris()
column_names= ['sepal_length','sepal_width','petal_length', 'class']
X= data.drop('class' , axis=-1)
y=data['class']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2 , random_state=42)
model= GaussianNB()
model.fit(X_train,y_train)