import pickle
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# load data
dataset = loadtxt('test.csv', delimiter=",")
# split data into X and y
X_test = dataset[:,0:8]
y_test = dataset[:,8]

# load model from file
loaded_model = pickle.load(open("pima_model.pkl", "rb"))

# make predictions for test data
y_pred = loaded_model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
