# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
# load data
dataset = loadtxt('train.csv', delimiter=",")
# split data into X and y
X_train = dataset[:,0:8]
y_train = dataset[:,8]

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open("pima_model.pkl", "wb"))

# make predictions for test data
#y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]
## evaluate predictions
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))
