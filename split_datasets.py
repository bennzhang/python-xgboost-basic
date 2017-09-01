# First XGBoost model for Pima Indians dataset
import numpy as np
from numpy import loadtxt
import pickle
# load data
dataset = loadtxt('pima-indians-diabetes.data', delimiter=",")
seed = 7
test_size = 0.3

a_train,a_test  = train_test_split(dataset, test_size=test_size, random_state=seed)
np.savetxt("train.csv",a_train,delimiter=",")
np.savetxt("test.csv",a_test,delimiter=",")
