# xgboost basic
An basic example to show using xgboost.
```
$ python -V
Python 3.6.0 :: Anaconda 4.3.1 (64-bit)
```
If xgboost is not installed, install it by running

```
$ pip install xgboost

$ python -V
Python 3.6.0 :: Anaconda 4.3.1 (64-bit)

>>> print(xgboost.__version__)
0.6
```

## Pima Indians diabetes dataset
You can learn more about this dataset on the [UCI Machine Learning Repository website](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)

Basiclly, this dataset is comprised of 8 input variables that describe medical details of patients and one output variable to indicate whether the patient will have an onset of diabetes within 5 years.

[Download dataset from here](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)

```
$ wget https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data
```

## Steps
  
```
$ python split_datasets.py
```
Split the whole datasets into two sets `train.csv` and `test.csv`. `train.csv` is for training model and `test.csv` is for testing the trained model. 

```
$ python create_model.py
```
Train model using xgboost and save it into `pima_model.pkl`

```
$ python predict.py

Accuracy: 81.17%
```
Based on the saved mode `pima_model.pkl`, make predictions on the test dataset `test.csv` and test the model.
