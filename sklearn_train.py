# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
import csv
from datetime import datetime
from joblib import dump, load

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def ingest():
    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    return data

def split(data):
    np.random.seed(40)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]


    return train_x, test_x, train_y, test_y

def train(train_x, train_y, test_x, test_y, alpha, l1_ratio):
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    return lr, rmse, mae, r2


def store_model(alpha, l1_ratio, rmse, mae, r2, model):

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    current_datetime = str(datetime.now()).replace(' ', '_')
    model_name = f'{current_datetime}_my_model'
    dump(model, 'model_store/' + model_name)
    with open('model_metrics/model_metrics.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([current_datetime, alpha, l1_ratio, rmse, mae, r2])
    return True

def main():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    data = ingest()
    train_x, test_x, train_y, test_y = split(data)
    estimator, rmse, mae, r2 = train(train_x, train_y, test_x,  test_y, alpha, l1_ratio)
    store_model(alpha, l1_ratio, rmse, mae, r2, estimator)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
