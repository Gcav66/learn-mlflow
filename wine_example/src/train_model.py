import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

from ml_utils import eval_metrics

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

DATA_URL = os.environ["CSV_URL"]
TARGET_VAR = os.environ["TARGET_VAR"]
TRAINING_PARAMS = {
    "alpha": float(os.environ["ALPHA"]),
    "l1_ratio": float(os.environ["L1_RATIO"]),
    "random_state": int(os.environ["RANDOM_STATE"])
}

def ingest_data():
    """Read the wine-quality csv file from env variable"""
    try:
        data = pd.read_csv(DATA_URL, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    return data

def split_data(data, target):
    """Assumes data is a pandas dataframe and target_var is a str"""
    np.random.seed(40)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    data_out = {
        "train_x": train.drop([target], axis=1),
        "test_x": test.drop([target], axis=1),
        "train_y": train[[target]],
        "test_y": test[[target]]
    }

    return data_out

def train_model(data: dict):
    params = TRAINING_PARAMS
    clf = ElasticNet(**params)
    clf.fit(data['train_x'], data['train_y'])
    return clf

def get_prediction(data: dict, model: ElasticNet):
    return model.predict(data["test_x"])

def evaluate_model(data: dict, prediction):
    (rmse, mae, r2) = eval_metrics(data['test_y'], prediction)
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    return metrics

def store_model(params: dict, metrics: dict, model: ElasticNet, target):
    """Assumes alpha, l1_ratio, rmse, mae, and r2 are floats
       Assumes model is a sklearn estimator"""
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (params['alpha'], params['l1_ratio']))
    print("  RMSE: %s" % metrics['rmse'])
    print("  MAE: %s" % metrics['mae'])
    print("  R2: %s" % metrics['r2'])


    mlflow.log_param("alpha", params['alpha'])
    mlflow.log_param("l1_ratio", params['l1_ratio'])
    mlflow.log_param("target_var", target)
    mlflow.log_metric("rmse", metrics['rmse'])
    mlflow.log_metric("r2", metrics['r2'])
    mlflow.log_metric("mae", metrics['mae'])

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetWineModel")
    else:
        mlflow.sklearn.log_model(model, "model")

def train():
    mlflow.start_run()
    df = ingest_data()
    data = split_data(df, TARGET_VAR)
    clf = train_model(data)
    predictions = get_prediction(data, clf)
    metrics = evaluate_model(data, predictions)
    store_model(TRAINING_PARAMS, metrics, clf, TARGET_VAR)
    mlflow.end_run()


if __name__ == "__main__":
    train()