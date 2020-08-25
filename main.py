#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier, XGBRegressor

import helpers

TESTS = [
    "LABEL_BaseExcess",
    "LABEL_Fibrinogen",
    "LABEL_AST",
    "LABEL_Alkalinephos",
    "LABEL_Bilirubin_total",
    "LABEL_Lactate",
    "LABEL_TroponinI",
    "LABEL_SaO2",
    "LABEL_Bilirubin_direct",
    "LABEL_EtCO2",
]
VITALS = ["LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"]

SEPSIS = "LABEL_Sepsis"

SEED = 42

def print_full(x):
    pd.set_option('display.max_columns', len(x))
    print(x)
    pd.reset_option('display.max_columns')

Xl = pd.read_csv("../../data/2/train_features.csv")
print_full(Xl)

def main():
    np.random.seed(SEED)
    X_train = helpers_local.preprocess(
        pd.read_csv("train_features.csv")
    ).sort_values("pid")
    X_test = helpers_local.preprocess(
        pd.read_csv("test_features.csv")
    ).sort_values("pid")
    y_train = pd.read_csv("train_labels.csv").sort_values("pid")
    print_full(X_train)
    y_test = X_test.copy(deep=True).sort_values("pid")
    y_test.drop(y_test.columns[1:], axis=1, inplace=True)
    for i in y_train.columns[1:]:
        y_test[i] = np.nan

    for col in TESTS:
        model = XGBClassifier()
        model.fit(X_train.drop("pid", axis=1), y_train[col].ravel())
        y_test[col] = model.predict_proba(X_test.drop("pid", axis=1))[:, 1]

    model = XGBClassifier()
    model.fit(X_train.drop("pid", axis=1), y_train[SEPSIS].ravel())
    y_test[SEPSIS] = model.predict_proba(X_test.drop("pid", axis=1))[:, 1]

    for col in VITALS:
        model = XGBRegressor(objective="reg:squarederror")
        model.fit(X_train.drop("pid", axis=1), y_train[col].ravel())
        y_test[col] = model.predict(X_test.drop("pid", axis=1))

    y_test.to_csv(
        "prediction.zip",
        index=False,
        sep=",",
        header=True,
        float_format="%.3f",
        compression="zip",
    )


if __name__ == "__main__":
    sys.exit(main())
