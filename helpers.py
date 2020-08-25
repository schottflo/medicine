#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Callable, List

import numpy as np
import pandas as pd

TESTS: List[str] = [
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


def preprocess(
    data: pd.DataFrame, aggregations: List[Callable[[List[float]], float]] = None
) -> pd.DataFrame:
    """
    Perform elementary feature engineering and pivot data to wide.

    Details: TODO

    Parameters
    ----------
    data : pd.DataFrame
        Data to be preprocessed.
    aggregations : List[Callable[[List[float]], float]]
        Aggregation functions to be performed on each test.

    Returns
    -------
    dat : pd.DataFrame
        Preprocessed data.
    """
    # Deep copys for safety.
    backup = data.copy(deep=True)
    dat = data.copy(deep=True)
    dat.columns = [i.lower() for i in dat.columns]
    # Save the first time observed for each `pid`.
    dat["first_time_observed"] = dat["pid"].map(
        dat.groupby(by="pid").agg(np.min)["time"].to_dict()
    )
    dat["time"] = pd.Series([i for q in range(dat.shape[0]) for i in range(1, 13)])

    dat = dat.pivot(index="pid", columns="time")
    # Drop the first eleven age columns and the last eleven
    # `first_time_observed_columns` since they are redundant.
    dat = dat.drop(
        dat.columns[[i for i in range(1, 12)] + [i for i in range(-1, -12, -1)]], axis=1
    ).reset_index()

    dat.columns = [
        f"{x[0]}_{x[1]}"
        if (x[0] != "age" and x[0] != "first_time_observed")
        else f"{x[0]}"
        for x in dat.columns.to_flat_index()
    ]

    dat.columns = ["pid"] + list(dat.columns)[1:]

    for col in backup.columns[3:]:
        dat[col + "_count"] = backup.groupby("pid").count()[col].values
        dat[col + "_mean"] = backup.groupby("pid").mean()[col].values
        dat[col + "_last_value"] = backup.groupby("pid").last()[col].values
        dat[col + "_median"] = backup.groupby("pid").median()[col].values
        if aggregations is not None:
            for f in aggregations:
                dat[col + "_" + f.__name__] = backup.groupby("pid").agg(f)[col].values

    return dat
