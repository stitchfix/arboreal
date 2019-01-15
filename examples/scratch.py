import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from core.datapoint import Datapoint
from core.dataset import Dataset, Metadata
from core.datatype import Datatype
from core.arboreal_tree import DecisionTree
from core.split import Split, create_val_set_set_set

from pprint import pprint as pp


if __name__ == "__main__":

    # Make some toy data points
    dp1 = Datapoint(
        _id=1, fit_rating=1, fitted=0.5, buttonup=0.2, summer="yes"
    )
    dp2 = Datapoint(
        _id=2, fit_rating=0, fitted=0.8, buttondown=0.3, summer="almost"
    )
    dp3 = Datapoint(
        _id=3,
        fit_rating=1,
        fitted=0.1,
        previous_purchases=2,
        buttondown=0.1,
        summer="yes",
    )
    dp4 = Datapoint(
        _id=4,
        fit_rating=1,
        fitted=0.2,
        previous_purchases=0,
        buttonup=0.9,
        winter="yes",
    )
    dp5 = Datapoint(
        _id=5, fit_rating=0, fitted=0.4, buttonup=0.6, summer="yes"
    )

    # Metadata
    m = Metadata()
    m.identifier = "_id"
    m.numericals = [
        "fit_rating",
        "fitted",
        "buttonup",
        "buttondown",
        "previous_purchases",
    ]
    m.categoricals = ["summer", "winter"]
    m.target = "fit_rating"

    # Dataset
    dataset = Dataset(metadata=m, datapoints=[dp1, dp2, dp3, dp4, dp5])

    # Fit Split
    s = Split()
    s.fit(dataset)
    print(f"Split: {s}")
    print(f"Has reward: {s.reward}")

    # Decision Tree
    t = DecisionTree(name="A Simple Tree")
    t.fit(dataset)
    print(f"Fit tree:")
    print(t)

    # Split fitting scratch
    vsss_fitted = create_val_set_set_set(
        dataset.unique_values_of_feature("fitted"), Datatype.numerical
    )
    vsss_summer = create_val_set_set_set(
        dataset.unique_values_of_feature("summer"), Datatype.categorical
    )
    vsss_r10 = create_val_set_set_set(set(range(10)), Datatype.categorical)

    # Transform scratch
    for dp in dataset.datapoints:
        prediction, prediction_datatype = t._transform(dp)
        print(f"Predicted: {prediction}, True: {dp[dataset.metadata.target]}")

    # Batch prediction
    print(f"Batch predictions:")
    predictions = t.transform(dataset)
    pp(predictions)
