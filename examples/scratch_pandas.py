import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# External:
from sklearn import datasets
import copy
import math
import numpy as np
import pandas as pd
import random

# Arboreal:
from core.dataset import Metadata, Dataset
from core.arboreal_tree import DecisionTree, ArborealTree, RandomForest


random.seed(42)  # set random seed


# Get a dataset (using the common iris dataset imported from sklearn)
# Load iris dataset and convert to Pandas DataFrame
iris = datasets.load_iris()
iris_df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]],
    columns=iris["feature_names"] + ["target"],
)
# Add an explicit identifier column
iris_df["identifier"] = range(1, len(iris_df) + 1)
# Create the datapoints by converting Pandas rows to dictionaries
datapoints = iris_df.to_dict(orient="records")
# Train/test split
train_fraction = 0.8
random.shuffle(datapoints)  # shuffle dataset
split_index = math.ceil(train_fraction * len(datapoints))
train_datapoints, test_datapoints = (
    datapoints[:split_index],
    datapoints[split_index:],
)
# Double check we're not cheating by letting the target into the test data
test_datapoints_for_eval = copy.deepcopy(test_datapoints)
for dp in test_datapoints:
    del dp["target"]


# And now to use Arboreal...

# Create the Arboreal Metadata for this dataset
m = Metadata()
m.identifier = "identifier"
m.numericals = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
m.categoricals = ["target"]
m.target = "target"

# Create an Arboreal Dataset for train and test
train_dataset = Dataset(metadata=m, datapoints=train_datapoints)
test_dataset = Dataset(
    metadata=m, datapoints=test_datapoints, validate_target=False
)

# Fit an ArborealTree on the train set
tree = ArborealTree()  # or DecisionTree() or RandomForest()
tree.fit(train_dataset)

# Predict data points in the test set
predictions = tree.transform(test_dataset)

# Evaluate our performance on the test set
results = []
prediction_datatypes = set()
for dp in test_datapoints_for_eval:
    target = dp["target"]
    prediction = predictions[dp["identifier"]]
    predicted_value = prediction[0]
    prediction_datatype = prediction[1]
    prediction_datatypes.add(prediction_datatype)
    assert (
        len(prediction_datatypes) == 1
    ), "All predictions should be of the same datatype"
    results.append((target, predicted_value))
accuracy = len([r for r in results if r[0] == r[1]]) / len(results)
print(f"ArborealTree Accuracy: {accuracy} (Datatype: {prediction_datatype})")
print(f"ArborealTree:")
print(tree)


#  Fit a DecisionTree (or RandomForest) on the train set
tree2 = DecisionTree()  # or RandomForest()
tree2.fit(train_dataset)
predictions2 = tree2.transform(test_dataset)
results2 = []
prediction_datatypes2 = set()
for dp in test_datapoints_for_eval:
    target = dp["target"]
    prediction = predictions2[dp["identifier"]]
    predicted_value = prediction[0]
    prediction_datatype = prediction[1]
    prediction_datatypes2.add(prediction_datatype)
    assert (
        len(prediction_datatypes2) == 1
    ), "All predictions should be of the same datatype"
    results2.append((target, predicted_value))
accuracy2 = len([r for r in results2 if r[0] == r[1]]) / len(results2)
print(f"DecisionTree Accuracy: {accuracy2} (Datatype: {prediction_datatype})")
print(f"DecisionTree:")
print(tree2)
