# Disclaimer
We make no warranty as to the quality, functionality, safety, or utility of this software.  This repository contains an exploration of some ideas.  We are making this repository public so that we can collaborate with members of the community outside of Stitch Fix.

# Arboreal
#### Tree based modeling for humans


## What is Arboreal?
Welcome!  Arboreal is a Python package for tree based machine learning.  It's designed to work with a variety of data types, and has an explicit priority of ease of use and extensibility over speed.

## What does using Arboreal look like?
```python
# First, grab a dataset (using the common iris dataset imported from sklearn)
# Load iris dataset and convert to Pandas DataFrame
from sklearn import datasets
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])
# Add an explicit identifier column
iris_df['identifier'] = range(1, len(iris_df) + 1)
# Create the datapoints by converting Pandas rows to dictionaries
datapoints = iris_df.to_dict(orient='records')
# Train/test split
train_fraction = 0.8
random.shuffle(datapoints)  # shuffle dataset
split_index = math.ceil(train_fraction * len(datapoints))
train_datapoints, test_datapoints = datapoints[:split_index], datapoints[split_index:]
# Double check we're not cheating by letting the target into the test data
test_datapoints_for_eval = copy.deepcopy(test_datapoints)
for dp in test_datapoints:
    del dp['target']


# And now to use Arboreal...

# Create the Arboreal Metadata for this dataset
m = Metadata()
m.identifier = 'identifier'
m.numericals = ['sepal length (cm)',
                'sepal width (cm)',
                'petal length (cm)',
                'petal width (cm)']
m.categoricals = ['target']
m.target = 'target'

# Create an Arboreal Dataset for train and test
train_dataset = Dataset(metadata=m,
                        datapoints=train_datapoints)
test_dataset = Dataset(metadata=m,
                       datapoints=test_datapoints,
                       validate_target=False)

# Fit an ArborealTree on the train set
tree = ArborealTree()  # or DecisionTree() or RandomForest()
tree.fit(train_dataset)

# Predict data points in the test set
predictions = tree.transform(test_dataset)

# Evaluate our performance on the test set
results = []
prediction_datatypes = set()
for dp in test_datapoints_for_eval:
    target = dp['target']
    prediction = predictions[dp['identifier']]
    predicted_value = prediction[0]
    prediction_datatype = prediction[1]
    prediction_datatypes.add(prediction_datatype)
    assert len(prediction_datatypes) == 1, "All predictions should be of the same datatype"
    results.append((target, predicted_value))
accuracy = len([r for r in results if r[0] == r[1]]) / len(results)
print(f"ArborealTree Accuracy: {accuracy} (Datatype: {prediction_datatype})")
print(f"ArborealTree:")
print(tree)
```


## Installation
To get started, you'll want to clone this repository and run the tests to ensure Arboreal is working on your system.

To install Arboreal's test dependencies:

`pip install -r requirements/test.txt`

And then to run tests:

`sniffer`

or

`python -m unittest discover`


## Examples
To see some examples of Arboreal in use, check out `examples/` and the `test/` directory.

## Development
Arboreal has some dependencies that make development nicer.  Try installing the `dev` dependencies with `pip install -r requirements/dev.txt` and then running `sniffer` in your terminal (pro tip: turn your volume down first!).


## Dependencies
Arboreal has different sets of dependencies corresponding to use cases.  One set of dependencies is used for tests, for example, while another set is used for development. To ensure you have exactly the dependencies desired for your use case, run:

`pip install -r requirements/{use_case}.txt`
