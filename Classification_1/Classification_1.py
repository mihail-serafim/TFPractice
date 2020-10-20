import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from input_fn import *

# This program classifies iris flowers into the three different classes of Setosa, Versicolor, and Verginica
# The given information about each flower is sepal length/width, and petal length/width
# To classify the various flowers, I use a DNNClassifier 

# defining some constants
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# using keras to grab data 
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

# and reading it into a pandas dataframe
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# popping the species column to use as labels
train_y = train.pop('Species')
test_y = test.pop('Species')

# creating feature columns to describe how inputs are used
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# build a DNN with 2 hidden layers, with 30 and 10 nodes respectively.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # 2 hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # choose between 3 classes.
    n_classes=3)

# training the classifier
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

# testing the classfier
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))

print('Accuracy: {accuracy:0.3f}\n'.format(**eval_result))

#simple script to prompt user for input and return classifier results
features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid: 
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: test_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))


