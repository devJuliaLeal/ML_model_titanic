from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
dftrain = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # gets a list of all unique values from given feature column
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
        feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name, dtype=tf.float32))

    def make_input_fn(data_df, label_df, num_epochs=9, shuffle=True, batch_size=32):
        def input_function():  # inner function, this will be returned
            # create tf.data.Dataset object with data and its label
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
            if shuffle:
                ds = ds.shuffle(1000)  # randomize order of data
            # split dataset into batches of 32 and repeat process for number of epochs
            ds = ds.batch(batch_size).repeat(num_epochs)
            return ds  # return a batch of the dataset
        return input_function  # return a function object for use

    # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
    train_input_fn = make_input_fn(dftrain, y_train)
    eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    # We create a linear estimtor by passing the feature columns we created earlier
    linear_est.train(train_input_fn)  # train
    # get model metrics/stats by testing on tetsing data
    result = linear_est.evaluate(eval_input_fn)

    clear_output()  # clears consoke output
    # the result variable is simply a dict of stats about our model
    print(result['accuracy'])
    result = list(linear_est.predict(eval_input_fn))
    print(dfeval.loc[3])
    print(y_eval.loc[3])
    print(result[3]['probabilities'][1])
