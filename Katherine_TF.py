import csv
import pandas as pd
import numpy as np
import math 
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers

dataframe = pd.read_csv('train.csv')
dataframe['user_verified'] = dataframe['user_verified'].astype(int)
dataframe.head()

dataframe = dataframe.drop(columns=['id', 'user_mentions', 'urls', 'hashtags', 'text'])

train, val = train_test_split(dataframe, test_size=0.3)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('retweet_count')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
                ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

x_train = df_to_dataset(dataframe = train)
x_eval = df_to_dataset(dataframe = val, shuffle=False)

feature_columns = []

# numeric cols
for header in ['timestamp', 'user_verified', 'user_statuses_count', 'user_followers_count', 'user_friends_count']:
        feature_columns.append(feature_column.numeric_column(header))


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([feature_layer, layers.Dense(128, activation='relu'), layers.Dense(128, activation='relu'), layers.Dropout(.1), layers.Dense(1)])

model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError(), metrics=['mean_absolute_error'])

model.fit(x_train, validation_data=x_eval, epochs=10)

loss, accuracy = model.evaluate(x_eval)
print("Accuracy", accuracy)