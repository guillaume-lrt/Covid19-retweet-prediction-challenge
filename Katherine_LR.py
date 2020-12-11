import csv
import pandas
import numpy as np
import math 
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers


#read train.csv file
xdata = pandas.read_csv('train.csv', usecols=[1,3,4,5,6])
ydata = pandas.read_csv('train.csv', usecols=[2])
xdata['user_verified'] = xdata['user_verified'].astype(int)
#print(xdata)

#convert pandas to np array
x_data = xdata.values 
y_data = ydata.values
x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_data, train_size = 0.7, test_size=0.3)
print(x_train.shape)
#print(y_train.shape)


#linear regression
modelLR = LinearRegression().fit(x_train, y_train )

#predict retweet in test set
y_pred = modelLR.predict(x_eval)

y_pred1 = np.where(y_pred<0, 0, y_pred)
y_pred2 = np.rint(np.sqrt(y_pred1))
print(y_pred2)

diff = np.absolute(y_pred2 - y_eval)
error = 1/ y_eval.shape[0] * np.sum(diff)
print(error)


#TF

#convert np arrays to tensors
x_train = tf.convert_to_tensor(x_train)
x_eval = tf.convert_to_tensor(x_eval)
y_train = tf.convert_to_tensor(y_train)
y_eval = tf.convert_to_tensor(y_eval)

feature_columns = []
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([feature_layer, layers.Dense(128, activation='relu'), layers.Dense(128, activation='relu'), layers.Dropout(.1), layers.Dense(1)])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(x_train, validation_data=x_eval, epochs=10)

loss, accuracy = model.evaluate(y_eval)
print("Accuracy", accuracy)