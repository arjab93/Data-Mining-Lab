import logging
import tensorflow as tf

# Set TensorFlow logging level to suppress warnings
tf.get_logger().setLevel(logging.ERROR)

# Set Keras logging level to suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Your code here

from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
import numpy as np
cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target
split = int(len(x) * 0.7)
trainx, testx = x[:split], x[split:]
trainy, testy = y[:split], y[split:]
print("Number of features: ", len(cancer.feature_names))
# print("Features: ", *cancer.feature_names)
print("Number of classes: ", len(cancer.target_names))
print("Class Labels: ", cancer.target_names)
# Define the keras model
model = Sequential()
model.add(Dense(128, input_dim=30, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
# Compile and fit the model
model.compile(loss="binary_crossentropy", optimizer="adam", 
metrics=["accuracy"])
model.fit(trainx, trainy, epochs=200, batch_size=16, verbose=0)
# Make class predictions with the model
yp = model.predict(testx)
pred = []
for x in yp:
 pred.append(np.round(x))
pred = np.array(pred)
pred = pred.ravel()
pred = pred.astype(int)
# print("Actual Class: ", *testy)
# print("Predicted Class: ", *yp)
print("\nConfusion Matrix:")
print(metrics.confusion_matrix(testy, pred))
print("\nClassification Measures:")
print("Accuracy:", metrics.accuracy_score(testy, pred))
print("Recall:", metrics.recall_score(testy, pred))
print("Precision:", metrics.precision_score(testy, pred))
print("F1-score:", metrics.f1_score(testy, pred))