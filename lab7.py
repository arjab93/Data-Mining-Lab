import logging
import tensorflow as tf

# Set TensorFlow logging level to suppress warnings
tf.get_logger().setLevel(logging.ERROR)

# Set Keras logging level to suppress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Your code here

import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
# Add the import statement for NumPy
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

dataset = pd.read_csv("iris.csv")
dataset = dataset.values
dataset = shuffle(dataset)
x = dataset[:, 0:4].astype(float)
y = dataset[:, 4]

encoder = LabelEncoder()
encoder.fit(y)
ey = encoder.transform(y)

dy = to_categorical(ey)

sc = StandardScaler().fit(x)
sx = sc.transform(x)

split = int(len(x) * 0.7)
trainx, testx = sx[:split], sx[split:]
trainy, testy = dy[:split], dy[split:]

model = Sequential()
model.add(Dense(64, input_dim=4, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(units=3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(trainx, trainy, epochs=20, batch_size=8, verbose=0)

yp = model.predict(testx)
yp = np.argmax(yp, axis=-1)
yp = yp.ravel()
a = list()
for i in range(len(testy)):
    d = np.argmax(testy[i])
    a.append(d)
a = np.array(a)
al = encoder.inverse_transform(a)
pl = encoder.inverse_transform(yp)

print(classification_report(al, pl))
