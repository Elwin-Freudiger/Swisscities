import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

df = pd.read_csv('data/csv/Features.csv')
# Convert the 'Location' column to categorical labels
df['Location'] = pd.Categorical(df['Location'])
df['Location'] = df['Location'].cat.codes

numeric_features = ['Mean', 'Stdev', 'Contrast', 'Correlation', 'Energy', 'Homogeneity']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

Numeric_train, Numeric_test = train_test_split(df[numeric_features], test_size=0.3, random_state=42)
y_train, y_test = train_test_split(df['Location'], test_size=0.3, random_state=42)
Image_string_train , Image_string_test = train_test_split(df[['RGB']], test_size=0.3, random_state=42)

y_train = np.array(y_train.to_list())
y_test = np.array(y_test.to_list())

Numeric_train = Numeric_train.to_numpy()
Numeric_test = Numeric_test.to_numpy()

Image_train = []
for i in Image_string_train['RGB'].to_numpy():
    array = eval(i)
    array = (array-np.min(array))/(np.max(array)-np.min(array))
    Image_train.append(array)
Image_train = np.array(Image_train)

Image_test = []
for i in Image_string_test['RGB'].to_numpy():
    array = eval(i)
    array = (array-np.min(array))/(np.max(array)-np.min(array))
    Image_test.append(array)
Image_test = np.array(Image_test)

model = Sequential()
model.add(Flatten(input_shape=(256, 3)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation= 'softmax'))

#compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train the model
model.fit(Image_train, y_train, epochs=10)

#test accuracy
test_loss, test_acc = model.evaluate(Image_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)

y_pred_probs = model.predict(Image_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
print(cm)
