import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

#open the csv
data = pd.read_csv('data/csv/Ten_cities.csv')

def string_to_array(string):
    elements = string.strip('[]').split(',')
    array = np.array(elements, dtype=int).reshape((500, 500))
    return array

data['Array'] = data['Array'].apply(string_to_array)

City_list = ['Zurich', 'Geneva', 'Basel', 'Lausanne', 'Bern', 'Winterthur', 'Luzern', 'St_Gallen', 'Lugano', 'Biel']
label_encoder = LabelEncoder()
data['Location'] = label_encoder.fit_transform(data['Location'])

#split the dataset
X = np.array(data['Array'].to_list())
y = np.array(data['Location'].to_list())
X_train, X_test, y_train, y_test = train_test_split(data['Array'], data['Location'], test_size=0.3)

#reduce the array between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

model = Sequential()
model.add(Flatten(input_shape=(500, 500)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation= 'softmax'))

#compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, epochs=10)

#test accuracy
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)

