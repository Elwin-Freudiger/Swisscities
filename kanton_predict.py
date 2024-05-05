import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten, Input


df = pd.read_csv('data/csv/Kanton_features.csv')
# Convert the 'Location' column to categorical labels
df['Location'] = pd.Categorical(df['Location'])
df['Location_code'] = df['Location'].cat.codes

numeric_features = ['Mean', 'Stdev', 'Contrast', 'Correlation', 'Energy', 'Homogeneity']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

Numeric_train, Numeric_test = train_test_split(df[numeric_features], test_size=0.3, random_state=42)
y_train, y_test = train_test_split(df['Location_code'], test_size=0.3, random_state=42)
Image_string_train , Image_string_test = train_test_split(df[['Distribution']], test_size=0.3, random_state=42)

y_train = np.array(y_train.to_list())
y_test = np.array(y_test.to_list())

Numeric_train = Numeric_train.to_numpy()
Numeric_test = Numeric_test.to_numpy()

Image_train = []
for i in Image_string_train['Distribution'].to_numpy():
    array = eval(i)
    array = (array-np.min(array))/(np.max(array)-np.min(array))
    Image_train.append(array)
Image_train = np.array(Image_train)

Image_test = []
for i in Image_string_test['Distribution'].to_numpy():
    array = eval(i)
    array = (array-np.min(array))/(np.max(array)-np.min(array))
    Image_test.append(array)
Image_test = np.array(Image_test)

numeric_input = Input(shape = (6, ), name = 'Numeric')
image_input = Input(shape = (256, 4), name = 'Image')

numeric_dense = Dense(64, activation='relu')(numeric_input)

image_flatten = Flatten()(image_input)
image_dense = Dense(128, activation = 'relu')(image_flatten)
image_drop = Dropout(0.5)(image_dense)
image_dense = Dense(128, activation = 'relu')(image_drop)

x = layers.concatenate([numeric_dense, image_dense])

xdense = Dense(64, activation = 'relu')(x)
output = Dense(25, activation = 'softmax')(xdense)

model = keras.Model(inputs=[numeric_input, image_input], outputs = output)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    {"Numeric": Numeric_train, "Image": Image_train},
    y_train, epochs=10, batch_size=32)

#test accuracy
test_loss, test_acc = model.evaluate([Numeric_test, Image_test],  y_test, verbose=2)
print('\nTest accuracy:', test_acc)

y_pred_probs = model.predict([Numeric_test, Image_test])
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
print(cm)