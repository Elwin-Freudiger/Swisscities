import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, Model, utils
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Dropout, Dense, Flatten, Input

# Read the CSV file
df = pd.read_csv('data/csv/top10_with_info.csv')

# Convert the 'Location' column to categorical labels
df['Location'] = pd.Categorical(df['Location'])
df['Location_code'] = df['Location'].cat.codes
city_names = df['Location'].cat.categories

# Standardize numeric features
numeric_features = ['Mean', 'Stdev', 'Contrast', 'Correlation', 'Energy', 'Homogeneity', 'temperature', 'sunshine', 'rainfall']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Split the data
Numeric_train, Numeric_test = train_test_split(df[numeric_features], test_size=0.3, random_state=42)
y_train, y_test = train_test_split(df['Location_code'], test_size=0.3, random_state=42)
Image_string_train, Image_string_test = train_test_split(df[['Distribution']], test_size=0.3, random_state=42)

y_train = np.array(y_train.to_list())
y_test = np.array(y_test.to_list())

Numeric_train = Numeric_train.to_numpy()
Numeric_test = Numeric_test.to_numpy()

# Function to scale each color channel separately
def scale_each_channel(image_str):
    array = np.array(eval(image_str))  # Convert the string to a numpy array
    scaled_array = np.zeros_like(array, dtype=np.float32)
    for channel in range(array.shape[1]):
        channel_data = array[:, channel]
        scaled_array[:, channel] = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data))
    return scaled_array

# Process Image_train data
Image_train = []
for i in Image_string_train['Distribution'].to_numpy():
    scaled_array = scale_each_channel(i)
    Image_train.append(scaled_array)
Image_train = np.array(Image_train)

# Process Image_test data
Image_test = []
for i in Image_string_test['Distribution'].to_numpy():
    scaled_array = scale_each_channel(i)
    Image_test.append(scaled_array)
Image_test = np.array(Image_test)

# Define the model
numeric_input = Input(shape=(9,), name='Numeric')
image_input = Input(shape=(256, 4), name='Image')

numeric_dense = Dense(64, activation='relu')(numeric_input)

image_flatten = Flatten()(image_input)
image_dense = Dense(128, activation='relu')(image_flatten)
image_drop = Dropout(0.5)(image_dense)
image_dense = Dense(128, activation='relu')(image_drop)

x = layers.concatenate([numeric_dense, image_dense])

xdense = Dense(64, activation='relu')(x)
output = Dense(10, activation='softmax')(xdense)

model = Model(inputs=[numeric_input, image_input], outputs=output)

model.summary()
utils.plot_model(model, "report/multimodel.png")

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.fit(
    {"Numeric": Numeric_train, "Image": Image_train},
    y_train, epochs=10, batch_size=32)

# Test accuracy
test_loss, test_acc = model.evaluate([Numeric_test, Image_test], y_test, verbose=2)
print('\nTest accuracy:', test_acc)

y_pred_probs = model.predict([Numeric_test, Image_test])
y_pred = np.argmax(y_pred_probs, axis=1)

# Plot and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=city_names, yticklabels=city_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('report/confusion_matrix_top10_withinfo.png')
plt.show()
