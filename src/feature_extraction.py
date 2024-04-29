import requests
import rasterio
import numpy as np
import csv
import cv2 as cv
import pandas as pd
import skimage

file_path = 'data/csv/Ten_cities_features.csv'
City_list = ['Zurich', 'Geneva', 'Basel', 'Lausanne', 'Bern', 'Winterthur', 'Luzern', 'St_Gallen', 'Lugano', 'Biel']

data = []

for city in City_list:
    path = f'data/city_link/{city}.csv'
    with open(path, 'r') as link_list:
        links = link_list.read().splitlines()
    
    for item in links:
        try:
            response = requests.get(item)
            if response.status_code == 200:
                with open('swissimage.tif', 'wb') as f:
                    f.write(response.content)
                
                with rasterio.open('swissimage.tif') as dataset:
                    img_array = dataset.read([1, 2, 3]).transpose((1, 2, 0))  # Correct dimension order for OpenCV
                    metadata = dataset.meta
                    corner = metadata['transform']
                    center_x, center_y = corner * (metadata['width'] // 2, metadata['height'] // 2)

                    mean = np.mean(img_array)
                    stdev = np.std(img_array)

                    gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
                    intensity, bins = np.histogram(gray.flatten(), bins=256, range=[0, 256])

                    red = cv.calcHist([img_array], [0], None, [256], [0, 256])
                    green = cv.calcHist([img_array], [1], None, [256], [0, 256])
                    blue = cv.calcHist([img_array], [2], None, [256], [0, 256])
                    rgb_distribution = np.column_stack((red.flatten(), green.flatten(), blue.flatten()))

                    co_matrix = skimage.feature.graycomatrix(gray, [5], [0], levels=256, normed=True)
                    contrast = skimage.feature.graycoprops(co_matrix, 'contrast')[0, 0]
                    correlation = skimage.feature.graycoprops(co_matrix, 'correlation')[0, 0]
                    energy = skimage.feature.graycoprops(co_matrix, 'energy')[0, 0]
                    homogeneity = skimage.feature.graycoprops(co_matrix, 'homogeneity')[0, 0]
                    

                    row = [city, center_x, center_y, mean, stdev, intensity.tolist(), rgb_distribution.tolist(), contrast, correlation, energy, homogeneity]
                    data.append(row)
        except Exception as e:
            print(f"Failed to process {item}: {e}")

df = pd.DataFrame(data, columns=['Location', 'East', 'North', 'Mean', 'Stdev', 'Intensity', 'RGB', 'Contrast', 'Correlation', 'Energy', 'Homogeneity'])
df.to_csv('data/csv/Features.csv', index=False)
print(df.head())