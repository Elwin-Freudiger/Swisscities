import requests
import rasterio
import numpy as np
import csv
import cv2 as cv
import pandas as pd
import random
import skimage
from alive_progress import alive_bar; import time

list_kanton = [
    'Bern', 'Graubünden', 'Valais', 'Ticino', 'Vaud', 'St.Gallen', 'Zürich', 'Fribourg',
    'Aargau', 'Luzern', 'Uri', 'Thurgau', 'Jura', 'Neuchâtel', 'Schwyz', 'Solothurn',
    'Glarus', 'Obwalden', 'Schaffhausen', 'Appenzell-Ausserrhoden',
    'Genève', 'Nidwalden', 'Appenzell-Innerrhoden', 'Zug', 'Basel']

data = []

#loop for every city in our list
for kanton in list_kanton:
    canton_links = []
    path = f'data/Kanton_link/{kanton}.csv'
    with open(path, 'r') as link_list:
        canton_links = link_list.read().split('\n')
    random_integers = random.sample(range(len(canton_links)), 200)
    with alive_bar(200) as bar:
        #read every link and download it's content
        for rank in random_integers:
            try:
                response = requests.get(canton_links[rank])
                if response.status_code == 200:
                    with open('swissimage2.tif', 'wb') as f:
                        f.write(response.content)
                    #open image with rasterio
                    with rasterio.open('swissimage2.tif') as dataset:
                        img_array = dataset.read([1, 2, 3]).transpose((1, 2, 0)) #read rgb array
                        metadata = dataset.meta #read metadata
                        corner = metadata['transform']
                        center_x, center_y = corner * (metadata['width']//2, metadata['height']//2)
                        #mean pixels and stdeviation
                        mean = np.mean(img_array)
                        stdev = np.std(img_array)

                        #get rgb and gray intensity distribution
                        gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
                        gray_dist = cv.calcHist([gray], [0], None, [256], [0,256])                    
                        red = cv.calcHist([img_array], [0], None, [256], [0, 256])
                        green = cv.calcHist([img_array], [1], None, [256], [0, 256])
                        blue = cv.calcHist([img_array], [2], None, [256], [0, 256])
                        distribution = np.column_stack((gray_dist.flatten(), red.flatten(), green.flatten(), blue.flatten()))

                        co_matrix = skimage.feature.graycomatrix(gray, [5], [0], levels=256, normed=True)
                        contrast = skimage.feature.graycoprops(co_matrix, 'contrast')[0, 0]
                        correlation = skimage.feature.graycoprops(co_matrix, 'correlation')[0, 0]
                        energy = skimage.feature.graycoprops(co_matrix, 'energy')[0, 0]
                        homogeneity = skimage.feature.graycoprops(co_matrix, 'homogeneity')[0, 0]

                        row = [kanton, center_x, center_y, mean, stdev, distribution, contrast, correlation, energy, homogeneity]
                        data.append(row)
            except Exception as e:
                (f"Failed to process {canton_links[rank]}: {e}")
            bar()
    print(f'{kanton} done')

df = pd.DataFrame(data, columns=['Location', 'East', 'North', 'Mean', 'Stdev', 'Distribution', 'Contrast', 'Correlation', 'Energy', 'Homogeneity'])
df.to_csv('data/csv/Kanton_features.csv', index=False)