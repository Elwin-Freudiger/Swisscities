import requests
import rasterio
import numpy as np
import csv
import cv2 as cv
import pandas as pd
from math import *

#for temperature
df = pd.read_csv('data/csv/Ten_cities_lower.csv')

info = ['temperature', 'sunshine', 'rainfall']

for file in info:
    path = f'data/{file}.tiff'
    with rasterio.open(path) as dataset:
        meta = dataset.meta
        matrix = dataset.read(1)
        func = meta['transform']

        def get_value(est, nord):
            x, y = ~func * (est, nord)
            x = floor(x)
            y = floor(y)
            return matrix[y, x]

    get_value_vector = np.vectorize(get_value)

    df[file] = get_value_vector(df['East'], df['North'])

df.to_csv('data/csv/Ten_cities_with_info.csv', index=False)
