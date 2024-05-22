import requests
import rasterio
import numpy as np
import csv
import cv2 as cv
import pandas as pd
from math import *
import concurrent.futures

#for temperature
df_top10 = pd.read_csv('data/csv/Features_top10.csv')
df_balanced = pd.read_csv('data/csv/Features_balanced.csv')

info = ['temperature', 'sunshine', 'rainfall']

def main(df, name):
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
    df.to_csv(f'data/csv/{name}_with_info.csv', index=False)

if __name__ == '__main__':
    main(df_top10, 'top10')
    main(df_balanced, 'balanced')
