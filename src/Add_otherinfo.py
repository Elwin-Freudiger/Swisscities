import requests
import rasterio
import numpy as np
import csv
import cv2 as cv
import pandas as pd
from math import *
import concurrent.futures


df_top10 = pd.read_csv('data/csv/Features_top10.csv')
df_balanced = pd.read_csv('data/csv/Features_balanced.csv')
#list of all file names
info = ['temperature', 'sunshine', 'rainfall']

def main(df, name):
    for file in info: 
        path = f'data/{file}.tiff' #open the file
        with rasterio.open(path) as dataset:
            meta = dataset.meta
            matrix = dataset.read(1)
            func = meta['transform'] #get the affine transofrmation

            def get_value(est, nord):
                x, y = ~func * (est, nord) #reverse it to find the value
                x = floor(x)
                y = floor(y)
                return matrix[y, x]

        get_value_vector = np.vectorize(get_value) #vectorize the function

        df[file] = get_value_vector(df['East'], df['North']) #create new column for the value
    df.to_csv(f'data/csv/{name}_with_info.csv', index=False) #save it to csv

if __name__ == '__main__':
    main(df_top10, 'top10')
    main(df_balanced, 'balanced')
