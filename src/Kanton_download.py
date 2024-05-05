import requests
import rasterio
import numpy as np
import csv
import cv2 as cv
import pandas as pd
import geopandas as gpd
import random

list_kanton = [
    'Graubünden', 'Bern', 'Valais', 'Ticino', 'Vaud', 'St.Gallen', 'Zürich', 'Fribourg',
    'Aargau', 'Luzern', 'Uri', 'Thurgau', 'Jura', 'Neuchâtel', 'Schwyz', 'Solothurn',
    'Glarus', 'Obwalden', 'Basel-Landschaft', 'Schaffhausen', 'Appenzell-Ausserrhoden',
    'Genève', 'Nidwalden', 'Appenzell-Innerrhoden', 'Zug', 'Basel'
]

#formula to convert rgb image into a grayscale
rgb_array = np.array([[0.2989], [0.5870], [0.1140]])
def rgb2gray(rgb):
    return np.dot(rgb.T, rgb_array).squeeze().T

def generate_unique_randoms(n, x):
    if n > x + 1:
        raise ValueError("n must be less than or equal to x+1")
    return random.sample(range(x + 1), n)

#open the csv and write the headers
with open('data/csv/Kanton.csv', 'w', newline='') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(['Location', 'East', 'North', 'Array'])

#loop for every city in our list
for kanton in list_kanton:
    path = f'data/Kanton_link/{kanton}.csv'
    
    links = []
    with open(path, 'r') as link_list:
        links = link_list.read().splitlines()

        random_integers = generate_unique_randoms(200, len(links))
        #read every link and download it's content
        for item in random_integers:
            try:
                response = requests.get(links[item])
                if response.status_code == 200:
                    with open('swissimage.tif', 'wb') as f:
                        f.write(response.content)
                    #open image with rasterio
                    with rasterio.open('swissimage.tif') as dataset:
                        img_array = dataset.read([1, 2, 3]) #read rgb array
                        metadata = dataset.meta #read metadata
                        corner = metadata['transform']
                        center_x, center_y = corner * (metadata['width']/2, metadata['height']/2)

                        gray = rgb2gray(img_array) #transform into grayscale array
                        gray_list = gray.round().astype(int).flatten().tolist()
                        #add line to csv
                        with open('data/csv/Kanton.csv', 'a', newline='') as file:
                            csvwriter = csv.writer(file)
                            csvwriter.writerow([kanton, center_x, center_y, gray_list])
            except Exception: #expection in case the link is broken
                print(f"Failed to process")
    print(f'{kanton} done')