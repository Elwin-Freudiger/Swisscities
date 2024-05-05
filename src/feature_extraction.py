import requests
import rasterio
import numpy as np
import csv
import cv2 as cv
import pandas as pd
import skimage
from alive_progress import alive_bar; import time
from concurrent.futures import ThreadPoolExecutor, as_completed

#City_list = ['Zurich', 'Geneva', 'Basel', 'Lausanne', 'Bern', 'Winterthur', 'Luzern', 'St_Gallen', 'Lugano', 'Biel']
Better_list = ['Bern', 'Zurich', 'Lugano', 'Lausanne', 'Chur', 'Schwyz', 'Glarus', 'Winterthur', 'Sarnen', 'Nendaz']

data = []

def extraction(link, city):
    try:
        response = requests.get(link, timeout=10)
        response.raise_for_status()
        with rasterio.io.MemoryFile(response.content) as file:
            with file.open() as dataset:
                img_array = dataset.read([1, 2, 3]).transpose((1, 2, 0))  # Correct dimension order for OpenCV
                metadata = dataset.meta
                corner = metadata['transform']
                center_x, center_y = corner * (metadata['width'] // 2, metadata['height'] // 2)

                mean = np.mean(img_array)
                stdev = np.std(img_array)

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
                
                row = [city, center_x, center_y, mean, stdev, distribution.tolist(), contrast, correlation, energy, homogeneity]
                return row
    except Exception as e:
        print(f"Failed to process {link}: {e}")
        return None
        

def main():
    for city in Better_list:
        path = f'data/city_link/{city}.csv'
        with open(path, 'r') as link_list:
            links = link_list.read().splitlines()
        with ThreadPoolExecutor(max_workers=10) as executor:
            with alive_bar(len(links)) as bar:
                future_images = {executor.submit(extraction, link, city): link for link in links}
                for future in as_completed(future_images):
                    row = future.result()
                    if row is not None:
                        data.append(row)
                    bar()
                    
    df = pd.DataFrame(data, columns=['Location', 'East', 'North', 'Mean', 'Stdev', 'Distribution', 'Contrast', 'Correlation', 'Energy', 'Homogeneity'])
    df.to_csv('data/csv/City_balanced.csv', index=False)

if __name__ == '__main__':
    main()