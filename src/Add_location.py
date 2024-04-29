import requests
import rasterio
import numpy as np
import csv
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

path = 'data/GeoJson/swissBOUNDARIES3D_1_5_LV95_LN02.gdb'
swiss_cantons = gpd.read_file(path, layer='TLM_KANTONSGEBIET')
swiss_cities = gpd.read_file(path, layer='TLM_HOHEITSGEBIET')

def which_kanton(east, north):
    kanton = 'foreign'
    inside = False
    coord = Point(east, north)
    i = 0
    while not inside and i <= 25:
        inside = coord.within(swiss_cantons['geometry'][i])
        if inside:
            kanton = swiss_cantons['NAME'][i]
            break
        i += 1
    return kanton
"""
#open file 
df = pd.read_csv('data/csv/All_images.csv')

which_kanton = np.vectorize(which_kanton)

df['Kanton'] = which_kanton(df['East'], df['North'])

df.to_csv('data/csv/Images_with_canton.csv', index=False)
"""

