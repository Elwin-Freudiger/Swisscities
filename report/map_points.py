import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import contextily as ctx
import xyzservices.providers as xyz

csv_file = 'data/csv/Features_balanced.csv'
df = pd.read_csv(csv_file)

# Define the EPSG 2056 CRS
crs_2056 = "EPSG:2056"

path = 'report/swissBOUNDARIES3D_1_5_LV95_LN02.gdb'
swiss_borders = gpd.read_file(path, layer='TLM_HOHEITSGRENZE')
canton_border = swiss_borders[swiss_borders['OBJEKTART'].isin([0])]

geometry = [Point(xy) for xy in zip(df['East'], df['North'])]

gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs=crs_2056)

fig, ax = plt.subplots(figsize=(10, 10))
canton_border['geometry'].plot(ax=ax, color='grey', alpha=0)
gdf_points.plot(ax=ax, color='red', markersize=1)

swiss_basemap = xyz.SwissFederalGeoportal.NationalMapColor
ctx.add_basemap(ax, crs=crs_2056, source=swiss_basemap)
plt.axis('off')
plt.savefig('report/balanced_map.png', dpi=300)
plt.show()
