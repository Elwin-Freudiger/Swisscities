import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#open the csv
data = pd.read_csv('data/full_list.csv')

for row in range(len(data)):
    data['Array'][row] = np.array(data['Array'][row]).reshape((500,500))

print(data['Array'][0])