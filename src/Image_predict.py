import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#open the csv
data = pd.read_csv('data/csv/full_list.csv')

def string_to_array(string):
    elements = string.strip('[]').split(',')
    array = np.array(elements, dtype=int).reshape((500, 500))
    return array

data['Array'] = data['Array'].apply(string_to_array)

print(data['Array'][0])