import pandas as pd
import numpy as np
import os
import pickle
import re
from sklearn.model_selection import train_test_split
current_path = os.path.dirname(__file__)


def save_variable(val, val_name):
    save_path = os.path.join(current_path, val_name)
    with open(save_path, 'wb') as file_pi:
        pickle.dump(val, file_pi)

# read data
data = pd.read_csv('carotid_downstream.csv')
data.dropna(subset=['processed_content'], axis=0, inplace=True)
selected_cols = ['ID', 'processed_content', 'RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA',
                 'LCCA', 'LEICA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']
data = data[selected_cols]
# id_data = data[['ID']]
# x_data = data[['processed_content']]
# y_data = data[['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA',
#                'LCCA', 'LEICA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']]

for i in range(10):
    training_data, testing_data = train_test_split(data, test_size=0.2, random_state=i)
    # dir_path = 'round_'+str(i)
    dir_path = str(os.path.join('carotid', 'round_' + str(i)))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    training_data.to_csv(os.path.join(dir_path, 'training_'+ str(i) + '.csv'), index=False)
    testing_data.to_csv(os.path.join(dir_path, 'testing_' + str(i) + '.csv'), index=False)
    save_variable(training_data, os.path.join(dir_path, 'training_bert.pickle'))
    save_variable(testing_data, os.path.join(dir_path, 'test_bert.pickle'))
    print(i)

print('done')