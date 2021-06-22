import pickle
import pandas as pd
import os

path_g1 = os.path.join('data', 'restroke', 'round_0', 'test_bert.pickle')
f_g1 = open(path_g1, 'rb')
data_g1 = pickle.load(f_g1)
print(data_g1.shape)

# path_g2 = os.path.join('data', 'restroke', 'round_1', 'training_bert.pickle')
# f_g2 = open(path_g2, 'rb')
# data_g2 = pickle.load(f_g2)


path_b1 = os.path.join('data', 'restroke', 'round_2', 'test_bert.pickle')
f_b1 = open(path_b1, 'rb')
data_b1 = pickle.load(f_b1)


# path_b2 = os.path.join('data', 'restroke', 'round_7', 'training_bert.pickle')
# f_b2 = open(path_b2, 'rb')
# data_b2 = pickle.load(f_b2)


gb = pd.merge(data_g1, data_b1, how='inner', on='ID')
print(gb.shape)

see = data_b1[~data_b1.ID.isin(gb.ID)]
print(see.shape)
see.to_csv('see.csv', index=False)

see2 = data_g1[~data_g1.ID.isin(gb.ID)]
print(see2.shape)
see2.to_csv('see2.csv', index=False)

# bb = pd.merge(data_b1, data_b2, how='inner', on='ID')
# print(bb.shape)


print('done')
