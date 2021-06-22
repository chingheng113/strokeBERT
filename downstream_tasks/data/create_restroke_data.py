import pandas as pd
import pickle
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import os
import re
current_path = os.path.dirname(__file__)


def clean_text(parg):
    if parg == parg:
        parg = re.sub(r'(:nil)', '', parg)
        # remove special characters
        parg = re.sub(r'(\s\.)|(\.{2,})|(:{2,})|(^;)|(;;;)|(=+>)|(={2})|(-+>)|(={2})|(\*{3})', '', parg)
        # fix bullet points
        parg = re.sub(r'(\d+[.{1}][\D])', r'. \1', parg)
        # remove Chinese
        parg = re.sub(r'[\u4e00-\u9fff]+', ' ', parg)
        # remove multi-spaces
        parg = re.sub(r'[\s\t]+', ' ', parg)
        # remove date
        parg = re.sub(r'\d{2,4}[/]\d{1,2}[/]\d{1,2}', '', parg)
    return parg


def save_variable(val, val_name):
    save_path = os.path.join(current_path, val_name)
    with open(save_path, 'wb') as file_pi:
        pickle.dump(val, file_pi)

days = ['m', '360', 'all']
for day in days:
    df = pd.read_csv('recurrent_stroke_ds_'+day+'.csv')
    df.dropna(axis=0, subset=['主訴', '病史', '住院治療經過'], inplace=True)

    df['processed_content'] = df['主訴']+". "+df['病史']+". "+df['住院治療經過']
    df['processed_content'] = df['processed_content'].apply(clean_text)
    data = df[['歸戶代號', 'processed_content', 'label']]

    data.rename(columns={'歸戶代號':'ID'}, inplace=True)
    for i in range(10):
        training_data, testing_data = train_test_split(data, test_size=0.2, random_state=i)
        # sample balance on training data
        # if training_data[training_data.label == 0].shape[0] > training_data[training_data.label == 1].shape[0]:
        #     resampled = resample(training_data[training_data.label == 0],
        #                          replace=False,
        #                          n_samples=training_data[training_data.label == 1].shape[0],
        #                          random_state=i)
        #     training_data = pd.concat([training_data[training_data.label == 1], resampled])
        # else:
        #     resampled = resample(training_data[training_data.label == 1],
        #                          replace=False,
        #                          n_samples=training_data[training_data.label == 0].shape[0],
        #                          random_state=i)
        #     training_data = pd.concat([training_data[training_data.label == 0], resampled])
        #
        dir_path = str(os.path.join('reStroke_'+day, 'round_' + str(i)))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        training_data.to_csv(os.path.join(dir_path, 'training_' + str(i) + '.csv'), index=False)
        testing_data.to_csv(os.path.join(dir_path, 'testing_' + str(i) + '.csv'), index=False)
        save_variable(training_data, os.path.join(dir_path, 'training_bert.pickle'))
        save_variable(testing_data, os.path.join(dir_path, 'test_bert.pickle'))
        print(i)

print('done')