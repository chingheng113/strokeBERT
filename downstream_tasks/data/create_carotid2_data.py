import pandas as pd
import os
import pickle
import re
current_path = os.path.dirname(__file__)

def clean_text(parg):
    if parg == parg:
        finding_start = re.search('findings:', parg, re.IGNORECASE)
        finding_end = re.search('IMPRESSION:', parg, re.IGNORECASE)
        if finding_end is None:
            finding_end = re.search('IMPRESSIONS:', parg, re.IGNORECASE)
        parg = parg[finding_start.regs[0][1]:finding_end.regs[0][0]]
        parg = re.sub(r'(:nil)', '', parg)
        # remove special characters
        parg = re.sub(r'(\s\.)|(\.{2,})|(:{2,})|(^;)|(;;;)|(=+>)|(={2})|(-+>)|(={2})|(\*{3})|(-{2,}|(_{2,}))', '', parg)
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

# read data
data = pd.read_csv('carotid2.csv')
data.rename(columns={'IDCode': 'ID', 'RTRESTXT': 'processed_content'}, inplace=True)
data.dropna(subset=['processed_content'], axis=0, inplace=True)
data['processed_content'] = data['processed_content'].apply(clean_text)

data.to_csv(os.path.join('carotid2', 'testing.csv'), index=False)
save_variable(data, os.path.join('carotid2', 'test_bert.pickle'))
# There is no training at all... this is independent dataset validation
# Here is same data but different file name... just for making code running
data.to_csv(os.path.join('carotid2', 'training.csv'), index=False)
save_variable(data, os.path.join('carotid2', 'training_bert.pickle'))


print('done')