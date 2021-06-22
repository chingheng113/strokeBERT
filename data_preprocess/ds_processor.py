import spacy
import re
import pandas as pd
import os
from collections import Counter
current_path = os.path.dirname(__file__)

selected_cols = ['主訴', '病史', '手術日期、方法及發現', '住院治療經過']

# select patients
note_icd9 = pd.read_csv('14653_出院病摘_1_icd9.csv')
print(note_icd9.shape)
selected_ids_icd9 = pd.read_csv('selected_ID_icd9_stroke.csv')
note_icd9 = pd.merge(note_icd9, selected_ids_icd9, on=['院區', '資料年月', '歸戶代號'])
print(note_icd9.shape)
note_icd9 = note_icd9[selected_cols]
#
# note_icd9 = note_icd9[0:100]

# see https://github.com/allenai/scispacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.3/en_core_sci_md-0.2.3.tar.gz
nlp = spacy.load('en_core_sci_md', disable=['tagger','ner'])
with open('../data/ds.txt', 'w', encoding="utf-8") as f:
    for inx, row in note_icd9.iterrows():
        for i in range(len(selected_cols)):
            # print(selected_cols[i])
            parg = str(row[selected_cols[i]])
            # remove special characters
            parg = re.sub(r'(^;)|(;;;)|(=+>)|(={2})|(-+>)|(={2})|(\*{3})', '', parg)
            # fix bullet points
            parg = re.sub(r'(\d+[.{1}][\D])', r'. \1', parg)
            # remove multi-spaces
            parg = re.sub(r' +', ' ', parg)
            # print(parg)
            # remove Chinese
            parg = re.sub(r'[\u4e00-\u9fff]+', '', parg)
            # remove date
            parg = re.sub(r'\d{2,4}[/]\d{1,2}[/]\d{1,2}', '', parg)
            sentences = nlp(parg)
            for sentence in sentences.sents:
                # remove bullet point number
                see_sentence = re.sub(r'(^\d+\.)|(^\s+)', '', sentence.text)
                # trim white space at the head and the end
                see_sentence = see_sentence.strip()
                # if the sentence is very short or doesn't contain any word or start with non-character, remove it!
                if not ((len(see_sentence.split()) < 3) | (see_sentence == 'nil') | (len(re.findall(r'[a-zA-Z_]{2}', see_sentence)) < 3)):
                    if not re.search(r'\W', see_sentence).start() == 0:
                        # print(see_sentence)
                        f.write(see_sentence)
                        f.write('\n')
        f.write('\n')
        print(inx)

print('done')
