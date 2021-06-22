import spacy
import re
import pandas as pd
import os
current_path = os.path.dirname(__file__)


note_1 = pd.read_csv('14653_X光科報告_1.csv')
print(note_1.shape)
note_2 = pd.read_csv('14653_X光科報告_2.csv')
print(note_2.shape)
note_icd9 = pd.concat([note_1, note_2], axis=0)
print(note_icd9.shape)
selected_ids_icd9 = pd.read_csv('selected_ID_icd9_stroke.csv')
note_icd9 = pd.merge(note_icd9, selected_ids_icd9, on=['院區', '資料年月', '歸戶代號'])
print(note_icd9.shape)
selected_cols = ['報告01']
note_icd9 = note_icd9[selected_cols]
# note_icd9 = note_icd9[0:100]

nlp = spacy.load('en_core_sci_md', disable=['tagger', 'ner'])
with open('../data/xnote.txt', 'w', encoding="utf-8") as f:
    for inx, row in note_icd9.iterrows():
        for i in range(len(selected_cols)):
            parg = str(row[selected_cols[i]])
            # remove special characters
            parg = re.sub(r'(^;)|(;;;)|(=+>)|(={2})|(-+>)|(={2})|(\*{3})', '', parg)
            # fix bullet points
            parg = re.sub(r'(\d+[.{1}][\D])', r'. \1', parg)
            # fix period
            parg = re.sub(r'(\s{1}[.])', '. ', parg)
            parg = re.sub(r'(\s{1}[.]\s{1})', '. ', parg)
            parg = re.sub(r'([.]+)', '.', parg)
            # remove multi-spaces
            parg = re.sub(' +', ' ', parg)
            # remove Chinese
            parg = re.sub(r'[\u4e00-\u9fff]+', '', parg)

            sentences = nlp(parg)
            for sentence in sentences.sents:
                if len(sentence) > 3:
                    # remove bullet point number
                    see_sentence = re.sub(r'(^\d+\.)|(^\s+)', '', sentence.text)
                    # trim white space at the head and the end
                    see_sentence = see_sentence.strip()
                    f.write(see_sentence)
                    f.write('\n')
        f.write('\n')
        print(inx)

print('done')
