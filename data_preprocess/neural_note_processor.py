import spacy
import re
import pandas as pd
import os
current_path = os.path.dirname(__file__)


note_1 = pd.read_csv('14653_腦神經內科報告_1.csv')

# remove M29-079: carotid report...
note_1 = note_1[note_1['檢查項目'] != 'M29-079']
# remove M28-057: PSYCHOPHYSIOLOGICAL report...
note_1 = note_1[note_1['檢查項目'] != 'M28-057']

selected_ids_icd9 = pd.read_csv('selected_ID_icd9_stroke.csv')
note_icd9 = pd.merge(note_1, selected_ids_icd9, on=['院區', '資料年月', '歸戶代號'])
print(note_icd9.shape)
selected_cols = ['檢查項目', '報告01']
note_icd9 = note_icd9[selected_cols]

# note_icd9 = note_icd9[0:100]

delimiters = ['Conclusion:', 'Comments:', 'Interpretation:', 'INTERPRETATION:', 'Doppler Findings:', 'COMMENTS:', 'ULTRASOUND DIAGNOSIS:']
nlp = spacy.load('en_core_sci_md', disable=['tagger', 'ner'])
with open('../data/nu_note.txt', 'w', encoding="utf-8") as f:
    for inx, row in note_icd9.iterrows():
        new_line_flag = False
        parg = str(row[selected_cols[1]])
        for delimiter in delimiters:
            start_position = parg.find(delimiter)
            if start_position != -1:
                sentence = parg[start_position:len(parg)]
                if re.search(r'[.]\s', sentence):
                    sub_sentence = sentence[0:re.search(r'[.]\s', sentence).start()]
                    p = re.sub(r' +', ' ', sub_sentence)
                    p = re.sub(r'[\u4e00-\u9fff]+', '', p)
                    p = re.sub(r'[*]+', '', p)
                    p = p.replace(delimiter+' ', '')
                    if not len(p.split()) < 3:
                        f.write(p)
                        f.write('\n')
                        new_line_flag = True
        if new_line_flag:
            f.write('\n')

print('done')
