import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import os
import pickle

model_name = 'sb_24'
print(model_name)
read_path = os.path.join('restroke_all', model_name, 'test_prediction.pickle')
with open(read_path, 'rb') as f:
    data = pickle.load(f)
    all_logits = data['all_logits']
    all_labels = data['all_labels']
    fpr, tpr, _ = roc_curve(all_labels, all_logits[:, 1])
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    p_label_b = (all_logits[:, 1] > 0.5).astype(int)
    print(classification_report(all_labels, p_label_b))
    print(confusion_matrix(all_labels, p_label_b))
    print('done')