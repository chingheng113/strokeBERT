import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import os
import pickle
import numpy as np

model_name = 'sb_24'
print(model_name)
read_path = os.path.join('carotid', model_name, 'test_prediction.pickle')
labels = ['RIICA', 'RACA', 'RMCA', 'RPCA', 'RIVA', 'BA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LIVA']
#labels = ['RCCA', 'REICA', 'REVA', 'LCCA', 'LEICA', 'LEVA']
with open(read_path, 'rb') as f:
    data = pickle.load(f)
    all_logits = data['all_logits']
    all_labels = data['all_labels']
    fprs = []
    tprs = []
    roc_aucs = []
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc = auc(fpr, tpr)
        if ~np.isnan(roc_auc):
            roc_aucs.append(auc(fpr, tpr))
            print(labels[i])
            print(auc(fpr, tpr))
            p_label_b = (all_logits[:, i] > 0.5).astype(int)

    print(round(np.mean(roc_aucs),3), round(np.std(roc_aucs),3))
    print('done')