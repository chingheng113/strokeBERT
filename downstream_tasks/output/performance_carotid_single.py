import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import os
import pickle

model_name = 's_all_0'
print(model_name)
read_path = os.path.join('carotid', model_name, 'test_prediction.pickle')
labels = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA',
          'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']
with open(read_path, 'rb') as f:
    data = pickle.load(f)
    all_logits = data['all_logits']
    all_labels = data['all_labels']
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(17):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(labels[i])
        print(roc_auc[i])
        p_label_b = (all_logits[:, i] > 0.5).astype(int)
        print(classification_report(all_labels[:, i], p_label_b))
        print(confusion_matrix(all_labels[:, i], p_label_b))
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print('------')
    print(roc_auc["micro"])
    print('done')