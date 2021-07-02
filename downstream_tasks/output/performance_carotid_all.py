import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, classification_report
import os
from scipy import interp
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

model_name = 'sb_all_'
labels = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA',
          'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']
for lab_inx in range(17):
    i = 0
    tprs = []
    aucs = []
    f1s = []
    for inx in range(10):
        read_path = os.path.join('carotid', model_name+str(inx), 'test_prediction.pickle')
        with open(read_path, 'rb') as f:
            data = pickle.load(f)
            all_logits = data['all_logits']
            all_labels = data['all_labels']
            mean_fpr = np.linspace(0, 1, 100)
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(all_labels[:, lab_inx], all_logits[:, lab_inx])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # plt.plot(fpr, tpr, lw=1, alpha=0.3,
            #          label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            y = np.where(all_logits[:, lab_inx] < 0, 0, 1)
            f1s.append(f1_score(all_labels[:, lab_inx], y, average='micro'))
            i += 1
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #          label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    mean_f1 = np.mean(f1s, axis=0)
    std_f1 = np.std(f1s)
    # print(model_name)
    # print(labels[lab_inx])
    print(round(mean_auc,3), round(std_auc,3))
    # print(mean_f1)
    # print(std_f1)
    # plt.plot(mean_fpr, mean_tpr, color='b',
    #          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #          lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()