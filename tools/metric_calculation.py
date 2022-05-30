import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def calc_roc_auc(pred_file, gt_file):
    preds = pd.read_csv(pred_file)
    gts = pd.read_csv(gt_file)

    scores = {}
    for i in range(len(preds)):
        if preds['imageA'][i] not in scores:
            scores[preds['imageA'][i]] = {}
        scores[preds['imageA'][i]][preds['imageB'][i]] = preds['prediction'][i]


    pred_scores, gt_scores = [], []
    for i in range(len(gts)):
        gt_scores.append(gts['label'][i])
        pred_scores.append(scores[gts['imageA'][i]][gts['imageB'][i]])

    pred_scores, gt_scores = np.asarray(pred_scores), np.asarray(gt_scores)

    auc_score = roc_auc_score(gt_scores, pred_scores)
    return auc_score