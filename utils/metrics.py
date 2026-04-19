from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

def fpr95(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = (tpr >= 0.95).argmax()
    return fpr[idx]

def ood_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)

def ood_aupr(y_true, y_score):
    return average_precision_score(y_true, y_score)