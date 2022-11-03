from sklearn import metrics
from sklearn.metrics import confusion_matrix


def prec_rec_f1_acc_mcc(y_true, y_pred, num_classes):
    performance_threshold_dict = dict()
    y_true_tmp = []
    for each_y_true in y_true:
        y_true_tmp.append(each_y_true)
    y_true = y_true_tmp

    y_pred_tmp = []
    for each_y_pred in y_pred:
        y_pred_tmp.append(each_y_pred)
    y_pred = y_pred_tmp
    if num_classes == 2:
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
        auroc = metrics.auc(fpr, tpr)
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred)
    else:
        precision = metrics.precision_score(y_true, y_pred, average='weighted')
        recall = metrics.recall_score(y_true, y_pred, average='weighted')
        f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
    accuracy = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    if num_classes == 2:
        performance_threshold_dict["Auroc"] = auroc
        performance_threshold_dict["Roc_auc"] = roc_auc
        performance_threshold_dict["TP"] = tp
        performance_threshold_dict["FP"] = fp
        performance_threshold_dict["TN"] = tn
        performance_threshold_dict["FN"] = fn

    performance_threshold_dict["Precision"] = precision
    performance_threshold_dict["Recall"] = recall
    performance_threshold_dict["F1-Score"] = f1_score
    performance_threshold_dict["Accuracy"] = accuracy
    performance_threshold_dict["MCC"] = mcc

    return performance_threshold_dict


def get_list_of_scores(num_classes):
    if num_classes == 2:
        return ["Auroc", "Roc_auc", "Precision", "Recall", "F1-Score", "Accuracy", "MCC", "TP", "FP", "TN", "FN"]
    else:
        return ["Precision", "Recall", "F1-Score", "Accuracy", "MCC"]
