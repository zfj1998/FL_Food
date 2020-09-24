import numpy as np
def get_metrics(y_true, y_pred):
    # 获取常见的4个值，用于系列指标计算
    TN, FP, FN, TP = np.fromiter((sum(
        bool(j >> 1) == bool(y_true[i]) and
        bool(j & 1) == bool(y_pred[i])
        for i in range(len(y_true))
    ) for j in range(4)), float)

    # Accuracy = (TN + TP) / (TN + FP + FN + TP + 1e-8)
    Precision = TP / (TP + FP + 1e-8)
    # True Positive Rate
    Recall = TP / (TP + FN + 1e-8)
    # False Positive Rate
    FPR = FP / (FP + TN + 1e-8)

    print("Precision", Precision)
    print("Recall", Recall)

    # F_measure = 2 * Recall * Precision / (Recall + Precision + 1e-8)
    # g_mean = np.sqrt((TN / (TN + FP + 1e-8)) * (TP / (TP + FN + 1e-8)))
    # Balance = 1 - np.sqrt((0 - FPR) ** 2 + (1 - Recall) ** 2) / np.sqrt(2)
    MCC = (TP * TN - FN * FP) / np.sqrt((TP + FN) * (TP + FP) * (FN + TN) * (FP + TN) + 1e-8)

    # 当F_measure中θ值为2时
    F_2 = 5 * Recall * Precision / (4 * Recall + Precision + 1e-8)
    # G_measure = 2 * Recall * (1 - FPR) / (Recall + (1 - FPR) + 1e-8)
    # NMI = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")

    # 返回所有指标值 vars() 函数返回对象object的属性和属性值的字典对象。
    y_pred = vars()
    # 该字典不返回'y_true', 'y_pred', "TN", "FP", "FN", "TP"这些key值
    return {k: y_pred[k] for k in reversed(list(y_pred)) if k not in ['y_true', 'y_pred', "TN", "FP", "FN", "TP", "FPR"]}
