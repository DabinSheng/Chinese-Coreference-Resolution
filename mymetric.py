import visdom
from sklearn.metrics import precision_recall_curve, auc
def mymetric(all_targets, all_predictions):
    viz = visdom.Visdom()
    precision, recall, _ = precision_recall_curve(all_targets, all_predictions)
    # 计算PR曲线下的面积
    pr_auc = auc(recall, precision)
    # 绘制Precision-Recall曲线
    viz.line(
        X=recall,
        Y=precision,
        opts=dict(title='Precision-Recall Curve', xlabel='Recall', ylabel='Precision')
    )
    # 显示PR曲线下的面积
    viz.text(
        f'AUC: {pr_auc}',
        opts=dict(title='Area Under Curve (AUC)')
    )