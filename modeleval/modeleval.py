import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.metrics import (
    precision_recall_curve, precision_score, f1_score, 
    auc, accuracy_score,mean_squared_error, confusion_matrix, 
    classification_report, roc_curve, roc_auc_score, 
    recall_score, make_scorer
)


class BaseEvaluator(object):
    """
    Take a base model object, generate model evaluation result on test data.
    Accepted models are base model in sklearn, xgboost, lightgbm and catboost.
    """
    def __init__(self):
        pass


class BinaryEvaluator(BaseEvaluator):
    """
    For binary classification problem, generate common evaluation metrics
    and plots which are useful to determine the threshold.
    """
    def __init__(self, model, eval_X, eval_y, threshold=0.5):
        super(BaseEvaluator,self).__init__()
        self.model = model
        self.eval_X = eval_X
        self.eval_y = eval_y
        self.threshold = threshold 
    def evaluate(self, metrics = "all"):
        """Make prediction and evaluate"""
        pred = self.model.predict(self.eval_X)
        if metrics == "base" or metrics == "all":
            print("Model Evluation")
            print("The accuracy is %0.003f" % (accuracy_score(pred, self.eval_y)))
            print("The recall for 1 is %0.3f" % (recall_score(self.eval_y, pred, pos_label=1)))
            print("The precision for 1 is %0.3f" % (precision_score(self.eval_y, pred, pos_label=1)))
            print("The F1-score is %0.3f" % f1_score(self.eval_y, pred, pos_label=1))
            print("Confusion Matrics")
            y_true = pd.Series(self.eval_y)
            y_pred = pd.Series(pred)
            confusion = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
            print(confusion)
        if metrics == "all":
            # AUC
            y_probs = self.model.predict_proba(self.eval_X)[:, 1]
            fpr, tpr, auc_thresholds = roc_curve(self.eval_y, y_probs)
            precisions, recalls, thresholds = precision_recall_curve(self.eval_y, y_probs)
            # ROC
            fig = plt.figure(figsize=(10, 20))
            ax1 = fig.add_subplot(411)
            ax1.set_title('ROC Curve')
            ax1.plot(fpr, tpr, linewidth=2)
            ax1.plot([0, 1], [0, 1], 'k--')
            ax1.axis([-0.005, 1, 0, 1.005])
            ax1.set_xticks(np.arange(0, 1, 0.05))
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate (Recall)')

            # Recall_Precision VS Decision Threshold Plot
            ax2 = fig.add_subplot(412)
            ax2.set_title('Precision and Recall vs Decision Threshold')
            ax2.plot(thresholds, precisions[:-1], 'b--', label='Precision')
            ax2.plot(thresholds, recalls[:-1], 'g-', label='Recall')
            ax2.set_ylabel('Score')
            ax2.set_xlabel('Decision Threshold')
            ax2.legend(loc='best')

            # Class Probability Distribution
            ax3 = fig.add_subplot(413)
            ax3.set_title('Class Probability Distribution')
            buy = ax3.hist(y_probs[self.eval_y == 1][:, 1], bins=40,
                           density=True, alpha=0.5)
            nonbuy = ax3.hist(y_probs[self.eval_y == 0][:, 1], bins=40,
                              density=True, alpha=0.5)
            # Feature importance
            if ("RandomForest" or "XGB") in str(type(self.model)): 
                ax4 = fig.add_subplot(414)
                ax4.set_title('Feature Imprtance')
                feature_importance = self.model.feature_importances_
                pd.Series(feature_importance, index=["0",'1','2','3']).nlargest(self.eval_X.shape[1]).plot(kind='barh')

#             # Save the plots
#             plot_path = eval_path + 'multiple_metrics_plot.png'
#             fig.savefig(
#                 plot_path,
#                 bbox_inches='tight'
#             )

#             # Write result into a txt
#             output = '\n'.join([
#                 'Model Evaluation:',
#                 '\tAccuracy: {accuracy:.3f}',
#                 '\tRecall for monetizers: {recall:.3f}',
#                 '\tPrecision for monetizers: {precision:.3f}',
#                 '\tF1 score: {f1:.3f}',
#                 '\tAUC: {auc:.3f}',
#                 '\n',
#                 'Confusion Matrix:',
#                 '{confusion}'
#             ]).format(
#                 accuracy  = accuracy_score(prediction, y_test),
#                 recall    = recall_score(y_test, prediction, pos_label=1),
#                 precision = precision_score(y_test, prediction, pos_label=1),
#                 f1        = f1_score(y_test, prediction, pos_label=1),
#                 auc       = auc(fpr, tpr),
#                 confusion = confusion.to_csv(
#                     sep=' ', index=True, header=True, index_label='Confusion')
#             )
#             result_path = eval_path + 'output.txt'
#             with open(result_path, 'w+') as f:
#                 f.write(output)

