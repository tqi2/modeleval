import numpy as np
import pandas as pd
import os
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
    A base evaluator object, generate model evaluation result on test data.
    Accepted models are base model in sklearn, xgboost, lightgbm and catboost.
    """
    def __init__(self):
        pass
    
    
    def ensure_path(self, file_path):
        """Helper function to create/ensure the path"""
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)


class BinaryEvaluator(BaseEvaluator):
    """
    For binary classification problem, generate common evaluation metrics
    and plots which are useful to evaluate your model performance and 
    determine the threshold.
    """
    def __init__(self):
        super(BinaryEvaluator, self).__init__()

        
    def evaluate(self, model, eval_X, eval_y, threshold=0.5, metrics = "all", save=False, save_folder="result"):
        """Make prediction and evaluation based on specified threshold.

        Parameters
        ----------
        model : sklearn/lightgbm/xgboost/catboost classification model object
            The model for evaluation.

        eval_X : ndarray or pd.DataFrame
            The test data's features.

        eval_y : ndarray
            The test data's labels.

        metrics : string, optional (default="all")
            The metrics for evaluating the model. If "base", return only common metrics. If "all", return
            plots including ROC curve, Precision_Recall vs threshold, class probability distribution and
            feature importance as well.

        save : bool, optional (default=False)
            Whether to save the result.

        save_folder : string (default="result")
            The folder path to save the result, default is the result folder in the current directory.

        """
        if isinstance(eval_X, pd.DataFrame):
            eval_X = eval_X.values
            X_cols = X.columns
        y_probs = model.predict_proba(eval_X)[:, 1]
        if threshold == 0.5:
            pred = model.predict(eval_X)
        else:
            pred = np.where(y_probs>threshold, 1, 0)       
        accuracy  = accuracy_score(pred, eval_y)
        recall_1    = recall_score(eval_y, pred, pos_label=1)
        precision_1 = precision_score(eval_y, pred, pos_label=1)
        recall_0    = recall_score(eval_y, pred, pos_label=0)
        precision_0 = precision_score(eval_y, pred, pos_label=0)
        f1        = f1_score(eval_y, pred, pos_label=1)
        roc_auc       = roc_auc_score(eval_y, y_probs)
        y_true = pd.Series(eval_y)
        y_pred = pd.Series(pred)
        confusion = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
        if metrics == "basic" or metrics == "all":
            print("Evaluation result of Threshold=={thres}".format(thres=threshold))
            print("---Common Metrics---")
            print("The accuracy is %0.4f" % accuracy)
            print("The recall for 1 is %0.4f" % recall_1)
            print("The precision for 1 is %0.4f" % precision_1)
            print("The recall for 0 is %0.4f" % recall_0)
            print("The precision for 0 is %0.4f" % precision_0)
            print("The F1-score is %0.4f" % f1)
            print("The ROC-AUC is %0.4f" % roc_auc)
            print("\n---Confusion Matrix---")
            print(confusion)
        if metrics == "all":
            # AUC
            fpr, tpr, auc_thresholds = roc_curve(eval_y, y_probs)
            precisions, recalls, thresholds = precision_recall_curve(eval_y, y_probs)

            # ROC
            fig = plt.figure(figsize=(10, 27))
            plt.subplots_adjust(hspace=0.25)
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
            ax3.set_ylabel('Density')
            ax3.set_xlabel('Predicted Probability')
            ax3.hist(y_probs[eval_y == 1], bins=40,
                           density=True, alpha=0.5)
            ax3.hist(y_probs[eval_y == 0], bins=40,
                              density=True, alpha=0.5)
            
            # Feature importance
            model_list = ["DecisionTree","RandomForest", "XGB", "LGBM"]
            if any(mod_name in str(type(model)) for mod_name in model_list): 
                ax4 = fig.add_subplot(414)
                ax4.set_title('Feature Importance')
                feature_importance = model.feature_importances_
                try:
                    X_cols
                except:
                    pd.Series(feature_importance, index=range(eval_X.shape[1])).nlargest(eval_X.shape[1]).plot(kind='barh')
                    ax4.set_ylabel('Column Index')
                else:
                    pd.Series(feature_importance, index=X_cols).nlargest(eval_X.shape[1]).plot(kind='barh')
        if save:
            super(BinaryEvaluator, self).ensure_path(save_folder)
            plot_path = save_folder + 'multiple_metrics_plots.png'
            if fig:
                fig.savefig(
                    plot_path,
                    bbox_inches='tight'
                )
            #Write result into a txt
            output = '\n'.join([
                '--Model Evaluation--',
                '\tAccuracy: {accuracy:.4f}',
                '\tRecall for 1: {recall_1:.4f}',
                '\tPrecision for 1: {precision_1:.4f}',
                '\tRecall for 0: {recall_0:.4f}',
                '\tPrecision for 0: {precision_0:.4f}',
                '\tF1 score: {f1:.4f}',
                '\tROC-AUC: {roc_auc:.4f}',
                '\n',
                '--Confusion Matrix--',
                '{confusion}'
            ]).format(
                accuracy = accuracy,
                recall_1 = recall_1,
                precision_1 = precision_1,
                recall_0 = recall_0,
                precision_0 = precision_0,
                f1 = f1,
                roc_auc = roc_auc,
                confusion = confusion
            )
            result_path = save_folder + 'output.txt'
            with open(result_path, 'w+') as f:
                f.write(output)
            

    def find_threshold(self, start=0, stop=1, step=0.1):
        """show the result of common metrics of the threshold within a given interval.

        Parameters
        ----------
        start : int or float
            The start point of the threshold.

        end : int or float
            The end point of the threshold.

        step: float
            The increment step of the threshold.
        """
        pass