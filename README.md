# modeleval

A python package for machine learning model's evaluation, fully compatible ***scikit-learn***,  ***xgboost*** and  ***LightGBM API***. 

By <a href="https://www.linkedin.com/in/tian-luke-qi/">Tian Qi</a>.

The idea is inspired by my internship mentor <a href="https://www.linkedin.com/in/kevinolivier/">Kevin Olivier</a> at [Ubisoft](https://www.ubisoft.com/en-us/), when evaluated my first machine learning model (a binary classification problem) in the industry, I found that looking at simple metric like accuracy was not enough. Based on company's business purpose, you may want to look at the recall, precision, F1 and auc as well to check your model's overall performance, also some plots like class-probability distribution, precision_recall vs threshold are useful tools to determine the probability threshold. It is, however, cumbersome to call the sklearn `metrics` many times and make those plots by hand.

## Functionality

There are currently only two `evaluator` objects, `BinaryEvaluator` and `RegressionEvaluator` for evaluating binary classification and linear regression model respectively. In the future, will integrate more type of model/task.

* `BinaryEvaluator()`: Generate useful metrics and plots for a binary classification model based on the threshold. Metrics include accuracy, recall and precision for two groups, F1 score, roc-auc, confusion matrix. Plots include ROC curve, Precision_Recall vs threshold, class probability distribution and feature importance if it can be obtained from the model. Also you can use the `find_threshold` method to generate metrics for different threshold to find the best for your purpose.

* `RegressionEvaluator()`: Similar as above but for regression model. Metrics includes MSE, RMSE, Mean/Median Absolute Error , Explained Variance Score ($R^2$). Plot includes residuals vs predicted values plot.

All evaluator result can be saved.

## Usage

Here is a [simple example](./example/examples_binary_classification.ipynb) to show the basic usage of this package.

## Installation

You can just install the package via:

```bash
$ pip install modeleval
```

