# modeleval

A python package that generates useful evaluation metrics and plots for machine learning models, fully compatible ***scikit-learn***, ***xgboost***, ***LightGBM*** and ***catboost API***. 

By <a href="https://tqi2.github.io/">Tian Qi</a>.

This idea was inspired by my internship mentor <a href="https://www.linkedin.com/in/kevinolivier/">Kevin Olivier</a> at [Ubisoft](https://www.ubisoft.com/en-us/), when evaluated my first machine learning model (a binary classification problem) in the industry, I found that looking at simple metric like accuracy was not enough. Based on company's business purpose, you may want to look at the recall, precision, F1 and auc as well to check your model's overall performance, also some plots like class-probability distribution, precision_recall vs threshold are useful tools to determine the probability threshold. It is, however, cumbersome to call the sklearn `metrics` many times and make those plots by hand.

## Functionality

There are currently only two `evaluator` objects, `BinaryEvaluator` and `RegressionEvaluator` for evaluating binary classification and regression model respectively. In the future, will integrate more type of model/task.

* `BinaryEvaluator()`: Use `evaluate` method to generate useful metrics and plots for a binary classification model with the given threshold. Metrics include accuracy, recall and precision for two groups, F1 score, roc-auc, confusion matrix. Plots include ROC curve, Precision_Recall vs threshold, class probability distribution and feature importance if it can be obtained from the model. With `ThresGridSearch` method, you can do a grid search on thresholds and sort result by the metric you specified. Also, the `find_best_model` method can compare the result of many models and give the model which has the best metric value you want.


* `RegressionEvaluator()`: Similar as above but for regression model. Metrics includes Mean Squared Error(MSE), Root Mean Squared Error(RMSE), Mean Absolute Error(MAE), Root Mean Squared Logarithmic Error (RMSLE), Explained Variance Score ![#r2](https://latex.codecogs.com/gif.latex?R%5E2). Plot has residuals vs predicted values plot.

All evaluators' result can be saved to given path.

## Usage

Here is a [example](./example/binary_classification_example.ipynb) to show the basic usage of `BinaryEvaluator()` and [example](./example/regression_example.ipynb) for `RegressionEvaluator()`.

## Installation

You can just install the package via:

```bash
$ pip install modeleval
```
or upgrade to the latest version:


```bash
$ pip install -U modeleval
```


