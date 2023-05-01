from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

from matplotlib.legend_handler import HandlerLine2D


class OneVOne:
    def __init__(self, trainable, first, second, **kwargs) -> None:
        
        best_learning_rate = 0.5
        best_n_estimators = 27
        best_max_depth = 9
        best_min_samples_split = 0.3
        best_min_samples_leaf = 0.1
        best_max_features = 45
        
        if kwargs.get("verbose", False):
            print("setting parameters...")

        if "features" in kwargs:
            feature_names = kwargs.get("features")

        # if "params" in kwargs:
        #     self.params = kwargs.get("params")
        # else:
        #     self.params = {
        #         "learning_rate": best_learning_rate,
        #         "loss": "deviance",
        #         "max_depth": best_max_depth,
        #         "max_features": best_max_features,
        #         "min_samples_leaf": best_min_samples_leaf,
        #         "min_samples_split": best_min_samples_split,
        #         "n_estimators": best_n_estimators,
        #         "subsample": 1,
        #     }


        x_first = trainable[first]['feat']
        x_second = trainable[second]['feat']
        x = np.concatenate((x_first, x_second))

        y_first = trainable[first]['mech']
        y_second = trainable[second]['mech']
        y = np.concatenate((y_first, y_second))
        print(y)

        y[y == int(first)] = 0
        y[y == int(second)] = 1
        print(y)
        

        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

        model = GradientBoostingClassifier()
        parameters = {
            'learning_rate' : sp_randFloat(),
            'subsample' : sp_randFloat(),
            'max_depth' : sp_randInt(1, 32),
            'max_features' : sp_randInt(1, 13),
            'min_samples_leaf' : sp_randFloat(loc=0.001, scale=0.499),
            'min_samples_split' : sp_randFloat(),
            'n_estimators': sp_randInt(1, 1000),
        }

        randm = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=500, n_jobs=-1, verbose=10)
        randm.fit(x_train, y_train)

        print(" Results from Random Search \n" )
        print("The best estimator across ALL searched params: \n", randm.best_estimator_)
        print("The best score across ALL searched params: \n", randm.best_score_)
        print("The best parameters across ALL searched params: \n", randm.best_params_)

        feature_importance = randm.best_estimator_.feature_importances_
        print("Feature importance: \n", feature_importance)
        print(feature_names)

        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.barh(pos, feature_importance[sorted_idx], align="center")
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.title("Feature Importance (MDI)")

        result = permutation_importance(
            randm, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2
        )
        sorted_idx = result.importances_mean.argsort()
        plt.subplot(1, 2, 2)
        plt.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            labels=np.array(feature_names)[sorted_idx],
        )
        plt.title("Permutation Importance (test set)")
        fig.tight_layout()
        plt.show()

        return 