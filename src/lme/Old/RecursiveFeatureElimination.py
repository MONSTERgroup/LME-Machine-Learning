import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

class RecursiveFeatureElimination:
    def __init__(self, trainable, predictable, **kwargs) -> None:

        if kwargs.get("verbose", False):
            print("setting parameters...")

        if "features" in kwargs:
            feature_names = kwargs.get("features")

        if kwargs.get("verbose", False):
            print("building classifiers...")

        X = np.concatenate((trainable["0"]["feat"], trainable["1"]["feat"], trainable["2"]["feat"], trainable["3"]["feat"]))
        y = np.concatenate((trainable["0"]["mech"], trainable["1"]["mech"], trainable["2"]["mech"], trainable["3"]["mech"]))

       # Create the RFE object and rank each pixel
        x_train_valid, x_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        x_train, x_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=42)

        tic = time.perf_counter()
        model = RandomForestClassifier()
        parameters = {
            "n_estimators": sp_randInt(10, 500),
            "criterion": ["gini", "entropy"],
            "max_depth": sp_randInt(1, 100),
            #"min_samples_split": sp_randInt(2, 10),
            "min_samples_leaf": sp_randInt(1, 10),
            #"max_features": sp_randInt(1, len(feature_names)),

        }

        print(f'Performing Extra Tree Classifier variable search')
        randm = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameters,
            cv = 5,
            n_iter=100,
            n_jobs=-1,
            verbose=1,
            refit = True
        )
        randm.fit(x_train, y_train)

        print(" Results from Random Search ")
        print("The best estimator across ALL searched params:", randm.best_estimator_)
        print("The best score across ALL searched params:", randm.best_score_)
        print("The best parameters across ALL searched params:", randm.best_params_)

        toc = time.perf_counter()
        print(f"Calculated in {toc - tic:0.4f} seconds")

        
        clf = randm.best_estimator_
        cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
        cal_clf.fit(x_validate, y_validate)
        score = cal_clf.score(x_test,y_test)

        classifier_full = RandomForestClassifier(
            n_estimators=randm.best_params_['n_estimators'],
            criterion=randm.best_params_['criterion'],
            max_depth=randm.best_params_['max_depth'],
            min_samples_split=2,
            min_samples_leaf=randm.best_params_['min_samples_leaf'],
            #max_features=randm.best_params_['max_features'],
            )

        min_features_to_select = 1  # Minimum number of features to consider
        cv = StratifiedKFold(5)

        rfecv = RFECV(
            estimator=classifier_full,
            step=1,
            cv=cv,
            scoring="accuracy",
            min_features_to_select=min_features_to_select,
            n_jobs=2,
        )
        rfecv.fit(X, y)

        print(f"Optimal number of features: {rfecv.n_features_}")
        print(rfecv.support_)
        print(rfecv.ranking_)
        print(feature_names)

        n_scores = len(rfecv.cv_results_["mean_test_score"])
        # plt.figure()
        # plt.xlabel("Number of features selected")
        # plt.ylabel("Mean test accuracy")
        # plt.errorbar(
        #     range(min_features_to_select, n_scores + min_features_to_select),
        #     rfecv.cv_results_["mean_test_score"],
        #     yerr=rfecv.cv_results_["std_test_score"],
        # )
        # plt.title("Recursive Feature Elimination \nwith correlated features")
        # plt.show()

        base_lr = LogisticRegression(max_iter=500)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        ovr = OneVsRestClassifier(base_lr)
        ovr.fit(x_train, y_train)
        Y_pred_ovr = ovr.predict(x_test)
        actual = pd.DataFrame(data=y_test, columns=["Actual"])
        self.report = classification_report(actual,Y_pred_ovr, output_dict=True)
        print(self.report)
        ovr_jaccard_score = jaccard_score(y_test, Y_pred_ovr, average=None)

        chains = [ClassifierChain(base_lr, order="random", random_state=i) for i in range(10)]
        for chain in chains:
            chain.fit(x_train, y_train)

        Y_pred_chains = np.array([chain.predict(x_test) for chain in chains])
        chain_jaccard_scores = [
            jaccard_score(x_test, Y_pred_chain >= 0.5, average=None)
            for Y_pred_chain in Y_pred_chains
        ]

        Y_pred_ensemble = Y_pred_chains.mean(axis=0)
        ensemble_jaccard_score = jaccard_score(
            x_test, Y_pred_ensemble >= 0.5, average=None
        )

        model_scores = [ovr_jaccard_score] + chain_jaccard_scores
        model_scores.append(ensemble_jaccard_score)

        model_names = (
            "Independent",
            "Chain 1",
            "Chain 2",
            "Chain 3",
            "Chain 4",
            "Chain 5",
            "Chain 6",
            "Chain 7",
            "Chain 8",
            "Chain 9",
            "Chain 10",
            "Ensemble",
        )

        x_pos = np.arange(len(model_names))

        # Plot the Jaccard similarity scores for the independent model, each of the
        # chains, and the ensemble (note that the vertical axis on this plot does
        # not begin at 0).

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.grid(True)
        ax.set_title("Classifier Chain Ensemble Performance Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation="vertical")
        ax.set_ylabel("Jaccard Similarity Score")
        ax.set_ylim([min(model_scores) * 0.9, max(model_scores) * 1.1])
        colors = ["r"] + ["b"] * len(chain_jaccard_scores) + ["g"]
        ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
        plt.tight_layout()
        plt.show()
        



        
 