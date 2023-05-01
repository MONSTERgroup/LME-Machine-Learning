import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import ExtraTreesClassifier

from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

class ExtraTreesEnsembleTest:
    def __init__(self, trainable, predictable, **kwargs) -> None:

        if kwargs.get("verbose", False):
            print("setting parameters...")

        if "features" in kwargs:
            feature_names = kwargs.get("features")

        if kwargs.get("verbose", False):
            print("building classifiers...")

        x = np.concatenate((trainable["0"]["feat"], trainable["1"]["feat"], trainable["2"]["feat"], trainable["3"]["feat"]))
        y = np.concatenate((trainable["0"]["mech"], trainable["1"]["mech"], trainable["2"]["mech"], trainable["3"]["mech"]))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        tic = time.perf_counter()
        model = ExtraTreesClassifier()
        parameters = {
            "n_estimators": sp_randInt(10, 500),
            "criterion": ["gini", "entropy"],
            "max_depth": sp_randInt(1, 100),
            #"min_samples_split": sp_randInt(2, 10),
            "min_samples_leaf": sp_randInt(1, 10),
            "max_features": sp_randInt(1, len(feature_names)),

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
        score = clf.score(x_test,y_test)
        print("The score after refit:",score)

        pred = clf.predict(x_test)
        # print(pred)
        # pred = pd.DataFrame(data=pred, columns=["Prediction"])
        # print(pred)
        actual = pd.DataFrame(data=y_test, columns=["Actual"])

        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=clf.classes_)
        plt.show()
        # sns.set()
        # cm = confusion_matrix(actual, pred)

        # ax = plt.subplot()
        # sns.heatmap(cm / np.sum(cm), annot=True, fmt=".2%", cmap="Blues")

        # # labels, title and ticks
        # ax.set_xlabel("Predicted labels")
        # ax.set_ylabel("True labels")
        # ax.set_title("Confusion Matrix")
        # ax.xaxis.set_ticklabels(["0", "1", "2", "3"])
        # ax.yaxis.set_ticklabels(["0", "1", "2", "3"])
        # plt.show()

        self.report = classification_report(actual,pred, output_dict=True)
        print(self.report)

        prediction_test = predictable["0"]["feat"]

        predictions = clf.predict_proba(prediction_test)
        print(predictions)
        pred = [np.where(p==np.max(p))[0][0] for p in predictions]

        display_pred = pd.DataFrame(data=pred, columns=["Prediction"])
        display_id = pd.DataFrame(data=predictable["0"]["id"], columns=["ID"])

        compare_predictions = pd.concat([display_id, display_pred], axis=1)
        print(compare_predictions)



        classifier_full = ExtraTreesClassifier(
            n_estimators=randm.best_params_['n_estimators'],
            criterion=randm.best_params_['criterion'],
            max_depth=randm.best_params_['max_depth'],
            min_samples_split=2,
            min_samples_leaf=randm.best_params_['min_samples_leaf'],
            max_features=randm.best_params_['max_features'],
            )
        classifier_full.fit(x,y)

        prediction_test = predictable["0"]["feat"]

        predictions = classifier_full.predict_proba(prediction_test)

        pred = [np.where(p==np.max(p))[0][0] for p in predictions]

        display_pred = pd.DataFrame(data=pred, columns=["Prediction"])
        display_id = pd.DataFrame(data=predictable["0"]["id"], columns=["ID"])

        compare_predictions = pd.concat([display_id, display_pred], axis=1)
        print(compare_predictions)
