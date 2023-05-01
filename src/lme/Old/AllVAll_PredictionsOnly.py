from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

from matplotlib.legend_handler import HandlerLine2D


class AllVAll_PredictionsOnly:
    def __init__(self, trainable, predictable, **kwargs) -> None:

        if kwargs.get("verbose", False):
            print("setting parameters...")

        if "features" in kwargs:
            feature_names = kwargs.get("features")

        # if not kwargs.get("genetic", False):
        #     self.params = {}

        #     self.params["01"] = {'learning_rate': 0.14460964141681998, 'max_depth': 9, 'max_features': 5, 'min_samples_leaf': 0.027127267243118, 'min_samples_split': 0.3725546395367668, 'n_estimators': 542, 'subsample': 0.7945532068183083}
        #     self.params["02"] = {'learning_rate': 0.17528554658298034, 'max_depth': 19, 'max_features': 61, 'min_samples_leaf': 0.21373876568924605, 'min_samples_split': 0.17830077198075767, 'n_estimators': 230, 'subsample': 0.6958616319367862}
        #     self.params["03"] = {'learning_rate': 0.25451251289455024, 'max_depth': 11, 'max_features': 27, 'min_samples_leaf': 0.0061240494986215645, 'min_samples_split': 0.931005132652906, 'n_estimators': 226, 'subsample': 0.9542023462571922}
        #     self.params["12"] = {'learning_rate': 0.6054325708217553, 'max_depth': 10, 'max_features': 60, 'min_samples_leaf': 0.2684584421879727, 'min_samples_split': 0.2617892399884044, 'n_estimators': 393, 'subsample': 0.8169639788630859}
        #     self.params["13"] = {'learning_rate': 0.9074038030926245, 'max_depth': 17, 'max_features': 41, 'min_samples_leaf': 0.09967639155410575, 'min_samples_split': 0.3029448258185029, 'n_estimators': 792, 'subsample': 0.5672707768083121}
        #     self.params["23"] = {'learning_rate': 0.04407844492043744, 'max_depth': 2, 'max_features': 27, 'min_samples_leaf': 0.25375798912904596, 'min_samples_split': 0.583263975885689, 'n_estimators': 261, 'subsample': 0.8195810025495328}
        # else:
        #     self.params = kwargs.get("params")

        if kwargs.get("verbose", False):
            print("building classifiers...")


        self.clf = {}
        self.x_test = {}
        self.y_test = {}

        for k in trainable.keys():
            for l in trainable.keys():
                if int(l) > int(k):
                    self.clf[f"{k}{l}"], self.x_test[f"{k}{l}"], self.y_test[f"{k}{l}"] = self.getBestClassifier(
                        trainable, k, l, verbose=False, features=feature_names
                    )

        # if kwargs.get("verbose", False):
        #     print("creating test splits")

        # self.feat_train = {}
        # self.feat_test = {}
        # self.mech_train = {}
        # self.mech_test = {}

        # for k in trainable.keys():
        #     (
        #         self.feat_train[int(k)],
        #         self.feat_test[int(k)],
        #         self.mech_train[int(k)],
        #         self.mech_test[int(k)],
        #     ) = train_test_split(
        #         trainable[k]["feat"],
        #         np.asarray(trainable[k]["mech"]),
        #         test_size=0.25,
        #         random_state=42,
        #     )

        # if kwargs.get("verbose", False):
        #     print("Concatenate all the things")

        # self.feat_train_pair = {}
        # self.mech_train_pair = {}
        # self.feat_test_pair = {}
        # self.mech_test_pair = {}

        # for k in trainable.keys():
        #     for l in trainable.keys():
        #         if int(l) > int(k):
        #             self.feat_train_pair[f"{k}{l}"] = np.concatenate(
        #                 (self.feat_train[int(k)], self.feat_train[int(l)])
        #             )
        #             self.mech_train_pair[f"{k}{l}"] = np.concatenate(
        #                 (self.mech_train[int(k)], self.mech_train[int(l)])
        #             )
        #             self.feat_test_pair[f"{k}{l}"] = np.concatenate(
        #                 (self.feat_test[int(k)], self.feat_test[int(l)])
        #             )
        #             self.mech_test_pair[f"{k}{l}"] = np.concatenate(
        #                 (self.mech_test[int(k)], self.mech_test[int(l)])
        #             )

        # if kwargs.get("verbose", False):
        #     print("Training...maybe?...")

        # for k in trainable.keys():
        #     for l in trainable.keys():
        #         if int(l) > int(k):
        #             self.clf[f"{k}{l}"].fit(
        #                 self.feat_train_pair[f"{k}{l}"], self.mech_train_pair[f"{k}{l}"]
        #             )

        # feat_test = np.concatenate([self.feat_test[int(k)] for k in trainable.keys()])
        # mech_test = np.concatenate([self.mech_test[int(k)] for k in trainable.keys()])

        feat_test = np.concatenate((self.x_test['01'], self.x_test['23']))
        mech_test = np.concatenate((self.y_test['01'], self.y_test['23']))
        
        # if kwargs.get("verbose", False):
        #     print("Predicting...maybe?...")

        self.prob = {}

        for k in trainable.keys():
            for l in trainable.keys():
                if int(l) > int(k):
                    self.prob[f"{k}{l}"] = self.clf[f"{k}{l}"].predict_proba(feat_test)

        self.prob_all = np.concatenate(
            [
                self.prob[f"{k}{l}"]
                for k in trainable.keys()
                for l in trainable.keys()
                if int(l) > int(k)
            ],
            axis=1,
        )

        displayable_prob_all = pd.DataFrame(
            data=self.prob_all,
            columns=[
                f"{k}{l}-{k if i == 1 else l}"
                for k in trainable.keys()
                for l in trainable.keys()
                for i in range(1, 3)
                if int(l) > int(k)
            ],
        )
        pd.set_option("display.max_rows", self.prob_all.shape[0] + 1)
        print(displayable_prob_all)

        preds = [0.0] * feat_test.shape[0] * 4
        preds = np.reshape(preds, (feat_test.shape[0], 4))

        for ii in [0, 2, 4]:
            for jj in range(0, feat_test.shape[0]):
                preds[jj, 0] = preds[jj, 0] + self.prob_all[jj, ii]
        for ii in [1, 6, 8]:
            for jj in range(0, feat_test.shape[0]):
                preds[jj, 1] = preds[jj, 1] + self.prob_all[jj, ii]
        for ii in [3, 7, 10]:
            for jj in range(0, feat_test.shape[0]):
                preds[jj, 2] = preds[jj, 2] + self.prob_all[jj, ii]
        for ii in [5, 9, 11]:
            for jj in range(0, feat_test.shape[0]):
                preds[jj, 3] = preds[jj, 3] + self.prob_all[jj, ii]

        displayable_preds = pd.DataFrame(data=preds, columns=["0", "1", "2", "3"])
        pd.set_option("display.max_rows", preds.shape[0] + 1)
        print(displayable_preds)

        modelpred = displayable_preds.idxmax(axis=1)
        pred = pd.DataFrame(data=modelpred, columns=["Prediction"])
        actual = pd.DataFrame(data=mech_test, columns=["Actual"])
        # display_side_by_side(pred, actual)

        compare_predictions = pd.concat([pred, actual], axis=1)
        print(compare_predictions)

        modelpredn = np.array([0] * feat_test.shape[0])
        for ii in range(0, feat_test.shape[0] - 1):
            modelpredn[ii] = int(pred.iloc[ii])

        #sns.set()
        #cm = confusion_matrix(actual, modelpredn, normalize="all")

        #ax = plt.subplot()
        #sns.heatmap(cm / np.sum(cm), annot=True, fmt=".2%", cmap="Blues")

        # labels, title and ticks
        #ax.set_xlabel("Predicted labels")
        #ax.set_ylabel("True labels")
        #ax.set_title("Confusion Matrix")
        #ax.xaxis.set_ticklabels(["0", "1", "2", "3"])
        #ax.yaxis.set_ticklabels(["0", "1", "2", "3"])
        #plt.show()

        #self.report = classification_report(actual, modelpredn, output_dict=True)
        #print(self.report)

        ### new stuff

        prediction_test = predictable["0"]["feat"]

        self.prediction_prob = {}

        for k in trainable.keys():
            for l in trainable.keys():
                if int(l) > int(k):
                    self.prediction_prob[f"{k}{l}"] = self.clf[f"{k}{l}"].predict_proba(prediction_test)

        self.prediction_prob_all = np.concatenate(
            [
                self.prediction_prob[f"{k}{l}"]
                for k in trainable.keys()
                for l in trainable.keys()
                if int(l) > int(k)
            ],
            axis=1,
        )

        displayable_prediction_prob_all = pd.DataFrame(
            data=self.prediction_prob_all,
            columns=[
                f"{k}{l}-{k if i == 1 else l}"
                for k in trainable.keys()
                for l in trainable.keys()
                for i in range(1, 3)
                if int(l) > int(k)
            ],
        )
        pd.set_option("display.max_rows", self.prediction_prob_all.shape[0] + 1)
        print(displayable_prediction_prob_all)

        preds = [0.0] * prediction_test.shape[0] * 4
        preds = np.reshape(preds, (prediction_test.shape[0], 4))

        for ii in [0, 2, 4]:
            for jj in range(0, prediction_test.shape[0]):
                preds[jj, 0] = preds[jj, 0] + self.prediction_prob_all[jj, ii]
        for ii in [1, 6, 8]:
            for jj in range(0, prediction_test.shape[0]):
                preds[jj, 1] = preds[jj, 1] + self.prediction_prob_all[jj, ii]
        for ii in [3, 7, 10]:
            for jj in range(0, prediction_test.shape[0]):
                preds[jj, 2] = preds[jj, 2] + self.prediction_prob_all[jj, ii]
        for ii in [5, 9, 11]:
            for jj in range(0, prediction_test.shape[0]):
                preds[jj, 3] = preds[jj, 3] + self.prediction_prob_all[jj, ii]

        displayable_preds = pd.DataFrame(data=preds, columns=["0", "1", "2", "3"])
        pd.set_option("display.max_rows", preds.shape[0] + 1)
        print(displayable_preds)

        modelpred = displayable_preds.idxmax(axis=1)
        pred = pd.DataFrame(data=modelpred, columns=["Prediction"])
        id = pd.DataFrame(data=predictable["0"]["id"], columns=["ID"])
        #actual = pd.DataFrame(data=mech_test, columns=["Actual"])
        # display_side_by_side(pred, actual)
        #print(pred)

        compare_predictions = pd.concat([id, pred], axis=1)
        print(compare_predictions)

        #modelpredn = np.array([0] * prediction_test.shape[0])
        #for ii in range(0, prediction_test.shape[0] - 1):
        #    modelpredn[ii] = int(pred.iloc[ii])

    def how_accurate(self) -> float:
        return self.report["accuracy"]

    def getBestClassifier(
        self, trainable, first, second, **kwargs
    ) -> GradientBoostingClassifier:

        # best_learning_rate = 0.5
        # best_n_estimators = 27
        # best_max_depth = 9
        # best_min_samples_split = 0.3
        # best_min_samples_leaf = 0.1
        # best_max_features = 45

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

        x_first = trainable[first]["feat"]
        x_second = trainable[second]["feat"]
        x = np.concatenate((x_first, x_second))

        y_first = trainable[first]["mech"]
        y_second = trainable[second]["mech"]

        print(y_first)
        print(y_second)
        y = np.concatenate((y_first, y_second))
        print(y)

        y[y == int(first)] = 0
        y[y == int(second)] = 1
        print(y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)

        model = GradientBoostingClassifier()
        parameters = {
            "learning_rate": sp_randFloat(),
            "subsample": sp_randFloat(),
            "max_depth": sp_randInt(1, 32),
            "max_features": sp_randInt(1, 13),
            "min_samples_leaf": sp_randFloat(loc=0.001, scale=0.499),
            "min_samples_split": sp_randFloat(),
            "n_estimators": sp_randInt(1, 1000),
        }
        print(f'Scoring {first}{second}')
        randm = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameters,
            cv = 5,
            n_iter=500,
            n_jobs=-1,
            verbose=1,
        )
        randm.fit(x_train, y_train)

        print(" Results from Random Search ")
        print("The best estimator across ALL searched params:", randm.best_estimator_)
        print("The best score across ALL searched params:", randm.best_score_)
        print("The best parameters across ALL searched params:", randm.best_params_)

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
        #plt.show()

        if (int(first) == 1 and (int(second) == 2 or int(second) == 3)):
            y_test[y_test == 1] = int(second)
            y_test[y_test == 0] = int(first)
            print(y_test)
        else:
            y_test[y_test == 0] = int(first)
            y_test[y_test == 1] = int(second)
            print(y_test)

        return randm.best_estimator_, x_test, y_test
